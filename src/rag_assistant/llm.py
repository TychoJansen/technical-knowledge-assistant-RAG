"""LLMs behind a common interface: a local Ollama server now; hosted APIs can be added as another LLM subclass."""

import json
import os
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from rag_assistant.config import DEFAULT_MODEL
from rag_assistant.logger import get_logger

logger = get_logger(__name__)

# OLLAMA_HOST is Ollama's own environment variable
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# Ollama's default context window is small and it silently drops the start of longer prompts, so set it explicitly.
# 8192 is the maximum of the judge model llama3 (`ollama show llama3`)
DEFAULT_NUM_CTX = 8192
DEFAULT_SEED = 42
_TIMEOUT_S = 1800


@dataclass(frozen=True)
class LLMResponse:
    """The result of one LLM call.

    Attributes:
        text: The generated text (a JSON string when a json_schema was given).
        model: The model that produced the text.
        input_tokens: Number of prompt tokens the model processed.
        output_tokens: Number of tokens generated.
        latency_s: Wall-clock time of the call in seconds.
        truncated: True if generation stopped at max_tokens, so the text may be incomplete (e.g. invalid JSON).
    """

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    truncated: bool


class LLM(ABC):
    """An LLM maps a prompt to generated text.

    Implementations only send what they are given.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a response to a prompt.

        Args:
            prompt: The user prompt.
            system: Optional system instructions.
            json_schema: Optional JSON schema; if given, the output must be JSON matching it.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            The generated text with token counts, latency and whether it was cut off.
        """

    def settings(self) -> dict[str, Any]:
        """Return everything about this LLM that changes its output, to store with evaluation results.

        Returns:
            A JSON-serialisable dict of settings.
        """
        raise NotImplementedError

    def unload(self) -> None:
        """Free the model and any cached state, so the next call starts fresh. No-op for backends without state."""


class OllamaLLM(LLM):
    """Local LLM served by Ollama (https://ollama.com), called over its HTTP API with the standard library.

    Deterministic by default (temperature 0, fixed seed).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        num_ctx: int = DEFAULT_NUM_CTX,
        temperature: float = 0.0,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """Configure the Ollama backend.

        Args:
            model: The Ollama model name, e.g. "llama3.2:3b" (must be pulled with `ollama pull` first).
            host: Base URL of the Ollama server.
            num_ctx: Context window in tokens; prompts longer than this are truncated by Ollama.
            temperature: Sampling temperature; 0 gives (near-)deterministic output.
            seed: Random seed, for reproducible output.
        """
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.seed = seed

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate a response with Ollama's /api/chat endpoint.

        Args:
            prompt: The user prompt.
            system: Optional system instructions.
            json_schema: Optional JSON schema; Ollama constrains the output to it.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            The generated text with token counts, latency and whether it was cut off.

        Raises:
            RuntimeError: If Ollama cannot be reached or the model is not pulled.
        """
        payload = self._build_payload(prompt, system, json_schema, max_tokens)
        start = time.perf_counter()
        data = self._post("/api/chat", payload)
        latency_s = time.perf_counter() - start
        response = self._parse_response(data, latency_s)
        if response.input_tokens >= self.num_ctx:
            logger.warning(
                f"Prompt filled the whole context window ({response.input_tokens} >= num_ctx {self.num_ctx}); "
                "Ollama may have dropped part of it"
            )
        if response.truncated:
            logger.warning(f"Output cut off at max_tokens={max_tokens}; the text may be incomplete")
        return response

    def settings(self) -> dict[str, Any]:
        """Return the model, its digest and the generation options (see LLM.settings).

        The digest identifies the exact weights: a tag such as "llama3.2:3b" can point to other weights after a new
        `ollama pull`.

        Returns:
            Dict with model, digest, num_ctx, temperature and seed.
        """
        return {
            "model": self.model,
            "digest": self.model_digest(),
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
            "seed": self.seed,
        }

    def unload(self) -> None:
        """Unload the model from Ollama's memory, which also clears its prompt cache.

        Ollama reuses the processed prompt of earlier calls, and a cached prompt can give a different answer than a
        fresh one. Unloading before a run makes every run start from the same state.
        """
        self._post("/api/generate", {"model": self.model, "keep_alive": 0})

    def model_digest(self) -> str | None:
        """Return the digest of the local model from Ollama's /api/tags.

        Returns:
            The digest, or None if the model is not pulled.
        """
        name = self.model if ":" in self.model else f"{self.model}:latest"
        for model in self._get("/api/tags").get("models", []):
            if model.get("name") == name:
                return model.get("digest")
        return None

    def _build_payload(
        self, prompt: str, system: str | None, json_schema: dict[str, Any] | None, max_tokens: int
    ) -> dict[str, Any]:
        """Build the /api/chat request body.

        Args:
            prompt: The user prompt.
            system: Optional system instructions.
            json_schema: Optional JSON schema for the output.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            The request body.
        """
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
                "temperature": self.temperature,
                "seed": self.seed,
            },
        }
        if json_schema is not None:
            payload["format"] = json_schema
        return payload

    def _parse_response(self, data: dict[str, Any], latency_s: float) -> LLMResponse:
        """Turn an /api/chat response into an LLMResponse.

        Args:
            data: The decoded JSON response.
            latency_s: Wall-clock time of the call in seconds.

        Returns:
            The response, with truncated=True if generation stopped at the token limit.
        """
        return LLMResponse(
            text=data["message"]["content"],
            model=data.get("model", self.model),
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            latency_s=latency_s,
            truncated=data.get("done_reason") == "length",
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST a JSON payload to an Ollama endpoint and return the decoded response (see _request)."""
        return self._request(path, json.dumps(payload).encode("utf-8"))

    def _get(self, path: str) -> dict[str, Any]:
        """GET an Ollama endpoint and return the decoded response (see _request)."""
        return self._request(path, None)

    def _request(self, path: str, body: bytes | None) -> dict[str, Any]:
        """Send a request to the Ollama server.

        Args:
            path: The endpoint, e.g. "/api/chat".
            body: The JSON body for a POST, or None for a GET.

        Returns:
            The decoded JSON response.

        Raises:
            RuntimeError: If the model is not pulled, Ollama returns an error, or the server cannot be reached.
        """
        request = urllib.request.Request(self.host + path, body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
                return json.load(response)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code == 404:
                raise RuntimeError(f"Ollama model {self.model!r} not found; run `ollama pull {self.model}`") from e
            raise RuntimeError(f"Ollama returned HTTP {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach Ollama at {self.host}; is it running (`ollama serve`)?") from e
