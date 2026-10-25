"""Unit tests for the Ollama backend; the HTTP call is replaced, so no Ollama server is needed."""

from rag_assistant.llm import OllamaLLM

OLLAMA_REPLY = {
    "model": "llama3.2:3b",
    "message": {"role": "assistant", "content": '{"answer": "ok"}'},
    "prompt_eval_count": 120,
    "eval_count": 8,
    "done_reason": "stop",
}


def make_llm(monkeypatch, reply=OLLAMA_REPLY):
    llm = OllamaLLM(model="llama3.2:3b", num_ctx=4096)
    sent = {}

    def fake_post(path, payload):
        sent["path"], sent["payload"] = path, payload
        return reply

    monkeypatch.setattr(llm, "_post", fake_post)
    return llm, sent


def test_payload_contains_system_prompt_schema_and_options(monkeypatch):
    llm, sent = make_llm(monkeypatch)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    llm.generate("question", system="instructions", json_schema=schema, max_tokens=50)

    payload = sent["payload"]
    assert sent["path"] == "/api/chat"
    assert payload["messages"] == [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "question"},
    ]
    assert payload["format"] == schema
    assert payload["options"] == {"num_ctx": 4096, "num_predict": 50, "temperature": 0.0, "seed": 42}
    assert payload["stream"] is False


def test_payload_without_system_or_schema(monkeypatch):
    llm, sent = make_llm(monkeypatch)
    llm.generate("question")
    assert sent["payload"]["messages"] == [{"role": "user", "content": "question"}]
    assert "format" not in sent["payload"]


def test_response_is_parsed(monkeypatch):
    llm, _ = make_llm(monkeypatch)
    response = llm.generate("question")
    assert response.text == '{"answer": "ok"}'
    assert response.model == "llama3.2:3b"
    assert (response.input_tokens, response.output_tokens) == (120, 8)
    assert response.truncated is False
    assert response.latency_s >= 0


def test_output_cut_off_at_max_tokens_is_flagged(monkeypatch):
    llm, _ = make_llm(monkeypatch, reply={**OLLAMA_REPLY, "done_reason": "length"})
    assert llm.generate("question").truncated is True


TAGS_REPLY = {"models": [{"name": "llama3:latest", "digest": "aaa"}, {"name": "llama3.2:3b", "digest": "bbb"}]}


def test_settings_contain_digest_and_generation_options(monkeypatch):
    llm = OllamaLLM(model="llama3.2:3b", num_ctx=4096, seed=7)
    monkeypatch.setattr(llm, "_get", lambda path: TAGS_REPLY)
    assert llm.settings() == {"model": "llama3.2:3b", "digest": "bbb", "num_ctx": 4096, "temperature": 0.0, "seed": 7}


def test_digest_of_model_without_tag_uses_latest(monkeypatch):
    llm = OllamaLLM(model="llama3")
    monkeypatch.setattr(llm, "_get", lambda path: TAGS_REPLY)
    assert llm.model_digest() == "aaa"


def test_digest_of_missing_model_is_none(monkeypatch):
    llm = OllamaLLM(model="qwen2.5:3b")
    monkeypatch.setattr(llm, "_get", lambda path: TAGS_REPLY)
    assert llm.model_digest() is None
