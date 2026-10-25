"""The prompt and output format for answering a question from retrieved corpus chunks."""

import hashlib
import json
from collections.abc import Sequence

# The sentence of every refusal. The assistant's answer rules give it to every refusal, whatever the model wrote;
INSUFFICIENT_TEXT = "The knowledge source does not contain enough information to answer this question."

SYSTEM_PROMPT = """You are a technical assistant for LangChain and related libraries.
Answer ONLY with information that appears in the sources provided by the user. Do not add facts, APIs or code from
your own knowledge.
- If the sources answer the question, or part of it, answer that part, set insufficient_information to false and say
  which part the sources do not cover.
- If the sources do not contain the answer, set insufficient_information to true, say so in one sentence and cite
  nothing, even if the sources are about the same topic.
- Every statement in the answer must be supported by a source you cite by its exact doc_id in cited_doc_ids. Only cite
  doc_ids that appear in the sources.
- Keep the answer concise. Include code from the sources when it helps."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "cited_doc_ids": {"type": "array", "items": {"type": "string"}},
        "insufficient_information": {"type": "boolean"},
    },
    "required": ["answer", "cited_doc_ids", "insufficient_information"],
}

TRUNCATION_MARKER = "\n[... truncated]"


def format_source(doc_id: str, text: str, max_chars: int) -> str:
    """Format one corpus chunk as a source block, cut to max_chars characters.

    Args:
        doc_id: The corpus doc_id of the chunk.
        text: The chunk text.
        max_chars: Maximum number of characters of the text to include.

    Returns:
        The source block, e.g. '<source doc_id="...">text</source>'.
    """
    body = text if len(text) <= max_chars else text[:max_chars] + TRUNCATION_MARKER
    return f'<source doc_id="{doc_id}">\n{body}\n</source>'


def build_prompt(question: str, sources: Sequence[tuple[str, str]], max_chars: int) -> str:
    """Build the user prompt: the sources followed by the question.

    Args:
        question: The user's question (title and body).
        sources: (doc_id, text) pairs of the retrieved chunks, best first.
        max_chars: Maximum number of characters per chunk.

    Returns:
        The user prompt.
    """
    blocks = "\n\n".join(format_source(doc_id, text, max_chars) for doc_id, text in sources)
    return f"Sources:\n\n{blocks}\n\nQuestion:\n{question}"


def prompt_fingerprint() -> str:
    """Return a short hash of everything that shapes the prompt, stored with evaluation runs.

    It covers the system prompt, the answer schema and the prompt layout (via a prompt built from placeholder
    values, including a truncated source), so any edit to them gives a new fingerprint without bumping a version.

    Returns:
        The first 12 hex characters of a SHA-256 hash.
    """
    example = build_prompt("QUESTION", [("DOC_ID", "TEXT TO CUT")], max_chars=4)
    content = json.dumps([SYSTEM_PROMPT, ANSWER_SCHEMA, example], sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
