"""LLM-as-judge for answer evaluation: are the nuggets in the answer, and do the cited chunks support it.

The judge sees evaluation data (nuggets). That is allowed here: the judge grades the assistant, it is not the
assistant. Nothing a judge sees is ever fed back to the assistant.

The citation judge must quote the supporting text from the source, and a citation only counts as supported if that
quote really occurs in the source. This turns the judge's opinion into checkable evidence.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

from rag_assistant.llm import LLM
from rag_assistant.logger import get_logger
from rag_assistant.prompt import format_source

logger = get_logger(__name__)

# Make sure runs of different versions are never mixed.
JUDGE_VERSION = 5

# A quote counts as found if this share of it occurs as one contiguous block in the source (after normalising case and
# whitespace). Below 1.0 to tolerate small copying errors of a small judge model; high enough to reject paraphrases.
QUOTE_MATCH_THRESHOLD = 0.8
# Quotes shorter than this (after normalising) are too unspecific to count as evidence.
MIN_QUOTE_CHARS = 15
# Answers shorter than this (after normalising) are too short to state a fact, e.g. "No" (see is_degenerate_answer).
MIN_ANSWER_CHARS = 20

NUGGET_SYSTEM_PROMPT = """You are a strict grader of answers to technical questions.
You get an answer and a list of facts labelled fact_1, fact_2, and so on. For each fact, decide whether the answer
states it: the same meaning, possibly in other words or with equivalent code. A fact counts only if the answer clearly
contains it. Give a verdict for every fact: true if the answer contains it, false if not. Do not use outside knowledge
and do not judge whether the answer is correct, only whether it contains the fact."""


def nugget_schema(n_facts: int) -> dict:
    """Build the nugget judge's output schema: one required true/false field per fact (fact_1, fact_2, ...).

    Required fields are part of the grammar Ollama enforces while generating, so the judge cannot close the object
    before it has judged every fact. (Version 4 asked for a list, which the judge often closed after the first fact.)

    Args:
        n_facts: The number of facts (nuggets) of the question.

    Returns:
        The JSON schema.
    """
    keys = [f"fact_{i}" for i in range(1, n_facts + 1)]
    return {"type": "object", "properties": {k: {"type": "boolean"} for k in keys}, "required": keys}


CITATION_SYSTEM_PROMPT = """You are a strict grader of citations.
You get a question, an answer and one source the answer cites. Decide whether the source supports the answer: it must
contain information that backs up at least one concrete claim or code example in the answer.
- If it does, set "supported" to true and copy into "quote" the sentence or code line from the SOURCE that backs up the
  answer, exactly as it is written in the source.
- If it does not (the source is about something else, or only shares some keywords with the answer), set "supported" to
  false and "quote" to an empty string."""

CITATION_SCHEMA = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}, "quote": {"type": "string"}},
    "required": ["supported", "quote"],
}


@dataclass(frozen=True)
class NuggetJudgment:
    """Per nugget whether the answer contains it, plus whether the judge output was usable."""

    present: tuple[bool, ...]
    parse_ok: bool
    latency_s: float


@dataclass(frozen=True)
class CitationJudgment:
    """Whether one cited chunk supports the answer.

    Attributes:
        doc_id: The cited chunk.
        supported: The final verdict: the judge said supported AND its quote was found in the source.
        judge_supported: The judge's own verdict, before the quote check.
        quote: The text the judge quoted as evidence.
        quote_found: Whether the quote occurs in the source (see quote_in_source).
        parse_ok: Whether the judge output was usable.
        latency_s: Time of the judge call.
    """

    doc_id: str
    supported: bool
    judge_supported: bool
    quote: str
    quote_found: bool
    parse_ok: bool
    latency_s: float


def judge_nuggets(judge: LLM, question: str, answer: str, nuggets: Sequence[str]) -> NuggetJudgment:
    """Ask the judge which nuggets the answer contains, in one call for all nuggets of the question.

    Args:
        judge: The judge LLM.
        question: The full question (title and body), for context.
        answer: The assistant's answer.
        nuggets: The nugget texts of the question.

    Returns:
        One boolean per nugget, in order. Nuggets the judge did not return count as not present.
    """
    keys = [f"fact_{i}" for i in range(1, len(nuggets) + 1)]
    facts = "\n".join(f"{key}: {text}" for key, text in zip(keys, nuggets))
    prompt = f"Question:\n{question}\n\nAnswer:\n{answer}\n\nFacts:\n{facts}"
    schema = nugget_schema(len(nuggets))
    response = judge.generate(prompt, system=NUGGET_SYSTEM_PROMPT, json_schema=schema, max_tokens=400)
    try:
        data = json.loads(response.text)
        verdicts = {key: data[key] for key in keys if isinstance(data.get(key), bool)}
    except (json.JSONDecodeError, AttributeError):
        verdicts = {}
    parse_ok = len(verdicts) == len(keys)
    if not parse_ok:
        logger.warning("Nugget judge output incomplete or invalid; missing nuggets count as not present")
    present = tuple(verdicts.get(key, False) for key in keys)
    return NuggetJudgment(present, parse_ok, response.latency_s)


def is_degenerate_answer(answer: str, title: str) -> bool:
    """Whether an answer is too empty to contain any fact, so it gets no nuggets without asking the judge.

    The judge was seen crediting facts to such answers (e.g. "No", or an answer that only repeats the question title).

    Args:
        answer: The assistant's answer.
        title: The question title.

    Returns:
        True if the normalised answer is shorter than MIN_ANSWER_CHARS, or equals or is part of the title.
    """
    a = normalize(answer).strip(" .!?:")
    t = normalize(title).strip(" .!?:")
    return len(a) < MIN_ANSWER_CHARS or a in t


def normalize(text: str) -> str:
    """Lowercase and collapse all whitespace to single spaces, for quote matching.

    Args:
        text: The text to normalise.

    Returns:
        The normalised text.
    """
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_source(quote: str, source: str, threshold: float = QUOTE_MATCH_THRESHOLD) -> bool:
    """Check whether a quote occurs in the source, tolerating small copying errors.

    Args:
        quote: The text the judge quoted.
        source: The source text the judge saw.
        threshold: The share of the quote that must occur as one contiguous block in the source.

    Returns:
        True if the normalised quote is long enough and its longest contiguous match in the normalised source covers
        at least `threshold` of the quote.
    """
    q, s = normalize(quote), normalize(source)
    if len(q) < MIN_QUOTE_CHARS:
        return False
    if q in s:
        return True
    match = SequenceMatcher(None, q, s, autojunk=False).find_longest_match(0, len(q), 0, len(s))
    return match.size / len(q) >= threshold


def judge_citation(judge: LLM, question: str, answer: str, doc_id: str, text: str, max_chars: int) -> CitationJudgment:
    """Ask the judge whether one cited chunk supports the answer, and verify its quote.

    Args:
        judge: The judge LLM.
        question: The full question.
        answer: The assistant's answer.
        doc_id: The cited chunk's doc_id.
        text: The chunk text.
        max_chars: Cut the chunk like the assistant's prompt did, so the judge sees what the assistant saw.

    Returns:
        The judgment. "Supported" requires the judge's yes AND a quote found in the source. Unusable output counts as
        not supported.
    """
    source = format_source(doc_id, text, max_chars)
    prompt = f"Question:\n{question}\n\nAnswer:\n{answer}\n\nCited source:\n{source}"
    response = judge.generate(prompt, system=CITATION_SYSTEM_PROMPT, json_schema=CITATION_SCHEMA, max_tokens=300)
    try:
        data = json.loads(response.text)
        judge_supported, quote = bool(data["supported"]), str(data["quote"])
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.warning(f"Citation judge output invalid for {doc_id}; counted as not supported")
        return CitationJudgment(doc_id, False, False, "", False, False, response.latency_s)
    # Check against the chunk text the judge saw, not the label around it: quoting the doc_id is no evidence.
    found = quote_in_source(quote, text[:max_chars])
    if judge_supported and not found:
        logger.debug(f"Judge said {doc_id} supports the answer, but its quote is not in the source: {quote[:80]!r}")
    return CitationJudgment(doc_id, judge_supported and found, judge_supported, quote, found, True, response.latency_s)
