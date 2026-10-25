"""The assistant: retrieve chunks for a question, let the LLM answer from them, and check the citations."""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag_assistant.config import DEFAULT_MAX_CHARS, DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_TOKENIZER, DEFAULT_TOP_K
from rag_assistant.data import load_corpus_map
from rag_assistant.llm import LLM, LLMResponse, OllamaLLM
from rag_assistant.logger import get_logger
from rag_assistant.prompt import ANSWER_SCHEMA, INSUFFICIENT_TEXT, SYSTEM_PROMPT, build_prompt
from rag_assistant.retrieval import RetrievedDoc, Retriever, load_retriever

logger = get_logger(__name__)

# Make sure runs of different versions are never mixed.
ANSWER_RULES_VERSION = 4

# A short answer with this wording is a refusal, also when the model did not set insufficient_information.
MAX_REFUSAL_CHARS = 100
_REFUSAL = re.compile(
    r"\b(can ?not|can't|unable to) (answer|help)"
    r"|(\bnot|n't) (contain |have |provide |include )?enough information"
    r"|\b(insufficient|no relevant) information",
    re.IGNORECASE,
)

# The byte-range suffix of a doc_id, e.g. "_0_2044" in "README.md_0_2044".
_BYTE_RANGE = re.compile(r"_\d+_\d+$")


@dataclass(frozen=True)
class Answer:
    """The assistant's answer to one question.

    Attributes:
        question: The question that was asked.
        text: The answer text.
        cited_doc_ids: Cited doc_ids that were among the retrieved sources (the valid citations).
        invalid_citations: Cited doc_ids that were NOT among the retrieved sources and could not be repaired; dropped
            from cited_doc_ids.
        insufficient_information: True if the model said the sources do not contain enough information.
        valid_output: False if the model's output could not be parsed (e.g. invalid or cut-off JSON).
        retrieved: The chunks given to the model, best first.
        llm_response: The raw LLM response, with token counts and latency.
        repaired_citations: (cited, repaired) pairs: a mangled doc_id that matched exactly one provided chunk, which is
            then in cited_doc_ids (see repair_citation).
    """

    question: str
    text: str
    cited_doc_ids: tuple[str, ...]
    invalid_citations: tuple[str, ...]
    insufficient_information: bool
    valid_output: bool
    retrieved: tuple[RetrievedDoc, ...]
    llm_response: LLMResponse
    repaired_citations: tuple[tuple[str, str], ...] = ()


class Assistant:
    """Answers questions from the corpus with retrieval plus an LLM, citing chunks by doc_id."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        corpus: Mapping[str, str],
        top_k: int = DEFAULT_TOP_K,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        """Configure the assistant.

        Args:
            retriever: Finds the chunks for a question.
            llm: Generates the answer.
            corpus: Mapping from doc_id to chunk text.
            top_k: Number of retrieved chunks given to the LLM.
            max_chars: Maximum number of characters per chunk in the prompt.
            max_tokens: Maximum number of tokens the LLM may generate.
        """
        self.retriever = retriever
        self.llm = llm
        self.corpus = corpus
        self.top_k = top_k
        self.max_chars = max_chars
        self.max_tokens = max_tokens

    def ask(self, question: str, retrieval_query: str | None = None) -> Answer:
        """Answer a question from the corpus.

        Args:
            question: The question (only the question text; never evaluation data). The model answers this.
            retrieval_query: Optional shorter text to search with (e.g. only the title, see rag_assistant.query);
                default: the question itself.

        Returns:
            The answer with its valid citations, dropped citations and the retrieved chunks.
        """
        retrieved = tuple(self.retriever.search(retrieval_query or question, self.top_k))
        sources = [(hit.doc_id, self.corpus[hit.doc_id]) for hit in retrieved]
        prompt = build_prompt(question, sources, self.max_chars)
        response = self.llm.generate(
            prompt, system=SYSTEM_PROMPT, json_schema=ANSWER_SCHEMA, max_tokens=self.max_tokens
        )
        answer = parse_answer(question, response, retrieved)
        logger.debug(
            f"Answered in {response.latency_s:.1f}s: valid={answer.valid_output} "
            f"insufficient={answer.insufficient_information} cited={list(answer.cited_doc_ids)} "
            f"invalid={list(answer.invalid_citations)}"
        )
        return answer


def _file_name(doc_id: str) -> str:
    """Return the last path part of a doc_id.

    Args:
        doc_id: A (possibly mangled) doc_id.

    Returns:
        The file name with byte range, e.g. "README.md_0_2044".
    """
    return doc_id.rsplit("/", 1)[-1]


def _file_stem(doc_id: str) -> str:
    """Return the file name without byte range and extension.

    Args:
        doc_id: A (possibly mangled) doc_id.

    Returns:
        The file stem, e.g. "README" (also "pinecone" for a mangled "pinecone.ipb").
    """
    return _BYTE_RANGE.sub("", _file_name(doc_id)).split(".", 1)[0]


def repair_citation(cited: str, provided: Sequence[str]) -> str | None:
    """Map a mangled citation to the provided chunk it clearly means, or None.

    The small model often copies a long doc_id wrongly: it leaves out the middle of the path, writes the wrong
    repository (langchain vs langchainjs) or cuts off the extension and byte range. A citation is repaired only if it
    points to exactly one provided chunk, first by file name with byte range, then by file name alone. A citation that
    matches nothing or more than one chunk stays invalid.

    Args:
        cited: The doc_id the model cited, which is not among the provided ones.
        provided: The doc_ids of the chunks given to the model.

    Returns:
        The provided doc_id it means, or None.
    """
    for key in (_file_name, _file_stem):
        matches = [doc_id for doc_id in provided if key(doc_id) == key(cited)]
        if len(matches) == 1:
            return matches[0]
    return None


def is_refusal_text(text: str) -> bool:
    """Return whether an answer is only a refusal, e.g. "I can't answer that question".

    The model sometimes refuses in words but leaves insufficient_information false, so the refusal would count as an
    answer (and a lenient judge may even credit facts to it). Only short answers count: a longer answer that names the
    part it cannot answer is a partial answer, which the prompt allows.

    Args:
        text: The answer text.

    Returns:
        True for the standard refusal sentence (INSUFFICIENT_TEXT), or a short answer with refusal wording.
    """
    if " ".join(text.split()).lower() == INSUFFICIENT_TEXT.lower():
        return True
    return len(text.strip()) <= MAX_REFUSAL_CHARS and bool(_REFUSAL.search(text))


def parse_answer(question: str, response: LLMResponse, retrieved: tuple[RetrievedDoc, ...]) -> Answer:
    """Parse the LLM's JSON output and check its citations against the retrieved chunks.

    Rules:
    - unparseable output gives valid_output=False and no citations;
    - an empty answer, insufficient_information=true, or a short answer that is only a refusal (is_refusal_text) is a
      refusal: it gets INSUFFICIENT_TEXT and no citations (the prompt asks for one sentence and no citations; the 3B
      model often writes the field name as its answer or cites anyway);
    - a citation to a chunk that was not provided is repaired if it matches exactly one provided chunk
      (repair_citation), and otherwise moved to invalid_citations.

    Args:
        question: The question that was asked.
        response: The LLM response, expected to be JSON matching ANSWER_SCHEMA.
        retrieved: The chunks that were given to the LLM.

    Returns:
        The checked answer.
    """
    try:
        data = json.loads(response.text)
        text = str(data["answer"])
        cited = [str(doc_id) for doc_id in data["cited_doc_ids"]]
        insufficient = bool(data["insufficient_information"])
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        reason = "output cut off at max_tokens" if response.truncated else f"invalid output ({type(e).__name__})"
        logger.warning(f"Could not parse the LLM answer: {reason}")
        return Answer(question, f"[No valid answer: {reason}]", (), (), False, False, retrieved, response)

    if not text.strip():
        # A small model sometimes returns an empty answer without flagging it; an empty answer is never useful.
        logger.warning("The LLM returned an empty answer; treating it as insufficient information")
        return Answer(question, INSUFFICIENT_TEXT, (), (), True, True, retrieved, response)
    if not insufficient and is_refusal_text(text):
        logger.warning(
            f"The LLM refused in words but did not set insufficient_information: {text!r}; treated as a refusal"
        )
        insufficient = True
    if insufficient:
        # A refusal cites nothing and gets one clear sentence, whatever the model wrote (often the field name itself).
        if cited:
            logger.debug(f"Ignored {len(cited)} citation(s) on a refusal: {cited}")
        return Answer(question, INSUFFICIENT_TEXT, (), (), True, True, retrieved, response)

    provided = [hit.doc_id for hit in retrieved]
    valid: list[str] = []
    invalid: list[str] = []
    repaired: list[tuple[str, str]] = []
    for doc_id in dict.fromkeys(cited):  # drop duplicates, keep order
        target = doc_id if doc_id in provided else repair_citation(doc_id, provided)
        if target is None:
            invalid.append(doc_id)
        elif target not in valid:
            valid.append(target)
            if target != doc_id:
                repaired.append((doc_id, target))
    if repaired:
        logger.warning(f"Repaired {len(repaired)} citation(s) to the provided chunk they clearly mean: {repaired}")
    if invalid:
        logger.warning(f"Dropped {len(invalid)} citation(s) to doc_ids that were not provided: {invalid}")
    return Answer(question, text, tuple(valid), tuple(invalid), False, True, retrieved, response, tuple(repaired))


def load_assistant(
    tokenizer: str = DEFAULT_TOKENIZER,
    model: str = DEFAULT_MODEL,
    top_k: int = DEFAULT_TOP_K,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Assistant:
    """Create an assistant from the saved BM25 index, the local corpus and a local Ollama model.

    Args:
        tokenizer: The tokenizer the index was built with.
        model: The Ollama model name.
        top_k: Number of retrieved chunks given to the LLM.
        max_chars: Maximum number of characters per chunk in the prompt.
        max_tokens: Maximum number of tokens the LLM may generate.

    Returns:
        A ready-to-use assistant.
    """
    return Assistant(
        retriever=load_retriever(tokenizer),
        llm=OllamaLLM(model=model),
        corpus=load_corpus_map("text"),
        top_k=top_k,
        max_chars=max_chars,
        max_tokens=max_tokens,
    )
