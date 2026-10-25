"""Abstention on insufficient context (`rag eval abstention`): does the assistant say the information is insufficient.

FreshStack only has answerable questions, so insufficient information is created from its own judgments: for each
question, the assistant gets only chunks that FreshStack's assessors judged and marked **not relevant** to that
question (the best BM25 matches among them, so they look as much like evidence as possible). None of these chunks
supports a key fact of the accepted answer, so the expected outcome is "insufficient information". Answering anyway
means answering without evidence: from the model's own knowledge or from chunks that do not answer the question.

The questions are real, on-topic questions of the split, so this follows the dev/test split and needs no judge.

Usage:
    poetry run rag eval abstention --limit 20       # the same seeded sample as `rag eval answers --limit 20`
    poetry run rag eval abstention                  # all dev questions (~35-40 s per question)
    poetry run rag eval abstention --split test     # final numbers only

Outputs in results/: abstention/<name>_<split>.jsonl, abstention_<name>_<split>.json and abstention_summary.md.
"""

import json
from collections.abc import Collection, Mapping
from typing import Any

from rag_assistant.assistant import ANSWER_RULES_VERSION, Assistant, load_assistant
from rag_assistant.config import (
    ABSTENTION_DIR,
    DEFAULT_MAX_CHARS,
    DEFAULT_MODEL,
    DEFAULT_TOKENIZER,
    DEFAULT_TOP_K,
    LOGS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
)
from rag_assistant.data import language, load_split_queries
from rag_assistant.evaluation.answer_eval import SAMPLE_SEED, mean, run, select_questions, warm_up
from rag_assistant.evaluation.report import format_table, load_reports
from rag_assistant.logger import add_run_log, get_logger
from rag_assistant.prompt import prompt_fingerprint
from rag_assistant.retrieval import BM25Retriever, RetrievedDoc, Retriever

logger = get_logger(__name__)

# Make sure runs of different versions are never mixed.
#   1: the top-k BM25 matches (full question) among the chunks judged not relevant to any nugget of the question,
#      excluding chunks whose text equals that of a relevant chunk.
CONTEXT_VERSION = 1

# Column label -> key in the report, for the summary table.
SUMMARY_COLUMNS = {
    "Correct abst": "correct_abstention_rate",
    "False answer": "false_answer_rate",
    "Invalid": "invalid_rate",
    "Cited false ans": "cited_false_answer_rate",
    "Chunks/question": "context_size_mean",
}


def insufficient_context_candidates(question: dict[str, Any], corpus: Mapping[str, str]) -> list[str]:
    """Return the chunks FreshStack judged not relevant to the question, and that cannot answer it.

    A chunk qualifies if an assessor judged it not relevant for at least one nugget, it is relevant for none (nor for
    the question), and its text is not identical to that of a relevant chunk (the corpus has duplicate files).

    Args:
        question: The query record with its nuggets and relevance judgments (evaluation data).
        corpus: Mapping from doc_id to chunk text.

    Returns:
        The qualifying doc_ids, sorted.
    """
    nuggets = question["nuggets"]
    relevant = set(question["relevant_doc_ids"]) | {d for n in nuggets for d in n["relevant_corpus_ids"]}
    relevant_texts = {corpus[d] for d in relevant if d in corpus}
    judged_not_relevant = {d for n in nuggets for d in n["non_relevant_corpus_ids"]} - relevant
    return sorted(d for d in judged_not_relevant if d in corpus and corpus[d] not in relevant_texts)


class RestrictedRetriever(Retriever):
    """A BM25 retriever that only returns documents from a fixed set (the insufficient context of one question)."""

    def __init__(self, base: BM25Retriever, doc_ids: Collection[str]) -> None:
        """Search with `base`, but only among `doc_ids`.

        Args:
            base: The BM25 retriever whose scores are used.
            doc_ids: The documents that may be returned.
        """
        self.base = base
        self.doc_ids = doc_ids

    def search(self, query: str, k: int) -> list[RetrievedDoc]:
        """Return the top-k documents of the fixed set for the query (see BM25Retriever.search_within).

        Args:
            query: The search query.
            k: The number of documents to return.

        Returns:
            The best-scoring documents of the fixed set, best first.
        """
        return self.base.search_within(query, k, self.doc_ids)


def evaluate_insufficient_context(question: dict[str, Any], assistant: Assistant) -> dict[str, Any]:
    """Ask one question with only insufficient context; the assistant should say the sources are not enough.

    Args:
        question: The query record (only question["question"] goes to the assistant; the judgments choose the context).
        assistant: The assistant under evaluation; its retriever must be a BM25Retriever.

    Returns:
        A per-question record with the answer and its citations.
    """
    candidates = insufficient_context_candidates(question, assistant.corpus)
    restricted = Assistant(
        RestrictedRetriever(assistant.retriever, candidates),
        assistant.llm,
        assistant.corpus,
        top_k=assistant.top_k,
        max_chars=assistant.max_chars,
        max_tokens=assistant.max_tokens,
    )
    answer = restricted.ask(question["question"])
    return {
        "qid": question["qid"],
        "kind": "insufficient_context",
        "language": language(question),
        "title": question["title"],
        "answer": answer.text,
        "valid_output": answer.valid_output,
        "insufficient_information": answer.insufficient_information,
        "retrieved": [hit.doc_id for hit in answer.retrieved],
        "n_candidates": len(candidates),
        "cited_doc_ids": list(answer.cited_doc_ids),
        "invalid_citations": list(answer.invalid_citations),
        "repaired_citations": [list(pair) for pair in answer.repaired_citations],
        "generation_s": answer.llm_response.latency_s,
        "prompt_tokens": answer.llm_response.input_tokens,
        "output_tokens": answer.llm_response.output_tokens,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Aggregate the per-question records: how often the assistant abstains, answers anyway, or fails.

    Args:
        records: The per-question records of the run.

    Returns:
        The run's metrics (see SUMMARY_COLUMNS), plus dropped citations, context size and generation time.
    """
    false_answers = [r for r in records if r["valid_output"] and not r["insufficient_information"]]
    return {
        "n": len(records),
        "correct_abstention_rate": mean([float(r["valid_output"] and r["insufficient_information"]) for r in records]),
        "false_answer_rate": len(false_answers) / len(records),
        "invalid_rate": mean([float(not r["valid_output"]) for r in records]),
        # Answering anyway AND citing one of the (non-relevant) chunks as if it backed the answer.
        "cited_false_answer_rate": mean([float(bool(r["cited_doc_ids"])) for r in false_answers]),
        "answers_with_dropped_citations_rate": mean([float(bool(r["invalid_citations"])) for r in records]),
        "context_size_mean": mean([float(len(r["retrieved"])) for r in records]),
        "generation_s_mean": mean([r["generation_s"] for r in records]),
    }


def write_summary(reports: list[dict[str, Any]]) -> str:
    """Rewrite abstention_summary.md from all saved abstention runs.

    Args:
        reports: The saved reports of all abstention runs.

    Returns:
        The summary table as Markdown.
    """
    rows = []
    for r in reports:
        # The prompt fingerprint, so runs made with different prompts are never read as comparable.
        cells = [r["name"], r["split"], r["config"].get("prompt", "-"), str(r["metrics"]["n"])]
        for key in SUMMARY_COLUMNS.values():
            value = r["metrics"].get(key)
            cells.append("-" if value is None else f"{value:.3f}")
        rows.append(cells)
    table = format_table(["Run", "Split", "Prompt", "n", *SUMMARY_COLUMNS], rows)
    path = RESULTS_DIR / "abstention_summary.md"
    path.write_text(
        "# Abstention on insufficient context\n\n"
        "Generated by `rag eval abstention`; do not edit by hand. Every question gets only chunks FreshStack judged "
        'not relevant to it, so the correct outcome is always "insufficient information".\n\n'
        "- **Correct abst**: share of questions where the assistant says the sources are not sufficient.\n"
        "- **Prompt**: fingerprint of the assistant prompt; compare only runs with the same one.\n"
        "- **False answer**: share where it answers anyway instead of saying the sources are not sufficient.\n"
        "- **Invalid**: share with unparseable output.\n"
        "- **Cited false ans**: share of those false answers that cite one of the non-relevant chunks as a source.\n"
        "- **Chunks/question**: mean number of chunks in the context (below top-k when few judged chunks match).\n\n"
        f"{table}\n",
        encoding="utf-8",
    )
    logger.info(f"[write] {path.relative_to(PROJECT_ROOT)}")
    return table


def evaluate_abstention(
    split: str = "dev",
    limit: int | None = None,
    tokenizer: str = DEFAULT_TOKENIZER,
    model: str = DEFAULT_MODEL,
    top_k: int = DEFAULT_TOP_K,
    max_chars: int = DEFAULT_MAX_CHARS,
    name: str | None = None,
    restart: bool = False,
) -> None:
    """Run the insufficient-context evaluation and write all outputs.

    Args:
        split: The data split ("dev" or "test"; test only for final numbers).
        limit: A seeded random sample of N questions (the same sample as `rag eval answers --limit N`).
        tokenizer: The tokenizer of the BM25 index, used to rank the non-relevant chunks.
        model: Ollama model of the assistant.
        top_k: Number of chunks given to the assistant.
        max_chars: Maximum number of characters per chunk.
        name: Run name (default: derived from the settings, as in `rag eval answers`).
        restart: Discard earlier results of this run.
    """
    if split == "test":
        logger.warning("Evaluating on the TEST split. Use it only for final numbers, never for tuning.")

    if not name:
        name = f"bm25_{tokenizer}_{model.replace(':', '-')}_k{top_k}_c{max_chars}"
        if limit:
            name += f"_sample{limit}"

    questions = select_questions(load_split_queries(split), limit)
    ABSTENTION_DIR.mkdir(parents=True, exist_ok=True)
    path = ABSTENTION_DIR / f"{name}_{split}.jsonl"
    if restart:
        path.unlink(missing_ok=True)
    add_run_log(LOGS_DIR / f"abstention_{name}_{split}.log", restart)

    assistant = load_assistant(tokenizer=tokenizer, model=model, top_k=top_k, max_chars=max_chars)
    # Everything that changes the answers, so a resumed run never mixes two systems.
    config = {
        "tokenizer": tokenizer,
        "top_k": top_k,
        "max_chars": max_chars,
        "max_tokens": assistant.max_tokens,
        "prompt": prompt_fingerprint(),
        "answer_rules": ANSWER_RULES_VERSION,
        "assistant_llm": assistant.llm.settings(),
        "context_version": CONTEXT_VERSION,
        "sample": {"limit": limit, "seed": SAMPLE_SEED} if limit else None,
    }
    warm_up(assistant.llm)
    records = run(questions, path, config, lambda q: evaluate_insufficient_context(q, assistant))

    report = {"name": name, "split": split, "config": config, "metrics": aggregate(records)}
    report_path = RESULTS_DIR / f"abstention_{name}_{split}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info(f"[write] {path.relative_to(PROJECT_ROOT)}")
    logger.info(f"[write] {report_path.relative_to(PROJECT_ROOT)}")

    table = write_summary(load_reports("abstention"))
    logger.info("Abstention results (all runs):\n" + table)
    logger.info("This run:\n" + json.dumps(report, indent=2))
