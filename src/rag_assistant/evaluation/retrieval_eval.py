"""Evaluate the BM25 retriever on the dev (default) or test split (`rag eval retrieval`).

The retriever only sees the question text (title + body). Relevance judgments and nuggets are used for scoring only.

Usage:
    poetry run rag eval retrieval --tokenizer word_stem
    poetry run rag eval retrieval --tokenizer word_stem --split test   # final numbers only

Outputs (in results/):
    runs/<name>_<split>.json        top-100 doc_ids with scores per qid (input for hybrid/reranking later)
    retrieval_<name>_<split>.json   metrics overall, per language and per query, plus latency
    retrieval_summary.md            one row per run, rebuilt from all retrieval_*.json files
    <name>_worst_<split>.md         the questions with the lowest Recall@50, for error analysis
"""

import json
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from rag_assistant.config import LOGS_DIR, PROJECT_ROOT, RESULTS_DIR, RUNS_DIR
from rag_assistant.data import language, load_split_queries
from rag_assistant.evaluation.metrics import (
    coverage_at_k,
    judged_doc_ids,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    unjudged_at_k,
)
from rag_assistant.evaluation.report import format_table, load_reports
from rag_assistant.logger import add_run_log, get_logger
from rag_assistant.query import retrieval_query
from rag_assistant.retrieval import RetrievedDoc, Retriever, load_retriever

logger = get_logger(__name__)

RUN_DEPTH = 100
N_WORST = 5
N_TOP_SHOWN = 5
LANGUAGES = ("js", "python")

# Column label -> function(retrieved doc_ids, query) -> score.
METRICS: dict[str, Callable[[list[str], dict[str, Any]], float]] = {
    "R@10": lambda r, q: recall_at_k(r, q["relevant_doc_ids"], 10),
    "R@50": lambda r, q: recall_at_k(r, q["relevant_doc_ids"], 50),
    "R@100": lambda r, q: recall_at_k(r, q["relevant_doc_ids"], 100),
    "nDCG@10": lambda r, q: ndcg_at_k(r, q["relevant_doc_ids"], 10),
    "MRR@10": lambda r, q: mrr_at_k(r, q["relevant_doc_ids"], 10),
    "Cov@20": lambda r, q: coverage_at_k(r, [n["relevant_corpus_ids"] for n in q["nuggets"]], 20),
    "Unj@10": lambda r, q: unjudged_at_k(r, judged_doc_ids(q), 10),
}


def run_retrieval(
    retriever: Retriever, queries: list[dict[str, Any]], k: int, query_mode: str = "full"
) -> tuple[dict[str, list[RetrievedDoc]], list[float]]:
    """Search every question and return the hits per qid and the latency per query in ms.

    Args:
        retriever: The retriever to evaluate.
        queries: A list of query dictionaries from the split JSON.
        k: The number of top documents to retrieve per query.
        query_mode: Which part of the question to search with (see rag_assistant.query).

    Returns:
        A tuple of (results, latencies), where results is a dict mapping qid to a list of RetrievedDoc, and latencies is a list of retrieval times in milliseconds.
    """
    results: dict[str, list[RetrievedDoc]] = {}
    latencies: list[float] = []
    for query in queries:
        start = time.perf_counter()
        results[query["qid"]] = retriever.search(retrieval_query(query, query_mode), k)
        latencies.append((time.perf_counter() - start) * 1000)
    return results, latencies


def score_queries(queries: list[dict[str, Any]], results: dict[str, list[RetrievedDoc]]) -> dict[str, dict[str, Any]]:
    """Compute all metrics per query.

    Args:
        queries: A list of query dictionaries from the split JSON.
        results: A dict mapping qid to a list of RetrievedDoc for that query.

    Returns:
        A dict mapping qid to a dict of metrics for that query, including the language and all scores defined in METRICS.
    """
    per_query = {}
    for query in queries:
        retrieved = [hit.doc_id for hit in results[query["qid"]]]
        scores = {name: fn(retrieved, query) for name, fn in METRICS.items()}
        per_query[query["qid"]] = {"language": language(query), **scores}
    return per_query


def aggregate(per_query: dict[str, dict[str, Any]], language_filter: str | None = None) -> dict[str, float]:
    """Average each metric over all queries, or over the queries of one language.

    Args:
        per_query: A dict mapping qid to a dict of metrics for that query.
        language_filter: If not None, only include queries of this language in the average.

    Returns:
        A dict mapping metric names to their average values.
    """
    rows = [m for m in per_query.values() if language_filter in (None, m["language"])]
    summary: dict[str, float] = {"n": len(rows)}
    for name in METRICS:
        summary[name] = float(np.mean([m[name] for m in rows])) if rows else float("nan")
    return summary


def save_run(name: str, split: str, results: dict[str, list[RetrievedDoc]]) -> None:
    """Save the raw ranked results so later stages (hybrid, reranking) can reuse them.

    Args:
        name: The run name (default bm25_<tokenizer>).
        split: The data split ("dev" or "test").
        results: A dict mapping qid to a list of RetrievedDoc for that query.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{name}_{split}.json"
    run = {qid: [{"doc_id": h.doc_id, "score": h.score} for h in hits] for qid, hits in results.items()}
    path.write_text(json.dumps({"name": name, "split": split, "depth": RUN_DEPTH, "results": run}), encoding="utf-8")
    logger.info(f"[write] {path.relative_to(PROJECT_ROOT)}")


def save_report(report: dict[str, Any]) -> None:
    """Save the metrics of one run.

    Args:
        report: A dict containing the run name, split, retriever, tokenizer, depth, metrics, latency and per-query scores.
    """
    path = RESULTS_DIR / f"retrieval_{report['name']}_{report['split']}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info(f"[write] {path.relative_to(PROJECT_ROOT)}")


def summary_tables(reports: list[dict[str, Any]]) -> tuple[str, str]:
    """Return the overall table and the per-language table for all runs.

    Args:
        reports: A list of report dicts, each containing the run name, split, retriever, tokenizer, depth, metrics, latency and per-query scores.

    Returns:
        A tuple of the overall table and the per-language table.
    """
    overall_rows = [
        [r["name"], r["split"], str(r["metrics"]["all"]["n"])]
        + [f"{r['metrics']['all'][m]:.3f}" for m in METRICS]
        + [f"{r['latency_ms']['mean']:.1f}", f"{r['latency_ms']['p95']:.1f}"]
        for r in reports
    ]
    overall = format_table(["Run", "Split", "n", *METRICS, "Lat mean ms", "Lat p95 ms"], overall_rows)

    lang_rows = [
        [r["name"], r["split"], lang, str(r["metrics"][lang]["n"])] + [f"{r['metrics'][lang][m]:.3f}" for m in METRICS]
        for r in reports
        for lang in LANGUAGES
    ]
    by_language = format_table(["Run", "Split", "Lang", "n", *METRICS], lang_rows)
    return overall, by_language


def write_summary(overall: str, by_language: str) -> None:
    """Rewrite retrieval_summary.md (the ablation table) from all saved runs.

    Args:
        overall: The overall table as a Markdown string.
        by_language: The per-language table as a Markdown string.
    """
    path = RESULTS_DIR / "retrieval_summary.md"
    path.write_text(
        "# Retrieval results\n\n"
        "Generated by `rag eval retrieval`; do not edit by hand. Recall is a lower bound: "
        "`Unj@10` is the share of the top-10 that was never judged.\n\n"
        f"## Overall\n\n{overall}\n\n## By language\n\n{by_language}\n",
        encoding="utf-8",
    )
    logger.info(f"[write] {path.relative_to(PROJECT_ROOT)}")


def doc_label(doc_id: str, query: dict[str, Any]) -> str:
    """Describe the judgment of a doc for a query: relevant, non-relevant or unjudged.

    Args:
        doc_id: The ID of the document.
        query: The query dictionary.

    Returns:
        A string indicating the judgment: "relevant", "non-relevant", or "unjudged".
    """
    if doc_id in query["relevant_doc_ids"]:
        return "relevant"
    return "non-relevant" if doc_id in judged_doc_ids(query) else "unjudged"


def write_worst(
    name: str,
    split: str,
    queries: list[dict[str, Any]],
    results: dict[str, list[RetrievedDoc]],
    per_query: dict[str, dict[str, Any]],
) -> None:
    """Write the questions with the lowest Recall@50 with their top hits and missed relevant docs.

    Args:
        name: The run name (default bm25_<tokenizer>).
        split: The data split ("dev" or "test").
        queries: A list of query dictionaries from the split JSON.
        results: A dict mapping qid to a list of RetrievedDoc for that query.
        per_query: A dict mapping qid to a dict of metrics for that query.
    """
    worst = sorted(queries, key=lambda q: (per_query[q["qid"]]["R@50"], q["qid"]))[:N_WORST]
    lines = [f"# {name}: {N_WORST} {split} questions with the lowest Recall@50\n"]
    for query in worst:
        hits = results[query["qid"]]
        rank_of = {h.doc_id: h.rank for h in hits}
        top50 = {h.doc_id for h in hits[:50]}
        missed = [d for d in query["relevant_doc_ids"] if d not in top50]
        m = per_query[query["qid"]]
        lines += [
            f"## {query['title']}\n",
            f"- qid: {query['qid']} | language: {m['language']} | tags: {', '.join(query['tags'])}",
            f"- question length: {len(query['question'])} chars",
            f"- R@50: {m['R@50']:.2f} | R@100: {m['R@100']:.2f} | Cov@20: {m['Cov@20']:.2f} | Unj@10: {m['Unj@10']:.2f}\n",
            f"Top-{N_TOP_SHOWN} retrieved:\n",
            *[
                f"{h.rank}. `{h.doc_id}` ({doc_label(h.doc_id, query)}, score {h.score:.1f})"
                for h in hits[:N_TOP_SHOWN]
            ],
            f"\nMissed relevant docs (not in top-50), {len(missed)} of {len(query['relevant_doc_ids'])}:\n",
            *[f"- `{d}`" + (f" (rank {rank_of[d]})" if d in rank_of else "") for d in missed],
            "",
        ]
    path = RESULTS_DIR / f"{name}_worst_{split}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"[write] {path.relative_to(PROJECT_ROOT)}")


def evaluate_retrieval(tokenizer: str, split: str = "dev", name: str | None = None, query_mode: str = "full") -> None:
    """Evaluate the BM25 retriever with this tokenizer on a split and write all outputs.

    Args:
        tokenizer: The tokenizer the index was built with.
        split: The data split ("dev" or "test"; test only for final numbers).
        name: The run name (default: bm25_<tokenizer>, plus _q-<mode> for another query mode than "full").
        query_mode: Which part of the question to search with: "full", "title" or "title_first_paragraph".
    """
    name = name or f"bm25_{tokenizer}" + ("" if query_mode == "full" else f"_q-{query_mode}")
    # A retrieval run is recomputed from scratch every time, so its log is too.
    add_run_log(LOGS_DIR / f"retrieval_{name}_{split}.log", restart=True)
    if split == "test":
        logger.warning("Evaluating on the TEST split. Use it only for final numbers, never for tuning.")

    queries = load_split_queries(split)
    retriever = load_retriever(tokenizer)
    logger.info(f"Searching {len(queries)} {split} questions with {name}")
    results, latencies = run_retrieval(retriever, queries, RUN_DEPTH, query_mode)
    per_query = score_queries(queries, results)

    report = {
        "name": name,
        "split": split,
        "retriever": "bm25",
        "tokenizer": tokenizer,
        "query_mode": query_mode,
        "depth": RUN_DEPTH,
        "metrics": {"all": aggregate(per_query), **{lang: aggregate(per_query, lang) for lang in LANGUAGES}},
        "latency_ms": {"mean": float(np.mean(latencies)), "p95": float(np.percentile(latencies, 95))},
        "per_query": per_query,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_run(name, split, results)
    save_report(report)
    write_worst(name, split, queries, results, per_query)

    overall, by_language = summary_tables(load_reports("retrieval"))
    write_summary(overall, by_language)
    logger.info("Retrieval results (all runs):\n" + overall + "\n\nBy language:\n" + by_language)
