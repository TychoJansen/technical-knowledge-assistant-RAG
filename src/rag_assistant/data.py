"""Loaders for the local data files created by `rag prepare` (see rag_assistant.prepare)."""

import json
from typing import Any

import pandas as pd

from rag_assistant.config import CORPUS_PATH, QUERIES_PATH, SPLITS_PATH

JS_TAGS = {"javascript", "node.js", "typescript", "langchain-js", "langchainjs"}


def load_corpus(columns: list[str] | None = None) -> pd.DataFrame:
    """Load the corpus chunks.

    Args:
        columns: Optional subset of columns to read.

    Returns:
        DataFrame with columns doc_id, text, url, start_byte, end_byte, repo (or the requested subset).
    """
    return pd.read_parquet(CORPUS_PATH, columns=columns)


def load_corpus_map(column: str) -> dict[str, Any]:
    """Load one corpus column as a mapping from doc_id to its value.

    Args:
        column: Corpus column to map to, e.g. "text" or "url".

    Returns:
        Dict from doc_id to the value of that column.
    """
    return load_corpus(columns=["doc_id", column]).set_index("doc_id")[column].to_dict()


def load_queries() -> list[dict[str, Any]]:
    """Load the evaluation queries (evaluation only, never input for the assistant).

    Returns:
        List of query records, with the keys described in rag_assistant.prepare.prepare_queries.
    """
    with QUERIES_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split() -> dict[str, Any]:
    """Load the fixed dev/test split.

    Returns:
        Dict with keys "seed" (int), "dev" and "test" (lists of query IDs).
    """
    return json.loads(SPLITS_PATH.read_text(encoding="utf-8"))


def load_split_queries(split: str) -> list[dict[str, Any]]:
    """Load the evaluation queries of one split, in split order.

    Args:
        split: Name of the split, "dev" or "test".

    Returns:
        List of query records in that split.
    """
    qids = load_split()[split]
    by_qid = {q["qid"]: q for q in load_queries()}
    return [by_qid[qid] for qid in qids]


def language(query: dict[str, Any]) -> str:
    """Classify a query as JavaScript or Python based on its tags.

    Args:
        query: Query record with a "tags" key.

    Returns:
        "js" if any tag is a JavaScript tag, otherwise "python".
    """
    return "js" if any(t.lower() in JS_TAGS for t in query["tags"]) else "python"
