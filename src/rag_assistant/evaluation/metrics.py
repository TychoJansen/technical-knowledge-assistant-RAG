"""Retrieval metrics as small pure functions over ranked doc_id lists (best first)."""

import math
from collections.abc import Collection, Sequence
from typing import Any


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Fraction of the relevant documents found in the top-k.

    Args:
        retrieved: Ranked doc_ids.
        relevant: The relevant doc_ids of the query.
        k: Cutoff.

    Returns:
        The recall, 0.0 if there are no relevant documents.
    """
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def ndcg_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """NDCG@k with binary relevance.

    Args:
        retrieved: Ranked doc_ids.
        relevant: The relevant doc_ids of the query.
        k: Cutoff.

    Returns:
        DCG of the top-k divided by the ideal DCG, 0.0 if there are no relevant documents.
    """
    relevant = set(relevant)
    dcg = sum(1 / math.log2(rank + 1) for rank, doc_id in enumerate(retrieved[:k], start=1) if doc_id in relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def mrr_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Reciprocal rank of the first relevant document in the top-k.

    Args:
        retrieved: Ranked doc_ids.
        relevant: The relevant doc_ids of the query.
        k: Cutoff.

    Returns:
        1 / rank of the first relevant document, 0.0 if there is none in the top-k.
    """
    relevant = set(relevant)
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1 / rank
    return 0.0


def coverage_at_k(retrieved: Sequence[str], nugget_relevant: Sequence[Collection[str]], k: int) -> float:
    """Fraction of nuggets supported by at least one top-k document (FreshStack Coverage@k).

    Args:
        retrieved: Ranked doc_ids.
        nugget_relevant: For each nugget of the query, the doc_ids that support it.
        k: Cutoff.

    Returns:
        The share of nuggets with a supporting document in the top-k, 0.0 if the query has no nuggets.
    """
    if not nugget_relevant:
        return 0.0
    top = set(retrieved[:k])
    return sum(1 for docs in nugget_relevant if top & set(docs)) / len(nugget_relevant)


def unjudged_at_k(retrieved: Sequence[str], judged: Collection[str], k: int) -> float:
    """Fraction of the top-k documents that have no relevance judgment at all for this query.

    Args:
        retrieved: Ranked doc_ids.
        judged: The doc_ids judged (relevant or not) for the query.
        k: Cutoff.

    Returns:
        The unjudged share of the top-k (of fewer if fewer were retrieved), 0.0 if nothing was retrieved.
    """
    top = retrieved[:k]
    if not top:
        return 0.0
    judged = set(judged)
    return sum(1 for doc_id in top if doc_id not in judged) / len(top)


def judged_doc_ids(query: dict[str, Any]) -> set[str]:
    """All doc_ids judged (relevant or non-relevant) for any nugget of the query.

    Args:
        query: A single query record (evaluation data).

    Returns:
        A set of all doc_ids that were judged for any nugget of the query.
    """
    return {d for n in query["nuggets"] for d in n["relevant_corpus_ids"] + n["non_relevant_corpus_ids"]}
