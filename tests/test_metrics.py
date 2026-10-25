"""Unit tests for the retrieval metrics on hand-made examples."""

import math

import pytest

from rag_assistant.evaluation.metrics import coverage_at_k, mrr_at_k, ndcg_at_k, recall_at_k, unjudged_at_k

RETRIEVED = ["a", "b", "c", "d", "e"]


def test_recall_counts_only_top_k():
    assert recall_at_k(RETRIEVED, {"b", "e", "z"}, k=2) == pytest.approx(1 / 3)
    assert recall_at_k(RETRIEVED, {"b", "e", "z"}, k=5) == pytest.approx(2 / 3)


def test_recall_without_relevant_docs_is_zero():
    assert recall_at_k(RETRIEVED, set(), k=5) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(RETRIEVED, {"a", "b"}, k=5) == pytest.approx(1.0)


def test_ndcg_binary_relevance():
    # One relevant doc at rank 2, two relevant docs in total.
    expected = (1 / math.log2(3)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(RETRIEVED, {"b", "z"}, k=5) == pytest.approx(expected)


def test_ndcg_ignores_hits_after_k():
    assert ndcg_at_k(RETRIEVED, {"e"}, k=3) == 0.0


def test_mrr_uses_first_relevant_rank():
    assert mrr_at_k(RETRIEVED, {"c", "d"}, k=10) == pytest.approx(1 / 3)
    assert mrr_at_k(RETRIEVED, {"e"}, k=3) == 0.0


def test_coverage_counts_supported_nuggets():
    nuggets = [{"a", "x"}, {"y"}, {"d"}]
    assert coverage_at_k(RETRIEVED, nuggets, k=2) == pytest.approx(1 / 3)
    assert coverage_at_k(RETRIEVED, nuggets, k=5) == pytest.approx(2 / 3)


def test_unjudged_fraction_of_top_k():
    assert unjudged_at_k(RETRIEVED, {"a", "c"}, k=4) == pytest.approx(2 / 4)
    assert unjudged_at_k([], {"a"}, k=10) == 0.0
