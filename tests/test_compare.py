"""Unit tests for the paired comparison of two answer-evaluation runs."""

import pytest

from rag_assistant.evaluation.compare import FRESHSTACK_METRICS, bootstrap_ci, paired_stats


def record(answered: bool, coverage: float, seconds: float) -> dict:
    return {
        "valid_output": True,
        "insufficient_information": not answered,
        "context_coverage": coverage,
        "citations": [],
        "generation_s": seconds,
        "nuggets_present": None,
    }


def test_bootstrap_ci_contains_the_mean_and_is_reproducible():
    diffs = [1.0, 2.0, 3.0, 4.0]
    low, high = bootstrap_ci(diffs, n=2000)
    assert low <= 2.5 <= high
    assert (low, high) == bootstrap_ci(diffs, n=2000)


def test_bootstrap_ci_of_constant_differences_is_that_constant():
    assert bootstrap_ci([-5.0] * 10, n=500) == (-5.0, -5.0)


def test_paired_stats_counts_and_differences():
    a = {"q1": record(True, 0.5, 40), "q2": record(True, 0.0, 50), "q3": record(False, 1.0, 30)}
    b = {"q1": record(True, 0.5, 30), "q2": record(False, 0.0, 35), "q3": record(True, 1.0, 20)}
    answered = paired_stats(a, b, FRESHSTACK_METRICS["answered"])
    assert answered["n"] == 3 and answered["b_higher"] == 1 and answered["b_lower"] == 1 and answered["same"] == 1
    speed = paired_stats(a, b, FRESHSTACK_METRICS["generation_s"])
    assert speed["diff"] == pytest.approx(-35 / 3)
    assert speed["ci_high"] < 0  # B is faster on every question


def test_paired_stats_skips_questions_without_a_value():
    a = {"q1": record(True, 0.5, 40)}
    b = {"q1": record(True, 0.5, 30)}
    assert paired_stats(a, b, FRESHSTACK_METRICS["nugget_recall (judge)"]) == {"n": 0}  # no judge in either run
