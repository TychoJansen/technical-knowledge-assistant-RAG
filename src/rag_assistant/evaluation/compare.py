"""Paired comparison of two runs of the same evaluation (`rag eval compare`).

Both runs answered the same questions, so they are compared per question: for each metric, the difference B - A per
question, its mean, and a 95% bootstrap confidence interval of that mean. A difference whose interval includes 0 is
not distinguishable from noise with this many questions.

Usage:
    poetry run rag eval compare --a <run A> --b <run B>               # answer runs (results/answers/<name>_dev.jsonl)
    poetry run rag eval compare --a <run A> --b <run B> --abstention  # abstention runs (results/abstention/...)
"""

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from rag_assistant.config import ABSTENTION_DIR, ANSWERS_DIR, PROJECT_ROOT, RESULTS_DIR
from rag_assistant.evaluation.report import format_table
from rag_assistant.logger import get_logger

logger = get_logger(__name__)

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42


def _answered(r: dict[str, Any]) -> bool:
    """Return whether a record is an answer: valid output and not a refusal."""
    return r["valid_output"] and not r["insufficient_information"]


def _citation_precision(r: dict[str, Any]) -> float | None:
    """Return the share of an answer's citations that FreshStack marks relevant, None without citations."""
    citations = r["citations"] if _answered(r) else []
    return float(np.mean([c["relevant_in_qrels"] for c in citations])) if citations else None


def _nugget_recall(r: dict[str, Any]) -> float | None:
    """Return the share of nuggets the judge found in the answer, None if the record was not judged."""
    present = r["nuggets_present"]
    return sum(present) / len(present) if present is not None else None


# Metric -> per-question value (None when it does not apply to that question). Deterministic metrics first.
FRESHSTACK_METRICS: dict[str, Callable[[dict[str, Any]], float | None]] = {
    "context_coverage": lambda r: r["context_coverage"],
    "answered": lambda r: float(_answered(r)),
    "false_abstention": lambda r: float(r["valid_output"] and r["insufficient_information"]),
    "citation_precision_qrels": _citation_precision,
    "generation_s": lambda r: r["generation_s"],
    "nugget_recall (judge)": _nugget_recall,
}
# Insufficient-context runs (rag eval abstention): every question should be answered with "insufficient".
ABSTENTION_METRICS: dict[str, Callable[[dict[str, Any]], float | None]] = {
    "correct_abstention": lambda r: float(r["valid_output"] and r["insufficient_information"]),
    "false_answer": lambda r: float(_answered(r)),
    "generation_s": lambda r: r["generation_s"],
}

# Evaluation -> (results directory, record kind, metrics).
EVALUATIONS: dict[str, tuple[Path, str, dict[str, Callable[[dict[str, Any]], float | None]]]] = {
    "answers": (ANSWERS_DIR, "freshstack", FRESHSTACK_METRICS),
    "abstention": (ABSTENTION_DIR, "insufficient_context", ABSTENTION_METRICS),
}


def load_run(name: str, split: str, results_dir: Path = ANSWERS_DIR) -> dict[str, dict[str, Any]]:
    """Load the per-question records of a run, keyed by qid.

    Args:
        name: The run name.
        split: The split of the run.
        results_dir: The directory with the run's JSONL file (answers or abstention).

    Returns:
        qid -> record.

    Raises:
        SystemExit: If the run has no results file.
    """
    path = results_dir / f"{name}_{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"No evaluation results at {path}")
    return {r["qid"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def bootstrap_ci(diffs: list[float], n: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """95% confidence interval of the mean of paired differences, by resampling questions with replacement.

    Args:
        diffs: Per-question differences B - A.
        n: Number of bootstrap resamples.
        seed: Random seed, so the interval is reproducible.

    Returns:
        The 2.5th and 97.5th percentile of the resampled means.
    """
    rng = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def paired_stats(a: dict[str, dict], b: dict[str, dict], metric: Callable[[dict], float | None]) -> dict[str, Any]:
    """Compare one metric per question over the questions both runs have a value for.

    Args:
        a: Run A, qid -> record.
        b: Run B, qid -> record.
        metric: Per-question value of the metric.

    Returns:
        n, the means of A and B, the mean difference B - A with its 95% bootstrap interval, and how many questions
        are better, worse or the same in B. Empty ({"n": 0}) if no question has a value in both runs.
    """
    pairs = [(metric(a[q]), metric(b[q])) for q in sorted(set(a) & set(b))]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if not pairs:
        return {"n": 0}
    diffs = [y - x for x, y in pairs]
    low, high = bootstrap_ci(diffs)
    return {
        "n": len(pairs),
        "mean_a": float(np.mean([x for x, _ in pairs])),
        "mean_b": float(np.mean([y for _, y in pairs])),
        "diff": float(np.mean(diffs)),
        "ci_low": low,
        "ci_high": high,
        "b_higher": sum(d > 0 for d in diffs),
        "b_lower": sum(d < 0 for d in diffs),
        "same": sum(d == 0 for d in diffs),
    }


def compare_runs(name_a: str, name_b: str, split: str = "dev", evaluation: str = "answers") -> dict[str, Any]:
    """Compare two runs of the same evaluation per question and write the result.

    Args:
        name_a: Run A (the reference, e.g. the baseline).
        name_b: Run B.
        split: The split of both runs.
        evaluation: "answers" (rag eval answers) or "abstention" (rag eval abstention).

    Returns:
        Per metric the paired statistics; also written to results/compare_[abstention_]<a>_vs_<b>_<split>.json and
        logged as a table.
    """
    results_dir, kind, metrics = EVALUATIONS[evaluation]
    # Records of other kinds are skipped (runs made before the hand-written abstention questions were removed hold
    a = {q: r for q, r in load_run(name_a, split, results_dir).items() if r["kind"] == kind}
    b = {q: r for q, r in load_run(name_b, split, results_dir).items() if r["kind"] == kind}
    common = set(a) & set(b)
    if set(a) != set(b):
        logger.warning(f"The runs have different questions; comparing only the {len(common)} they share")

    result: dict[str, Any] = {"a": name_a, "b": name_b, "split": split, "evaluation": evaluation, "metrics": {}}
    for metric_name, metric in metrics.items():
        stats = paired_stats({q: a[q] for q in common}, {q: b[q] for q in common}, metric)
        if stats["n"]:
            result["metrics"][metric_name] = stats

    rows = []
    for metric_name, s in result["metrics"].items():
        significant = "yes" if s["ci_low"] > 0 or s["ci_high"] < 0 else "no"
        rows.append(
            [
                metric_name,
                str(s["n"]),
                f"{s['mean_a']:.3f}",
                f"{s['mean_b']:.3f}",
                f"{s['diff']:+.3f}",
                f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]",
                significant,
                f"{s['b_higher']}/{s['b_lower']}/{s['same']}",
            ]
        )
    table = format_table(["Metric", "n", "A", "B", "B - A", "95% CI", "Clear?", "B higher/lower/same"], rows)

    prefix = "compare_" if evaluation == "answers" else f"compare_{evaluation}_"
    out = RESULTS_DIR / f"{prefix}{name_a}_vs_{name_b}_{split}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info(f"[write] {Path(out).relative_to(PROJECT_ROOT)}")
    logger.info(f"A = {name_a}\nB = {name_b}\n{table}\n'Clear?' = yes when the 95% interval excludes 0.")
    return result
