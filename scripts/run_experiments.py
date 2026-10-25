"""Run the phase-1 and phase-2 evaluations one after another (docs/how_to_run.md, steps 4 and 5).

Every step is an ordinary `rag` command in its own process, so each run keeps its own log in logs/, resumes when the
script is started again after an interruption, and refuses changed settings, exactly as when run by hand.

Usage:
    poetry run python scripts/run_experiments.py all              # phase1 + phase2a in one go (~6.5 h)
    poetry run python scripts/run_experiments.py phase1           # all 7 setups, 20-question sample (~4 h)
    poetry run python scripts/run_experiments.py phase2a          # the baseline on all dev questions, with the judge
    poetry run python scripts/run_experiments.py phase2b --setup 1   # the second setup chosen after phase 1
    poetry run python scripts/run_experiments.py all --dry-run    # only print the commands

Phase 2a ends with a check of the judge's health. If it reports a problem, look at the judge
before running phase 2b, which judges the second setup with the same judge.
"""

import argparse
import json
import subprocess
import sys
import time

from rag_assistant.config import (
    ANSWERS_DIR,
    DEFAULT_MAX_CHARS,
    DEFAULT_MODEL,
    DEFAULT_TOKENIZER,
    DEFAULT_TOP_K,
    RESULTS_DIR,
)

# The setups of docs/how_to_run.md, step 4: label -> (description, options of `rag eval answers/abstention`).
SETUPS: dict[str, tuple[str, list[str]]] = {
    "B": ("baseline", []),
    "1": ("depth: fewer, fuller chunks", ["--top-k", "2", "--max-chars", "4000"]),
    "2": ("speed: less text", ["--top-k", "3", "--max-chars", "1500"]),
    "3": ("other tokenizer", ["--tokenizer", "identifier_split"]),
    "4": ("more context", ["--top-k", "6", "--max-chars", "2000"]),
    "5": ("breadth: many short chunks", ["--top-k", "8", "--max-chars", "1000"]),
    "6": ("less truncation, same chunks", ["--top-k", "4", "--max-chars", "3000"]),
}
PHASE1_LIMIT = 20

# Judge health gate after the first judged run (phase 2). Thresholds are rules of thumb, not proofs: a run that fails
# them is stopped for a human look, not declared wrong.
MAX_NUGGET_JUDGE_FAILURES = 0.10  # share of judged answers whose nugget judgment was incomplete or invalid
MIN_SUPPORT_VS_PRECISION = 0.5  # citation support should not be far below citation precision against FreshStack


def run_name(setup: str, limit: int | None = None, no_judge: bool = False) -> str:
    """Return the run name `rag eval answers/abstention` derives from a setup's options (see evaluate_answers)."""
    options = dict(zip(SETUPS[setup][1][::2], SETUPS[setup][1][1::2]))
    tokenizer = options.get("--tokenizer", DEFAULT_TOKENIZER)
    top_k = options.get("--top-k", DEFAULT_TOP_K)
    max_chars = options.get("--max-chars", DEFAULT_MAX_CHARS)
    name = f"bm25_{tokenizer}_{DEFAULT_MODEL.replace(':', '-')}_k{top_k}_c{max_chars}"
    if limit:
        name += f"_sample{limit}"
    if no_judge:
        name += "_nojudge"
    return name


def rag(args: list[str], dry_run: bool) -> None:
    """Run one `rag` command in its own process; stop the script if it fails."""
    print(f"\n=== rag {' '.join(args)}", flush=True)
    if dry_run:
        return
    start = time.perf_counter()
    result = subprocess.run([sys.executable, "-m", "rag_assistant.cli", *args])
    if result.returncode != 0:
        sys.exit(
            f"Stopped: `rag {' '.join(args)}` failed (exit code {result.returncode}). Fix it and start the "
            "script again; finished questions are skipped."
        )
    print(f"=== done in {(time.perf_counter() - start) / 60:.1f} min", flush=True)


def phase1(setups: list[str], dry_run: bool) -> None:
    """Answer run (no judge) and abstention run per setup on the 20-question sample, then compare with the baseline."""
    limit = ["--limit", str(PHASE1_LIMIT)]
    for setup in setups:
        print(f"\n##### Setup {setup}: {SETUPS[setup][0]}", flush=True)
        rag(["eval", "answers", *limit, "--no-judge", *SETUPS[setup][1]], dry_run)
        rag(["eval", "abstention", *limit, *SETUPS[setup][1]], dry_run)

    baseline_answers = run_name("B", PHASE1_LIMIT, no_judge=True)
    if not dry_run and not (ANSWERS_DIR / f"{baseline_answers}_dev.jsonl").exists():
        print("\nNo baseline run yet: run setup B to get the comparisons.")
        return
    for setup in [s for s in setups if s != "B"]:
        rag(["eval", "compare", "--a", baseline_answers, "--b", run_name(setup, PHASE1_LIMIT, no_judge=True)], dry_run)
        rag(
            [
                "eval",
                "compare",
                "--abstention",
                "--a",
                run_name("B", PHASE1_LIMIT),
                "--b",
                run_name(setup, PHASE1_LIMIT),
            ],
            dry_run,
        )


def judge_health(name: str) -> list[str]:
    """Check the first judged run for the judge defects found before (design decision 5.10); return the problems."""
    report = json.loads((RESULTS_DIR / f"answer_{name}_dev.json").read_text(encoding="utf-8"))["freshstack"]
    records = [
        json.loads(line) for line in (ANSWERS_DIR / f"{name}_dev.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    judged = [r for r in records if r["valid_output"] and not r["insufficient_information"] and r["judge_s"]]
    failures = sum(not r["nugget_judge_ok"] for r in judged) / len(judged) if judged else 0.0
    support, precision = report["citation_support"], report["citation_precision_qrels"]
    print(
        f"\nJudge health of {name}: nugget-judge failures {failures:.0%} of {len(judged)} judged answers; "
        f"citation support {support}, citation precision vs FreshStack {precision}"
    )
    problems = []
    if failures > MAX_NUGGET_JUDGE_FAILURES:
        problems.append(
            f"the nugget judge failed on {failures:.0%} of the judged answers (see the warnings in "
            f"logs/answers_{name}_dev.log)"
        )
    if support is not None and precision and support < MIN_SUPPORT_VS_PRECISION * precision:
        problems.append(
            f"citation support ({support:.3f}) is far below citation precision ({precision:.3f}): the "
            "judge rejects chunks FreshStack marks relevant"
        )
    return problems


def phase2a(dry_run: bool) -> None:
    """Run the baseline on all dev questions: abstention run, answer run with the judge, then the judge health check.

    The health check comes last, so a warning stops nothing: the runs are done, and the second setup (phase2b) is
    only judged once the judge has been found fine (docs/how_to_run.md, step 5b).
    """
    rag(["eval", "abstention", *SETUPS["B"][1]], dry_run)
    rag(["eval", "answers", *SETUPS["B"][1]], dry_run)
    if dry_run:
        return
    problems = judge_health(run_name("B"))
    if problems:
        sys.exit(
            "\nThe judge may be broken; do not run phase2b yet:\n- "
            + "\n- ".join(problems)
            + "\nLook at the run log and a few records (docs/how_to_run.md, step 5b). If the judge needs a fix, raise "
            "JUDGE_VERSION and rerun the baseline with `rag eval answers --restart`."
        )
    print(
        "Judge health looks fine. Still read a few answers and their nugget verdicts by eye "
        "(docs/how_to_run.md, step 5b)."
    )


def phase2b(setup: str, dry_run: bool) -> None:
    """Run the second chosen setup on all dev questions (answers with the judge, abstention); compare with the baseline."""
    rag(["eval", "abstention", *SETUPS[setup][1]], dry_run)
    rag(["eval", "answers", *SETUPS[setup][1]], dry_run)
    rag(["eval", "compare", "--a", run_name("B"), "--b", run_name(setup)], dry_run)
    rag(["eval", "compare", "--abstention", "--a", run_name("B"), "--b", run_name(setup)], dry_run)


def main() -> None:
    """Parse the arguments and run the chosen phase."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("all", "phase1", "phase2a", "phase2b"))
    parser.add_argument("--setups", nargs="+", choices=list(SETUPS), help="phase1: setup labels (default: all)")
    parser.add_argument("--setup", choices=[s for s in SETUPS if s != "B"], help="phase2b: the second chosen setup")
    parser.add_argument("--dry-run", action="store_true", help="only print the commands")
    args = parser.parse_args()

    if args.phase in ("all", "phase1"):
        setups = args.setups or list(SETUPS)
        phase1(sorted(setups, key=lambda s: s != "B"), args.dry_run)  # baseline first, so the comparisons can run
    if args.phase in ("all", "phase2a"):
        phase2a(args.dry_run)
    if args.phase == "phase2b":
        if not args.setup:
            parser.error("phase2b needs the second setup chosen after phase 1, e.g. --setup 1")
        phase2b(args.setup, args.dry_run)


if __name__ == "__main__":
    main()
