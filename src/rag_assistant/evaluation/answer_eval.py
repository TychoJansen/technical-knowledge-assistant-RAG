"""End-to-end evaluation of the assistant (`rag eval answers`): retrieval quality, answer quality, citation support.

For every question of a split the assistant answers (retrieve -> LLM -> checks), then a judge LLM grades the answer:
- Retrieval: Coverage@k and Recall@k of the k chunks actually given to the model (the ceiling for the answer).
- Answer quality: nugget recall (which key facts of the reference answer are in the answer), how often the assistant
  answers, wrongly abstains or produces invalid output. (Whether it says so when the information is insufficient is
  measured separately, by `rag eval abstention`.)
- Citation support: per cited chunk, whether the judge finds it supports the answer; citation precision against the
  FreshStack judgments; dropped citations and answers without citations.

The assistant only sees the question. Nuggets and judgments are used by the judge and the metrics only.

Usage:
    poetry run rag eval answers --limit 10              # quick run on a seeded sample of 10 dev questions
    poetry run rag eval answers                         # all dev questions (~1.5-2 h on a laptop CPU)
    poetry run rag eval answers --split test            # final numbers only

Results are written per question as they come in, so an interrupted run resumes where it stopped (use --restart to
start over). Outputs in results/: answers/<name>_<split>.jsonl, answer_<name>_<split>.json and answer_summary.md.
"""

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from rag_assistant.assistant import ANSWER_RULES_VERSION, Assistant, load_assistant
from rag_assistant.config import (
    ANSWERS_DIR,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MAX_CHARS,
    DEFAULT_MODEL,
    DEFAULT_QUERY_MODE,
    DEFAULT_TOKENIZER,
    DEFAULT_TOP_K,
    LOGS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
)
from rag_assistant.data import language, load_split_queries
from rag_assistant.evaluation.judge import (
    JUDGE_VERSION,
    CitationJudgment,
    is_degenerate_answer,
    judge_citation,
    judge_nuggets,
)
from rag_assistant.evaluation.metrics import coverage_at_k, judged_doc_ids, recall_at_k
from rag_assistant.evaluation.report import format_table, load_reports
from rag_assistant.llm import LLM, OllamaLLM
from rag_assistant.logger import add_run_log, get_logger
from rag_assistant.prompt import prompt_fingerprint
from rag_assistant.query import retrieval_query

logger = get_logger(__name__)

# Seed for the random sample of questions with --limit, so every setup gets the same sample.
SAMPLE_SEED = 42

# Column label -> (section, key) in the report, for the summary table.
SUMMARY_COLUMNS = {
    "Ctx Cov": ("freshstack", "context_coverage"),
    "Ctx R": ("freshstack", "context_recall"),
    "Nugget R": ("freshstack", "nugget_recall"),
    "Nugget R|ans": ("freshstack", "nugget_recall_answered"),
    "Answered": ("freshstack", "answered_rate"),
    "False abst": ("freshstack", "false_abstention_rate"),
    "Invalid": ("freshstack", "invalid_rate"),
    "Cit support": ("freshstack", "citation_support"),
    "Cit prec": ("freshstack", "citation_precision_qrels"),
    "Uncited ans": ("freshstack", "uncited_answer_rate"),
}


def evaluate_freshstack(
    question: dict[str, Any], assistant: Assistant, judge: LLM | None, query_mode: str = DEFAULT_QUERY_MODE
) -> dict[str, Any]:
    """Answer one FreshStack question and grade the answer and its citations.

    Args:
        question: The query record (evaluation data; only question["question"] goes to the assistant).
        assistant: The assistant under evaluation.
        judge: The judge LLM, or None to skip judging (--no-judge): the judge fields are then None and only the
            deterministic metrics can be computed.
        query_mode: Which part of the question to search with (see rag_assistant.query); the model always answers
            the full question.

    Returns:
        A per-question record with the answer, the context metrics and the judgments.
    """
    answer = assistant.ask(question["question"], retrieval_query(question, query_mode))
    retrieved_ids = [hit.doc_id for hit in answer.retrieved]
    nuggets = question["nuggets"]
    relevant = set(question["relevant_doc_ids"])
    judged = judged_doc_ids(question)

    answered = answer.valid_output and not answer.insufficient_information
    citation_judgments: dict[str, CitationJudgment] = {}
    if judge is None:
        # --no-judge: no verdicts at all (None, not False), so the judge metrics stay empty instead of 0.
        nuggets_present, nugget_judge_ok, judge_s = None, None, None
    elif answered and is_degenerate_answer(answer.text, question["title"]):
        # Too short or only the title: no facts to credit and no claims to support, so no judge call (floor).
        nuggets_present, nugget_judge_ok, judge_s = [False] * len(nuggets), True, 0.0
        citation_judgments = {d: CitationJudgment(d, False, False, "", False, True, 0.0) for d in answer.cited_doc_ids}
    elif answered:
        nugget_judgment = judge_nuggets(judge, question["question"], answer.text, [n["text"] for n in nuggets])
        nuggets_present = list(nugget_judgment.present)
        nugget_judge_ok = nugget_judgment.parse_ok
        citation_judgments = {
            d: judge_citation(judge, question["question"], answer.text, d, assistant.corpus[d], assistant.max_chars)
            for d in answer.cited_doc_ids
        }
        judge_s = nugget_judgment.latency_s + sum(c.latency_s for c in citation_judgments.values())
    else:
        # An abstained or invalid answer contains no nuggets and has no citations to judge.
        nuggets_present, nugget_judge_ok, judge_s = [False] * len(nuggets), True, 0.0

    return {
        "qid": question["qid"],
        "kind": "freshstack",
        "language": language(question),
        "title": question["title"],
        "answer": answer.text,
        "valid_output": answer.valid_output,
        "insufficient_information": answer.insufficient_information,
        "retrieved": retrieved_ids,
        "cited_doc_ids": list(answer.cited_doc_ids),
        "invalid_citations": list(answer.invalid_citations),
        "repaired_citations": [list(pair) for pair in answer.repaired_citations],
        "degenerate_answer": answered and is_degenerate_answer(answer.text, question["title"]),
        "context_coverage": coverage_at_k(
            retrieved_ids, [n["relevant_corpus_ids"] for n in nuggets], len(retrieved_ids)
        ),
        "context_recall": recall_at_k(retrieved_ids, relevant, len(retrieved_ids)),
        "nuggets": [n["text"] for n in nuggets],
        "nuggets_present": nuggets_present,
        "nugget_judge_ok": nugget_judge_ok,
        "citations": [citation_record(d, citation_judgments.get(d), relevant, judged) for d in answer.cited_doc_ids],
        "generation_s": answer.llm_response.latency_s,
        "judge_s": judge_s,
        "prompt_tokens": answer.llm_response.input_tokens,
        "output_tokens": answer.llm_response.output_tokens,
    }


def citation_record(
    doc_id: str, judgment: CitationJudgment | None, relevant: set[str], judged: set[str]
) -> dict[str, Any]:
    """One citation of an answer: the deterministic checks always, the judge's verdict only if it was judged.

    Args:
        doc_id: The cited chunk.
        judgment: The judge's verdict, or None if the citation was not judged (--no-judge).
        relevant: The question's relevant doc_ids (FreshStack).
        judged: The question's judged doc_ids, relevant or not.

    Returns:
        The citation record; the judge fields are None when not judged.
    """
    return {
        "doc_id": doc_id,
        "supported": judgment.supported if judgment else None,
        "judge_supported": judgment.judge_supported if judgment else None,
        "quote": judgment.quote if judgment else None,
        "quote_found": judgment.quote_found if judgment else None,
        "judge_ok": judgment.parse_ok if judgment else None,
        "relevant_in_qrels": doc_id in relevant,
        "judged_in_qrels": doc_id in judged,
    }


def select_questions(questions: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Return all questions, or a seeded random sample of `limit` of them (in split order).

    A random sample instead of the first N: the split is sorted by question id, so the first N are the oldest
    questions. The fixed seed gives every setup exactly the same sample.

    Args:
        questions: The questions of the split, in split order.
        limit: The sample size, or None for all questions.

    Returns:
        The selected questions, in split order.
    """
    if not limit or limit >= len(questions):
        return questions
    chosen = set(random.Random(SAMPLE_SEED).sample(range(len(questions)), limit))
    return [q for i, q in enumerate(questions) if i in chosen]


def warm_up(llm: LLM) -> None:
    """Start a run from a clean, loaded model.

    First unload the model, which clears Ollama's prompt cache: prompts cached by an earlier run would otherwise be
    answered faster and possibly differently. Then load it with one short call, so the first
    timed question does not include loading the model.

    Args:
        llm: The model to unload and load again.
    """
    logger.info(f"Unloading and warming up {llm.settings().get('model', 'the LLM')}")
    llm.unload()
    llm.generate("Reply with OK.", max_tokens=5)


def mean(values: list[float]) -> float | None:
    """Mean of a list, None when empty (written as null in JSON, '-' in the summary table).

    Args:
        values: The values to average.

    Returns:
        The mean, or None for an empty list.
    """
    return float(np.mean(values)) if values else None


def aggregate_freshstack(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Aggregate the FreshStack per-question records into answer, citation and context metrics.

    Rates are over all questions unless the name says otherwise. Citation metrics are micro-averages over all valid
    citations of answered questions.

    Args:
        records: The per-question records of the run.

    Returns:
        The run's metrics by name; judge metrics are None when nothing was judged.
    """
    answered = [r for r in records if r["valid_output"] and not r["insufficient_information"]]
    citations = [c for r in answered for c in r["citations"]]
    # Judge metrics only over what was judged; a --no-judge run has nothing judged, so they come out as None.
    judged_records = [r for r in records if r["nuggets_present"] is not None]
    judged_answered = [r for r in answered if r["nuggets_present"] is not None]
    judged_citations = [c for c in citations if c["supported"] is not None]
    recall_per_q = [sum(r["nuggets_present"]) / len(r["nuggets_present"]) for r in judged_records]
    recall_answered = [sum(r["nuggets_present"]) / len(r["nuggets_present"]) for r in judged_answered]
    return {
        "n": len(records),
        "context_coverage": mean([r["context_coverage"] for r in records]),
        "context_recall": mean([r["context_recall"] for r in records]),
        "nugget_recall": mean(recall_per_q),
        "nugget_recall_answered": mean(recall_answered),
        "answered_rate": len(answered) / len(records),
        "false_abstention_rate": mean([float(r["valid_output"] and r["insufficient_information"]) for r in records]),
        "invalid_rate": mean([float(not r["valid_output"]) for r in records]),
        "n_citations": len(citations),
        "citation_support": mean([float(c["supported"]) for c in judged_citations]),
        "citation_judge_yes_rate": mean([float(c["judge_supported"]) for c in judged_citations]),
        "citation_yes_without_quote_rate": mean(
            [float(c["judge_supported"] and not c["quote_found"]) for c in judged_citations]
        ),
        "citation_precision_qrels": mean([float(c["relevant_in_qrels"]) for c in citations]),
        "citation_unjudged_rate": mean([float(not c["judged_in_qrels"]) for c in citations]),
        "uncited_answer_rate": mean([float(not r["cited_doc_ids"]) for r in answered]),
        "answers_with_dropped_citations_rate": mean([float(bool(r["invalid_citations"])) for r in records]),
        "answers_with_repaired_citations_rate": mean([float(bool(r.get("repaired_citations"))) for r in records]),
        "judge_parse_failure_rate": mean(
            [float(not r["nugget_judge_ok"]) for r in judged_answered]
            + [float(not c["judge_ok"]) for c in judged_citations]
        ),
        "generation_s_mean": mean([r["generation_s"] for r in records]),
        "judge_s_mean": mean([r["judge_s"] for r in judged_answered]),
    }


def load_records(path: Path, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load already evaluated questions for resuming; refuse to mix results of a different configuration.

    A half-written last line (a run stopped while writing) is dropped, so that question is evaluated again.

    Args:
        path: The run's JSONL results file.
        config: The configuration of the run being started.

    Returns:
        qid -> record of the questions already done; empty if the file does not exist.

    Raises:
        SystemExit: If a line other than the last is invalid, or the file was made with another configuration.
    """
    if not path.exists():
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i < len(lines) - 1:
                raise SystemExit(f"{path} has an invalid line {i + 1}; fix or remove it, or use --restart")
            # A run stopped while writing leaves a half-written last line: drop it, that question is redone.
            logger.warning(f"Ignoring a half-written last line in {path}; that question will be evaluated again")
            path.write_text("".join(f"{kept}\n" for kept in lines[:-1]), encoding="utf-8")
    if records and records[0].get("config") != config:
        raise SystemExit(f"{path} was made with a different configuration; use --restart or another --name")
    return {r["qid"]: r for r in records}


def run(
    questions: list[dict[str, Any]],
    path: Path,
    config: dict[str, Any],
    evaluate: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Evaluate all questions not yet in the results file, appending each record as soon as it is done.

    Args:
        questions: The questions to evaluate, in order.
        path: The JSONL results file (one record per question); existing records are kept and skipped.
        config: The run configuration; a file made with another configuration is refused (see load_records).
        evaluate: Turns one question into its record (e.g. evaluate_freshstack with the assistant and judge bound).

    Returns:
        The records of all questions, in the order of `questions`.
    """
    done = load_records(path, config)
    todo = [q for q in questions if q["qid"] not in done]
    logger.info(f"{len(done)} questions already done, {len(todo)} to go")
    with path.open("a", encoding="utf-8") as f:
        for i, q in enumerate(todo, start=1):
            record = evaluate(q)
            record["config"] = config
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            done[q["qid"]] = record
            logger.info(
                f"[{i}/{len(todo)}] {q['qid']} ({record['kind']}): insufficient={record['insufficient_information']} "
                f"cited={len(record['cited_doc_ids'])} gen={record['generation_s']:.0f}s"
                + (f" judge={record['judge_s']:.0f}s" if record.get("judge_s") is not None else "")
            )
    return [done[q["qid"]] for q in questions]


def summary_table(reports: list[dict[str, Any]]) -> str:
    """Render one row per run with the main metrics.

    Args:
        reports: The saved reports of all answer runs.

    Returns:
        The summary table as Markdown.
    """
    rows = []
    for r in reports:
        # The prompt fingerprint, so runs made with different prompts are never read as comparable.
        cells = [r["name"], r["split"], r["config"].get("prompt", "-"), str(r["freshstack"]["n"])]
        for section, key in SUMMARY_COLUMNS.values():
            value = r.get(section, {}).get(key)
            cells.append("-" if value is None else f"{value:.3f}")
        rows.append(cells)
    return format_table(["Run", "Split", "Prompt", "n", *SUMMARY_COLUMNS], rows)


def write_summary(table: str) -> None:
    """Rewrite answer_summary.md from all saved answer-evaluation runs.

    Args:
        table: The summary table (see summary_table).
    """
    path = RESULTS_DIR / "answer_summary.md"
    path.write_text(
        "# Answer evaluation results\n\n"
        "Generated by `rag eval answers`; do not edit by hand.\n\n"
        "- **Prompt**: fingerprint of the assistant prompt; compare only runs with the same one.\n"
        "- **Ctx Cov / Ctx R**: Coverage@k and Recall@k of the k chunks given to the model (retrieval ceiling).\n"
        "- **Nugget R**: share of each question's nuggets the judge finds in the answer (0 when abstained).\n"
        "- **Nugget R|ans**: the same, over answered questions only.\n"
        "- **Answered / False abst / Invalid**: share of (answerable) FreshStack questions answered, wrongly "
        "abstained, or with unparseable output.\n"
        "- **Cit support**: share of valid citations the judge finds supporting the answer, with a quote that is "
        "verified to occur in the source.\n"
        "- **Cit prec**: share of citations that are relevant according to the FreshStack judgments (lower bound).\n"
        "- **Uncited ans**: share of answers without any valid citation.\n\n"
        "Whether the assistant says so when the information is insufficient: see `abstention_summary.md` "
        "(`rag eval abstention`).\n\n"
        f"{table}\n",
        encoding="utf-8",
    )
    logger.info(f"[write] {path.relative_to(PROJECT_ROOT)}")


def evaluate_answers(
    split: str = "dev",
    limit: int | None = None,
    tokenizer: str = DEFAULT_TOKENIZER,
    model: str = DEFAULT_MODEL,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    top_k: int = DEFAULT_TOP_K,
    max_chars: int = DEFAULT_MAX_CHARS,
    name: str | None = None,
    restart: bool = False,
    no_judge: bool = False,
    query_mode: str = DEFAULT_QUERY_MODE,
) -> None:
    """Run the evaluation and write all outputs.

    Args:
        split: The data split ("dev" or "test"; test only for final numbers).
        limit: A seeded random sample of N questions of the split (for quick runs; the same sample every time).
        tokenizer: The tokenizer of the BM25 index.
        model: Ollama model of the assistant.
        judge_model: Ollama model of the judge.
        top_k: Number of retrieved chunks given to the assistant.
        max_chars: Maximum number of characters per chunk.
        name: Run name (default: derived from the settings).
        restart: Discard earlier results of this run.
        no_judge: Skip the judge: only the deterministic metrics (context, answered/abstained/invalid, citation
            precision against FreshStack, speed). Much faster; the judge metrics are None.
        query_mode: Which part of the question to search with: "full", "title" or "title_first_paragraph". The model
            always answers the full question.
    """
    if split == "test":
        logger.warning("Evaluating on the TEST split. Use it only for final numbers, never for tuning.")

    if not name:
        name = f"bm25_{tokenizer}_{model.replace(':', '-')}_k{top_k}_c{max_chars}"
        if query_mode != DEFAULT_QUERY_MODE:
            name += f"_q-{query_mode}"
        if limit:
            name += f"_sample{limit}"
        if no_judge:
            name += "_nojudge"

    questions = select_questions(load_split_queries(split), limit)

    ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
    path = ANSWERS_DIR / f"{name}_{split}.jsonl"
    if restart:
        path.unlink(missing_ok=True)
    add_run_log(LOGS_DIR / f"answers_{name}_{split}.log", restart)

    assistant = load_assistant(tokenizer=tokenizer, model=model, top_k=top_k, max_chars=max_chars)
    judge = None if no_judge else OllamaLLM(model=judge_model)
    # Everything that changes the answers or the judgments, so a resumed run never mixes two systems.
    config = {
        "tokenizer": tokenizer,
        "query_mode": query_mode,
        "top_k": top_k,
        "max_chars": max_chars,
        "max_tokens": assistant.max_tokens,
        "prompt": prompt_fingerprint(),
        "answer_rules": ANSWER_RULES_VERSION,
        "assistant_llm": assistant.llm.settings(),
        "judge_llm": judge.settings() if judge else None,
        "judge_version": JUDGE_VERSION if judge else None,
        "sample": {"limit": limit, "seed": SAMPLE_SEED} if limit else None,
    }
    warm_up(assistant.llm)
    if judge:
        warm_up(judge)
    records = run(questions, path, config, lambda q: evaluate_freshstack(q, assistant, judge, query_mode))

    report = {"name": name, "split": split, "config": config, "freshstack": aggregate_freshstack(records)}
    report_path = RESULTS_DIR / f"answer_{name}_{split}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info(f"[write] {path.relative_to(PROJECT_ROOT)}")
    logger.info(f"[write] {report_path.relative_to(PROJECT_ROOT)}")

    table = summary_table(load_reports("answer"))
    write_summary(table)
    logger.info("Answer evaluation results (all runs):\n" + table)
    logger.info("This run:\n" + json.dumps(report, indent=2))
