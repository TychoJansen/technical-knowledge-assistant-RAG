"""Command-line interface: `poetry run rag <command>`.

Commands:
    rag prepare [--force] [--resplit]           download the data once and create the dev/test split
    rag index [--tokenizer NAME] [--force]      build the BM25 index for a tokenizer
    rag ask ["question"] [--show-sources]       answer one question, or interactively without a question
    rag eval retrieval [--split test]           evaluate retrieval (dev by default; test only for final numbers)
    rag eval answers [--limit 10] [--no-judge]  evaluate answers end-to-end (with a judge LLM unless --no-judge)
    rag eval abstention [--limit 20]            does it say "insufficient" when the context cannot answer (no judge)
    rag eval compare --a RUN --b RUN            compare two runs per question, with confidence intervals

Run `rag <command> --help` for all options. `ask` and `eval answers/abstention` need a running Ollama with the model
pulled.
"""

import argparse
import sys
import time
from typing import TYPE_CHECKING

from rag_assistant.config import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MAX_CHARS,
    DEFAULT_MODEL,
    DEFAULT_QUERY_MODE,
    DEFAULT_TOKENIZER,
    DEFAULT_TOP_K,
)
from rag_assistant.logger import get_logger, start_new_log_file
from rag_assistant.query import QUERY_MODES
from rag_assistant.tokenizers import TOKENIZERS

if TYPE_CHECKING:
    from rag_assistant.assistant import Answer

# Commands import their modules lazily, so e.g. `rag ask` does not load the Hugging Face datasets library.

logger = get_logger(__name__)

EXIT_WORDS = {"", "exit", "quit", "q"}


def cmd_prepare(args: argparse.Namespace) -> None:
    """Download the data and create the split (see rag_assistant.prepare).

    Args:
        args: The parsed command-line arguments.
    """
    from rag_assistant.prepare import prepare

    prepare(force=args.force, resplit=args.resplit)


def cmd_index(args: argparse.Namespace) -> None:
    """Build the BM25 index for a tokenizer, unless it already exists.

    Args:
        args: The parsed command-line arguments.
    """
    from rag_assistant.retrieval import build_index, index_dir, index_exists

    name = index_dir(args.tokenizer).name
    if index_exists(args.tokenizer) and not args.force:
        logger.info(f"[skip] {name} index already exists (use --force to rebuild)")
        return
    start = time.perf_counter()
    build_index(args.tokenizer)
    logger.info(f"[write] {name} index built in {time.perf_counter() - start:.0f}s")


def print_answer(answer: "Answer", urls: dict[str, str], show_sources: bool) -> None:
    """Print an answer with its citations in a readable form.

    Args:
        answer: The assistant's answer.
        urls: Mapping from doc_id to the GitHub URL of its file.
        show_sources: If True, also list every retrieved chunk, not only the cited ones.
    """
    print()
    if not answer.valid_output:
        print("[!] The model did not produce a valid answer.")
    elif answer.insufficient_information:
        print("[!] The knowledge source does not contain enough information to answer this.")
    print(answer.text)

    if answer.cited_doc_ids:
        print("\nSources:")
        for doc_id in answer.cited_doc_ids:
            print(f"  - {doc_id}\n    {urls.get(doc_id, '')}")
    elif answer.valid_output and not answer.insufficient_information:
        print("\n[!] The answer cites no sources; treat it with caution.")
    if answer.invalid_citations:
        print(f"\n[!] Dropped citations to chunks that were not retrieved: {', '.join(answer.invalid_citations)}")
    if show_sources:
        print("\nRetrieved chunks:")
        for hit in answer.retrieved:
            marker = "*" if hit.doc_id in answer.cited_doc_ids else " "
            print(f"  {marker} {hit.rank}. {hit.doc_id} (score {hit.score:.1f})")

    r = answer.llm_response
    print(f"\n({r.model}, {r.latency_s:.0f}s, {r.input_tokens} prompt + {r.output_tokens} output tokens)\n")


def cmd_ask(args: argparse.Namespace) -> None:
    """Answer one question, or questions typed interactively until an empty line or 'exit'.

    Args:
        args: The parsed command-line arguments.
    """
    from rag_assistant.assistant import load_assistant
    from rag_assistant.data import load_corpus_map

    # The Windows console may not be UTF-8; avoid crashes on characters in answers.
    sys.stdout.reconfigure(errors="replace")

    assistant = load_assistant(tokenizer=args.tokenizer, model=args.model, top_k=args.top_k, max_chars=args.max_chars)
    urls = load_corpus_map("url")

    if args.question:
        print_answer(assistant.ask(args.question), urls, args.show_sources)
        return

    print(f"Ask a question about LangChain ({args.model}, top-{args.top_k}). Empty line or 'exit' to stop.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in EXIT_WORDS:
            break
        print("Thinking...")
        print_answer(assistant.ask(question), urls, args.show_sources)


def cmd_eval_retrieval(args: argparse.Namespace) -> None:
    """Evaluate retrieval (see rag_assistant.evaluation.retrieval_eval).

    Args:
        args: The parsed command-line arguments.
    """
    from rag_assistant.evaluation.retrieval_eval import evaluate_retrieval

    evaluate_retrieval(tokenizer=args.tokenizer, split=args.split, name=args.name, query_mode=args.query_mode)


def cmd_eval_answers(args: argparse.Namespace) -> None:
    """Evaluate answers end-to-end (see rag_assistant.evaluation.answer_eval).

    Args:
        args: The parsed command-line arguments.
    """
    from rag_assistant.evaluation.answer_eval import evaluate_answers

    evaluate_answers(
        split=args.split,
        limit=args.limit,
        tokenizer=args.tokenizer,
        model=args.model,
        judge_model=args.judge_model,
        top_k=args.top_k,
        max_chars=args.max_chars,
        name=args.name,
        restart=args.restart,
        no_judge=args.no_judge,
        query_mode=args.query_mode,
    )


def cmd_eval_abstention(args: argparse.Namespace) -> None:
    """Evaluate abstention on insufficient context (see rag_assistant.evaluation.abstention_eval).

    Args:
        args: The parsed command-line arguments.
    """
    from rag_assistant.evaluation.abstention_eval import evaluate_abstention

    evaluate_abstention(
        split=args.split,
        limit=args.limit,
        tokenizer=args.tokenizer,
        model=args.model,
        top_k=args.top_k,
        max_chars=args.max_chars,
        name=args.name,
        restart=args.restart,
    )


def cmd_eval_compare(args: argparse.Namespace) -> None:
    """Compare two runs of the same evaluation per question (see rag_assistant.evaluation.compare).

    Args:
        args: The parsed command-line arguments.
    """
    from rag_assistant.evaluation.compare import compare_runs

    compare_runs(args.a, args.b, split=args.split, evaluation="abstention" if args.abstention else "answers")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with one subcommand per step.

    Returns:
        The parser; each subcommand sets `func` to its cmd_* function.
    """
    parser = argparse.ArgumentParser(
        prog="rag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add_tokenizer(p: argparse.ArgumentParser) -> None:
        p.add_argument("--tokenizer", choices=list(TOKENIZERS), default=DEFAULT_TOKENIZER)

    def add_query_mode(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--query-mode",
            choices=QUERY_MODES,
            default=DEFAULT_QUERY_MODE,
            help="which part of the question to search with (the model always answers the full question)",
        )

    def add_assistant_options(p: argparse.ArgumentParser) -> None:
        add_tokenizer(p)
        p.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model of the assistant")
        p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="number of chunks given to the model")
        p.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="max characters per chunk")

    p = commands.add_parser("prepare", help="download the data and create the dev/test split")
    p.add_argument("--force", action="store_true", help="re-download corpus and queries (does not touch splits)")
    p.add_argument("--resplit", action="store_true", help="recreate data/splits.json")
    p.set_defaults(func=cmd_prepare)

    p = commands.add_parser("index", help="build the BM25 index for a tokenizer")
    add_tokenizer(p)
    p.add_argument("--force", action="store_true", help="rebuild even if the index already exists")
    p.set_defaults(func=cmd_index)

    p = commands.add_parser("ask", help="ask the assistant questions")
    p.add_argument("question", nargs="?", help="answer this question and exit (default: interactive)")
    add_assistant_options(p)
    p.add_argument("--show-sources", action="store_true", help="list all retrieved chunks")
    p.set_defaults(func=cmd_ask)

    evals = commands.add_parser("eval", help="evaluate retrieval, answers or abstention").add_subparsers(
        dest="eval", required=True
    )

    p = evals.add_parser("retrieval", help="evaluate retrieval on a split")
    add_tokenizer(p)
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--name", help="run name (default: bm25_<tokenizer>)")
    add_query_mode(p)
    p.set_defaults(func=cmd_eval_retrieval)

    p = evals.add_parser("answers", help="evaluate answers end-to-end with a judge LLM")
    add_assistant_options(p)
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--limit", type=int, help="a seeded random sample of N questions (the same every run)")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Ollama model of the judge")
    add_query_mode(p)
    p.add_argument("--name", help="run name (default: derived from the settings)")
    p.add_argument("--restart", action="store_true", help="discard earlier results of this run")
    p.add_argument(
        "--no-judge", action="store_true", help="skip the judge: deterministic metrics only (2.5-3.5x faster)"
    )
    p.set_defaults(func=cmd_eval_answers)

    p = evals.add_parser(
        "abstention",
        help="does the assistant say 'insufficient' when it only gets chunks known not to answer the question",
    )
    add_assistant_options(p)
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--limit", type=int, help="a seeded random sample of N questions (the same as `eval answers`)")
    p.add_argument("--name", help="run name (default: derived from the settings)")
    p.add_argument("--restart", action="store_true", help="discard earlier results of this run")
    p.set_defaults(func=cmd_eval_abstention)

    p = evals.add_parser("compare", help="compare two runs per question, with confidence intervals")
    p.add_argument("--a", required=True, help="run A, the reference (name as in results/answers/<name>_<split>.jsonl)")
    p.add_argument("--b", required=True, help="run B")
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--abstention", action="store_true", help="compare `eval abstention` runs instead of answer runs")
    p.set_defaults(func=cmd_eval_compare)

    return parser


def main() -> None:
    """Parse the command line and run the chosen command."""
    args = build_parser().parse_args()
    # A fresh logs.log per command, after parsing so `rag --help` or a typo does not clear the previous run's log.
    start_new_log_file()
    logger.info(f"Run started with args: { {k: v for k, v in vars(args).items() if k != 'func'} }")
    args.func(args)


if __name__ == "__main__":
    main()
