"""Shared project paths and default settings. Import from here instead of defining them in each module."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
CORPUS_PATH = DATA_DIR / "corpus.parquet"
QUERIES_PATH = DATA_DIR / "queries.json"
SPLITS_PATH = DATA_DIR / "splits.json"
STATS_PATH = DATA_DIR / "stats.json"

INDEX_DIR = PROJECT_ROOT / "indexes"

RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
ANSWERS_DIR = RESULTS_DIR / "answers"
ABSTENTION_DIR = RESULTS_DIR / "abstention"

LOG_PATH = PROJECT_ROOT / "logs.log"
# One log file per evaluation run (logger.add_run_log), kept next to the latest-command logs.log.
LOGS_DIR = PROJECT_ROOT / "logs"

# Defaults shared by the assistant, the CLI and the evaluations.
DEFAULT_TOKENIZER = "word_stem"
DEFAULT_QUERY_MODE = "full"
DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_JUDGE_MODEL = "llama3"
DEFAULT_TOP_K = 4
DEFAULT_MAX_CHARS = 2000
DEFAULT_MAX_TOKENS = 800
