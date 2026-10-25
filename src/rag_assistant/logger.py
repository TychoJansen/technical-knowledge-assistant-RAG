"""Shared logger: logs to the console and to logs.log in the project root.

Every `rag` command starts a fresh logs.log, so that file holds the latest command only. An
evaluation run also keeps its own log in logs/.
"""

import logging
import sys
from pathlib import Path

from rag_assistant.config import LOG_PATH

_ROOT_LOGGER_NAME = "rag"
_FILE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONSOLE_FORMAT = "%(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _file_handler(path: Path) -> logging.FileHandler:
    """Create a DEBUG-level file handler that appends to a log file.

    Args:
        path: The log file.

    Returns:
        The file handler, with the file log format.
    """
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def _configure_root() -> logging.Logger:
    """Attach console and file handlers to the project root logger once.

    Returns:
        The project root logger.
    """
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    if root.handlers:
        return root

    root.setLevel(logging.DEBUG)
    root.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
    root.addHandler(console)

    root.addHandler(_file_handler(LOG_PATH))

    return root


def start_new_log_file() -> None:
    """Empty logs.log so it only holds the run that starts now.

    Called once at the start of every `rag` command. The file handler is closed first; it reopens the file (in append
    mode) at the next log message, so logging continues into the emptied file.
    """
    root = _configure_root()
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()
    LOG_PATH.write_text("", encoding="utf-8")


def add_run_log(path: Path, restart: bool = False) -> None:
    """Also log everything of this run to its own file, e.g. logs/answers_<run name>_<split>.log.

    The file starts with what this command logged so far (copied from logs.log: the arguments and the start), and
    then receives every further message. A resumed run appends to the same file, so it holds the whole run; with
    `restart` the earlier log is removed together with the earlier results.

    Args:
        path: The run's log file.
        restart: Remove an existing log of this run first.
    """
    root = _configure_root()
    path.parent.mkdir(parents=True, exist_ok=True)
    for handler in list(root.handlers):
        # Never two handlers on one file (and Windows cannot remove a file that is still open).
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == path.resolve():
            handler.close()
            root.removeHandler(handler)
    if restart:
        path.unlink(missing_ok=True)
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.flush()
    so_far = LOG_PATH.read_text(encoding="utf-8") if LOG_PATH.exists() else ""
    with path.open("a", encoding="utf-8") as f:
        f.write(so_far)
    root.addHandler(_file_handler(path))
    root.info(f"Run log: {path}")


def get_logger(name: str) -> logging.Logger:
    """Return a project logger that writes to the console and to logs.log.

    Args:
        name: Name of the module requesting the logger, usually ``__name__``.

    Returns:
        A child logger of the project root logger.
    """
    _configure_root()
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
