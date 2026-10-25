"""Unit test for starting a fresh log file per run, on a temporary log file and a separate logger."""

import logging

from rag_assistant import logger as log_module


def test_start_new_log_file_keeps_only_the_new_run(tmp_path, monkeypatch):
    log_path = tmp_path / "logs.log"
    monkeypatch.setattr(log_module, "LOG_PATH", log_path)
    monkeypatch.setattr(log_module, "_ROOT_LOGGER_NAME", "rag_test_logger")  # do not touch the real logs.log
    try:
        log = log_module.get_logger("test")
        log.info("previous run")
        assert "previous run" in log_path.read_text(encoding="utf-8")

        log_module.start_new_log_file()
        log.info("new run")

        text = log_path.read_text(encoding="utf-8")
        assert "new run" in text
        assert "previous run" not in text
    finally:
        root = logging.getLogger("rag_test_logger")
        for handler in list(root.handlers):
            handler.close()
            root.removeHandler(handler)


def test_run_log_starts_with_the_command_so_far_and_keeps_receiving_messages(tmp_path, monkeypatch):
    log_path, run_log = tmp_path / "logs.log", tmp_path / "logs" / "answers_x_dev.log"
    monkeypatch.setattr(log_module, "LOG_PATH", log_path)
    monkeypatch.setattr(log_module, "_ROOT_LOGGER_NAME", "rag_test_run_logger")
    try:
        log = log_module.get_logger("test")
        log.info("run started with args")
        log_module.add_run_log(run_log)
        log.info("question 1 done")
        text = run_log.read_text(encoding="utf-8")
        assert "run started with args" in text and "question 1 done" in text

        log_module.add_run_log(run_log, restart=True)  # --restart: the earlier run log is removed
        assert "question 1 done" not in run_log.read_text(encoding="utf-8").split("Run log:")[-1]
    finally:
        root = logging.getLogger("rag_test_run_logger")
        for handler in list(root.handlers):
            handler.close()
            root.removeHandler(handler)
