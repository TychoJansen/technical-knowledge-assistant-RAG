"""Helpers for the saved reports and the Markdown summary tables the evaluations write."""

import json
from typing import Any

from rag_assistant.config import RESULTS_DIR


def load_reports(prefix: str) -> list[dict[str, Any]]:
    """Load all saved reports results/<prefix>_*.json, sorted by split and name.

    Args:
        prefix: The report kind: "retrieval", "answer" or "abstention".

    Returns:
        A list of report dicts, one per evaluated run.
    """
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in RESULTS_DIR.glob(f"{prefix}_*.json")]
    return sorted(reports, key=lambda r: (r["split"], r["name"]))


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table with padded columns, so it also reads well in a terminal.

    Args:
        headers: A list of column headers.
        rows: A list of rows, each a list of column values.

    Returns:
        A string containing the Markdown table.
    """
    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows)]
    lines = [headers, ["-" * w for w in widths], *rows]
    return "\n".join("| " + " | ".join(str(c).ljust(w) for c, w in zip(line, widths)) + " |" for line in lines)
