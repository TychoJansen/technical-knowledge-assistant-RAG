"""The text used to search the corpus: the full question, or a shorter part of it (`--query-mode`)."""

from typing import Any

QUERY_MODES = ("full", "title", "title_first_paragraph")


def first_paragraph(text: str) -> str:
    """Return the text up to the first blank line, stripped.

    Args:
        text: The text, e.g. a question body.

    Returns:
        The first paragraph (the whole text if it has no blank line).
    """
    return text.strip().split("\n\n", 1)[0].strip()


def retrieval_query(question: dict[str, Any], mode: str) -> str:
    """Build the search query for a question record.

    Args:
        question: A query record with "question" (title and body), "title" and "body". Only the question text is
            used, never evaluation data.
        mode: "full" (title and body), "title" (the title only) or "title_first_paragraph" (the title and the first
            paragraph of the body).

    Returns:
        The text to search with.

    Raises:
        ValueError: If the mode is unknown.
    """
    if mode == "full":
        return question["question"]
    if mode == "title":
        return question["title"]
    if mode == "title_first_paragraph":
        return f"{question['title']}\n\n{first_paragraph(question['body'])}"
    raise ValueError(f"Unknown query mode {mode!r}, choose from {QUERY_MODES}")
