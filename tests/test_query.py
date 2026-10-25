"""Unit tests for the retrieval query modes (`--query-mode`)."""

import pytest

from rag_assistant.query import QUERY_MODES, first_paragraph, retrieval_query

QUESTION = {
    "title": "JSONLoader fails",
    "body": "I load a JSON file with JSONLoader.\nIt fails.\n\nTraceback (most recent call last):\n  ...",
}
QUESTION["question"] = f"{QUESTION['title']}\n\n{QUESTION['body']}"


def test_first_paragraph_stops_at_the_first_blank_line():
    assert first_paragraph("  one\ntwo\n\nthree\n\nfour") == "one\ntwo"
    assert first_paragraph("no blank line") == "no blank line"


def test_full_mode_is_the_question_itself():
    assert retrieval_query(QUESTION, "full") == QUESTION["question"]


def test_title_mode_is_only_the_title():
    assert retrieval_query(QUESTION, "title") == "JSONLoader fails"


def test_title_first_paragraph_mode_drops_the_rest_of_the_body():
    query = retrieval_query(QUESTION, "title_first_paragraph")
    assert query == "JSONLoader fails\n\nI load a JSON file with JSONLoader.\nIt fails."
    assert "Traceback" not in query


def test_every_listed_mode_works_and_an_unknown_mode_fails():
    for mode in QUERY_MODES:
        assert retrieval_query(QUESTION, mode)
    with pytest.raises(ValueError):
        retrieval_query(QUESTION, "body")
