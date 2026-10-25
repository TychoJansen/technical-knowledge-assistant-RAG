"""Unit tests for the insufficient-context evaluation, with a tiny BM25 index and a fake LLM (no data or Ollama)."""

import json

import pytest

from rag_assistant.assistant import Assistant
from rag_assistant.evaluation.abstention_eval import (
    RestrictedRetriever,
    aggregate,
    evaluate_insufficient_context,
    insufficient_context_candidates,
)
from rag_assistant.llm import LLM, LLMResponse
from rag_assistant.retrieval import BM25Retriever

CORPUS = {
    "rel": "load a json file with the json loader",
    "dup": "load a json file with the json loader",  # same text as the relevant chunk
    "neg1": "the json loader reads json lines",
    "neg2": "split text into chunks with a text splitter",
    "neg3": "vector store with chroma",
    "both": "json loader jq schema",  # judged not relevant for one nugget, relevant for another
}
QUESTION = {
    "qid": "q1",
    "question": "How do I use the json loader?",
    "title": "json loader",
    "tags": ["python"],
    "relevant_doc_ids": ["rel"],
    "nuggets": [
        {"text": "use JSONLoader", "relevant_corpus_ids": ["rel"], "non_relevant_corpus_ids": ["dup", "neg1", "both"]},
        {"text": "set jq_schema", "relevant_corpus_ids": ["both"], "non_relevant_corpus_ids": ["neg2", "neg3", "gone"]},
    ],
}


@pytest.fixture(scope="module")
def bm25():
    return BM25Retriever.build(list(CORPUS), list(CORPUS.values()), "word_stem")


class FakeLLM(LLM):
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def generate(self, prompt, *, system=None, json_schema=None, max_tokens=1024):
        self.prompts.append(prompt)
        return LLMResponse(self.text, "fake", 10, 5, 0.1, False)


def test_candidates_are_judged_not_relevant_and_never_relevant():
    # "both" is relevant for a nugget, "dup" duplicates a relevant chunk, "gone" is not in the corpus.
    assert insufficient_context_candidates(QUESTION, CORPUS) == ["neg1", "neg2", "neg3"]


def test_search_within_returns_only_the_allowed_documents_with_the_normal_scores(bm25):
    hits = bm25.search_within("json loader", 4, ["neg1", "neg2", "unknown"])
    assert [h.doc_id for h in hits] == ["neg1"]  # neg2 has no matching term (score 0), unknown ids are ignored
    full = {h.doc_id: h.score for h in bm25.search("json loader", 10)}
    assert hits[0].score == pytest.approx(full["neg1"])
    assert bm25.search_within("json loader", 4, []) == []


def test_restricted_retriever_ranks_within_its_set(bm25):
    hits = RestrictedRetriever(bm25, ["neg1", "rel", "neg3"]).search("json loader", 2)
    assert [h.doc_id for h in hits] == ["rel", "neg1"]


def test_the_assistant_only_sees_the_insufficient_context(bm25):
    llm = FakeLLM(json.dumps({"answer": "", "cited_doc_ids": [], "insufficient_information": True}))
    assistant = Assistant(bm25, llm, CORPUS, top_k=4, max_chars=100)
    record = evaluate_insufficient_context(QUESTION, assistant)
    assert set(record["retrieved"]) <= {"neg1", "neg2", "neg3"}
    assert "rel" not in llm.prompts[0] and CORPUS["rel"] not in llm.prompts[0]
    assert record["kind"] == "insufficient_context" and record["insufficient_information"]
    assert record["n_candidates"] == 3


def test_aggregate():
    def rec(valid, insufficient, cited=()):
        return {
            "valid_output": valid,
            "insufficient_information": insufficient,
            "cited_doc_ids": list(cited),
            "invalid_citations": [],
            "retrieved": ["a", "b"],
            "generation_s": 10.0,
        }

    m = aggregate([rec(True, True), rec(True, False, ["a"]), rec(True, False), rec(False, False)])
    assert m["correct_abstention_rate"] == pytest.approx(0.25)
    assert m["false_answer_rate"] == pytest.approx(0.5)
    assert m["invalid_rate"] == pytest.approx(0.25)
    assert m["cited_false_answer_rate"] == pytest.approx(0.5)
    assert m["context_size_mean"] == pytest.approx(2.0)
