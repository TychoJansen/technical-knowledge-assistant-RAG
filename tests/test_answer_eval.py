"""Unit tests for the judge parsing and the answer-evaluation aggregates, with a fake judge LLM."""

import json

import pytest

from rag_assistant.evaluation.answer_eval import aggregate_freshstack
from rag_assistant.evaluation.judge import (
    CITATION_SCHEMA,
    judge_citation,
    judge_nuggets,
    nugget_schema,
    quote_in_source,
)
from rag_assistant.llm import LLM, LLMResponse


class FakeJudge(LLM):
    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, prompt, *, system=None, json_schema=None, max_tokens=1024):
        return LLMResponse(self.text, "fake", 10, 5, 0.5, False)


def test_judge_nuggets_maps_fields_to_nuggets_in_order():
    judge = FakeJudge(json.dumps({"fact_2": True, "fact_1": False}))
    result = judge_nuggets(judge, "q", "answer", ["fact one", "fact two"])
    assert result.present == (False, True)
    assert result.parse_ok


def test_judge_nuggets_missing_facts_count_as_absent():
    judge = FakeJudge(json.dumps({"fact_1": True}))
    result = judge_nuggets(judge, "q", "answer", ["a", "b", "c"])
    assert result.present == (True, False, False)
    assert not result.parse_ok


def test_nugget_schema_requires_a_verdict_for_every_fact():
    schema = nugget_schema(3)
    assert schema["required"] == ["fact_1", "fact_2", "fact_3"]
    assert all(schema["properties"][k] == {"type": "boolean"} for k in schema["required"])


def test_nugget_judge_sends_the_per_question_schema_and_labels():
    seen = {}

    class RecordingJudge(FakeJudge):
        def generate(self, prompt, *, system=None, json_schema=None, max_tokens=1024):
            seen["prompt"], seen["schema"] = prompt, json_schema
            return super().generate(prompt, system=system, json_schema=json_schema, max_tokens=max_tokens)

    judge_nuggets(RecordingJudge('{"fact_1": true, "fact_2": false}'), "q", "answer", ["one", "two"])
    assert seen["schema"] == nugget_schema(2)
    assert "fact_1: one" in seen["prompt"] and "fact_2: two" in seen["prompt"]


def test_citation_schema_asks_for_the_verdict_before_the_quote():
    assert list(CITATION_SCHEMA["properties"]) == ["supported", "quote"]


def test_judge_nuggets_invalid_json():
    result = judge_nuggets(FakeJudge("not json"), "q", "answer", ["a"])
    assert result.present == (False,)
    assert not result.parse_ok


SOURCE_TEXT = "Use JSONLoader with jq_schema='.messages[]' to load each message as a separate document."


def test_quote_in_source_exact_and_normalised():
    assert quote_in_source("jq_schema='.messages[]' to load each message", SOURCE_TEXT)
    assert quote_in_source("USE  JSONLoader with\njq_schema", SOURCE_TEXT)  # case and whitespace differ


def test_quote_in_source_tolerates_a_small_copying_error():
    assert quote_in_source("Use JSONLoader with jq_schema='.messages[]' to load each mesage", SOURCE_TEXT)


def test_quote_in_source_rejects_paraphrase_and_short_quotes():
    assert not quote_in_source("Load every chat message as its own document with a JSON loader", SOURCE_TEXT)
    assert not quote_in_source("JSONLoader", SOURCE_TEXT)  # too short to be evidence
    assert not quote_in_source("", SOURCE_TEXT)


def test_judge_citation_supported_needs_a_verified_quote():
    verified = judge_citation(
        FakeJudge(json.dumps({"quote": "jq_schema='.messages[]' to load each message", "supported": True})),
        "q",
        "a",
        "d1",
        SOURCE_TEXT,
        1000,
    )
    assert verified.supported and verified.judge_supported and verified.quote_found and verified.parse_ok

    invented = judge_citation(
        FakeJudge(json.dumps({"quote": "call loader.split() to get documents", "supported": True})),
        "q",
        "a",
        "d1",
        SOURCE_TEXT,
        1000,
    )
    assert invented.judge_supported and not invented.quote_found
    assert not invented.supported  # the judge said yes, but its evidence is not in the source


def test_judge_citation_no_and_invalid_output():
    no = judge_citation(FakeJudge('{"quote": "", "supported": false}'), "q", "a", "d1", SOURCE_TEXT, 1000)
    assert not no.supported and no.parse_ok
    invalid = judge_citation(FakeJudge("{"), "q", "a", "d1", SOURCE_TEXT, 1000)
    assert not invalid.supported and not invalid.parse_ok


def record(present, insufficient=False, valid=True, cited=("d1",), supported=(True,), relevant=(True,)):
    citations = [
        {
            "doc_id": d,
            "supported": s,
            "judge_supported": s,
            "quote_found": s,
            "judge_ok": True,
            "relevant_in_qrels": r,
            "judged_in_qrels": True,
        }
        for d, s, r in zip(cited, supported, relevant)
    ]
    return {
        "valid_output": valid,
        "insufficient_information": insufficient,
        "nuggets_present": list(present),
        "nugget_judge_ok": True,
        "cited_doc_ids": list(cited),
        "invalid_citations": [],
        "citations": citations,
        "context_coverage": 1.0,
        "context_recall": 0.5,
        "generation_s": 30.0,
        "judge_s": 20.0,
    }


def test_aggregate_freshstack():
    records = [
        record([True, False]),  # answered, half the nuggets, 1 supported relevant citation
        record([True, True], cited=("d1", "d2"), supported=(True, False), relevant=(False, False)),
        record([False, False], insufficient=True, cited=(), supported=(), relevant=()),  # false abstention
        record([False], valid=False, cited=(), supported=(), relevant=()),  # invalid output
    ]
    m = aggregate_freshstack(records)
    assert m["n"] == 4
    assert m["answered_rate"] == pytest.approx(0.5)
    assert m["false_abstention_rate"] == pytest.approx(0.25)
    assert m["invalid_rate"] == pytest.approx(0.25)
    assert m["nugget_recall"] == pytest.approx((0.5 + 1.0 + 0 + 0) / 4)
    assert m["nugget_recall_answered"] == pytest.approx(0.75)
    assert m["n_citations"] == 3
    assert m["citation_support"] == pytest.approx(2 / 3)
    assert m["citation_precision_qrels"] == pytest.approx(1 / 3)
    assert m["uncited_answer_rate"] == pytest.approx(0.0)


class StubAssistant:
    """Returns a fixed answer citing d1, without retrieval or an LLM."""

    corpus = {"d1": "some text", "d2": "other text"}
    max_chars = 100

    def ask(self, question, retrieval_query=None):
        from rag_assistant.assistant import Answer
        from rag_assistant.retrieval import RetrievedDoc

        response = LLMResponse('{"answer": "use X"}', "fake", 10, 5, 0.1, False)
        retrieved = (RetrievedDoc("d1", 2.0, 1), RetrievedDoc("d2", 1.0, 2))
        return Answer(question, "use X", ("d1",), (), False, True, retrieved, response)


QUESTION = {
    "qid": "q1",
    "question": "How?",
    "title": "How?",
    "tags": ["python"],
    "relevant_doc_ids": ["d1"],
    "nuggets": [{"text": "uses X", "relevant_corpus_ids": ["d1"], "non_relevant_corpus_ids": ["d2"]}],
}


def test_no_judge_keeps_deterministic_fields_and_leaves_judge_fields_empty():
    from rag_assistant.evaluation.answer_eval import evaluate_freshstack

    r = evaluate_freshstack(QUESTION, StubAssistant(), judge=None)
    assert r["nuggets_present"] is None and r["nugget_judge_ok"] is None and r["judge_s"] is None
    assert r["context_coverage"] == 1.0
    assert r["citations"] == [
        {
            "doc_id": "d1",
            "supported": None,
            "judge_supported": None,
            "quote": None,
            "quote_found": None,
            "judge_ok": None,
            "relevant_in_qrels": True,
            "judged_in_qrels": True,
        }
    ]


def test_query_mode_changes_only_the_search_query():
    from rag_assistant.evaluation.answer_eval import evaluate_freshstack

    calls = []

    class RecordingAssistant(StubAssistant):
        def ask(self, question, retrieval_query=None):
            calls.append((question, retrieval_query))
            return super().ask(question, retrieval_query)

    full = "How?\n\nbody"
    question = {**QUESTION, "question": full, "body": "body"}
    evaluate_freshstack(question, RecordingAssistant(), judge=None, query_mode="title")
    evaluate_freshstack(question, RecordingAssistant(), judge=None)
    assert calls == [(full, "How?"), (full, full)]


def test_no_judge_aggregate_gives_none_for_judge_metrics_only():
    from rag_assistant.evaluation.answer_eval import evaluate_freshstack

    m = aggregate_freshstack([evaluate_freshstack(QUESTION, StubAssistant(), judge=None)])
    for key in [
        "nugget_recall",
        "nugget_recall_answered",
        "citation_support",
        "judge_parse_failure_rate",
        "judge_s_mean",
    ]:
        assert m[key] is None, key
    assert m["answered_rate"] == 1.0
    assert m["citation_precision_qrels"] == 1.0
    assert m["context_coverage"] == 1.0


def test_quote_check_ignores_the_source_label():
    # Quoting the doc_id (it is in the <source doc_id=...> label the judge sees) is no evidence.
    doc_id = "langchain/docs/docs/how_to/document_loader_json.mdx_0_8249"
    judgment = judge_citation(
        FakeJudge(json.dumps({"quote": doc_id, "supported": True})),
        "q",
        "a",
        doc_id,
        "Unrelated text about cats.",
        1000,
    )
    assert judgment.judge_supported and not judgment.quote_found and not judgment.supported


def test_quote_check_only_uses_the_text_the_judge_saw():
    text = "a" * 50 + " the supporting sentence is here after the cut"
    judgment = judge_citation(
        FakeJudge(json.dumps({"quote": "the supporting sentence is here", "supported": True})), "q", "a", "d1", text, 50
    )
    assert not judgment.supported  # the quote is beyond max_chars, so the judge could not have seen it


def test_is_degenerate_answer():
    from rag_assistant.evaluation.judge import is_degenerate_answer

    assert is_degenerate_answer("No", "Is X possible?")
    assert is_degenerate_answer("  How to load JSON files  ", "How to load JSON files in LangChain?")
    assert not is_degenerate_answer("Use JSONLoader with jq_schema='.messages[]'.", "How to load JSON files?")


def test_degenerate_answer_gets_no_nuggets_without_calling_the_judge():
    from rag_assistant.evaluation.answer_eval import evaluate_freshstack

    class ExplodingJudge(LLM):
        def generate(self, prompt, *, system=None, json_schema=None, max_tokens=1024):
            raise AssertionError("the judge should not be called for a degenerate answer")

    r = evaluate_freshstack(QUESTION, StubAssistant(), judge=ExplodingJudge())  # StubAssistant answers "use X"
    assert r["degenerate_answer"] is True
    assert r["nuggets_present"] == [False] and r["judge_s"] == 0.0
    assert r["citations"][0]["supported"] is False and r["citations"][0]["relevant_in_qrels"] is True


def test_select_questions_is_a_seeded_sample_in_split_order():
    from rag_assistant.evaluation.answer_eval import select_questions

    questions = [{"qid": str(i)} for i in range(60)]
    sample = select_questions(questions, 20)
    assert len(sample) == 20
    assert sample == select_questions(questions, 20)  # the same every time
    assert [q["qid"] for q in sample] == sorted((q["qid"] for q in sample), key=int)  # split order kept
    assert sample != questions[:20]  # not simply the first 20
    assert select_questions(questions, None) == questions


def test_warm_up_unloads_then_calls_the_llm_once():
    """The warm-up first unloads the model (clearing Ollama's prompt cache), then loads it with one call."""
    from rag_assistant.evaluation.answer_eval import warm_up

    calls = []

    class CountingLLM(FakeJudge):
        def generate(self, prompt, **kwargs):
            calls.append("generate")
            return super().generate(prompt, **kwargs)

        def unload(self):
            calls.append("unload")

        def settings(self):
            return {"model": "fake"}

    warm_up(CountingLLM("OK"))
    assert calls == ["unload", "generate"]


def test_ollama_unload_sends_keep_alive_zero(monkeypatch):
    """Unloading asks Ollama to drop the model right away (keep_alive 0)."""
    from rag_assistant.llm import OllamaLLM

    llm = OllamaLLM(model="llama3.2:3b")
    sent = {}
    monkeypatch.setattr(llm, "_post", lambda path, payload: sent.update(path=path, payload=payload) or {})
    llm.unload()
    assert sent == {"path": "/api/generate", "payload": {"model": "llama3.2:3b", "keep_alive": 0}}


def test_resume_drops_a_half_written_last_line(tmp_path):
    """A run stopped while writing leaves a broken last line; resuming drops it and redoes that question."""
    from rag_assistant.evaluation.answer_eval import load_records

    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps({"qid": "q1", "config": {"x": 1}}) + '\n{"qid": "q2", "conf', encoding="utf-8")
    assert list(load_records(path, {"x": 1})) == ["q1"]
    assert path.read_text(encoding="utf-8").count("\n") == 1  # the broken line is gone from the file


def test_resume_refuses_a_broken_line_in_the_middle(tmp_path):
    """A broken line that is not the last one is not a stopped run: stop instead of silently dropping data."""
    from rag_assistant.evaluation.answer_eval import load_records

    path = tmp_path / "run.jsonl"
    path.write_text('{"qid": "q1", "co\n' + json.dumps({"qid": "q2", "config": {}}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_records(path, {})
