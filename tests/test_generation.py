"""Unit tests for the prompt and the assistant, with a fake retriever and a fake LLM (no index or Ollama needed)."""

import json

import rag_assistant.prompt as prompt_module
from rag_assistant.assistant import INSUFFICIENT_TEXT, Assistant, is_refusal_text, parse_answer, repair_citation
from rag_assistant.llm import LLM, LLMResponse
from rag_assistant.prompt import TRUNCATION_MARKER, build_prompt, format_source, prompt_fingerprint
from rag_assistant.retrieval import RetrievedDoc, Retriever

CORPUS = {"a.py_0_10": "alpha text", "b.md_0_10": "beta text", "c.ts_0_10": "gamma text"}
RETRIEVED = (RetrievedDoc("a.py_0_10", 3.0, 1), RetrievedDoc("b.md_0_10", 2.0, 2))


def llm_response(text: str, truncated: bool = False) -> LLMResponse:
    return LLMResponse(text=text, model="fake", input_tokens=10, output_tokens=5, latency_s=0.1, truncated=truncated)


def json_answer(cited: list[str], insufficient: bool = False) -> str:
    return json.dumps({"answer": "the answer", "cited_doc_ids": cited, "insufficient_information": insufficient})


class FakeRetriever(Retriever):
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, k: int) -> list[RetrievedDoc]:
        self.queries.append(query)
        return list(RETRIEVED[:k])


class FakeLLM(LLM):
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    def generate(self, prompt, *, system=None, json_schema=None, max_tokens=1024):
        self.prompts.append(prompt)
        return llm_response(self.text)


def test_format_source_truncates_long_text():
    assert format_source("x", "short", max_chars=10) == '<source doc_id="x">\nshort\n</source>'
    assert TRUNCATION_MARKER in format_source("x", "a" * 50, max_chars=10)


def test_build_prompt_contains_sources_in_order_and_question():
    prompt = build_prompt("my question", [("a", "first"), ("b", "second")], max_chars=100)
    assert prompt.index('doc_id="a"') < prompt.index('doc_id="b"') < prompt.index("my question")


def test_valid_citations_are_kept():
    answer = parse_answer("q", llm_response(json_answer(["a.py_0_10"])), RETRIEVED)
    assert answer.valid_output
    assert answer.cited_doc_ids == ("a.py_0_10",)
    assert answer.invalid_citations == ()


def test_citations_to_unretrieved_docs_are_dropped():
    answer = parse_answer("q", llm_response(json_answer(["a.py_0_10", "c.ts_0_10", "a.py_0_10"])), RETRIEVED)
    assert answer.cited_doc_ids == ("a.py_0_10",)
    assert answer.invalid_citations == ("c.ts_0_10",)


def test_insufficient_information_is_passed_through():
    answer = parse_answer("q", llm_response(json_answer([], insufficient=True)), RETRIEVED)
    assert answer.insufficient_information
    assert answer.valid_output


def test_empty_answer_counts_as_insufficient_information():
    text = json.dumps({"answer": "  ", "cited_doc_ids": ["a.py_0_10"], "insufficient_information": False})
    answer = parse_answer("q", llm_response(text), RETRIEVED)
    assert answer.insufficient_information
    assert answer.valid_output
    assert answer.cited_doc_ids == ()
    assert answer.text.strip()


def test_a_refusal_gets_the_standard_text_and_no_citations():
    text = json.dumps(
        {"answer": "insufficient_information", "cited_doc_ids": ["a.py_0_10"], "insufficient_information": True}
    )
    answer = parse_answer("q", llm_response(text), RETRIEVED)
    assert answer.insufficient_information and answer.valid_output
    assert answer.text == INSUFFICIENT_TEXT
    assert answer.cited_doc_ids == () and answer.invalid_citations == ()


def test_a_short_refusal_in_words_counts_as_a_refusal():
    text = json.dumps(
        {"answer": "I can't answer that question", "cited_doc_ids": ["a.py_0_10"], "insufficient_information": False}
    )
    answer = parse_answer("q", llm_response(text), RETRIEVED)
    assert answer.insufficient_information and answer.text == INSUFFICIENT_TEXT and answer.cited_doc_ids == ()


def test_the_standard_sentence_without_the_flag_counts_as_a_refusal():
    # The model wrote the exact sentence the prompt asks for, but left insufficient_information false.
    text = json.dumps({"answer": INSUFFICIENT_TEXT, "cited_doc_ids": [], "insufficient_information": False})
    answer = parse_answer("q", llm_response(text), RETRIEVED)
    assert answer.insufficient_information and answer.text == INSUFFICIENT_TEXT


def test_is_refusal_text_recognises_contain_enough_information_wordings():
    assert is_refusal_text(INSUFFICIENT_TEXT)
    assert is_refusal_text("The sources do not contain enough information to answer this question.")
    assert is_refusal_text("The knowledge source doesn't contain enough information.")
    assert is_refusal_text("The sources do not have enough information to answer this.")


def test_is_refusal_text_leaves_partial_answers_alone():
    assert is_refusal_text("I cannot answer this question.")
    assert is_refusal_text("The sources contain no relevant information.")
    partial = (
        "Use Pinecone.from_existing_index with the same embeddings. "
        "The part I cannot answer is how to filter by metadata, because the sources do not cover it."
    )
    assert not is_refusal_text(partial)  # longer: a partial answer, which the prompt allows
    assert not is_refusal_text("Call loader.load() to get the documents.")


PROVIDED = [
    "llama_index/llama-index-integrations/llms/llama-index-llms-dashscope/README.md_0_2044",
    "langchain/docs/docs/integrations/vectorstores/pinecone.ipynb_6363_13158",
    "langchain/libs/text-splitters/tests/unit_tests/test_text_splitters.py_0_8012",
    "langchain/libs/text-splitters/tests/unit_tests/test_text_splitters.py_8015_15553",
]


def test_repair_citation_by_file_name_and_byte_range():
    # The middle of the path left out, or the wrong repository: the file name with byte range is unique.
    assert repair_citation("llama_index/llms-dashscope/README.md_0_2044", PROVIDED) == PROVIDED[0]
    assert repair_citation("langchainjs/docs/vectorstores/pinecone.ipynb_6363_13158", PROVIDED) == PROVIDED[1]


def test_repair_citation_by_file_name_alone():
    # Extension and byte range cut off ("pinecone.ipb"): the file name alone is still unique.
    assert repair_citation("langchain/docs/docs/integrations/vectorstores/pinecone.ipb", PROVIDED) == PROVIDED[1]


def test_no_repair_when_nothing_or_more_than_one_chunk_matches():
    assert repair_citation("langchain/.../test_character_text_splitter.py_0_8015", PROVIDED) is None
    assert repair_citation("insufficient_information", PROVIDED) is None
    assert repair_citation("tests/test_text_splitters.py", PROVIDED) is None  # two chunks of that file


def test_parse_answer_repairs_and_reports_mangled_citations():
    retrieved = (RetrievedDoc("repo/docs/a.py_0_10", 3.0, 1), RetrievedDoc("repo/docs/b.md_0_10", 2.0, 2))
    cited = ["other/a.py_0_10", "repo/docs/a.py_0_10", "unknown.py_0_5"]
    answer = parse_answer("q", llm_response(json_answer(cited)), retrieved)
    assert answer.cited_doc_ids == ("repo/docs/a.py_0_10",)  # repaired, and the duplicate is not counted twice
    assert answer.repaired_citations == (("other/a.py_0_10", "repo/docs/a.py_0_10"),)
    assert answer.invalid_citations == ("unknown.py_0_5",)


def test_invalid_json_gives_invalid_output():
    answer = parse_answer("q", llm_response('{"answer": "cut of', truncated=True), RETRIEVED)
    assert not answer.valid_output
    assert answer.cited_doc_ids == ()
    assert "max_tokens" in answer.text


def test_missing_field_gives_invalid_output():
    answer = parse_answer("q", llm_response('{"answer": "no citations field"}'), RETRIEVED)
    assert not answer.valid_output


def test_assistant_sends_only_top_k_retrieved_sources_and_the_question():
    llm = FakeLLM(json_answer(["a.py_0_10"]))
    assistant = Assistant(FakeRetriever(), llm, CORPUS, top_k=1, max_chars=100)
    answer = assistant.ask("what is alpha?")
    prompt = llm.prompts[0]
    assert "alpha text" in prompt and "what is alpha?" in prompt
    assert "beta text" not in prompt
    assert answer.retrieved == RETRIEVED[:1]


def test_assistant_searches_with_the_retrieval_query_but_answers_the_full_question():
    retriever, llm = FakeRetriever(), FakeLLM(json_answer(["a.py_0_10"]))
    assistant = Assistant(retriever, llm, CORPUS, top_k=1, max_chars=100)
    assistant.ask("Title\n\nlong body with logs", retrieval_query="Title")
    assistant.ask("only a question")
    assert retriever.queries == ["Title", "only a question"]
    assert "long body with logs" in llm.prompts[0]


def test_prompt_fingerprint_changes_when_the_prompt_changes(monkeypatch):
    original = prompt_fingerprint()
    assert original == prompt_fingerprint()
    monkeypatch.setattr(prompt_module, "SYSTEM_PROMPT", prompt_module.SYSTEM_PROMPT + " Be brief.")
    assert prompt_fingerprint() != original
    monkeypatch.undo()
    monkeypatch.setattr(prompt_module, "TRUNCATION_MARKER", "\n[cut]")
    assert prompt_fingerprint() != original
