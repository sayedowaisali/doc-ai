import uuid

from app.services.chat import build_context, build_overview, extract_citations
from app.services.llm import _cost
from app.services.retrieval import RetrievedChunk


def _chunks() -> list[RetrievedChunk]:
    doc_id = uuid.uuid4()
    return [
        RetrievedChunk(doc_id, "a.pdf", 1, "First passage text", 0.9),
        RetrievedChunk(doc_id, "a.pdf", 4, "Second passage text", 0.8),
        RetrievedChunk(doc_id, "b.docx", 1, "Third passage text", 0.7),
    ]


def test_context_numbers_passages_with_source():
    context = build_context(_chunks())
    assert context.startswith("[1] (document: a.pdf, page 1)")
    assert "[3] (document: b.docx, page 1)" in context


def test_citations_only_include_cited_passages_and_ignore_invalid_numbers():
    citations = extract_citations("Fact one [2]. Fact two [2][3]. Bogus [9] and [0].", _chunks())
    assert [c["index"] for c in citations] == [2, 3]
    assert citations[0]["page_number"] == 4
    assert citations[1]["filename"] == "b.docx"


def test_no_citations_for_uncited_answer():
    assert extract_citations("I couldn't find this in the provided documents.", _chunks()) == []


def test_overview_includes_summary_and_key_points():
    from types import SimpleNamespace

    doc = SimpleNamespace(filename="a.pdf", summary="About policies.", insights={"key_points": ["P1", "P2"]})
    overview = build_overview([doc])
    assert "Document: a.pdf" in overview and "Summary: About policies." in overview
    assert "- P1\n- P2" in overview


def test_cost_calculation_and_unknown_model():
    assert _cost("gpt-4o-mini", 1_000_000, 1_000_000) == 0.75
    assert _cost("some-future-model", 1000, 1000) == 0.0
