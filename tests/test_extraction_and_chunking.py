import pytest
from docx import Document as DocxDocument

from app.services.chunking import Chunk, select_text_for_insights, split_pages
from app.services.extraction import ExtractionError, Page, clean_pdf_text, extract_pages


def test_extract_text_file_strips_nul_bytes(tmp_path):
    path = tmp_path / "note.txt"
    path.write_bytes(b"hello\x00 world")
    assert extract_pages(path) == [Page(1, "hello world")]


def test_extract_docx_includes_paragraphs_and_tables(tmp_path):
    doc = DocxDocument()
    doc.add_paragraph("Quarterly report")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Revenue"
    table.rows[0].cells[1].text = "10M"
    path = tmp_path / "report.docx"
    doc.save(path)

    text = extract_pages(path)[0].text
    assert "Quarterly report" in text
    assert "Revenue | 10M" in text


def test_pdf_word_per_line_text_is_restored_but_paragraphs_are_kept():
    broken = "interface\n \nwith\n \ndocuments\n   \n \nis\n\nNew paragraph"
    assert clean_pdf_text(broken) == "interface with documents is\n\nNew paragraph"


def test_empty_file_raises_clear_error(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n")
    with pytest.raises(ExtractionError, match="No text"):
        extract_pages(path)


def test_corrupted_pdf_raises_clear_error(tmp_path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"not really a pdf")
    with pytest.raises(ExtractionError):
        extract_pages(path)


def test_chunks_keep_page_numbers_and_respect_size():
    long_text = " ".join(f"word{i}" for i in range(3000))
    chunks = split_pages([Page(1, "short page"), Page(2, long_text)])

    assert chunks[0].page_number == 1
    assert {c.page_number for c in chunks[1:]} == {2}
    assert len(chunks) > 2
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.token_count <= 800 for c in chunks)


def test_insight_text_is_sampled_across_whole_document_when_over_budget():
    chunks = [Chunk(i, 1, f"chunk-{i}", 100) for i in range(100)]
    text = select_text_for_insights(chunks, token_budget=1000)

    assert "chunk-0" in text
    assert "chunk-90" in text  # not just the first pages
    assert text.count("chunk-") == 10


def test_insight_text_is_whole_document_when_under_budget():
    chunks = [Chunk(i, 1, f"chunk-{i}", 10) for i in range(3)]
    assert select_text_for_insights(chunks, 1000) == "chunk-0\n\nchunk-1\n\nchunk-2"
