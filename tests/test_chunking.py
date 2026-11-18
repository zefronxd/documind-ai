import pytest

from src.chunking import chunk_document, split_text
from src.pdf_processor import PageText, ProcessedDocument


def test_chunks_respect_size_and_overlap():
    text = "word " * 600
    chunks = split_text(text, 500, 100)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        split_text("hello", 100, 100)


def test_empty_text_gives_no_chunks():
    assert split_text("   ", 100, 10) == []


def test_page_numbers_preserved():
    doc = ProcessedDocument(
        doc_id="d1", name="a.pdf", size_bytes=1, page_count=2,
        pages=[PageText(1, "First page text. " * 5), PageText(2, "Second page text. " * 5)],
    )
    chunks = chunk_document(doc, 60, 10)
    assert {c.page_number for c in chunks} == {1, 2}
    assert all(c.doc_name == "a.pdf" for c in chunks)
