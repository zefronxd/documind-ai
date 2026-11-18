import sys

import fitz
import pytest

from src import ocr
from src.ocr import OCRError, _result_to_text, needs_ocr, ocr_document
from src.pdf_processor import extract_pdf


def blank_pdf(pages=2) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def fake_engine(image):
    # image must be an RGB/BGR numpy array of shape (h, w, 3)
    assert image.ndim == 3 and image.shape[2] == 3
    text = "Scanned invoice total amount is 500 rupees and was paid on time"
    return [[[[0, 0], [100, 0], [100, 20], [0, 20]], text, 0.95]], 0.01


def test_result_to_text_groups_lines_in_reading_order():
    result = [
        [[[120, 0], [200, 0], [200, 20], [120, 20]], "World", 0.9],
        [[[0, 2], [100, 2], [100, 22], [0, 22]], "Hello", 0.9],
        [[[0, 50], [100, 50], [100, 70], [0, 70]], "Next line", 0.9],
    ]
    assert _result_to_text(result) == "Hello World\nNext line"


def test_empty_ocr_result_gives_empty_text():
    assert _result_to_text(None) == ""


def test_ocr_fills_scanned_pages():
    data = blank_pdf(2)
    doc = extract_pdf("scan.pdf", data)
    assert doc.is_scanned and needs_ocr(doc)

    done = ocr_document(doc, data, engine=fake_engine)

    assert done == 2
    assert doc.ocr_pages == [1, 2]
    assert "invoice" in doc.pages[0].text
    assert doc.is_scanned is False
    assert doc.pages[1].page_number == 2  # page numbers stay intact for citations


def test_text_pdf_does_not_need_ocr():
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "This page has a real text layer with plenty of characters in it.")
    data = pdf.tobytes()
    pdf.close()
    doc = extract_pdf("text.pdf", data)
    assert needs_ocr(doc) is False


def test_missing_engine_gives_friendly_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)  # makes the import fail
    ocr.get_engine.cache_clear()
    data = blank_pdf(1)
    doc = extract_pdf("scan.pdf", data)
    with pytest.raises(OCRError):
        ocr_document(doc, data)
    ocr.get_engine.cache_clear()
