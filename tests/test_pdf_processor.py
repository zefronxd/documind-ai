import fitz
import pytest

from src.pdf_processor import PDFProcessingError, extract_pdf, validate_pdf_upload


def make_pdf(page_texts, **save_kwargs) -> bytes:
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = doc.tobytes(**save_kwargs)
    doc.close()
    return data


def test_extract_preserves_page_numbers():
    result = extract_pdf("a.pdf", make_pdf(["Hello page one", "Second page text"]))
    assert result.page_count == 2
    assert [p.page_number for p in result.pages] == [1, 2]
    assert "Hello" in result.pages[0].text
    assert "Second" in result.pages[1].text


def test_empty_pdf_flagged_as_scanned():
    result = extract_pdf("blank.pdf", make_pdf(["", ""]))
    assert result.total_chars == 0
    assert result.is_scanned is True


def test_rejects_non_pdf_bytes():
    with pytest.raises(PDFProcessingError):
        validate_pdf_upload("fake.pdf", b"this is not a pdf")


def test_rejects_wrong_extension():
    with pytest.raises(PDFProcessingError):
        validate_pdf_upload("notes.txt", b"%PDF-1.4")


def test_password_protected_pdf_rejected():
    data = make_pdf(
        ["secret"],
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="user123",
        owner_pw="owner123",
    )
    with pytest.raises(PDFProcessingError):
        extract_pdf("locked.pdf", data)
