"""PDF validation and page-wise text extraction using PyMuPDF."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from utils.config import (
    MAX_PAGES_PER_DOC,
    MAX_UPLOAD_MB,
    SCANNED_MIN_CHARS_PER_PAGE,
)

SCAN_WARNING = (
    "Little or no extractable text found. This looks like a scanned/image-based PDF."
)


class PDFProcessingError(Exception):
    """Raised with a user-friendly message when a PDF cannot be processed."""


@dataclass
class PageText:
    page_number: int  # 1-based, preserved for citations
    text: str


@dataclass
class ProcessedDocument:
    doc_id: str
    name: str
    size_bytes: int
    page_count: int  # pages in the original file
    pages: list[PageText] = field(default_factory=list)  # pages actually processed
    is_scanned: bool = False
    warnings: list[str] = field(default_factory=list)
    ocr_pages: list[int] = field(default_factory=list)  # pages whose text came from OCR

    @property
    def pages_processed(self) -> int:
        return len(self.pages)

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)

    @property
    def has_text(self) -> bool:
        return self.total_chars > 0


def compute_doc_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def validate_pdf_upload(filename: str, data: bytes) -> None:
    """Cheap checks before parsing. Raises PDFProcessingError."""
    if not filename.lower().endswith(".pdf"):
        raise PDFProcessingError("Only PDF files (.pdf) are supported.")
    if not data:
        raise PDFProcessingError("The file is empty (0 bytes).")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise PDFProcessingError(f"File is larger than the {MAX_UPLOAD_MB} MB limit.")
    if not data[:1024].lstrip().startswith(b"%PDF"):
        raise PDFProcessingError("This file does not look like a valid PDF.")


def refresh_scan_status(doc: ProcessedDocument) -> None:
    """Recompute is_scanned and the scan warning (also called again after OCR)."""
    doc.warnings = [w for w in doc.warnings if w != SCAN_WARNING]
    chars_per_page = doc.total_chars / max(doc.pages_processed, 1)
    doc.is_scanned = doc.total_chars == 0 or chars_per_page < SCANNED_MIN_CHARS_PER_PAGE
    if doc.is_scanned:
        doc.warnings.append(SCAN_WARNING)


def extract_pdf(filename: str, data: bytes) -> ProcessedDocument:
    """Extract text page by page. The document is only read, never executed."""
    validate_pdf_upload(filename, data)

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PDFProcessingError("The PDF is corrupted or could not be opened.") from exc

    with doc:
        if doc.needs_pass:
            raise PDFProcessingError(
                "The PDF is password-protected. Please remove the password and upload again."
            )
        if doc.page_count == 0:
            raise PDFProcessingError("The PDF has no pages.")

        result = ProcessedDocument(
            doc_id=compute_doc_id(data),
            name=filename,
            size_bytes=len(data),
            page_count=doc.page_count,
        )

        limit = min(doc.page_count, MAX_PAGES_PER_DOC)
        if doc.page_count > limit:
            result.warnings.append(
                f"Large document: only the first {limit} of {doc.page_count} pages were processed."
            )

        failed_pages: list[int] = []
        for index in range(limit):
            try:
                text = doc.load_page(index).get_text("text")
                text = text.replace("\x00", "").strip()
            except Exception:
                text = ""
                failed_pages.append(index + 1)
            result.pages.append(PageText(page_number=index + 1, text=text))

        if failed_pages:
            result.warnings.append(
                f"Text could not be read from page(s): {', '.join(map(str, failed_pages[:10]))}"
                + (" ..." if len(failed_pages) > 10 else "")
            )

    refresh_scan_status(result)
    return result
