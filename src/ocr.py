"""Offline OCR for scanned PDF pages using RapidOCR (pip-only, no Tesseract install needed)."""
from __future__ import annotations

from functools import lru_cache
from typing import Callable

import fitz  # PyMuPDF
import numpy as np

from src.pdf_processor import ProcessedDocument, refresh_scan_status
from utils.config import OCR_DPI, OCR_MAX_PAGES, SCANNED_MIN_CHARS_PER_PAGE

LOW_TEXT_PAGE_RATIO = 0.3  # run OCR if at least 30% of pages have almost no text


class OCRError(Exception):
    """User-friendly OCR failure."""


@lru_cache(maxsize=1)
def get_engine():
    """Load the OCR engine once. Models are bundled inside the pip package."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise OCRError(
            "OCR engine is not installed. Run: pip install rapidocr-onnxruntime"
        ) from exc
    return RapidOCR()


def _result_to_text(result) -> str:
    """Convert RapidOCR output [[box, text, score], ...] into reading-order text lines."""
    if not result:
        return ""
    boxes = []
    for item in result:
        box, text = item[0], item[1]
        xs = [point[0] for point in box]
        ys = [point[1] for point in box]
        boxes.append((min(ys), max(ys), min(xs), str(text)))
    boxes.sort()

    lines: list[list[tuple[float, str]]] = []
    current: list[tuple[float, str]] = []
    line_top = line_bottom = 0.0
    for top, bottom, left, text in boxes:
        if current and top < line_top + 0.6 * (line_bottom - line_top):
            current.append((left, text))  # same visual line
            line_bottom = max(line_bottom, bottom)
        else:
            if current:
                lines.append(current)
            current = [(left, text)]
            line_top, line_bottom = top, bottom
    if current:
        lines.append(current)
    return "\n".join(" ".join(t for _, t in sorted(line)) for line in lines)


def ocr_page(page, engine=None) -> str:
    """Render one PyMuPDF page to an image and run OCR on it."""
    engine = engine or get_engine()
    pix = page.get_pixmap(dpi=OCR_DPI, colorspace=fitz.csRGB, alpha=False)
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    image = np.ascontiguousarray(image[:, :, ::-1])  # RGB -> BGR (OpenCV convention)
    result, _ = engine(image)
    return _result_to_text(result)


def pages_needing_ocr(doc: ProcessedDocument) -> list:
    return [p for p in doc.pages if len(p.text) < SCANNED_MIN_CHARS_PER_PAGE]


def needs_ocr(doc: ProcessedDocument) -> bool:
    if doc.pages_processed == 0:
        return False
    low = len(pages_needing_ocr(doc))
    return doc.is_scanned or low / doc.pages_processed >= LOW_TEXT_PAGE_RATIO


def ocr_document(
    doc: ProcessedDocument,
    data: bytes,
    progress: Callable[[int, int], None] | None = None,
    engine=None,
) -> int:
    """OCR the low-text pages of `doc` in place. Returns how many pages got text."""
    targets = pages_needing_ocr(doc)
    if not targets:
        return 0
    limited = targets[:OCR_MAX_PAGES]
    if len(targets) > len(limited):
        doc.warnings.append(
            f"OCR was limited to the first {len(limited)} of {len(targets)} pages that need it "
            f"(OCR_MAX_PAGES)."
        )

    engine = engine or get_engine()  # raises OCRError if the engine is missing
    failed: list[int] = []
    done_pages: list[int] = []

    try:
        pdf = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise OCRError("Could not reopen the PDF for OCR.") from exc

    with pdf:
        for i, page in enumerate(limited, start=1):
            try:
                text = ocr_page(pdf.load_page(page.page_number - 1), engine).strip()
            except Exception:
                failed.append(page.page_number)
                text = ""
            if text:
                page.text = text
                done_pages.append(page.page_number)
            if progress:
                progress(i, len(limited))

    doc.ocr_pages = sorted(set(doc.ocr_pages) | set(done_pages))
    if done_pages:
        shown = ", ".join(map(str, done_pages[:10])) + (" ..." if len(done_pages) > 10 else "")
        doc.warnings.append(
            f"OCR was used on page(s) {shown}. The text may contain recognition errors."
        )
    if failed:
        doc.warnings.append(f"OCR failed on page(s): {', '.join(map(str, failed[:10]))}")
    refresh_scan_status(doc)
    return len(done_pages)
