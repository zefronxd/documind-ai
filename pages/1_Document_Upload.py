"""Document Upload page: validate, extract (with optional OCR) and manage PDFs."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ocr import OCRError, needs_ocr, ocr_document
from src.pdf_processor import (
    PDFProcessingError,
    compute_doc_id,
    extract_pdf,
    validate_pdf_upload,
)
from utils.config import MAX_PAGES_PER_DOC, MAX_UPLOAD_MB, OCR_MAX_PAGES
from utils.helpers import format_file_size, get_documents, init_session_state, page_header

init_session_state()
page_header("Document Upload", "Upload one or more PDFs. Text is extracted page by page.")
st.caption(f"Max {MAX_UPLOAD_MB} MB per file · first {MAX_PAGES_PER_DOC} pages processed per document.")

use_ocr = st.checkbox(
    "Run OCR on scanned / image-based pages",
    value=True,
    help=f"Offline OCR (RapidOCR). Slower: about 1-3 seconds per page, up to {OCR_MAX_PAGES} pages per document.",
)

docs = get_documents()

uploaded = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state['uploader_key']}",
)

if uploaded:
    existing_ids = {d.doc_id for d in docs}
    added = 0
    for f in uploaded:
        data = f.getvalue()
        try:
            validate_pdf_upload(f.name, data)
            if compute_doc_id(data) in existing_ids:
                continue  # already loaded in this session
            with st.spinner(f"Extracting text from {f.name}..."):
                doc = extract_pdf(f.name, data)
        except PDFProcessingError as exc:
            st.error(f"**{f.name}**: {exc}")
            continue
        except Exception:
            st.error(f"**{f.name}**: unexpected error while reading this file.")
            continue

        if use_ocr and needs_ocr(doc):
            bar = st.progress(0.0, text=f"OCR: {f.name}")
            try:
                ocr_document(
                    doc,
                    data,
                    progress=lambda done, total, name=f.name, b=bar: b.progress(
                        done / total, text=f"OCR {name}: page {done}/{total}"
                    ),
                )
            except OCRError as exc:
                st.warning(f"**{f.name}**: {exc}")
            except Exception:
                st.warning(f"**{f.name}**: OCR failed unexpectedly. Extracted text is used as is.")
            finally:
                bar.empty()

        docs.append(doc)
        existing_ids.add(doc.doc_id)
        added += 1
    if added:
        st.session_state["docs_version"] += 1
        st.success(f"{added} document(s) processed successfully.")

st.divider()

if not docs:
    st.info("No documents loaded yet.")
    st.stop()


def text_status(d) -> str:
    if d.is_scanned:
        return "Scanned / low text"
    if d.ocr_pages:
        return f"Text OK (OCR on {len(d.ocr_pages)} pages)"
    return "Text OK"


st.subheader(f"Loaded documents ({len(docs)})")
table = pd.DataFrame(
    [
        {
            "File name": d.name,
            "Size": format_file_size(d.size_bytes),
            "Pages": d.page_count,
            "Pages processed": d.pages_processed,
            "Characters": f"{d.total_chars:,}",
            "Text status": text_status(d),
        }
        for d in docs
    ]
)
st.dataframe(table, hide_index=True, use_container_width=True)

for doc in docs:
    with st.expander(f"📄 {doc.name}"):
        for warning in doc.warnings:
            st.warning(warning)
        page_no = st.selectbox(
            "Preview page",
            options=[p.page_number for p in doc.pages],
            key=f"page_{doc.doc_id}",
        )
        preview = next(p.text for p in doc.pages if p.page_number == page_no)
        source = "OCR" if page_no in doc.ocr_pages else "PDF text layer"
        st.text_area(
            f"Extracted text (first 2000 characters) · source: {source}",
            value=preview[:2000] or "(no extractable text on this page)",
            height=200,
            disabled=True,
            key=f"preview_{doc.doc_id}_{page_no}",
        )
        if st.button("Remove this document", key=f"rm_{doc.doc_id}"):
            st.session_state["documents"] = [d for d in docs if d.doc_id != doc.doc_id]
            st.session_state["docs_version"] += 1
            st.session_state["uploader_key"] += 1  # reset uploader so it won't re-add the file
            st.rerun()

if st.button("Clear all documents", type="secondary"):
    st.session_state["documents"] = []
    st.session_state["docs_version"] += 1
    st.session_state["uploader_key"] += 1
    st.rerun()
