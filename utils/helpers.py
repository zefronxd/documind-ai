"""Shared Streamlit helpers: session state, theme, guards, index management."""
from __future__ import annotations

import copy
import uuid

import streamlit as st

from utils.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, has_api_key

_DEFAULT_STATE = {
    "documents": [],        # list[ProcessedDocument]
    "docs_version": 0,      # bumped whenever documents change (triggers index rebuild)
    "uploader_key": 0,      # bumped to reset the file uploader widget
    "questions_asked": 0,
    "chat_history": [],     # list of {"role", "content", "citations"}
    "summaries": {},        # doc_id -> (DocumentSummary, truncated)
    "extractions": {},      # doc_id -> (ExtractedInfo, truncated)
    "comparisons": {},      # "idA:idB" -> (ComparisonResult, truncated)
    "rag": None,            # RAGPipeline
    "rag_signature": None,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
}


def init_session_state() -> None:
    for key, value in _DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(value)
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = uuid.uuid4().hex[:8]


def get_documents() -> list:
    return st.session_state["documents"]


def format_file_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def require_documents(need_llm: bool = True) -> list:
    """Page guard. Stops the page with a clear message if prerequisites are missing."""
    docs = get_documents()
    if not docs:
        st.info("No documents loaded yet. Please upload PDFs on the **Document Upload** page first.")
        st.stop()
    if need_llm and not has_api_key():
        st.error(
            "GEMINI_API_KEY is missing. Copy `.env.example` to `.env`, add your Gemini API key "
            "and restart the app. PDF upload and text extraction still work without it."
        )
        st.stop()
    usable = [d for d in docs if d.has_text]
    if not usable:
        st.warning(
            "None of the uploaded PDFs contain extractable text (they look scanned). "
            "Re-upload them with the OCR option enabled on the Document Upload page."
        )
        st.stop()
    return usable


def ensure_index():
    """Build (or reuse) the FAISS index for the current documents and settings."""
    from src.rag_pipeline import RAGPipeline  # lazy: heavy imports

    signature = (
        st.session_state["docs_version"],
        st.session_state["chunk_size"],
        st.session_state["chunk_overlap"],
    )
    if st.session_state["rag"] is None or st.session_state["rag_signature"] != signature:
        pipeline = RAGPipeline(session_dir=DATA_DIR / "sessions" / st.session_state["session_id"])
        docs = [d for d in get_documents() if d.has_text]
        with st.spinner("Building vector index (first run downloads the embedding model, ~90 MB)..."):
            pipeline.build_index(docs, st.session_state["chunk_size"], st.session_state["chunk_overlap"])
        st.session_state["rag"] = pipeline
        st.session_state["rag_signature"] = signature
    return st.session_state["rag"]


def render_list(items: list[str]) -> None:
    if not items:
        st.caption("Nothing found in the document.")
        return
    for item in items:
        st.markdown(f"- {item}")


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; max-width: 1200px; }
        section[data-testid="stSidebar"] { background: #0B1F3A; }
        section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
        .dm-header h1 { color: #0B1F3A; margin-bottom: 0; font-weight: 700; }
        .dm-header p { color: #475569; margin-top: 0.2rem; }
        .dm-kpi {
            background: #FFFFFF; border: 1px solid #E2E8F0; border-left: 5px solid #1E3A8A;
            border-radius: 10px; padding: 1rem 1.2rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        }
        .dm-kpi .label { color: #64748B; font-size: 0.85rem; }
        .dm-kpi .value { color: #0B1F3A; font-size: 1.9rem; font-weight: 700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="dm-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str | int) -> None:
    st.markdown(
        f'<div class="dm-kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )
