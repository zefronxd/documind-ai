"""About page: project details and architecture."""
from __future__ import annotations

import streamlit as st

from utils.config import APP_NAME, APP_TAGLINE, GEMINI_MODEL
from utils.helpers import init_session_state, page_header

init_session_state()
page_header("About Project", f"{APP_NAME}: {APP_TAGLINE}")

st.markdown(
    """
**DocuMind AI** is a B.Tech (AI & Data Science) project that lets users upload PDFs and then
summarize them, ask questions with source citations (RAG), extract structured data and compare documents.
"""
)

st.subheader("Architecture")
st.graphviz_chart(
    """
digraph G {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fillcolor="#E2E8F0", fontname="Helvetica"];
  "PDF upload" -> "PyMuPDF text extraction (+ OCR for scans)" -> "Page-wise chunking" -> "MiniLM embeddings" -> "FAISS index";
  "User question" -> "Query embedding";
  "Query embedding" -> "FAISS index" [label="top-k"];
  "FAISS index" -> "Gemini LLM" -> "Answer + citations";
}
"""
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Technology stack")
    st.markdown(
        f"""
- Streamlit (UI), PyMuPDF (PDF text), RapidOCR (scanned pages)
- Sentence-Transformers `all-MiniLM-L6-v2` (embeddings)
- FAISS CPU (vector search, cosine similarity)
- Google Gemini via `google-genai` (model: `{GEMINI_MODEL}`)
- Pydantic (validation), Pandas / Plotly (tables and charts)
"""
    )
with c2:
    st.subheader("Safety design")
    st.markdown(
        """
- Document text is treated as **untrusted data** (prompt-injection defence)
- Answers only from retrieved context, else "insufficient information"
- Citations are built from retrieved chunks, never typed by the model
- API key is read from `.env` and never displayed
"""
    )
st.subheader("Known limitations")
st.markdown("- OCR is slower and can make recognition errors  \n- Index is per session  \n- Output quality depends on the LLM")
