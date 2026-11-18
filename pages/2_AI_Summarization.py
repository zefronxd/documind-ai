"""AI Summarization page."""
from __future__ import annotations

import streamlit as st

from src.llm_service import LLMError
from src.summarizer import summarize_document
from utils.helpers import init_session_state, page_header, render_list, require_documents

init_session_state()
page_header("AI Summarization", "Short and detailed summaries, key points, findings, dates and entities.")
docs = require_documents()

names = {d.doc_id: d.name for d in docs}
doc_id = st.selectbox("Select a document", options=list(names), format_func=names.get)
doc = next(d for d in docs if d.doc_id == doc_id)

cache = st.session_state["summaries"]
if st.button("Generate summary", type="primary"):
    with st.spinner("The AI model is reading your document..."):
        try:
            cache[doc_id] = summarize_document(doc)
        except LLMError as exc:
            st.error(str(exc))

result = cache.get(doc_id)
if result:
    summary, truncated = result
    if truncated:
        st.warning("The document was very long, so an equal share of each page was sent to the model.")
    tab1, tab2, tab3 = st.tabs(["Summary", "Points & findings", "Dates & entities"])
    with tab1:
        st.subheader("Short summary")
        st.write(summary.short_summary or "Not available.")
        st.subheader("Detailed summary")
        st.write(summary.detailed_summary or "Not available.")
    with tab2:
        st.subheader("Important points")
        render_list(summary.important_points)
        st.subheader("Key findings")
        render_list(summary.key_findings)
    with tab3:
        st.subheader("Important dates")
        render_list(summary.important_dates)
        st.subheader("Important entities")
        render_list(summary.important_entities)
else:
    st.info("Select a document and click **Generate summary**.")
