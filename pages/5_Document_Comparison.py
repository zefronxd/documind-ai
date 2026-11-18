"""Document Comparison page."""
from __future__ import annotations

import streamlit as st

from src.document_comparator import compare_documents
from src.llm_service import LLMError
from utils.helpers import init_session_state, page_header, render_list, require_documents

init_session_state()
page_header("Document Comparison", "Compare two uploaded documents: similarities, differences, gaps and contradictions.")
docs = require_documents()

if len(docs) < 2:
    st.info("Upload at least two PDFs with extractable text to use this feature.")
    st.stop()

names = {d.doc_id: d.name for d in docs}
col1, col2 = st.columns(2)
with col1:
    id_a = st.selectbox("Document A", options=list(names), format_func=names.get, key="cmp_a")
with col2:
    id_b = st.selectbox(
        "Document B", options=[i for i in names if i != id_a], format_func=names.get, key="cmp_b"
    )

doc_a = next(d for d in docs if d.doc_id == id_a)
doc_b = next(d for d in docs if d.doc_id == id_b)
cache_key = f"{id_a}:{id_b}"

if st.button("Compare documents", type="primary"):
    with st.spinner("Comparing documents..."):
        try:
            st.session_state["comparisons"][cache_key] = compare_documents(doc_a, doc_b)
        except LLMError as exc:
            st.error(str(exc))

result = st.session_state["comparisons"].get(cache_key)
if result:
    comparison, truncated = result
    if truncated:
        st.warning("One or both documents were very long, so an equal share of each page was analyzed.")
    s1, s2 = st.columns(2)
    with s1:
        st.subheader(f"Summary: {doc_a.name}")
        st.write(comparison.summary_document_a or "Not available.")
    with s2:
        st.subheader(f"Summary: {doc_b.name}")
        st.write(comparison.summary_document_b or "Not available.")
    st.subheader("Similarities")
    render_list(comparison.similarities)
    st.subheader("Key differences")
    render_list(comparison.key_differences)
    st.subheader("Important missing information")
    render_list(comparison.missing_information)
    st.subheader("Contradictory statements")
    if comparison.contradictions:
        render_list(comparison.contradictions)
    else:
        st.caption("No direct contradictions were found in the text.")
else:
    st.info("Choose two documents and click **Compare documents**.")
