"""Ask Documents: RAG chat with conversation history and citations."""
from __future__ import annotations

import streamlit as st

from src.llm_service import LLMError
from utils.config import TOP_K
from utils.helpers import ensure_index, init_session_state, page_header, require_documents

init_session_state()
page_header("Ask Documents", "Chat with your PDFs. Answers come only from the uploaded documents, with sources.")
docs = require_documents()

with st.expander("Retrieval settings"):
    size = st.slider("Chunk size (characters)", 300, 2000, st.session_state["chunk_size"], step=100, key="w_chunk_size")
    max_overlap = min(500, size // 2)
    overlap = st.slider(
        "Chunk overlap (characters)", 0, max_overlap, min(st.session_state["chunk_overlap"], max_overlap),
        step=50, key="w_chunk_overlap",
    )
    top_k = st.slider("Chunks retrieved per question", 1, 10, TOP_K, key="w_top_k")
    st.session_state["chunk_size"] = size
    st.session_state["chunk_overlap"] = overlap

names = {d.doc_id: d.name for d in docs}
selected = st.multiselect("Search in", options=list(names), default=list(names), format_func=names.get)
if not selected:
    st.warning("Select at least one document.")
    st.stop()

try:
    pipeline = ensure_index()
except Exception as exc:
    st.error(
        f"Could not build the vector index ({type(exc).__name__}). "
        "On first run the embedding model must be downloaded, so check your internet connection."
    )
    st.stop()

if st.button("🗑️ Clear chat"):
    st.session_state["chat_history"] = []
    st.rerun()

for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        citations = message.get("citations") or []
        if citations:
            with st.expander(f"Sources ({len(citations)})"):
                for c in citations:
                    st.markdown(f"**{c['doc_name']}** · page {c['page_number']} · relevance {c['score']:.2f}")
                    st.text(c["snippet"])

question = st.chat_input("Ask a question about your documents...")
if question:
    history = [(m["role"], m["content"]) for m in st.session_state["chat_history"]]
    with st.spinner("Searching documents and generating answer..."):
        try:
            result = pipeline.answer(question, history=history, k=top_k, doc_ids=set(selected))
        except LLMError as exc:
            st.error(str(exc))
            st.stop()
    st.session_state["chat_history"].append({"role": "user", "content": question, "citations": []})
    st.session_state["chat_history"].append(
        {"role": "assistant", "content": result.answer, "citations": [c.to_dict() for c in result.citations]}
    )
    st.session_state["questions_asked"] += 1
    st.rerun()
