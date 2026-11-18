"""Information Extraction page with JSON / CSV export."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.extractor import extract_information, info_to_rows
from src.llm_service import LLMError
from utils.helpers import init_session_state, page_header, require_documents

init_session_state()
page_header("Information Extraction", "Names, organizations, dates, amounts, keywords and topics as structured data.")
docs = require_documents()

names = {d.doc_id: d.name for d in docs}
doc_id = st.selectbox("Select a document", options=list(names), format_func=names.get)
doc = next(d for d in docs if d.doc_id == doc_id)

cache = st.session_state["extractions"]
if st.button("Extract information", type="primary"):
    with st.spinner("Extracting structured information..."):
        try:
            cache[doc_id] = extract_information(doc)
        except LLMError as exc:
            st.error(str(exc))

result = cache.get(doc_id)
if result:
    info, truncated = result
    if truncated:
        st.warning("The document was very long, so an equal share of each page was analyzed.")
    rows = info_to_rows(info)
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        df = pd.DataFrame(columns=["Category", "Value"])
        st.info("No structured information was found in this document.")

    payload = {"document": doc.name, **info.model_dump()}
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Download JSON",
            data=json.dumps(payload, indent=2, ensure_ascii=False),
            file_name=f"{doc.name.rsplit('.', 1)[0]}_extracted.json",
            mime="application/json",
        )
    with c2:
        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{doc.name.rsplit('.', 1)[0]}_extracted.csv",
            mime="text/csv",
        )
    with st.expander("View raw JSON"):
        st.json(payload)
else:
    st.info("Select a document and click **Extract information**.")
