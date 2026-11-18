"""DocuMind AI - entry point. Handles navigation and the Dashboard page."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.config import APP_NAME, APP_TAGLINE, has_api_key
from utils.helpers import (
    apply_theme,
    get_documents,
    init_session_state,
    kpi_card,
    page_header,
)

st.set_page_config(page_title=APP_NAME, page_icon="🧠", layout="wide")

ROOT = Path(__file__).parent

# (file, sidebar title, icon, short description, needs LLM?)
FEATURE_PAGES = [
    ("pages/1_Document_Upload.py", "Document Upload", "📄", "Upload PDFs and extract text page by page (with OCR)", False),
    ("pages/2_AI_Summarization.py", "AI Summarization", "📝", "Short/detailed summary, key points, dates, entities", True),
    ("pages/3_Ask_Documents.py", "Ask Documents", "💬", "RAG question answering and chat with citations", True),
    ("pages/4_Information_Extraction.py", "Information Extraction", "🔎", "Structured JSON/CSV extraction", True),
    ("pages/5_Document_Comparison.py", "Document Comparison", "⚖️", "Compare two documents", True),
    ("pages/6_About_Project.py", "About Project", "ℹ️", "Project details and architecture", False),
]


def render_dashboard() -> None:
    page_header("Dashboard", APP_TAGLINE)
    docs = get_documents()

    cols = st.columns(4)
    with cols[0]:
        kpi_card("Documents uploaded", len(docs))
    with cols[1]:
        kpi_card("Pages processed", f"{sum(d.pages_processed for d in docs):,}")
    with cols[2]:
        kpi_card("Extracted characters", f"{sum(d.total_chars for d in docs):,}")
    with cols[3]:
        kpi_card("Questions asked", st.session_state["questions_asked"])

    st.write("")
    if has_api_key():
        st.success("Gemini API key detected. LLM-powered features are enabled.")
    else:
        st.warning(
            "GEMINI_API_KEY not found. PDF upload and text extraction work, but "
            "LLM features need a key. Copy `.env.example` to `.env`, add your key and restart the app."
        )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Available analysis features")
        rows = []
        for path, title, _icon, desc, needs_llm in FEATURE_PAGES:
            if not (ROOT / path).exists():
                status = "Not built yet"
            elif needs_llm and not has_api_key():
                status = "Needs API key"
            else:
                status = "Ready"
            rows.append({"Feature": title, "What it does": desc, "Status": status})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    with right:
        st.subheader("Pages per document")
        if docs:
            df = pd.DataFrame(
                {"Document": [d.name for d in docs], "Pages": [d.pages_processed for d in docs]}
            )
            fig = px.bar(df, x="Document", y="Pages", color_discrete_sequence=["#1E3A8A"])
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No documents yet. Go to **Document Upload** to add PDFs.")


init_session_state()
apply_theme()

pages = [st.Page(render_dashboard, title="Dashboard", icon="📊", url_path="dashboard", default=True)]
for path, title, icon, _desc, _llm in FEATURE_PAGES:
    if (ROOT / path).exists():
        pages.append(st.Page(path, title=title, icon=icon))

st.sidebar.markdown(f"### 🧠 {APP_NAME}")
st.navigation(pages).run()
