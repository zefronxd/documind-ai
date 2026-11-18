"""AI summarization of a document using the LLM."""
from __future__ import annotations

from pydantic import Field

from src.llm_service import (
    UNTRUSTED_RULES,
    LenientModel,
    LLMProvider,
    generate_json,
    get_llm,
    prepare_document_text,
    schema_hint,
    wrap_document,
)

SYSTEM = UNTRUSTED_RULES + " You are a precise document analyst."


class DocumentSummary(LenientModel):
    short_summary: str = ""
    detailed_summary: str = ""
    important_points: list[str] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    important_dates: list[str] = Field(default_factory=list)
    important_entities: list[str] = Field(default_factory=list)


def summarize_document(doc, llm: LLMProvider | None = None) -> tuple[DocumentSummary, bool]:
    """Return (summary, truncated). truncated=True if the text had to be shortened."""
    llm = llm or get_llm()
    text, truncated = prepare_document_text(doc)
    prompt = (
        "Analyze the document below and return ONLY a JSON object matching this schema:\n"
        f"{schema_hint(DocumentSummary)}\n\n"
        "Rules:\n"
        "- Use only information present in the document. Do not guess or add outside knowledge.\n"
        "- short_summary: 2-3 sentences. detailed_summary: 2-4 paragraphs.\n"
        "- important_points / key_findings: concise bullet strings (mention page numbers like (p. 3) when clear).\n"
        "- important_dates: dates exactly as written, with a few words of context.\n"
        "- important_entities: people, organizations, places, products named in the document.\n"
        "- If something is not in the document, return an empty list for it "
        "(or 'Not stated in the document' for text fields).\n\n"
        f"{wrap_document(doc.name, text)}"
    )
    return generate_json(llm, prompt, DocumentSummary, system=SYSTEM), truncated
