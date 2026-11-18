"""Structured information extraction with Pydantic validation."""
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

SYSTEM = UNTRUSTED_RULES + " You extract structured data from documents."


class ExtractedInfo(LenientModel):
    document_type: str = "Unknown"
    main_topics: list[str] = Field(default_factory=list)
    person_names: list[str] = Field(default_factory=list)
    organization_names: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    monetary_amounts: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


_CATEGORIES = [
    ("Document type", "document_type"),
    ("Main topic", "main_topics"),
    ("Person", "person_names"),
    ("Organization", "organization_names"),
    ("Date", "dates"),
    ("Monetary amount", "monetary_amounts"),
    ("Keyword", "keywords"),
]


def extract_information(doc, llm: LLMProvider | None = None) -> tuple[ExtractedInfo, bool]:
    llm = llm or get_llm()
    text, truncated = prepare_document_text(doc)
    prompt = (
        "Extract structured information from the document below. "
        f"Return ONLY a JSON object matching this schema:\n{schema_hint(ExtractedInfo)}\n\n"
        "Rules:\n"
        "- Include only items that literally appear in the document. Never invent values.\n"
        "- Copy dates and monetary amounts exactly as written (with currency symbol).\n"
        "- document_type: e.g. research paper, invoice, resume, contract, report, article, other.\n"
        "- Use empty lists when nothing is found. Remove duplicates.\n\n"
        f"{wrap_document(doc.name, text)}"
    )
    return generate_json(llm, prompt, ExtractedInfo, system=SYSTEM), truncated


def info_to_rows(info: ExtractedInfo) -> list[dict]:
    """Flatten to [{'Category', 'Value'}] rows for the table and CSV export."""
    rows = []
    for label, field_name in _CATEGORIES:
        value = getattr(info, field_name)
        values = [value] if isinstance(value, str) else value
        for item in values:
            if str(item).strip():
                rows.append({"Category": label, "Value": str(item).strip()})
    return rows
