"""LLM-based comparison of two documents."""
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
from utils.config import MAX_LLM_CHARS

SYSTEM = UNTRUSTED_RULES + " You compare two documents objectively."


class ComparisonResult(LenientModel):
    summary_document_a: str = ""
    summary_document_b: str = ""
    similarities: list[str] = Field(default_factory=list)
    key_differences: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)


def compare_documents(doc_a, doc_b, llm: LLMProvider | None = None) -> tuple[ComparisonResult, bool]:
    llm = llm or get_llm()
    text_a, trunc_a = prepare_document_text(doc_a, MAX_LLM_CHARS // 2)
    text_b, trunc_b = prepare_document_text(doc_b, MAX_LLM_CHARS // 2)
    prompt = (
        "Compare Document A and Document B below. "
        f"Return ONLY a JSON object matching this schema:\n{schema_hint(ComparisonResult)}\n\n"
        "Rules:\n"
        "- Base everything strictly on the two documents. Do not use outside knowledge.\n"
        "- missing_information: important topics covered in one document but absent from the other.\n"
        "- contradictions: ONLY statements that directly conflict, quoting the specific claims from both. "
        "Return an empty list if there are none.\n"
        "- Use empty lists instead of guessing.\n\n"
        f"{wrap_document(doc_a.name, text_a)}\n\n{wrap_document(doc_b.name, text_b)}"
    )
    return generate_json(llm, prompt, ComparisonResult, system=SYSTEM), (trunc_a or trunc_b)
