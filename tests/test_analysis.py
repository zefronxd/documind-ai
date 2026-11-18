from src.document_comparator import compare_documents
from src.extractor import extract_information, info_to_rows
from src.pdf_processor import PageText, ProcessedDocument
from src.summarizer import summarize_document
from tests.fakes import FakeLLM


def make_doc(name, text):
    return ProcessedDocument(name, f"{name}.pdf", 1, 1, pages=[PageText(1, text)])


def test_summarizer_wraps_document_and_parses():
    llm = FakeLLM(['{"short_summary": "Short.", "important_dates": ["1 Jan 2024"]}'])
    summary, truncated = summarize_document(make_doc("a", "Report dated 1 Jan 2024."), llm)
    assert summary.short_summary == "Short." and summary.important_dates == ["1 Jan 2024"]
    assert '<document name="a.pdf">' in llm.prompts[0] and not truncated


def test_extractor_rows():
    llm = FakeLLM(['{"document_type": "invoice", "person_names": ["Asha"], "monetary_amounts": ["Rs. 500"]}'])
    info, _ = extract_information(make_doc("b", "Invoice for Asha Rs. 500"), llm)
    rows = info_to_rows(info)
    assert {"Category": "Person", "Value": "Asha"} in rows
    assert {"Category": "Document type", "Value": "invoice"} in rows


def test_comparison():
    llm = FakeLLM(['{"similarities": ["Both cover AI"], "contradictions": []}'])
    result, _ = compare_documents(make_doc("a", "AI text"), make_doc("b", "More AI text"), llm)
    assert result.similarities == ["Both cover AI"] and result.contradictions == []
