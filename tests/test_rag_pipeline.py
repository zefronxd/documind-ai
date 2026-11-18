from src.pdf_processor import PageText, ProcessedDocument
from src.rag_pipeline import INSUFFICIENT, RAGPipeline
from tests.fakes import FakeLLM, fake_embed

DOC = ProcessedDocument(
    "d1", "geo.pdf", 1, 2,
    pages=[PageText(1, "The capital of France is Paris."), PageText(2, "Photosynthesis converts sunlight into energy.")],
)


def make(replies):
    pipe = RAGPipeline(embed_fn=fake_embed, llm_factory=lambda: FakeLLM(replies))
    pipe.build_index([DOC], 500, 50)
    return pipe


def test_grounded_answer_has_real_citation():
    pipe = make(['{"answer": "Paris", "sufficient": true, "sources": [1]}'])
    result = pipe.answer("What is the capital of France?")
    assert result.grounded and result.answer == "Paris"
    assert result.citations[0].page_number == 1 and result.citations[0].doc_name == "geo.pdf"


def test_unrelated_question_skips_llm():
    result = make([]).answer("quantum entanglement zebra")  # empty reply list: LLM must not be called
    assert result.answer == INSUFFICIENT and result.citations == []


def test_model_says_insufficient():
    result = make(['{"answer": "", "sufficient": false, "sources": []}']).answer("capital of France?")
    assert result.answer == INSUFFICIENT and not result.grounded


def test_invented_source_ids_are_rejected():
    result = make(['{"answer": "Paris", "sufficient": true, "sources": [99]}']).answer("capital of France?")
    assert result.answer == INSUFFICIENT and result.citations == []
