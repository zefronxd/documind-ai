# DocuMind AI: Abstract, Architecture, Demo and Viva

## Project Abstract
DocuMind AI is an intelligent document analysis system built with Generative AI and Retrieval-Augmented
Generation (RAG). Reading long PDF documents such as research papers, reports and contracts is time consuming,
and general-purpose chatbots may produce answers that are not supported by the document. DocuMind AI addresses
this by grounding every answer in the user's own files. The application extracts text from uploaded PDFs page by
page using PyMuPDF, applies offline OCR to scanned pages, splits the text into overlapping chunks, converts the
chunks into vector embeddings using the Sentence-Transformers model all-MiniLM-L6-v2 and stores them in a FAISS
index. When a user asks a question, the most relevant chunks are retrieved by cosine similarity and passed to the
Google Gemini large language model, which generates a concise answer. The system displays the source document and
page number for each answer and explicitly states when the documents do not contain enough information. In
addition to question answering, the system provides AI-based summarization, structured information extraction
(names, organizations, dates, amounts, keywords) validated with Pydantic and exportable as JSON and CSV, and
comparison of two documents. Security measures include file validation, upload limits, treating document text as
untrusted data to resist prompt injection, and safe handling of API keys. The Streamlit interface offers a
dashboard with real session statistics. The project demonstrates practical use of embeddings, vector search and
LLMs in a modular, testable design.

## Architecture Explanation
1. **Ingestion layer** (`pdf_processor.py`, `ocr.py`): validates the file, opens it with PyMuPDF, reads text per
   page, detects scanned pages, runs offline OCR (RapidOCR) on them and rejects corrupt or password-protected files.
2. **Indexing layer** (`chunking.py`, `embeddings.py`, `vector_store.py`): chunks stay inside one page so page
   numbers remain exact; embeddings are normalised, so FAISS `IndexFlatIP` gives cosine similarity.
3. **Retrieval + generation layer** (`rag_pipeline.py`, `llm_service.py`): top-k retrieval with a relevance
   threshold, a JSON-structured LLM answer, and citations built from retrieved chunk metadata.
4. **Analysis layer** (`summarizer.py`, `extractor.py`, `document_comparator.py`): one prompt each, output
   validated by Pydantic models.
5. **Presentation layer** (`app.py`, `pages/`): Streamlit multipage UI; all numbers come from session state.
6. **Provider abstraction**: `LLMProvider` is an abstract class; `GeminiProvider` is one implementation, so
   another LLM can be added without touching the pipeline.

## Project Demonstration Steps
1. Start the app and open the **Dashboard** (all counters are zero, API key status is shown).
2. Go to **Document Upload**, upload 2 PDFs; show size, page count, preview. Try a `.txt` file renamed to `.pdf`
   to show the error message. Upload a scanned PDF with OCR enabled and show "source: OCR" in the preview.
3. Open **AI Summarization**, generate a summary and show dates and entities.
4. Open **Ask Documents**: ask a question whose answer is in the PDF, open the "Sources" expander and verify the
   page number in the PDF. Ask a follow-up question. Then ask an unrelated question to show the
   "insufficient information" reply.
5. Open **Information Extraction**, extract, show the table and download JSON and CSV.
6. Open **Document Comparison**, compare the two PDFs.
7. Return to the **Dashboard** to show updated statistics. Run `python -m pytest tests -v` to show tests.

## Viva Questions and Answers
1. **What is RAG?** Retrieval-Augmented Generation: relevant text is retrieved from a knowledge source and given
   to an LLM as context, so answers are grounded in real data instead of only the model's memory.
2. **Why RAG instead of sending the whole PDF to the LLM?** Long documents exceed context limits, cost more and
   dilute relevance. Retrieval sends only the relevant parts and enables page-level citations.
3. **What is an embedding?** A numeric vector representing the meaning of text; similar meanings give nearby vectors.
4. **Why all-MiniLM-L6-v2?** It is small (about 90 MB), fast on CPU, gives 384-dimensional vectors and has good
   semantic quality for its size.
5. **What is FAISS?** Facebook AI Similarity Search, a library for fast nearest-neighbour search over vectors.
6. **Why `IndexFlatIP` with normalised vectors?** Inner product of unit vectors equals cosine similarity; a flat
   index is exact and fast enough for document-sized collections.
7. **Why chunk with overlap?** Overlap keeps sentences that fall on a chunk boundary available in at least one chunk.
8. **Why chunk per page?** So each chunk has an exact page number for trustworthy citations.
9. **How do you prevent hallucination?** Answer only from retrieved chunks, relevance threshold, a "sufficient"
   flag from the LLM, low temperature and an explicit "insufficient information" reply.
10. **How are citations kept honest?** The LLM only returns chunk ids; the app maps ids to real retrieved
    chunks and ignores invalid ids, so the model cannot invent a page number.
11. **What is prompt injection and how do you defend?** Malicious text inside a document tries to give the LLM
    new instructions. We wrap document text in tags, escape the tags, and tell the model to treat it purely as data.
12. **Why Pydantic?** It validates the LLM's JSON, applies defaults and makes malformed output detectable.
13. **What if the LLM returns invalid JSON?** The text is cleaned, retried once with a repair instruction, and then
    a friendly error is shown.
14. **How do you handle scanned PDFs?** We detect pages with almost no text layer and run offline OCR (RapidOCR)
    on them: the page is rendered to an image with PyMuPDF and the recognised text replaces the empty page text,
    keeping the page number. OCR text can contain recognition errors, so the app shows a warning.
15. **What happens without an API key?** Upload and extraction still work; LLM features show a setup message
    and never return fake answers.
16. **Why a provider abstraction?** `LLMProvider` lets us swap Gemini for another model without changing the RAG code.
17. **What are the limitations?** OCR errors on poor scans, per-session index, dependence on LLM quality and API quota.
18. **How is the project tested?** pytest with a fake LLM, deterministic fake embeddings and a fake OCR engine for
    offline tests, plus manual end-to-end checks with the real models.
19. **How would you improve retrieval?** Hybrid BM25 + vector search, cross-encoder re-ranking and query rewriting.
20. **Why Streamlit?** Pure-Python UI with fast development, suitable for a 2-3 hour prototype.
