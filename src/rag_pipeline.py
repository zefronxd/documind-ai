"""RAG pipeline: chunk -> embed -> FAISS -> retrieve -> grounded answer with real citations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from pydantic import Field

from src.chunking import chunk_document
from src.llm_service import (
    UNTRUSTED_RULES,
    LenientModel,
    LLMProvider,
    generate_json,
    get_llm,
    neutralize,
    schema_hint,
)
from src.vector_store import VectorStore
from utils.config import MIN_RELEVANCE, TOP_K

INSUFFICIENT = "The uploaded documents do not provide sufficient information to answer this question."

RAG_SYSTEM = UNTRUSTED_RULES + (
    " You answer questions using ONLY the numbered excerpts provided. "
    "If the excerpts do not contain the answer, say so. Never use outside knowledge."
)


class RAGResponse(LenientModel):
    answer: str = ""
    sufficient: bool = False
    sources: list[int] = Field(default_factory=list)


@dataclass
class Citation:
    doc_name: str
    page_number: int
    score: float
    snippet: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RAGAnswer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = False


class RAGPipeline:
    def __init__(
        self,
        embed_fn: Callable[[list[str]], np.ndarray] | None = None,
        llm_factory: Callable[[], LLMProvider] = get_llm,
        session_dir: Path | None = None,
    ):
        self._embed_fn = embed_fn
        self._llm_factory = llm_factory
        self.session_dir = session_dir
        self.store: VectorStore | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._embed_fn is None:
            from src.embeddings import embed_texts

            self._embed_fn = embed_texts
        return self._embed_fn(texts)

    def build_index(self, docs: list, chunk_size: int, overlap: int) -> int:
        chunks = []
        for doc in docs:
            chunks.extend(chunk_document(doc, chunk_size, overlap))
        if not chunks:
            self.store = None
            return 0
        vectors = self._embed([c.text for c in chunks])
        self.store = VectorStore.build(chunks, vectors)
        if self.session_dir:
            self.store.save(self.session_dir)  # persisted locally for this session
        return len(chunks)

    def retrieve(self, query: str, k: int = TOP_K, doc_ids: set[str] | None = None):
        if self.store is None:
            return []
        query_vec = self._embed([query])[0]
        return self.store.search(query_vec, k, doc_ids)

    def answer(
        self,
        question: str,
        history: list[tuple[str, str]] | None = None,
        k: int = TOP_K,
        doc_ids: set[str] | None = None,
        min_score: float = MIN_RELEVANCE,
    ) -> RAGAnswer:
        history = history or []
        query = question
        if history and len(question.split()) < 8:  # short follow-up: add the previous question
            previous = [text for role, text in history if role == "user"]
            if previous:
                query = f"{previous[-1]} {question}"

        hits = [(c, s) for c, s in self.retrieve(query, k, doc_ids) if s >= min_score]
        if not hits:
            return RAGAnswer(INSUFFICIENT, [], False)  # no LLM call for unrelated questions

        excerpts = "\n\n".join(
            f'<chunk id="{i}" source="{c.doc_name}" page="{c.page_number}">\n{neutralize(c.text)}\n</chunk>'
            for i, (c, _) in enumerate(hits, start=1)
        )
        chat = "\n".join(f"{role}: {text[:500]}" for role, text in history[-6:]) or "(none)"
        prompt = (
            f"Earlier conversation (context only, not a source of facts):\n{chat}\n\n"
            f"Excerpts:\n{excerpts}\n\n"
            f"Question: {question}\n\n"
            f"Return ONLY JSON matching this schema: {schema_hint(RAGResponse)}\n"
            '- "answer": concise answer grounded only in the excerpts, in the language of the question.\n'
            '- "sufficient": true only if the excerpts really contain the answer; otherwise false and answer "".\n'
            '- "sources": list of the integer chunk ids you actually used.'
        )

        response = generate_json(self._llm_factory(), prompt, RAGResponse, system=RAG_SYSTEM)

        used: list[int] = []
        for source in response.sources:
            if 1 <= source <= len(hits) and source not in used:  # ignore invented ids
                used.append(source)
        if not response.sufficient or not response.answer.strip() or not used:
            return RAGAnswer(INSUFFICIENT, [], False)

        citations: list[Citation] = []
        seen: set[tuple[str, int]] = set()
        for source in used:
            chunk, score = hits[source - 1]
            key = (chunk.doc_id, chunk.page_number)
            if key in seen:
                continue
            seen.add(key)
            citations.append(Citation(chunk.doc_name, chunk.page_number, round(score, 3), chunk.text[:300]))
        return RAGAnswer(response.answer.strip(), citations, True)
