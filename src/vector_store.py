"""FAISS vector store (cosine similarity) with local save/load."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.chunking import Chunk


class VectorStore:
    def __init__(self, index, chunks: list[Chunk]):
        self.index = index
        self.chunks = chunks

    @classmethod
    def build(cls, chunks: list[Chunk], vectors: np.ndarray) -> "VectorStore":
        import faiss

        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if len(chunks) == 0:
            raise ValueError("Cannot build an index with no chunks")
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        index = faiss.IndexFlatIP(vectors.shape[1])  # inner product == cosine for normalised vectors
        index.add(vectors)
        return cls(index, list(chunks))

    def search(self, query_vec: np.ndarray, k: int = 5, doc_ids: set[str] | None = None):
        """Return up to k (Chunk, score) pairs, best first. Optionally restrict to doc_ids."""
        total = self.index.ntotal
        if total == 0 or k <= 0:
            return []
        q = np.ascontiguousarray(np.asarray(query_vec).reshape(1, -1), dtype="float32")
        fetch = total if doc_ids else min(k, total)
        scores, idxs = self.index.search(q, fetch)
        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            if doc_ids and chunk.doc_id not in doc_ids:
                continue
            results.append((chunk, float(score)))
            if len(results) >= k:
                break
        return results

    def save(self, directory: Path) -> None:
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        (directory / "chunks.json").write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        import faiss

        directory = Path(directory)
        index = faiss.read_index(str(directory / "index.faiss"))
        raw = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        return cls(index, [Chunk(**item) for item in raw])
