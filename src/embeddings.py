"""Sentence-Transformers embeddings (all-MiniLM-L6-v2). Model loads lazily, once."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Return L2-normalised float32 vectors, shape (len(texts), 384).

    Normalised vectors let FAISS inner product behave as cosine similarity.
    """
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    vectors = get_model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.ascontiguousarray(vectors, dtype="float32")
