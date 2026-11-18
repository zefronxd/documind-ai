"""Page-aware text chunking with configurable size and overlap."""
from __future__ import annotations

from dataclasses import dataclass

from utils.config import CHUNK_OVERLAP, CHUNK_SIZE


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: int  # 1-based, copied from the source page (used for citations)
    text: str


_BREAKS = ("\n\n", "\n", ". ", "? ", "! ", " ")


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph/sentence/word boundaries."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window_start = start + max(overlap + 1, chunk_size // 2)
            for sep in _BREAKS:
                pos = text.rfind(sep, window_start, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        next_start = end - overlap
        start = next_start if next_start > start else end
    return chunks


def chunk_document(doc, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Chunk each page separately so every chunk keeps an exact page number."""
    chunks: list[Chunk] = []
    for page in doc.pages:
        for i, piece in enumerate(split_text(page.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}-p{page.page_number}-c{i}",
                    doc_id=doc.doc_id,
                    doc_name=doc.name,
                    page_number=page.page_number,
                    text=piece,
                )
            )
    return chunks
