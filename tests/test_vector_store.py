from src.chunking import Chunk
from src.vector_store import VectorStore
from tests.fakes import fake_embed


def make_chunks():
    texts = ["the capital of france is paris", "photosynthesis converts sunlight to energy", "cats and dogs are pets"]
    chunks = [Chunk(f"c{i}", "d1" if i < 2 else "d2", "x.pdf", i + 1, t) for i, t in enumerate(texts)]
    return chunks, fake_embed(texts)


def test_similarity_search_returns_best_chunk():
    chunks, vecs = make_chunks()
    store = VectorStore.build(chunks, vecs)
    hits = store.search(fake_embed(["what is the capital of france"])[0], k=2)
    assert hits[0][0].page_number == 1


def test_doc_filter():
    chunks, vecs = make_chunks()
    store = VectorStore.build(chunks, vecs)
    hits = store.search(fake_embed(["capital of france"])[0], k=3, doc_ids={"d2"})
    assert all(c.doc_id == "d2" for c, _ in hits)


def test_save_and_load(tmp_path):
    chunks, vecs = make_chunks()
    VectorStore.build(chunks, vecs).save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded.chunks) == 3
    assert loaded.search(fake_embed(["photosynthesis sunlight"])[0], k=1)[0][0].page_number == 2
