from data_ingestion.loader import Document
from indexing.embedder import HashingEmbedder
from indexing.vector_store import FaissVectorStore


def test_faiss_vector_store_build_and_search() -> None:
    docs = [
        Document(doc_id="1", content="Python programming"),
        Document(doc_id="2", content="Machine learning"),
    ]
    store = FaissVectorStore(embedder=HashingEmbedder(), use_faiss=False)
    store.build(docs)
    results = store.search("Python", k=1)
    assert results
    assert results[0][0].doc_id == "1"
