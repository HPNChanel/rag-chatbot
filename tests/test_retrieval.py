from data_ingestion.loader import Document
from indexing.embedder import HashingEmbedder
from indexing.vector_store import FaissVectorStore
from retrieval.hybrid import HybridRetriever


def test_hybrid_retriever_returns_results() -> None:
    docs = [
        Document(doc_id="1", content="Python programming language"),
        Document(doc_id="2", content="Data science with statistics"),
    ]
    embedder = HashingEmbedder()
    store = FaissVectorStore(embedder=embedder, use_faiss=False)
    store.build(docs)
    retriever = HybridRetriever(docs, store)
    results = retriever.search("Python", k=2)
    assert results
    assert any(doc.doc_id == "1" for doc, _ in results)
