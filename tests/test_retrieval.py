import pytest

from data_ingestion.loader import Document
from indexing.embedder import HashingEmbedder
from indexing.vector_store import FaissVectorStore
from retrieval import BM25Retriever, DenseRetriever, DualStagePRFRetriever


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(doc_id="1", content="Space shuttles use rockets for orbital missions"),
        Document(doc_id="2", content="Mars missions require precise orbital mechanics"),
        Document(doc_id="3", content="Rocket propulsion relies on liquid fuel"),
    ]


def test_bm25_retriever_ranks_relevant_document(sample_documents: list[Document]) -> None:
    retriever = BM25Retriever(sample_documents)
    results = retriever.search("orbital missions", k=2)
    assert results
    assert results[0][0].doc_id in {"1", "2"}


def test_dense_retriever_returns_results(sample_documents: list[Document]) -> None:
    embedder = HashingEmbedder(dim=64)
    store = FaissVectorStore(embedder=embedder, use_faiss=False)
    store.build(sample_documents)
    retriever = DenseRetriever(store)
    results = retriever.search("rocket fuel", k=2)
    assert results
    assert any(doc.doc_id == "3" for doc, _ in results)


def test_dual_stage_prf_expands_query(sample_documents: list[Document]) -> None:
    base_retriever = BM25Retriever(sample_documents)
    prf_retriever = DualStagePRFRetriever(
        base_retriever,
        feedback_depth=1,
        expansion_terms=5,
        re_rank_strategy="sum",
        feedback_weight=0.7,
    )
    initial_results = base_retriever.search("space mission", k=2)
    assert all(doc.doc_id != "3" for doc, _ in initial_results)
    prf_results = prf_retriever.search("space mission", k=3)
    assert any(doc.doc_id == "3" for doc, _ in prf_results)
