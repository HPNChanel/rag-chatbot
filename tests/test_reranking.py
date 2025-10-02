from data_ingestion.loader import Document
from reranking.coverage import CoverageAwareReranker


def test_reranker_balances_similarity_and_coverage() -> None:
    doc_short = Document(doc_id="1", content="short text", metadata={"document_id": "A"})
    doc_long = Document(doc_id="2", content="long text " * 10, metadata={"document_id": "B"})
    candidates = [(doc_short, 0.9), (doc_long, 0.4)]
    reranker = CoverageAwareReranker(similarity_weight=0.8)
    reranked = reranker.rerank(candidates, top_k=2)
    assert reranked[0][0].doc_id == "1"
    assert reranked[1][0].doc_id == "2"


def test_reranker_promotes_diverse_sources() -> None:
    doc_a1 = Document(doc_id="1", content="topic A first", metadata={"document_id": "A"})
    doc_a2 = Document(doc_id="2", content="topic A second", metadata={"document_id": "A"})
    doc_b = Document(doc_id="3", content="topic B", metadata={"document_id": "B"})
    candidates = [(doc_a1, 0.8), (doc_a2, 0.79), (doc_b, 0.6)]
    reranker = CoverageAwareReranker(similarity_weight=0.8, diversity_bias=0.3)
    reranked = reranker.rerank(candidates, top_k=3)
    assert reranked[0][0].doc_id == "1"
    # Ensure the second document provides coverage from a different source
    assert reranked[1][0].metadata["document_id"] == "B"
