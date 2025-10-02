from data_ingestion.loader import Document
from reranking.coverage import CoverageReranker


def test_reranker_orders_by_coverage() -> None:
    doc_short = Document(doc_id="1", content="short text")
    doc_long = Document(doc_id="2", content="long text " * 10)
    candidates = [(doc_short, 0.5), (doc_long, 0.4)]
    reranker = CoverageReranker()
    reranked = reranker.rerank(candidates, top_k=2)
    assert reranked[0][0].doc_id == "2"
