from pathlib import Path

from data_ingestion.loader import Document, DocumentLoader
from generation.dummy import EchoGenerator
from indexing.embedder import get_default_embedder, HashingEmbedder
from indexing.index import HybridIndex, VectorIndex
from pipelines.chatbot import RAGPipeline
from retrieval import BM25Retriever, DualStagePRFRetriever
from reranking.coverage import CoverageAwareReranker


def test_end_to_end_query(tmp_path: Path) -> None:
    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    (data_dir / "doc1.txt").write_text("Python is great for scripting", encoding="utf-8")
    (data_dir / "doc2.txt").write_text("Rust is great for systems", encoding="utf-8")

    pipeline = RAGPipeline(
        data_path=data_dir,
        embedder=get_default_embedder(prefer_lightweight=True),
        generator=EchoGenerator(),
        use_faiss=False,
    )
    docs = pipeline.query("Python", top_k=1)
    assert docs
    response = pipeline.chat("Python", top_k=1)
    assert "Python" in response


def test_loader_and_index_integration(tmp_path: Path) -> None:
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    (data_dir / "doc1.txt").write_text("Python enables rapid application development", encoding="utf-8")
    (data_dir / "doc2.md").write_text("## Heading\n\nRust focuses on performance.", encoding="utf-8")

    loader = DocumentLoader(data_dir, chunk_size=20)
    documents = loader.load()
    assert documents

    index = HybridIndex(
        vector_index=VectorIndex(embedder=HashingEmbedder(dim=64), use_faiss=False),
        dense_weight=0.6,
        sparse_weight=0.4,
    )
    index.build(documents)

    results = index.query("rapid development", top_k=2)
    assert results
    assert any("python" in result.document.content.lower() for result in results)


def test_retrieval_reranking_pipeline() -> None:
    documents = [
        Document(doc_id="a::1", content="Python supports async programming", metadata={"document_id": "a"}),
        Document(doc_id="a::2", content="Asyncio provides event loops", metadata={"document_id": "a"}),
        Document(doc_id="b::1", content="Rust offers fearless concurrency", metadata={"document_id": "b"}),
    ]
    base_retriever = BM25Retriever(documents)
    prf = DualStagePRFRetriever(base_retriever, feedback_depth=1, expansion_terms=5, feedback_weight=0.5)
    candidates = prf.search("async concurrency", k=3)
    assert len(candidates) == 3
    reranker = CoverageAwareReranker(similarity_weight=0.6, diversity_bias=0.25)
    reranked = reranker.rerank(candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0][1] >= reranked[1][1]
    # Ensure reranker promotes documents from different parents when available
    assert reranked[0][0].metadata["document_id"] != reranked[1][0].metadata["document_id"]
