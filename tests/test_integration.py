from pathlib import Path

from data_ingestion.loader import Document, DocumentLoader
from evaluation import (
    build_human_eval_payload,
    corpus_bleu_score,
    evaluate_retrieval,
    meteor_score,
    rouge_l_score,
)
from generation.citation_pipeline import CitationFirstPipeline
from generation.dummy import EchoGenerator
from indexing.embedder import get_default_embedder, HashingEmbedder
from indexing.index import HybridIndex, VectorIndex
from pipelines.chatbot import RAGPipeline
from retrieval import BM25Retriever, DualStagePRFRetriever
from reranking.coverage import CoverageAwareReranker


class DeterministicGenerator:
    """Mock generator producing grounded answers for integration tests."""

    def generate(
        self,
        prompt: str,
        context: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return "Python emphasises readability. [doc_python]"


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


def test_query_to_generation_to_evaluation() -> None:
    documents = [
        Document(doc_id="doc_python", content="Python prioritises readability and batteries-included tooling", metadata={}),
        Document(doc_id="doc_rust", content="Rust ensures memory safety", metadata={}),
    ]

    retriever = BM25Retriever(documents)
    query = "Why do developers like Python?"
    query_id = "q_python"
    retrieved = retriever.search(query, k=2)
    retrieved_docs = [doc for doc, _ in retrieved]
    retrieved_ids = [doc.doc_id for doc in retrieved_docs]

    pipeline = CitationFirstPipeline(DeterministicGenerator())
    generation = pipeline.generate(query, retrieved_docs)

    assert generation.citations == ["doc_python"]

    relevance = {query_id: {"doc_python"}}
    retrieval_metrics = evaluate_retrieval(relevance, {query_id: retrieved_ids}, (1, 2))
    assert retrieval_metrics["precision"][1] == 1.0

    references = ["Developers appreciate Python for its readability and batteries included philosophy."]
    answers = [generation.answer]

    bleu = corpus_bleu_score(references, answers)
    rouge_l = rouge_l_score(references, answers)
    meteor = meteor_score(references, answers)

    assert bleu > 0
    assert rouge_l > 0
    assert meteor > 0

    human_template = build_human_eval_payload(query, references[0], generation.answer, generation.citations)
    serialised = human_template.to_json()
    assert "doc_python" in serialised
