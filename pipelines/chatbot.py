"""High-level orchestration for the RAG chatbot."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable, List, Protocol

from data_ingestion.loader import Document, DocumentLoader
from generation.openai_generator import OpenAIGenerator
from indexing.embedder import BaseEmbedder, get_default_embedder
from indexing.vector_store import FaissVectorStore, is_faiss_available
from reranking.coverage import CoverageReranker
from retrieval.hybrid import HybridRetriever


class SupportsGenerate(Protocol):  # pragma: no cover - structural typing helper
    def generate(self, prompt: str, context: Iterable[str] | None = None, max_tokens: int = 256) -> str:
        ...


@dataclass(slots=True)
class RetrievalResult:
    """Container describing a retrieved document and its scoring signals."""

    document: Document
    candidate_score: float
    reranked_score: float
    candidate_rank: int
    reranked_rank: int


@dataclass(slots=True)
class RetrievalResponse:
    """Detailed payload returned by :meth:`RAGPipeline.query_with_details`."""

    query: str
    results: List[RetrievalResult]
    metrics: dict[str, float]


@dataclass(slots=True)
class ChatResponse:
    """Full payload returned by :meth:`RAGPipeline.chat_with_details`."""

    query: str
    answer: str
    citations: List[dict[str, str]]
    retrieval: RetrievalResponse
    metrics: dict[str, float]


class RAGPipeline:
    """End-to-end pipeline for retrieval augmented generation."""

    def __init__(
        self,
        data_path: Path | str,
        embedder: BaseEmbedder | None = None,
        generator: SupportsGenerate | None = None,
        use_faiss: bool | None = None,
    ) -> None:
        loader = DocumentLoader(data_path)
        self.documents = loader.load()
        if not self.documents:
            raise ValueError(f"No documents found under {data_path}")
        if embedder is None:
            prefer_lightweight = use_faiss is False or (use_faiss is None and not is_faiss_available())
            embedder = get_default_embedder(prefer_lightweight=prefer_lightweight)
        self.embedder = embedder
        self.vector_store = FaissVectorStore(self.embedder, use_faiss=use_faiss)
        self.vector_store.build(self.documents)
        self.retriever = HybridRetriever(self.documents, self.vector_store)
        self.reranker = CoverageReranker()
        self.generator = generator or OpenAIGenerator()

    def query_with_details(self, query: str, top_k: int = 5) -> RetrievalResponse:
        """Return a rich retrieval response including scoring diagnostics."""

        start = time.perf_counter()
        candidates = self.retriever.search(query, k=top_k * 2)
        retrieval_latency = time.perf_counter() - start

        rerank_start = time.perf_counter()
        reranked = self.reranker.rerank(candidates, top_k=top_k)
        rerank_latency = time.perf_counter() - rerank_start

        candidate_ranks = {doc.doc_id: rank for rank, (doc, _) in enumerate(candidates, start=1)}
        candidate_scores = {doc.doc_id: float(score) for doc, score in candidates}

        results: List[RetrievalResult] = []
        for rerank_pos, (doc, rerank_score) in enumerate(reranked, start=1):
            results.append(
                RetrievalResult(
                    document=doc,
                    candidate_score=candidate_scores.get(doc.doc_id, 0.0),
                    reranked_score=float(rerank_score),
                    candidate_rank=candidate_ranks.get(doc.doc_id, rerank_pos),
                    reranked_rank=rerank_pos,
                )
            )

        reranker_impact = 0.0
        if results:
            reranker_impact = sum(
                result.candidate_rank - result.reranked_rank for result in results
            ) / len(results)

        metrics = {
            "retrieval_time": retrieval_latency,
            "rerank_time": rerank_latency,
            "total_time": retrieval_latency + rerank_latency,
            "candidate_count": float(len(candidates)),
            "reranker_impact": reranker_impact,
        }

        return RetrievalResponse(query=query, results=results, metrics=metrics)

    def chat_with_details(self, query: str, top_k: int = 5) -> ChatResponse:
        """Generate an answer with citations and diagnostic metadata."""

        retrieval = self.query_with_details(query, top_k=top_k)
        generation_start = time.perf_counter()
        context = [result.document.content for result in retrieval.results]
        answer = self.generator.generate(query, context=context)
        generation_latency = time.perf_counter() - generation_start

        citations: List[dict[str, str]] = []
        for result in retrieval.results:
            metadata = result.document.metadata
            citations.append(
                {
                    "doc_id": result.document.doc_id,
                    "title": metadata.get("title", result.document.doc_id),
                    "source": metadata.get("source", ""),
                    "excerpt": result.document.content[:280],
                    "rank": str(result.reranked_rank),
                }
            )

        metrics = {
            **retrieval.metrics,
            "generation_time": generation_latency,
            "end_to_end_time": retrieval.metrics["total_time"] + generation_latency,
        }

        return ChatResponse(
            query=query,
            answer=answer,
            citations=citations,
            retrieval=retrieval,
            metrics=metrics,
        )

    def query(self, query: str, top_k: int = 5) -> List[Document]:
        """Backward compatible helper returning only the documents."""

        detailed = self.query_with_details(query, top_k=top_k)
        return [result.document for result in detailed.results]

    def chat(self, query: str, top_k: int = 5) -> str:
        detailed = self.chat_with_details(query, top_k=top_k)
        return detailed.answer
