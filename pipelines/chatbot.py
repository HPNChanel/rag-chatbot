"""High-level orchestration for the RAG chatbot."""
from __future__ import annotations

from pathlib import Path
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

    def query(self, query: str, top_k: int = 5) -> List[Document]:
        """Return top ``top_k`` documents for the query."""
        candidates = self.retriever.search(query, k=top_k * 2)
        reranked = self.reranker.rerank(candidates, top_k=top_k)
        return [doc for doc, _ in reranked]

    def chat(self, query: str, top_k: int = 5) -> str:
        docs = self.query(query, top_k=top_k)
        context = [doc.content for doc in docs]
        return self.generator.generate(query, context=context)
