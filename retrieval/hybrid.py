"""Hybrid retriever combining dense and lexical search."""
from __future__ import annotations

from typing import Iterable, List, Tuple

from data_ingestion.loader import Document
from indexing.vector_store import FaissVectorStore
from .bm25 import BM25Retriever


class HybridRetriever:
    """Combine BM25 and dense retrieval with configurable weights."""

    def __init__(
        self,
        documents: Iterable[Document],
        vector_store: FaissVectorStore,
        lexical_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> None:
        self.documents = list(documents)
        self.vector_store = vector_store
        self.lexical_retriever = BM25Retriever(self.documents)
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight

    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        lexical_results = self.lexical_retriever.search(query, k=k * 2)
        dense_results = self.vector_store.search(query, k=k * 2)
        combined_scores: dict[str, float] = {}

        for doc, score in lexical_results:
            combined_scores[doc.doc_id] = combined_scores.get(doc.doc_id, 0.0) + score * self.lexical_weight
        for doc, score in dense_results:
            combined_scores[doc.doc_id] = combined_scores.get(doc.doc_id, 0.0) + score * self.dense_weight

        ranked_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[:k]
        doc_lookup = {doc.doc_id: doc for doc in self.documents}
        return [(doc_lookup[doc_id], combined_scores[doc_id]) for doc_id in ranked_ids if doc_id in doc_lookup]
