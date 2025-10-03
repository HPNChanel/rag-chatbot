"""BM25-based lexical retriever."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Tuple

try:  # pragma: no cover - optional dependency
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None  # type: ignore

from data_ingestion.loader import Document


class BM25Retriever:
    """Simple BM25 retriever over an in-memory corpus."""

    def __init__(self, documents: Iterable[Document]):
        self.documents: List[Document] = list(documents)
        self._tokenized = [doc.content.lower().split() for doc in self.documents]
        self.model = BM25Okapi(self._tokenized) if BM25Okapi is not None else None

    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        if self.model is not None:
            scores_array = self.model.get_scores(query.lower().split())
            scored = [
                (idx, float(score)) for idx, score in enumerate(scores_array)
            ]
            scored.sort(key=lambda item: (item[1], -item[0]), reverse=True)
            return [
                (self.documents[idx], score)
                for idx, score in scored[:k]
            ]
        return self._fallback_search(query, k)

    def _fallback_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        query_terms = query.lower().split()
        query_counts = Counter(query_terms)
        scored: List[Tuple[int, float]] = []
        for idx, tokens in enumerate(self._tokenized):
            doc_counts = Counter(tokens)
            score = sum(query_counts[term] * doc_counts.get(term, 0) for term in query_counts)
            scored.append((idx, float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(self.documents[idx], score) for idx, score in scored[:k]]
