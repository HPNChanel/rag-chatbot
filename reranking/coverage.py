"""Coverage-aware reranking implementation."""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

from data_ingestion.loader import Document

CoverageFunction = Callable[[Document], float]


def default_coverage(doc: Document) -> float:
    """Default coverage score using document length as a proxy."""

    return float(len(doc.content.split()))


class CoverageReranker:
    """Rerank retrieved documents based on coverage signals."""

    def __init__(self, coverage_fn: CoverageFunction | None = None):
        self.coverage_fn = coverage_fn or default_coverage

    def rerank(self, candidates: Sequence[Tuple[Document, float]], top_k: int = 5) -> List[Tuple[Document, float]]:
        reranked = []
        for doc, score in candidates:
            coverage_score = self.coverage_fn(doc)
            reranked.append((doc, score + coverage_score))
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked[:top_k]
