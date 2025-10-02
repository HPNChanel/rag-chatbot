"""Coverage-aware reranking implementation."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Sequence, Tuple

from data_ingestion.loader import Document

CoverageFunction = Callable[[Document], float]


def default_coverage(doc: Document) -> float:
    """Default coverage score using document length as a proxy."""

    return float(len(doc.content.split()))


class CoverageAwareReranker:
    """Combine semantic similarity with coverage and diversity signals."""

    def __init__(
        self,
        coverage_fn: CoverageFunction | None = None,
        *,
        similarity_weight: float = 0.7,
        diversity_bias: float = 0.15,
    ) -> None:
        if not 0.0 <= similarity_weight <= 1.0:
            raise ValueError("similarity_weight must be between 0 and 1")
        if diversity_bias < 0.0:
            raise ValueError("diversity_bias must be non-negative")
        self.coverage_fn = coverage_fn or default_coverage
        self.similarity_weight = similarity_weight
        self.diversity_bias = diversity_bias

    def rerank(self, candidates: Sequence[Tuple[Document, float]], top_k: int = 5) -> List[Tuple[Document, float]]:
        if top_k <= 0:
            return []
        if not candidates:
            return []
        coverage_scores: Dict[str, float] = {}
        for doc, _ in candidates:
            coverage_scores[doc.doc_id] = max(self.coverage_fn(doc), 0.0)
        max_coverage = max(coverage_scores.values(), default=1.0) or 1.0
        base_scores: List[Tuple[Document, float]] = []
        for doc, similarity in candidates:
            coverage_norm = coverage_scores[doc.doc_id] / max_coverage
            score = self.similarity_weight * float(similarity) + (1 - self.similarity_weight) * coverage_norm
            base_scores.append((doc, score))
        selected: List[Tuple[Document, float]] = []
        usage_counts: Dict[str, int] = defaultdict(int)
        remaining = base_scores.copy()
        while remaining and len(selected) < top_k:
            best_index = -1
            best_adjusted = float("-inf")
            for idx, (doc, score) in enumerate(remaining):
                group_id = doc.metadata.get("document_id", doc.metadata.get("source", doc.doc_id))
                penalty = self.diversity_bias * usage_counts[group_id]
                bonus = self.diversity_bias if usage_counts[group_id] == 0 else 0.0
                adjusted_score = score + bonus - penalty
                if adjusted_score > best_adjusted:
                    best_adjusted = adjusted_score
                    best_index = idx
            if best_index == -1:
                break
            doc, _ = remaining.pop(best_index)
            group_id = doc.metadata.get("document_id", doc.metadata.get("source", doc.doc_id))
            usage_counts[group_id] += 1
            selected.append((doc, best_adjusted))
        return selected


class CoverageReranker(CoverageAwareReranker):
    """Backward compatible alias for the previous class name."""


__all__ = ["CoverageAwareReranker", "CoverageReranker", "default_coverage"]
