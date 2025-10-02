"""Dual-stage pseudo relevance feedback retriever."""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from data_ingestion.loader import Document

from . import RetrieverProtocol

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}
_TOKEN_PATTERN = re.compile(r"\b\w+\b")


class DualStagePRFRetriever:
    """Augment a base retriever with pseudo relevance feedback."""

    def __init__(
        self,
        base_retriever: RetrieverProtocol,
        *,
        feedback_depth: int = 3,
        expansion_terms: int = 10,
        re_rank_strategy: str = "sum",
        feedback_weight: float = 0.6,
    ) -> None:
        if feedback_depth <= 0:
            raise ValueError("feedback_depth must be positive")
        if expansion_terms <= 0:
            raise ValueError("expansion_terms must be positive")
        if not 0.0 <= feedback_weight <= 1.0:
            raise ValueError("feedback_weight must be between 0 and 1")
        strategy = re_rank_strategy.lower()
        if strategy not in {"sum", "max", "second"}:
            raise ValueError("re_rank_strategy must be one of 'sum', 'max', or 'second'")
        self.base_retriever = base_retriever
        self.feedback_depth = feedback_depth
        self.expansion_terms = expansion_terms
        self.re_rank_strategy = strategy
        self.feedback_weight = feedback_weight

    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        if not query:
            return []
        first_stage = list(self.base_retriever.search(query, k=max(k, self.feedback_depth)))
        if not first_stage:
            return []
        feedback_docs = [doc for doc, _ in first_stage[: self.feedback_depth]]
        expansion = self._build_feedback_query(feedback_docs)
        if not expansion:
            return first_stage[:k]
        expanded_query = f"{query} {expansion}".strip()
        second_stage = list(self.base_retriever.search(expanded_query, k=k))
        combined = self._combine_results(first_stage, second_stage)
        combined.sort(key=lambda item: item[1], reverse=True)
        return combined[:k]

    def _build_feedback_query(self, documents: Sequence[Document]) -> str:
        tokens: List[str] = []
        for document in documents:
            tokens.extend(_TOKEN_PATTERN.findall(document.content.lower()))
        filtered = [token for token in tokens if len(token) > 2 and token not in _STOPWORDS]
        if not filtered:
            return ""
        most_common = [term for term, _ in Counter(filtered).most_common(self.expansion_terms)]
        return " ".join(dict.fromkeys(most_common))

    def _combine_results(
        self,
        first_stage: Sequence[Tuple[Document, float]],
        second_stage: Sequence[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        lookup: Dict[str, Document] = {}
        first_scores: Dict[str, float] = {}
        for doc, score in first_stage:
            lookup[doc.doc_id] = doc
            first_scores[doc.doc_id] = float(score)
        second_scores: Dict[str, float] = {}
        for doc, score in second_stage:
            lookup[doc.doc_id] = doc
            second_scores[doc.doc_id] = float(score)
        combined: List[Tuple[Document, float]] = []
        for doc_id, document in lookup.items():
            base_score = first_scores.get(doc_id)
            feedback_score = second_scores.get(doc_id)
            combined_score = self._combine_scores(base_score, feedback_score)
            combined.append((document, combined_score))
        return combined

    def _combine_scores(self, base: float | None, feedback: float | None) -> float:
        if self.re_rank_strategy == "second":
            return feedback if feedback is not None else (base or 0.0)
        if self.re_rank_strategy == "max":
            return max(base or 0.0, feedback or 0.0)
        base_score = base or 0.0
        feedback_score = feedback or 0.0
        return (1 - self.feedback_weight) * base_score + self.feedback_weight * feedback_score


__all__ = ["DualStagePRFRetriever"]
