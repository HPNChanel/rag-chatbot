"""Reranking components used for ordering retrieved documents."""
from __future__ import annotations

from typing import Protocol, Sequence, Tuple

from data_ingestion.loader import Document


class RerankerProtocol(Protocol):
    """Protocol describing a reranker implementation."""

    def rerank(self, candidates: Sequence[Tuple[Document, float]], top_k: int = 5):
        """Return a reranked subset of ``candidates``."""


from .coverage import CoverageAwareReranker

__all__ = ["RerankerProtocol", "CoverageAwareReranker"]
