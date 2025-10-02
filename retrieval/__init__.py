"""Retrieval strategies used across the chatbot stack."""
from __future__ import annotations

from typing import Protocol, Sequence, Tuple

from data_ingestion.loader import Document


class RetrieverProtocol(Protocol):
    """Protocol for retrievers returning scored documents."""

    def search(self, query: str, k: int = 5) -> Sequence[Tuple[Document, float]]:
        """Return the ``k`` most relevant documents for ``query``."""


from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .prf import DualStagePRFRetriever

__all__ = [
    "RetrieverProtocol",
    "BM25Retriever",
    "DenseRetriever",
    "DualStagePRFRetriever",
]
