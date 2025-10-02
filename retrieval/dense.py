"""Dense retriever backed by :class:`~indexing.vector_store.FaissVectorStore`."""
from __future__ import annotations

from typing import Iterable, List, Tuple

from data_ingestion.loader import Document
from indexing.vector_store import FaissVectorStore


class DenseRetriever:
    """Adapter exposing the vector store through the retriever protocol."""

    def __init__(self, vector_store: FaissVectorStore) -> None:
        self.vector_store = vector_store

    def build_if_required(self, documents: Iterable[Document]) -> None:
        """Ensure the underlying store is initialised with ``documents``."""

        if getattr(self.vector_store, "id_to_doc", None):  # store already built
            return
        self.vector_store.build(list(documents))

    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        return list(self.vector_store.search(query, k=k))


__all__ = ["DenseRetriever"]
