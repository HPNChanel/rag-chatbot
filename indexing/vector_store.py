"""Vector store utilities backed by FAISS with lightweight fallbacks."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import faiss
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from data_ingestion.loader import Document
from .embedder import BaseEmbedder


def is_faiss_available() -> bool:
    """Return ``True`` when FAISS and numpy are importable."""

    return faiss is not None and np is not None


class FaissVectorStore:
    """A vector store that prefers FAISS but degrades gracefully."""

    def __init__(self, embedder: BaseEmbedder, use_faiss: bool | None = None) -> None:
        self.embedder = embedder
        if use_faiss is None:
            self.uses_faiss = is_faiss_available()
        else:
            if use_faiss and not is_faiss_available():
                raise ImportError("FAISS and numpy must be installed when use_faiss=True")
            self.uses_faiss = use_faiss
        self.index = None
        self.id_to_doc: Dict[int, Document] = {}
        self._normalized_embeddings: List[List[float]] | None = None

    def build(self, documents: Sequence[Document]) -> None:
        """Build the backing index for the provided documents."""

        docs = list(documents)
        if not docs:
            raise ValueError("No documents provided for indexing")
        embeddings = _ensure_2d_list(self.embedder.embed(doc.content for doc in docs))
        if self.uses_faiss:
            matrix = _to_numpy(embeddings)
            dimension = matrix.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # type: ignore[arg-type]
            faiss.normalize_L2(matrix)
            self.index.add(matrix)
        else:
            self._normalized_embeddings = _normalize_vectors(embeddings)
        self.id_to_doc = {idx: doc for idx, doc in enumerate(docs)}

    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Return the top ``k`` documents for ``query``."""

        if not self.id_to_doc:
            raise RuntimeError("Index has not been built")
        query_embeddings = _ensure_2d_list(self.embedder.embed([query]))
        if not query_embeddings:
            return []
        if self.uses_faiss:
            if self.index is None:
                raise RuntimeError("Index has not been built")
            query_matrix = _to_numpy(query_embeddings)
            faiss.normalize_L2(query_matrix)
            scores, indices = self.index.search(query_matrix, k)
            results: List[Tuple[Document, float]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                doc = self.id_to_doc.get(int(idx))
                if doc is not None:
                    results.append((doc, float(score)))
            return results
        if self._normalized_embeddings is None:
            raise RuntimeError("Index has not been built")
        normalized_query = _normalize_vectors(query_embeddings)[0]
        scored: List[Tuple[Document, float]] = []
        for idx, doc in self.id_to_doc.items():
            score = _dot(self._normalized_embeddings[idx], normalized_query)
            scored.append((doc, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Incrementally add documents to an existing index."""

        docs = list(documents)
        if not docs:
            return
        if not self.id_to_doc:
            self.build(docs)
            return
        embeddings = _ensure_2d_list(self.embedder.embed(doc.content for doc in docs))
        if self.uses_faiss:
            if self.index is None:
                raise RuntimeError("Index has not been built")
            matrix = _to_numpy(embeddings)
            faiss.normalize_L2(matrix)
            self.index.add(matrix)
        else:
            normalized = _normalize_vectors(embeddings)
            if self._normalized_embeddings is None:
                self._normalized_embeddings = normalized
            else:
                if normalized and len(normalized[0]) != len(self._normalized_embeddings[0]):
                    raise ValueError("Embedding dimension mismatch")
                self._normalized_embeddings.extend(normalized)
        start_idx = len(self.id_to_doc)
        for offset, doc in enumerate(docs):
            self.id_to_doc[start_idx + offset] = doc


def _ensure_2d_list(embeddings: Iterable[Iterable[float]]) -> List[List[float]]:
    rows = list(embeddings)
    if not rows:
        return []
    first = rows[0]
    if hasattr(first, "tolist"):
        converted = [row.tolist() for row in rows]
        if converted and isinstance(converted[0], (int, float)):
            return [list(map(float, converted))]
        return [list(map(float, row)) for row in converted]
    if isinstance(first, (list, tuple)):
        return [list(map(float, row)) for row in rows]
    # Treat as a flat vector of scalars
    return [list(map(float, rows))]


def _to_numpy(embeddings: List[List[float]]):
    if np is None:
        raise ImportError("numpy is required for FAISS operations")
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")
    return matrix


def _normalize_vectors(vectors: List[List[float]]) -> List[List[float]]:
    normalized: List[List[float]] = []
    for vector in vectors:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            normalized.append([0.0 for value in vector])
        else:
            normalized.append([value / norm for value in vector])
    return normalized


def _dot(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    return float(sum(a * b for a, b in zip(vec_a, vec_b)))


__all__ = ["FaissVectorStore", "is_faiss_available"]
