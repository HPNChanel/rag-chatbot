"""Embedding utilities for turning text into vectors."""
from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import re
from typing import Iterable, List, Sequence

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover - optional heavy dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore


ArrayLike = Sequence[Sequence[float]]


class BaseEmbedder(ABC):
    """Abstract interface for embedding text into vectors."""

    @abstractmethod
    def embed(self, texts: Iterable[str]) -> ArrayLike:
        """Return a 2D collection of embeddings."""


class SentenceTransformerEmbedder(BaseEmbedder):
    """SentenceTransformers embedder with a configurable model name."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required for SentenceTransformerEmbedder")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: Iterable[str]) -> ArrayLike:
        if np is None:
            raise ImportError("numpy is required to use SentenceTransformerEmbedder")
        return np.asarray(self.model.encode(list(texts), convert_to_numpy=True), dtype=np.float32)


class TfidfEmbedder(BaseEmbedder):
    """TF-IDF embedder that fits once and reuses the trained vocabulary."""

    def __init__(self) -> None:
        if TfidfVectorizer is None:
            raise ImportError("scikit-learn is required for TfidfEmbedder")
        self.vectorizer = TfidfVectorizer()
        self._is_fitted = False

    def embed(self, texts: Iterable[str]) -> ArrayLike:
        text_list = list(texts)
        if not text_list:
            return []
        if not self._is_fitted:
            matrix = self.vectorizer.fit_transform(text_list)
            self._is_fitted = True
        else:
            matrix = self.vectorizer.transform(text_list)
        if np is None:
            return matrix.toarray().tolist()
        return matrix.toarray().astype(np.float32)


class HashingEmbedder(BaseEmbedder):
    """Deterministic hashing embedder with no external dependencies."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: Iterable[str]) -> ArrayLike:
        vectors: List[List[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in self._tokenize(text):
                index = self._hash_token(token)
                vector[index] += 1.0
            vectors.append(vector)
        if np is not None:
            return np.asarray(vectors, dtype=np.float32)
        return vectors

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _hash_token(self, token: str) -> int:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="big") % self.dim


def get_default_embedder(prefer_lightweight: bool = False) -> BaseEmbedder:
    """Return the default embedder for the project."""

    if not prefer_lightweight and SentenceTransformer is not None and np is not None:
        return SentenceTransformerEmbedder()
    if TfidfVectorizer is not None and np is not None and not prefer_lightweight:
        return TfidfEmbedder()
    return HashingEmbedder()


__all__ = [
    "ArrayLike",
    "BaseEmbedder",
    "HashingEmbedder",
    "SentenceTransformerEmbedder",
    "TfidfEmbedder",
    "get_default_embedder",
]
