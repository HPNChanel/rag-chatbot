"""Indexing utilities for dense, sparse, and hybrid retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import json
import logging
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency handled via requirements
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

try:  # pragma: no cover - optional dependency handled via requirements
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover - optional dependency handled via requirements
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    BM25Okapi = None  # type: ignore

from data_ingestion.loader import Document
from .embedder import BaseEmbedder, HashingEmbedder, SentenceTransformerEmbedder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Structured representation of a retrieval match."""

    document: Document
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None


class VectorIndex:
    """FAISS-backed dense retrieval index."""

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        *,
        use_faiss: Optional[bool] = None,
        normalize: bool = True,
    ) -> None:
        self.embedder = embedder or _default_embedder()
        self.normalize = normalize
        self.use_faiss = self._resolve_faiss_flag(use_faiss)
        self._documents: List[Document] = []
        self._faiss_index = None
        self._normalized_embeddings = None

    @staticmethod
    def _resolve_faiss_flag(flag: Optional[bool]) -> bool:
        if flag is None:
            return faiss is not None and np is not None
        if flag and (faiss is None or np is None):
            raise ImportError("FAISS and numpy are required when use_faiss=True")
        return flag

    @property
    def documents(self) -> Sequence[Document]:
        return self._documents

    def build(self, documents: Sequence[Document]) -> None:
        """Build the dense index from ``documents``."""

        if not documents:
            raise ValueError("Cannot build an index with zero documents")
        self._documents = list(documents)
        if self.use_faiss:
            if np is None:
                raise ImportError("numpy is required for FAISS indexing")
            matrix = self._embed_texts([doc.content for doc in self._documents])
            if matrix.size == 0:
                raise ValueError("Embedder returned no embeddings")
            if self.normalize:
                matrix = _normalize_matrix(matrix)
            dimension = matrix.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dimension)  # type: ignore[arg-type]
            self._faiss_index.add(matrix)
        else:
            embeddings = list(self.embedder.embed([doc.content for doc in self._documents]))
            if not embeddings:
                raise ValueError("Embedder returned no embeddings")
            vectors = [_ensure_float_list(vec) for vec in embeddings]
            if self.normalize:
                vectors = [_normalize_list(vec) for vec in vectors]
            self._normalized_embeddings = vectors

    def query(self, text: str, top_k: int = 5) -> List[RetrievalResult]:
        """Return the ``top_k`` most similar documents for ``text``."""

        if not self._documents:
            raise RuntimeError("Index has not been built")
        query_embeddings = list(self.embedder.embed([text]))
        if not query_embeddings:
            return []
        if self.use_faiss:
            if self._faiss_index is None:
                raise RuntimeError("FAISS index has not been built")
            if np is None:
                raise ImportError("numpy is required for FAISS indexing")
            query_matrix = self._embed_texts([text])
            if self.normalize:
                query_matrix = _normalize_matrix(query_matrix)
            scores, indices = self._faiss_index.search(query_matrix, top_k)  # type: ignore[arg-type]
        else:
            if self._normalized_embeddings is None:
                raise RuntimeError("Index has not been built")
            query_vector = _ensure_float_list(query_embeddings[0])
            if self.normalize:
                query_vector = _normalize_list(query_vector)
            scores, indices = _brute_force_cosine_list(self._normalized_embeddings, query_vector, top_k)
        results: List[RetrievalResult] = []
        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue
            document = self._documents[int(index)]
            results.append(RetrievalResult(document=document, score=float(score), dense_score=float(score)))
        return results

    def save(self, directory: Path | str) -> None:
        """Persist the index to ``directory``."""

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        docs_payload = [doc.to_json() for doc in self._documents]
        (path / "documents.json").write_text(json.dumps(docs_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        config = {
            "normalize": self.normalize,
            "use_faiss": self.use_faiss,
            "embedder": self._serialize_embedder(),
        }
        if self.use_faiss:
            config["embedding_format"] = "faiss"
        else:
            config["embedding_format"] = "json"
            embeddings = self._normalized_embeddings
            if embeddings is None:
                raise RuntimeError("Index has not been built")
            (path / "embeddings.json").write_text(json.dumps(embeddings, ensure_ascii=False), encoding="utf-8")
        (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        if self.use_faiss:
            if self._faiss_index is None:
                raise RuntimeError("FAISS index has not been built")
            faiss.write_index(self._faiss_index, str(path / "index.faiss"))  # type: ignore[arg-type]

    @classmethod
    def load(
        cls,
        directory: Path | str,
        *,
        embedder: Optional[BaseEmbedder] = None,
    ) -> "VectorIndex":
        """Load an index previously saved with :meth:`save`."""

        path = Path(directory)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        embedder = embedder or _deserialize_embedder(config.get("embedder"))
        index = cls(embedder=embedder, use_faiss=config.get("use_faiss", True), normalize=config.get("normalize", True))
        documents = [Document.from_json(item) for item in json.loads((path / "documents.json").read_text(encoding="utf-8"))]
        index._documents = documents
        if index.use_faiss:
            if faiss is None:
                raise ImportError("faiss is required to load a FAISS-backed index")
            index._faiss_index = faiss.read_index(str(path / "index.faiss"))  # type: ignore[arg-type]
        else:
            fmt = config.get("embedding_format", "json")
            if fmt == "json":
                data = json.loads((path / "embeddings.json").read_text(encoding="utf-8"))
                index._normalized_embeddings = [[float(value) for value in vector] for vector in data]
            elif fmt == "npy":
                if np is None:
                    raise ImportError("numpy is required to load embeddings saved in npy format")
                index._normalized_embeddings = np.load(path / "embeddings.npy").tolist()
            else:
                raise ValueError(f"Unsupported embedding format: {fmt}")
        return index

    def _embed_texts(self, texts: Iterable[str]) -> "np.ndarray":
        if np is None:
            raise ImportError("numpy is required for dense indexing")
        matrix = np.asarray(list(self.embedder.embed(texts)), dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix")
        return matrix

    def _serialize_embedder(self) -> Dict[str, str]:
        if isinstance(self.embedder, SentenceTransformerEmbedder):
            model_name = getattr(self.embedder.model, "name_or_path", "")
            return {"type": "sentence-transformer", "model_name": model_name}
        if isinstance(self.embedder, HashingEmbedder):
            return {"type": "hashing", "dim": getattr(self.embedder, "dim", 256)}
        return {"type": self.embedder.__class__.__name__}


class BM25Index:
    """Sparse BM25 index built with :mod:`rank_bm25`."""

    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]] = None) -> None:
        self.tokenizer = tokenizer or _default_tokenizer
        self._documents: List[Document] = []
        self._tokenized_corpus: List[List[str]] = []
        self._model = None

    @property
    def documents(self) -> Sequence[Document]:
        return self._documents

    def build(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("Cannot build an index with zero documents")
        self._documents = list(documents)
        self._tokenized_corpus = [self.tokenizer(doc.content) for doc in self._documents]
        if BM25Okapi is not None and np is not None:
            self._model = BM25Okapi(self._tokenized_corpus)
        else:
            self._model = None

    def query(self, text: str, top_k: int = 5) -> List[RetrievalResult]:
        tokens = self.tokenizer(text)
        if self._model is not None and np is not None:
            scores = self._model.get_scores(tokens)
            ranked_indices = np.argsort(scores)[::-1][:top_k]
            results: List[RetrievalResult] = []
            for idx in ranked_indices:
                results.append(
                    RetrievalResult(
                        document=self._documents[int(idx)],
                        score=float(scores[idx]),
                        sparse_score=float(scores[idx]),
                    )
                )
            return results
        return self._fallback_search(tokens, top_k)

    def _fallback_search(self, tokens: List[str], top_k: int) -> List[RetrievalResult]:
        if not self._documents:
            return []
        query_counts = Counter(tokens)
        scored: List[Tuple[int, float]] = []
        for idx, doc_tokens in enumerate(self._tokenized_corpus):
            doc_counts = Counter(doc_tokens)
            score = float(sum(query_counts[token] * doc_counts.get(token, 0) for token in query_counts))
            scored.append((idx, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        results: List[RetrievalResult] = []
        for idx, score in scored[:top_k]:
            results.append(
                RetrievalResult(
                    document=self._documents[idx],
                    score=score,
                    sparse_score=score,
                )
            )
        return results

    def save(self, directory: Path | str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": [doc.to_json() for doc in self._documents],
        }
        (path / "bm25.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        directory: Path | str,
        *,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> "BM25Index":
        path = Path(directory)
        payload = json.loads((path / "bm25.json").read_text(encoding="utf-8"))
        documents = [Document.from_json(item) for item in payload.get("documents", [])]
        index = cls(tokenizer=tokenizer)
        if documents:
            index.build(documents)
        return index


class HybridIndex:
    """Combine dense and sparse indexes for hybrid retrieval."""

    def __init__(
        self,
        vector_index: Optional[VectorIndex] = None,
        bm25_index: Optional[BM25Index] = None,
        *,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> None:
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError("dense_weight must be between 0 and 1")
        if not 0.0 <= sparse_weight <= 1.0:
            raise ValueError("sparse_weight must be between 0 and 1")
        if dense_weight == 0.0 and sparse_weight == 0.0:
            raise ValueError("At least one of dense_weight or sparse_weight must be non-zero")
        self.vector_index = vector_index or VectorIndex()
        self.bm25_index = bm25_index or BM25Index()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def build(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("Cannot build an index with zero documents")
        self.vector_index.build(documents)
        self.bm25_index.build(documents)

    def query(self, text: str, top_k: int = 5) -> List[RetrievalResult]:
        dense_results = self.vector_index.query(text, top_k=top_k)
        sparse_results = self.bm25_index.query(text, top_k=top_k)
        score_map: Dict[str, RetrievalResult] = {}

        def _accumulate(results: Iterable[RetrievalResult], weight: float, attr: str) -> None:
            for result in results:
                doc_id = result.document.doc_id
                base = score_map.get(doc_id)
                if base is None:
                    base = RetrievalResult(document=result.document, score=0.0)
                    score_map[doc_id] = base
                score = getattr(result, attr) if getattr(result, attr) is not None else result.score
                if attr == "dense_score":
                    base.dense_score = score
                else:
                    base.sparse_score = score
                base.score += weight * score

        _accumulate(dense_results, self.dense_weight, "dense_score")
        _accumulate(sparse_results, self.sparse_weight, "sparse_score")
        ranked = sorted(score_map.values(), key=lambda item: item.score, reverse=True)
        return ranked[:top_k]

    def save(self, directory: Path | str) -> None:
        path = Path(directory)
        self.vector_index.save(path / "dense")
        self.bm25_index.save(path / "bm25")
        config = {
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
        }
        (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        directory: Path | str,
        *,
        embedder: Optional[BaseEmbedder] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> "HybridIndex":
        path = Path(directory)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        vector_index = VectorIndex.load(path / "dense", embedder=embedder)
        bm25_index = BM25Index.load(path / "bm25", tokenizer=tokenizer)
        return cls(
            vector_index=vector_index,
            bm25_index=bm25_index,
            dense_weight=config.get("dense_weight", 0.5),
            sparse_weight=config.get("sparse_weight", 0.5),
        )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _default_embedder() -> BaseEmbedder:
    try:
        return SentenceTransformerEmbedder()
    except Exception:  # pragma: no cover - fall back to hashing embedder for lightweight usage
        logger.warning("Falling back to HashingEmbedder due to SentenceTransformer availability issues")
        return HashingEmbedder()


def _normalize_matrix(matrix: "np.ndarray") -> "np.ndarray":
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _ensure_float_list(vector: Iterable[float]) -> List[float]:
    if isinstance(vector, list):
        return [float(value) for value in vector]
    if hasattr(vector, "tolist"):
        converted = vector.tolist()
        if isinstance(converted, list):
            return [float(value) for value in converted]
        return [float(converted)]
    return [float(value) for value in vector]


def _normalize_list(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return [0.0 for value in vector]
    return [float(value) / norm for value in vector]


def _brute_force_cosine_list(
    embeddings: Sequence[Sequence[float]], query: Sequence[float], top_k: int
) -> Tuple[List[List[float]], List[List[int]]]:
    scores = [_dot(vector, query) for vector in embeddings]
    ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
    top_scores = [scores[idx] for idx in ranked_indices]
    while len(top_scores) < top_k:
        ranked_indices.append(-1)
        top_scores.append(-1.0)
    return [top_scores], [ranked_indices]


def _dot(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    return float(sum(a * b for a, b in zip(vec_a, vec_b)))


def _default_tokenizer(text: str) -> List[str]:
    return text.lower().split()


def _deserialize_embedder(config: Optional[Dict[str, str]]) -> BaseEmbedder:
    if not config:
        return _default_embedder()
    if config.get("type") == "sentence-transformer":
        model_name = config.get("model_name") or "sentence-transformers/all-MiniLM-L6-v2"
        return SentenceTransformerEmbedder(model_name)
    if config.get("type") == "hashing":
        dim = int(config.get("dim", 256))
        return HashingEmbedder(dim=dim)
    logger.warning("Unknown embedder type %s, using default", config.get("type"))
    return _default_embedder()


__all__ = [
    "BM25Index",
    "HybridIndex",
    "RetrievalResult",
    "VectorIndex",
]
