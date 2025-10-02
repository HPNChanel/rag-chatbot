from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from data_ingestion.loader import Document
from indexing.embedder import HashingEmbedder
from indexing.index import BM25Index, HybridIndex, VectorIndex


@pytest.fixture()
def sample_documents() -> List[Document]:
    return [
        Document(doc_id="doc::chunk_0", content="python programming language", metadata={"title": "doc"}),
        Document(doc_id="doc::chunk_1", content="machine learning and statistics", metadata={"title": "doc"}),
        Document(doc_id="doc::chunk_2", content="deep learning with python", metadata={"title": "doc"}),
    ]


def test_vector_index_build_query_and_persist(tmp_path: Path, sample_documents: List[Document]) -> None:
    index = VectorIndex(embedder=HashingEmbedder(dim=64), use_faiss=False)
    index.build(sample_documents)
    results = index.query("python", top_k=2)
    assert results
    assert any("python" in result.document.content for result in results)

    save_path = tmp_path / "vector"
    index.save(save_path)
    loaded = VectorIndex.load(save_path, embedder=HashingEmbedder(dim=64))
    loaded_results = loaded.query("python", top_k=2)
    assert loaded_results[0].document.doc_id == results[0].document.doc_id


def test_bm25_index_build_and_query(sample_documents: List[Document]) -> None:
    index = BM25Index()
    index.build(sample_documents)
    results = index.query("statistics", top_k=1)
    assert results
    assert "statistics" in results[0].document.content


def test_hybrid_index_combines_scores(sample_documents: List[Document]) -> None:
    hybrid = HybridIndex(
        vector_index=VectorIndex(embedder=HashingEmbedder(dim=64), use_faiss=False),
        dense_weight=0.7,
        sparse_weight=0.3,
    )
    hybrid.build(sample_documents)
    results = hybrid.query("python learning", top_k=3)
    assert results
    # Ensure results include both dense and sparse contributions
    assert any(result.dense_score is not None for result in results)
    assert any(result.sparse_score is not None for result in results)


def test_hybrid_index_persistence(tmp_path: Path, sample_documents: List[Document]) -> None:
    hybrid = HybridIndex(vector_index=VectorIndex(embedder=HashingEmbedder(dim=32), use_faiss=False))
    hybrid.build(sample_documents)
    hybrid.save(tmp_path)

    loaded = HybridIndex.load(tmp_path, embedder=HashingEmbedder(dim=32))
    results = loaded.query("python", top_k=1)
    assert results
    assert "python" in results[0].document.content
