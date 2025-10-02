from pathlib import Path

from data_ingestion.loader import DocumentLoader
from generation.dummy import EchoGenerator
from indexing.embedder import get_default_embedder, HashingEmbedder
from indexing.index import HybridIndex, VectorIndex
from pipelines.chatbot import RAGPipeline


def test_end_to_end_query(tmp_path: Path) -> None:
    data_dir = tmp_path / "docs"
    data_dir.mkdir()
    (data_dir / "doc1.txt").write_text("Python is great for scripting", encoding="utf-8")
    (data_dir / "doc2.txt").write_text("Rust is great for systems", encoding="utf-8")

    pipeline = RAGPipeline(
        data_path=data_dir,
        embedder=get_default_embedder(prefer_lightweight=True),
        generator=EchoGenerator(),
        use_faiss=False,
    )
    docs = pipeline.query("Python", top_k=1)
    assert docs
    response = pipeline.chat("Python", top_k=1)
    assert "Python" in response


def test_loader_and_index_integration(tmp_path: Path) -> None:
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    (data_dir / "doc1.txt").write_text("Python enables rapid application development", encoding="utf-8")
    (data_dir / "doc2.md").write_text("## Heading\n\nRust focuses on performance.", encoding="utf-8")

    loader = DocumentLoader(data_dir, chunk_size=20)
    documents = loader.load()
    assert documents

    index = HybridIndex(
        vector_index=VectorIndex(embedder=HashingEmbedder(dim=64), use_faiss=False),
        dense_weight=0.6,
        sparse_weight=0.4,
    )
    index.build(documents)

    results = index.query("rapid development", top_k=2)
    assert results
    assert any("python" in result.document.content.lower() for result in results)
