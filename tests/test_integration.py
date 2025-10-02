from pathlib import Path

from generation.dummy import EchoGenerator
from indexing.embedder import get_default_embedder
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
