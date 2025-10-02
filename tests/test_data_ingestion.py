from pathlib import Path

from data_ingestion.loader import DocumentLoader


def test_load_text(tmp_path: Path) -> None:
    file_path = tmp_path / "doc.txt"
    file_path.write_text("Hello world", encoding="utf-8")
    loader = DocumentLoader(tmp_path)
    docs = loader.load()
    assert len(docs) == 1
    assert docs[0].content == "Hello world"
