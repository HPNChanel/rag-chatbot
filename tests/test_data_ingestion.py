from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List

import pytest

from data_ingestion.loader import DocumentLoader


def _create_pdf_bytes(text: str) -> bytes:
    """Create a minimal PDF document containing ``text``."""

    buffer = BytesIO()
    buffer.write(b"%PDF-1.4\n")
    offsets: List[int] = []

    def _write_obj(obj_num: int, content: bytes) -> None:
        offsets.append(buffer.tell())
        buffer.write(f"{obj_num} 0 obj\n".encode("utf-8"))
        buffer.write(content)
        buffer.write(b"\nendobj\n")

    _write_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    _write_obj(2, b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>")
    _write_obj(
        3,
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
    )
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("utf-8")
    _write_obj(4, b"<< /Length " + str(len(stream)).encode("utf-8") + b" >>\nstream\n" + stream + b"\nendstream")
    _write_obj(5, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    xref_offset = buffer.tell()
    buffer.write(f"xref\n0 {len(offsets) + 1}\n".encode("utf-8"))
    buffer.write(b"0000000000 65535 f \n")
    for offset in offsets:
        buffer.write(f"{offset:010d} 00000 n \n".encode("utf-8"))
    buffer.write(b"trailer\n<< /Size ")
    buffer.write(str(len(offsets) + 1).encode("utf-8"))
    buffer.write(b" /Root 1 0 R >>\nstartxref\n")
    buffer.write(str(xref_offset).encode("utf-8"))
    buffer.write(b"\n%%EOF")
    return buffer.getvalue()


def test_load_text_file_with_chunking(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("one two three four five six", encoding="utf-8")
    loader = DocumentLoader(tmp_path, chunk_size=3, chunk_overlap=1)
    docs = loader.load()
    assert len(docs) == 3
    assert docs[0].content == "one two three"
    assert docs[0].metadata["chunk_id"].endswith("chunk_0")
    assert docs[0].metadata["title"] == "sample"


def test_load_markdown_file(tmp_path: Path) -> None:
    file_path = tmp_path / "readme.md"
    file_path.write_text("# Title\n\nThis is *markdown* content.", encoding="utf-8")
    loader = DocumentLoader(tmp_path, chunk_size=10, chunk_overlap=2)
    docs = loader.load()
    assert docs
    assert "this is" in docs[0].content.lower()
    assert docs[0].metadata["source"] == str(file_path)


def test_load_pdf_file(tmp_path: Path) -> None:
    pytest.importorskip("PyPDF2")
    pdf_bytes = _create_pdf_bytes("Hello PDF world")
    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(pdf_bytes)
    loader = DocumentLoader(tmp_path, chunk_size=50)
    docs = loader.load()
    assert docs
    assert any("hello" in doc.content.lower() for doc in docs)
    assert docs[0].metadata["source"] == str(pdf_path)
