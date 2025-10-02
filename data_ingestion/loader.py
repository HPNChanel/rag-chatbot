"""Utilities for loading documents from various sources."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import logging

try:
    from markdown2 import markdown
except ImportError:  # pragma: no cover - dependency handled via requirements
    markdown = None  # type: ignore

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """In-memory representation of a document."""

    doc_id: str
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)


class DocumentLoader:
    """Load documents from common file formats."""

    def __init__(self, base_path: Path | str):
        self.base_path = Path(base_path)

    def load(self) -> List[Document]:
        """Recursively load all supported documents under ``base_path``."""
        documents: List[Document] = []
        for path in self.base_path.rglob("*"):
            if path.is_file():
                doc = self._load_file(path)
                if doc:
                    documents.append(doc)
        return documents

    def _load_file(self, path: Path) -> Optional[Document]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf(path)
        if suffix in {".md", ".markdown"}:
            return self._load_markdown(path)
        if suffix in {".txt", ""}:
            return self._load_text(path)
        logger.debug("Skipping unsupported file type: %s", path)
        return None

    def _load_text(self, path: Path) -> Document:
        content = path.read_text(encoding="utf-8")
        return Document(doc_id=str(path.relative_to(self.base_path)), content=content, metadata={"source": str(path)})

    def _load_markdown(self, path: Path) -> Document:
        raw = path.read_text(encoding="utf-8")
        if markdown is None:
            logger.warning("markdown2 not installed, returning raw Markdown text")
            content = raw
        else:
            content = markdown(raw)
        return Document(doc_id=str(path.relative_to(self.base_path)), content=content, metadata={"source": str(path), "format": "markdown"})

    def _load_pdf(self, path: Path) -> Document:
        if PdfReader is None:
            raise ImportError("pypdf is required to load PDF files")
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        content = "\n".join(pages)
        return Document(doc_id=str(path.relative_to(self.base_path)), content=content, metadata={"source": str(path), "format": "pdf"})


def load_documents_from_paths(paths: Iterable[Path | str]) -> List[Document]:
    """Load documents from a heterogeneous collection of paths."""
    documents: List[Document] = []
    for path in paths:
        loader = DocumentLoader(path)
        documents.extend(loader.load())
    return documents
