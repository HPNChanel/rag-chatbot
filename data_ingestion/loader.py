"""Utilities for loading and preprocessing documents for retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

try:  # pragma: no cover - handled via optional dependency in requirements
    from markdown2 import markdown
except ImportError:  # pragma: no cover
    markdown = None  # type: ignore

try:  # pragma: no cover - dependency declared in requirements
    from PyPDF2 import PdfReader
except ImportError:  # pragma: no cover
    try:  # pragma: no cover - fallback to pypdf which offers the same API
        from pypdf import PdfReader  # type: ignore
    except ImportError:  # pragma: no cover
        PdfReader = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass
class Document:
    """In-memory representation of a processed document chunk."""

    doc_id: str
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> Dict[str, str]:
        """Return a JSON-serialisable representation of the document."""

        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_json(payload: Dict[str, str]) -> "Document":
        """Instantiate a document from a JSON payload."""

        return Document(
            doc_id=payload["doc_id"],
            content=payload["content"],
            metadata=dict(payload.get("metadata", {})),
        )


class DocumentLoader:
    """Load documents from disk and chunk them into retrieval friendly units."""

    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt", ""}

    def __init__(
        self,
        base_path: Path | str,
        *,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        self.base_path = Path(base_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must not be negative")
        if self.chunk_overlap >= self.chunk_size:
            logger.warning(
                "chunk_overlap (%s) >= chunk_size (%s); overlap will be clamped",
                self.chunk_overlap,
                self.chunk_size,
            )

    def load(self) -> List[Document]:
        """Recursively load and chunk all supported documents under ``base_path``."""

        if not self.base_path.exists():
            raise FileNotFoundError(f"Data directory {self.base_path} does not exist")
        documents: List[Document] = []
        for path in sorted(self.base_path.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                logger.debug("Skipping unsupported file type: %s", path)
                continue
            try:
                documents.extend(self._load_file(path))
            except Exception as exc:  # pragma: no cover - logged for observability
                logger.exception("Failed to load %s: %s", path, exc)
        return documents

    def _load_file(self, path: Path) -> List[Document]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = self._read_pdf(path)
        elif suffix in {".md", ".markdown"}:
            text = self._read_markdown(path)
        else:
            text = self._read_text(path)
        return list(self._chunk_document(path, text))

    def _read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def _read_markdown(self, path: Path) -> str:
        raw = path.read_text(encoding="utf-8")
        if markdown is None:
            logger.warning("markdown2 not installed, returning raw Markdown text")
            return raw
        html = markdown(raw)
        # Strip HTML tags to provide cleaner text for downstream components.
        return re.sub(r"<[^>]+>", " ", html)

    def _read_pdf(self, path: Path) -> str:
        if PdfReader is None:
            raise ImportError("PyPDF2 or pypdf is required to load PDF files")
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    def _chunk_document(self, path: Path, text: str) -> Iterator[Document]:
        clean = clean_text(text)
        if not clean:
            return iter(())
        doc_id = self._relative_id(path)
        title = path.stem

        def _generator() -> Iterator[Document]:
            for idx, chunk in enumerate(chunk_text(clean, self.chunk_size, self.chunk_overlap)):
                chunk_id = f"{doc_id}::chunk_{idx}"
                yield Document(
                    doc_id=chunk_id,
                    content=chunk,
                    metadata={
                        "title": title,
                        "source": str(path),
                        "chunk_id": chunk_id,
                        "document_id": doc_id,
                    },
                )

        return _generator()

    def _relative_id(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.base_path))
        except ValueError:  # pragma: no cover - defensive guard
            return str(path)


def clean_text(text: str) -> str:
    """Normalise whitespace and strip leading/trailing spaces."""

    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\r\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterator[str]:
    """Split ``text`` into overlapping word-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    words = text.split()
    if not words:
        return iter(())
    if chunk_size >= len(words):
        def _single() -> Iterator[str]:
            yield " ".join(words)

        return _single()
    overlap = min(chunk_overlap, chunk_size - 1) if chunk_size > 1 else 0
    step = max(1, chunk_size - overlap)
    start = 0
    def _generator() -> Iterator[str]:
        nonlocal start
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            if not chunk_words:
                break
            yield " ".join(chunk_words)
            start += step

    return _generator()


def load_documents_from_paths(
    paths: Iterable[Path | str], *, chunk_size: int = 500, chunk_overlap: int = 50
) -> List[Document]:
    """Load documents from a heterogeneous collection of paths."""

    documents: List[Document] = []
    for path in paths:
        loader = DocumentLoader(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents.extend(loader.load())
    return documents


def save_documents(documents: Sequence[Document], destination: Path | str) -> None:
    """Persist a list of documents to disk for reproducibility."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = [doc.to_json() for doc in documents]
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_documents(path: Path | str) -> List[Document]:
    """Load pre-processed documents saved via :func:`save_documents`."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Document.from_json(item) for item in data]
