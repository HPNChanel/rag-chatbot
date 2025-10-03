"""Citation-aware orchestration utilities for text generation."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Protocol, Sequence

from data_ingestion.loader import Document


class SupportsGenerate(Protocol):  # pragma: no cover - runtime structural typing
    """Protocol describing the subset of generator behaviour we rely on."""

    def generate(
        self,
        prompt: str,
        context: Iterable[str] | None = None,
        max_tokens: int | None = None,
    ) -> str:
        ...


@dataclass(slots=True)
class GeneratedAnswer:
    """Structured representation of the LLM output."""

    answer: str
    citations: List[str]


class CitationFirstPipeline:
    """Encapsulate the "citation-first" prompting strategy.

    The pipeline decorates retrieved documents with explicit identifiers before
    forwarding them to the underlying language model. The generated response is
    post-processed to extract the cited document identifiers which can then be
    surfaced alongside the final answer.
    """

    citation_pattern = re.compile(r"\[(?P<doc>[^\[\]]+)\]")

    def __init__(
        self,
        generator: SupportsGenerate,
        *,
        instruction: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.generator = generator
        self.instruction = instruction or (
            "You are a retrieval-augmented assistant. Use ONLY the provided "
            "documents to answer the user's question. Cite evidence inline "
            "using the format [doc_id] where doc_id is taken from the "
            "document headers."
        )
        self.max_tokens = max_tokens

    def _format_documents(self, documents: Sequence[Document]) -> List[str]:
        formatted: List[str] = []
        for document in documents:
            formatted.append(f"[{document.doc_id}] {document.content}")
        return formatted

    def _build_prompt(self, query: str) -> str:
        return (
            f"{self.instruction}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    def _extract_citations(self, answer: str, candidate_ids: Sequence[str]) -> List[str]:
        candidates = set(candidate_ids)
        citations: List[str] = []
        for match in self.citation_pattern.finditer(answer):
            doc_id = match.group("doc")
            if doc_id in candidates and doc_id not in citations:
                citations.append(doc_id)
        return citations

    def generate(self, query: str, documents: Sequence[Document]) -> GeneratedAnswer:
        """Generate an answer for ``query`` grounded in ``documents``."""

        context = self._format_documents(documents)
        prompt = self._build_prompt(query)
        answer = self.generator.generate(prompt, context=context, max_tokens=self.max_tokens)
        citations = self._extract_citations(answer, [doc.doc_id for doc in documents])
        return GeneratedAnswer(answer=answer, citations=citations)


__all__ = ["CitationFirstPipeline", "GeneratedAnswer", "SupportsGenerate"]
