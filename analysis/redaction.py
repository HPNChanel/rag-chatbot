"""Utilities for privacy and safety redaction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List

try:  # pragma: no cover - optional dependency
    import spacy
except Exception:  # pragma: no cover - graceful fallback
    spacy = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import nltk
    from nltk import ne_chunk, pos_tag, word_tokenize
    from nltk.tree import Tree
except Exception:  # pragma: no cover - graceful fallback
    nltk = None  # type: ignore[assignment]
    ne_chunk = pos_tag = word_tokenize = Tree = None  # type: ignore[assignment]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s\-()]{7,}\d)")
CREDIT_CARD_RE = re.compile(r"(?<!\d)(?:\d[\s-]?){13,16}(?!\d)")
SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")


def redact_pii(text: str) -> str:
    """Redact common PII patterns using deterministic placeholders."""

    if not text:
        return text
    redacted = EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    redacted = PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    redacted = CREDIT_CARD_RE.sub("[REDACTED_CARD]", redacted)
    redacted = SSN_RE.sub("[REDACTED_SSN]", redacted)
    redacted = _redact_named_entities(redacted)
    return redacted


def redact_sensitive_terms(text: str, terms: Iterable[str]) -> str:
    """Redact provided sensitive terms case-insensitively."""

    if not text:
        return text
    redacted = text
    for term in terms:
        if not term:
            continue
        pattern = re.compile(re.escape(term), flags=re.IGNORECASE)
        redacted = pattern.sub("[REDACTED_TERM]", redacted)
    return redacted


def _redact_named_entities(text: str) -> str:
    """Apply lightweight NER-based redaction when NLP models are available."""

    if spacy is not None:
        try:  # pragma: no cover - depends on external model availability
            nlp = (
                spacy.blank("en")
                if not spacy.util.is_package("en_core_web_sm")
                else spacy.load("en_core_web_sm")
            )
            doc = nlp(text)
            spans = [span for span in doc.ents if span.label_ in {"PERSON", "ORG"}]
            return _replace_spans(text, spans)
        except Exception:
            pass
    if nltk is not None and word_tokenize is not None:
        try:  # pragma: no cover - optional path
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            chunks = ne_chunk(tagged)
            spans = []
            idx = 0
            for node in chunks:
                if isinstance(node, Tree) and node.label() in {
                    "PERSON",
                    "ORGANIZATION",
                }:
                    current_tokens = [token for token, _ in node.leaves()]
                    length = len(current_tokens)
                    spans.append((idx, idx + length))
                    idx += length
                else:
                    idx += 1
            if not spans:
                return text
            redacted = list(tokens)
            for start, end in spans:
                redacted[start:end] = ["[REDACTED_NAME]"]
            return " ".join(redacted)
        except LookupError:
            pass
    return text


def _replace_spans(text: str, spans: Iterable[object]) -> str:
    result = text
    offset = 0
    for span in spans:
        try:
            start = span.start_char  # type: ignore[attr-defined]
            end = span.end_char  # type: ignore[attr-defined]
        except AttributeError:
            continue
        result = result[: start + offset] + "[REDACTED_NAME]" + result[end + offset :]
        offset += len("[REDACTED_NAME]") - (end - start)
    return result


@dataclass
class RedactionSettings:
    enable_redaction: bool = True
    redact_pii: bool = True
    sensitive_terms: List[str] = field(default_factory=list)


class TextRedactor:
    """Configurable text redaction helper."""

    def __init__(self, settings: RedactionSettings | None = None) -> None:
        self.settings = settings or RedactionSettings()

    def redact(self, text: str) -> str:
        if not self.settings.enable_redaction:
            return text
        redacted = text
        if self.settings.redact_pii:
            redacted = redact_pii(redacted)
        if self.settings.sensitive_terms:
            redacted = redact_sensitive_terms(redacted, self.settings.sensitive_terms)
        return redacted

    def redact_many(self, texts: Iterable[str]) -> List[str]:
        return [self.redact(text) for text in texts]


__all__ = [
    "TextRedactor",
    "RedactionSettings",
    "redact_pii",
    "redact_sensitive_terms",
]
