"""Dummy generator useful for tests and offline runs."""
from __future__ import annotations

from typing import Sequence


class EchoGenerator:
    """Generator that returns the prompt appended with context."""

    def generate(self, prompt: str, context: Sequence[str] | None = None, max_tokens: int = 256) -> str:
        context_block = " | ".join(context) if context else ""
        return f"{prompt} :: {context_block}".strip()
