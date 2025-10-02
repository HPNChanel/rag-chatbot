"""Wrapper around the OpenAI Responses API."""
from __future__ import annotations

import os
from typing import Any, List, Sequence

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


class OpenAIGenerator:
    """Generate responses using OpenAI's chat completions API."""

    def __init__(self, model: str = "gpt-4o-mini", client: Any | None = None) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if client is None and OpenAI is None:
            raise ImportError("The openai package is required to use OpenAIGenerator")
        if not api_key and client is None:
            raise EnvironmentError("OPENAI_API_KEY environment variable is required")
        self.client = client or OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        prompt: str,
        context: Sequence[str] | None = None,
        max_tokens: int = 256,
    ) -> str:
        messages: List[dict[str, str]] = [{"role": "system", "content": "You are a helpful assistant."}]
        if context:
            context_block = "\n\n".join(context)
            messages.append({"role": "user", "content": f"Context:\n{context_block}"})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
        )
        message = response.choices[0].message.content or ""
        return message.strip()


__all__ = ["OpenAIGenerator"]
