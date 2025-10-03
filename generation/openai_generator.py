"""Wrapper around the OpenAI Responses API."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass(slots=True)
class OpenAIConfig:
    """Configuration values used when calling the OpenAI API."""

    model: str = "gpt-4o-mini"
    system_prompt: str = "You are a helpful assistant."
    temperature: float = 0.2
    max_tokens: int = 256
    extra_params: MutableMapping[str, Any] = field(default_factory=dict)


class OpenAIGenerator:
    """Generate responses using OpenAI's chat completions API.

    The implementation is intentionally lightweight and focuses on making
    dependency injection and configuration simple so that unit tests can supply
    mock clients without needing network access.
    """

    def __init__(
        self,
        config: OpenAIConfig | None = None,
        client: Any | None = None,
    ) -> None:
        self.config = config or OpenAIConfig()
        api_key = os.getenv("OPENAI_API_KEY")
        if client is None and OpenAI is None:
            raise ImportError("The openai package is required to use OpenAIGenerator")
        if not api_key and client is None:
            raise EnvironmentError("OPENAI_API_KEY environment variable is required")
        self.client = client or OpenAI(api_key=api_key)

    def _build_messages(self, prompt: str, context: Sequence[str] | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.config.system_prompt}
        ]
        if context:
            context_block = "\n\n".join(context)
            messages.append({"role": "user", "content": f"Context:\n{context_block}"})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self,
        prompt: str,
        context: Sequence[str] | None = None,
        max_tokens: int | None = None,
        *,
        request_params: Mapping[str, Any] | None = None,
    ) -> str:
        """Return the generated text for ``prompt`` and ``context``.

        Parameters
        ----------
        prompt:
            The instruction or question to send to the model.
        context:
            Optional contextual passages appended to the conversation.
        max_tokens:
            Overrides the configured ``max_tokens`` when supplied.
        request_params:
            Additional key/value pairs forwarded to the OpenAI client. These
            values override any provided in :class:`OpenAIConfig.extra_params`.
        """

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": self._build_messages(prompt, context),
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        payload.update(self.config.extra_params)
        if request_params:
            payload.update(request_params)
        response = self.client.chat.completions.create(**payload)
        message = response.choices[0].message.content or ""
        return message.strip()


__all__ = ["OpenAIGenerator", "OpenAIConfig"]
