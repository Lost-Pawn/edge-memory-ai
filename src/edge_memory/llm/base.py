"""LLM provider protocol used by RAG and conflict-merge."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def available(self) -> bool: ...

    def generate(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 512
    ) -> str: ...
