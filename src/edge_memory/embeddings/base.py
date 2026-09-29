"""Embedding provider protocol.

Keeping this behind a Protocol lets the store and tests swap in a deterministic fake
embedder without pulling in FastEmbed or any model weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class SparseVector(BaseModel):
    """A sparse embedding as parallel index/value arrays (Qdrant sparse format)."""

    indices: list[int]
    values: list[float]


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Produces dense + sparse text embeddings and dense image embeddings."""

    @property
    def dense_dim(self) -> int: ...

    def embed_text(self, texts: list[str]) -> list[list[float]]: ...

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]: ...

    def embed_image(self, paths: list[Path]) -> list[list[float]]: ...
