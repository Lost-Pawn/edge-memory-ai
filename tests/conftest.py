"""Shared fixtures and lightweight fakes (no model weights, no SDK needed)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from edge_memory.config import CloudSettings, EdgeSettings
from edge_memory.embeddings.base import SparseVector


class FakeEmbedder:
    """Deterministic hash-based embeddings so tests need no models or network."""

    dim = 8

    @property
    def dense_dim(self) -> int:
        return self.dim

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        out = []
        for t in texts:
            tokens = {abs(hash(w)) % 1000 for w in t.lower().split()}
            idx = sorted(tokens)
            out.append(SparseVector(indices=idx, values=[1.0] * len(idx)))
        return out

    def embed_image(self, paths: list[Path]) -> list[list[float]]:
        return [self._vec(str(p)) for p in paths]

    def _vec(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[: self.dim]]


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def edge_settings(tmp_path: Path) -> EdgeSettings:
    return EdgeSettings(
        device_id="edge-test",
        data_dir=tmp_path / "edge",
        models_dir=tmp_path / "models",
        llm_enabled=False,
        storage_budget=5,
        importance_sync_threshold=0.4,
    )


@pytest.fixture
def cloud_settings() -> CloudSettings:
    return CloudSettings(collection="test_memory", dense_dim=8)
