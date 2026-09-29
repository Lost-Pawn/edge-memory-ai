"""Runtime configuration, loaded from environment / .env.

All settings are validated by pydantic. Nothing here triggers a network call or a
model download — model fetching is an explicit, separate action (see cli.py).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EdgeSettings(BaseSettings):
    """Configuration for an edge device node."""

    model_config = SettingsConfigDict(
        env_prefix="EDGE_", env_file=".env", extra="ignore"
    )

    # Identity / storage
    device_id: str = "edge-A"
    data_dir: Path = Path("./data/edge-A")
    api_host: str = "0.0.0.0"
    api_port: int = 8001

    # Embeddings (on-device, cached)
    models_dir: Path = Path("./models")
    dense_model: str = "Qdrant/clip-ViT-B-32-text"
    image_model: str = "Qdrant/clip-ViT-B-32-vision"
    sparse_model: str = "Qdrant/bm25"
    dense_dim: int = 512
    local_files_only: bool = True

    # Local LLM (in-process llama.cpp)
    llm_enabled: bool = True
    llm_model_path: Path = Path("./models/llm/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    llm_ctx: int = 4096
    llm_threads: int = 4
    llm_gpu_layers: int = 0

    # Sync policy
    storage_budget: int = 10_000
    importance_sync_threshold: float = 0.4
    dedupe_cosine_threshold: float = 0.95

    # Cloud sync
    cloud_url: str = "http://localhost:8000"
    sync_interval_seconds: int = 15

    # Download helpers (used only by explicit CLI download commands)
    download_llm_repo: str = Field("bartowski/Llama-3.2-3B-Instruct-GGUF")
    download_llm_file: str = Field("Llama-3.2-3B-Instruct-Q4_K_M.gguf")

    @property
    def shard_dir(self) -> Path:
        return self.data_dir / "shards"

    @property
    def outbox_path(self) -> Path:
        return self.data_dir / "outbox.db"

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.shard_dir, self.models_dir):
            path.mkdir(parents=True, exist_ok=True)


class CloudSettings(BaseSettings):
    """Configuration for the cloud sync / aggregation service."""

    model_config = SettingsConfigDict(
        env_prefix="CLOUD_", env_file=".env", extra="ignore"
    )

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection: str = "global_memory"
    dense_dim: int = 512


@lru_cache
def get_edge_settings() -> EdgeSettings:
    return EdgeSettings()


@lru_cache
def get_cloud_settings() -> CloudSettings:
    return CloudSettings()
