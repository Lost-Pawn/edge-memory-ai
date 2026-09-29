"""Cloud sync service FastAPI application (the central brain)."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cloud_sync.aggregator import Aggregator
from cloud_sync.qdrant_repo import QdrantRepo
from edge_memory.config import get_cloud_settings, get_edge_settings
from edge_memory.embeddings import FastEmbedProvider
from edge_memory.logging import configure_logging, get_logger
from edge_memory.sync.protocol import (
    PullRequest,
    PullResponse,
    PushRequest,
    PushResponse,
)

log = get_logger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_cloud_settings()
    # Reuse the edge embedding config so dense/sparse spaces match end to end.
    embedder = FastEmbedProvider(get_edge_settings())
    repo = QdrantRepo(settings, embedder)
    app.state.aggregator = Aggregator(settings, repo)
    app.state.repo = repo
    log.info("cloud brain up", collection=settings.collection)
    yield


app = FastAPI(title="EdgeMemoryAi — Cloud Brain", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/sync/push", response_model=PushResponse)
async def push(request: PushRequest) -> PushResponse:
    return app.state.aggregator.push(request)


@app.post("/sync/pull", response_model=PullResponse)
async def pull(request: PullRequest) -> PullResponse:
    return app.state.aggregator.pull(request)


@app.get("/stats")
async def stats() -> dict[str, int]:
    return {"total_memories": app.state.repo.count()}
