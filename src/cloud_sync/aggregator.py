"""Aggregation logic: apply conflict resolution + semantic dedupe, then persist.

This is where the "intelligence" of the edge-to-cloud workflow lives. An incoming memory
from an edge device is reconciled against what the global brain already knows before it is
stored, so the cloud stays consistent even when many devices edit concurrently offline.
"""

from __future__ import annotations

from edge_memory.config import CloudSettings
from edge_memory.logging import get_logger
from edge_memory.models import Memory, SyncState
from edge_memory.sync.conflict import is_duplicate, resolve
from edge_memory.sync.protocol import (
    ConflictReport,
    PullRequest,
    PullResponse,
    PushRequest,
    PushResponse,
)
from cloud_sync.qdrant_repo import QdrantRepo

log = get_logger(__name__)


class Aggregator:
    def __init__(self, settings: CloudSettings, repo: QdrantRepo) -> None:
        self._settings = settings
        self._repo = repo
        self._dedupe_threshold = 0.95

    def push(self, request: PushRequest) -> PushResponse:
        accepted: list[str] = []
        conflicts: list[ConflictReport] = []
        rejected: list[str] = []

        for memory in request.memories:
            try:
                report = self._store_one(memory)
                if report is not None:
                    conflicts.append(report)
                else:
                    accepted.append(memory.id)
            except Exception as exc:  # pragma: no cover - defensive per-item
                log.warning("push item failed", id=memory.id, error=str(exc))
                rejected.append(memory.id)

        log.info(
            "push handled",
            device=request.device_id,
            accepted=len(accepted),
            conflicts=len(conflicts),
            rejected=len(rejected),
        )
        return PushResponse(accepted=accepted, conflicts=conflicts, rejected=rejected)

    def _store_one(self, memory: Memory) -> ConflictReport | None:
        # 1. Same-id conflict resolution.
        existing = self._repo.get(memory.id)
        winner, report = resolve(memory, existing)

        # 2. Semantic dedupe against a different existing memory.
        if existing is None:
            near = self._repo.nearest(winner.text, limit=1)
            if near and is_duplicate(near[0][1], self._dedupe_threshold):
                dup, score = near[0]
                merged = dup.model_copy(
                    update={
                        "version": max(dup.version, winner.version) + 1,
                        "access_count": dup.access_count + winner.access_count,
                    }
                )
                self._repo.upsert(merged)
                return ConflictReport(
                    memory_id=winner.id,
                    resolution="merged",
                    detail=f"semantic dedupe (cosine={score:.3f}) into {dup.id}",
                    winner=merged,
                )

        winner.sync_state = SyncState.SYNCED
        winner.origin = "cloud"
        self._repo.upsert(winner)
        return report

    def pull(self, request: PullRequest) -> PullResponse:
        memories, cursor = self._repo.updated_since(
            request.since, exclude_device=request.device_id, limit=request.limit
        )
        # Optional push-down filter (e.g. only memories for this device's site/role).
        if request.filters:
            memories = [
                m
                for m in memories
                if all(m.metadata.get(k) == v for k, v in request.filters.items())
            ]
        return PullResponse(memories=memories, cursor=cursor)
