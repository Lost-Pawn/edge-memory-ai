"""Core domain models shared across the edge stack and the sync protocol.

A ``Memory`` is the unit of knowledge stored on device. Its payload fields carry the
metadata that drives the sync-policy engine and conflict resolution, so this schema is
the contract between the edge store, the outbox, and the cloud service.
"""

from __future__ import annotations

import time
import uuid
from enum import StrEnum

from pydantic import BaseModel, Field


class Sensitivity(StrEnum):
    """Data-classification tier that gates whether a memory may leave the device."""

    PII = "pii"          # personal / sensitive -> never syncs, stays local-only
    INTERNAL = "internal"  # may sync to the private cloud brain
    PUBLIC = "public"      # freely shareable


class SyncState(StrEnum):
    """Lifecycle of a memory with respect to the cloud."""

    LOCAL_ONLY = "local_only"  # policy decided it stays on device
    PENDING = "pending"        # queued in the outbox, awaiting push
    SYNCED = "synced"          # confirmed present on the cloud brain
    CONFLICT = "conflict"      # concurrent edits detected, needs resolution
    FAILED = "failed"          # push failed after retries


class MemoryKind(StrEnum):
    TEXT = "text"
    IMAGE = "image"


def _now() -> float:
    return time.time()


def new_id() -> str:
    # Qdrant requires point ids to be an unsigned int or a dashed UUID string.
    return str(uuid.uuid4())


class Memory(BaseModel):
    """A single semantic memory item."""

    id: str = Field(default_factory=new_id)
    text: str
    kind: MemoryKind = MemoryKind.TEXT

    # Provenance / versioning (drives conflict resolution)
    device_id: str
    origin: str = "edge"  # "edge" | "cloud"
    version: int = 1
    created_at: float = Field(default_factory=_now)
    updated_at: float = Field(default_factory=_now)

    # Policy-derived classification
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    importance: float = 0.5
    sync_state: SyncState = SyncState.LOCAL_ONLY
    policy_reason: str = ""

    # Usage signals (feed LRU eviction + importance)
    access_count: int = 0
    last_accessed_at: float = Field(default_factory=_now)

    # Soft delete so deletions propagate through sync (tombstone)
    deleted: bool = False

    # Free-form domain metadata (e.g. machine_id, site, author)
    metadata: dict[str, str] = Field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        """Serialise for storage in a Qdrant point payload."""
        return self.model_dump(exclude={"id"})

    @classmethod
    def from_payload(cls, point_id: str, payload: dict[str, object]) -> Memory:
        return cls(id=point_id, **payload)  # type: ignore[arg-type]

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed_at = _now()


class SearchHit(BaseModel):
    """A scored search result returned to the UI."""

    memory: Memory
    score: float
    source_shard: str  # "mutable" | "immutable" -> where the hit came from


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    latency_ms: float
    query: str
    used_hybrid: bool


class SyncStatus(BaseModel):
    """Snapshot of sync health for the dashboard."""

    device_id: str
    online: bool
    pending: int
    synced: int
    local_only: int
    conflicts: int
    failed: int
    last_sync_at: float | None = None
    last_error: str | None = None
