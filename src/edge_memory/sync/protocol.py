"""Wire protocol shared by the edge sync worker and the cloud service.

Delta-based sync (default): the edge pushes changed memories and pulls memories updated
since its last cursor. This is more robust to demo than raw snapshot transfer, while still
using Qdrant Server as the authoritative cloud brain. (Snapshot-based sync via
``EdgeShard.unpack_snapshot`` remains available as an alternative; see README.)
"""

from __future__ import annotations

from pydantic import BaseModel

from edge_memory.models import Memory


class PushRequest(BaseModel):
    device_id: str
    memories: list[Memory]


class ConflictReport(BaseModel):
    memory_id: str
    resolution: str  # "kept_incoming" | "kept_existing" | "merged"
    detail: str
    winner: Memory


class PushResponse(BaseModel):
    accepted: list[str]          # memory ids stored as-is
    conflicts: list[ConflictReport]
    rejected: list[str]          # ids refused (e.g. malformed)


class PullRequest(BaseModel):
    device_id: str
    since: float = 0.0           # cursor: only memories updated after this ts
    limit: int = 500
    # Optional domain filter for "push down relevant subset" (e.g. site/role).
    filters: dict[str, str] = {}


class PullResponse(BaseModel):
    memories: list[Memory]
    cursor: float                # advance the edge cursor to this value
