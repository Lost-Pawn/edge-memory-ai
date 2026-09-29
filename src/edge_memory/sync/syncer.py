"""The sync worker: drains the outbox (push) and pulls cloud knowledge (pull).

Runs as a periodic asyncio task. When offline, pushes are no-ops (work stays queued in
the outbox) and the app keeps functioning fully on local memory. When connectivity
returns, the queue drains and fresh cloud knowledge flows into the immutable shard.
"""

from __future__ import annotations

import time

from edge_memory.config import EdgeSettings
from edge_memory.events import EventBus, EventType
from edge_memory.exceptions import OfflineError
from edge_memory.logging import get_logger
from edge_memory.models import Memory, SyncState
from edge_memory.store import EdgeStore, Outbox
from edge_memory.store.outbox import OpType
from edge_memory.sync.cloud_client import CloudClient
from edge_memory.sync.connectivity import ConnectivityMonitor
from edge_memory.sync.protocol import PullRequest, PushRequest

log = get_logger(__name__)


class Syncer:
    def __init__(
        self,
        settings: EdgeSettings,
        store: EdgeStore,
        outbox: Outbox,
        connectivity: ConnectivityMonitor,
        client: CloudClient,
        events: EventBus,
    ) -> None:
        self._settings = settings
        self._store = store
        self._outbox = outbox
        self._conn = connectivity
        self._client = client
        self._events = events
        self._cursor: float = 0.0
        self.last_sync_at: float | None = None
        self.last_error: str | None = None

    async def sync_once(self) -> dict[str, int]:
        """Run one push+pull cycle if online. Returns a small stats dict."""
        if not await self._conn.is_online():
            self._events.emit(EventType.CONNECTIVITY, "offline: sync skipped, work queued")
            return {"pushed": 0, "pulled": 0, "skipped": 1}

        pushed = await self._push()
        pulled = await self._pull()
        self.last_sync_at = time.time()
        self.last_error = None
        return {"pushed": pushed, "pulled": pulled, "skipped": 0}

    async def _push(self) -> int:
        ops = self._outbox.pending(limit=200)
        if not ops:
            return 0
        memories = [Memory.model_validate(op.payload) for op in ops]
        try:
            resp = await self._client.push(
                PushRequest(device_id=self._settings.device_id, memories=memories)
            )
        except OfflineError as exc:
            self.last_error = str(exc)
            self._events.emit(EventType.CONNECTIVITY, "push failed: offline")
            return 0

        accepted = set(resp.accepted) | {c.memory_id for c in resp.conflicts}
        for op in ops:
            if op.memory_id in accepted:
                self._outbox.mark_synced(op.op_id)
                self._mark_local_state(op.memory_id, SyncState.SYNCED)
            else:
                self._outbox.mark_failed(op.op_id, "rejected by cloud")

        for conflict in resp.conflicts:
            self._store.upsert(conflict.winner)  # adopt the resolved winner locally
            self._events.emit(
                EventType.CONFLICT,
                f"conflict resolved ({conflict.resolution}) for {conflict.memory_id}",
                resolution=conflict.resolution,
                detail=conflict.detail,
            )

        self._events.emit(
            EventType.SYNC_PUSH,
            f"pushed {len(accepted)} memories to cloud",
            count=len(accepted),
            conflicts=len(resp.conflicts),
        )
        return len(accepted)

    async def _pull(self) -> int:
        try:
            resp = await self._client.pull(
                PullRequest(
                    device_id=self._settings.device_id,
                    since=self._cursor,
                    filters={},
                )
            )
        except OfflineError as exc:
            self.last_error = str(exc)
            return 0

        # Store cloud knowledge in the immutable mirror shard.
        fresh = [m for m in resp.memories if m.device_id != self._settings.device_id]
        count = self._store.upsert_many(fresh, into="immutable")
        self._cursor = max(self._cursor, resp.cursor)
        if count:
            self._events.emit(
                EventType.SYNC_PULL,
                f"pulled {count} memories from cloud brain",
                count=count,
            )
        return count

    def _mark_local_state(self, memory_id: str, state: SyncState) -> None:
        try:
            memory = self._store.get(memory_id)
            memory.sync_state = state
            self._store.upsert(memory)
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("failed to update local sync state", id=memory_id, error=str(exc))
