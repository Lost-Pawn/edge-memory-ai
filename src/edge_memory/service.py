"""MemoryService — the application core that wires every component together.

This is the single object the API layer talks to. It owns the store, outbox, policy
engine, RAG pipeline, syncer, connectivity monitor, and event bus, and exposes the
high-level use cases: ingest, search, ask, delete, sync, and status.
"""

from __future__ import annotations

import time
import uuid

from edge_memory.config import EdgeSettings
from edge_memory.embeddings import FastEmbedProvider
from edge_memory.events import EventBus, EventType
from edge_memory.llm.llama_cpp_provider import build_llm
from edge_memory.logging import get_logger
from edge_memory.models import Memory, SearchResponse, SyncState, SyncStatus
from edge_memory.policy import SyncPolicyEngine
from edge_memory.rag import RagAnswer, RagPipeline
from edge_memory.store import EdgeStore, Outbox
from edge_memory.store.outbox import OpType
from edge_memory.sync import ConnectivityMonitor, Syncer
from edge_memory.sync.cloud_client import CloudClient

log = get_logger(__name__)


class MemoryService:
    def __init__(self, settings: EdgeSettings) -> None:
        settings.ensure_dirs()
        self.settings = settings
        self.events = EventBus()
        self.embedder = FastEmbedProvider(settings)
        self.store = EdgeStore(settings, self.embedder)
        self.outbox = Outbox(settings.outbox_path)
        self.policy = SyncPolicyEngine(settings)
        self.llm = build_llm(settings)
        self.rag = RagPipeline(self.store, self.llm)
        self.connectivity = ConnectivityMonitor(settings.cloud_url)
        self.syncer = Syncer(
            settings,
            self.store,
            self.outbox,
            self.connectivity,
            CloudClient(settings.cloud_url),
            self.events,
        )

    # ── ingest ────────────────────────────────────────────────────────────────
    def ingest(self, text: str, metadata: dict[str, str] | None = None) -> Memory:
        memory = Memory(
            text=text,
            device_id=self.settings.device_id,
            metadata=metadata or {},
        )
        self.policy.apply(memory)
        self.store.upsert(memory)
        self.events.emit(
            EventType.MEMORY_ADDED,
            f"stored memory {memory.id[:8]}",
            id=memory.id,
            sync_state=memory.sync_state,
        )
        self.events.emit(EventType.POLICY_DECISION, memory.policy_reason, id=memory.id)

        if memory.sync_state == SyncState.PENDING:
            self.outbox.enqueue(
                op_id=uuid.uuid4().hex,
                memory_id=memory.id,
                op_type=OpType.UPSERT,
                payload=memory.model_dump(),
            )
        self._enforce_budget()
        return memory

    def update(self, memory_id: str, text: str) -> Memory:
        memory = self.store.get(memory_id)
        memory.text = text
        memory.version += 1
        memory.updated_at = time.time()
        self.policy.apply(memory)
        self.store.upsert(memory)
        if memory.sync_state == SyncState.PENDING:
            self.outbox.enqueue(
                uuid.uuid4().hex, memory.id, OpType.UPSERT, memory.model_dump()
            )
        self.events.emit(EventType.MEMORY_UPDATED, f"updated {memory_id[:8]}", id=memory_id)
        return memory

    def delete(self, memory_id: str) -> Memory:
        memory = self.store.tombstone(memory_id)
        if memory.sensitivity.value != "pii":
            self.outbox.enqueue(
                uuid.uuid4().hex, memory.id, OpType.DELETE, memory.model_dump()
            )
        self.events.emit(EventType.MEMORY_DELETED, f"deleted {memory_id[:8]}", id=memory_id)
        return memory

    # ── query ────────────────────────────────────────────────────────────────
    def search(self, query: str, *, limit: int = 10, hybrid: bool = True) -> SearchResponse:
        start = time.perf_counter()
        hits = self.store.hybrid_search(query, limit=limit, use_hybrid=hybrid)
        latency_ms = (time.perf_counter() - start) * 1000
        self.events.emit(
            EventType.SEARCH,
            f"search '{query[:40]}' -> {len(hits)} hits in {latency_ms:.1f}ms",
            latency_ms=latency_ms,
            hybrid=hybrid,
        )
        return SearchResponse(
            hits=hits, latency_ms=round(latency_ms, 2), query=query, used_hybrid=hybrid
        )

    def ask(self, question: str, *, top_k: int = 5) -> RagAnswer:
        return self.rag.ask(question, top_k=top_k)

    # ── sync ─────────────────────────────────────────────────────────────────
    async def sync_now(self) -> dict[str, int]:
        return await self.syncer.sync_once()

    def set_offline(self, offline: bool) -> None:
        self.connectivity.force_offline(offline)
        self.events.emit(
            EventType.CONNECTIVITY,
            "forced offline" if offline else "back online",
            offline=offline,
        )

    def status(self, online: bool) -> SyncStatus:
        memories = self.store.list_memories(limit=10**6, include_deleted=True)
        by_state: dict[str, int] = {}
        for m in memories:
            by_state[m.sync_state] = by_state.get(m.sync_state, 0) + 1
        return SyncStatus(
            device_id=self.settings.device_id,
            online=online,
            pending=self.outbox.counts().get("pending", 0),
            synced=by_state.get(SyncState.SYNCED, 0),
            local_only=by_state.get(SyncState.LOCAL_ONLY, 0),
            conflicts=by_state.get(SyncState.CONFLICT, 0),
            failed=self.outbox.counts().get("failed", 0),
            last_sync_at=self.syncer.last_sync_at,
            last_error=self.syncer.last_error,
        )

    def _enforce_budget(self) -> None:
        memories = self.store.list_memories(limit=10**6)
        for victim in self.policy.select_evictions(memories):
            self.store.hard_delete(victim.id)
            self.events.emit(
                EventType.MEMORY_EVICTED,
                f"evicted {victim.id[:8]} (synced, low value)",
                id=victim.id,
            )

    def close(self) -> None:
        self.store.close()
        self.outbox.close()
