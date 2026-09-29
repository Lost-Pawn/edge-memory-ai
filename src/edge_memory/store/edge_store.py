"""Qdrant Edge store using the documented dual-shard pattern.

Two in-process shards live on the device:

* ``mutable``   — every local write lands here first.
* ``immutable`` — a mirror of cloud knowledge, refreshed from the cloud brain.

A search runs against **both** shards, results are de-duplicated by id, and dense +
sparse results are fused with Reciprocal Rank Fusion (RRF) in Python. Doing fusion
ourselves keeps hybrid search correct and independent of any single SDK's fusion API.

SDK note: the exact Qdrant Edge Python classes (``Point``, ``QueryRequest`` …) are
imported lazily and confined to the private ``_backend`` helpers below. If your installed
``qdrant-edge-py`` differs, those few helpers are the only place to adjust.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from pathlib import Path

from edge_memory.config import EdgeSettings
from edge_memory.embeddings.base import EmbeddingProvider, SparseVector
from edge_memory.exceptions import MemoryNotFoundError
from edge_memory.logging import get_logger
from edge_memory.models import Memory, SearchHit

log = get_logger(__name__)

DENSE_VECTOR = "text"
SPARSE_VECTOR = "bm25"
RRF_K = 60  # standard RRF damping constant


class EdgeStore:
    """Embedded vector memory for a single device."""

    def __init__(self, settings: EdgeSettings, embedder: EmbeddingProvider) -> None:
        self._settings = settings
        self._embedder = embedder
        self._lock = threading.RLock()
        self._mutable = self._open_shard(settings.shard_dir / "mutable")
        self._immutable = self._open_shard(settings.shard_dir / "immutable")

    # ── shard lifecycle ──────────────────────────────────────────────────────
    def _open_shard(self, path: Path):  # type: ignore[no-untyped-def]
        from qdrant_edge import (
            Distance,
            EdgeConfig,
            EdgeShard,
            EdgeSparseVectorParams,
            EdgeVectorParams,
            Modifier,
        )

        path.mkdir(parents=True, exist_ok=True)
        if any(path.iterdir()):
            return EdgeShard.load(str(path))

        config = EdgeConfig(
            vectors={
                DENSE_VECTOR: EdgeVectorParams(
                    size=self._embedder.dense_dim, distance=Distance.Cosine
                )
            },
            sparse_vectors={SPARSE_VECTOR: EdgeSparseVectorParams(modifier=Modifier.Idf)},
        )
        log.info("creating edge shard", path=str(path))
        return EdgeShard.create(str(path), config)

    def close(self) -> None:
        for shard in (self._mutable, self._immutable):
            with _suppress():
                shard.close()

    # ── writes ───────────────────────────────────────────────────────────────
    def upsert(self, memory: Memory, *, into: str = "mutable") -> Memory:
        """Embed and store a memory into the chosen shard."""
        dense = self._embedder.embed_text([memory.text])[0]
        sparse = self._embedder.embed_sparse([memory.text])[0]
        point = self._to_point(memory, dense, sparse)
        shard = self._mutable if into == "mutable" else self._immutable
        with self._lock:
            self._apply_upsert(shard, point)
        return memory

    def upsert_many(self, memories: list[Memory], *, into: str = "immutable") -> int:
        if not memories:
            return 0
        texts = [m.text for m in memories]
        dense = self._embedder.embed_text(texts)
        sparse = self._embedder.embed_sparse(texts)
        points = [self._to_point(m, dense[i], sparse[i]) for i, m in enumerate(memories)]
        shard = self._mutable if into == "mutable" else self._immutable
        with self._lock:
            for point in points:
                self._apply_upsert(shard, point)
        return len(points)

    def tombstone(self, memory_id: str) -> Memory:
        """Soft-delete so the deletion can propagate through sync."""
        memory = self.get(memory_id)
        memory.deleted = True
        memory.version += 1
        return self.upsert(memory)

    def hard_delete(self, memory_id: str, *, shard_name: str = "mutable") -> None:
        from qdrant_edge import UpdateOperation

        shard = self._mutable if shard_name == "mutable" else self._immutable
        with self._lock:
            shard.update(UpdateOperation.delete_points([memory_id]))

    # ── reads ────────────────────────────────────────────────────────────────
    def get(self, memory_id: str) -> Memory:
        for shard in (self._mutable, self._immutable):
            records = shard.retrieve([memory_id], with_payload=True, with_vector=False)
            if records:
                return self._record_to_memory(records[0])
        raise MemoryNotFoundError(memory_id)

    def list_memories(
        self, *, limit: int = 100, include_deleted: bool = False
    ) -> list[Memory]:
        out: dict[str, Memory] = {}
        for shard in (self._mutable, self._immutable):
            records, _ = shard.scroll(self._scroll_request(limit))
            for rec in records:
                mem = self._record_to_memory(rec)
                if mem.deleted and not include_deleted:
                    continue
                out[mem.id] = mem
        return sorted(out.values(), key=lambda m: m.updated_at, reverse=True)[:limit]

    def count(self, *, include_deleted: bool = False) -> int:
        return len(self.list_memories(limit=10**6, include_deleted=include_deleted))

    def hybrid_search(
        self, query: str, *, limit: int = 10, use_hybrid: bool = True
    ) -> list[SearchHit]:
        """Dense + sparse search across both shards, fused with RRF."""
        dense_q = self._embedder.embed_text([query])[0]
        ranked: list[tuple[str, float, str, Memory]] = []

        dense_lists: list[list[tuple[str, Memory]]] = []
        sparse_lists: list[list[tuple[str, Memory]]] = []
        for name, shard in (("mutable", self._mutable), ("immutable", self._immutable)):
            dense_lists.append(
                [
                    (m.id, m)
                    for m in self._dense_results(shard, dense_q, limit * 5, name)
                ]
            )
            if use_hybrid:
                sparse_q = self._embedder.embed_sparse([query])[0]
                sparse_lists.append(
                    [
                        (m.id, m)
                        for m in self._sparse_results(shard, sparse_q, limit * 5, name)
                    ]
                )

        fused = self._rrf(dense_lists + sparse_lists)
        for mem, score, src in fused[:limit]:
            ranked.append((mem.id, score, src, mem))
        return [
            SearchHit(memory=mem, score=score, source_shard=src)
            for (_id, score, src, mem) in ranked
        ]

    # ── RRF fusion ───────────────────────────────────────────────────────────
    @staticmethod
    def _rrf(
        ranked_lists: list[list[tuple[str, Memory]]],
    ) -> list[tuple[Memory, float, str]]:
        scores: dict[str, float] = defaultdict(float)
        memories: dict[str, Memory] = {}
        source: dict[str, str] = {}
        for lst in ranked_lists:
            for rank, (mid, mem) in enumerate(lst):
                scores[mid] += 1.0 / (RRF_K + rank + 1)
                memories[mid] = mem
                source.setdefault(mid, mem.metadata.get("_src", "mutable"))
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(memories[mid], sc, source.get(mid, "mutable")) for mid, sc in ordered]

    # ── private backend adapters (SDK-specific bits live here) ───────────────
    def _to_point(self, memory: Memory, dense: list[float], sparse: SparseVector):  # type: ignore[no-untyped-def]
        from qdrant_edge import Point
        from qdrant_edge import SparseVector as EdgeSparse

        sparse_vec = EdgeSparse(indices=sparse.indices, values=sparse.values)
        return Point(
            id=memory.id,
            vector={DENSE_VECTOR: dense, SPARSE_VECTOR: sparse_vec},
            payload=memory.to_payload(),
        )

    def _apply_upsert(self, shard, point) -> None:  # type: ignore[no-untyped-def]
        from qdrant_edge import UpdateOperation

        shard.update(UpdateOperation.upsert_points([point]))

    def _dense_results(self, shard, vector, limit, src):  # type: ignore[no-untyped-def]
        from qdrant_edge import Query, QueryRequest

        req = QueryRequest(
            query=Query.Nearest(vector, using=DENSE_VECTOR),
            limit=limit,
            with_payload=True,
        )
        return [self._record_to_memory(r, src) for r in shard.query(req)]

    def _sparse_results(self, shard, sparse: SparseVector, limit, src):  # type: ignore[no-untyped-def]
        from qdrant_edge import Query, QueryRequest
        from qdrant_edge import SparseVector as EdgeSparse

        req = QueryRequest(
            query=Query.Nearest(
                EdgeSparse(indices=sparse.indices, values=sparse.values),
                using=SPARSE_VECTOR,
            ),
            limit=limit,
            with_payload=True,
        )
        return [self._record_to_memory(r, src) for r in shard.query(req)]

    def _scroll_request(self, limit: int):  # type: ignore[no-untyped-def]
        from qdrant_edge import ScrollRequest

        return ScrollRequest(limit=limit, with_payload=True)

    @staticmethod
    def _record_to_memory(record, src: str = "mutable") -> Memory:  # type: ignore[no-untyped-def]
        payload = dict(record.payload or {})
        mem = Memory.from_payload(str(record.id), payload)
        mem.metadata = {**mem.metadata, "_src": src}
        return mem


class _suppress:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_: object) -> bool:
        return True
