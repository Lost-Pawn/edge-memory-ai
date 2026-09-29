"""Repository over Qdrant Server (the authoritative global memory collection).

Uses the standard ``qdrant-client``. Stores the same named vectors as the edge shards
(dense ``text`` + sparse ``bm25``) so that dense/sparse semantics match end to end.
"""

from __future__ import annotations

from edge_memory.config import CloudSettings
from edge_memory.embeddings.base import EmbeddingProvider
from edge_memory.logging import get_logger
from edge_memory.models import Memory

log = get_logger(__name__)

DENSE_VECTOR = "text"
SPARSE_VECTOR = "bm25"


class QdrantRepo:
    def __init__(self, settings: CloudSettings, embedder: EmbeddingProvider) -> None:
        from qdrant_client import QdrantClient

        self._settings = settings
        self._embedder = embedder
        self._client = QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key or None
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from qdrant_client import models

        name = self._settings.collection
        if self._client.collection_exists(name):
            return
        log.info("creating cloud collection", collection=name)
        self._client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=self._settings.dense_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )
        self._client.create_payload_index(
            name, "updated_at", field_schema="float"
        )

    def upsert(self, memory: Memory) -> None:
        from qdrant_client import models

        dense = self._embedder.embed_text([memory.text])[0]
        sparse = self._embedder.embed_sparse([memory.text])[0]
        point = models.PointStruct(
            id=memory.id,
            vector={
                DENSE_VECTOR: dense,
                SPARSE_VECTOR: models.SparseVector(
                    indices=sparse.indices, values=sparse.values
                ),
            },
            payload=memory.to_payload(),
        )
        self._client.upsert(self._settings.collection, [point])

    def get(self, memory_id: str) -> Memory | None:
        records = self._client.retrieve(
            self._settings.collection, ids=[memory_id], with_payload=True
        )
        if not records:
            return None
        return Memory.from_payload(str(records[0].id), records[0].payload or {})

    def nearest(self, text: str, limit: int = 1) -> list[tuple[Memory, float]]:
        dense = self._embedder.embed_text([text])[0]
        results = self._client.query_points(
            self._settings.collection,
            query=dense,
            using=DENSE_VECTOR,
            limit=limit,
            with_payload=True,
        ).points
        return [
            (Memory.from_payload(str(p.id), p.payload or {}), float(p.score))
            for p in results
        ]

    def updated_since(
        self, since: float, *, exclude_device: str, limit: int = 500
    ) -> tuple[list[Memory], float]:
        from qdrant_client import models

        flt = models.Filter(
            must=[
                models.FieldCondition(
                    key="updated_at", range=models.Range(gt=since)
                )
            ],
            must_not=[
                models.FieldCondition(
                    key="device_id", match=models.MatchValue(value=exclude_device)
                )
            ],
        )
        records, _ = self._client.scroll(
            self._settings.collection,
            scroll_filter=flt,
            limit=limit,
            with_payload=True,
        )
        memories = [
            Memory.from_payload(str(r.id), r.payload or {}) for r in records
        ]
        cursor = max((m.updated_at for m in memories), default=since)
        return memories, cursor

    def count(self) -> int:
        return self._client.count(self._settings.collection, exact=True).count
