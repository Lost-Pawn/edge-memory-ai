"""Durable SQLite outbox for the sync queue.

Every memory the policy engine marks for sync is enqueued here as an operation. The
sync worker drains the outbox when the cloud is reachable, retrying with backoff. Because
it is on-disk, queued work survives process restarts and long offline periods.
"""

from __future__ import annotations

import json
import sqlite3
import time
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

_SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
    op_id      TEXT PRIMARY KEY,
    memory_id  TEXT NOT NULL,
    op_type    TEXT NOT NULL,
    payload    TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending',
    attempts   INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox(status);
"""


class OpType(StrEnum):
    UPSERT = "upsert"
    DELETE = "delete"


class OpStatus(StrEnum):
    PENDING = "pending"
    SYNCED = "synced"
    FAILED = "failed"


class OutboxOp(BaseModel):
    op_id: str
    memory_id: str
    op_type: OpType
    payload: dict[str, object]
    status: OpStatus
    attempts: int
    created_at: float
    updated_at: float
    last_error: str | None = None


class Outbox:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def enqueue(
        self, op_id: str, memory_id: str, op_type: OpType, payload: dict[str, object]
    ) -> None:
        now = time.time()
        self._conn.execute(
            """INSERT INTO outbox (op_id, memory_id, op_type, payload, status,
                                   attempts, created_at, updated_at)
               VALUES (?, ?, ?, ?, 'pending', 0, ?, ?)
               ON CONFLICT(op_id) DO UPDATE SET
                   payload=excluded.payload, status='pending', updated_at=excluded.updated_at
            """,
            (op_id, memory_id, op_type, json.dumps(payload), now, now),
        )
        self._conn.commit()

    def pending(self, limit: int = 100) -> list[OutboxOp]:
        rows = self._conn.execute(
            "SELECT * FROM outbox WHERE status = 'pending' ORDER BY created_at LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row(r) for r in rows]

    def mark_synced(self, op_id: str) -> None:
        self._conn.execute(
            "UPDATE outbox SET status='synced', updated_at=? WHERE op_id=?",
            (time.time(), op_id),
        )
        self._conn.commit()

    def mark_failed(self, op_id: str, error: str) -> None:
        self._conn.execute(
            """UPDATE outbox
               SET status=CASE WHEN attempts + 1 >= 5 THEN 'failed' ELSE 'pending' END,
                   attempts=attempts + 1, last_error=?, updated_at=?
               WHERE op_id=?""",
            (error, time.time(), op_id),
        )
        self._conn.commit()

    def counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM outbox GROUP BY status"
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row(r: sqlite3.Row) -> OutboxOp:
        return OutboxOp(
            op_id=r["op_id"],
            memory_id=r["memory_id"],
            op_type=OpType(r["op_type"]),
            payload=json.loads(r["payload"]),
            status=OpStatus(r["status"]),
            attempts=r["attempts"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            last_error=r["last_error"],
        )
