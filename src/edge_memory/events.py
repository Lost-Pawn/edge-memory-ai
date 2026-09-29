"""In-process event bus for the activity feed and live dashboard updates.

Events are appended to a bounded ring buffer (persisted best-effort) and fanned out to
any subscribed asyncio queues, which the WebSocket endpoint drains to the UI.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from enum import StrEnum

from pydantic import BaseModel, Field


class EventType(StrEnum):
    MEMORY_ADDED = "memory_added"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_DELETED = "memory_deleted"
    MEMORY_EVICTED = "memory_evicted"
    SEARCH = "search"
    POLICY_DECISION = "policy_decision"
    SYNC_PUSH = "sync_push"
    SYNC_PULL = "sync_pull"
    CONFLICT = "conflict"
    CONNECTIVITY = "connectivity"
    ERROR = "error"


class Event(BaseModel):
    type: EventType
    message: str
    at: float = Field(default_factory=time.time)
    data: dict[str, object] = Field(default_factory=dict)


class EventBus:
    """Fan-out bus with a bounded history buffer."""

    def __init__(self, history: int = 500) -> None:
        self._history: deque[Event] = deque(maxlen=history)
        self._subscribers: set[asyncio.Queue[Event]] = set()

    def emit(self, type_: EventType, message: str, **data: object) -> Event:
        event = Event(type=type_, message=message, data=data)
        self._history.append(event)
        for queue in list(self._subscribers):
            # Never block the emitter; drop for slow consumers.
            with _suppress_full(queue):
                queue.put_nowait(event)
        return event

    def recent(self, limit: int = 100) -> list[Event]:
        items = list(self._history)
        return items[-limit:][::-1]

    def subscribe(self) -> asyncio.Queue[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        self._subscribers.discard(queue)


class _suppress_full:
    """Context manager that swallows asyncio.QueueFull."""

    def __init__(self, _queue: asyncio.Queue[Event]) -> None:
        self._queue = _queue

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, *_: object) -> bool:
        return exc_type is asyncio.QueueFull
