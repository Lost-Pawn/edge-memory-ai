"""Edge device FastAPI application.

Wires a :class:`MemoryService` into HTTP + WebSocket endpoints and runs a background sync
loop. The loop periodically drains the outbox and pulls cloud knowledge; when offline it
is a cheap no-op, so the device keeps serving searches and ingests either way.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from edge_memory.api.schemas import (
    AskRequest,
    IngestRequest,
    OfflineRequest,
    SearchRequest,
    UpdateRequest,
)
from edge_memory.config import get_edge_settings
from edge_memory.exceptions import MemoryNotFoundError
from edge_memory.logging import configure_logging, get_logger
from edge_memory.service import MemoryService

log = get_logger(__name__)


async def _sync_loop(service: MemoryService) -> None:
    interval = service.settings.sync_interval_seconds
    while True:
        try:
            await service.sync_now()
        except Exception as exc:  # pragma: no cover - keep the loop alive
            log.warning("sync loop error", error=str(exc))
        await asyncio.sleep(interval)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_edge_settings()
    service = MemoryService(settings)
    app.state.service = service
    task = asyncio.create_task(_sync_loop(service))
    log.info("edge device up", device_id=settings.device_id, port=settings.api_port)
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        service.close()


app = FastAPI(title="EdgeMemoryAi — Edge Device", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _svc(app: FastAPI) -> MemoryService:
    return app.state.service  # type: ignore[no-any-return]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "device_id": _svc(app).settings.device_id}


@app.post("/memories")
async def ingest(req: IngestRequest) -> dict[str, object]:
    memory = _svc(app).ingest(req.text, req.metadata)
    return memory.model_dump()


@app.get("/memories")
async def list_memories(limit: int = 100, include_deleted: bool = False) -> list[dict]:
    return [
        m.model_dump()
        for m in _svc(app).store.list_memories(
            limit=limit, include_deleted=include_deleted
        )
    ]


@app.get("/memories/{memory_id}")
async def get_memory(memory_id: str) -> dict:
    try:
        return _svc(app).store.get(memory_id).model_dump()
    except MemoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail="memory not found") from exc


@app.patch("/memories/{memory_id}")
async def update_memory(memory_id: str, req: UpdateRequest) -> dict:
    try:
        return _svc(app).update(memory_id, req.text).model_dump()
    except MemoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail="memory not found") from exc


@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str) -> dict:
    try:
        return _svc(app).delete(memory_id).model_dump()
    except MemoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail="memory not found") from exc


@app.post("/search")
async def search(req: SearchRequest) -> dict:
    return _svc(app).search(req.query, limit=req.limit, hybrid=req.hybrid).model_dump()


@app.post("/ask")
async def ask(req: AskRequest) -> dict:
    return _svc(app).ask(req.question, top_k=req.top_k).model_dump()


@app.get("/sync/status")
async def sync_status() -> dict:
    service = _svc(app)
    online = await service.connectivity.is_online()
    return service.status(online).model_dump()


@app.post("/sync/now")
async def sync_now() -> dict[str, int]:
    return await _svc(app).sync_now()


@app.post("/sync/offline")
async def set_offline(req: OfflineRequest) -> dict[str, bool]:
    _svc(app).set_offline(req.offline)
    return {"forced_offline": req.offline}


@app.get("/activity")
async def activity(limit: int = 100) -> list[dict]:
    return [e.model_dump() for e in _svc(app).events.recent(limit)]


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    service = _svc(app)
    queue = service.events.subscribe()
    try:
        for event in service.events.recent(50)[::-1]:
            await websocket.send_json(event.model_dump())
        while True:
            event = await queue.get()
            await websocket.send_json(event.model_dump())
    except WebSocketDisconnect:
        pass
    finally:
        service.events.unsubscribe(queue)
