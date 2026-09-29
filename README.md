# EdgeMemoryAi — AI-Powered Edge Memory & Intelligence Platform

Offline-first edge intelligence built on **Qdrant Edge**. Each device keeps its own
searchable semantic memory, runs hybrid (dense + sparse) vector search with no network,
decides on its own what stays local vs. what syncs, and reconciles with a central cloud
brain (Qdrant Server) when connectivity returns — resolving conflicts and de-duplicating
along the way. Built for Qdrant Problem Statement 03.

## Why it meets the brief

| Requirement | How it is met |
|---|---|
| Searchable semantic memory on device | Embedded `qdrant-edge-py` dual-shard store, in-process |
| Low-latency hybrid search offline | Dense (CLIP) + sparse (BM25) fused with RRF, all on-device |
| Fully offline | FastEmbed cached models (`local_files_only`) + in-process llama.cpp; no cloud calls |
| Dynamically decide local vs. sync | `SyncPolicyEngine`: PII → local-only, importance/usage → queue, budget → LRU evict |
| Intermittent connectivity | SQLite outbox + connectivity monitor; app never blocks on the network |
| Sync edge ↔ server | Delta push/pull to the cloud brain; work drains when back online |
| Evolving data + conflicts | Versioning, last-writer-wins with history, optional LLM merge, semantic dedupe |
| User-facing interface | React dashboard: memory browser, search scores, live sync status, activity log |
| Meaningful edge-to-cloud workflow | Device A learns offline → syncs → Device B benefits; cloud aggregates + pushes down |

## Architecture

```
 Edge device (per node)                         Cloud brain
 ┌───────────────────────────────┐             ┌──────────────────────────┐
 │ FastEmbed (CLIP dense + BM25)  │             │ cloud_sync (FastAPI)     │
 │ Qdrant Edge: mutable+immutable │  push/pull  │  Aggregator: conflict +  │
 │ SyncPolicyEngine  ┐            │ ◀─────────▶ │  dedupe                  │
 │ Outbox (SQLite)   ├─ Syncer    │             │ Qdrant Server (global)   │
 │ llama.cpp RAG     ┘            │             └──────────────────────────┘
 │ FastAPI + WebSocket ──────────────▶ React dashboard
 └───────────────────────────────┘
```

The dual-shard pattern follows Qdrant's own edge-to-cloud guidance: local writes land in
a **mutable** shard, cloud knowledge mirrors into an **immutable** shard, and search runs
across both with RRF fusion.

## Quick start (Docker, full stack)

```bash
cp .env.example .env
docker compose build
# One-time, EXPLICIT model download into ./models (nothing downloads automatically):
docker compose run --rm edge-a edge-memory download-models   # embeddings (~350MB)
docker compose run --rm edge-a edge-memory download-llm       # GGUF LLM (~2-4GB)
docker compose up -d
```

- Dashboard: http://localhost:5173
- Edge A API: http://localhost:8001/docs · Edge B: http://localhost:8002/docs
- Cloud brain: http://localhost:8000/docs · Qdrant: http://localhost:6333/dashboard

## Local dev (no Docker)

```bash
make install         # core + dev deps (no models)
make install-llm     # in-process llama.cpp
make download-models # EXPLICIT: embeddings + GGUF LLM
make qdrant          # Qdrant Server
make cloud           # cloud brain :8000
make edge            # edge A :8001
make edge-b          # edge B :8002
make frontend        # dashboard :5173
```

## Model policy

Nothing downloads at runtime. FastEmbed runs with `local_files_only=True`; the LLM loads a
local GGUF path. Weights are fetched only by the explicit `download-models` / `download-llm`
commands. Missing weights degrade gracefully (retrieval keeps working; RAG falls back to
extractive answers).

## Demo script

1. Device A: toggle **Simulate offline**. Ingest field notes → instant hybrid search.
2. Ingest a note containing an email → tagged `local-only (PII)`, never queued.
3. **Go online** → outbox drains, states flip `pending → synced`.
4. Device B pulls the synced knowledge → benefits from A's memory.
5. Edit the same fact on A and B while offline → reconnect → conflict resolved (LWW/merge).

## Tests & quality

```bash
make test   # pytest (unit logic runs without models or the SDK)
make lint   # ruff
make type   # mypy
```

## Layout

```
src/edge_memory/   embeddings · llm · store (edge_store, outbox) · policy · sync · rag · api · cli
src/cloud_sync/    qdrant_repo · aggregator · api
frontend/          React + Vite dashboard
deploy/            Dockerfiles · deploy script
tests/             unit tests (policy, pii, conflict, outbox, …)
```
