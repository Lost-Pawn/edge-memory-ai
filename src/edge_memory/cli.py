"""Command-line interface for the edge device.

Model downloads live ONLY behind the explicit ``download-models`` and ``download-llm``
commands. No other command fetches weights.
"""

from __future__ import annotations

import asyncio

import typer

from edge_memory.config import get_edge_settings
from edge_memory.logging import configure_logging

app = typer.Typer(help="EdgeMemoryAi edge device CLI", no_args_is_help=True)


@app.command("download-models")
def download_models() -> None:
    """Explicitly download + cache the FastEmbed embedding models (needs network)."""
    from edge_memory.embeddings.fastembed_provider import download_models as dl

    configure_logging()
    dl(get_edge_settings())
    typer.echo("embedding models cached")


@app.command("download-llm")
def download_llm() -> None:
    """Explicitly download the local GGUF LLM (needs network)."""
    from edge_memory.llm.llama_cpp_provider import download_llm as dl

    configure_logging()
    dl(get_edge_settings())
    typer.echo("LLM downloaded")


@app.command()
def ingest(text: str) -> None:
    """Store a memory locally (offline)."""
    from edge_memory.service import MemoryService

    configure_logging()
    service = MemoryService(get_edge_settings())
    memory = service.ingest(text)
    typer.echo(f"{memory.id}  [{memory.sync_state}]  {memory.policy_reason}")
    service.close()


@app.command()
def search(query: str, limit: int = 5, hybrid: bool = True) -> None:
    """Hybrid search over local memory (offline)."""
    from edge_memory.service import MemoryService

    configure_logging()
    service = MemoryService(get_edge_settings())
    resp = service.search(query, limit=limit, hybrid=hybrid)
    typer.echo(f"{len(resp.hits)} hits in {resp.latency_ms}ms")
    for hit in resp.hits:
        typer.echo(f"  {hit.score:.4f}  [{hit.source_shard}]  {hit.memory.text[:80]}")
    service.close()


@app.command()
def ask(question: str, top_k: int = 5) -> None:
    """Ask a RAG question over local memory (offline)."""
    from edge_memory.service import MemoryService

    configure_logging()
    service = MemoryService(get_edge_settings())
    answer = service.ask(question, top_k=top_k)
    typer.echo(answer.answer)
    service.close()


@app.command()
def sync() -> None:
    """Run one sync cycle now."""
    from edge_memory.service import MemoryService

    configure_logging()
    service = MemoryService(get_edge_settings())
    stats = asyncio.run(service.sync_now())
    typer.echo(str(stats))
    service.close()


if __name__ == "__main__":
    app()
