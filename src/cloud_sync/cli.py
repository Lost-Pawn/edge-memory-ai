"""Cloud sync service CLI."""

from __future__ import annotations

import typer
import uvicorn

from edge_memory.config import get_cloud_settings

app = typer.Typer(help="EdgeMemoryAi cloud brain CLI", no_args_is_help=True)


@app.command()
def serve() -> None:
    """Run the cloud sync service."""
    settings = get_cloud_settings()
    uvicorn.run(
        "cloud_sync.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
    )


if __name__ == "__main__":
    app()
