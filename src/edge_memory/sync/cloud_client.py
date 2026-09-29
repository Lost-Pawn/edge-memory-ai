"""HTTP client the edge uses to talk to the cloud sync service."""

from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from edge_memory.exceptions import OfflineError
from edge_memory.sync.protocol import (
    PullRequest,
    PullResponse,
    PushRequest,
    PushResponse,
)


class CloudClient:
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    async def push(self, request: PushRequest) -> PushResponse:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base}/sync/push", json=request.model_dump()
                )
                resp.raise_for_status()
                return PushResponse.model_validate(resp.json())
        except (httpx.HTTPError, OSError) as exc:
            raise OfflineError(str(exc)) from exc

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    async def pull(self, request: PullRequest) -> PullResponse:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base}/sync/pull", json=request.model_dump()
                )
                resp.raise_for_status()
                return PullResponse.model_validate(resp.json())
        except (httpx.HTTPError, OSError) as exc:
            raise OfflineError(str(exc)) from exc
