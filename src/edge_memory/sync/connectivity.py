"""Connectivity monitor.

Tracks whether the cloud brain is reachable. Supports a manual override so a demo can
simulate going offline/online without physically toggling the network — the UI exposes
this as a switch, and there is also a real health probe.
"""

from __future__ import annotations

import httpx

from edge_memory.logging import get_logger

log = get_logger(__name__)


class ConnectivityMonitor:
    def __init__(self, cloud_url: str, timeout: float = 2.0) -> None:
        self._cloud_url = cloud_url.rstrip("/")
        self._timeout = timeout
        self._forced_offline = False
        self._last_online: bool = False

    @property
    def forced_offline(self) -> bool:
        return self._forced_offline

    def force_offline(self, value: bool) -> None:
        """Demo switch: pretend the network is down regardless of reality."""
        self._forced_offline = value
        log.info("connectivity override", forced_offline=value)

    async def is_online(self) -> bool:
        if self._forced_offline:
            self._last_online = False
            return False
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self._cloud_url}/health")
                self._last_online = resp.status_code == 200
        except (httpx.HTTPError, OSError):
            self._last_online = False
        return self._last_online

    @property
    def last_online(self) -> bool:
        return self._last_online
