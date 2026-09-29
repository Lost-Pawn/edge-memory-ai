"""Domain exceptions."""

from __future__ import annotations


class EdgeMemoryError(Exception):
    """Base error for the edge memory platform."""


class ModelNotFoundError(EdgeMemoryError):
    """A required model file/weight is missing and downloads are disabled."""


class MemoryNotFoundError(EdgeMemoryError):
    """A requested memory id does not exist."""


class SyncError(EdgeMemoryError):
    """A sync operation failed (network, server, or conflict handling)."""


class OfflineError(SyncError):
    """The cloud is unreachable; the operation was queued instead."""
