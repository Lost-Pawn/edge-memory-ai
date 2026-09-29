"""Conflict resolution + semantic dedupe.

Used by the cloud service when two devices edit the same logical memory while offline.

Strategy:

* **Same id, diverging versions** → Last-Writer-Wins on ``updated_at``. The losing copy
  is preserved in the winner's ``metadata['superseded']`` as history, and the event is
  reported as a conflict so the UI can surface it.
* **Optional LLM merge** → when a merge function is provided (a local LLM), genuinely
  conflicting text can be merged into a single reconciled memory instead of dropped.
* **Semantic dedupe** → near-duplicate memories (cosine >= threshold) are merged so the
  same fact learned on many devices does not bloat the global brain.
"""

from __future__ import annotations

from collections.abc import Callable

from edge_memory.models import Memory
from edge_memory.sync.protocol import ConflictReport

MergeFn = Callable[[Memory, Memory], str]


def resolve(
    incoming: Memory,
    existing: Memory | None,
    *,
    merge_fn: MergeFn | None = None,
) -> tuple[Memory, ConflictReport | None]:
    """Return the winning memory and, if there was a conflict, a report."""
    if existing is None:
        return incoming, None

    # No real divergence: incoming is a strictly newer version of the same lineage.
    if incoming.version > existing.version and _same_lineage(incoming, existing):
        return incoming, None

    # Concurrent edits: both changed relative to a shared base.
    if _concurrent(incoming, existing):
        if merge_fn is not None:
            merged_text = merge_fn(existing, incoming)
            winner = incoming.model_copy(
                update={
                    "text": merged_text,
                    "version": max(incoming.version, existing.version) + 1,
                }
            )
            return winner, ConflictReport(
                memory_id=incoming.id,
                resolution="merged",
                detail="LLM-merged two concurrent edits",
                winner=winner,
            )
        # Last-writer-wins, keep the loser as history.
        winner, loser = (
            (incoming, existing)
            if incoming.updated_at >= existing.updated_at
            else (existing, incoming)
        )
        winner = winner.model_copy(deep=True)
        winner.metadata = {**winner.metadata, "superseded": loser.text}
        return winner, ConflictReport(
            memory_id=incoming.id,
            resolution="kept_incoming"
            if winner is incoming
            else "kept_existing",
            detail="last-writer-wins on updated_at",
            winner=winner,
        )

    # Existing is newer / equal — keep it.
    return existing, None


def is_duplicate(score: float, threshold: float) -> bool:
    return score >= threshold


def _same_lineage(a: Memory, b: Memory) -> bool:
    return a.id == b.id


def _concurrent(incoming: Memory, existing: Memory) -> bool:
    """Both edited independently: neither version strictly descends from the other."""
    return (
        incoming.version == existing.version
        and incoming.text != existing.text
    ) or (
        incoming.device_id != existing.device_id
        and abs(incoming.version - existing.version) <= 1
        and incoming.text != existing.text
    )
