"""The sync-policy engine.

For every ingested memory it decides:

1. **Sensitivity** — PII is forced ``local_only`` and never syncs (privacy guarantee).
2. **Importance** — a 0..1 score from length, explicit hints, and access frequency.
3. **Sync state** — important, non-PII memories are queued for the cloud brain.

It also enforces a **storage budget**: when the mutable shard exceeds the budget, the
least-valuable already-synced memories are evicted (LRU-by-value), because their content
is safe on the cloud and can be pulled back on demand.

Every decision carries a human-readable ``reason`` string that the UI shows verbatim, so
the "dynamically decide what stays local" behaviour is visible, not a black box.
"""

from __future__ import annotations

from pydantic import BaseModel

from edge_memory.config import EdgeSettings
from edge_memory.models import Memory, Sensitivity, SyncState
from edge_memory.policy import pii

_IMPORTANCE_HINTS = ("error", "failure", "fix", "safety", "critical", "warning", "spec")


class PolicyDecision(BaseModel):
    sensitivity: Sensitivity
    importance: float
    sync_state: SyncState
    reason: str


class SyncPolicyEngine:
    def __init__(self, settings: EdgeSettings) -> None:
        self._settings = settings

    def classify(self, memory: Memory) -> PolicyDecision:
        """Decide sensitivity, importance, and sync state for a new/updated memory."""
        categories = pii.detect(memory.text)
        importance = self._score_importance(memory)

        if categories:
            return PolicyDecision(
                sensitivity=Sensitivity.PII,
                importance=importance,
                sync_state=SyncState.LOCAL_ONLY,
                reason=f"local-only: contains PII ({', '.join(categories)})",
            )

        # Caller may pin sensitivity via metadata (e.g. manual "public").
        declared = memory.metadata.get("sensitivity")
        sensitivity = (
            Sensitivity(declared)
            if declared in Sensitivity.__members__.values()
            else Sensitivity.INTERNAL
        )

        if importance >= self._settings.importance_sync_threshold:
            return PolicyDecision(
                sensitivity=sensitivity,
                importance=importance,
                sync_state=SyncState.PENDING,
                reason=(
                    f"queued for sync: importance {importance:.2f} "
                    f">= {self._settings.importance_sync_threshold:.2f}"
                ),
            )

        return PolicyDecision(
            sensitivity=sensitivity,
            importance=importance,
            sync_state=SyncState.LOCAL_ONLY,
            reason=(
                f"local-only: importance {importance:.2f} below sync threshold "
                f"{self._settings.importance_sync_threshold:.2f}"
            ),
        )

    def apply(self, memory: Memory) -> Memory:
        decision = self.classify(memory)
        memory.sensitivity = decision.sensitivity
        memory.importance = decision.importance
        memory.sync_state = decision.sync_state
        memory.policy_reason = decision.reason
        return memory

    def _score_importance(self, memory: Memory) -> float:
        text = memory.text.lower()
        score = 0.3
        score += min(len(memory.text) / 800.0, 0.3)  # richer text = more useful
        if any(h in text for h in _IMPORTANCE_HINTS):
            score += 0.3
        score += min(memory.access_count * 0.05, 0.2)  # frequently used = valuable
        try:
            score = max(score, float(memory.metadata.get("importance", score)))
        except (TypeError, ValueError):
            pass
        return round(min(score, 1.0), 3)

    # ── eviction ──────────────────────────────────────────────────────────────
    def select_evictions(self, memories: list[Memory]) -> list[Memory]:
        """Pick already-synced, low-value memories to drop when over budget."""
        budget = self._settings.storage_budget
        if len(memories) <= budget:
            return []
        evictable = [
            m
            for m in memories
            if m.sync_state == SyncState.SYNCED and m.sensitivity != Sensitivity.PII
        ]
        # Lowest value first: importance, then least-recently accessed.
        evictable.sort(key=lambda m: (m.importance, m.last_accessed_at))
        overflow = len(memories) - budget
        return evictable[:overflow]
