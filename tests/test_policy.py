from edge_memory.models import Memory, Sensitivity, SyncState
from edge_memory.policy import SyncPolicyEngine


def _mem(text: str, **kw) -> Memory:
    return Memory(text=text, device_id="edge-test", **kw)


def test_pii_forces_local_only(edge_settings):
    engine = SyncPolicyEngine(edge_settings)
    decision = engine.classify(_mem("patient email is john@hospital.org"))
    assert decision.sensitivity == Sensitivity.PII
    assert decision.sync_state == SyncState.LOCAL_ONLY
    assert "PII" in decision.reason


def test_important_memory_is_queued(edge_settings):
    engine = SyncPolicyEngine(edge_settings)
    decision = engine.classify(
        _mem("CRITICAL safety fix: shut valve before service to avoid failure")
    )
    assert decision.sync_state == SyncState.PENDING
    assert decision.importance >= edge_settings.importance_sync_threshold


def test_low_value_stays_local(edge_settings):
    engine = SyncPolicyEngine(edge_settings)
    decision = engine.classify(_mem("ok"))
    assert decision.sync_state == SyncState.LOCAL_ONLY


def test_eviction_targets_synced_low_value(edge_settings):
    engine = SyncPolicyEngine(edge_settings)  # budget = 5
    memories = []
    for i in range(8):
        m = _mem(f"note {i}", importance=0.2, sync_state=SyncState.SYNCED)
        m.importance = 0.2
        m.sync_state = SyncState.SYNCED
        memories.append(m)
    victims = engine.select_evictions(memories)
    assert len(victims) == 3  # 8 - budget(5)
    assert all(v.sync_state == SyncState.SYNCED for v in victims)


def test_pii_is_never_evicted(edge_settings):
    engine = SyncPolicyEngine(edge_settings)
    memories = [
        _mem("secret a@b.com", sensitivity=Sensitivity.PII, sync_state=SyncState.LOCAL_ONLY)
        for _ in range(8)
    ]
    for m in memories:
        m.sensitivity = Sensitivity.PII
        m.sync_state = SyncState.LOCAL_ONLY
    assert engine.select_evictions(memories) == []
