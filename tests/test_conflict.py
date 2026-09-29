from edge_memory.models import Memory
from edge_memory.sync.conflict import is_duplicate, resolve


def _mem(text: str, *, device: str, version: int, ts: float) -> Memory:
    return Memory(
        text=text, device_id=device, version=version, updated_at=ts, id="shared-id"
    )


def test_no_existing_accepts_incoming():
    incoming = _mem("a", device="A", version=1, ts=1.0)
    winner, report = resolve(incoming, None)
    assert winner is incoming
    assert report is None


def test_newer_version_same_lineage_wins_cleanly():
    existing = _mem("a", device="A", version=1, ts=1.0)
    incoming = _mem("a2", device="A", version=2, ts=2.0)
    winner, report = resolve(incoming, existing)
    assert winner.version == 2
    assert report is None


def test_concurrent_edits_last_writer_wins():
    existing = _mem("from B", device="B", version=1, ts=5.0)
    incoming = _mem("from A", device="A", version=1, ts=9.0)
    winner, report = resolve(incoming, existing)
    assert winner.text == "from A"
    assert report is not None
    assert report.resolution == "kept_incoming"
    assert winner.metadata["superseded"] == "from B"


def test_concurrent_edits_llm_merge():
    existing = _mem("torque 40Nm", device="B", version=1, ts=5.0)
    incoming = _mem("torque 45Nm", device="A", version=1, ts=9.0)
    winner, report = resolve(
        incoming, existing, merge_fn=lambda a, b: f"{a.text} | {b.text}"
    )
    assert report.resolution == "merged"
    assert "40Nm" in winner.text and "45Nm" in winner.text


def test_is_duplicate_threshold():
    assert is_duplicate(0.97, 0.95)
    assert not is_duplicate(0.90, 0.95)
