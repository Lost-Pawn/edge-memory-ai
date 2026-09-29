from edge_memory.store.outbox import Outbox, OpType


def test_enqueue_and_pending(tmp_path):
    ob = Outbox(tmp_path / "outbox.db")
    ob.enqueue("op1", "mem1", OpType.UPSERT, {"text": "hi"})
    pending = ob.pending()
    assert len(pending) == 1
    assert pending[0].memory_id == "mem1"
    ob.close()


def test_mark_synced_removes_from_pending(tmp_path):
    ob = Outbox(tmp_path / "outbox.db")
    ob.enqueue("op1", "mem1", OpType.UPSERT, {"text": "hi"})
    ob.mark_synced("op1")
    assert ob.pending() == []
    assert ob.counts().get("synced") == 1
    ob.close()


def test_failed_retries_then_gives_up(tmp_path):
    ob = Outbox(tmp_path / "outbox.db")
    ob.enqueue("op1", "mem1", OpType.UPSERT, {"text": "hi"})
    for _ in range(4):
        ob.mark_failed("op1", "boom")
        assert ob.pending(), "should retry while under attempt cap"
    ob.mark_failed("op1", "boom")  # 5th attempt -> failed
    assert ob.pending() == []
    assert ob.counts().get("failed") == 1
    ob.close()


def test_enqueue_is_idempotent_on_op_id(tmp_path):
    ob = Outbox(tmp_path / "outbox.db")
    ob.enqueue("op1", "mem1", OpType.UPSERT, {"text": "v1"})
    ob.enqueue("op1", "mem1", OpType.UPSERT, {"text": "v2"})
    pending = ob.pending()
    assert len(pending) == 1
    assert pending[0].payload["text"] == "v2"
    ob.close()
