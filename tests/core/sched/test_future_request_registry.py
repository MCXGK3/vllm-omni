import time

from vllm_omni.core.sched.future_request_registry import FutureRequestRegistry


def test_transfer_ready_publication_supersedes_scheduled(tmp_path):
    registry = FutureRequestRegistry(str(tmp_path / "future.sqlite3"), ttl_s=60)
    now = time.time()
    registry.publish(
        request_id="req-1",
        source_stage=0,
        target_stage=1,
        sequence=10,
        published_at=now,
        ready_at=now + 5,
        payload={"publication_node": "scheduled"},
    )
    registry.publish(
        request_id="req-1",
        source_stage=0,
        target_stage=1,
        sequence=1,
        published_at=now + 1,
        ready_at=now + 2,
        payload={"publication_node": "transfer_ready"},
    )

    hints = registry.get_pending(
        target_stage=1,
        now=now + 1.1,
        ready_after=now + 1.1,
    )

    assert len(hints) == 1
    assert hints[0].payload["publication_node"] == "transfer_ready"
    assert hints[0].ready_at == now + 2


def test_expired_latest_publication_does_not_resurrect_older_eta(tmp_path):
    registry = FutureRequestRegistry(str(tmp_path / "future.sqlite3"), ttl_s=60)
    now = time.time()
    registry.publish(
        request_id="req-1",
        source_stage=0,
        target_stage=1,
        sequence=10,
        published_at=now,
        ready_at=now + 10,
        payload={"publication_node": "scheduled"},
    )
    registry.publish(
        request_id="req-1",
        source_stage=0,
        target_stage=1,
        sequence=1,
        published_at=now + 1,
        ready_at=now + 2,
        payload={"publication_node": "transfer_ready"},
    )

    hints = registry.get_pending(
        target_stage=1,
        now=now + 3,
        ready_after=now + 3,
    )

    assert hints == []


def test_stage_state_round_trip_and_staleness(tmp_path):
    registry = FutureRequestRegistry(str(tmp_path / "future.sqlite3"), ttl_s=60)
    now = time.time()
    registry.publish_stage_state(
        stage_id=1,
        updated_at=now,
        payload={"recv_queue_depth": 3, "get_ewma_ms": 2.5},
    )

    state = registry.get_stage_state(1, now=now + 0.5, max_age_s=1)
    assert state is not None
    assert state["recv_queue_depth"] == 3
    assert state["get_ewma_ms"] == 2.5
    assert registry.get_stage_state(1, now=now + 2, max_age_s=1) is None
