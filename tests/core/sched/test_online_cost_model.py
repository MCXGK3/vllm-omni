import json
import time
from unittest.mock import patch

from vllm_omni.core.sched.future_request_registry import (
    FutureRequestHint,
    FutureRequestRegistry,
)
from vllm_omni.core.sched.omni_ar_scheduler_cost_aware import (
    FutureAwareSchedulerMixin,
    _BatchItem,
)
from vllm_omni.core.sched.online_cost_model import (
    OnlineCostModelPredictor,
    OnlineCostModelTrainer,
    record_feature_vector,
)


def _record(q: int, c: int = 0, prompt: int = 4096) -> dict:
    return {
        "requests": [
            {
                "scheduled_tokens": q,
                "computed_tokens_before": c,
                "prompt_tokens": prompt,
            }
        ]
    }


def test_online_fit_persists_and_guards_unseen_paths(tmp_path, monkeypatch):
    model_path = tmp_path / "cost_model.json"
    monkeypatch.setenv("VLLM_OMNI_COST_MODEL_PATH", str(model_path))
    monkeypatch.setenv("VLLM_OMNI_COST_MODEL_MIN_SAMPLES", "4")
    monkeypatch.setenv("VLLM_OMNI_COST_MODEL_REFIT_INTERVAL", "1")
    monkeypatch.setenv("VLLM_OMNI_COST_MODEL_MIN_PATH_SAMPLES", "2")
    monkeypatch.setenv("VLLM_OMNI_COST_MODEL_MIN_VALIDATION_SAMPLES", "1")

    trainer = OnlineCostModelTrainer(stage_id=0, graph_max_tokens=128)
    prefill_records = [_record(q) for q in (128, 256, 384, 512)]
    for index, record in enumerate(prefill_records):
        trainer.observe(record, 10.0 + index)
    trainer.observe(_record(1, c=4096, prompt=4096), 2.0)
    trainer.wait_until_idle()

    document = json.loads(model_path.read_text(encoding="utf-8"))
    stage = document["stages"]["0"]
    assert stage["sample_count"] == 5
    assert stage["path_counts"]["prefill_none"] == 4
    assert stage["path_counts"]["decode_full"] == 1
    assert stage["path_models"]["prefill_none"]["accepted"]

    predictor = OnlineCostModelPredictor(stage_id=0)
    assert predictor.predict(record_feature_vector(prefill_records[0], 128)) is not None
    assert predictor.predict(record_feature_vector(_record(1, 4096, 4096), 128)) is None


def test_no_model_before_minimum_samples(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_COST_MODEL_PATH", str(tmp_path / "model.json"))
    monkeypatch.setenv("VLLM_OMNI_COST_MODEL_MIN_SAMPLES", "3")
    trainer = OnlineCostModelTrainer(stage_id=1, graph_max_tokens=128)
    trainer.observe(_record(128), 12.0)
    trainer.observe(_record(256), 13.0)
    trainer.wait_until_idle()
    assert not trainer.model_path.exists()


def test_future_registry_filters_expired_hints_before_limit(tmp_path):
    registry = FutureRequestRegistry(str(tmp_path / "future.sqlite3"))
    now = time.time()
    # More stale rows than the internal result window must not crowd out the
    # one actionable hint sorted after them.
    for index in range(20):
        registry.publish(
            request_id=f"stale-{index}",
            source_stage=0,
            target_stage=1,
            sequence=index,
            published_at=now - 1,
            ready_at=now - 0.5,
            payload={},
        )
    registry.publish(
        request_id="future",
        source_stage=0,
        target_stage=1,
        sequence=100,
        published_at=now,
        ready_at=now + 0.5,
        payload={},
    )

    hints = registry.get_pending(
        target_stage=1,
        now=now,
        ready_after=now,
        max_items=2,
    )
    assert [hint.request_id for hint in hints] == ["future"]


def test_future_wait_holds_until_positive_deadline():
    now = 1_000.0
    hint = FutureRequestHint(
        request_id="future",
        source_stage=0,
        target_stage=1,
        sequence=7,
        published_at=now - 0.01,
        ready_at=now + 0.02,
        payload={"next_scheduled_tokens": 128, "prompt_tokens": 128},
    )
    scheduler = object.__new__(FutureAwareSchedulerMixin)
    scheduler._cost_aware_wait_until = None
    scheduler._cost_aware_wait_keys = set()
    scheduler._future_waited_keys = set()
    scheduler._cost_aware_future_wait_ms = 100.0
    scheduler._cost_aware_min_wait_ms = 0.1
    scheduler._cost_aware_min_gain_ms = 0.0
    scheduler._cost_aware_budget_ms = 0.0
    scheduler.max_num_scheduled_tokens = 4096
    scheduler._stage_id = lambda: 1
    scheduler._cost_decisions_enabled = lambda: True
    scheduler._future_hints = lambda ready_after=None: (
        [hint] if ready_after is None or hint.ready_at > ready_after else []
    )
    scheduler._running_items = lambda: []
    scheduler._current_candidate_items = lambda waiting: [
        _BatchItem("current", 128, 0, 128)
    ]
    scheduler._request_item = lambda request: _BatchItem("current", 128, 0, 128)
    scheduler._hint_item = lambda value: _BatchItem("future", 128, 0, 128)
    # Separate batches cost 50 + 50 ms; a highly efficient joined batch costs
    # 20 ms. After waiting 20 ms, mean request latency improves by 35 ms.
    scheduler._predict_ms = lambda items: 50.0 if len(items) == 1 else 20.0

    with patch(
        "vllm_omni.core.sched.omni_ar_scheduler_cost_aware.time.time",
        return_value=now,
    ):
        assert scheduler._should_wait_for_future([object()])
    assert scheduler._cost_aware_wait_until == hint.ready_at
    assert scheduler._cost_aware_wait_keys == {"future"}
    assert ("future", 7) in scheduler._future_waited_keys

    # A subsequent poll before ready_at must preserve the hold.
    with patch(
        "vllm_omni.core.sched.omni_ar_scheduler_cost_aware.time.time",
        return_value=now + 0.01,
    ):
        assert scheduler._should_wait_for_future([object()])


def test_latency_objective_rejects_gpu_work_only_false_gain():
    scheduler = object.__new__(FutureAwareSchedulerMixin)
    current = [_BatchItem("current", 128, 0, 128)]
    future = [_BatchItem("future", 128, 0, 128)]
    # The old objective reported 50 + 50 - 20 - 60 = +20 ms. In request
    # completion time, waiting makes the current request finish at 80 ms and
    # the future request at 60 ms, versus 50 ms and 80 ms without waiting.
    scheduler._predict_ms = lambda items: 50.0 if len(items) == 1 else 60.0
    benefit, current_ms, future_ms, joined_ms = (
        scheduler._mean_latency_benefit_ms(current, future, 20.0)
    )
    assert (current_ms, future_ms, joined_ms) == (50.0, 50.0, 60.0)
    assert benefit == -5.0


def test_scheduler_does_not_wait_for_negative_latency_benefit():
    now = 1_000.0
    hint = FutureRequestHint(
        request_id="future",
        source_stage=0,
        target_stage=1,
        sequence=8,
        published_at=now - 0.01,
        ready_at=now + 0.02,
        payload={"next_scheduled_tokens": 128, "prompt_tokens": 128},
    )
    scheduler = object.__new__(FutureAwareSchedulerMixin)
    scheduler._cost_aware_wait_until = None
    scheduler._cost_aware_wait_keys = set()
    scheduler._future_waited_keys = set()
    scheduler._cost_aware_future_wait_ms = 100.0
    scheduler._cost_aware_min_wait_ms = 0.1
    scheduler._cost_aware_min_gain_ms = 0.0
    scheduler._cost_aware_budget_ms = 0.0
    scheduler._stage_id = lambda: 1
    scheduler._cost_decisions_enabled = lambda: True
    scheduler._future_hints = lambda ready_after=None: [hint]
    scheduler._current_candidate_items = lambda waiting: [
        _BatchItem("current", 128, 0, 128)
    ]
    scheduler._hint_item = lambda value: _BatchItem("future", 128, 0, 128)
    scheduler._predict_ms = lambda items: 50.0 if len(items) == 1 else 60.0

    with patch(
        "vllm_omni.core.sched.omni_ar_scheduler_cost_aware.time.time",
        return_value=now,
    ):
        assert not scheduler._should_wait_for_future([object()])
    assert scheduler._cost_aware_wait_until is None
    assert not scheduler._future_waited_keys
