"""Future-aware cost-based Omni AR schedulers.

This module intentionally sits on top of the existing Omni scheduler.  It
only controls which waiting requests are exposed to the vLLM scheduler in a
step; KV-cache allocation, request state transitions, and output handling are
still performed by the original scheduler.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from vllm.v1.request import Request, RequestStatus

from vllm_omni.config.future_scheduler_config import load_future_scheduler_config
from vllm_omni.core.sched.future_request_registry import (
    FutureRequestHint,
    FutureRequestRegistry,
)
from vllm_omni.core.sched.omni_ar_scheduler import (
    OmniARAsyncScheduler,
    OmniARScheduler,
)
from vllm_omni.core.sched.online_cost_model import (
    OnlineCostModelPredictor,
    feature_vector,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BatchItem:
    request_id: str
    q: int
    c: int
    prompt_tokens: int


@dataclass(frozen=True)
class _BatchFeatures:
    n: int
    a: int
    s: int
    b: int
    m: int
    c: int
    d: int
    p: int
    state: str
    mode: str


class FutureAwareSchedulerMixin:
    """Admission control based on predicted current and future batch cost."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        future_config = load_future_scheduler_config()
        decision_config = future_config.decision
        registry_config = future_config.registry
        transfer_config = future_config.transfer
        prediction_config = future_config.prediction
        diagnostics_config = future_config.diagnostics
        self._future_registry = FutureRequestRegistry(
            path=registry_config.path,
            ttl_s=registry_config.ttl_s,
        )
        self._future_waited_keys: set[tuple[str, int]] = set()
        # A wait decision must survive until the next scheduler poll.  The
        # old implementation only parked requests for one call to
        # ``schedule()``, so a tight scheduler loop could admit them before
        # the announced downstream request became available.
        self._cost_aware_wait_until: float | None = None
        # Track request ids, rather than publication sequence numbers.  The
        # upstream stage can publish a newer sequence while the same request
        # is still in flight; that must not end an active hold prematurely.
        self._cost_aware_wait_keys: set[str] = set()
        self._cost_aware_step = 0
        self._cost_aware_budget_ms = decision_config.budget_ms
        self._cost_aware_future_wait_ms = decision_config.max_wait_ms
        # The upstream model forward finishing is earlier than the request
        # becoming schedulable downstream. Account for output postprocessing,
        # IPC queueing/transfer and downstream ingestion in the published ETA.
        self._future_delivery_margin_ms = max(
            0.0, transfer_config.delivery_margin_ms
        )
        # Avoid parking a batch for a sub-resolution interval. Such a decision
        # only adds an empty scheduler iteration and was previously reported as
        # an active wait even when the computed duration was exactly zero.
        self._cost_aware_min_wait_ms = max(0.0, decision_config.min_wait_ms)
        self._cost_aware_min_gain_ms = decision_config.min_gain_ms
        self._future_max_items = decision_config.max_future_items
        self._future_stage1_token_offset = transfer_config.stage1_token_offset
        self._future_stage_state_max_age_s = max(
            0.1, registry_config.stage_state_max_age_s
        )
        self._future_stage_state_interval_s = max(
            0.001,
            registry_config.stage_state_interval_ms / 1000.0,
        )
        self._future_last_stage_state_at = 0.0
        self._future_put_default_ms = max(0.01, transfer_config.put_default_ms)
        self._future_get_default_ms = max(0.01, transfer_config.get_default_ms)
        self._future_decode_queue_margin_ms = max(
            0.0, decision_config.decode_queue_margin_ms
        )
        self._graph_capture_max_tokens = prediction_config.graph_capture_max_tokens
        self._cost_model_mode = prediction_config.mode.strip().lower()
        self._future_diag_enabled = diagnostics_config.enabled
        self._future_diag_log_every = max(1, diagnostics_config.log_every)
        self._future_diag_calls = 0
        self._future_diag_counts: dict[str, int] = {}
        valid_modes = {"static", "collect", "learn", "fitted"}
        if self._cost_model_mode not in valid_modes:
            logger.warning(
                "Unknown VLLM_OMNI_COST_MODEL_MODE=%r; using static",
                self._cost_model_mode,
            )
            self._cost_model_mode = "static"
        self._online_cost_predictor = (
            OnlineCostModelPredictor(self._stage_id())
            if self._cost_model_mode != "static"
            else None
        )
        logger.info(
            "Cost-aware predictor mode stage=%s mode=%s",
            self._stage_id(),
            self._cost_model_mode,
        )

    def _future_diag_inc(self, reason: str, amount: int = 1) -> None:
        if not self._future_diag_enabled:
            return
        self._future_diag_counts[reason] = (
            self._future_diag_counts.get(reason, 0) + amount
        )

    def _future_diag_log(self, *, force: bool = False) -> None:
        if not self._future_diag_enabled:
            return
        if not force and self._future_diag_calls % self._future_diag_log_every:
            return
        logger.info(
            "Future-aware diagnostics stage=%s calls=%s counts=%s",
            self._stage_id(),
            self._future_diag_calls,
            dict(sorted(self._future_diag_counts.items())),
        )

    def _future_diag_registry_stats(self) -> None:
        if not self._future_diag_enabled:
            return
        stats = getattr(self._future_registry, "last_query_stats", {})
        for name in (
            "raw_rows",
            "known_filtered",
            "malformed_filtered",
            "newer_replaced",
        ):
            value = int(stats.get(name, 0))
            if value:
                self._future_diag_inc(f"registry_{name}", value)

    # ------------------------------------------------------------------
    # Feature extraction and path-specific predictor
    # ------------------------------------------------------------------

    def _stage_id(self) -> int:
        return int(getattr(self.vllm_config.model_config, "stage_id", 0))

    @staticmethod
    def _request_prompt_tokens(request: Request) -> int:
        return int(getattr(request, "num_prompt_tokens", 0) or 0)

    def _request_q(self, request: Request, token_budget: int | None = None) -> int:
        computed = max(
            0,
            int(getattr(request, "num_computed_tokens", 0))
            - int(getattr(request, "num_output_placeholders", 0)),
        )
        prompt = self._request_prompt_tokens(request)
        if computed < prompt:
            remaining = prompt - computed
        else:
            # The cost-aware layer only needs the next decode step for a
            # running request.  The base scheduler remains authoritative for
            # speculative decoding and max-token edge cases.
            remaining = 1
        threshold = int(getattr(self.scheduler_config, "long_prefill_token_threshold", 0))
        if threshold > 0:
            remaining = min(remaining, threshold)
        if token_budget is not None:
            remaining = min(remaining, max(token_budget, 0))
        return max(0, int(remaining))

    def _request_item(self, request: Request, token_budget: int | None = None) -> _BatchItem:
        return _BatchItem(
            request_id=str(request.request_id),
            q=self._request_q(request, token_budget),
            c=max(0, int(getattr(request, "num_computed_tokens", 0))),
            prompt_tokens=self._request_prompt_tokens(request),
        )

    def _hint_item(self, hint: FutureRequestHint) -> _BatchItem:
        payload = hint.payload
        q = int(payload.get("next_scheduled_tokens", payload.get("prompt_tokens", 1)))
        c = int(payload.get("computed_tokens_before", 0))
        prompt = int(payload.get("next_prompt_tokens", payload.get("prompt_tokens", q)))
        return _BatchItem(hint.request_id, max(q, 1), max(c, 0), max(prompt, q))

    def _path(self, items: list[_BatchItem]) -> tuple[str, str]:
        if not items:
            return "empty", "NONE"
        has_prefill = any(item.q > 1 for item in items)
        has_decode = any(item.q == 1 and item.c >= item.prompt_tokens for item in items)
        if has_prefill and has_decode:
            state = "mixed"
        elif has_prefill:
            state = "chunk" if any(item.c > 0 for item in items if item.q > 1) else "prefill"
        else:
            state = "decode"

        if state == "decode":
            mode = "FULL"
        elif state == "mixed" and sum(item.q for item in items) <= self._graph_capture_max_tokens:
            mode = "PIECEWISE"
        else:
            mode = "NONE"
        return state, mode

    def _features(self, items: list[_BatchItem]) -> _BatchFeatures:
        state, mode = self._path(items)
        n = sum(item.q for item in items)
        a = sum(item.q * item.c + item.q * (item.q + 1) // 2 for item in items)
        s = sum(item.c + item.q for item in items)
        return _BatchFeatures(
            n=n,
            a=a,
            s=s,
            b=len(items),
            m=max((item.c + item.q for item in items), default=0),
            c=sum(item.c for item in items),
            d=sum(item.q for item in items if item.q == 1 and item.c >= item.prompt_tokens),
            p=sum(1 for item in items if item.q > 1),
            state=state,
            mode=mode,
        )

    def _online_feature_vector(self, f: _BatchFeatures) -> list[float]:
        return feature_vector(
            n=f.n,
            a=f.a,
            b=f.b,
            m=f.m,
            c=f.c,
            s=f.s,
            d=f.d,
            p=f.p,
            state=f.state,
            mode=f.mode,
        )

    def _online_model_ready(self) -> bool:
        return bool(
            self._online_cost_predictor is not None
            and self._online_cost_predictor.ready
        )

    def _cost_decisions_enabled(self) -> bool:
        if self._cost_model_mode == "static":
            return True
        if self._cost_model_mode == "collect":
            return False
        return self._online_model_ready()

    def _predict_ms(self, items: list[_BatchItem]) -> float:
        """Predict synchronized forward time using the current fitted model.

        The coefficients are deliberately kept in this research scheduler,
        rather than changing the production model runner.  They are the
        current Qwen3-Omni/A100 coefficients and should be replaced by a
        loaded model once enough online samples are available.
        """
        f = self._features(items)
        if f.state == "empty":
            return 0.0
        if self._online_cost_predictor is not None:
            prediction = self._online_cost_predictor.predict(
                self._online_feature_vector(f)
            )
            if prediction is not None:
                return prediction
        if self._stage_id() == 0:
            base = 61.9975
            beta = (67.2946, -5.4266, -12.0575, 11.3847, -1.8995, 11.9480, 8.0227, 0.6761)
            offsets = {
                ("decode", "FULL"): -89.4193,
                ("prefill", "NONE"): 0.0,
                ("chunk", "NONE"): -69.8116,
                ("mixed", "NONE"): -132.0535,
                ("mixed", "PIECEWISE"): -116.9849,
            }
        else:
            base = 78.7162
            beta = (-0.7459, 2.3164, -1.2926, 0.1236, -0.0083, 0.6791, -2.2023, 0.9993)
            offsets = {
                ("decode", "FULL"): -75.6408,
                ("prefill", "NONE"): 0.0,
                ("chunk", "NONE"): -54.7430,
                ("mixed", "NONE"): -44.3215,
                ("mixed", "PIECEWISE"): -14.1843,
            }
        value = (
            base
            + beta[0] * f.n / 1e3
            + beta[1] * f.a / 1e6
            + beta[2] * (0 if f.b <= 0 else __import__("math").log2(1 + f.b))
            + beta[3] * f.m / 1e3
            + beta[4] * f.c / 1e3
            + beta[5] * f.d
            + beta[6] * f.p
            + beta[7] * 0.0
            + offsets.get((f.state, f.mode), 0.0)
        )
        # A fitted linear model can produce a negative value for tiny or
        # unusual batches; the scheduler must never use a negative cost.
        return max(float(value), 0.05)

    # ------------------------------------------------------------------
    # Future publication and admission
    # ------------------------------------------------------------------

    def _future_adapter_snapshot(self) -> dict[str, float | int]:
        adapter = getattr(self, "chunk_transfer_adapter", None)
        snapshot = getattr(adapter, "future_queue_snapshot", None)
        if callable(snapshot):
            try:
                return dict(snapshot())
            except Exception:
                logger.debug("Failed to read transfer queue snapshot", exc_info=True)
        return {
            "send_queue_depth": 0,
            "recv_queue_depth": 0,
            "put_ewma_ms": self._future_put_default_ms,
            "get_ewma_ms": self._future_get_default_ms,
        }

    def _publish_stage_state(self, waiting: list[Request]) -> None:
        """Expose downstream queue state to upstream ETA estimators."""
        now = time.time()
        if now - self._future_last_stage_state_at < self._future_stage_state_interval_s:
            return
        self._future_last_stage_state_at = now
        snapshot = self._future_adapter_snapshot()
        current_items = self._current_candidate_items(waiting)
        scheduler_backlog_ms = self._predict_ms(current_items) if current_items else 0.0
        self._future_registry.publish_stage_state(
            stage_id=self._stage_id(),
            updated_at=now,
            payload={
                **snapshot,
                "ready_waiting_count": len(waiting),
                "running_count": len(self.running),
                "scheduler_backlog_ms": scheduler_backlog_ms,
            },
        )

    def _future_target_state(self, target_stage: int, now: float) -> dict[str, Any]:
        return self._future_registry.get_stage_state(
            target_stage,
            now=now,
            max_age_s=self._future_stage_state_max_age_s,
        ) or {}

    def _next_decode_items(self, items: list[_BatchItem]) -> list[_BatchItem]:
        """Approximate the first decode batch after the current prefill step."""
        decode_items: list[_BatchItem] = []
        for item in items:
            computed_after = item.c + item.q
            if item.q > 1 and computed_after < item.prompt_tokens:
                continue
            decode_items.append(
                _BatchItem(
                    request_id=item.request_id,
                    q=1,
                    c=max(computed_after, item.prompt_tokens),
                    prompt_tokens=item.prompt_tokens,
                )
            )
        return decode_items

    def _scheduled_eta_components(
        self,
        *,
        item: _BatchItem,
        items: list[_BatchItem],
        item_rank: int,
        target_stage: int,
        now: float,
    ) -> dict[str, float]:
        """Estimate schedule-to-downstream-schedulable latency for one request."""
        current_forward_ms = self._predict_ms(items)
        remaining_prefill_ms = 0.0
        decode_ms = 0.0
        if self._stage_id() == 0 and item.q > 1:
            computed_after = item.c + item.q
            remaining = max(0, item.prompt_tokens - computed_after)
            if remaining:
                remaining_prefill_ms = self._predict_ms(
                    [
                        _BatchItem(
                            request_id=item.request_id,
                            q=remaining,
                            c=computed_after,
                            prompt_tokens=item.prompt_tokens,
                        )
                    ]
                )
            decode_items = self._next_decode_items(items)
            if not any(x.request_id == item.request_id for x in decode_items):
                decode_items.append(
                    _BatchItem(
                        request_id=item.request_id,
                        q=1,
                        c=item.prompt_tokens,
                        prompt_tokens=item.prompt_tokens,
                    )
                )
            decode_ms = self._predict_ms(decode_items)

        compute_horizon_ms = (
            current_forward_ms
            + remaining_prefill_ms
            + self._future_decode_queue_margin_ms
            + decode_ms
        )
        source_state = self._future_adapter_snapshot()
        target_state = self._future_target_state(target_stage, now)
        put_ms = max(
            0.01,
            float(source_state.get("put_ewma_ms", self._future_put_default_ms)),
        )
        get_ms = max(
            0.01,
            float(target_state.get("get_ewma_ms", self._future_get_default_ms)),
        )
        send_depth = max(0, int(source_state.get("send_queue_depth", 0)))
        recv_depth = max(0, int(target_state.get("recv_queue_depth", 0)))

        # Both transfer workers run concurrently with model compute.  Only the
        # residual queue work at the predicted enqueue/arrival time contributes
        # to this request's ETA. Requests from this batch are serialized by rank.
        send_queue_ms = max(0.0, send_depth * put_ms - compute_horizon_ms)
        send_queue_ms += item_rank * put_ms
        recv_queue_ms = max(0.0, recv_depth * get_ms - compute_horizon_ms)
        recv_queue_ms += item_rank * get_ms
        target_scheduler_ms = max(
            0.0,
            float(target_state.get("scheduler_backlog_ms", 0.0))
            - compute_horizon_ms,
        )
        return {
            "current_forward": current_forward_ms,
            "remaining_prefill": remaining_prefill_ms,
            "decode_queue": self._future_decode_queue_margin_ms,
            "next_decode": decode_ms,
            "send_queue": send_queue_ms,
            "put": put_ms,
            "recv_queue": recv_queue_ms,
            "get": get_ms,
            "target_scheduler": target_scheduler_ms,
            "delivery_margin": self._future_delivery_margin_ms,
        }

    def _publish_next_stage_hints(self, scheduler_output: Any) -> None:
        source_stage = self._stage_id()
        target_stage = source_stage + 1
        if not scheduler_output.num_scheduled_tokens:
            return
        requests = []
        items = []
        for req_id, q in scheduler_output.num_scheduled_tokens.items():
            request = self.requests.get(req_id)
            if request is None:
                continue
            item = self._request_item(request, int(q))
            # Use the scheduler's actual q for this step.  The base scheduler
            # remains authoritative when long-prefill or KV-cache constraints
            # change the amount scheduled. vLLM has already advanced
            # num_computed_tokens when this method runs, so recover the
            # pre-step context length used by the attention computation.
            computed_before = max(0, item.c - int(q))
            item = _BatchItem(
                item.request_id, int(q), computed_before, item.prompt_tokens
            )
            requests.append((request, item))
            items.append(item)
        if not requests:
            return

        now = time.time()
        for item_rank, (request, item) in enumerate(requests):
            prompt_tokens = item.prompt_tokens
            next_prompt_tokens = prompt_tokens
            if target_stage == 1 and source_stage == 0:
                next_prompt_tokens += self._future_stage1_token_offset
            components = self._scheduled_eta_components(
                item=item,
                items=items,
                item_rank=item_rank,
                target_stage=target_stage,
                now=now,
            )
            eta_ms = sum(components.values())
            ready_at = now + eta_ms / 1000.0
            self._future_registry.publish(
                request_id=item.request_id,
                source_stage=source_stage,
                target_stage=target_stage,
                sequence=time.time_ns(),
                published_at=now,
                ready_at=ready_at,
                payload={
                    "publication_node": "scheduled",
                    "source_stage": source_stage,
                    "target_stage": target_stage,
                    "prompt_tokens": prompt_tokens,
                    "next_prompt_tokens": next_prompt_tokens,
                    "computed_tokens_before": item.c,
                    "next_scheduled_tokens": item.q,
                    "phase": "decode" if item.q == 1 and item.c >= prompt_tokens else "prefill",
                    "batch_size": len(items),
                    "batch_tokens": sum(x.q for x in items),
                    "eta_components_ms": components,
                    "ready_at": ready_at,
                },
            )

    def _known_request_ids(self) -> set[str]:
        return {
            str(request.request_id)
            for request in list(self.running) + list(self.waiting)
        }

    def _future_hints(
        self, *, ready_after: float | None = None
    ) -> list[FutureRequestHint]:
        return self._future_registry.get_pending(
            target_stage=self._stage_id(),
            known_request_ids=self._known_request_ids(),
            ready_after=ready_after,
            max_items=self._future_max_items,
        )

    def _waiting_requests(self) -> list[Request]:
        return [request for request in self.waiting if request.status == RequestStatus.WAITING]

    def _running_items(self) -> list[_BatchItem]:
        return [self._request_item(request) for request in self.running if not request.is_finished()]

    def _select_admitted(self, waiting: list[Request]) -> tuple[list[Request], list[Request]]:
        if not waiting:
            return [], []
        if not self._cost_decisions_enabled():
            return list(waiting), []
        selected: list[Request] = []
        current_items = self._running_items()
        current_tokens = sum(item.q for item in current_items)
        token_limit = int(self.max_num_scheduled_tokens)
        budget = self._cost_aware_budget_ms

        # Keep FCFS order by default.  The gain is from admission based on
        # the batch prediction, not from reordering requests and starving old
        # requests.
        for request in waiting:
            remaining_budget = max(0, token_limit - current_tokens)
            q = self._request_q(request, remaining_budget)
            if q <= 0:
                continue
            item = self._request_item(request, q)
            candidate_items = current_items + [item]
            candidate_time = self._predict_ms(candidate_items)
            over_time = budget > 0 and candidate_time > budget
            over_tokens = sum(x.q for x in candidate_items) > token_limit
            if (over_time or over_tokens) and not selected:
                # Never deadlock a batch because its first request alone is
                # larger than the configured time budget.
                selected.append(request)
                current_items = candidate_items
                current_tokens += q
                continue
            if over_time or over_tokens:
                # Preserve FCFS in the first implementation.  Skipping a
                # large request in order to admit a later small request is a
                # useful extension, but it needs an explicit aging policy to
                # avoid starving the large request.
                break
            selected.append(request)
            current_items = candidate_items
            current_tokens += q
        admitted_ids = {id(request) for request in selected}
        rejected = [request for request in waiting if id(request) not in admitted_ids]
        return selected, rejected

    def _current_candidate_items(self, waiting: list[Request]) -> list[_BatchItem]:
        """Approximate the batch that the base scheduler would run now."""
        items = self._running_items()
        token_limit = int(self.max_num_scheduled_tokens)
        used_tokens = sum(item.q for item in items)
        for request in waiting:
            remaining = max(0, token_limit - used_tokens)
            q = self._request_q(request, remaining)
            if q <= 0:
                break
            item = self._request_item(request, q)
            items.append(item)
            used_tokens += item.q
            if used_tokens >= token_limit:
                break
        return items

    def _mean_latency_benefit_ms(
        self,
        current_items: list[_BatchItem],
        future_items: list[_BatchItem],
        wait_ms: float,
    ) -> tuple[float, float, float, float]:
        """Compare no-wait and wait-and-join mean request completion time.

        Future requests are predicted as the batch they would naturally form,
        not as a sum of single-request executions. The wait penalty is paid by
        every request already eligible to run.
        """
        if not current_items or not future_items:
            return float("-inf"), 0.0, 0.0, 0.0
        current_time = self._predict_ms(current_items)
        future_batch_time = self._predict_ms(future_items)
        joined_time = self._predict_ms(current_items + future_items)
        current_count = len(current_items)
        future_count = len(future_items)

        # No wait: current requests finish after current_time. Future requests
        # arrive after wait_ms and then either wait for the current batch or run
        # immediately as one natural batch.
        no_wait_sum = (
            current_count * current_time
            + future_count
            * (max(current_time - wait_ms, 0.0) + future_batch_time)
        )
        # Wait and join: current requests pay the full wait; future latency is
        # measured from their expected arrival at the selected deadline.
        joined_sum = (
            current_count * (wait_ms + joined_time)
            + future_count * joined_time
        )
        benefit = (no_wait_sum - joined_sum) / (current_count + future_count)
        return benefit, current_time, future_batch_time, joined_time

    def _should_wait_for_future(self, waiting: list[Request]) -> bool:
        self._future_diag_calls += 1
        if not waiting:
            self._future_diag_inc("reject_empty_waiting")
            self._cost_aware_wait_until = None
            self._cost_aware_wait_keys.clear()
            return False
        if not self._cost_decisions_enabled():
            self._future_diag_inc("reject_cost_decisions_disabled")
            self._cost_aware_wait_until = None
            self._cost_aware_wait_keys.clear()
            return False
        now = time.time()

        # Preserve the wait decision across scheduler polls.  Release early
        # when one of the announced requests has already entered this stage;
        # otherwise hold the current waiting requests until the prediction
        # deadline expires.
        if self._cost_aware_wait_until is not None:
            pending_ids = {
                hint.request_id
                for hint in self._future_hints(ready_after=now)
            }
            still_pending = bool(self._cost_aware_wait_keys & pending_ids)
            if now < self._cost_aware_wait_until and still_pending:
                return True
            self._cost_aware_wait_until = None
            self._cost_aware_wait_keys.clear()

        if self._cost_aware_future_wait_ms <= 0:
            self._future_diag_inc("reject_wait_window_disabled")
            return False
        # Expired ETAs are not future work. Filtering them in the registry is
        # important because its ORDER BY ready_at would otherwise let stale
        # rows occupy the entire max_items window.
        hints = self._future_hints(ready_after=now)
        self._future_diag_registry_stats()
        if not hints:
            self._future_diag_inc("reject_no_hints_after_registry_filter")
            return False
        current_items = self._current_candidate_items(waiting)
        if not current_items:
            self._future_diag_inc("reject_no_current_items")
            return False
        not_waited_hints = [
            hint
            for hint in hints
            if (hint.request_id, hint.sequence) not in self._future_waited_keys
        ]
        future_hints = [
            hint
            for hint in not_waited_hints
            if self._cost_aware_min_wait_ms / 1000.0
            <= hint.ready_at - now
            <= self._cost_aware_future_wait_ms / 1000.0
        ]
        self._future_diag_inc(
            "reject_already_waited_hints", len(hints) - len(not_waited_hints)
        )
        self._future_diag_inc(
            "reject_hints_outside_eta_window",
            len(not_waited_hints) - len(future_hints),
        )
        if not future_hints:
            self._future_diag_inc("reject_no_hints_in_eta_window")
            return False
        # Hints have different ETAs. Evaluate every ready-time prefix and wait
        # only for the prefix with the best latency benefit. Using all hints
        # with the earliest ETA previously predicted a joined batch that could
        # not actually exist at the release deadline.
        future_hints.sort(key=lambda hint: (hint.ready_at, hint.sequence))
        best = None
        for end in range(1, len(future_hints) + 1):
            selected_hints = future_hints[:end]
            wait_ms = max(
                0.0,
                (max(hint.ready_at for hint in selected_hints) - now) * 1000.0,
            )
            if wait_ms < self._cost_aware_min_wait_ms:
                continue
            selected_items = [self._hint_item(hint) for hint in selected_hints]
            benefit, current_time, future_batch_time, joined_time = (
                self._mean_latency_benefit_ms(
                    current_items, selected_items, wait_ms
                )
            )
            within_budget = (
                self._cost_aware_budget_ms <= 0
                or joined_time <= self._cost_aware_budget_ms
            )
            if not within_budget:
                self._future_diag_inc("reject_candidate_budget")
                continue
            candidate = (
                benefit,
                -wait_ms,
                selected_hints,
                current_time,
                future_batch_time,
                joined_time,
                wait_ms,
            )
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None or best[0] < self._cost_aware_min_gain_ms:
            if best is None:
                self._future_diag_inc("reject_no_budget_valid_candidate")
            else:
                self._future_diag_inc("reject_gain_below_threshold")
            return False

        (
            benefit,
            _,
            selected_hints,
            current_time,
            future_batch_time,
            joined_time,
            wait_ms,
        ) = best
        for hint in selected_hints:
            self._future_waited_keys.add((hint.request_id, hint.sequence))
        self._cost_aware_wait_until = max(
            hint.ready_at for hint in selected_hints
        )
        self._cost_aware_wait_keys = {
            hint.request_id for hint in selected_hints
        }
        logger.info(
            "Future-aware scheduler delaying stage=%s: current=%.2fms "
            "future_batch=%.2fms joined=%.2fms wait=%.2fms "
            "mean_latency_benefit=%.2fms current_reqs=%s future_reqs=%s hints=%s",
            self._stage_id(), current_time, future_batch_time, joined_time,
            wait_ms, benefit, len(current_items), len(selected_hints),
            [hint.request_id for hint in selected_hints],
        )
        return True

    def _run_with_parked_waiting(self, parked: list[Request]):
        if parked:
            self.waiting.remove_requests(parked)
        try:
            return super().schedule()
        finally:
            # Restore in original order.  The request queue policy will apply
            # again, but FCFS queues preserve this order through prepend.
            for request in reversed(parked):
                self.waiting.prepend_request(request)

    def schedule(self):
        self._cost_aware_step += 1
        adapter = getattr(self, "chunk_transfer_adapter", None)
        chunk_queues_preprocessed = False
        if adapter is not None:
            # Async IPC pre-creates downstream Request objects before their
            # chunks arrive.  Remove those not-ready placeholders first so
            # they remain visible as registry hints, not as current requests.
            adapter.process_pending_chunks(self.waiting, self.running)
            self._omni_chunk_queues_preprocessed = True
            chunk_queues_preprocessed = True

        try:
            waiting = self._waiting_requests()
            self._publish_stage_state(waiting)

            # A future-aware hold is implemented by temporarily hiding waiting
            # requests and returning an empty normal SchedulerOutput.  This keeps
            # the engine protocol intact and allows the next input-queue poll to
            # receive the announced downstream request.
            if self._should_wait_for_future(waiting):
                output = self._run_with_parked_waiting(waiting)
                self._publish_next_stage_hints(output)
                return output

            admitted, rejected = self._select_admitted(waiting)
            if waiting and rejected:
                # Hide rejected requests while the original scheduler schedules
                # running requests and the admitted prefix.
                all_waiting = list(waiting)
                self.waiting.remove_requests(all_waiting)
                for request in admitted:
                    self.waiting.add_request(request)
                try:
                    output = super().schedule()
                finally:
                    # Restore requests that were not exposed in this step.
                    for request in reversed(rejected):
                        self.waiting.prepend_request(request)
            else:
                output = super().schedule()

            self._publish_next_stage_hints(output)
            return output
        finally:
            if chunk_queues_preprocessed:
                self._omni_chunk_queues_preprocessed = False
                adapter.restore_queues(self.waiting, self.running)
            self._future_diag_log()

    def has_unfinished_requests(self) -> bool:
        # Requests parked for a cost-aware step are restored before returning
        # from schedule(), so the base implementation remains authoritative.
        unfinished = super().has_unfinished_requests()
        if not unfinished:
            self._future_diag_log(force=True)
        return unfinished


class CostAwareOmniARScheduler(FutureAwareSchedulerMixin, OmniARScheduler):
    """Synchronous future-aware Omni AR scheduler."""


class CostAwareOmniARAsyncScheduler(FutureAwareSchedulerMixin, OmniARAsyncScheduler):
    """Async future-aware Omni AR scheduler."""
