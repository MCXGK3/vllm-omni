from __future__ import annotations

import time
from collections import deque
from typing import Any

from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request, RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import (
    OmniARAsyncScheduler,
    OmniARScheduler,
)

logger = init_logger(__name__)

# Sentinel batch-index for requests that do NOT belong to any batch.
_NO_BATCH = 2**30


class BatchOrderOmniARScheduler(OmniARScheduler):
    """Omni AR scheduler that enforces a **batch-sequential prefill order**.

    Requests are partitioned into ordered batches.  A request belongs to a batch
    when the **last segment** of its ``request_id`` (split by ``-``) equals one
    of the suffixes listed for that batch.  For example::

        req_id = "chatcmpl-bench-e1191862-1"  → last segment = "1"

    A batch is **only** allowed to begin prefill when:

    1. A request matching every suffix in that batch has arrived.
    2. A request matching every suffix in **all previous batches** has
       completed prefill (or finished).

    This guarantees that prefill runs in strict batch order — earlier batches
    always finish prefill before later batches start.

    Parameters
    ----------
    batch_order : list[list[str]]
        Ordered list of batches.  Each batch is a list of **suffixes** that are
        matched against the last ``-``-separated segment of each ``request_id``.
        Example: ``[["01", "02"], ["03"], ["04", "05"]]``
        - ``req_id = "chat-01"`` → batch 0
        - ``req_id = "chat-02"`` → batch 0
        - ``req_id = "chat-03"`` → batch 1
        - ``req_id = "chat-11"`` → no batch
    """

    def __init__(self, *args, batch_order: list[list[str]] | None = None, **kwargs):
        super().__init__(*args, **kwargs)

        # ---- batch specification -------------------------------------------
        self._batch_order: list[list[str]] = batch_order or ([["2","3","6","8"],["1","9"]] if self.stage_id==1 else [])
        if self.stage_id==0:
            self._batch_order=[['1','2'],['3','4'],['5','6'],['7','8']]
        elif self.stage_id==1:
            self._batch_order=[]
            
        # suffix → batch_index (0-based).  If a suffix appears in multiple
        # batches the constructor raises ValueError.
        self._suffix_to_batch: dict[str, int] = {}
        for i, batch in enumerate(self._batch_order):
            for suffix in batch:
                if suffix in self._suffix_to_batch:
                    raise ValueError(
                        f"Suffix {suffix!r} appears in multiple batches "
                        f"({self._suffix_to_batch[suffix]} and {i})."
                    )
                self._suffix_to_batch[suffix] = i

        self._num_batches = len(self._batch_order)

        # ---- runtime state -------------------------------------------------
        # Per batch: set of *suffixes* for which a matching request has arrived.
        self._batch_suffixes_arrived: list[set[str]] = [
            set() for _ in range(self._num_batches)
        ]
        # Which batch indices have completed prefill.
        self._batch_prefill_done: set[int] = set()

        # Pending batch requests held back from the waiting queue.
        # req_id → Request
        self._pending_batch_requests: dict[str, Request] = {}

        # Effective "arrival time" used for ordering: the wall-clock time at
        # which each request's *initial* chunk became available.  Falls back
        # to Request.arrival_time when a request has no recorded chunk time.
        # req_id → time.time()
        self._req_chunk_arrival_time: dict[str, float] = {}

        # Suffixes whose matching request has finished prefill (or been
        # aborted / finished by other means).
        self._suffix_prefill_done: set[str] = set()

        # Cache: req_id → (batch_idx, suffix) to avoid repeated suffix scans.
        self._req_batch_cache: dict[str, tuple[int, str]] = {}

        # ---- atomic-batch prefill state -----------------------------------
        # True when a batch's requests have been promoted to the waiting
        # queue but have not yet all completed prefill.  While this flag is
        # set, no new batch may be promoted.
        self._batch_prefill_in_progress: bool = False

        # The index of the batch that is currently being prefilled (valid
        # only when _batch_prefill_in_progress is True).
        self._active_batch_idx: int = -1

        logger.info("BATCH SCHEDULER!!!")

    # ------------------------------------------------------------------
    # Suffix matching
    # ------------------------------------------------------------------

    def _match_req_id(self, req_id: str) -> tuple[int, str] | None:
        """Return ``(batch_idx, suffix)`` for *req_id*, or *None*.

        Matching is done by splitting ``req_id`` on ``-`` and comparing the
        **last segment** against each suffix.
        """
        cached = self._req_batch_cache.get(req_id)
        if cached is not None:
            return cached if cached[0] != _NO_BATCH else None

        last_segment = req_id.rsplit("-", 1)[-1] if "-" in req_id else req_id
        batch_idx = self._suffix_to_batch.get(last_segment)
        if batch_idx is not None:
            result = (batch_idx, last_segment)
            self._req_batch_cache[req_id] = result
            return result

        self._req_batch_cache[req_id] = (_NO_BATCH, "")
        return None

    def _is_batch_member(self, req_id: str, batch_idx: int) -> bool:
        """Return True if *req_id* belongs to batch *batch_idx*."""
        match = self._match_req_id(req_id)
        return match is not None and match[0] == batch_idx

    # ------------------------------------------------------------------
    # Batch eligibility
    # ------------------------------------------------------------------

    def _is_batch_eligible(self, batch_idx: int) -> bool:
        """Return True if *batch_idx* may start prefill."""
        if batch_idx >= self._num_batches:
            return False

        # All previous batches must have completed prefill.
        for b in range(batch_idx):
            if b not in self._batch_prefill_done:
                return False

        # Every suffix of this batch must have a matching request arrived.
        expected = set(self._batch_order[batch_idx])
        arrived = self._batch_suffixes_arrived[batch_idx]
        return expected == arrived

    def _find_next_eligible_batch(self) -> int | None:
        """Find the first batch index that is eligible and not yet promoted.

        Returns ``None`` if no batch is ready.
        """
        for i in range(self._num_batches):
            if i in self._batch_prefill_done:
                continue
            if self._batch_prefill_in_progress and self._active_batch_idx == i:
                continue
            if self._is_batch_eligible(i):
                return i
        return None

    def _promote_eligible_batches(self) -> None:
        """Move requests from *_pending_batch_requests* into the waiting queue.

        Only **one** batch is promoted at a time.  Before promoting we check
        that the total token requirement of every request in the batch fits
        within ``max_num_scheduled_tokens``, so that all requests in the batch
        can be prefilled in a single scheduling iteration.
        """
        # Do not promote while a batch is already being prefilled.
        if self._batch_prefill_in_progress:
            return

        batch_idx = self._find_next_eligible_batch()
        if batch_idx is None:
            return

        # ---- budget check: ensure all requests fit in one iteration --------
        suffixes = set(self._batch_order[batch_idx])
        total_new_tokens = 0
        batch_requests: list[tuple[str, Request, str]] = []
        for req_id, request in list(self._pending_batch_requests.items()):
            match = self._match_req_id(req_id)
            if match is None:
                continue
            _, suffix = match
            if suffix in suffixes:
                remaining = request.num_prompt_tokens - request.num_computed_tokens
                threshold = self.scheduler_config.long_prefill_token_threshold
                if 0 < threshold < remaining:
                    remaining = threshold
                total_new_tokens += remaining
                batch_requests.append((req_id, request, suffix))

        # The batch must fit within BOTH the per-iteration token budget and
        # the max-num-seqs cap, otherwise the underlying vLLM scheduler would
        # split it across iterations (breaking atomic batch prefill).
        if (
            total_new_tokens > self.max_num_scheduled_tokens
            or len(self.running) + len(batch_requests) > self.max_num_running_reqs
        ):
            logger.warning(
                "Batch %d requires ~%d tokens (max_num_scheduled_tokens=%d) "
                "with %d running + %d batch requests "
                "(max_num_running_reqs=%d). Delaying batch until enough "
                "budget is available.",
                batch_idx,
                total_new_tokens,
                self.max_num_scheduled_tokens,
                len(self.running),
                len(batch_requests),
                self.max_num_running_reqs,
            )
            return

        # ---- schedulability check: all requests must be WAITING -----------
        # In the async pipeline, batch requests are set to WAITING_FOR_CHUNK
        # in add_request while chunk data is being loaded.  Do not promote
        # until every request has finished loading (status == WAITING),
        # otherwise the batch would be marked "in progress" but none of its
        # requests could actually be scheduled.
        for _, request, _ in batch_requests:
            if request.status != RequestStatus.WAITING:
                logger.debug(
                    "Batch %d: request %s not ready (status=%s), "
                    "deferring promotion.",
                    batch_idx, request.request_id, request.status,
                )
                return

        # ---- promote every request of the batch ---------------------------
        promoted = 0
        for req_id, request, suffix in batch_requests:
            del self._pending_batch_requests[req_id]
            # Mark the request as "chunk ready" on the adapter so that
            # process_pending_chunks() does NOT re-register it for polling
            # via load_async() before it is actually scheduled as prefill.
            # Without this, the background recv_loop immediately fetches the
            # next (decode-phase) chunk for this request and overwrites
            # request.additional_information — dropping embed.prefill (per
            # the adapter's "keys absent from the new chunk are dropped"
            # merge rule) — which later crashes talker_preprocess_prefill
            # with `KeyError: 'prefill'`.
            if self.chunk_transfer_adapter is not None:
                self.chunk_transfer_adapter.requests_with_ready_chunks.add(req_id)
            self._enqueue_waiting_request(request)
            promoted += 1
            logger.debug(
                "Batch %d: promoted %s (suffix %s) to waiting queue.",
                batch_idx, req_id, suffix,
            )

        self._batch_prefill_in_progress = True
        self._active_batch_idx = batch_idx
        logger.info(
            "Batch %d is now eligible (%d requests promoted, ~%d tokens).",
            batch_idx, promoted, total_new_tokens,
        )

    # ------------------------------------------------------------------
    # Prefill-completion tracking
    # ------------------------------------------------------------------

    def _check_batch_prefill_completion(
        self, req_id: str, num_computed_tokens: int,
        num_tokens_scheduled: int, num_prompt_tokens: int,
    ) -> None:
        """If *req_id* just completed prefill, update batch state and
        potentially unlock the next batch."""
        match = self._match_req_id(req_id)
        if match is None:
            return
        batch_idx, suffix = match

        if suffix in self._suffix_prefill_done:
            return

        # Detect the transition: prefill was NOT done, now IS done.
        if num_computed_tokens < num_prompt_tokens:
            return
        tokens_before = num_computed_tokens - num_tokens_scheduled
        if tokens_before >= num_prompt_tokens:
            return  # prefill was already done in a previous step

        # ---- prefill just completed for this request --------------------
        self._suffix_prefill_done.add(suffix)
        logger.debug(
            "Request %s (batch %d, suffix %s) completed prefill.",
            req_id, batch_idx, suffix,
        )

        # Check if every suffix in this batch has its prefill done.
        batch_suffixes = self._batch_order[batch_idx]
        if all(s in self._suffix_prefill_done for s in batch_suffixes):
            self._batch_prefill_done.add(batch_idx)
            logger.info(
                "Batch %d prefill complete (%d/%d suffixes).",
                batch_idx, len(batch_suffixes), len(batch_suffixes),
            )
            # Clear in-progress flag so the next batch can be promoted.
            self._batch_prefill_in_progress = False
            self._active_batch_idx = -1
            self._promote_eligible_batches()

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        """Intercept batch requests; hold them until their batch is eligible."""
        req_id = request.request_id

        # Streaming-update path: delegate entirely to parent.
        existing = self.requests.get(req_id)
        if existing is not None:
            super().add_request(request)
            return

        match = self._match_req_id(req_id)
        if match is None:
            # Not part of any batch — normal processing.
            super().add_request(request)
            return

        batch_idx, suffix = match

        # ---- batch request: hold until eligible --------------------------
        # Normal setup (mirrors Scheduler.add_request).
        if request.resumable:
            request.streaming_queue = deque()
        self.requests[req_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

        # Mark arrival (tracked by suffix, not full req_id).
        self._batch_suffixes_arrived[batch_idx].add(suffix)
        self._pending_batch_requests[req_id] = request

        # Start chunk loading immediately so that by the time the batch
        # becomes eligible, data may already be ready.
        if self.chunk_transfer_adapter is not None:
            self.chunk_transfer_adapter.load_async(request)
            request.status = RequestStatus.WAITING_FOR_CHUNK

        logger.debug(
            "Batch %d: request %s arrived (suffix %s, %d/%d suffixes present).",
            batch_idx,
            req_id,
            suffix,
            len(self._batch_suffixes_arrived[batch_idx]),
            len(self._batch_order[batch_idx]),
        )

        # Check if this (or subsequent) batches just became eligible.
        self._promote_eligible_batches()

    # ------------------------------------------------------------------
    # Chunk-readiness for pending batch requests
    # ------------------------------------------------------------------

    def _process_pending_batch_chunks(self) -> None:
        """Check if pending batch requests have finished loading chunk data.

        When the chunk-transfer adapter finishes loading data for a request
        it adds the request id to ``_finished_load_reqs``.  We poll that set
        here and transition matching pending requests from
        ``WAITING_FOR_CHUNK`` back to ``WAITING`` so that
        :meth:`_promote_eligible_batches` can promote them.
        """
        if self.chunk_transfer_adapter is None:
            return

        finished = self.chunk_transfer_adapter._finished_load_reqs

        # Record the wall-clock time each request's *initial* chunk became
        # available.  This is used as the effective "arrival time" for
        # ordering (see _sort_key in schedule()), so requests are ordered by
        # when their input data actually arrived rather than submission time.
        if finished:
            now = time.time()
            for req_id in finished:
                self._req_chunk_arrival_time.setdefault(req_id, now)
                request=self.requests.get(req_id)
                request.arrival_time=now

        if not finished:
            return

        for req_id, request in list(self._pending_batch_requests.items()):
            if request.status == RequestStatus.WAITING_FOR_CHUNK:
                if req_id in finished:
                    request.status = RequestStatus.WAITING
                    finished.discard(req_id)
                    logger.debug(
                        "Pending batch request %s chunk loaded → WAITING.",
                        req_id,
                    )

        # Re-check promotion now that some requests may have become WAITING.
        self._promote_eligible_batches()

    def schedule(self):
        """Sort running by batch order, ensure atomic batch prefill,
        then delegate to parent schedule()."""
        # Check chunk readiness for pending batch requests before the
        # parent schedule loop processes waiting/running requests.
        self._process_pending_batch_chunks()

        if self.running and self._batch_order:
            def _sort_key(r: Request) -> tuple[int, float, str]:
                match = self._match_req_id(r.request_id)
                batch_idx = match[0] if match else _NO_BATCH
                # Order by initial-chunk arrival time when recorded, otherwise
                # fall back to the request submission time.
                arrival = self._req_chunk_arrival_time.get(
                    r.request_id, r.arrival_time
                )
                return (batch_idx, arrival, r.request_id)
            self.running.sort(key=_sort_key)

        # ---- all-or-nothing: ensure every batch waiting request can be
        # scheduled in this iteration ------------------------------------
        if self._batch_prefill_in_progress and self.waiting:
            # Collect batch requests currently in the waiting queue.
            batch_waiting: list[Request] = []
            non_batch_waiting: list[Request] = []
            for req in self.waiting:
                match = self._match_req_id(req.request_id)
                if match is not None and match[0] == self._active_batch_idx:
                    batch_waiting.append(req)
                else:
                    non_batch_waiting.append(req)

            if batch_waiting:
                # Compute total new tokens needed for the first scheduling
                # of every batch request (considering long_prefill threshold).
                total_new_tokens = 0
                for req in batch_waiting:
                    remaining = req.num_prompt_tokens - req.num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < remaining:
                        remaining = threshold
                    total_new_tokens += remaining

                # The batch must fit within BOTH the per-iteration token
                # budget AND the max-num-seqs cap, otherwise the underlying
                # vLLM scheduler would split it across iterations (breaking
                # atomic batch prefill).
                exceeds_tokens = total_new_tokens > self.max_num_scheduled_tokens
                exceeds_seqs = (
                    len(self.running) + len(batch_waiting) > self.max_num_running_reqs
                )
                if exceeds_tokens or exceeds_seqs:
                    # Cannot fit all in one iteration — move the ENTIRE batch
                    # back to pending to wait for a larger budget.
                    logger.warning(
                        "Batch %d waiting requests need ~%d tokens "
                        "(max_num_scheduled_tokens=%d) with %d running + %d "
                        "batch requests (max_num_running_reqs=%d). "
                        "Holding back the entire batch.",
                        self._active_batch_idx,
                        total_new_tokens,
                        self.max_num_scheduled_tokens,
                        len(self.running),
                        len(batch_waiting),
                        self.max_num_running_reqs,
                    )
                    self.waiting.remove_requests(batch_waiting)
                    for req in batch_waiting:
                        self._pending_batch_requests[req.request_id] = req
                        # Un-mark chunk readiness: the request is back in
                        # pending and not about to be scheduled.
                        if self.chunk_transfer_adapter is not None:
                            self.chunk_transfer_adapter.requests_with_ready_chunks.discard(
                                req.request_id,
                            )
                    self._batch_prefill_in_progress = False
                    self._active_batch_idx = -1
                    # Try to promote the next eligible batch instead.
                    self._promote_eligible_batches()

        # ---- reserve the whole iteration for the active batch -----------
        # While a batch prefill is in progress and its requests are still in
        # the waiting queue (i.e. the iteration in which the batch is about to
        # be scheduled), park every non-batch waiting request so the
        # underlying vLLM scheduler sees ONLY the batch requests.  Without
        # this, non-batch requests ahead of the batch in the FIFO waiting
        # queue would consume the per-iteration token budget and/or num_seqs
        # slots, splitting the batch across iterations.
        parked: list[Request] = []
        batch_ids_in_waiting: set[str] = set()
        if self._batch_prefill_in_progress and self.waiting:
            batch_ids_in_waiting = {
                req.request_id for req in self.waiting
                if self._is_batch_member(req.request_id, self._active_batch_idx)
            }
            if batch_ids_in_waiting:
                parked = [
                    req for req in self.waiting
                    if not self._is_batch_member(req.request_id, self._active_batch_idx)
                ]
                if parked:
                    self.waiting.remove_requests(parked)

        try:
            scheduler_output = super().schedule()

            # ---- post-check: the active batch must have been scheduled
            # atomically (all its waiting requests scheduled in this round).
            if self._batch_prefill_in_progress and batch_ids_in_waiting:
                scheduled_ids = {
                    getattr(nr, "req_id", None)
                    for nr in getattr(scheduler_output, "scheduled_new_reqs", []) or []
                }
                missing = [rid for rid in batch_ids_in_waiting if rid not in scheduled_ids]
                if missing:
                    logger.warning(
                        "Batch %d was NOT scheduled atomically: %d/%d requests "
                        "missing from scheduled_new_reqs (e.g. %s). This "
                        "indicates an unexpected scheduler constraint "
                        "(KV cache / num_seqs).",
                        self._active_batch_idx,
                        len(missing),
                        len(batch_ids_in_waiting),
                        missing[:3],
                    )

            return scheduler_output
        finally:
            # Restore parked requests (in original order) for the next round.
            if parked:
                for req in parked:
                    self.waiting.add_request(req)

    def update_from_output(self, scheduler_output, model_runner_output):
        """Track prefill completion for batch-aware ordering."""
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        # Call parent to do the actual update.
        result = super().update_from_output(scheduler_output, model_runner_output)

        # Scan for requests that just completed prefill.
        for req_id, num_tokens in num_scheduled_tokens.items():
            request = self.requests.get(req_id)
            if request is None:
                continue
            self._check_batch_prefill_completion(
                req_id=req_id,
                num_computed_tokens=request.num_computed_tokens,
                num_tokens_scheduled=num_tokens,
                num_prompt_tokens=request.num_prompt_tokens,
            )

        return result

    def has_unfinished_requests(self) -> bool:
        """Also count pending batch requests held outside the main queues."""
        if self._pending_batch_requests:
            return True
        return super().has_unfinished_requests()

    def finish_requests(
        self, request_ids: Any, finished_status: RequestStatus,
    ) -> list[tuple[str, int]]:
        """Clean up batch state for finished/aborted requests."""
        if isinstance(request_ids, str):
            ids_iter = (request_ids,)
        elif request_ids is not None:
            ids_iter = request_ids
        else:
            ids_iter = tuple(self.requests.keys())

        for req_id in ids_iter:
            self._pending_batch_requests.pop(req_id, None)
            self._req_chunk_arrival_time.pop(req_id, None)

            match = self._match_req_id(req_id)
            if match is not None:
                batch_idx, suffix = match
                # Treat finished requests as prefill-done so they don't
                # block their batch from advancing.
                if suffix not in self._suffix_prefill_done:
                    self._suffix_prefill_done.add(suffix)
                    # Check if this unblocks the batch.
                    batch_suffixes = self._batch_order[batch_idx]
                    if all(s in self._suffix_prefill_done for s in batch_suffixes):
                        self._batch_prefill_done.add(batch_idx)
                        logger.info(
                            "Batch %d prefill complete (request %s finished).",
                            batch_idx, req_id,
                        )
                        self._batch_prefill_in_progress = False
                        self._active_batch_idx = -1
                        self._promote_eligible_batches()

        return super().finish_requests(request_ids, finished_status)


class BatchOrderOmniARAsyncScheduler(BatchOrderOmniARScheduler, OmniARAsyncScheduler):
    """Async variant of :class:`BatchOrderOmniARScheduler`.

    Inherits the batch-barrier logic from ``BatchOrderOmniARScheduler``;
    async scheduling machinery comes from ``OmniARAsyncScheduler``.
    """
