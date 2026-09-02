from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Callable

from vllm.logger import init_logger
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
)
from vllm.v1.request import Request

from vllm_omni.core.sched.omni_ar_scheduler import (
    OmniARAsyncScheduler,
    OmniARScheduler,
)

logger = init_logger(__name__)

# Default sort key: sort by arrival_time, then request_id as tiebreaker.
_DEFAULT_SORT_KEY: Callable[[Request], tuple[float, str]] = (
    lambda r: (r.arrival_time, r.request_id)
)


class FixedOrderRequestQueue(RequestQueue):
    """A request queue that maintains requests in a fixed, deterministic order.

    Requests are always ordered by a configurable ``sort_key`` (default: arrival
    time, then request_id).  Unlike ``PriorityRequestQueue``, this queue ignores
    the ``priority`` field entirely — the ordering depends ONLY on the sort key.
    """

    def __init__(
        self,
        sort_key: Callable[[Request], object] | None = None,
    ) -> None:
        self._list: list[Request] = []
        self._sort_key = sort_key or _DEFAULT_SORT_KEY

    # ---- helpers -----------------------------------------------------------

    def _sorted_insert(self, request: Request) -> None:
        """Insert *request* so that ``_list`` stays sorted by ``_sort_key``."""
        key = self._sort_key(request)
        # linear scan — queue sizes are typically small enough for this to be fine
        for i, existing in enumerate(self._list):
            if key < self._sort_key(existing):
                self._list.insert(i, request)
                return
        self._list.append(request)

    # ---- RequestQueue interface --------------------------------------------

    def add_request(self, request: Request) -> None:
        self._sorted_insert(request)

    def pop_request(self) -> Request:
        if not self._list:
            raise IndexError("pop from an empty queue")
        return self._list.pop(0)

    def peek_request(self) -> Request:
        if not self._list:
            raise IndexError("peek from an empty queue")
        return self._list[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend — still respects sort order, so this is equivalent to add."""
        self._sorted_insert(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for req in requests:
            self._sorted_insert(req)

    def remove_request(self, request: Request) -> None:
        self._list.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        to_remove = requests if isinstance(requests, set) else set(requests)
        self._list = [r for r in self._list if r not in to_remove]

    def __bool__(self) -> bool:
        return bool(self._list)

    def __len__(self) -> int:
        return len(self._list)

    def __iter__(self) -> Iterator[Request]:
        return iter(self._list)


class FixedOrderOmniARScheduler(OmniARScheduler):
    """Omni AR scheduler that guarantees a **fixed, deterministic** processing
    order for requests.

    Every scheduling step:

    1. **Running requests** are sorted by *arrival_time* (then *request_id*)
       before tokens are allocated — so earlier-arriving requests always get
       token budget first.
    2. **Waiting / skipped queues** use ``FixedOrderRequestQueue`` which ignores
       ``priority`` and keeps requests in arrival order at all times.
    3. **Queue selection** compares heads of both waiting queues to always pick
       the earliest-arriving request, regardless of which queue it sits in.

    This makes the scheduler fully FCFS — priority-based reordering and
    arbitrary list-order dependencies are removed.
    """

    def __init__(self, *args, sort_key=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Replace the waiting queues with fixed-order versions.
        self._replace_with_fixed_order_queue("waiting", sort_key)
        self._replace_with_fixed_order_queue("skipped_waiting", sort_key)

        # Keep the sort key for use inside schedule().
        self._fixed_order_sort_key = sort_key or _DEFAULT_SORT_KEY

        # Override policy to FCFS so that internal create_request_queue calls
        # inside schedule() also produce deterministic queues.
        self.policy = SchedulingPolicy.FCFS

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _replace_with_fixed_order_queue(
        self,
        attr: str,
        sort_key: Callable[[Request], object] | None,
    ) -> None:
        """Drain *attr* queue and replace it with a ``FixedOrderRequestQueue``,
        re-inserting all queued requests in fixed order."""
        old_queue: RequestQueue = getattr(self, attr)
        new_queue = FixedOrderRequestQueue(sort_key=sort_key)
        while old_queue:
            try:
                req = old_queue.pop_request()
            except IndexError:
                break
            new_queue.add_request(req)
        setattr(self, attr, new_queue)
        logger.debug(
            "Replaced self.%s with FixedOrderRequestQueue (%d requests).",
            attr,
            len(new_queue),
        )

    def _compare_queue_heads(
        self, q1: RequestQueue, q2: RequestQueue
    ) -> RequestQueue | None:
        """Return the queue whose head request sorts earlier (or *q1* on tie).

        Returns *None* if both queues are empty.
        """
        if not q1 and not q2:
            return None
        if not q1:
            return q2
        if not q2:
            return q1
        k1 = self._fixed_order_sort_key(q1.peek_request())
        k2 = self._fixed_order_sort_key(q2.peek_request())
        return q1 if k1 <= k2 else q2

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        """Compare heads of both fixed-order queues so the earliest-arriving
        request across *waiting* and *skipped_waiting* is always chosen."""
        return self._compare_queue_heads(self.waiting, self.skipped_waiting)

    def schedule(self):
        """Sort running requests before delegating to the parent schedule()."""
        # ---- sort running by fixed key ----------------------------------
        if self.running:
            self.running.sort(key=self._fixed_order_sort_key)

        # ---- delegate to OmniARScheduler.schedule() --------------------
        return super().schedule()


class FixedOrderOmniARAsyncScheduler(FixedOrderOmniARScheduler, OmniARAsyncScheduler):
    """Async variant of :class:`FixedOrderOmniARScheduler`.

    Inherits the fixed-order __init__ and schedule() from
    ``FixedOrderOmniARScheduler``; the async machinery comes from
    ``OmniARAsyncScheduler``.
    """
