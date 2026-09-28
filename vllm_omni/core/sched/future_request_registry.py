"""Cross-process registry for requests that are expected by a downstream stage.

The stage processes in vLLM-Omni are separate processes.  A small SQLite
database in /dev/shm is a better fit than a Python singleton: it is visible to
all stage processes, survives multiprocessing spawn, and does not add a
network service to the serving path.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FutureRequestHint:
    request_id: str
    source_stage: int
    target_stage: int
    sequence: int
    published_at: float
    ready_at: float
    payload: dict[str, Any]


class FutureRequestRegistry:
    """A best-effort, TTL-backed cross-process future-request registry."""

    def __init__(self, path: str | None = None, ttl_s: float = 120.0):
        raw_path = path or os.getenv(
            "VLLM_OMNI_FUTURE_REGISTRY_PATH",
            "/dev/shm/vllm_omni_future_requests.sqlite3",
        )
        self.path = Path(raw_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_s = max(float(ttl_s), 1.0)
        self._conn: sqlite3.Connection | None = None
        # Lightweight diagnostics consumed by the future-aware scheduler.
        # This intentionally records only counts, never request payloads.
        self.last_query_stats: dict[str, int] = {}
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.path),
                timeout=0.05,
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA busy_timeout=50")
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _initialize(self) -> None:
        try:
            conn = self._connect()
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS future_request_hints (
                    source_stage INTEGER NOT NULL,
                    target_stage INTEGER NOT NULL,
                    request_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    published_at REAL NOT NULL,
                    ready_at REAL NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (source_stage, target_stage, request_id, sequence)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_future_hints_target "
                "ON future_request_hints(target_stage, published_at)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS future_stage_states (
                    stage_id INTEGER PRIMARY KEY,
                    updated_at REAL NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.commit()
        except sqlite3.Error:
            # The scheduler must remain usable if the optional registry is
            # temporarily unavailable.  Calls below are also best effort.
            return

    def purge(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        cutoff = now - self.ttl_s
        try:
            conn = self._connect()
            conn.execute(
                "DELETE FROM future_request_hints WHERE published_at < ?",
                (cutoff,),
            )
            conn.commit()
        except sqlite3.Error:
            return

    def publish_stage_state(
        self,
        *,
        stage_id: int,
        payload: dict[str, Any],
        updated_at: float | None = None,
    ) -> None:
        """Publish a compact queue/service snapshot for ETA estimation."""
        updated_at = time.time() if updated_at is None else updated_at
        try:
            conn = self._connect()
            conn.execute(
                """
                INSERT OR REPLACE INTO future_stage_states
                (stage_id, updated_at, payload)
                VALUES (?, ?, ?)
                """,
                (
                    int(stage_id),
                    float(updated_at),
                    json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
                ),
            )
            conn.commit()
        except sqlite3.Error:
            return

    def get_stage_state(
        self,
        stage_id: int,
        *,
        now: float | None = None,
        max_age_s: float = 2.0,
    ) -> dict[str, Any] | None:
        """Return the latest non-stale queue/service snapshot for a stage."""
        now = time.time() if now is None else now
        try:
            row = self._connect().execute(
                """
                SELECT updated_at, payload
                FROM future_stage_states
                WHERE stage_id = ?
                """,
                (int(stage_id),),
            ).fetchone()
        except sqlite3.Error:
            return None
        if row is None or now - float(row[0]) > max(float(max_age_s), 0.0):
            return None
        try:
            payload = json.loads(row[1])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        payload["updated_at"] = float(row[0])
        return payload

    def publish(
        self,
        *,
        request_id: str,
        source_stage: int,
        target_stage: int,
        sequence: int,
        ready_at: float,
        payload: dict[str, Any],
        published_at: float | None = None,
    ) -> None:
        published_at = time.time() if published_at is None else published_at
        try:
            conn = self._connect()
            conn.execute(
                """
                INSERT OR REPLACE INTO future_request_hints
                (source_stage, target_stage, request_id, sequence,
                 published_at, ready_at, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(source_stage),
                    int(target_stage),
                    str(request_id),
                    int(sequence),
                    float(published_at),
                    float(ready_at),
                    json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
                ),
            )
            conn.commit()
        except sqlite3.Error:
            return

    def get_pending(
        self,
        *,
        target_stage: int,
        known_request_ids: set[str] | None = None,
        now: float | None = None,
        ready_after: float | None = None,
        max_items: int = 64,
    ) -> list[FutureRequestHint]:
        now = time.time() if now is None else now
        cutoff = now - self.ttl_s
        known_request_ids = known_request_ids or set()
        self.last_query_stats = {
            "raw_rows": 0,
            "known_filtered": 0,
            "malformed_filtered": 0,
            "newer_replaced": 0,
        }
        self.purge(now)
        try:
            conn = self._connect()
            fetch_limit = max(
                int(max_items) * 8,
                int(max_items) + len(known_request_ids) * 4,
            )
            if ready_after is None:
                ready_filter = ""
                parameters = (int(target_stage), float(cutoff), fetch_limit)
            else:
                ready_filter = "AND ready_at > ?"
                parameters = (
                    int(target_stage),
                    float(cutoff),
                    float(ready_after),
                    fetch_limit,
                )
            records = conn.execute(
                f"""
                WITH ranked AS (
                    SELECT request_id, source_stage, target_stage, sequence,
                           published_at, ready_at, payload,
                           ROW_NUMBER() OVER (
                               PARTITION BY request_id
                               ORDER BY published_at DESC, sequence DESC
                           ) AS publication_rank
                    FROM future_request_hints
                    WHERE target_stage = ? AND published_at >= ?
                )
                SELECT request_id, source_stage, target_stage, sequence,
                       published_at, ready_at, payload
                FROM ranked
                WHERE publication_rank = 1 {ready_filter}
                ORDER BY published_at DESC, sequence DESC
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            self.last_query_stats["raw_rows"] = len(records)
        except sqlite3.Error:
            return []

        # Keep only the newest announcement for each request.  A chunked
        # request can be published at every upstream scheduler step.
        newest: dict[str, FutureRequestHint] = {}
        for row in records:
            request_id = str(row[0])
            if request_id in known_request_ids:
                self.last_query_stats["known_filtered"] += 1
                continue
            try:
                hint = FutureRequestHint(
                    request_id=request_id,
                    source_stage=int(row[1]),
                    target_stage=int(row[2]),
                    sequence=int(row[3]),
                    published_at=float(row[4]),
                    ready_at=float(row[5]),
                    payload=json.loads(row[6]),
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                self.last_query_stats["malformed_filtered"] += 1
                continue
            previous = newest.get(request_id)
            hint_order = (hint.published_at, hint.sequence)
            previous_order = (
                (previous.published_at, previous.sequence)
                if previous is not None
                else None
            )
            if previous is None or hint_order > previous_order:
                if previous is not None:
                    self.last_query_stats["newer_replaced"] += 1
                newest[request_id] = hint
        return sorted(newest.values(), key=lambda item: (item.ready_at, item.sequence))[:max_items]
