"""Online ridge model for AR batch execution time.

The GPU runner records asynchronous CUDA-event samples.  A trainer fits one
model per Omni stage and persists it as JSON; scheduler processes periodically
reload the file without sharing Python state with the worker.
"""

from __future__ import annotations

import atexit
import fcntl
import json
import logging
import math
import os
import queue
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vllm_omni.config.future_scheduler_config import load_future_scheduler_config

logger = logging.getLogger(__name__)

FEATURE_NAMES = (
    "intercept",
    "N_per_1k",
    "A_per_1m",
    "log2_1_plus_B",
    "M_per_1k",
    "C_per_1k",
    "S_per_1k",
    "D",
    "P",
    "path_decode_full",
    "path_chunk_none",
    "path_mixed_none",
    "path_mixed_piecewise",
)
BASIC_FEATURE_NAMES = FEATURE_NAMES[:9]


def path_name_from_vector(values: Sequence[float]) -> str:
    flags = values[9:13]
    names = ("decode_full", "chunk_none", "mixed_none", "mixed_piecewise")
    for name, flag in zip(names, flags):
        if flag > 0.5:
            return name
    return "prefill_none"


def static_prediction_ms(stage_id: int, values: Sequence[float]) -> float:
    """Evaluate the original fixed model for holdout acceptance checks."""
    if stage_id == 0:
        base = 61.9975
        beta = (67.2946, -5.4266, -12.0575, 11.3847, -1.8995, 11.9480, 8.0227)
        offsets = {
            "decode_full": -89.4193,
            "prefill_none": 0.0,
            "chunk_none": -69.8116,
            "mixed_none": -132.0535,
            "mixed_piecewise": -116.9849,
        }
    else:
        base = 78.7162
        beta = (-0.7459, 2.3164, -1.2926, 0.1236, -0.0083, 0.6791, -2.2023)
        offsets = {
            "decode_full": -75.6408,
            "prefill_none": 0.0,
            "chunk_none": -54.7430,
            "mixed_none": -44.3215,
            "mixed_piecewise": -14.1843,
        }
    value = (
        base
        + beta[0] * values[1]
        + beta[1] * values[2]
        + beta[2] * values[3]
        + beta[3] * values[4]
        + beta[4] * values[5]
        + beta[5] * values[7]
        + beta[6] * values[8]
        + offsets[path_name_from_vector(values)]
    )
    return max(float(value), 0.05)


def model_path_from_env() -> Path:
    """Return the configured model path (name retained for compatibility)."""
    return Path(load_future_scheduler_config().prediction.model_path)


def feature_vector(
    *,
    n: int,
    a: int,
    b: int,
    m: int,
    c: int,
    s: int,
    d: int,
    p: int,
    state: str,
    mode: str,
) -> list[float]:
    path = (state, mode)
    return [
        1.0,
        n / 1e3,
        a / 1e6,
        math.log2(1 + max(b, 0)),
        m / 1e3,
        c / 1e3,
        s / 1e3,
        float(d),
        float(p),
        float(path == ("decode", "FULL")),
        float(path == ("chunk", "NONE")),
        float(path == ("mixed", "NONE")),
        float(path == ("mixed", "PIECEWISE")),
    ]


def record_feature_vector(record: Mapping[str, Any], graph_max_tokens: int) -> list[float] | None:
    items: list[tuple[int, int, int]] = []
    for request in record.get("requests", []):
        try:
            q = int(request["scheduled_tokens"])
            c = int(request["computed_tokens_before"])
            prompt = int(request["prompt_tokens"])
        except (KeyError, TypeError, ValueError):
            return None
        if q > 0:
            items.append((q, max(c, 0), max(prompt, q)))
    if not items:
        return None

    has_prefill = any(q > 1 for q, _, _ in items)
    has_decode = any(q == 1 and c >= prompt for q, c, prompt in items)
    if has_prefill and has_decode:
        state = "mixed"
    elif has_prefill:
        state = "chunk" if any(c > 0 for q, c, _ in items if q > 1) else "prefill"
    else:
        state = "decode"
    n = sum(q for q, _, _ in items)
    if state == "decode":
        mode = "FULL"
    elif state == "mixed" and n <= graph_max_tokens:
        mode = "PIECEWISE"
    else:
        mode = "NONE"
    return feature_vector(
        n=n,
        a=sum(q * c + q * (q + 1) // 2 for q, c, _ in items),
        b=len(items),
        m=max(c + q for q, c, _ in items),
        c=sum(c for _, c, _ in items),
        s=sum(c + q for q, c, _ in items),
        d=sum(q for q, c, prompt in items if q == 1 and c >= prompt),
        p=sum(1 for q, _, _ in items if q > 1),
        state=state,
        mode=mode,
    )


class OnlineCostModelTrainer:
    """Collect resolved CUDA-event samples and periodically refit a ridge model."""

    def __init__(self, stage_id: int, graph_max_tokens: int) -> None:
        prediction_config = load_future_scheduler_config().prediction
        self.stage_id = int(stage_id)
        self.graph_max_tokens = int(graph_max_tokens)
        self.model_path = Path(prediction_config.model_path)
        self.sample_path = Path(f"{self.model_path}.stage{self.stage_id}.samples.jsonl")
        self.min_samples = prediction_config.min_samples
        self.refit_interval = prediction_config.refit_interval
        self.max_samples = prediction_config.max_samples
        self.ridge = prediction_config.ridge
        self.min_path_samples = prediction_config.min_path_samples
        self.min_validation_samples = prediction_config.min_validation_samples
        self.accept_ratio = prediction_config.accept_ratio
        self.samples: list[tuple[list[float], float]] = []
        self._observations_seen = 0
        self._last_fit_observation = 0
        self._samples_since_compaction = 0
        self._closed = False
        self._load_samples()
        self._observations_seen = len(self.samples)
        self._maybe_fit(force=True)
        self._sample_queue: queue.Queue[tuple[Mapping[str, Any], float] | None] = (
            queue.Queue()
        )
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"omni-cost-fit-stage-{self.stage_id}",
            daemon=True,
        )
        self._worker.start()
        atexit.register(self.close)

    def _load_samples(self) -> None:
        try:
            with self.sample_path.open("r", encoding="utf-8") as file:
                lines = deque(file, maxlen=self.max_samples)
        except FileNotFoundError:
            return
        for line in lines:
            try:
                row = json.loads(line)
                x = [float(value) for value in row["x"]]
                y = float(row["duration_ms"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if len(x) == len(FEATURE_NAMES) and math.isfinite(y) and y > 0:
                self.samples.append((x, y))

    def observe(self, record: Mapping[str, Any], duration_ms: float) -> None:
        """Queue a sample without doing file I/O or fitting on the hot path."""
        if not self._closed:
            self._sample_queue.put((dict(record), float(duration_ms)))

    def _worker_loop(self) -> None:
        while True:
            item = self._sample_queue.get()
            try:
                if item is None:
                    return
                self._consume(*item)
            except Exception:
                logger.exception("Failed to update online cost model stage=%s", self.stage_id)
            finally:
                self._sample_queue.task_done()

    def _consume(self, record: Mapping[str, Any], duration_ms: float) -> None:
        if not math.isfinite(duration_ms) or not 0.01 <= duration_ms <= 120_000:
            return
        x = record_feature_vector(record, self.graph_max_tokens)
        if x is None:
            return
        row = {
            "timestamp": time.time(),
            "stage_id": self.stage_id,
            "duration_ms": float(duration_ms),
            "x": x,
            "feature_names": FEATURE_NAMES,
            "batch_id": record.get("batch_id"),
        }
        self.sample_path.parent.mkdir(parents=True, exist_ok=True)
        with self.sample_path.open("a", encoding="utf-8") as file:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        self.samples.append((x, float(duration_ms)))
        self._observations_seen += 1
        if len(self.samples) > self.max_samples:
            self.samples = self.samples[-self.max_samples :]
        self._samples_since_compaction += 1
        if self._samples_since_compaction >= self.max_samples:
            self._compact_sample_file()
            self._samples_since_compaction = 0
        self._maybe_fit()

    def _compact_sample_file(self) -> None:
        """Bound the persistent JSONL size to the configured fitting window."""
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.sample_path.parent, delete=False
        ) as temp_file:
            for x, duration_ms in self.samples:
                temp_file.write(
                    json.dumps(
                        {
                            "timestamp": time.time(),
                            "stage_id": self.stage_id,
                            "duration_ms": duration_ms,
                            "x": x,
                            "feature_names": FEATURE_NAMES,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            temp_name = temp_file.name
        os.replace(temp_name, self.sample_path)

    def wait_until_idle(self) -> None:
        """Testing/maintenance hook; never called by the forward hot path."""
        self._sample_queue.join()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._sample_queue.put(None)
        self._sample_queue.join()
        self._worker.join(timeout=5)

    def _maybe_fit(self, *, force: bool = False) -> None:
        count = len(self.samples)
        if count < self.min_samples:
            return
        if (
            not force
            and self._observations_seen - self._last_fit_observation
            < self.refit_interval
        ):
            return
        grouped: dict[str, list[tuple[list[float], float]]] = {}
        for sample in self.samples:
            grouped.setdefault(path_name_from_vector(sample[0]), []).append(sample)
        path_models: dict[str, dict[str, Any]] = {}
        path_counts: dict[str, int] = {}
        for path in (
            "prefill_none",
            "decode_full",
            "chunk_none",
            "mixed_none",
            "mixed_piecewise",
        ):
            path_samples = grouped.get(path, [])
            path_counts[path] = len(path_samples)
            if len(path_samples) < self.min_path_samples:
                continue
            validation_count = max(
                self.min_validation_samples,
                int(math.ceil(len(path_samples) * 0.2)),
            )
            if validation_count >= len(path_samples):
                continue
            train_samples = path_samples[:-validation_count]
            validation_samples = path_samples[-validation_count:]
            coefficients = self._fit_coefficients(train_samples)
            validation = self._evaluate(coefficients, validation_samples)
            static_errors = [
                abs(y - static_prediction_ms(self.stage_id, x))
                for x, y in validation_samples
            ]
            static_mae = float(np.mean(static_errors))
            accepted = validation["mae_ms"] <= static_mae * self.accept_ratio
            if accepted:
                coefficients = self._fit_coefficients(path_samples)
            path_models[path] = {
                "accepted": accepted,
                "feature_names": BASIC_FEATURE_NAMES,
                "coefficients": coefficients.tolist(),
                "sample_count": len(path_samples),
                "train_count": len(train_samples),
                "validation_count": validation_count,
                "validation_metrics": validation,
                "static_validation_mae_ms": static_mae,
            }
        accepted_paths = [
            path for path, model in path_models.items() if model["accepted"]
        ]
        payload = {
            "schema_version": 2,
            "stage_id": self.stage_id,
            "feature_names": FEATURE_NAMES,
            "model_feature_names": BASIC_FEATURE_NAMES,
            "sample_count": count,
            "path_counts": path_counts,
            "path_models": path_models,
            "accepted_paths": accepted_paths,
            "ridge": self.ridge,
            "updated_at": time.time(),
        }
        self._write_stage_model(payload)
        self._last_fit_observation = self._observations_seen
        logger.info(
            "Online cost model fitted stage=%s samples=%s accepted_paths=%s",
            self.stage_id,
            count,
            accepted_paths,
        )

    def _fit_coefficients(
        self, samples: Sequence[tuple[list[float], float]]
    ) -> np.ndarray:
        x = np.asarray(
            [sample[0][: len(BASIC_FEATURE_NAMES)] for sample in samples],
            dtype=np.float64,
        )
        y = np.asarray([sample[1] for sample in samples], dtype=np.float64)
        penalty = np.eye(x.shape[1], dtype=np.float64) * self.ridge
        penalty[0, 0] = 0.0
        return np.linalg.pinv(x.T @ x + penalty) @ x.T @ y

    @staticmethod
    def _evaluate(
        coefficients: np.ndarray,
        samples: Sequence[tuple[list[float], float]],
    ) -> dict[str, float]:
        x = np.asarray(
            [sample[0][: len(BASIC_FEATURE_NAMES)] for sample in samples],
            dtype=np.float64,
        )
        y = np.asarray([sample[1] for sample in samples], dtype=np.float64)
        predictions = np.maximum(x @ coefficients, 0.05)
        residuals = y - predictions
        return {
            "mae_ms": float(np.mean(np.abs(residuals))),
            "rmse_ms": float(np.sqrt(np.mean(residuals**2))),
            "mape_pct": float(
                np.mean(np.abs(residuals) / np.maximum(y, 0.1)) * 100
            ),
        }

    def _write_stage_model(self, stage_payload: Mapping[str, Any]) -> None:
        path = self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = Path(f"{path}.lock")
        with lock_path.open("a", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                try:
                    document = json.loads(path.read_text(encoding="utf-8"))
                except (FileNotFoundError, json.JSONDecodeError):
                    document = {"schema_version": 1, "stages": {}}
                document.setdefault("stages", {})[str(self.stage_id)] = dict(stage_payload)
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=path.parent, delete=False
                ) as temp_file:
                    json.dump(document, temp_file, ensure_ascii=False, indent=2)
                    temp_name = temp_file.name
                os.replace(temp_name, path)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class OnlineCostModelPredictor:
    """Reload a persisted stage model when its file changes."""

    def __init__(self, stage_id: int) -> None:
        self.stage_id = int(stage_id)
        self.path = model_path_from_env()
        self.path_models: dict[str, np.ndarray] = {}
        self.metadata: dict[str, Any] = {}
        self._mtime_ns = -1
        self._last_check = 0.0

    @property
    def ready(self) -> bool:
        self.reload()
        return bool(self.path_models)

    def reload(self) -> None:
        now = time.monotonic()
        if now - self._last_check < 0.25:
            return
        self._last_check = now
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return
        if stat.st_mtime_ns == self._mtime_ns:
            return
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
            stage = document["stages"][str(self.stage_id)]
            if tuple(stage["feature_names"]) != FEATURE_NAMES:
                raise ValueError("feature schema mismatch")
            if tuple(stage["model_feature_names"]) != BASIC_FEATURE_NAMES:
                raise ValueError("model feature schema mismatch")
            path_models = {}
            for path, model in stage.get("path_models", {}).items():
                if not model.get("accepted", False):
                    continue
                coefficients = np.asarray(model["coefficients"], dtype=np.float64)
                if coefficients.shape != (len(BASIC_FEATURE_NAMES),):
                    raise ValueError("coefficient count mismatch")
                path_models[path] = coefficients
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            logger.warning("Cannot load online cost model %s: %s", self.path, error)
            self._mtime_ns = stat.st_mtime_ns
            return
        was_ready = bool(self.path_models)
        self.path_models = path_models
        self.metadata = dict(stage)
        self._mtime_ns = stat.st_mtime_ns
        if not was_ready:
            logger.info(
                "Online cost model activated stage=%s samples=%s paths=%s",
                self.stage_id,
                stage.get("sample_count"),
                sorted(path_models),
            )

    def predict(self, values: Sequence[float]) -> float | None:
        self.reload()
        if len(values) != len(FEATURE_NAMES):
            return None
        path = path_name_from_vector(values)
        coefficients = self.path_models.get(path)
        if coefficients is None:
            return None
        value = float(
            np.dot(
                coefficients,
                np.asarray(values[: len(BASIC_FEATURE_NAMES)], dtype=np.float64),
            )
        )
        return max(value, 0.05) if math.isfinite(value) else None
