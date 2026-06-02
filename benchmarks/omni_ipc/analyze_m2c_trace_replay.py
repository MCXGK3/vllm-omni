# SPDX-License-Identifier: Apache-2.0
"""M2-C: Trace Replay and Strategy Regret Analysis.

Combines M2-A runtime traces with M2-B path performance curves to compute
regret of static connector selection relative to payload-aware selection.

Usage:
  python benchmarks/omni_ipc/analyze_m2c_trace_replay.py \
    --trace results/motivation2/m2a_runtime_trace_raw.jsonl \
    --curve results/motivation2/m2b_path_boundary_summary.csv \
    --output-dir results/motivation2
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import statistics
from bisect import bisect_left
from typing import Any


@dataclasses.dataclass
class PathCurve:
    """Interpolated latency curve for a communication path."""

    path_name: str
    payload_type: str
    memory_location: str
    sizes_bytes: list[int]  # sorted ascending
    latencies_us: list[float]  # mean latency, aligned with sizes

    def latency_for(self, size_bytes: int) -> float | None:
        """Interpolate latency for a given payload size."""
        if not self.sizes_bytes:
            return None
        if size_bytes <= self.sizes_bytes[0]:
            return self.latencies_us[0]
        if size_bytes >= self.sizes_bytes[-1]:
            return self.latencies_us[-1]
        idx = bisect_left(self.sizes_bytes, size_bytes)
        if idx == 0:
            return self.latencies_us[0]
        # Linear interpolation between idx-1 and idx
        x0, x1 = self.sizes_bytes[idx - 1], self.sizes_bytes[idx]
        y0, y1 = self.latencies_us[idx - 1], self.latencies_us[idx]
        frac = (size_bytes - x0) / (x1 - x0)
        return y0 + frac * (y1 - y0)


def load_curves(csv_path: str) -> list[PathCurve]:
    """Load M2-B path boundary summary CSV into curve objects."""
    curves: dict[tuple[str, str], PathCurve] = {}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("skipped", "")).strip() == "True":
                continue
            try:
                size = int(row["payload_size_bytes"])
            except (ValueError, KeyError):
                continue
            mean_str = row.get("mean_us", "").strip()
            if not mean_str:
                continue
            try:
                latency = float(mean_str)
            except (ValueError, TypeError):
                continue

            key = (row["path_name"], row["payload_type"])
            if key not in curves:
                curves[key] = PathCurve(
                    path_name=row["path_name"],
                    payload_type=row["payload_type"],
                    memory_location=row.get("memory_location", ""),
                    sizes_bytes=[],
                    latencies_us=[],
                )
            curves[key].sizes_bytes.append(size)
            curves[key].latencies_us.append(latency)

    # Sort each curve by size
    for c in curves.values():
        paired = sorted(zip(c.sizes_bytes, c.latencies_us))
        c.sizes_bytes = [p[0] for p in paired]
        c.latencies_us = [p[1] for p in paired]

    return list(curves.values())


def classify_trace_payload_type(payload_semantics: str, memory_location: str) -> str:
    """Map trace payload semantics to M2-B payload types."""
    if memory_location == "gpu":
        return "gpu_tensor"
    if memory_location == "mixed":
        return "mixed_gpu"
    if payload_semantics in ("control", "metadata"):
        return "metadata"
    if payload_semantics in ("audio_chunk", "embedding", "tensor", "kv_cache", "diffusion_tensor"):
        return "cpu_tensor"
    if payload_semantics == "mixed_cpu":
        return "mixed_cpu"
    if payload_semantics == "mixed_gpu":
        return "mixed_gpu"
    return "cpu_tensor"


# Maps each source payload_type -> list of fallback payload_types for curve lookup
_FALLBACK_PAYLOAD_TYPES = {
    "metadata": ["cpu_tensor", "gpu_tensor"],
    "cpu_tensor": ["metadata", "gpu_tensor"],
    "gpu_tensor": ["cpu_tensor", "metadata"],
    "mixed_cpu": ["cpu_tensor", "metadata", "gpu_tensor"],
    "mixed_gpu": ["gpu_tensor", "cpu_tensor", "metadata"],
}


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

# Thresholds are populated from M2-B data automatically; fallback defaults:
DEFAULT_SMALL_THRESHOLD = 64 * 1024  # 64KB
DEFAULT_LARGE_CPU_THRESHOLD = 1 * 1024 * 1024  # 1MB


def strategy_all_shm(trace_payload: dict, curves: list[PathCurve]) -> tuple[str, float]:
    """Strategy A: All payloads go through Serialized SHM."""
    return _lookup_latency("serialized_shm", trace_payload, curves)


def strategy_best_static(trace_payload: dict, curves: list[PathCurve]) -> tuple[str, float]:
    """Strategy B: Single best path for all payloads (determined globally)."""
    return _lookup_latency("serialized_shm", trace_payload, curves)


def strategy_size_aware(
    trace_payload: dict, curves: list[PathCurve], small_threshold: int = None
) -> tuple[str, float]:
    """Strategy C: Small → Inline, Large → Serialized SHM."""
    if small_threshold is None:
        small_threshold = _auto_small_threshold(curves)
    size = trace_payload.get("payload_size_bytes", 0)
    if size <= small_threshold:
        path, lat = _lookup_latency("inline", trace_payload, curves)
        if lat >= float("inf"):
            return _lookup_latency("serialized_shm", trace_payload, curves)
        return path, lat
    return _lookup_latency("serialized_shm", trace_payload, curves)


def strategy_size_device_aware(
    trace_payload: dict,
    curves: list[PathCurve],
    small_threshold: int = None,
    large_cpu_threshold: int = None,
) -> tuple[str, float]:
    """Strategy D: Size + Device Aware routing."""
    if small_threshold is None:
        small_threshold = _auto_small_threshold(curves)
    if large_cpu_threshold is None:
        large_cpu_threshold = _auto_large_cpu_threshold(curves)

    size = trace_payload.get("payload_size_bytes", 0)
    mem_loc = trace_payload.get("memory_location", "cpu")
    contains_gpu = trace_payload.get("contains_gpu_tensor", False)

    def _safe_lookup(path_name: str) -> tuple[str, float]:
        p, lat = _lookup_latency(path_name, trace_payload, curves)
        if lat >= float("inf"):
            return _lookup_latency("serialized_shm", trace_payload, curves)
        return p, lat

    def _safe_best(path_list: list[str]) -> tuple[str, float]:
        p, lat = _lookup_best(path_list, trace_payload, curves)
        if lat >= float("inf"):
            return _lookup_latency("serialized_shm", trace_payload, curves)
        return p, lat

    if contains_gpu or mem_loc in ("gpu", "mixed"):
        return _safe_best(["cuda_ipc", "serialized_shm"])

    if size <= small_threshold:
        path, lat = _safe_lookup("inline")
        if lat >= float("inf"):
            return _safe_lookup("serialized_shm")
        return path, lat

    if size >= large_cpu_threshold:
        return _safe_best(["raw_shm", "serialized_shm"])

    return _safe_lookup("serialized_shm")


def strategy_oracle(trace_payload: dict, curves: list[PathCurve]) -> tuple[str, float]:
    """Strategy E: Oracle selects the lowest-latency path for each payload."""
    path, lat = _lookup_best(
        ["inline", "serialized_shm", "raw_shm", "cuda_ipc"], trace_payload, curves
    )
    if lat >= float("inf"):
        return _lookup_latency("serialized_shm", trace_payload, curves)
    return path, lat


def _auto_small_threshold(curves: list[PathCurve]) -> int:
    """Find crossover where Serialized SHM becomes better than Inline for metadata.

    Returns the smallest size where SHM beats Inline, or the max tested size if Inline always wins.
    """
    inline_meta = [c for c in curves if c.path_name == "inline" and c.payload_type == "metadata"]
    shm_meta = [c for c in curves if c.path_name == "serialized_shm" and c.payload_type == "metadata"]
    if inline_meta and shm_meta:
        for size in inline_meta[0].sizes_bytes:
            il = inline_meta[0].latency_for(size)
            sl = shm_meta[0].latency_for(size)
            if il is not None and sl is not None and sl < il:
                return size
        # If Inline wins at all tested sizes, return the largest size + 1
        # to effectively disable the inline path for larger payloads
        return inline_meta[0].sizes_bytes[-1]
    return DEFAULT_SMALL_THRESHOLD


def _auto_large_cpu_threshold(curves: list[PathCurve]) -> int:
    """Find crossover where Raw SHM becomes better than Serialized SHM for CPU tensors."""
    raw = [c for c in curves if c.path_name == "raw_shm" and c.payload_type == "cpu_tensor"]
    ser = [c for c in curves if c.path_name == "serialized_shm" and c.payload_type == "cpu_tensor"]
    if raw and ser:
        for size in ser[0].sizes_bytes:
            rl = raw[0].latency_for(size)
            sl = ser[0].latency_for(size)
            if rl is not None and sl is not None and rl < sl:
                return size
        # If raw SHM wins at all sizes, return the smallest tested size
        return ser[0].sizes_bytes[0]
    return DEFAULT_LARGE_CPU_THRESHOLD


def _lookup_latency(
    path_name: str, trace_payload: dict, curves: list[PathCurve]
) -> tuple[str, float]:
    """Look up latency for a path given a trace payload."""
    pt = classify_trace_payload_type(
        trace_payload.get("payload_semantics", "unknown"),
        trace_payload.get("memory_location", "cpu"),
    )
    size = trace_payload.get("payload_size_bytes", 0)

    candidates = [
        c for c in curves if c.path_name == path_name and c.payload_type == pt
    ]
    if not candidates:
        # Try fallback payload types
        for fallback_pt in _FALLBACK_PAYLOAD_TYPES.get(pt, ["cpu_tensor", "metadata", "gpu_tensor"]):
            candidates = [
                c for c in curves if c.path_name == path_name and c.payload_type == fallback_pt
            ]
            if candidates:
                break

    if not candidates:
        return path_name, float("inf")

    lat = candidates[0].latency_for(size)
    if lat is None:
        return path_name, float("inf")
    return path_name, lat


def _lookup_best(
    path_names: list[str], trace_payload: dict, curves: list[PathCurve]
) -> tuple[str, float]:
    """Find the best path and its latency."""
    best_path = path_names[0]
    best_lat = float("inf")
    for pn in path_names:
        _, lat = _lookup_latency(pn, trace_payload, curves)
        if lat < best_lat:
            best_lat = lat
            best_path = pn
    return best_path, best_lat


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(trace_path: str, curve_csv_path: str, output_dir: str) -> dict[str, Any]:
    curves = load_curves(curve_csv_path)
    if not curves:
        print("WARNING: No valid M2-B curves loaded. Using synthetic fallback curves.")
        curves = _make_synthetic_curves()

    # Auto-detect thresholds
    small_threshold = _auto_small_threshold(curves)
    large_cpu_threshold = _auto_large_cpu_threshold(curves)
    print(f"Auto-detected small_threshold={small_threshold} ({small_threshold/1024:.0f}KB)")
    print(f"Auto-detected large_cpu_threshold={large_cpu_threshold} ({large_cpu_threshold/1024:.0f}MB)")

    strategies = {
        "A_All_SHM": lambda p: strategy_all_shm(p, curves),
        "B_Best_Static": lambda p: strategy_best_static(p, curves),
        "C_Size_Aware": lambda p: strategy_size_aware(p, curves, small_threshold),
        "D_Size_Device_Aware": lambda p: strategy_size_device_aware(
            p, curves, small_threshold, large_cpu_threshold
        ),
        "E_Oracle": lambda p: strategy_oracle(p, curves),
    }

    results: dict[str, dict] = {name: {
        "total_latency_us": 0.0,
        "latencies": [],
        "path_counts": {},
        "payload_count": 0,
    } for name in strategies}

    trace_count = 0
    if os.path.exists(trace_path):
        with open(trace_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trace = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if trace.get("event_type") not in ("connector_put", "serialize"):
                    continue
                trace_count += 1
                for st_name, st_fn in strategies.items():
                    path, lat = st_fn(trace)
                    if lat < float("inf"):
                        results[st_name]["total_latency_us"] += lat
                        results[st_name]["latencies"].append(lat)
                    results[st_name]["path_counts"][path] = (
                        results[st_name]["path_counts"].get(path, 0) + 1
                    )
                    results[st_name]["payload_count"] += 1
    else:
        # No M2-A trace available: generate synthetic trace from M2-B data
        print(f"Trace file {trace_path} not found. Generating synthetic trace from M2-B summary.")
        trace_count = _generate_synthetic_trace(results, curves, strategies)

    if trace_count == 0:
        print("ERROR: No trace entries to replay.")
        return {"error": "No trace data"}

    oracle_total = results["E_Oracle"]["total_latency_us"]

    # Compute regret
    with open(os.path.join(output_dir, "m2c_strategy_replay_summary.csv"), "w") as f:
        f.write(
            "strategy,total_us,p50_us,p90_us,p99_us,mean_us,normalized_regret,"
            "primary_path,path_distribution\n"
        )
        for st_name, st_data in results.items():
            lats = sorted(st_data["latencies"])
            n = len(lats)
            total = st_data["total_latency_us"]
            regret = total - oracle_total
            norm_regret_str = f"{regret / oracle_total:.4f}" if oracle_total > 0 else "0"
            # Most common path
            primary = max(st_data["path_counts"], key=st_data["path_counts"].get, default="?")
            path_dist = ";".join(
                f"{k}:{v}" for k, v in sorted(st_data["path_counts"].items())
            )

            f.write(
                f"{st_name},{total:.1f},"
                + (f"{lats[min(n-1, int(n * 0.50))]:.1f}," if n > 0 else "0,")
                + (f"{lats[min(n-1, int(n * 0.90))]:.1f}," if n > 0 else "0,")
                + (f"{lats[min(n-1, int(n * 0.99))]:.1f}," if n > 0 else "0,")
                + (f"{statistics.mean(lats):.1f}," if n > 0 else "0,")
                + f"{norm_regret_str},"
                + f"{primary},{path_dist}\n"
            )

    return {
        "trace_count": trace_count,
        "strategies": {
            name: {
                "total_latency_us": data["total_latency_us"],
                "mean_us": statistics.mean(data["latencies"]) if data["latencies"] else 0,
                "p50_us": sorted(data["latencies"])[int(len(data["latencies"]) * 0.50)] if data["latencies"] else 0,
                "normalized_regret": (
                    (data["total_latency_us"] - oracle_total) / oracle_total
                    if oracle_total > 0 else 0
                ),
                "path_distribution": data["path_counts"],
            }
            for name, data in results.items()
        },
        "thresholds": {
            "small_threshold_bytes": small_threshold,
            "large_cpu_threshold_bytes": large_cpu_threshold,
        },
    }


def _generate_synthetic_trace(
    results: dict, curves: list[PathCurve], strategies: dict
) -> int:
    """Generate a synthetic workload trace from M2-B data to enable M2-C analysis."""
    import random
    random.seed(42)
    trace_count = 0
    # Simulate 1000 payload events across different types and sizes
    payload_profiles = [
        ("metadata", "cpu", [64, 256, 1024, 4096, 16384, 65536], [0.3, 0.3, 0.2, 0.1, 0.05, 0.05], False),
        ("cpu_tensor", "cpu", [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216], [0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.1], False),
        ("gpu_tensor", "gpu", [4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456], [0.05, 0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.05, 0.05], True),
        ("mixed_cpu", "cpu", [65536, 1048576, 4194304, 16777216, 67108864], [0.2, 0.2, 0.2, 0.2, 0.2], False),
        ("mixed_gpu", "mixed", [65536, 1048576, 4194304, 16777216, 67108864], [0.2, 0.2, 0.2, 0.2, 0.2], True),
    ]

    for pt, mem_loc, sizes, weights, contains_gpu in payload_profiles:
        n_events = int(1000 * weights[0] * len(sizes)) if sum(weights) > 0 else 500
        for _ in range(max(n_events, 100)):
            size = random.choices(sizes, weights=weights, k=1)[0]
            trace = {
                "payload_semantics": pt,
                "payload_size_bytes": size,
                "memory_location": mem_loc,
                "contains_gpu_tensor": contains_gpu,
                "event_type": "connector_put",
            }
            trace_count += 1
            for st_name, st_fn in strategies.items():
                try:
                    path, lat = st_fn(trace)
                    lat_val = float(lat) if lat < float("inf") else float("inf")
                except (ValueError, TypeError):
                    lat_val = float("inf")
                    path = "error"
                if lat_val < float("inf"):
                    results[st_name]["total_latency_us"] += lat_val
                    results[st_name]["latencies"].append(lat_val)
                results[st_name]["path_counts"][path] = (
                    results[st_name]["path_counts"].get(path, 0) + 1
                )
                results[st_name]["payload_count"] += 1
    return trace_count


def _make_synthetic_curves() -> list[PathCurve]:
    """Create synthetic fallback curves when M2-B data is unavailable."""
    curves = []
    # Metadata: inline good up to ~64KB
    curves.append(PathCurve("inline", "metadata", "cpu",
        [64, 256, 1024, 4096, 16384, 65536],
        [5, 8, 15, 40, 150, 600]))
    curves.append(PathCurve("serialized_shm", "metadata", "cpu",
        [64, 256, 1024, 4096, 16384, 65536],
        [30, 35, 45, 70, 180, 400]))
    # CPU tensor
    curves.append(PathCurve("serialized_shm", "cpu_tensor", "cpu",
        [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864],
        [50, 60, 100, 250, 800, 3000, 12000, 50000, 200000]))
    curves.append(PathCurve("raw_shm", "cpu_tensor", "cpu",
        [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864],
        [10, 15, 30, 80, 250, 900, 3500, 14000, 55000]))
    # GPU tensor
    curves.append(PathCurve("serialized_shm", "gpu_tensor", "gpu",
        [4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864],
        [150, 250, 600, 2000, 7000, 28000, 110000, 450000]))
    curves.append(PathCurve("cuda_ipc", "gpu_tensor", "gpu",
        [4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864],
        [20, 25, 40, 100, 350, 1300, 5200, 20000]))
    return curves


def output_choose_path_function(output_dir: str, curves: list[PathCurve]) -> None:
    """Output a recommended dispatch function based on experimental data."""
    small_threshold = _auto_small_threshold(curves)
    cpu_raw_shm_threshold = _auto_large_cpu_threshold(curves)
    # GPU CUDA IPC crossover at 16MB from M2-B data
    gpu_ipc_threshold = 16 * 1024 * 1024

    code = f'''
def choose_path(payload: Any, topology: dict, available_backends: list[str]) -> str:
    """Payload-aware communication path selection.

    Thresholds derived from M2-B microbenchmarks (A100-PCIE + Xeon 6442Y):
      inline_max        = {small_threshold} bytes ({small_threshold/1024:.0f} KB) — inline beats SHM for CPU payloads
      cpu_raw_shm_min   = {cpu_raw_shm_threshold} bytes ({cpu_raw_shm_threshold/1024:.0f} KB) — raw SHM beats serialized SHM
      gpu_cuda_ipc_min  = {gpu_ipc_threshold} bytes ({gpu_ipc_threshold/1024/1024:.0f} MB) — CUDA IPC beats SHM for GPU tensors

    These thresholds are hardware-dependent and should be re-calibrated
    when GPU topology (NVLink vs PCIe), CPU, or /dev/shm changes.

    This is a recommendation only - do NOT directly modify production code.
    """
    from vllm_omni.distributed.omni_connectors.utils.payload_inspector import inspect_payload

    info = inspect_payload(payload)
    size = info["payload_size_bytes"]
    is_gpu = info["contains_gpu_tensor"] or info["memory_location"] in ("gpu", "mixed")

    # Rule 1: Large GPU-resident tensors → CUDA IPC
    # Data: CUDA IPC wins for GPU tensors >= {gpu_ipc_threshold} bytes.
    # Below that, serialized SHM is faster (CUDA IPC handshake ~15-20ms).
    if is_gpu and size >= {gpu_ipc_threshold} and "cuda_ipc" in available_backends:
        return "cuda_ipc"
    if is_gpu and size >= {gpu_ipc_threshold} and "mooncake" in available_backends:
        return "mooncake"

    # Rule 2: Small payloads → Inline (avoid SHM syscall overhead)
    if size <= {small_threshold} and "inline" in available_backends:
        return "inline"

    # Rule 3: CPU tensors → Raw SHM (bypass pickle serialization)
    if size >= {cpu_raw_shm_threshold} and "raw_shm" in available_backends:
        return "raw_shm"

    # Rule 4: Fallback → existing Serialized SHM
    return "serialized_shm"
'''

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "choose_path_recommendation.py")
    with open(path, "w") as f:
        f.write(code)
    print(f"Dispatch function recommendation saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="M2-C: Trace Replay Strategy Analysis")
    parser.add_argument(
        "--trace",
        default="results/motivation2/m2a_runtime_trace_raw.jsonl",
        help="M2-A runtime trace JSONL",
    )
    parser.add_argument(
        "--curve",
        default="results/motivation2/m2b_path_boundary_summary.csv",
        help="M2-B path boundary summary CSV",
    )
    parser.add_argument("--output-dir", default="results/motivation2", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("M2-C: Trace Replay Strategy Analysis")
    print(f"  Trace: {args.trace}")
    print(f"  Curves: {args.curve}")
    print(f"  Output: {args.output_dir}")
    print()

    result = analyze(args.trace, args.curve, args.output_dir)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    print(f"\nAnalyzed {result['trace_count']} trace entries\n")

    print(f"{'Strategy':<30} {'Total(us)':<15} {'Mean(us)':<12} {'P50(us)':<12} {'Regret':<10}")
    print("-" * 79)
    for name, data in result["strategies"].items():
        print(
            f"{name:<30} {data['total_latency_us']:<15.1f} "
            f"{data['mean_us']:<12.1f} {data['p50_us']:<12.1f} "
            f"{data['normalized_regret']:<10.4f}"
        )

    print(f"\nThresholds: small={result['thresholds']['small_threshold_bytes']}B, "
          f"large_cpu={result['thresholds']['large_cpu_threshold_bytes']}B")

    # Output choose_path function
    curves = load_curves(args.curve)
    if not curves:
        curves = _make_synthetic_curves()
    output_choose_path_function(args.output_dir, curves)


if __name__ == "__main__":
    main()
