# SPDX-License-Identifier: Apache-2.0
"""Generate figures for Motivation 2 experiment report.

Uses matplotlib (no seaborn).

Usage:
  python benchmarks/omni_ipc/generate_motivation2_figures.py \
    --input-dir results/motivation2 \
    --output-dir results/motivation2/figures
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Consistent styling
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 8,
        "figure.figsize": (8, 5),
    }
)

COLORS = {
    "inline": "#2ecc71",
    "serialized_shm": "#e74c3c",
    "raw_shm": "#3498db",
    "cuda_ipc": "#9b59b6",
    "mooncake": "#e67e22",
    "ucx": "#1abc9c",
}

PATH_LABELS = {
    "inline": "Inline / Queue",
    "serialized_shm": "Serialized SHM",
    "raw_shm": "Raw CPU SHM",
    "cuda_ipc": "CUDA IPC",
    "mooncake": "Mooncake Fast Path",
    "ucx": "UCX / RDMA",
}


# ---------------------------------------------------------------------------
# M2-A figures
# ---------------------------------------------------------------------------

def plot_m2a_payload_size_cdf(input_dir: str, output_dir: str) -> str | None:
    """Plot CDF of payload sizes from M2-A trace."""
    trace_path = os.path.join(input_dir, "m2a_runtime_trace_raw.jsonl")
    if not os.path.exists(trace_path):
        return "skipped: no M2-A trace file"

    sizes = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sz = rec.get("payload_size_bytes", 0)
            if sz > 0:
                sizes.append(sz)

    if not sizes:
        return "skipped: no size data in trace"

    sizes_sorted = sorted(sizes)
    y = np.arange(1, len(sizes_sorted) + 1) / len(sizes_sorted)

    fig, ax = plt.subplots()
    ax.plot(sizes_sorted, y, linewidth=2)
    ax.set_xscale("log")
    ax.set_xlabel("Payload Size (bytes)")
    ax.set_ylabel("CDF")
    ax.set_title("M2-A: Payload Size Distribution (CDF)")
    ax.grid(True, alpha=0.3)
    ax.axvline(x=np.percentile(sizes_sorted, 50), color="green", linestyle="--", label="P50")
    ax.axvline(x=np.percentile(sizes_sorted, 90), color="orange", linestyle="--", label="P90")
    ax.axvline(x=np.percentile(sizes_sorted, 99), color="red", linestyle="--", label="P99")
    ax.legend()

    path = os.path.join(output_dir, "m2a_payload_size_cdf.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_m2a_payload_memory_location(input_dir: str, output_dir: str) -> str | None:
    """Plot pie chart of payload memory locations."""
    trace_path = os.path.join(input_dir, "m2a_runtime_trace_raw.jsonl")
    if not os.path.exists(trace_path):
        return "skipped: no M2-A trace file"

    locations = defaultdict(int)
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            loc = rec.get("memory_location", "unknown")
            locations[loc] += 1

    if not locations:
        return "skipped: no location data in trace"

    fig, ax = plt.subplots()
    labels = list(locations.keys())
    values = list(locations.values())
    colors_list = ["#3498db", "#e74c3c", "#f39c12", "#95a5a6"]
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%",
        colors=colors_list[:len(labels)], startangle=90
    )
    ax.set_title("M2-A: Payload Memory Location Distribution")

    path = os.path.join(output_dir, "m2a_payload_memory_location.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_m2a_payload_type_distribution(input_dir: str, output_dir: str) -> str | None:
    """Plot bar chart of payload semantic types."""
    trace_path = os.path.join(input_dir, "m2a_runtime_trace_raw.jsonl")
    if not os.path.exists(trace_path):
        return "skipped: no M2-A trace file"

    types = defaultdict(int)
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ptype = rec.get("payload_semantics", rec.get("payload_type_summary", "unknown"))
            types[ptype] += 1

    if not types:
        return "skipped: no type data in trace"

    fig, ax = plt.subplots()
    labels = list(types.keys())
    values = list(types.values())
    bars = ax.bar(range(len(labels)), values, color=plt.cm.Set3(np.linspace(0, 1, len(labels))))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("M2-A: Payload Type Distribution")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(v),
                ha="center", va="bottom", fontsize=8)

    path = os.path.join(output_dir, "m2a_payload_type_distribution.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_m2a_runtime_overhead_breakdown(input_dir: str, output_dir: str) -> str | None:
    """Plot stacked bar of serialization vs put/get overhead."""
    trace_path = os.path.join(input_dir, "m2a_runtime_trace_raw.jsonl")
    if not os.path.exists(trace_path):
        return "skipped: no M2-A trace file"

    overheads = {"serialize": 0.0, "deserialize": 0.0, "put": 0.0, "get": 0.0, "shm_rw": 0.0}
    counts = defaultdict(int)
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            evt = rec.get("event_type", "")
            if evt == "serialize":
                overheads["serialize"] += rec.get("serialize_time_us", 0)
                counts["serialize"] += 1
            elif evt == "deserialize":
                overheads["deserialize"] += rec.get("deserialize_time_us", 0)
                counts["deserialize"] += 1
            elif evt == "connector_put":
                overheads["put"] += rec.get("put_time_us", 0)
                counts["put"] += 1
            elif evt == "connector_get":
                overheads["get"] += rec.get("get_time_us", 0)
                counts["get"] += 1

    total = sum(overheads.values())
    if total == 0:
        return "skipped: no overhead data in trace"

    fig, ax = plt.subplots()
    labels = list(overheads.keys())
    values = [overheads[k] for k in labels]
    pcts = [v / total * 100 for v in values]

    colors_list = ["#e74c3c", "#c0392b", "#3498db", "#2980b9", "#95a5a6"]
    bars = ax.bar(labels, pcts, color=colors_list[:len(labels)])
    ax.set_ylabel("% of Total Communication Overhead")
    ax.set_title("M2-A: Runtime Overhead Breakdown")
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", fontsize=8)

    path = os.path.join(output_dir, "m2a_runtime_overhead_breakdown.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# M2-B figures
# ---------------------------------------------------------------------------

def _load_m2b_summary(input_dir: str) -> list[dict]:
    """Load M2-B summary CSV as list of dicts."""
    path = os.path.join(input_dir, "m2b_path_boundary_summary.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_m2b_metadata_latency(input_dir: str, output_dir: str) -> str | None:
    """Plot metadata payload latency vs size for different paths."""
    rows = _load_m2b_summary(input_dir)
    meta_rows = [r for r in rows if r.get("payload_type") == "metadata" and r.get("skipped") != "True"
                 and r.get("mean_us")]
    if not meta_rows:
        return "skipped: no metadata benchmarks in M2-B"

    fig, ax = plt.subplots()
    for path_name in sorted(set(r["path_name"] for r in meta_rows)):
        path_rows = sorted(meta_rows, key=lambda r: int(r["payload_size_bytes"]))
        path_rows = [r for r in path_rows if r["path_name"] == path_name]
        if not path_rows:
            continue
        sizes = [int(r["payload_size_bytes"]) for r in path_rows]
        latencies = [float(r["mean_us"]) for r in path_rows]
        label = PATH_LABELS.get(path_name, path_name)
        ax.plot(sizes, latencies, "o-", linewidth=2, markersize=4,
                label=label, color=COLORS.get(path_name, "#333333"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Payload Size (bytes)")
    ax.set_ylabel("Mean Latency (us)")
    ax.set_title("M2-B: Metadata / Control Message Latency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "m2b_metadata_latency.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_m2b_cpu_tensor_latency(input_dir: str, output_dir: str) -> str | None:
    """Plot CPU tensor latency vs size for different paths."""
    rows = _load_m2b_summary(input_dir)
    cpu_rows = [r for r in rows if r.get("payload_type") == "cpu_tensor" and r.get("skipped") != "True"
                and r.get("mean_us")]
    if not cpu_rows:
        return "skipped: no CPU tensor benchmarks in M2-B"

    fig, ax = plt.subplots()
    for path_name in sorted(set(r["path_name"] for r in cpu_rows)):
        path_rows = [r for r in cpu_rows if r["path_name"] == path_name]
        if not path_rows:
            continue
        path_rows.sort(key=lambda r: int(r["payload_size_bytes"]))
        sizes = [int(r["payload_size_bytes"]) for r in path_rows]
        latencies = [float(r["mean_us"]) for r in path_rows]
        label = PATH_LABELS.get(path_name, path_name)
        ax.plot(sizes, latencies, "o-", linewidth=2, markersize=4,
                label=label, color=COLORS.get(path_name, "#333333"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tensor Size (bytes)")
    ax.set_ylabel("Mean Latency (us)")
    ax.set_title("M2-B: CPU Tensor Communication Latency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "m2b_cpu_tensor_latency.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_m2b_gpu_tensor_latency(input_dir: str, output_dir: str) -> str | None:
    """Plot GPU tensor latency vs size for different paths."""
    rows = _load_m2b_summary(input_dir)
    gpu_rows = [r for r in rows if r.get("payload_type") == "gpu_tensor" and r.get("skipped") != "True"
                and r.get("mean_us")]
    if not gpu_rows:
        return "skipped: no GPU tensor benchmarks in M2-B"

    fig, ax = plt.subplots()
    for path_name in sorted(set(r["path_name"] for r in gpu_rows)):
        path_rows = [r for r in gpu_rows if r["path_name"] == path_name]
        if not path_rows:
            continue
        path_rows.sort(key=lambda r: int(r["payload_size_bytes"]))
        sizes = [int(r["payload_size_bytes"]) for r in path_rows]
        latencies = [float(r["mean_us"]) for r in path_rows]
        label = PATH_LABELS.get(path_name, path_name)
        ax.plot(sizes, latencies, "o-", linewidth=2, markersize=4,
                label=label, color=COLORS.get(path_name, "#333333"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tensor Size (bytes)")
    ax.set_ylabel("Mean Latency (us)")
    ax.set_title("M2-B: GPU Tensor Communication Latency")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "m2b_gpu_tensor_latency.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_m2b_gpu_tensor_bandwidth(input_dir: str, output_dir: str) -> str | None:
    """Plot GPU tensor effective bandwidth."""
    rows = _load_m2b_summary(input_dir)
    gpu_rows = [r for r in rows if r.get("payload_type") == "gpu_tensor" and r.get("skipped") != "True"
                and r.get("effective_bandwidth_gbps") and float(r.get("effective_bandwidth_gbps", 0)) > 0]
    if not gpu_rows:
        return "skipped: no GPU bandwidth data in M2-B"

    fig, ax = plt.subplots()
    for path_name in sorted(set(r["path_name"] for r in gpu_rows)):
        path_rows = [r for r in gpu_rows if r["path_name"] == path_name]
        path_rows.sort(key=lambda r: int(r["payload_size_bytes"]))
        sizes = [int(r["payload_size_bytes"]) for r in path_rows]
        bw = [float(r["effective_bandwidth_gbps"]) for r in path_rows]
        label = PATH_LABELS.get(path_name, path_name)
        ax.plot(sizes, bw, "s-", linewidth=2, markersize=4,
                label=label, color=COLORS.get(path_name, "#333333"))

    ax.set_xscale("log")
    ax.set_xlabel("Tensor Size (bytes)")
    ax.set_ylabel("Effective Bandwidth (GB/s)")
    ax.set_title("M2-B: GPU Tensor Transfer Bandwidth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "m2b_gpu_tensor_bandwidth.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_m2b_gpu_shm_breakdown(input_dir: str, output_dir: str) -> str | None:
    """Plot GPU tensor SHM path breakdown (D2H, serialization, SHM, deser, H2D)."""
    # This data is in the raw JSONL
    raw_path = os.path.join(input_dir, "m2b_path_boundary_raw.jsonl")
    if not os.path.exists(raw_path):
        return "skipped: no raw M2-B data"

    # Find gpu_shm breakdown records
    records = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("path_name") == "serialized_shm" and rec.get("payload_type") == "gpu_tensor":
                if rec.get("d2h") and not rec.get("skipped"):
                    records.append(rec)

    if not records:
        return "skipped: no GPU SHM breakdown data"

    # Aggregate by size
    by_size = defaultdict(list)
    for rec in records:
        sz = rec["payload_size_bytes"]
        by_size[sz].append(rec)

    sizes = sorted(by_size.keys())

    fig, ax = plt.subplots()
    x = range(len(sizes))
    width = 0.15

    d2h_means = []
    ser_means = []
    shm_means = []
    deser_means = []
    h2d_means = []

    for sz in sizes:
        recs = by_size[sz]
        d2h_means.append(np.mean([r["d2h"].get("mean_us", 0) if isinstance(r.get("d2h"), dict) else 0 for r in recs]))
        ser_means.append(np.mean([r.get("serialize", {}).get("mean_us", 0) if isinstance(r.get("serialize"), dict) else 0 for r in recs]))
        shm_means.append(np.mean([r.get("shm_rw", {}).get("mean_us", 0) if isinstance(r.get("shm_rw"), dict) else 0 for r in recs]))
        deser_means.append(np.mean([r.get("deserialize", {}).get("mean_us", 0) if isinstance(r.get("deserialize"), dict) else 0 for r in recs]))
        h2d_means.append(np.mean([r.get("h2d", {}).get("mean_us", 0) if isinstance(r.get("h2d"), dict) else 0 for r in recs]))

    ax.bar(x, d2h_means, width, label="D2H Copy", color="#e74c3c")
    ax.bar([i + width for i in x], ser_means, width, label="Serialize", color="#f39c12")
    ax.bar([i + 2*width for i in x], shm_means, width, label="SHM Read/Write", color="#3498db")
    ax.bar([i + 3*width for i in x], deser_means, width, label="Deserialize", color="#2ecc71")
    ax.bar([i + 4*width for i in x], h2d_means, width, label="H2D Copy", color="#9b59b6")

    ax.set_xticks([i + 2*width for i in x])
    ax.set_xticklabels([_format_size(s) for s in sizes], rotation=45, ha="right")
    ax.set_ylabel("Mean Latency (us)")
    ax.set_title("M2-B: GPU Tensor SHM Path Breakdown")
    ax.legend()

    path = os.path.join(output_dir, "m2b_gpu_shm_breakdown.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_m2b_cuda_ipc_breakdown(input_dir: str, output_dir: str) -> str | None:
    """Plot CUDA IPC breakdown (metadata vs D2D copy)."""
    raw_path = os.path.join(input_dir, "m2b_path_boundary_raw.jsonl")
    if not os.path.exists(raw_path):
        return "skipped: no raw M2-B data"

    records = []
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("path_name") == "cuda_ipc" and not rec.get("skipped"):
                records.append(rec)

    if not records:
        return "skipped: no CUDA IPC breakdown data"

    by_size = defaultdict(list)
    for rec in records:
        sz = rec["payload_size_bytes"]
        by_size[sz].append(rec)

    sizes = sorted(by_size.keys())

    fig, ax = plt.subplots()
    x = range(len(sizes))
    width = 0.3

    meta_means = [np.mean([r.get("metadata", {}).get("mean_us", 0) if isinstance(r.get("metadata"), dict) else 0
                          for r in by_size[sz]]) for sz in sizes]
    d2d_means = [np.mean([r.get("d2d_copy", {}).get("mean_us", 0) if isinstance(r.get("d2d_copy"), dict) else 0
                          for r in by_size[sz]]) for sz in sizes]

    ax.bar(x, meta_means, width, label="Metadata Transfer", color="#3498db")
    ax.bar([i + width for i in x], d2d_means, width, label="D2D Copy", color="#9b59b6")

    ax.set_xticks([i + width/2 for i in x])
    ax.set_xticklabels([_format_size(s) for s in sizes], rotation=45, ha="right")
    ax.set_ylabel("Mean Latency (us)")
    ax.set_title("M2-B: CUDA IPC Path Breakdown")
    ax.legend()

    path = os.path.join(output_dir, "m2b_cuda_ipc_breakdown.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_m2b_best_path_matrix(input_dir: str, output_dir: str) -> str | None:
    """Plot best path matrix as a colored table."""
    rows = _load_m2b_summary(input_dir)
    valid = [r for r in rows if r.get("skipped") != "True" and r.get("mean_us")]

    if not valid:
        return "skipped: no valid M2-B benchmarks"

    # Group by (payload_type, size) and find best path
    groups = defaultdict(list)
    for r in valid:
        key = (r["payload_type"], int(r["payload_size_bytes"]))
        groups[key].append((r["path_name"], float(r["mean_us"])))

    # Determine best path per group
    best_matrix = {}
    for (pt, sz), candidates in groups.items():
        best = min(candidates, key=lambda x: x[1])
        best_matrix[(pt, sz)] = best[0]

    # Create matrix
    payload_types = sorted(set(k[0] for k in best_matrix))
    sizes = sorted(set(k[1] for k in best_matrix))

    if len(payload_types) == 0 or len(sizes) == 0:
        return "skipped: insufficient data for matrix"

    # Build a color map for paths
    all_paths = sorted(set(best_matrix.values()))
    path_to_color = {
        p: plt.cm.tab10(i / max(len(all_paths), 1)) for i, p in enumerate(all_paths)
    }

    fig, ax = plt.subplots(figsize=(max(10, len(sizes) * 1.2), max(4, len(payload_types) * 0.8)))

    # Fill matrix
    matrix_data = np.zeros((len(payload_types), len(sizes)))
    matrix_text = np.empty((len(payload_types), len(sizes)), dtype=object)

    for i, pt in enumerate(payload_types):
        for j, sz in enumerate(sizes):
            best = best_matrix.get((pt, sz), "N/A")
            matrix_text[i, j] = best[:3] if best != "N/A" else "--"
            # Assign a numeric value for coloring
            if best in all_paths:
                matrix_data[i, j] = all_paths.index(best)
            else:
                matrix_data[i, j] = -1

    cmap = plt.cm.tab10
    im = ax.imshow(matrix_data, cmap=cmap, aspect="auto", vmin=0, vmax=max(len(all_paths)-1, 1))

    # Add text
    for i in range(len(payload_types)):
        for j in range(len(sizes)):
            ax.text(j, i, matrix_text[i, j], ha="center", va="center",
                    fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([_format_size(s) for s in sizes], rotation=45, ha="right")
    ax.set_yticks(range(len(payload_types)))
    ax.set_yticklabels(payload_types)
    ax.set_title("M2-B: Best Path Matrix")
    ax.set_xlabel("Payload Size")

    # Add color legend for paths
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=cmap(i / max(len(all_paths)-1, 1)), label=p)
        for i, p in enumerate(all_paths)
    ]
    ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(1.02, 1),
              borderaxespad=0)

    path = os.path.join(output_dir, "m2b_best_path_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# M2-C figures
# ---------------------------------------------------------------------------

def plot_m2c_strategy_total_time(input_dir: str, output_dir: str) -> str | None:
    """Bar chart of total communication time per M2-C strategy."""
    csv_path = os.path.join(input_dir, "m2c_strategy_replay_summary.csv")
    if not os.path.exists(csv_path):
        return "skipped: no M2-C summary"

    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return "skipped: empty M2-C summary"

    fig, ax = plt.subplots()
    names = [r["strategy"] for r in rows]
    totals = [float(r["total_us"]) / 1e6 for r in rows]  # Convert to seconds

    colors_list = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]
    bars = ax.bar(range(len(names)), totals, color=colors_list[:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Total Communication Time (s)")
    ax.set_title("M2-C: Strategy Total Communication Time")
    for bar, v in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}s", ha="center", fontsize=8)

    path = os.path.join(output_dir, "m2c_strategy_total_time.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_m2c_strategy_regret(input_dir: str, output_dir: str) -> str | None:
    """Bar chart of normalized regret per M2-C strategy."""
    csv_path = os.path.join(input_dir, "m2c_strategy_replay_summary.csv")
    if not os.path.exists(csv_path):
        return "skipped: no M2-C summary"

    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return "skipped: empty M2-C summary"

    fig, ax = plt.subplots()
    names = []
    regrets = []
    for r in rows:
        if r["strategy"] == "E_Oracle":
            continue
        nr_str = r.get("normalized_regret", "inf").strip()
        if nr_str in ("inf", "nan", ""):
            continue
        try:
            reg = float(nr_str) * 100
        except (ValueError, TypeError):
            continue
        names.append(r["strategy"])
        regrets.append(reg)

    if not names:
        return "skipped: only Oracle in data"

    colors_list = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71"]
    bars = ax.bar(range(len(names)), regrets, color=colors_list[:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Normalized Regret (%)")
    ax.set_title("M2-C: Selection Regret vs Oracle")
    for bar, v in zip(bars, regrets):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{v:.1f}%", ha="center", fontsize=8)

    path = os.path.join(output_dir, "m2c_strategy_regret.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_m2c_strategy_path_distribution(input_dir: str, output_dir: str) -> str | None:
    """Stacked bar of path distribution per strategy."""
    csv_path = os.path.join(input_dir, "m2c_strategy_replay_summary.csv")
    if not os.path.exists(csv_path):
        return "skipped: no M2-C summary"

    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return "skipped: empty M2-C summary"

    # Parse path distributions
    strat_paths = {}
    for r in rows:
        dist_str = r.get("path_distribution", "")
        dist = {}
        for pair in dist_str.split(";"):
            if ":" in pair:
                k, v = pair.split(":", 1)
                try:
                    dist[k] = int(v)
                except ValueError:
                    pass
        if dist:
            strat_paths[r["strategy"]] = dist

    if not strat_paths:
        return "skipped: no path distribution data"

    all_paths = sorted(set(p for d in strat_paths.values() for p in d))

    fig, ax = plt.subplots()
    strategies = list(strat_paths.keys())
    x = range(len(strategies))

    bottom = np.zeros(len(strategies))
    for pi, path in enumerate(all_paths):
        values = [strat_paths[st].get(path, 0) for st in strategies]
        ax.bar(x, values, bottom=bottom, label=PATH_LABELS.get(path, path),
               color=COLORS.get(path, "#95a5a6"))
        bottom += np.array(values)

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Payload Count")
    ax.set_title("M2-C: Strategy Path Distribution")
    ax.legend(fontsize=7)

    path = os.path.join(output_dir, "m2c_strategy_path_distribution.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.0f}KB"
    return f"{size_bytes/(1024*1024):.0f}MB"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIGURES = {
    "m2a_payload_size_cdf": plot_m2a_payload_size_cdf,
    "m2a_payload_memory_location": plot_m2a_payload_memory_location,
    "m2a_payload_type_distribution": plot_m2a_payload_type_distribution,
    "m2a_runtime_overhead_breakdown": plot_m2a_runtime_overhead_breakdown,
    "m2b_metadata_latency": plot_m2b_metadata_latency,
    "m2b_cpu_tensor_latency": plot_m2b_cpu_tensor_latency,
    "m2b_gpu_tensor_latency": plot_m2b_gpu_tensor_latency,
    "m2b_gpu_tensor_bandwidth": plot_m2b_gpu_tensor_bandwidth,
    "m2b_gpu_shm_breakdown": plot_m2b_gpu_shm_breakdown,
    "m2b_cuda_ipc_breakdown": plot_m2b_cuda_ipc_breakdown,
    "m2b_best_path_matrix": plot_m2b_best_path_matrix,
    "m2c_strategy_total_time": plot_m2c_strategy_total_time,
    "m2c_strategy_regret": plot_m2c_strategy_regret,
    "m2c_strategy_path_distribution": plot_m2c_strategy_path_distribution,
}


def main():
    parser = argparse.ArgumentParser(description="Generate Motivation 2 figures")
    parser.add_argument("--input-dir", default="results/motivation2", help="Results input directory")
    parser.add_argument("--output-dir", default="results/motivation2/figures", help="Figures output directory")
    parser.add_argument("--figure", nargs="*", choices=list(FIGURES.keys()),
                       help="Specific figures to generate (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    to_generate = args.figure if args.figure else list(FIGURES.keys())

    print(f"Generating {len(to_generate)} figures...")
    results = {}
    for name in to_generate:
        func = FIGURES[name]
        try:
            result = func(args.input_dir, args.output_dir)
            results[name] = result if result else "generated"
            print(f"  {name}: {results[name]}")
        except Exception as e:
            results[name] = f"ERROR: {e}"
            print(f"  {name}: ERROR: {e}")

    # Summary
    ok = sum(1 for v in results.values() if v and not str(v).startswith("skipped") and not str(v).startswith("ERROR"))
    skipped = sum(1 for v in results.values() if str(v).startswith("skipped"))
    errors = sum(1 for v in results.values() if str(v).startswith("ERROR"))
    print(f"\nDone: {ok} generated, {skipped} skipped, {errors} errors")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
