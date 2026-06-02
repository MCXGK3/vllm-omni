# SPDX-License-Identifier: Apache-2.0
"""M2-B: Multi-Path Communication Boundary Microbenchmark.

Measures end-to-end latency for different payload types across different
communication paths to identify crossover points and best-path matrices.

Usage:
  python benchmarks/omni_ipc/profile_m2b_path_boundary.py \
    --output-dir results/motivation2 \
    --warmup 20 --iterations 200 \
    --paths inline serialized_shm raw_shm cuda_ipc \
    --gpu-id 0
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import statistics
import sys
import time
import uuid
from typing import Any

import torch

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from benchmarks.omni_ipc.common import compute_stats, format_bytes


# ---------------------------------------------------------------------------
# Payload constructors
# ---------------------------------------------------------------------------

def make_metadata_payload(size_bytes: int) -> dict:
    """Construct a metadata/control dict of approximately *size_bytes* bytes."""
    # Use a repeated string field to hit target size
    overhead = 256  # approximate overhead of dict + fixed fields
    payload_size = max(0, size_bytes - overhead)
    filler = "x" * payload_size if payload_size > 0 else ""
    return {
        "request_id": str(uuid.uuid4()),
        "chunk_id": 1,
        "finished": False,
        "metadata": {"key": filler},
    }


def make_cpu_tensor(size_bytes: int) -> torch.Tensor:
    """Create a CPU tensor of approximately *size_bytes*."""
    n = max(1, size_bytes // 4)  # float32
    return torch.randn(n, device="cpu")


def make_gpu_tensor(size_bytes: int) -> torch.Tensor:
    """Create a GPU tensor of approximately *size_bytes*."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    n = max(1, size_bytes // 4)
    return torch.randn(n, device="cuda")


def make_mixed_cpu_payload(tensor_bytes: int, meta_bytes: int = 1024) -> dict:
    """Dict with CPU tensor + metadata."""
    return {
        "request_id": str(uuid.uuid4()),
        "finished": False,
        "metadata": {"key": "x" * max(0, meta_bytes - 200)},
        "tensor": make_cpu_tensor(tensor_bytes),
    }


def make_mixed_gpu_payload(tensor_bytes: int, meta_bytes: int = 1024) -> dict:
    """Dict with GPU tensor + metadata."""
    return {
        "request_id": str(uuid.uuid4()),
        "finished": False,
        "metadata": {"key": "x" * max(0, meta_bytes - 200)},
        "tensor": make_gpu_tensor(tensor_bytes),
    }


# ---------------------------------------------------------------------------
# Path implementations (each runs in a subprocess pair: sender / receiver)
# ---------------------------------------------------------------------------

SENTINEL = b"__DONE__"


def _path_inline_sender(send_q: mp.Queue, payload: Any, iterations: int, warmup: int) -> None:
    total = warmup + iterations
    for i in range(total):
        t0 = time.perf_counter_ns()
        pickled = pickle.dumps(payload)
        send_q.put(pickled)
        t1 = time.perf_counter_ns()
    send_q.put(SENTINEL)


def _path_inline_receiver(recv_q: mp.Queue, result_q: mp.Queue, iterations: int, warmup: int) -> None:
    total = warmup + iterations
    for i in range(total):
        data = recv_q.get()
        if data == SENTINEL:
            break
        obj = pickle.loads(data) if isinstance(data, bytes) else data
    result_q.put({"status": "ok"})


def measure_inline(payload: Any, iterations: int, warmup: int) -> dict[str, Any]:
    """Measure inline/queue baseline latency."""
    ctx = mp.get_context("spawn")
    send_q = ctx.Queue()
    result_q = ctx.Queue()

    sender = ctx.Process(target=_path_inline_measure_sender, args=(send_q, payload, iterations, warmup))
    receiver = ctx.Process(target=_path_inline_measure_receiver, args=(send_q, result_q, iterations, warmup))

    receiver.start()
    sender.start()
    sender.join()
    receiver.join()

    result_q.get()  # status: ok
    latencies = []
    while not result_q.empty():
        latencies.append(result_q.get())
    # Re-measure inline more carefully
    return _measure_inline_single_process(payload, iterations, warmup)


def _path_inline_measure_sender(send_q: mp.Queue, payload: Any, iterations: int, warmup: int) -> None:
    total = warmup + iterations
    for i in range(total):
        t0 = time.perf_counter_ns()
        pickled = pickle.dumps(payload)
        send_q.put((pickled, t0))
    send_q.put(SENTINEL)


def _path_inline_measure_receiver(
    recv_q: mp.Queue, result_q: mp.Queue, iterations: int, warmup: int
) -> None:
    total = warmup + iterations
    for i in range(total):
        item = recv_q.get()
        if item == SENTINEL:
            break
        pickled, t_send = item
        t_recv = time.perf_counter_ns()
        obj = pickle.loads(pickled) if isinstance(pickled, bytes) else pickled
        if i >= warmup:
            result_q.put((t_recv - t_send) / 1000.0)  # us


def _measure_inline_single_process(payload: Any, iterations: int, warmup: int) -> dict[str, Any]:
    """Inline measurement: pickle/unpickle in-process as baseline."""
    latencies = []
    total = warmup + iterations
    for i in range(total):
        t0 = time.perf_counter_ns()
        pickled = pickle.dumps(payload)
        obj = pickle.loads(pickled)
        t1 = time.perf_counter_ns()
        if i >= warmup:
            latencies.append((t1 - t0) / 1000.0)

    stats = compute_stats(latencies)
    return {
        **stats,
        "total_bytes_sample": len(pickle.dumps(payload)),
    }


# ---------------------------------------------------------------------------
# Path 1: Serialized SHM (current default path)
# ---------------------------------------------------------------------------

def _write_shm_bytes(payload: bytes, name: str | None = None) -> dict[str, Any]:
    """Write bytes into POSIX Shared Memory."""
    from multiprocessing import shared_memory

    shm = shared_memory.SharedMemory(create=True, size=len(payload), name=name)
    nbytes = len(payload)
    # Copy directly without holding a memoryview reference
    shm.buf[:nbytes] = payload
    shm.close()
    return {"name": shm.name, "size": nbytes}


def _read_shm_bytes(meta: dict[str, Any]) -> bytes:
    """Read bytes from POSIX Shared Memory and unlink."""
    from multiprocessing import shared_memory

    shm = shared_memory.SharedMemory(name=meta["name"])
    size = meta["size"]
    # Copy data out before closing
    data = bytes(shm.buf[:size])
    shm.close()
    shm.unlink()
    return data


def _path_serialized_shm_sender(send_q: mp.Queue, payload: Any, iterations: int, warmup: int) -> None:
    total = warmup + iterations
    for i in range(total):
        t0 = time.perf_counter_ns()

        # Phase 1: Serialize (pickle for simplicity, no vllm dependency)
        serialized = pickle.dumps(payload)
        t1 = time.perf_counter_ns()

        # Phase 2: SHM write
        key = f"m2b_ser_shm_{os.getpid()}_{i}"
        meta = _write_shm_bytes(serialized, name=key)
        t2 = time.perf_counter_ns()

        send_q.put((meta, t0, t1, t2))
    send_q.put(SENTINEL)


def _path_serialized_shm_receiver(
    recv_q: mp.Queue, result_q: mp.Queue, iterations: int, warmup: int
) -> None:
    total = warmup + iterations
    for i in range(total):
        item = recv_q.get()
        if item == SENTINEL:
            break
        meta, t0, t1, t2 = item

        t3 = time.perf_counter_ns()
        data_bytes = _read_shm_bytes(meta)
        t4 = time.perf_counter_ns()

        obj = pickle.loads(data_bytes)
        t5 = time.perf_counter_ns()

        if i >= warmup:
            result_q.put(
                {
                    "end_to_end_us": (t5 - t0) / 1000.0,
                    "serialize_us": (t1 - t0) / 1000.0,
                    "shm_write_us": (t2 - t1) / 1000.0,
                    "shm_read_us": (t4 - t3) / 1000.0,
                    "deserialize_us": (t5 - t4) / 1000.0,
                    "total_bytes": len(data_bytes),
                }
            )


def measure_serialized_shm(payload: Any, iterations: int, warmup: int) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    send_q = ctx.Queue()
    result_q = ctx.Queue()

    sender = ctx.Process(
        target=_path_serialized_shm_sender, args=(send_q, payload, iterations, warmup)
    )
    receiver = ctx.Process(
        target=_path_serialized_shm_receiver, args=(send_q, result_q, iterations, warmup)
    )

    receiver.start()
    sender.start()
    sender.join(timeout=120)
    receiver.join(timeout=120)

    records = []
    while not result_q.empty():
        try:
            records.append(result_q.get_nowait())
        except Exception:
            break

    if not records:
        return {"skipped": True, "skip_reason": "No measurements collected"}

    e2e = [r["end_to_end_us"] for r in records]
    serialize_times = [r["serialize_us"] for r in records]
    deserialize_times = [r["deserialize_us"] for r in records]
    shm_write_times = [r["shm_write_us"] for r in records]
    shm_read_times = [r["shm_read_us"] for r in records]
    total_bytes = records[0]["total_bytes"] if records else 0

    return {
        "end_to_end": compute_stats(e2e),
        "serialize": compute_stats(serialize_times),
        "deserialize": compute_stats(deserialize_times),
        "shm_write": compute_stats(shm_write_times),
        "shm_read": compute_stats(shm_read_times),
        "total_bytes": total_bytes,
        "effective_bandwidth_gbps_mean": (total_bytes / 1e9) / (statistics.mean(e2e) / 1e6)
        if e2e
        else 0,
    }


# ---------------------------------------------------------------------------
# Path 2: Raw CPU SHM (no serialization)
# ---------------------------------------------------------------------------

def _path_raw_shm_sender(send_q: mp.Queue, tensor: torch.Tensor, iterations: int, warmup: int) -> None:
    from multiprocessing import shared_memory

    import numpy as np

    total = warmup + iterations
    for i in range(total):
        t0 = time.perf_counter_ns()

        # Ensure contiguous CPU tensor
        if tensor.is_cuda:
            t_cpu = tensor.cpu().contiguous()
        else:
            t_cpu = tensor.contiguous()
        arr = t_cpu.numpy() if t_cpu.dtype != torch.bfloat16 else t_cpu.float().numpy()
        nbytes = arr.nbytes

        shm = shared_memory.SharedMemory(create=True, size=nbytes)
        shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf[:nbytes])
        np.copyto(shm_arr, arr)
        t1 = time.perf_counter_ns()

        send_q.put(
            (shm.name, tensor.shape, str(tensor.dtype), nbytes, t0, t1)
        )
    send_q.put(SENTINEL)


def _path_raw_shm_receiver(
    recv_q: mp.Queue, result_q: mp.Queue, iterations: int, warmup: int
) -> None:
    from multiprocessing import shared_memory

    import numpy as np

    total = warmup + iterations
    for i in range(total):
        item = recv_q.get()
        if item == SENTINEL:
            break
        name, shape, dtype_str, nbytes, t0, t1 = item

        t2 = time.perf_counter_ns()
        shm = shared_memory.SharedMemory(name=name)
        try:
            torch_dtype = getattr(torch, dtype_str.replace("torch.", ""))
            arr = np.ndarray(shape, dtype=np.float32, buffer=shm.buf[:nbytes])
            tensor = torch.from_numpy(arr.copy())
            if tensor.dtype != torch_dtype:
                tensor = tensor.to(torch_dtype)
            t3 = time.perf_counter_ns()
        finally:
            shm.close()
            shm.unlink()

        if i >= warmup:
            result_q.put(
                {
                    "end_to_end_us": (t3 - t0) / 1000.0,
                    "shm_write_us": (t1 - t0) / 1000.0,
                    "shm_read_us": (t3 - t2) / 1000.0,
                }
            )


def measure_raw_shm(
    tensor: torch.Tensor, iterations: int, warmup: int
) -> dict[str, Any]:
    if tensor.is_cuda:
        return {"skipped": True, "skip_reason": "Raw SHM only for CPU tensors"}

    ctx = mp.get_context("spawn")
    send_q = ctx.Queue()
    result_q = ctx.Queue()

    sender = ctx.Process(
        target=_path_raw_shm_sender, args=(send_q, tensor, iterations, warmup)
    )
    receiver = ctx.Process(
        target=_path_raw_shm_receiver, args=(send_q, result_q, iterations, warmup)
    )

    receiver.start()
    sender.start()
    sender.join(timeout=120)
    receiver.join(timeout=120)

    records = []
    while not result_q.empty():
        try:
            records.append(result_q.get_nowait())
        except Exception:
            break

    if not records:
        return {"skipped": True, "skip_reason": "No measurements collected"}

    e2e = [r["end_to_end_us"] for r in records]
    shm_write = [r["shm_write_us"] for r in records]
    shm_read = [r["shm_read_us"] for r in records]

    return {
        "end_to_end": compute_stats(e2e),
        "shm_write": compute_stats(shm_write),
        "shm_read": compute_stats(shm_read),
    }


# ---------------------------------------------------------------------------
# Path 3A: CUDA IPC via torch.multiprocessing tensor sharing
# ---------------------------------------------------------------------------

def _path_cuda_ipc_sender(send_q: mp.Queue, done_event, tensor: torch.Tensor, iterations: int, warmup: int) -> None:
    # Clone to avoid "tensor received from another process" error
    tensor = tensor.clone()
    total = warmup + iterations
    for i in range(total):
        t0 = time.perf_counter_ns()
        tensor_cpu_shape = tensor.shape
        tensor_cpu_dtype = str(tensor.dtype)
        tensor_cpu_device = str(tensor.device)
        t1 = time.perf_counter_ns()

        send_q.put((tensor, t0, t1, tensor_cpu_shape, tensor_cpu_dtype, tensor_cpu_device))
    send_q.put(SENTINEL)
    # Wait for receiver to finish before exiting (CUDA IPC requires sender to stay alive)
    done_event.wait(timeout=30)


def _path_cuda_ipc_receiver(
    recv_q: mp.Queue, result_q: mp.Queue, done_event, gpu_id: int, iterations: int, warmup: int
) -> None:
    if gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    total = warmup + iterations
    for i in range(total):
        item = recv_q.get()
        if item == SENTINEL:
            break
        tensor_sent, t0, t1, shape, dtype_str, device_str = item

        t2 = time.perf_counter_ns()
        # For a baseline measurement, do a D2D copy
        result = tensor_sent.clone()
        torch.cuda.synchronize()
        t3 = time.perf_counter_ns()

        if i >= warmup:
            result_q.put(
                {
                    "end_to_end_us": (t3 - t0) / 1000.0,
                    "metadata_us": (t1 - t0) / 1000.0,
                    "d2d_copy_us": (t3 - t2) / 1000.0,
                    "tensor_bytes": tensor_sent.numel() * tensor_sent.element_size(),
                }
            )
    done_event.set()


def measure_cuda_ipc(
    tensor: torch.Tensor, gpu_id: int, iterations: int, warmup: int
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"skipped": True, "skip_reason": "CUDA not available"}
    if not tensor.is_cuda:
        return {"skipped": True, "skip_reason": "CUDA IPC requires GPU tensor"}

    # Use torch.multiprocessing which handles CUDA IPC tensor sharing
    mp_ctx = torch.multiprocessing.get_context("spawn")
    send_q = mp_ctx.Queue()
    result_q = mp_ctx.Queue()
    done_event = mp_ctx.Event()

    sender = mp_ctx.Process(
        target=_path_cuda_ipc_sender, args=(send_q, done_event, tensor, iterations, warmup)
    )
    receiver = mp_ctx.Process(
        target=_path_cuda_ipc_receiver,
        args=(send_q, result_q, done_event, gpu_id, iterations, warmup),
    )

    receiver.start()
    sender.start()
    sender.join(timeout=120)
    receiver.join(timeout=120)

    records = []
    while not result_q.empty():
        try:
            records.append(result_q.get_nowait())
        except Exception:
            break

    if not records:
        return {"skipped": True, "skip_reason": "No measurements collected"}

    e2e = [r["end_to_end_us"] for r in records]
    metadata = [r["metadata_us"] for r in records]
    d2d_copy = [r["d2d_copy_us"] for r in records]
    tensor_bytes = records[0]["tensor_bytes"] if records else 0

    return {
        "end_to_end": compute_stats(e2e),
        "metadata": compute_stats(metadata),
        "d2d_copy": compute_stats(d2d_copy),
        "tensor_bytes": tensor_bytes,
        "effective_bandwidth_gbps_mean": (
            (tensor_bytes / 1e9) / (statistics.mean(e2e) / 1e6) if e2e else 0
        ),
        "note": "torch_mp_cuda_sharing_baseline (not production CUDA IPC connector)",
    }


# ---------------------------------------------------------------------------
# GPU tensor via SHM (includes D2H + H2D)
# ---------------------------------------------------------------------------

def measure_gpu_shm_breakdown(
    tensor: torch.Tensor, iterations: int, warmup: int
) -> dict[str, Any]:
    """Measure GPU tensor path through SHM with breakdown of D2H/H2D costs."""
    if not tensor.is_cuda:
        return {"skipped": True, "skip_reason": "GPU SHM breakdown requires GPU tensor"}

    # Pre-serialize to get total bytes (use pickle to avoid vllm dependency)
    t_cpu_test = tensor.cpu()
    torch.cuda.synchronize()
    serialized = pickle.dumps(t_cpu_test)
    total_bytes = len(serialized)
    del t_cpu_test, serialized

    d2h_times = []
    h2d_times = []
    serialize_times = []
    deserialize_times = []
    shm_times = []

    for i in range(warmup + iterations):
        # Simulate full path: D2H → serialize → SHM write → SHM read → deserialize → H2D
        t0 = time.perf_counter_ns()
        t_cpu = tensor.cpu()
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()

        s = pickle.dumps(t_cpu)
        t2 = time.perf_counter_ns()

        key = f"m2b_gpu_shm_{os.getpid()}_{i}"
        meta = _write_shm_bytes(s, name=key)
        t3 = time.perf_counter_ns()

        data = _read_shm_bytes(meta)
        t4 = time.perf_counter_ns()

        obj = pickle.loads(data)
        t5 = time.perf_counter_ns()

        if isinstance(obj, torch.Tensor):
            t_gpu = obj.cuda()
            torch.cuda.synchronize()
        t6 = time.perf_counter_ns()

        if i >= warmup:
            d2h_times.append((t1 - t0) / 1000.0)
            serialize_times.append((t2 - t1) / 1000.0)
            shm_times.append(((t3 - t2) + (t4 - t3)) / 1000.0)
            deserialize_times.append((t5 - t4) / 1000.0)
            h2d_times.append((t6 - t5) / 1000.0)

    e2e = [
        d2h_times[i] + serialize_times[i] + shm_times[i] + deserialize_times[i] + h2d_times[i]
        for i in range(len(d2h_times))
    ]

    return {
        "end_to_end": compute_stats(e2e),
        "d2h": compute_stats(d2h_times),
        "serialize": compute_stats(serialize_times),
        "shm_rw": compute_stats(shm_times),
        "deserialize": compute_stats(deserialize_times),
        "h2d": compute_stats(h2d_times),
        "total_bytes": total_bytes,
        "d2h_pct": statistics.mean(d2h_times) / statistics.mean(e2e) * 100 if e2e else 0,
        "h2d_pct": statistics.mean(h2d_times) / statistics.mean(e2e) * 100 if e2e else 0,
        "effective_bandwidth_gbps_mean": (total_bytes / 1e9) / (statistics.mean(e2e) / 1e6)
        if e2e
        else 0,
    }


# ---------------------------------------------------------------------------
# Path runner
# ---------------------------------------------------------------------------

SIZE_MAP = {
    "metadata": [64, 256, 1024, 4096, 16384, 65536],
    "cpu_tensor": [1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216,
                   67108864, 268435456],
    "gpu_tensor": [4096, 16384, 65536, 262144, 1048576, 4194304, 16777216,
                   67108864, 268435456],
    "mixed_cpu": [65536, 1048576, 4194304, 16777216, 67108864],
    "mixed_gpu": [65536, 1048576, 4194304, 16777216, 67108864],
}


def run_path(
    path_name: str,
    payload_type: str,
    size_bytes: int,
    iterations: int,
    warmup: int,
    gpu_id: int,
) -> dict[str, Any]:
    """Run a single configuration and return results."""
    base_record = {
        "path_name": path_name,
        "payload_type": payload_type,
        "payload_size_bytes": size_bytes,
        "warmup": warmup,
        "iterations": iterations,
        "gpu_id": gpu_id,
    }

    try:
        if payload_type == "metadata":
            payload = make_metadata_payload(size_bytes)
            actual_size = len(pickle.dumps(payload))
        elif payload_type == "cpu_tensor":
            payload = make_cpu_tensor(size_bytes)
            actual_size = payload.numel() * payload.element_size()
        elif payload_type == "gpu_tensor":
            if not torch.cuda.is_available():
                return {**base_record, "skipped": True, "skip_reason": "CUDA not available"}
            payload = make_gpu_tensor(size_bytes)
            actual_size = payload.numel() * payload.element_size()
        elif payload_type == "mixed_cpu":
            payload = make_mixed_cpu_payload(size_bytes)
            actual_size = size_bytes + 1024
        elif payload_type == "mixed_gpu":
            if not torch.cuda.is_available():
                return {**base_record, "skipped": True, "skip_reason": "CUDA not available"}
            payload = make_mixed_gpu_payload(size_bytes)
            actual_size = size_bytes + 1024
        else:
            return {**base_record, "skipped": True, "skip_reason": f"Unknown payload type: {payload_type}"}

        base_record["payload_size_bytes"] = actual_size
        base_record["memory_location"] = (
            "gpu" if "gpu" in payload_type else "cpu"
        )

        # Route to correct path
        if path_name == "inline":
            if payload_type not in ("metadata", "mixed_cpu"):
                return {**base_record, "skipped": True, "skip_reason": "Inline only for CPU/metadata payloads"}
            result = measure_inline(payload, iterations, warmup)

        elif path_name == "serialized_shm":
            if "gpu" in payload_type and isinstance(payload, torch.Tensor) and payload.is_cuda:
                result = measure_gpu_shm_breakdown(payload, iterations, warmup)
            else:
                result = measure_serialized_shm(payload, iterations, warmup)

        elif path_name == "raw_shm":
            if payload_type == "cpu_tensor":
                result = measure_raw_shm(payload, iterations, warmup)
            elif payload_type == "mixed_cpu":
                tensor_part = payload["tensor"]
                result = measure_raw_shm(tensor_part, iterations, warmup)
            else:
                return {**base_record, "skipped": True, "skip_reason": "Raw SHM only for CPU tensors"}

        elif path_name == "cuda_ipc":
            if payload_type == "gpu_tensor":
                result = measure_cuda_ipc(payload, gpu_id, iterations, warmup)
            elif payload_type == "mixed_gpu":
                result = measure_cuda_ipc(payload["tensor"], gpu_id, iterations, warmup)
            else:
                return {**base_record, "skipped": True, "skip_reason": "CUDA IPC only for GPU tensors"}

        elif path_name == "mooncake":
            return {**base_record, "skipped": True, "skip_reason": "Mooncake not available in environment"}

        elif path_name == "ucx":
            return {**base_record, "skipped": True, "skip_reason": "UCX not available in environment"}

        else:
            return {**base_record, "skipped": True, "skip_reason": f"Unknown path: {path_name}"}

        return {**base_record, **result, "success": True, "skipped": False}

    except Exception as e:
        import traceback

        return {
            **base_record,
            "success": False,
            "skipped": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description="M2-B: Multi-Path Boundary Microbenchmark")
    parser.add_argument("--output-dir", default="results/motivation2", help="Output directory")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=200, help="Measurement iterations")
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["inline", "serialized_shm", "raw_shm", "cuda_ipc"],
        choices=["inline", "serialized_shm", "raw_shm", "cuda_ipc", "mooncake", "ucx"],
        help="Paths to benchmark",
    )
    parser.add_argument(
        "--payload-types",
        nargs="+",
        default=["metadata", "cpu_tensor", "gpu_tensor", "mixed_cpu", "mixed_gpu"],
        help="Payload types to benchmark",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=None,
        help="Override size list (bytes)",
    )
    args = parser.parse_args()

    if args.gpu_id >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    raw_output = os.path.join(output_dir, "m2b_path_boundary_raw.jsonl")
    summary_output = os.path.join(output_dir, "m2b_path_boundary_summary.csv")

    print(f"M2-B: Multi-Path Boundary Microbenchmark")
    print(f"  Output: {output_dir}")
    print(f"  Paths: {args.paths}")
    print(f"  Payload types: {args.payload_types}")
    print(f"  Warmup: {args.warmup}, Iterations: {args.iterations}")
    print(f"  GPU: {args.gpu_id}")
    print()

    total_configs = sum(
        len(args.sizes or SIZE_MAP.get(pt, []))
        for pt in args.payload_types
    ) * len(args.paths)
    config_idx = 0

    with open(raw_output, "w") as f_raw, open(summary_output, "w") as f_summary:
        f_summary.write(
            "path_name,payload_type,payload_size_bytes,memory_location,"
            "mean_us,p50_us,p90_us,p99_us,std_us,min_us,max_us,"
            "serialize_us,deserialize_us,d2h_us,h2d_us,shm_us,metadata_us,"
            "effective_bandwidth_gbps,success,skipped,skip_reason\n"
        )

        for pt in args.payload_types:
            sizes = args.sizes or SIZE_MAP.get(pt, [])
            for size_bytes in sizes:
                for path_name in args.paths:
                    config_idx += 1
                    print(
                        f"  [{config_idx}/{total_configs}] {path_name} | {pt} | "
                        f"{format_bytes(size_bytes)} ... ",
                        end="",
                        flush=True,
                    )

                    record = run_path(
                        path_name=path_name,
                        payload_type=pt,
                        size_bytes=size_bytes,
                        iterations=args.iterations,
                        warmup=args.warmup,
                        gpu_id=args.gpu_id,
                    )

                    f_raw.write(json.dumps(record) + "\n")
                    f_raw.flush()

                    if record.get("skipped"):
                        print(f"SKIPPED: {record.get('skip_reason', '?')}")
                        f_summary.write(
                            f"{path_name},{pt},{size_bytes},,"
                            f",,,,,,,,,,,,"
                            f"False,True,{record.get('skip_reason', '')}\n"
                        )
                    elif not record.get("success"):
                        print(f"FAILED: {record.get('error', '?')}")
                    else:
                        e2e = record.get("end_to_end", {})
                        print(
                            f"P50={e2e.get('p50_us', 0):.0f}us "
                            f"P90={e2e.get('p90_us', 0):.0f}us "
                            f"mean={e2e.get('mean_us', 0):.0f}us"
                        )
                        ser_stats = record.get("serialize", {})
                        deser_stats = record.get("deserialize", {})
                        shm_stats = record.get("shm_write", {}) if record.get("shm_write") else record.get("shm_rw", {})
                        d2h_stats = record.get("d2h", {})
                        h2d_stats = record.get("h2d", {})
                        meta_stats = record.get("metadata", {})
                        bw = record.get("effective_bandwidth_gbps_mean", 0)

                        f_summary.write(
                            f"{path_name},{pt},{record.get('payload_size_bytes', size_bytes)},"
                            f"{record.get('memory_location', '')},"
                            f"{e2e.get('mean_us', '')},{e2e.get('p50_us', '')},"
                            f"{e2e.get('p90_us', '')},{e2e.get('p99_us', '')},"
                            f"{e2e.get('std_us', '')},{e2e.get('min_us', '')},"
                            f"{e2e.get('max_us', '')},"
                            f"{ser_stats.get('mean_us', '')},"
                            f"{deser_stats.get('mean_us', '')},"
                            f"{d2h_stats.get('mean_us', '')},"
                            f"{h2d_stats.get('mean_us', '')},"
                            f"{shm_stats.get('mean_us', '')},"
                            f"{meta_stats.get('mean_us', '')},"
                            f"{bw},"
                            f"True,False,\n"
                        )
                    f_summary.flush()

    print(f"\nDone. Raw data: {raw_output}")
    print(f"Summary: {summary_output}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
