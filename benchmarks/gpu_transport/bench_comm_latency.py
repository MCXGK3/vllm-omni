#!/usr/bin/env python3
"""Pure communication latency benchmark.

Measures end-to-end connector put/get latency without model inference.
Tests SHM, CUDA IPC, and CUDA Copy across a configurable GPU route.

Usage:
    python benchmarks/gpu_transport/bench_comm_latency.py --mode shm --sizes 4 8 16 32
    python benchmarks/gpu_transport/bench_comm_latency.py --mode all --src-device 0 --dst-device 4
    python benchmarks/gpu_transport/bench_comm_latency.py --mode all --concurrency 1 2 4 8 16
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import sys
import time
import uuid
from multiprocessing.connection import Connection
from typing import Any

import torch

SEPARATOR = "=" * 70


def _make_connector(mode: str, is_producer: bool, src_device: int, dst_device: int):
    """Create connector for producer or consumer."""
    if mode == "shm":
        from vllm_omni.distributed.omni_connectors.connectors.shm_connector import (
            SharedMemoryConnector,
        )
        return SharedMemoryConnector({"stage_id": 0 if is_producer else 1, "shm_threshold_bytes": 0})
    else:
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector,
        )
        extra: dict[str, Any] = {
            "stage_id": 0 if is_producer else 1,
            "shm_threshold_bytes": 0,
            "gpu_transport_mode": mode,
            "src_device": src_device,
            "dst_device": dst_device,
            "release_timeout_ms": 60000,
            "gpu_transport_min_bytes": 0,
        }
        return UniIPCConnector(extra)


def consumer_proc(mode: str, sizes_mb: list[int], iterations: int,
                  src_device: int, dst_device: int,
                  ack_conn: Connection | None,
                  meta_queue: mp.Queue, done_queue: mp.Queue,
                  result_queue: mp.Queue):
    """Consumer process - runs in spawned subprocess."""
    try:
        print(f"[consumer] started, mode={mode}", file=sys.stderr, flush=True)
        torch.cuda.set_device(dst_device)
        connector = _make_connector(mode, is_producer=False,
                                    src_device=src_device,
                                    dst_device=dst_device)
        if ack_conn is not None and hasattr(connector, "set_ack_conns"):
            connector.set_ack_conns(None, ack_conn)
        print(f"[consumer] ready, waiting for data...", file=sys.stderr, flush=True)

        results: list[dict] = []
        expected = iterations * len(sizes_mb)

        for _ in range(expected):
            meta_item = meta_queue.get()
            if meta_item is None:
                break
            key, metadata, size_mb = meta_item

            torch.cuda.synchronize(dst_device)
            t_cpu = time.perf_counter()

            result = connector.get("0", "1", key, metadata=metadata)

            t_cpu_end = time.perf_counter()
            cpu_ms = (t_cpu_end - t_cpu) * 1000.0
            data, data_size = result if result else (None, 0)

            # Measure first full-tensor consumption. For CUDA IPC this reads
            # producer GPU memory through the IPC view; for CUDA Copy it reads
            # the local copied tensor; for SHM it reads the CPU tensor.
            access_ms = 0.0
            if data is not None and isinstance(data, dict) and "tensor" in data:
                t = data["tensor"]
                torch.cuda.synchronize(dst_device)
                ta = time.perf_counter()
                _ = t.sum().item()
                if t.is_cuda:
                    torch.cuda.synchronize(t.device)
                access_ms = (time.perf_counter() - ta) * 1000.0

            results.append({
                "key": key,
                "size_mb": size_mb,
                "get_cpu_ms": round(cpu_ms, 3),
                "access_ms": round(access_ms, 3),
                "data_size": data_size,
            })

            if mode != "shm" and hasattr(connector, "release_gpu_tensors"):
                connector.release_gpu_tensors(key)

            # Signal producer: done consuming
            done_queue.put({"key": key})

        result_queue.put(results)
        print(f"[consumer] done, {len(results)} results", file=sys.stderr, flush=True)
    except Exception as e:
        import traceback
        print(f"[consumer] ERROR: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        result_queue.put({"error": str(e)})


def worker_proc(mode, sizes_mb, iterations, src_device, dst_device, ack_conn,
                meta_queue, done_queue, result_q):
    """Wrapper target for spawn Process."""
    consumer_proc(mode, sizes_mb, iterations, src_device, dst_device, ack_conn,
                  meta_queue, done_queue, result_q)


def run_benchmark(mode: str, sizes_mb: list[int], iterations: int,
                  src_device: int, dst_device: int, concurrency: int):
    """Run producer-consumer benchmark for one mode."""
    print(f"\n{SEPARATOR}")
    print(f"Mode: {mode.upper()}")
    print(f"Route: cuda:{src_device} -> cuda:{dst_device}")
    print(f"Max in-flight requests: {concurrency}")
    print(f"Tensor sizes: {sizes_mb} MB")
    print(f"Total iterations: {iterations}")
    print(f"{SEPARATOR}")
    torch.cuda.set_device(src_device)
    if mode != "shm":
        try:
            can_p2p = torch.cuda.can_device_access_peer(src_device, dst_device)
        except Exception:
            can_p2p = False
        print(f"P2P access cuda:{src_device}->cuda:{dst_device}: {can_p2p}")

    # ACK channels for CUDA IPC/Copy
    if mode != "shm":
        from vllm_omni.distributed.gpu_transport.control_channel import ControlChannelPair
        chan = ControlChannelPair()
        producer_ack = chan.producer_conn
        consumer_ack = chan.consumer_conn
    else:
        producer_ack = None
        consumer_ack = None

    # Producer connector
    producer = _make_connector(mode, is_producer=True,
                               src_device=src_device,
                               dst_device=dst_device)
    if producer_ack is not None and hasattr(producer, "set_ack_conns"):
        producer.set_ack_conns(producer_ack, None)
    if hasattr(producer, "_init_transport"):
        producer._init_transport()

    # Spawn consumer
    ctx = mp.get_context("spawn")
    meta_queue: mp.Queue = ctx.Queue()
    done_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()

    p = ctx.Process(
        target=worker_proc,
        args=(mode, sizes_mb, iterations, src_device, dst_device,
              consumer_ack, meta_queue, done_queue, result_queue),
    )
    p.start()
    print(f"[producer] consumer spawned, pid={p.pid}", file=sys.stderr, flush=True)

    put_results: list[dict] = []
    outstanding: dict[str, dict[str, Any]] = {}
    total_expected = iterations * len(sizes_mb)
    submitted_count = 0
    done_count = 0

    def drain_one_done() -> None:
        nonlocal done_count
        done_item = done_queue.get()
        done_key = done_item["key"] if isinstance(done_item, dict) else done_item
        record = outstanding.pop(done_key)
        e2e_ms = (time.perf_counter() - record["t_e2e"]) * 1000.0
        done_count += 1

        put_results.append({
            "key": done_key,
            "size_mb": record["size_mb"],
            "put_cpu_ms": round(record["put_cpu_ms"], 3),
            "e2e_ms": round(e2e_ms, 3),
            "meta_size": record["meta_size"],
            "success": record["success"],
        })

        if done_count == 1 or done_count % len(sizes_mb) == 1:
            print(
                f"[producer] done={done_count}/{total_expected} "
                f"size={record['size_mb']}MB put={record['put_cpu_ms']:.1f}ms "
                f"e2e={e2e_ms:.1f}ms in_flight={len(outstanding)}",
                file=sys.stderr,
                flush=True,
            )

    for i in range(iterations):
        for size_mb in sizes_mb:
            # Create tensor on the producer GPU. This is outside the measured
            # communication window.
            n_elements = (size_mb * 1024 * 1024) // 2
            tensor = torch.randn(n_elements, dtype=torch.bfloat16,
                                 device=f"cuda:{src_device}")
            key = f"bench-{uuid.uuid4().hex[:8]}"
            data = {"tensor": tensor}

            torch.cuda.synchronize(src_device)
            t_e2e = time.perf_counter()
            t_cpu = time.perf_counter()

            success, size, metadata = producer.put("0", "1", key, data)

            torch.cuda.synchronize(src_device)
            t_cpu_end = time.perf_counter()
            cpu_ms = (t_cpu_end - t_cpu) * 1000.0

            # Send metadata to consumer
            meta_queue.put((key, metadata, size_mb))
            submitted_count += 1

            outstanding[key] = {
                "size_mb": size_mb,
                "put_cpu_ms": cpu_ms,
                "t_e2e": t_e2e,
                "meta_size": size,
                "success": success,
            }

            while len(outstanding) >= concurrency:
                drain_one_done()

    while outstanding:
        drain_one_done()

    if submitted_count != total_expected:
        print(
            f"[producer] WARNING: submitted {submitted_count}, expected {total_expected}",
            file=sys.stderr,
            flush=True,
        )

    # Clean shutdown
    meta_queue.put(None)
    p.join(timeout=10)
    if hasattr(producer, "close"):
        producer.close()
    if p.is_alive():
        p.terminate()
        p.join()

    # Collect results
    get_results = []
    try:
        r = result_queue.get(timeout=5)
        if isinstance(r, list):
            get_results = r
    except Exception:
        pass

    return put_results, get_results


def print_stats(name: str, values: list[float], unit: str = "ms"):
    """Print p50/p95/p99 statistics."""
    if not values:
        print(f"  {name}: NO DATA")
        return
    p50 = statistics.median(values)
    p95 = statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values)
    p99 = statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
    mean = statistics.mean(values)
    print(f"  {name:20s}  mean={mean:8.2f}{unit}  p50={p50:8.2f}{unit}  "
          f"p95={p95:8.2f}{unit}  p99={p99:8.2f}{unit}  n={len(values)}")


def main():
    parser = argparse.ArgumentParser(description="Pure communication latency benchmark")
    parser.add_argument("--mode", default="all",
                        choices=["shm", "cuda_ipc", "cuda_copy", "all"])
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--src-device", type=int, default=0,
                        help="Producer CUDA device index.")
    parser.add_argument("--dst-device", type=int, default=4,
                        help="Consumer CUDA device index.")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1],
                        help="Maximum in-flight requests to test, e.g. 1 2 4 8 16.")
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()

    if any(c <= 0 for c in args.concurrency):
        parser.error("--concurrency values must be positive integers")

    modes = ["shm", "cuda_ipc", "cuda_copy"] if args.mode == "all" else [args.mode]
    total_iter = args.warmup + args.iterations
    all_results: dict[tuple[str, int], tuple[list, list]] = {}

    for mode in modes:
        for concurrency in args.concurrency:
            put_results, get_results = run_benchmark(
                mode, args.sizes, total_iter, args.src_device,
                args.dst_device, concurrency)

            warmup_n = args.warmup * len(args.sizes)
            put_results = put_results[warmup_n:]
            get_results = get_results[warmup_n:]

            if not get_results:
                print(
                    f"\n!!! {mode} concurrency={concurrency}: "
                    "consumer returned NO results !!!",
                    file=sys.stderr,
                )
                continue

            all_results[(mode, concurrency)] = (put_results, get_results)

            get_by_key = {r["key"]: r for r in get_results}

            print(
                f"\n--- {mode.upper()} concurrency={concurrency} "
                f"(after {args.warmup} warmup) ---")
            for size_mb in args.sizes:
                size_put = [r for r in put_results if r["size_mb"] == size_mb]
                size_get = [
                    r for r in get_results if r["size_mb"] == size_mb]
                put_vals = [r["put_cpu_ms"] for r in size_put]
                e2e_vals = [r["e2e_ms"] for r in size_put]
                get_vals = [r["get_cpu_ms"] for r in size_get]
                access_vals = [r["access_ms"] for r in size_get]
                component_total_vals = []
                for pr in size_put:
                    gr = get_by_key.get(pr["key"])
                    if gr is not None:
                        component_total_vals.append(
                            pr["put_cpu_ms"] + gr["get_cpu_ms"] +
                            gr["access_ms"])

                print(f"\n  Size={size_mb}MB:")
                print_stats("e2e wall", e2e_vals)
                print_stats("put()", put_vals)
                print_stats("get()", get_vals)
                print_stats("full consume", access_vals)
                print_stats("component sum", component_total_vals)

    if args.csv and all_results:
        print(f"\n{SEPARATOR}")
        print("CSV: mode,concurrency,size_mb,e2e_ms,put_ms,get_ms,access_ms,component_sum_ms")
        for (mode, concurrency), (put_r, get_r) in all_results.items():
            get_by_key = {r["key"]: r for r in get_r}
            for pr in put_r:
                gr = get_by_key.get(pr["key"])
                if gr is not None and pr["size_mb"] == gr["size_mb"]:
                    component_sum = (
                        pr['put_cpu_ms'] + gr['get_cpu_ms'] +
                        gr.get('access_ms', 0))
                    print(f"{mode},{concurrency},{pr['size_mb']},{pr['e2e_ms']},"
                          f"{pr['put_cpu_ms']},{gr['get_cpu_ms']},"
                          f"{gr.get('access_ms', 0)},{component_sum}")


if __name__ == "__main__":
    main()
