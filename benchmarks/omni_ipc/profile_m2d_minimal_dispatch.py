# SPDX-License-Identifier: Apache-2.0
"""M2-D: Experimental Payload-Aware Dispatcher (Minimal Prototype).

This is a lightweight demonstration dispatcher that uses thresholds from
M2-B experiments to route payloads to different communication paths.

Usage:
  VLLM_OMNI_EXPERIMENTAL_PAYLOAD_AWARE_DISPATCH=1 python -m vllm_omni.entrypoints.omni ...

Status: DEFERRED — Full integration requires modifying production connector paths.
This file provides the design blueprint and standalone validator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# Thresholds derived from M2-B experiments on A100 + Xeon Gold 6442Y
# These should be re-calibrated when hardware changes.
THRESHOLDS = {
    "small_payload": 64 * 1024,        # 64KB — below this, inline/pickle is faster than SHM
    "large_cpu_tensor": 1 * 1024 * 1024,  # 1MB — above this, raw SHM beats serialized SHM
}


@dataclass
class PayloadInfo:
    payload_type_summary: str = "unknown"
    payload_size_bytes: int = 0
    tensor_count: int = 0
    cpu_tensor_bytes: int = 0
    gpu_tensor_bytes: int = 0
    contains_gpu_tensor: bool = False
    memory_location: str = "cpu"
    max_tensor_size_bytes: int = 0


@dataclass
class TopologyInfo:
    same_node: bool = True
    nvlink_available: bool = False
    pcie_available: bool = True
    rdma_available: bool = False


class ExperimentalPayloadAwareDispatcher:
    """Experimental dispatcher for payload-aware communication path selection.

    This is a prototype ONLY. Do NOT use in production without proper testing
    and integration with the connector factory.
    """

    def __init__(self, topology: TopologyInfo | None = None):
        self.topology = topology or TopologyInfo()
        self._decision_log: list[dict] = []

    @staticmethod
    def inspect(payload: Any) -> PayloadInfo:
        """Inspect a payload without triggering GPU→CPU copies."""
        info = {
            "payload_type_summary": "unknown",
            "payload_size_bytes": 0,
            "tensor_count": 0,
            "cpu_tensor_bytes": 0,
            "gpu_tensor_bytes": 0,
            "contains_gpu_tensor": False,
            "memory_location": "cpu",
        }
        if isinstance(payload, torch.Tensor):
            nbytes = payload.numel() * payload.element_size()
            info["tensor_count"] = 1
            info["payload_size_bytes"] = nbytes
            info["max_tensor_size_bytes"] = nbytes
            if payload.is_cuda:
                info["gpu_tensor_bytes"] = nbytes
                info["contains_gpu_tensor"] = True
                info["memory_location"] = "gpu"
                info["payload_type_summary"] = "tensor"
            else:
                info["cpu_tensor_bytes"] = nbytes
                info["payload_type_summary"] = "tensor"
        elif isinstance(payload, dict):
            # Recurse to find tensors
            info["payload_type_summary"] = "mixed"
            for v in payload.values():
                if isinstance(v, torch.Tensor):
                    nbytes = v.numel() * v.element_size()
                    info["tensor_count"] += 1
                    info["payload_size_bytes"] += nbytes
                    if v.is_cuda:
                        info["gpu_tensor_bytes"] += nbytes
                        info["contains_gpu_tensor"] = True
                        info["memory_location"] = "mixed"
                    else:
                        info["cpu_tensor_bytes"] += nbytes
        elif isinstance(payload, bytes):
            info["payload_size_bytes"] = len(payload)
            info["payload_type_summary"] = "bytes"
        else:
            import pickle
            try:
                info["payload_size_bytes"] = len(pickle.dumps(payload))
            except Exception:
                pass
            info["payload_type_summary"] = "metadata"
        return PayloadInfo(**info)

    def choose_path(
        self,
        payload_info: PayloadInfo,
        available_backends: list[str],
    ) -> str:
        """Select the best communication path for a given payload.

        Priority rules (from M2-B data):
        1. GPU-resident tensor → CUDA IPC (same node) or Mooncake (cross-node)
        2. Small CPU payload → Inline (avoid SHM overhead)
        3. Large CPU tensor → Raw SHM (bypass serialization)
        4. Fallback → Serialized SHM
        """
        size = payload_info.payload_size_bytes
        is_gpu = payload_info.contains_gpu_tensor or payload_info.memory_location in ("gpu", "mixed")

        # Rule 1: GPU tensors
        if is_gpu:
            if self.topology.same_node and "cuda_ipc" in available_backends:
                return "cuda_ipc"
            if self.topology.rdma_available and "mooncake" in available_backends:
                return "mooncake"
            return "serialized_shm"

        # Rule 2: Small CPU payload
        if size <= THRESHOLDS["small_payload"] and "inline" in available_backends:
            return "inline"

        # Rule 3: Large CPU tensor
        if size >= THRESHOLDS["large_cpu_tensor"] and "raw_shm" in available_backends:
            return "raw_shm"

        # Rule 4: Fallback
        return "serialized_shm"

    def log_decision(self, **kwargs) -> None:
        self._decision_log.append(kwargs)

    def get_stats(self) -> dict[str, int]:
        from collections import Counter

        return dict(Counter(d["chosen_path"] for d in self._decision_log))


# ---------------------------------------------------------------------------
# Standalone validation
# ---------------------------------------------------------------------------

def validate_dispatcher() -> None:
    """Validate the dispatcher with representative payloads."""
    print("M2-D: Payload-Aware Dispatcher Validation")
    print("=" * 60)

    dispatcher = ExperimentalPayloadAwareDispatcher()
    available = ["inline", "serialized_shm", "raw_shm", "cuda_ipc"]

    # Test payloads
    test_cases = [
        ("small_metadata", {"request_id": "test", "chunk_id": 1, "finished": False}, "inline"),
        ("cpu_tensor_1kb", torch.randn(256, device="cpu"), "inline"),
        ("cpu_tensor_64kb", torch.randn(16384, device="cpu"), "inline"),
        ("cpu_tensor_1mb", torch.randn(262144, device="cpu"), "raw_shm"),
        ("cpu_tensor_64mb", torch.randn(16777216, device="cpu"), "raw_shm"),
        ("gpu_tensor_4kb", torch.randn(1024, device="cuda"), "cuda_ipc"),
        ("gpu_tensor_64mb", torch.randn(16777216, device="cuda"), "cuda_ipc"),
        ("mixed_gpu", {"tensor": torch.randn(1024, device="cuda"), "meta": "data"}, "cuda_ipc"),
    ]

    for name, payload, expected in test_cases:
        try:
            info = dispatcher.inspect(payload)
            chosen = dispatcher.choose_path(info, available)
            match = "✓" if chosen == expected else f"✗ (expected {expected})"
            print(
                f"  {name:25s} | size={info.payload_size_bytes:>10d}B | "
                f"loc={info.memory_location:5s} | chosen={chosen:15s} {match}"
            )
        except Exception as e:
            print(f"  {name:25s} | ERROR: {e}")

    print()
    print("Recommendation: This dispatcher blueprint should be integrated")
    print("into OmniConnectorFactory as a protocol selection layer between")
    print("the application and the connector backends.")
    print()
    print("Status: DEFERRED — production integration requires:")
    print("  1. Integration with OmniConnectorModelRunnerMixin._send_single_request")
    print("  2. Multi-backend connector pool management")
    print("  3. Dynamic path fallback on connector failure")


if __name__ == "__main__":
    validate_dispatcher()
