# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Strategy 1: Zero-copy GPU tensor transport via CUDA IPC.

Producer exports IPC handle via torch.multiprocessing.reductions.
Consumer reconstructs tensor that directly views producer's allocation.
Producer retains the tensor until consumer sends release ACK.
"""
from __future__ import annotations

import time
import uuid

import torch

from .protocol import GPUTensorTransport, TensorMetadata, TransportHandle
from .config import GPUTransportConfig
from .tensor_registry import TensorRegistry
from .ipc_utils import extract_ipc_args, rebuild_from_ipc_args
from .logging import get_logger

logger = get_logger(__name__)


class CudaIpcTransport:
    """Zero-copy CUDA IPC transport."""

    def __init__(self, config: GPUTransportConfig):
        self._config = config
        self._registry = TensorRegistry(timeout_ms=config.release_timeout_ms)
        self._ipc_args_store: dict[str, tuple] = {}
        if config.enable_peer_access:
            self._ensure_peer_access(config.src_device, config.dst_device)

    @staticmethod
    def _ensure_peer_access(src: int, dst: int) -> None:
        for i in (src, dst):
            for j in (src, dst):
                if i != j and not torch.cuda.can_device_access_peer(i, j):
                    try:
                        torch.cuda.device(i).enable_peer_access(j)
                    except Exception as e:
                        logger.warning(
                            "Failed to enable P2P access device %d -> %d: %s", i, j, e)

    def send(
        self,
        tensor: torch.Tensor,
        *,
        dst_rank: int,
        tensor_id: str | None = None,
    ) -> TransportHandle:
        if not tensor.is_cuda:
            raise ValueError(f"Only CUDA tensors supported, got device={tensor.device}")
        if not tensor.is_contiguous():
            logger.warning("send: non-contiguous tensor %s, calling .contiguous()", tensor_id)
            tensor = tensor.contiguous()

        tid = tensor_id or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        # Ensure all prior work is done before exporting
        torch.cuda.current_stream(tensor.device).synchronize()
        t1 = time.perf_counter()

        ipc_args = extract_ipc_args(tensor)
        t2 = time.perf_counter()

        metadata = TensorMetadata.from_tensor(
            tensor,
            tensor_id=tid,
            mode="cuda_ipc",
            src_device=str(tensor.device),
            dst_device=f"cuda:{self._config.dst_device}",
        )
        metadata.send_start_ts = t0
        metadata.ipc_meta_ready_ts = t2

        metadata.ipc_args = ipc_args

        self._registry.register(tid, tensor)
        self._ipc_args_store[tid] = ipc_args

        logger.debug(
            "send: id=%s shape=%s dtype=%s nbytes=%d sync_ms=%.3f ipc_ms=%.3f",
            tid, metadata.shape, metadata.dtype, metadata.nbytes,
            (t1 - t0) * 1000, (t2 - t1) * 1000,
        )
        return TransportHandle(tensor_id=tid, metadata=metadata)

    def recv(
        self,
        handle: TransportHandle,
        *,
        src_rank: int,
        dst_device: torch.device | str,
    ) -> torch.Tensor:
        meta = handle.metadata
        if meta.mode != "cuda_ipc":
            raise ValueError(f"Expected mode=cuda_ipc, got {meta.mode}")

        ipc_args = meta.ipc_args
        if ipc_args is None:
            raise ValueError("TransportHandle has no IPC args — was metadata set by caller?")

        t0 = time.perf_counter()
        # Set the device context so rebuild happens on the right GPU
        dst_dev = torch.device(dst_device) if isinstance(dst_device, str) else dst_device
        with torch.cuda.device(dst_dev):
            tensor = rebuild_from_ipc_args(ipc_args)
        t1 = time.perf_counter()

        logger.debug(
            "recv: id=%s shape=%s dtype=%s rebuild_ms=%.3f",
            meta.tensor_id, meta.shape, meta.dtype, (t1 - t0) * 1000,
        )
        return tensor

    def release(self, tensor_id: str) -> None:
        self._ipc_args_store.pop(tensor_id, None)
        self._registry.release(tensor_id)

    def close(self) -> None:
        self._ipc_args_store.clear()
        self._registry.clear()

    def cleanup_timeouts(self) -> list[str]:
        return self._registry.cleanup_timeouts()

    @property
    def registry_size(self) -> int:
        return self._registry.size
