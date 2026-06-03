# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""UniIPC connector: SHM/inline for metadata + CUDA IPC for GPU tensors.

Composes ``SharedMemoryConnector`` for standard metadata serialization
and uses the ``gpu_transport`` package to keep GPU tensors on-device
across process boundaries.
"""
from __future__ import annotations

import os
import time
from typing import Any

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase
from .shm_connector import SharedMemoryConnector

logger = get_connector_logger(__name__)


class UniIPCConnector(OmniConnectorBase):
    """GPU tensor-aware connector using composition with SharedMemoryConnector.

    Configuration keys (in ``extra`` dict of ConnectorSpec):

    - ``stage_id`` (int): stage identifier (passed to SHM connector)
    - ``device`` (str): default device, e.g. ``"cuda:0"``
    - ``shm_threshold_bytes`` (int): inline-vs-SHM threshold (default 65536)
    - ``gpu_transport_mode`` (str): ``"cuda_ipc"`` or ``"cuda_copy"``
    - ``src_device`` (int): producer GPU index
    - ``dst_device`` (int): consumer GPU index
    - ``release_timeout_ms`` (float): registry timeout (default 10000)
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = config.get("stage_id", -1)

        # SHM / inline delegate
        self._shm = SharedMemoryConnector(config)

        # GPU transport config
        self._transport_mode: str = config.get("gpu_transport_mode", "cuda_ipc")
        self._src_device: int = int(config.get("src_device", 0))
        self._dst_device: int = int(config.get("dst_device", 1))
        self._release_timeout_ms: float = float(
            config.get("release_timeout_ms", 10_000.0))
        self._transport: Any = None  # lazy-init

        self._metrics: dict[str, int] = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "gpu_tensors_sent": 0,
            "gpu_tensors_recv": 0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_transport(self):
        """Lazy-init the GPU transport.

        Uses importlib to load gpu_transport submodules without triggering
        the full ``vllm_omni.__init__`` import chain.
        """
        if self._transport is not None:
            return

        import importlib.util
        import sys

        gpu_base = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "gpu_transport",
        )

        def _load(name, fname):
            full = f"vllm_omni.distributed.gpu_transport.{name}"
            if full in sys.modules:
                return sys.modules[full]
            path = os.path.join(gpu_base, fname)
            spec = importlib.util.spec_from_file_location(full, path)
            mod = importlib.util.module_from_spec(spec)
            mod.__package__ = "vllm_omni.distributed.gpu_transport"
            sys.modules[full] = mod
            spec.loader.exec_module(mod)
            return mod

        _load("config", "config.py")
        # Load dependencies needed by create_transport imports
        _load("logging", "logging.py")
        _load("protocol", "protocol.py")
        _load("tensor_registry", "tensor_registry.py")
        _load("ipc_utils", "ipc_utils.py")
        _load("cuda_ipc_transport", "cuda_ipc_transport.py")
        _load("cuda_copy_transport", "cuda_copy_transport.py")

        from vllm_omni.distributed.gpu_transport.config import GPUTransportConfig
        from vllm_omni.distributed.gpu_transport import create_transport

        transport_config = GPUTransportConfig(
            mode=self._transport_mode,
            src_device=self._src_device,
            dst_device=self._dst_device,
            release_timeout_ms=self._release_timeout_ms,
        )
        self._transport = create_transport(transport_config)

    @staticmethod
    def _has_gpu(obj: Any) -> bool:
        """Return True if *obj* contains any CUDA tensors (inline, no imports)."""
        import torch
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            return True
        if isinstance(obj, dict):
            return any(UniIPCConnector._has_gpu(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(UniIPCConnector._has_gpu(v) for v in obj)
        return False

    def _split(self, obj: Any) -> Any:
        """Replace GPU tensors with ``__gpux__`` markers using the transport."""
        if not self._has_gpu(obj):
            return obj
        self._init_transport()
        from vllm_omni.distributed.gpu_transport.split import (
            split_gpu_tensors)
        obj = split_gpu_tensors(obj, self._transport)
        self._metrics["gpu_tensors_sent"] += 1
        return obj

    def _reassemble(self, obj: Any) -> Any:
        """Replace ``__gpux__`` markers with real GPU tensors."""
        if not isinstance(obj, dict):
            return obj
        # Check for __gpux__ markers without importing from split
        if not self._has_markers(obj):
            return obj
        self._init_transport()
        from vllm_omni.distributed.gpu_transport.split import (
            reassemble_gpu_tensors)
        obj = reassemble_gpu_tensors(obj, self._transport)
        self._metrics["gpu_tensors_recv"] += 1
        return obj

    @staticmethod
    def _has_markers(obj: Any) -> bool:
        """Return True if *obj* contains any ``__gpux__`` markers."""
        if isinstance(obj, dict):
            if obj.get("__gpux__"):
                return True
            return any(UniIPCConnector._has_markers(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(UniIPCConnector._has_markers(v) for v in obj)
        return False

    # ------------------------------------------------------------------
    # OmniConnectorBase interface
    # ------------------------------------------------------------------

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """Split GPU tensors, delegate metadata to SHM connector."""
        try:
            stripped = self._split(data)
            success, size, metadata = self._shm.put(
                from_stage, to_stage, put_key, stripped)
            if not success:
                return False, 0, None
            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size
            return True, size, metadata
        except Exception:
            logger.exception("UniIPC put failed for key=%s", put_key)
            return False, 0, None

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        """Retrieve via SHM connector, then reassemble GPU tensors."""
        try:
            result = self._shm.get(from_stage, to_stage, get_key, metadata)
            if result is None:
                return None
            obj, size = result
            obj = self._reassemble(obj)
            self._metrics["gets"] += 1
            return obj, size
        except Exception:
            logger.exception("UniIPC get failed for key=%s", get_key)
            return None

    def cleanup(self, request_id: str) -> None:
        """Clean SHM segments and release transport-held tensors."""
        self._shm.cleanup(request_id)

    def close(self) -> None:
        """Release SHM connector and GPU transport."""
        self._shm.close()
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "transport_mode": self._transport_mode,
            "src_device": self._src_device,
            "dst_device": self._dst_device,
            **self._metrics,
        }
