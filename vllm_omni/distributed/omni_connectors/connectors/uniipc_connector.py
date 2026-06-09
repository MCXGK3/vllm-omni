# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""UniIPC connector: SHM/inline for metadata + CUDA IPC for GPU tensors.

Composes ``SharedMemoryConnector`` for standard metadata serialization
and uses the ``gpu_transport`` package to keep GPU tensors on-device
across process boundaries.
"""
from __future__ import annotations

import os
import queue
import threading
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
        self._transport_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._deferred_ack_queue: queue.Queue[tuple[Any | None, list[str]] | None] = queue.Queue()
        self._deferred_ack_thread: threading.Thread | None = None
        self._deferred_ack_running = False

        # Size threshold: skip GPU transport for small tensors (0 = disabled)
        self._gpu_transport_min_bytes: int = int(
            config.get("gpu_transport_min_bytes", 65536))

        # Memory pressure threshold: fall back to inline when free ratio < threshold
        # 0.0 disables pressure checking
        self._gpu_memory_pressure_threshold: float = float(
            config.get("gpu_memory_pressure_threshold", 0.0))

        # ACK connections for GPU tensor release notifications
        self._ack_conn: Any = config.get("ack_conn", None)  # producer side: receive ACKs
        self._consumer_ack_conn: Any = config.get("consumer_ack_conn", None)  # consumer side: send ACKs

        self._metrics: dict[str, Any] = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "gpu_tensors_sent": 0,
            "gpu_tensors_recv": 0,
            "gpu_tensors_inlined": 0,
            "inline_bytes": 0,
            "pressure_fallbacks": 0,
            "put_total_ms": 0.0,
            "get_total_ms": 0.0,
        }

        # Track GPU transport tensor_ids by put_key for per-request ACK.
        # Key: put_key (e.g. "req-123_0_0"), Value: list of tensor_id strings.
        self._pending_gpu_tensors: dict[str, list[str]] = {}

        # Track GPU transport tensor_ids received via get() for per-request ACK.
        # Key: get_key, Value: list of tensor_id strings.
        self._received_gpu_tensors: dict[str, list[str]] = {}

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
        with self._transport_lock:
            if self._transport is not None:
                return

            # "none" mode: no transport needed, router always returns "inline"
            if self._transport_mode == "none":
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

            # Ensure the gpu_transport package itself is loaded (not a stub),
            # so create_transport is importable from it.
            pkg_name = "vllm_omni.distributed.gpu_transport"
            pkg_path = os.path.join(gpu_base, "__init__.py")
            if os.path.isfile(pkg_path):
                spec = importlib.util.spec_from_file_location(pkg_name, pkg_path)
                pkg_mod = importlib.util.module_from_spec(spec)
                pkg_mod.__package__ = pkg_name
                pkg_mod.__path__ = [gpu_base]
                sys.modules[pkg_name] = pkg_mod
                spec.loader.exec_module(pkg_mod)

            from vllm_omni.distributed.gpu_transport.config import GPUTransportConfig
            from vllm_omni.distributed.gpu_transport import create_transport

            transport_config = GPUTransportConfig(
                mode=self._transport_mode,
                src_device=self._src_device,
                dst_device=self._dst_device,
                release_timeout_ms=self._release_timeout_ms,
                ack_conn=self._ack_conn,
                consumer_ack_conn=self._consumer_ack_conn,
            )
            self._transport = create_transport(transport_config)

            # Start ACK thread if transport supports it and ack_conn is wired
            if self._transport is not None and self._ack_conn is not None:
                if hasattr(self._transport, '_start_ack_thread'):
                    self._transport._start_ack_thread()

    def _start_deferred_ack_thread(self) -> None:
        """Start the consumer-side CUDA-event ACK worker."""
        if self._deferred_ack_running:
            return
        self._deferred_ack_running = True
        self._deferred_ack_thread = threading.Thread(
            target=self._deferred_ack_loop,
            name=f"uniipc-ack-stage-{self.stage_id}",
            daemon=True,
        )
        self._deferred_ack_thread.start()

    def _deferred_ack_loop(self) -> None:
        while self._deferred_ack_running:
            item = self._deferred_ack_queue.get()
            if item is None:
                break
            event, tensor_ids = item
            try:
                if event is not None:
                    event.synchronize()
                self.notify_gpu_tensors_consumed(tensor_ids)
            except Exception:
                logger.warning(
                    "Deferred GPU tensor ACK failed for ids=%s",
                    tensor_ids,
                    exc_info=True,
                )
        self._deferred_ack_running = False

    def _shutdown_deferred_ack_thread(self) -> None:
        if self._deferred_ack_thread is None:
            return
        self._deferred_ack_queue.put(None)
        if self._deferred_ack_thread.is_alive():
            self._deferred_ack_thread.join(timeout=2.0)
        self._deferred_ack_thread = None
        self._deferred_ack_running = False

    def _enqueue_received_tensor_ack(
        self,
        tensor_ids: list[str],
        *,
        wait_cuda_event: bool,
    ) -> None:
        if not tensor_ids:
            return
        event = None
        if wait_cuda_event:
            try:
                import torch

                if torch.cuda.is_available():
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream())
            except Exception:
                logger.debug(
                    "Failed to record CUDA event for GPU tensor ACK; ACKing immediately",
                    exc_info=True,
                )
                event = None
        if event is None:
            self.notify_gpu_tensors_consumed(tensor_ids)
            return
        self._start_deferred_ack_thread()
        self._deferred_ack_queue.put((event, list(tensor_ids)))

    def _route_tensor(self, tensor: "torch.Tensor") -> str:
        """Decide transport strategy for a single GPU tensor.

        Returns ``"inline"`` to serialize to CPU bytes in the marker,
        or ``"ipc"`` to use the GPU transport (CUDA IPC / copy).

        Checks are evaluated in order: mode, size, memory pressure.
        """
        import torch

        nbytes = tensor.numel() * tensor.element_size()

        # Dimension 1: mode "none" always inlines
        if self._transport_mode == "none":
            with self._state_lock:
                self._metrics["gpu_tensors_inlined"] += 1
                self._metrics["inline_bytes"] += nbytes
            return "inline"

        # Dimension 2: size threshold
        if nbytes < self._gpu_transport_min_bytes:
            with self._state_lock:
                self._metrics["gpu_tensors_inlined"] += 1
                self._metrics["inline_bytes"] += nbytes
            return "inline"

        # Dimension 3: memory pressure (producer-side only for cuda_ipc)
        if (self._transport_mode == "cuda_ipc"
                and self._gpu_memory_pressure_threshold > 0.0):
            free, total = torch.cuda.mem_get_info(tensor.device)
            ratio = free / total
            if ratio < self._gpu_memory_pressure_threshold:
                with self._state_lock:
                    self._metrics["gpu_tensors_inlined"] += 1
                    self._metrics["inline_bytes"] += nbytes
                    self._metrics["pressure_fallbacks"] += 1
                return "inline"

        return "ipc"

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

    def _split(self, obj: Any) -> tuple[Any, list[str]]:
        """Replace GPU tensors with ``__gpux__`` markers using the transport.

        Returns ``(stripped_obj, tensor_ids)`` — *tensor_ids* are collected
        during the walk, avoiding a second traversal via ``_collect_tensor_ids``.
        """
        if not self._has_gpu(obj):
            return obj, []
        self._init_transport()
        from vllm_omni.distributed.gpu_transport.split import (
            split_gpu_tensors)
        stripped, tensor_ids = split_gpu_tensors(
            obj, self._transport,
            router=self._route_tensor,
            dst_device=f"cuda:{self._dst_device}",
            return_tensor_ids=True,
        )
        with self._state_lock:
            self._metrics["gpu_tensors_sent"] += 1
        return stripped, tensor_ids

    def _reassemble(self, obj: Any, skip_marker_check: bool = False) -> tuple[Any, list[str]]:
        """Replace ``__gpux__`` markers with real GPU tensors.

        Returns ``(restored_obj, tensor_ids)`` — *tensor_ids* are collected
        during the walk, avoiding a separate ``_collect_tensor_ids`` pass.

        When *skip_marker_check* is True, the caller guarantees that *obj*
        contains markers, avoiding a redundant full-payload walk.
        """
        if not skip_marker_check and not self._has_markers(obj):
            return obj, []
        self._init_transport()
        from vllm_omni.distributed.gpu_transport.split import (
            reassemble_gpu_tensors)
        obj, tensor_ids = reassemble_gpu_tensors(
            obj, self._transport, return_tensor_ids=True)
        with self._state_lock:
            self._metrics["gpu_tensors_recv"] += 1
        return obj, tensor_ids

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

    @staticmethod
    def _collect_tensor_ids(obj: Any) -> list[str]:
        """Walk *obj* and collect all tensor_ids from ``__gpux__`` markers."""
        ids: list[str] = []
        if isinstance(obj, dict):
            if obj.get("__gpux__"):
                tid = obj.get("tensor_id")
                if tid:
                    ids.append(tid)
            else:
                for v in obj.values():
                    ids.extend(UniIPCConnector._collect_tensor_ids(v))
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                ids.extend(UniIPCConnector._collect_tensor_ids(v))
        return ids

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
        t0 = time.perf_counter()
        tensor_ids = None
        timing = {}
        try:
            t_split_start = time.perf_counter()
            stripped, tensor_ids = self._split(data)
            timing["split_ms"] = (time.perf_counter() - t_split_start) * 1000.0
            if tensor_ids:
                with self._state_lock:
                    self._pending_gpu_tensors[put_key] = tensor_ids
            t_shm_start = time.perf_counter()
            success, size, metadata = self._shm.put(
                from_stage, to_stage, put_key, stripped)
            timing["shm_put_ms"] = (time.perf_counter() - t_shm_start) * 1000.0
            if not success:
                self._release_registered_tensors(put_key, tensor_ids)
                return False, 0, None
            timing["put_total_ms"] = (time.perf_counter() - t0) * 1000.0
            with self._state_lock:
                self._metrics["puts"] += 1
                self._metrics["bytes_transferred"] += size
                self._metrics["put_total_ms"] += timing["put_total_ms"]
            if tensor_ids:
                logger.debug(
                    "TIMING put: key=%s total=%.3fms split=%.3fms shm=%.3fms size=%d num_tids=%d",
                    put_key, timing["put_total_ms"], timing["split_ms"],
                    timing["shm_put_ms"], size, len(tensor_ids or []),
                )
            return True, size, metadata
        except Exception:
            logger.exception("UniIPC put failed for key=%s", put_key)
            self._release_registered_tensors(put_key, tensor_ids)
            return False, 0, None

    def _release_registered_tensors(
        self, put_key: str, tensor_ids: list[str] | None,
    ) -> None:
        """Release GPU transport registrations from a failed put."""
        if tensor_ids:
            with self._state_lock:
                self._pending_gpu_tensors.pop(put_key, None)
        transport = self._transport
        if tensor_ids and transport is not None:
            for tid in tensor_ids:
                try:
                    transport.release(tid)
                except Exception:
                    pass

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        """Retrieve via SHM connector, then reassemble GPU tensors.

        CUDA IPC tensors are ACKed during request cleanup, not here.
        The rebuilt tensor is a zero-copy view of producer memory, so
        producer-side lifetime must cover downstream GPU work.
        """
        t0 = time.perf_counter()
        timing = {}
        try:
            t_shm_start = time.perf_counter()
            result = self._shm.get(from_stage, to_stage, get_key, metadata)
            timing["shm_get_ms"] = (time.perf_counter() - t_shm_start) * 1000.0
            if result is None:
                return None
            obj, size = result
            # Reassembly collects tensor IDs during the walk.  The
            # _has_markers check here informs _reassemble so it can
            # skip a redundant full-payload walk.
            has_markers_flag = self._has_markers(obj)
            t_reassemble_start = time.perf_counter()
            obj, tensor_ids = self._reassemble(obj, skip_marker_check=has_markers_flag)
            timing["reassemble_ms"] = (time.perf_counter() - t_reassemble_start) * 1000.0
            if tensor_ids and self._transport_mode == "cuda_ipc":
                with self._state_lock:
                    self._received_gpu_tensors[get_key] = tensor_ids
                timing["ack_ms"] = 0.0
            else:
                timing["ack_ms"] = 0.0
            timing["get_total_ms"] = (time.perf_counter() - t0) * 1000.0
            with self._state_lock:
                self._metrics["gets"] += 1
                self._metrics["get_total_ms"] += timing["get_total_ms"]
            if has_markers_flag:
                logger.debug(
                    "TIMING get: key=%s total=%.3fms shm=%.3fms reassemble=%.3fms ack=%.3fms num_tids=%d",
                    get_key, timing["get_total_ms"], timing["shm_get_ms"],
                    timing["reassemble_ms"], timing["ack_ms"],
                    len(tensor_ids or []),
                )
            return obj, size
        except Exception:
            logger.exception("UniIPC get failed for key=%s", get_key)
            return None

    def release_gpu_tensors(self, request_id: str) -> None:
        """Release GPU transport tensors for *request_id*.

        Tensors produced by this connector are released from the local
        registry.  Tensors consumed by this connector are ACKed to their
        producer after the request is done using the zero-copy IPC view.
        """
        prefix = f"{request_id}_"
        with self._state_lock:
            put_keys = [
                k for k in self._pending_gpu_tensors
                if k == request_id or k.startswith(prefix)
            ]
            recv_keys = [
                k for k in self._received_gpu_tensors
                if k == request_id or k.startswith(prefix)
            ]
            pending_tids = [
                tid
                for key in put_keys
                for tid in self._pending_gpu_tensors.pop(key, [])
            ]
            received_tids = [
                tid
                for key in recv_keys
                for tid in self._received_gpu_tensors.pop(key, [])
            ]
        total_tids = len(pending_tids) + len(received_tids)
        if total_tids:
            logger.debug(
                "release_gpu_tensors: req=%s put_keys=%d recv_keys=%d total_tids=%d",
                request_id, len(put_keys), len(recv_keys), total_tids,
            )
        transport = self._transport
        if transport is not None:
            try:
                if hasattr(transport, "release_many"):
                    transport.release_many(pending_tids)
                else:
                    for tid in pending_tids:
                        transport.release(tid)
            except Exception:
                logger.debug(
                    "release_gpu_tensors: local release failed for %d ids",
                    len(pending_tids), exc_info=True)
        self.notify_gpu_tensors_consumed(received_tids)

    def release_received_gpu_tensors(
        self,
        get_key: str,
        *,
        wait_cuda_event: bool = True,
    ) -> None:
        """ACK received CUDA IPC tensors for one connector get key.

        This is the fast-path lifetime hook for zero-copy CUDA IPC.  Call it
        from the consumer after the payload identified by *get_key* has been
        submitted to the GPU.  When *wait_cuda_event* is True, ACK is deferred
        until work already enqueued on the current CUDA stream completes.
        """
        with self._state_lock:
            tensor_ids = self._received_gpu_tensors.pop(get_key, [])
        self._enqueue_received_tensor_ack(
            tensor_ids,
            wait_cuda_event=wait_cuda_event,
        )

    def release_pending_gpu_tensors(self, put_key: str) -> None:
        """Release locally produced GPU tensors for one connector put key."""
        with self._state_lock:
            tensor_ids = self._pending_gpu_tensors.pop(put_key, [])
        transport = self._transport
        if transport is None:
            return
        try:
            if hasattr(transport, "release_many"):
                transport.release_many(tensor_ids)
            else:
                for tid in tensor_ids:
                    transport.release(tid)
        except Exception:
            logger.debug(
                "release_pending_gpu_tensors: local release failed for %d ids",
                len(tensor_ids),
                exc_info=True,
            )

    def notify_gpu_tensor_consumed(self, tensor_id: str) -> None:
        """Notify producer that a GPU tensor (cuda_ipc mode) is no longer needed."""
        if self._transport is not None and hasattr(self._transport, 'notify_consumed'):
            self._transport.notify_consumed(tensor_id)

    def notify_gpu_tensors_consumed(self, tensor_ids: list[str]) -> None:
        """Notify producer that multiple GPU tensors are no longer needed."""
        if not tensor_ids or self._transport is None:
            return
        if hasattr(self._transport, "notify_consumed_many"):
            self._transport.notify_consumed_many(tensor_ids)
            return
        for tid in tensor_ids:
            self.notify_gpu_tensor_consumed(tid)

    def cleanup(self, request_id: str) -> None:
        """Clean SHM segments and release transport-held tensors."""
        self._shm.cleanup(request_id)
        self.release_gpu_tensors(request_id)

    def close(self) -> None:
        """Release SHM connector and GPU transport."""
        with self._state_lock:
            pending_tids = [
                tid
                for key in list(self._pending_gpu_tensors)
                for tid in self._pending_gpu_tensors.pop(key, [])
            ]
            received_tids = [
                tid
                for key in list(self._received_gpu_tensors)
                for tid in self._received_gpu_tensors.pop(key, [])
            ]
        transport = self._transport
        if transport is not None:
            try:
                if hasattr(transport, "release_many"):
                    transport.release_many(pending_tids)
                else:
                    for tid in pending_tids:
                        transport.release(tid)
            except Exception:
                pass
        self.notify_gpu_tensors_consumed(received_tids)
        self._shutdown_deferred_ack_thread()
        self._shm.close()
        with self._transport_lock:
            if self._transport is not None:
                if hasattr(self._transport, "close"):
                    self._transport.close()
                self._transport = None

    def set_ack_conns(self, ack_conn: Any, consumer_ack_conn: Any) -> None:
        """Wire ACK pipe connections after construction.

        Called after the connector is created, so that the
        ``multiprocessing.Connection`` objects never enter vLLM's
        config hash computation.
        """
        self._ack_conn = ack_conn
        self._consumer_ack_conn = consumer_ack_conn
        if self._transport is not None:
            self._transport._ack_conn = ack_conn
            self._transport._consumer_ack_conn = consumer_ack_conn
            self._transport._consumer_ctrl = None  # invalidate cached ctrl
            if ack_conn is not None and hasattr(self._transport, "_start_ack_thread"):
                self._transport._start_ack_thread()

    @property
    def dst_device(self) -> str:
        """Configured destination GPU device string (e.g. ``cuda:7``)."""
        return f"cuda:{self._dst_device}"

    def health(self) -> dict[str, Any]:
        with self._state_lock:
            metrics = dict(self._metrics)
        result = {
            "status": "healthy",
            "transport_mode": self._transport_mode,
            "src_device": self._src_device,
            "dst_device": self._dst_device,
            "gpu_transport_min_bytes": self._gpu_transport_min_bytes,
            "gpu_memory_pressure_threshold": self._gpu_memory_pressure_threshold,
            **metrics,
        }
        # Add real-time registry size if transport is active
        if self._transport is not None and hasattr(self._transport, 'registry_size'):
            result["current_registry_size"] = self._transport.registry_size
        else:
            result["current_registry_size"] = 0
        # Add current GPU free memory ratio
        try:
            import torch
            free, total = torch.cuda.mem_get_info(self._src_device)
            result["gpu_free_memory_ratio"] = round(free / total, 4)
        except Exception:
            result["gpu_free_memory_ratio"] = 0.0
        return result
