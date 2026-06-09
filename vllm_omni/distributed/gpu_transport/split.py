# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Payload split/reassemble for GPU tensor transport.

Walks a nested dict/list structure, replaces GPU tensors with lightweight
``__gpux__`` markers (safe for Pipe/msgpack serialization), and restores
them on the receiver side.
"""
from __future__ import annotations

import io
import pickle
import uuid
from typing import Any

import torch

from .protocol import TensorMetadata, TransportHandle
from .logging import get_logger

logger = get_logger(__name__)

_GPUX_MARKER = "__gpux__"


def has_gpu_tensors(obj: Any) -> bool:
    """Return True if *obj* contains any CUDA tensors."""
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
        return True
    if isinstance(obj, dict):
        return any(has_gpu_tensors(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(has_gpu_tensors(v) for v in obj)
    return False


def _make_inline_marker(tensor: torch.Tensor, dst_device: str) -> dict:
    """Serialize a GPU tensor to CPU bytes and wrap in a ``__gpux__`` marker."""
    cpu_tensor = tensor.detach().cpu().contiguous()
    buf = io.BytesIO()
    torch.save(cpu_tensor, buf)
    nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
    return {
        _GPUX_MARKER: True,
        "tensor_id": uuid.uuid4().hex[:12],
        "meta": {
            "shape": list(cpu_tensor.shape),
            "dtype": str(cpu_tensor.dtype),
            "nbytes": nbytes,
            "dst_device": dst_device,
        },
        "inline_data": buf.getvalue(),
    }


def _resolve_local_device(dst_device: str) -> torch.device:
    """Resolve *dst_device* to a valid local CUDA device.

    In multi-stage deployments the producer sets ``dst_device`` to the
    physical GPU index (e.g. ``cuda:7``), but vLLM may remap that GPU
    to a different index in the consumer process.  Falls back to the
    current CUDA device when the requested device is not available.
    """
    try:
        dev = torch.device(dst_device)
        # Trigger device context to validate the ordinal is reachable.
        with torch.cuda.device(dev):
            pass
        return dev
    except (RuntimeError, torch.AcceleratorError):
        return torch.device(torch.cuda.current_device())


def _recover_inline_tensor(marker: dict) -> torch.Tensor:
    """Recover a GPU tensor from an inline marker."""
    buf = io.BytesIO(marker["inline_data"])
    tensor = torch.load(buf, weights_only=True)
    dst_device = marker["meta"].get("dst_device", "cuda:0")
    return tensor.to(_resolve_local_device(dst_device))


def split_gpu_tensors(
    obj: Any,
    transport: Any,
    router: Any = None,
    dst_device: str | None = None,
    return_tensor_ids: bool = False,
) -> Any:
    """Walk *obj* and replace every CUDA tensor with a ``__gpux__`` marker.

    By default returns only the stripped object for compatibility with older
    callers.  If *return_tensor_ids* is True, returns
    ``(stripped_obj, tensor_ids)`` where *tensor_ids* is a flat list of IPC
    tensor IDs encountered during the walk.

    If *router* is provided, it is called for each tensor and must return
    ``"inline"`` or ``"ipc"``.  ``"inline"`` serializes the tensor to CPU
    bytes via ``_make_inline_marker``; ``"ipc"`` uses the GPU transport.

    If *router* is ``None``, all tensors go through the GPU transport
    (original behaviour).
    """
    stripped, tensor_ids = _split_gpu_tensors_impl(
        obj, transport, router=router, dst_device=dst_device)
    if return_tensor_ids:
        return stripped, tensor_ids
    return stripped


def _split_gpu_tensors_impl(
    obj: Any,
    transport: Any,
    router: Any = None,
    dst_device: str | None = None,
) -> tuple[Any, list[str]]:
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
        if router is not None:
            decision = router(obj)
            if decision == "inline":
                marker = _make_inline_marker(
                    obj, dst_device=dst_device or str(obj.device)
                )
                return marker, []
        # Original IPC path
        handle = transport.send(obj, dst_rank=transport._config.dst_device)
        marker = {
            _GPUX_MARKER: True,
            "tensor_id": handle.tensor_id,
            "meta": handle.metadata.to_dict(),
            "ipc_args": pickle.dumps(handle.metadata.ipc_args),
        }
        logger.debug(
            "split: replaced tensor id=%s shape=%s",
            handle.tensor_id,
            handle.metadata.shape,
        )
        return marker, [handle.tensor_id]

    if isinstance(obj, dict):
        result: dict[str, Any] = {}
        ids: list[str] = []
        for k, v in obj.items():
            result[k], sub_ids = _split_gpu_tensors_impl(
                v, transport, router=router, dst_device=dst_device)
            ids.extend(sub_ids)
        return result, ids

    if isinstance(obj, (list, tuple)):
        items: list[Any] = []
        ids: list[str] = []
        for v in obj:
            item, sub_ids = _split_gpu_tensors_impl(
                v, transport, router=router, dst_device=dst_device)
            items.append(item)
            ids.extend(sub_ids)
        return type(obj)(items), ids

    return obj, []


def reassemble_gpu_tensors(
    obj: Any,
    transport: Any,
    return_tensor_ids: bool = False,
) -> Any:
    """Walk *obj* and replace every ``__gpux__`` marker with a real GPU tensor.

    By default returns only the restored object for compatibility with older
    callers.  If *return_tensor_ids* is True, returns
    ``(restored_obj, tensor_ids)`` so callers do not need a separate
    ``_collect_tensor_ids`` pass.

    Handles both IPC markers (original path) and inline markers
    (``inline_data`` key present).
    """
    restored, tensor_ids = _reassemble_gpu_tensors_impl(obj, transport)
    if return_tensor_ids:
        return restored, tensor_ids
    return restored


def _reassemble_gpu_tensors_impl(obj: Any, transport: Any) -> tuple[Any, list[str]]:
    if isinstance(obj, dict) and obj.get(_GPUX_MARKER):
        tid = obj.get("tensor_id", "")
        # Inline path: recover from CPU bytes
        if "inline_data" in obj:
            return _recover_inline_tensor(obj), []

        # IPC path (original)
        meta = TensorMetadata.from_dict(obj["meta"])
        meta.ipc_args = (
            pickle.loads(obj["ipc_args"])
            if isinstance(obj["ipc_args"], bytes)
            else obj["ipc_args"]
        )
        handle = TransportHandle(tensor_id=tid, metadata=meta)
        tensor = transport.recv(
            handle,
            src_rank=transport._config.src_device,
            dst_device=_resolve_local_device(meta.dst_device),
        )
        return tensor, [tid] if tid else []

    if isinstance(obj, dict):
        result: dict[str, Any] = {}
        ids: list[str] = []
        for k, v in obj.items():
            result[k], sub_ids = _reassemble_gpu_tensors_impl(v, transport)
            ids.extend(sub_ids)
        return result, ids

    if isinstance(obj, (list, tuple)):
        items: list[Any] = []
        ids: list[str] = []
        for v in obj:
            item, sub_ids = _reassemble_gpu_tensors_impl(v, transport)
            items.append(item)
            ids.extend(sub_ids)
        return type(obj)(items), ids

    return obj, []
