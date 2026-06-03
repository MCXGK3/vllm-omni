# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Payload split/reassemble for GPU tensor transport.

Walks a nested dict/list structure, replaces GPU tensors with lightweight
``__gpux__`` markers (safe for Pipe/msgpack serialization), and restores
them on the receiver side.
"""
from __future__ import annotations

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


def split_gpu_tensors(obj: Any, transport: Any) -> Any:
    """Walk *obj* and replace every CUDA tensor with a ``__gpux__`` marker.

    The marker contains ``tensor_id`` and ``TensorMetadata.to_dict()`` plus IPC args.
    The transport's ``send()`` is called for each tensor to store it in
    the registry and extract IPC args.

    Returns a (possibly modified) copy of *obj* with no GPU tensors.
    """
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
        handle = transport.send(obj, dst_rank=transport._config.dst_device)
        handle.metadata.ipc_args = transport._ipc_args_store[handle.tensor_id]
        marker = {
            _GPUX_MARKER: True,
            "tensor_id": handle.tensor_id,
            "meta": handle.metadata.to_dict(),
            "ipc_args": handle.metadata.ipc_args,
        }
        logger.debug("split: replaced tensor id=%s shape=%s", handle.tensor_id, handle.metadata.shape)
        return marker

    if isinstance(obj, dict):
        return {k: split_gpu_tensors(v, transport) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return type(obj)(split_gpu_tensors(v, transport) for v in obj)

    return obj


def reassemble_gpu_tensors(obj: Any, transport: Any) -> Any:
    """Walk *obj* and replace every ``__gpux__`` marker with a real GPU tensor.

    The transport's ``recv()`` is called to reconstruct the tensor from
    IPC args. The caller is responsible for calling ``release()`` after
    the tensor is no longer needed.
    """
    if isinstance(obj, dict) and obj.get(_GPUX_MARKER):
        meta = TensorMetadata.from_dict(obj["meta"])
        meta.ipc_args = obj["ipc_args"]
        handle = TransportHandle(tensor_id=obj["tensor_id"], metadata=meta)
        tensor = transport.recv(handle, src_rank=transport._config.src_device,
                               dst_device=meta.dst_device)
        logger.debug("reassemble: restored tensor id=%s shape=%s", meta.tensor_id, meta.shape)
        return tensor

    if isinstance(obj, dict):
        return {k: reassemble_gpu_tensors(v, transport) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return type(obj)(reassemble_gpu_tensors(v, transport) for v in obj)

    return obj
