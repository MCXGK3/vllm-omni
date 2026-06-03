# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from .protocol import GPUTensorTransport, TensorMetadata, TransportHandle
from .config import GPUTransportConfig, TransportMode
from .logging import get_logger


def create_transport(config: GPUTransportConfig):
    """Factory: returns the correct transport based on config.mode."""
    if config.mode == "cuda_ipc":
        from .cuda_ipc_transport import CudaIpcTransport
        return CudaIpcTransport(config)
    elif config.mode == "cuda_copy":
        from .cuda_copy_transport import CudaCopyTransport
        return CudaCopyTransport(config)
    raise ValueError(f"Unknown transport mode: {config.mode}")


__all__ = [
    "GPUTensorTransport", "TensorMetadata", "TransportHandle",
    "GPUTransportConfig", "TransportMode", "get_logger",
    "CudaIpcTransport", "CudaCopyTransport", "create_transport",
]
