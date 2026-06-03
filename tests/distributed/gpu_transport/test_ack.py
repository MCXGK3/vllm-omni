# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the ACK thread in CUDA IPC transport.

Verifies that the background ACK thread correctly releases tensors
when release ACKs arrive over the control pipe, and that timeout
cleanup works correctly.
"""
from __future__ import annotations

import importlib.util
import sys
import time
import types

import pytest
import torch
from multiprocessing import Pipe

# ---------------------------------------------------------------------------
# Import hack: set up the vllm_omni package hierarchy in sys.modules WITHOUT
# triggering vllm_omni/__init__.py (which imports vllm, which may fail when
# its dependencies are mismatched, e.g. an incompatible transformers version).
# ---------------------------------------------------------------------------

_VLLM_OMNI_BASE = "/home/multimodal/vllm-omni/vllm_omni"
_GPU_TRANSPORT_PKG = "vllm_omni.distributed.gpu_transport"
_GPU_TRANSPORT_BASE = f"{_VLLM_OMNI_BASE}/distributed/gpu_transport"


def _setup_vllm_omni_hierarchy():
    """Create stub packages for the vllm_omni hierarchy and load
    gpu_transport submodules via importlib.

    Idempotent -- safe to call multiple times.
    """
    # 1. Stub packages above gpu_transport
    for name, path in [
        ("vllm_omni", _VLLM_OMNI_BASE),
        ("vllm_omni.distributed", f"{_VLLM_OMNI_BASE}/distributed"),
    ]:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [path]
            mod.__package__ = name
            sys.modules[name] = mod

    # 2. Stub for gpu_transport itself
    if _GPU_TRANSPORT_PKG not in sys.modules:
        pkg_mod = types.ModuleType(_GPU_TRANSPORT_PKG)
        pkg_mod.__path__ = [_GPU_TRANSPORT_BASE]
        pkg_mod.__package__ = _GPU_TRANSPORT_PKG
        sys.modules[_GPU_TRANSPORT_PKG] = pkg_mod

    # 3. Load submodules in dependency order
    def _load(name, file):
        full = f"{_GPU_TRANSPORT_PKG}.{name}"
        if full in sys.modules:
            return sys.modules[full]
        path = f"{_GPU_TRANSPORT_BASE}/{file}"
        spec = importlib.util.spec_from_file_location(full, path)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = _GPU_TRANSPORT_PKG
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    # Layer 1: no intra-package deps
    _load("logging", "logging.py")
    _load("protocol", "protocol.py")
    _load("config", "config.py")

    # Layer 2: depend on logging
    _load("tensor_registry", "tensor_registry.py")
    _load("ipc_utils", "ipc_utils.py")
    _load("control_channel", "control_channel.py")

    # Layer 3: depend on everything above
    _load("cuda_ipc_transport", "cuda_ipc_transport.py")
    _load("cuda_copy_transport", "cuda_copy_transport.py")


_setup_vllm_omni_hierarchy()

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from vllm_omni.distributed.gpu_transport.config import GPUTransportConfig
from vllm_omni.distributed.gpu_transport.cuda_ipc_transport import CudaIpcTransport
from vllm_omni.distributed.gpu_transport.control_channel import ConsumerControl

pytestmark = [pytest.mark.gpu]


class TestAckThread:
    def test_ack_thread_releases_tensor(self):
        p1, p2 = Pipe()
        transport = CudaIpcTransport(GPUTransportConfig(mode="cuda_ipc"))

        # Start ACK thread manually
        transport._ack_conn = p1
        transport._start_ack_thread()

        t = torch.ones(10, device="cuda:0")
        transport._registry.register("test-ack", t)
        assert transport.registry_size == 1

        # Consumer sends release ACK
        consumer = ConsumerControl(p2)
        consumer.send_ack("test-ack", ack_type="release")
        time.sleep(0.3)

        assert transport.registry_size == 0, "ACK thread should have released"

        transport.shutdown_ack_thread()
        transport.close()
        p1.close()
        p2.close()

    def test_no_ack_conn_no_thread(self):
        transport = CudaIpcTransport(GPUTransportConfig(mode="cuda_ipc"))
        assert transport._ack_thread is None, (
            "Should not start thread without ack_conn"
        )
        transport.close()

    def test_timeout_cleanup_releases_stale(self):
        transport = CudaIpcTransport(
            GPUTransportConfig(mode="cuda_ipc", release_timeout_ms=10)
        )
        t = torch.ones(10, device="cuda:0")
        transport._registry.register("test-timeout", t)
        assert transport.registry_size == 1

        time.sleep(0.1)
        stale = transport.cleanup_timeouts()
        assert "test-timeout" in stale
        assert transport.registry_size == 0
        transport.close()

    def test_ack_thread_handles_shutdown(self):
        p1, p2 = Pipe()
        transport = CudaIpcTransport(GPUTransportConfig(mode="cuda_ipc"))
        transport._ack_conn = p1
        transport._start_ack_thread()

        # Send shutdown via pipe
        p2.send({"type": "shutdown"})
        time.sleep(0.3)

        # Thread should have stopped
        transport._ack_running = False  # ensure
        if transport._ack_thread:
            transport._ack_thread.join(timeout=1)
        assert not (
            transport._ack_thread and transport._ack_thread.is_alive()
        )
        transport.close()
        p1.close()
        p2.close()
