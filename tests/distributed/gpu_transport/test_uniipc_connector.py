"""Integration tests for UniIPCConnector."""
from __future__ import annotations

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Need at least 2 GPUs for GPU transport")]


class TestUniIPCConnector:
    def test_put_get_without_gpu_tensor(self):
        """Pure metadata payload passes through unchanged."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector)

        config = {
            "stage_id": 0,
            "device": "cuda:0",
            "shm_threshold_bytes": 65536,
            "gpu_transport_mode": "cuda_ipc",
            "src_device": 0,
            "dst_device": 1,
        }
        connector = UniIPCConnector(config)
        data = {"a": 1, "b": "hello", "c": [1, 2, 3]}

        success, size, metadata = connector.put("0", "1", "test-cpu", data)
        assert success is True
        assert size > 0

        result = connector.get("0", "1", "test-cpu", metadata)
        assert result is not None
        obj, _ = result
        assert obj == data

        connector.cleanup("test-cpu")
        connector.close()

    def test_factory_creates_connector(self):
        """OmniConnectorFactory can create UniIPCConnector from spec."""
        from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
        from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector)

        spec = ConnectorSpec(
            name="UniIPCConnector",
            extra={
                "stage_id": 0,
                "gpu_transport_mode": "cuda_ipc",
                "src_device": 0,
                "dst_device": 1,
            },
        )
        connector = OmniConnectorFactory.create_connector(spec)
        assert isinstance(connector, UniIPCConnector)
        assert connector._transport_mode == "cuda_ipc"
        connector.close()

    def test_close_idempotent(self):
        """close() is safe to call multiple times."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector)

        connector = UniIPCConnector({
            "stage_id": 0,
            "gpu_transport_mode": "cuda_ipc",
            "src_device": 0,
            "dst_device": 1,
        })
        connector.close()
        connector.close()  # second close should not raise

    def test_health_reports_metrics(self):
        """health() returns expected fields."""
        from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import (
            UniIPCConnector)

        connector = UniIPCConnector({
            "stage_id": 0,
            "gpu_transport_mode": "cuda_copy",
            "src_device": 0,
            "dst_device": 1,
        })
        h = connector.health()
        assert h["status"] == "healthy"
        assert h["transport_mode"] == "cuda_copy"
        assert "puts" in h
        assert "gets" in h
        assert "gpu_tensors_sent" in h
        assert "gpu_tensors_recv" in h
        connector.close()

    def test_factory_list_includes_connector(self):
        """UniIPCConnector is in the factory registry."""
        from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
        connectors = OmniConnectorFactory.list_registered_connectors()
        assert "UniIPCConnector" in connectors
