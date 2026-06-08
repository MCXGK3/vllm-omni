# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Omni Connector Factory — creates connector instances from ConnectorSpec."""

from .base import OmniConnectorBase
from .utils.config import ConnectorSpec
from .utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


class OmniConnectorFactory:
    """Singleton factory for creating OmniConnector instances from specs."""

    _registry: dict[str, type[OmniConnectorBase]] = {}

    @classmethod
    def register_connector(
        cls, name: str, connector_cls: type[OmniConnectorBase],
    ) -> None:
        """Register a connector class."""
        cls._registry[name] = connector_cls
        logger.info(f"Registered connector: {name}")

    @classmethod
    def create_connector(cls, spec: ConnectorSpec) -> OmniConnectorBase:
        """Create a connector from specification."""
        if spec.name not in cls._registry:
            raise ValueError(f"Unknown connector: {spec.name}. Available: {list(cls._registry.keys())}")

        constructor = cls._registry[spec.name]
        try:
            connector = constructor(spec.extra)
        except Exception as e:
            logger.error(f"Failed to create connector {spec.name}: {e}")
            raise ValueError(f"Failed to create connector {spec.name}: {e}")
        # Wire ACK pipes passed through process spawn kwargs.
        extra = getattr(spec, "extra", {}) or {}
        stage_id = int(extra.get("stage_id", -1))
        if stage_id >= 0 and hasattr(connector, "set_ack_conns"):
            try:
                from vllm_omni.engine.stage_engine_core_proc import _WORKER_ACK_PIPES
                pipes = _WORKER_ACK_PIPES.get(stage_id, {})
                ack_conn = pipes.get("ack_conn")
                consumer_ack_conn = pipes.get("consumer_ack_conn")
                if ack_conn or consumer_ack_conn:
                    connector.set_ack_conns(ack_conn, consumer_ack_conn)
            except Exception:
                pass
        logger.info(f"Created connector: {spec.name}")
        return connector

    @classmethod
    def list_registered_connectors(cls) -> list[str]:
        """List all registered connector names."""
        return list(cls._registry.keys())
