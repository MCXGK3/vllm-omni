# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from typing import Any

from .utils.logging import get_connector_logger

try:
    from .connectors.base import OmniConnectorBase
    from .utils.config import ConnectorSpec
except ImportError:
    # Fallback for direct execution
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from omni_connectors.connectors.base import OmniConnectorBase
    from omni_connectors.utils.config import ConnectorSpec

logger = get_connector_logger(__name__)


class OmniConnectorFactory:
    """Factory for creating OmniConnectors."""

    _registry: dict[str, Callable[[dict[str, Any]], OmniConnectorBase]] = {}

    @classmethod
    def register_connector(cls, name: str, constructor: Callable[[dict[str, Any]], OmniConnectorBase]) -> None:
        """Register a connector constructor."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")
        cls._registry[name] = constructor
        logger.debug(f"Registered connector: {name}")

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
        # Wire ACK pipes that were stripped from extra before vLLM
        # config hash computation (see async_omni_engine.py).
        extra = getattr(spec, "extra", {}) or {}
        stage_id = int(extra.get("stage_id", -1))
        has_setter = hasattr(connector, "set_ack_conns")
        logger.info(
            "Created connector: %s (stage_id=%d has_set_ack=%s extra_keys=%s)",
            spec.name, stage_id, has_setter,
            list(extra.keys()) if extra else [],
        )
        if stage_id >= 0 and has_setter:
            try:
                # Use class-level dict on UniIPCConnector — survives fork
                # where module-level variables are reset on re-import.
                from vllm_omni.distributed.omni_connectors.connectors.uniipc_connector import \
                    UniIPCConnector as _UIPC
                pipes = _UIPC._stage_ack_pipes.get(stage_id, {})
                ack_conn = pipes.get("ack_conn")
                consumer_ack_conn = pipes.get("consumer_ack_conn")
                logger.info(
                    "[Stage-%s] _stage_ack_pipes entry: ack=%s consumer_ack=%s",
                    stage_id, ack_conn is not None, consumer_ack_conn is not None,
                )
                if ack_conn or consumer_ack_conn:
                    connector.set_ack_conns(ack_conn, consumer_ack_conn)
                    logger.info(
                        "[Stage-%s] Wired ACK pipes to %s",
                        stage_id, spec.name,
                    )
            except Exception as e:
                logger.warning(
                    "[Stage-%s] Failed to wire ACK pipes to %s: %s",
                    stage_id, spec.name, e,
                )
        return connector

    @classmethod
    def list_registered_connectors(cls) -> list[str]:
        """List all registered connector names."""
        return list(cls._registry.keys())


# Register built-in connectors with lazy imports
def _create_mooncake_store_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.mooncake_store_connector import MooncakeStoreConnector
    except ImportError:
        # Fallback import
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.mooncake_store_connector import MooncakeStoreConnector
    return MooncakeStoreConnector(config)


def _create_shm_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.shm_connector import SharedMemoryConnector
    except ImportError:
        # Fallback import
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.shm_connector import SharedMemoryConnector
    return SharedMemoryConnector(config)


def _create_uniipc_connector(config: dict[str, Any]) -> OmniConnectorBase:
    from .connectors.uniipc_connector import UniIPCConnector
    return UniIPCConnector(config)


def _create_yuanrong_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.yuanrong_connector import YuanrongConnector
    except ImportError:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.yuanrong_connector import YuanrongConnector
    return YuanrongConnector(config)


def _create_mooncake_transfer_engine_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.mooncake_transfer_engine_connector import MooncakeTransferEngineConnector
    except ImportError:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.mooncake_transfer_engine_connector import MooncakeTransferEngineConnector
    return MooncakeTransferEngineConnector(config)


def _create_uniipc_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.uniipc_connector import UniIPCConnector
    except ImportError:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.uniipc_connector import UniIPCConnector
    return UniIPCConnector(config)


# Register connectors
OmniConnectorFactory.register_connector("MooncakeStoreConnector", _create_mooncake_store_connector)
OmniConnectorFactory.register_connector("MooncakeTransferEngineConnector", _create_mooncake_transfer_engine_connector)
OmniConnectorFactory.register_connector("SharedMemoryConnector", _create_shm_connector)
OmniConnectorFactory.register_connector("UniIPCConnector", _create_uniipc_connector)
OmniConnectorFactory.register_connector("YuanrongConnector", _create_yuanrong_connector)
# Backward-compatible aliases – will be removed in the future
OmniConnectorFactory.register_connector("MooncakeConnector", _create_mooncake_store_connector)
