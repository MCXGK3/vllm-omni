"""Configuration for the experimental future-aware AR scheduler.

The scheduler spans several processes and modules, so keeping its knobs in
one YAML document is less error-prone than exporting many environment
variables.  ``VLLM_OMNI_FUTURE_SCHEDULER_CONFIG`` is the only recommended
environment variable.  Legacy variables remain supported as final overrides
so existing experiment scripts continue to work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_omni.config.yaml_util import load_yaml_config, to_dict

CONFIG_PATH_ENV = "VLLM_OMNI_FUTURE_SCHEDULER_CONFIG"


@dataclass(frozen=True)
class FutureSchedulerDecisionConfig:
    budget_ms: float = 0.0
    max_wait_ms: float = 50.0
    min_wait_ms: float = 0.1
    min_gain_ms: float = 10.0
    max_future_items: int = 8
    decode_queue_margin_ms: float = 0.0


@dataclass(frozen=True)
class FutureSchedulerRegistryConfig:
    path: str = "/dev/shm/vllm_omni_future_requests.sqlite3"
    ttl_s: float = 120.0
    stage_state_max_age_s: float = 2.0
    stage_state_interval_ms: float = 100.0


@dataclass(frozen=True)
class FutureSchedulerTransferConfig:
    delivery_margin_ms: float = 10.0
    stage1_token_offset: int = 6
    put_default_ms: float = 1.0
    get_default_ms: float = 1.0
    ewma_alpha: float = 0.2
    ipc_base_ms: float = 0.5
    ipc_bandwidth_gbps: float = 20.0


@dataclass(frozen=True)
class FutureSchedulerPredictionConfig:
    mode: str = "static"
    graph_capture_max_tokens: int = 128
    model_path: str = "/dev/shm/vllm_omni_online_cost_model.json"
    min_samples: int = 32
    refit_interval: int = 8
    max_samples: int = 2048
    ridge: float = 0.01
    min_path_samples: int = 16
    min_validation_samples: int = 3
    accept_ratio: float = 1.0


@dataclass(frozen=True)
class FutureSchedulerDiagnosticsConfig:
    enabled: bool = False
    log_every: int = 500


@dataclass(frozen=True)
class FutureSchedulerConfig:
    enabled: bool = False
    decision: FutureSchedulerDecisionConfig = FutureSchedulerDecisionConfig()
    registry: FutureSchedulerRegistryConfig = FutureSchedulerRegistryConfig()
    transfer: FutureSchedulerTransferConfig = FutureSchedulerTransferConfig()
    prediction: FutureSchedulerPredictionConfig = FutureSchedulerPredictionConfig()
    diagnostics: FutureSchedulerDiagnosticsConfig = FutureSchedulerDiagnosticsConfig()


_SECTION_TYPES = {
    "decision": FutureSchedulerDecisionConfig,
    "registry": FutureSchedulerRegistryConfig,
    "transfer": FutureSchedulerTransferConfig,
    "prediction": FutureSchedulerPredictionConfig,
    "diagnostics": FutureSchedulerDiagnosticsConfig,
}

# Deprecated compatibility layer. Values explicitly exported by an old
# experiment take precedence over the YAML file, matching normal environment
# override semantics.
_LEGACY_OVERRIDES: dict[str, tuple[str, str, type]] = {
    "VLLM_OMNI_COST_AWARE_BUDGET_MS": ("decision", "budget_ms", float),
    "VLLM_OMNI_COST_AWARE_FUTURE_WAIT_MS": ("decision", "max_wait_ms", float),
    "VLLM_OMNI_COST_AWARE_MIN_WAIT_MS": ("decision", "min_wait_ms", float),
    "VLLM_OMNI_COST_AWARE_MIN_GAIN_MS": ("decision", "min_gain_ms", float),
    "VLLM_OMNI_FUTURE_MAX_ITEMS": ("decision", "max_future_items", int),
    "VLLM_OMNI_FUTURE_DECODE_QUEUE_MARGIN_MS": (
        "decision",
        "decode_queue_margin_ms",
        float,
    ),
    "VLLM_OMNI_FUTURE_REGISTRY_PATH": ("registry", "path", str),
    "VLLM_OMNI_FUTURE_REGISTRY_TTL_S": ("registry", "ttl_s", float),
    "VLLM_OMNI_FUTURE_STAGE_STATE_MAX_AGE_S": (
        "registry",
        "stage_state_max_age_s",
        float,
    ),
    "VLLM_OMNI_FUTURE_STAGE_STATE_INTERVAL_MS": (
        "registry",
        "stage_state_interval_ms",
        float,
    ),
    "VLLM_OMNI_FUTURE_DELIVERY_MARGIN_MS": (
        "transfer",
        "delivery_margin_ms",
        float,
    ),
    "VLLM_OMNI_FUTURE_STAGE1_TOKEN_OFFSET": (
        "transfer",
        "stage1_token_offset",
        int,
    ),
    "VLLM_OMNI_FUTURE_PUT_DEFAULT_MS": ("transfer", "put_default_ms", float),
    "VLLM_OMNI_FUTURE_GET_DEFAULT_MS": ("transfer", "get_default_ms", float),
    "VLLM_OMNI_FUTURE_TRANSFER_EWMA_ALPHA": (
        "transfer",
        "ewma_alpha",
        float,
    ),
    "VLLM_OMNI_FUTURE_IPC_BASE_MS": ("transfer", "ipc_base_ms", float),
    "VLLM_OMNI_FUTURE_IPC_BANDWIDTH_GBPS": (
        "transfer",
        "ipc_bandwidth_gbps",
        float,
    ),
    "VLLM_OMNI_COST_MODEL_MODE": ("prediction", "mode", str),
    "VLLM_OMNI_COST_AWARE_GRAPH_MAX_TOKENS": (
        "prediction",
        "graph_capture_max_tokens",
        int,
    ),
    "VLLM_OMNI_COST_MODEL_PATH": ("prediction", "model_path", str),
    "VLLM_OMNI_COST_MODEL_MIN_SAMPLES": ("prediction", "min_samples", int),
    "VLLM_OMNI_COST_MODEL_REFIT_INTERVAL": (
        "prediction",
        "refit_interval",
        int,
    ),
    "VLLM_OMNI_COST_MODEL_MAX_SAMPLES": ("prediction", "max_samples", int),
    "VLLM_OMNI_COST_MODEL_RIDGE": ("prediction", "ridge", float),
    "VLLM_OMNI_COST_MODEL_MIN_PATH_SAMPLES": (
        "prediction",
        "min_path_samples",
        int,
    ),
    "VLLM_OMNI_COST_MODEL_MIN_VALIDATION_SAMPLES": (
        "prediction",
        "min_validation_samples",
        int,
    ),
    "VLLM_OMNI_COST_MODEL_ACCEPT_RATIO": (
        "prediction",
        "accept_ratio",
        float,
    ),
    "VLLM_OMNI_FUTURE_DIAGNOSTICS_LOG_EVERY": (
        "diagnostics",
        "log_every",
        int,
    ),
}


def _parse_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _load_document(path: str) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Future scheduler config does not exist: {config_path}")
    document = to_dict(load_yaml_config(config_path))
    if not isinstance(document, dict):
        raise ValueError("Future scheduler config root must be a mapping")
    # Allow the same block to be embedded in a larger YAML document.
    nested = document.get("future_scheduler")
    if nested is not None:
        if not isinstance(nested, dict):
            raise ValueError("future_scheduler must be a mapping")
        document = nested
    return document


def _validate(config: FutureSchedulerConfig) -> None:
    if config.prediction.mode not in {"static", "collect", "learn", "fitted"}:
        raise ValueError(f"Unknown prediction.mode: {config.prediction.mode!r}")
    if config.decision.min_wait_ms < 0:
        raise ValueError("decision.min_wait_ms must be non-negative")
    if config.decision.max_wait_ms < config.decision.min_wait_ms:
        raise ValueError("decision.max_wait_ms must be >= decision.min_wait_ms")
    if config.decision.max_future_items <= 0:
        raise ValueError("decision.max_future_items must be positive")
    if config.registry.ttl_s <= 0 or config.registry.stage_state_interval_ms <= 0:
        raise ValueError("registry TTL and state interval must be positive")
    if not 0 < config.transfer.ewma_alpha <= 1:
        raise ValueError("transfer.ewma_alpha must be in (0, 1]")
    if config.transfer.ipc_bandwidth_gbps <= 0:
        raise ValueError("transfer.ipc_bandwidth_gbps must be positive")
    if config.prediction.graph_capture_max_tokens <= 0:
        raise ValueError("prediction.graph_capture_max_tokens must be positive")


def load_future_scheduler_config() -> FutureSchedulerConfig:
    """Load defaults, optional YAML, then deprecated environment overrides."""
    path = os.getenv(CONFIG_PATH_ENV)
    document = _load_document(path) if path else {}
    allowed = {"enabled", *_SECTION_TYPES}
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ValueError(f"Unknown future scheduler config keys: {unknown}")

    enabled = _parse_bool(document.get("enabled", False), name="enabled")
    sections: dict[str, dict[str, Any]] = {}
    for section_name in _SECTION_TYPES:
        raw_section = document.get(section_name, {})
        if not isinstance(raw_section, dict):
            raise ValueError(f"{section_name} must be a mapping")
        sections[section_name] = dict(raw_section)

    legacy_enabled = os.getenv("VLLM_OMNI_COST_AWARE_SCHEDULER")
    if legacy_enabled is not None:
        enabled = _parse_bool(
            legacy_enabled,
            name="VLLM_OMNI_COST_AWARE_SCHEDULER",
        )
    legacy_diagnostics = os.getenv("VLLM_OMNI_FUTURE_DIAGNOSTICS")
    if legacy_diagnostics is not None:
        sections["diagnostics"]["enabled"] = _parse_bool(
            legacy_diagnostics,
            name="VLLM_OMNI_FUTURE_DIAGNOSTICS",
        )
    for env_name, (section_name, field_name, converter) in _LEGACY_OVERRIDES.items():
        value = os.getenv(env_name)
        if value is not None:
            sections[section_name][field_name] = converter(value)

    try:
        config = FutureSchedulerConfig(
            enabled=enabled,
            **{
                name: section_type(**sections[name])
                for name, section_type in _SECTION_TYPES.items()
            },
        )
    except TypeError as error:
        raise ValueError(f"Invalid future scheduler config field: {error}") from error
    _validate(config)
    return config
