from __future__ import annotations

import pytest

from vllm_omni.config.future_scheduler_config import (
    CONFIG_PATH_ENV,
    _LEGACY_OVERRIDES,
    load_future_scheduler_config,
)


def _clear_scheduler_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CONFIG_PATH_ENV, raising=False)
    monkeypatch.delenv("VLLM_OMNI_COST_AWARE_SCHEDULER", raising=False)
    monkeypatch.delenv("VLLM_OMNI_FUTURE_DIAGNOSTICS", raising=False)
    for name in _LEGACY_OVERRIDES:
        monkeypatch.delenv(name, raising=False)


def test_defaults_do_not_enable_scheduler(monkeypatch):
    _clear_scheduler_environment(monkeypatch)

    config = load_future_scheduler_config()

    assert not config.enabled
    assert config.decision.max_wait_ms == 50
    assert config.transfer.delivery_margin_ms == 10


def test_load_grouped_yaml(monkeypatch, tmp_path):
    _clear_scheduler_environment(monkeypatch)
    path = tmp_path / "future.yaml"
    path.write_text(
        """
enabled: true
decision:
  max_wait_ms: 75
  min_gain_ms: 3
prediction:
  mode: learn
  min_samples: 64
registry:
  path: /tmp/future-test.sqlite3
""",
        encoding="utf-8",
    )
    monkeypatch.setenv(CONFIG_PATH_ENV, str(path))

    config = load_future_scheduler_config()

    assert config.enabled
    assert config.decision.max_wait_ms == 75
    assert config.decision.min_gain_ms == 3
    assert config.prediction.mode == "learn"
    assert config.prediction.min_samples == 64
    assert config.registry.path == "/tmp/future-test.sqlite3"
    assert config.transfer.ewma_alpha == 0.2


def test_legacy_environment_overrides_yaml(monkeypatch, tmp_path):
    _clear_scheduler_environment(monkeypatch)
    path = tmp_path / "future.yaml"
    path.write_text(
        """
enabled: false
decision:
  max_wait_ms: 75
diagnostics:
  enabled: false
""",
        encoding="utf-8",
    )
    monkeypatch.setenv(CONFIG_PATH_ENV, str(path))
    monkeypatch.setenv("VLLM_OMNI_COST_AWARE_SCHEDULER", "1")
    monkeypatch.setenv("VLLM_OMNI_COST_AWARE_FUTURE_WAIT_MS", "125")
    monkeypatch.setenv("VLLM_OMNI_FUTURE_DIAGNOSTICS", "true")

    config = load_future_scheduler_config()

    assert config.enabled
    assert config.decision.max_wait_ms == 125
    assert config.diagnostics.enabled


def test_unknown_field_fails_fast(monkeypatch, tmp_path):
    _clear_scheduler_environment(monkeypatch)
    path = tmp_path / "future.yaml"
    path.write_text("decision:\n  max_wiat_ms: 5\n", encoding="utf-8")
    monkeypatch.setenv(CONFIG_PATH_ENV, str(path))

    with pytest.raises(ValueError, match="Invalid future scheduler config field"):
        load_future_scheduler_config()
