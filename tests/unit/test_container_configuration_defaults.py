"""Tests for FlextContainer configuration integration with FlextConfig."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import FlextConfig, FlextContainer

if TYPE_CHECKING:
    import pytest


def test_container_reads_flext_config_from_environment(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """FlextContainer should honor environment-based FlextConfig values."""

    FlextConfig.reset_global_instance()
    monkeypatch.setenv("FLEXT_MAX_WORKERS", "9")
    monkeypatch.setenv("FLEXT_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("FLEXT_ENVIRONMENT", "staging")

    try:
        container = FlextContainer()
        config = container.get_config()

        assert config["max_workers"] == 9
        assert config["timeout_seconds"] == 45.0
        assert config["environment"] == "staging"
    finally:
        FlextConfig.reset_global_instance()


def test_configure_container_preserves_overrides() -> None:
    """Overrides supplied to FlextContainer should persist across updates."""

    FlextConfig.reset_global_instance()
    FlextConfig.set_global_instance(
        FlextConfig.create(
            max_workers=12,
            timeout_seconds=90,
            environment="production",
        )
    )

    try:
        container = FlextContainer()

        initial = container.get_config()
        assert initial["max_workers"] == 12
        assert initial["timeout_seconds"] == 90.0
        assert initial["environment"] == "production"

        container.configure_container({"timeout_seconds": 15})

        after_timeout = container.get_config()
        assert after_timeout["max_workers"] == 12
        assert after_timeout["timeout_seconds"] == 15.0
        assert after_timeout["environment"] == "production"

        container.configure_container({"environment": "qa"})

        final_config = container.get_config()
        assert final_config["max_workers"] == 12
        assert final_config["timeout_seconds"] == 15.0
        assert final_config["environment"] == "qa"
    finally:
        FlextConfig.reset_global_instance()
