"""Behavior contract for flext_core config/settings — ADR-005 canonical singletons.

Asserts the locked operator law: ``config`` and ``settings`` are PRE-INSTANTIATED
namespaced singletons imported directly (``from flext_core import config, settings``)
and used directly (no ``self.`` accessor, not embedded in classes). ``config`` is an
OPEN pydantic-settings object (no declared model fields, ``extra='allow'``) auto-loaded
from ``config/*.yaml``. All legacy access forms are exterminated: no ``apply_override``
shim, no ``def settings(self) -> XSettings: return XSettings.fetch_global()`` overrides.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import flext_core as fc
from flext_core import (
    FlextConfig,
    FlextSettings,
    config,
    settings,
)


class TestsFlextCoreConfigSettingsCanonical:
    """Public-contract behaviour for the canonical config/settings singletons."""

    def test_config_is_preinstantiated_frozen_singleton(self) -> None:
        """S1: ``config`` is a ready-to-use frozen FlextConfig instance; mutation raises."""
        assert isinstance(config, FlextConfig)
        assert config is FlextConfig.fetch_global()
        with pytest.raises(ValidationError):
            setattr(config, "anything", "mutated")

    def test_settings_is_preinstantiated_usable_singleton(self) -> None:
        """S2: ``settings`` is a ready-to-use FlextSettings instance used directly."""
        assert isinstance(settings, FlextSettings)
        assert isinstance(settings.model_dump(), dict)

    def test_config_is_open_no_model(self) -> None:
        """S3: config is OPEN (extra=allow, zero declared fields — no app_name)."""
        assert config.model_config.get("extra") == "allow"
        assert "app_name" not in type(config).model_fields

    def test_direct_import_usage_no_self_accessor(self) -> None:
        """S4: root exposes config/settings as instances; no legacy self.settings property."""
        for name in ("config", "settings"):
            assert hasattr(fc, name), name
        # legacy apply_override shim exterminated
        assert getattr(FlextSettings.fetch_global(), "apply_override", None) is None
