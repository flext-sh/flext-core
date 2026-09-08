"""Behavior contract for flext_core config/settings — ADR-005 canonical singletons.

Asserts the locked operator law: ``config`` and ``settings`` are PRE-INSTANTIATED
namespaced singletons imported directly (``from flext_core import config, settings``)
and used directly (no ``self.`` accessor, not embedded in classes). ``config`` is an
OPEN pydantic-settings object (no declared model fields, ``extra='allow'``) auto-loaded
from ``config/*.yaml``. All legacy access forms are exterminated: no ``apply_override``
shim, no ``def settings(self) -> XSettings: return XSettings.fetch_global()`` overrides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_tests import tm
from pydantic import ValidationError

import flext_core as fc
from flext_core import FlextConfig, FlextSettings, config, settings

if TYPE_CHECKING:
    from pathlib import Path


class TestsFlextCoreConfigSettingsCanonical:
    """Public-contract behaviour for the canonical config/settings singletons."""

    def test_config_is_preinstantiated_frozen_singleton(self) -> None:
        """S1: ``config`` is a ready-to-use frozen FlextConfig instance; mutation raises."""
        assert isinstance(config, FlextConfig)
        assert config is FlextConfig.fetch_global()
        tm.rejects_assignment(config, "anything", "mutated", expected=ValidationError)

    def test_settings_is_preinstantiated_usable_singleton(self) -> None:
        """S2: ``settings`` is a ready-to-use FlextSettings instance used directly."""
        assert isinstance(settings, FlextSettings)
        assert isinstance(settings.model_dump(), dict)

    def test_config_subclasses_keep_independent_singletons(
        self, tmp_path: Path
    ) -> None:
        """Creating and resetting a child never borrows or resets its parent slot."""

        class ParentConfig(FlextConfig):
            CONFIG_DIR = str(tmp_path)

        parent = ParentConfig.fetch_global()

        class ChildConfig(ParentConfig):
            pass

        class SiblingConfig(ParentConfig):
            pass

        child = ChildConfig.fetch_global()
        sibling = SiblingConfig.fetch_global()
        assert type(parent) is ParentConfig
        assert type(child) is ChildConfig
        assert type(sibling) is SiblingConfig
        assert ChildConfig.fetch_global() is child
        ChildConfig.reset_for_testing()
        assert ChildConfig.fetch_global() is not child
        assert ParentConfig.fetch_global() is parent
        assert SiblingConfig.fetch_global() is sibling

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
