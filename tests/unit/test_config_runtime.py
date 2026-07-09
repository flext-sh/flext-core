"""Behavior contract for flext_core.FlextConfig — ADR-005 §7 runtime object.

Asserts the public config surface: frozen singleton, dual independent
``self.settings``/``self.config`` accessors on a service, clean root import
(no cycle), and removal of the legacy ``apply_override`` shim (ADR-005 §8).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from flext_core import FlextConfig, FlextService, FlextSettings, config


class TestsFlextCoreConfigRuntime:
    """Public-contract behaviour for the exported config facade."""

    def test_config_singleton_frozen(self) -> None:
        """S1: config is a stable frozen singleton; mutation raises."""
        first = FlextConfig.fetch_global()
        second = FlextConfig.fetch_global()
        assert first is second
        assert config is FlextConfig
        with pytest.raises(ValidationError):
            first.app_name = "mutated"  # frozen

    def test_service_dual_accessor_independent(self) -> None:
        """S2: self.settings (mutable) and self.config (frozen) are independent."""

        class _Svc(FlextService):
            def execute(self) -> object:
                return None

        svc = _Svc()
        assert isinstance(svc.settings, FlextSettings)
        assert isinstance(svc.config, FlextConfig)
        assert type(svc.settings) is not type(svc.config)
        # settings never exposes config.* and vice versa
        assert not hasattr(svc.settings, "fetch_config")
        assert not isinstance(svc.config, FlextSettings)

    def test_root_facades_and_config_import_clean(self) -> None:
        """S3: root facades + config all resolve, no import cycle."""
        import flext_core as fc

        for name in ("c", "t", "p", "m", "u", "settings", "config"):
            assert hasattr(fc, name), name

    def test_apply_override_removed(self) -> None:
        """S4: legacy apply_override shim is gone; update_global works."""
        settings = FlextSettings.fetch_global()
        assert getattr(settings, "apply_override", None) is None
        updated = FlextSettings.update_global(app_name="renamed-via-update")
        assert updated.app_name == "renamed-via-update"
