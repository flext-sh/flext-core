"""Container configuration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from flext_tests import tm

from flext_core._settings import FlextSettings
from flext_core.container import FlextContainer
from tests.models import TestsFlextModels, m

if TYPE_CHECKING:
    from tests.protocols import p
    from tests.typings import p, t


class TestsFlextCoreContainerConfig:
    @pytest.mark.parametrize(
        "settings",
        TestsFlextModels.Tests.ContainerScenarios.CONFIG_SCENARIOS,
        ids=str,
    )
    def test_configure_container(
        self,
        settings: t.ScalarMapping,
        clean_container: p.Container,
    ) -> None:
        """Test container configuration."""
        container = clean_container
        original_settings = container.snapshot()
        container.apply(settings)
        settings_result = container.snapshot()
        tm.that(
            settings_result,
            is_=m.ConfigMap,
            none=False,
            msg="Container settings must be a ConfigMap",
        )
        for key, value in settings.items():
            if key in original_settings.root:
                tm.that(
                    settings_result.root.get(key),
                    eq=value,
                    msg=f"Settings key {key} must be updated through configure()",
                )
            else:
                tm.that(
                    key in settings_result.root,
                    eq=False,
                    msg=f"Unknown settings key {key} must not leak into public settings",
                )
        if not settings:
            assert settings_result.root == original_settings.root, (
                "Empty configure() input must preserve existing settings"
            )

    def test_with_config_fluent(self, clean_container: p.Container) -> None:
        """Test fluent interface for configuration."""
        container = clean_container
        settings: t.ScalarMapping = {"max_services": 32}
        result = container.apply(settings)
        tm.that(
            result is container,
            eq=True,
            msg="with_config must return self for fluent interface",
        )
        settings_result = container.snapshot()
        tm.that(
            settings_result,
            is_=m.ConfigMap,
            none=False,
            msg="resolve_settings must return a ConfigMap",
        )
        tm.that(
            settings_result.root,
            none=False,
            msg="Settings must be accessible after configure()",
        )
        tm.that(
            settings_result.root.get("max_services"),
            eq=32,
            msg="configure() must expose applied public settings values",
        )

    def test_get_settings(self) -> None:
        """Test retrieving current settings."""
        container = FlextContainer()
        settings = container.snapshot()
        tm.that(
            settings,
            is_=m.ConfigMap,
            none=False,
            msg="resolve_settings must return ConfigMap",
        )
        tm.that(
            "enable_singleton" in settings.root,
            eq=True,
            msg="Config must contain enable_singleton",
        )
        tm.that(
            "max_services" in settings.root,
            eq=True,
            msg="Config must contain max_services",
        )

    def test_apply_none_is_noop_returning_self(
        self, clean_container: p.Container
    ) -> None:
        """apply(None) must be a no-op that preserves settings and returns self."""
        container = clean_container
        before = container.snapshot()
        result = container.apply(None)
        tm.that(
            result is container,
            eq=True,
            msg="apply(None) must return self for fluent chaining",
        )
        tm.that(
            container.snapshot().root,
            eq=before.root,
            msg="apply(None) must leave existing settings unchanged",
        )

    def test_apply_is_idempotent(self, clean_container: p.Container) -> None:
        """Applying the same overrides twice must yield identical public settings."""
        container = clean_container
        settings: t.ScalarMapping = {"max_services": 16, "enable_singleton": True}
        first = container.apply(settings).snapshot()
        second = container.apply(settings).snapshot()
        tm.that(
            second.root,
            eq=first.root,
            msg="Re-applying identical overrides must be idempotent",
        )
        tm.that(
            second.root.get("max_services"),
            eq=16,
            msg="Idempotent apply must retain the applied value",
        )

    def test_config_property(self) -> None:
        """Test accessing settings via property."""
        container = FlextContainer()
        settings = container.settings
        tm.that(
            settings,
            is_=FlextSettings,
            msg="Container settings property must expose FlextSettings",
        )
        assert isinstance(settings, FlextSettings)
        tm.that(
            settings.log_level,
            eq=FlextSettings.fetch_global().log_level,
            msg="Container settings property must reflect the bound public settings",
        )
