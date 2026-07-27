"""Integration contract for FlextSettings — real pydantic-settings behaviour.

Exercises the current canonical surface (singleton lifecycle, env-var loading,
precedence, clone, trace-requires-debug) using the universal scalar fields.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from flext_core import FlextContainer, FlextSettings
from flext_tests import tm


class TestsFlextSettingsIntegration:
    """End-to-end behaviour of the FlextSettings singleton + env loading."""

    def setup_method(self) -> None:
        """Reset global settings before each integration scenario."""
        FlextSettings.reset_for_testing()

    def teardown_method(self) -> None:
        """Reset global settings after each integration scenario."""
        FlextSettings.reset_for_testing()

    def test_fetch_global_returns_same_singleton_instance(self) -> None:
        """fetch_global() always yields the identical global instance."""
        first = FlextSettings.fetch_global()
        second = FlextSettings.fetch_global()
        third = FlextSettings.fetch_global()
        tm.that(first is second is third, eq=True)

    def test_reset_for_testing_replaces_the_global_instance(self) -> None:
        """reset_for_testing() causes the next fetch_global() to be new."""
        before = FlextSettings.fetch_global()
        FlextSettings.reset_for_testing()
        after = FlextSettings.fetch_global()
        tm.that(after is not before, eq=True)

    def test_container_resolves_settings_to_the_global_singleton(self) -> None:
        """The container's settings resolve to the same global instance."""
        global_settings = FlextSettings.fetch_global()
        container = FlextContainer()
        tm.that(container, none=False)
        tm.that(FlextSettings.fetch_global() is global_settings, eq=True)

    def test_default_settings_expose_documented_public_defaults(self) -> None:
        """Universal fields default to documented scalar values."""
        settings = FlextSettings.fetch_global()
        tm.that(settings.debug, eq=False)
        tm.that(settings.trace, eq=False)
        tm.that(
            {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}, has=settings.log_level
        )
        tm.that(settings.timezone, eq="UTC")
        tm.that(settings.async_logging, eq=True)

    def test_environment_variables_override_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FLEXT_-prefixed env vars populate the settings fields."""
        monkeypatch.setenv("FLEXT_LOG_LEVEL", "ERROR")
        FlextSettings.reset_for_testing()
        settings = FlextSettings.fetch_global()
        tm.that(settings.log_level, eq="ERROR")

    def test_explicit_overrides_win_over_defaults(self) -> None:
        """update_global overrides beat the declared defaults."""
        updated = FlextSettings.update_global(log_level="WARNING")
        tm.that(updated.log_level, eq="WARNING")
        tm.that(FlextSettings.fetch_global().log_level, eq="WARNING")

    def test_model_validate_builds_settings_from_a_mapping(self) -> None:
        """model_validate constructs settings from a plain mapping."""
        with FlextSettings.singleton_disabled():
            settings = FlextSettings.model_validate({"log_level": "DEBUG"})
        tm.that(settings.log_level, eq="DEBUG")

    def test_trace_without_debug_is_rejected(self) -> None:
        """trace=True without debug raises the documented invariant error."""
        with pytest.raises(ValidationError, match="trace mode requires debug"):
            FlextSettings.update_global(trace=True, debug=False)

    def test_clone_produces_independent_copy_with_overrides(self) -> None:
        """Clone applies overrides without mutating the source."""
        with FlextSettings.singleton_disabled():
            original = FlextSettings(log_level="INFO")
        clone = original.clone(log_level="CRITICAL")
        tm.that(clone.log_level, eq="CRITICAL")
        tm.that(original.log_level, eq="INFO")
