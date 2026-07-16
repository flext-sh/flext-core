"""Behavior contract for flext_core.FlextSettings — public API only.

Tests the minimal canonical surface: ``fetch_global``/``update_global``/
``clone``/``reset_for_testing`` and the universal scalar fields
(``debug``/``trace``/``log_level``/``timezone``/``async_logging``). Namespaced
project fields are plain nested Pydantic-2 model Fields; there is no namespace
registry, ``app_name`` field, ``validate_overrides`` or ``clone_for_injection``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from flext_core import FlextSettings


class TestsFlextCoreSettings:
    """Public-contract behaviour for the exported settings facade."""

    def test_fetch_global_returns_stable_singleton(self) -> None:
        """fetch_global returns the same instance across calls."""
        first = FlextSettings.fetch_global()
        second = FlextSettings.fetch_global()
        assert first is second

    def test_universal_fields_present_with_defaults(self) -> None:
        """The five universal runtime fields exist with scalar defaults."""
        s = FlextSettings.fetch_global()
        assert isinstance(s.debug, bool)
        assert isinstance(s.trace, bool)
        assert isinstance(s.log_level, str)
        assert isinstance(s.timezone, str)
        assert isinstance(s.async_logging, bool)

    def test_update_global_replaces_singleton_and_propagates(self) -> None:
        """update_global installs a new singleton reflected by fetch_global."""
        FlextSettings.reset_for_testing()
        updated = FlextSettings.update_global(log_level="ERROR")
        assert updated.log_level == "ERROR"
        assert FlextSettings.fetch_global().log_level == "ERROR"
        FlextSettings.reset_for_testing()

    def test_update_global_rejects_unknown_field(self) -> None:
        """update_global raises for keys that are not declared fields."""
        with pytest.raises(ValueError, match="Unknown settings override"):
            FlextSettings.update_global(typo_field="x")

    def test_clone_produces_isolated_snapshot(self) -> None:
        """Clone returns an isolated copy without mutating the singleton."""
        FlextSettings.reset_for_testing()
        original = FlextSettings.fetch_global()
        original_level = original.log_level
        snapshot = original.clone(log_level="CRITICAL")
        assert snapshot.log_level == "CRITICAL"
        assert original.log_level == original_level
        assert FlextSettings.fetch_global().log_level == original_level
        FlextSettings.reset_for_testing()

    def test_fetch_global_overrides_returns_isolated_clone(self) -> None:
        """fetch_global(overrides=...) yields a clone, not the singleton."""
        FlextSettings.reset_for_testing()
        singleton = FlextSettings.fetch_global()
        singleton_level = singleton.log_level
        derived = FlextSettings.fetch_global(overrides={"log_level": "WARNING"})
        assert derived.log_level == "WARNING"
        assert FlextSettings.fetch_global().log_level == singleton_level
        FlextSettings.reset_for_testing()

    def test_trace_requires_debug_invariant(self) -> None:
        """trace=True without debug raises the documented validation error."""
        with pytest.raises(ValidationError):
            FlextSettings.update_global(trace=True, debug=False)
        FlextSettings.reset_for_testing()

    def test_reset_for_testing_drops_singleton(self) -> None:
        """reset_for_testing forces a fresh instance on next fetch_global."""
        first = FlextSettings.fetch_global()
        FlextSettings.reset_for_testing()
        second = FlextSettings.fetch_global()
        assert first is not second
