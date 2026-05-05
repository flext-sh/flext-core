"""Behavior contract for flext_core.FlextSettings — public API only."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from flext_core import FlextSettings
from tests import t


@pytest.fixture(autouse=True)
def reset_flext_settings_singleton() -> Generator[None]:
    """Isolate singleton state across settings tests."""
    FlextSettings.reset_for_testing()
    try:
        yield
    finally:
        FlextSettings.reset_for_testing()


class TestsFlextSettings:
    """Coverage for the exported ``TestsFlextSettings`` test surface."""

    def test_reset_for_testing_drops_cached_singleton(self) -> None:
        first = FlextSettings.fetch_global()
        assert FlextSettings.fetch_global() is first

        FlextSettings.reset_for_testing()

        second = FlextSettings.fetch_global()
        assert second is not first

    def test_update_global_replaces_singleton_and_propagates(self) -> None:
        first = FlextSettings.fetch_global()
        assert first.app_name != "test-via-update"

        updated = FlextSettings.update_global(app_name="test-via-update")

        assert updated is FlextSettings.fetch_global()
        assert updated.app_name == "test-via-update"
        assert updated is not first

    def test_update_global_rejects_unknown_field_typo(self) -> None:
        FlextSettings.fetch_global()
        with pytest.raises(ValueError, match=r"FlextSettings.*nonexistent_field"):
            FlextSettings.update_global(nonexistent_field=42)

    def test_clone_returns_isolated_snapshot_without_mutating_global(self) -> None:
        original = FlextSettings.fetch_global()
        original_app_name = original.app_name

        snapshot = original.clone(app_name="cloned-snapshot")

        assert snapshot is not original
        assert snapshot.app_name == "cloned-snapshot"
        assert original.app_name == original_app_name
        assert FlextSettings.fetch_global().app_name == original_app_name

    def test_clone_for_injection_returns_none_for_default_path(self) -> None:
        result = FlextSettings.clone_for_injection(None)
        assert result is None

    def test_clone_for_injection_clones_provided_instance(self) -> None:
        original = FlextSettings.fetch_global()
        cloned = FlextSettings.clone_for_injection(original)
        assert cloned is not None
        assert cloned is not original
        assert cloned.app_name == original.app_name


__all__: t.MutableSequenceOf[str] = ["TestsFlextSettings"]
