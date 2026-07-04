"""Behavior contract for flext_core.FlextSettings — public API only."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from flext_core import FlextSettings, FlextSettingsBase

if TYPE_CHECKING:
    from tests.typings import t


class TestsFlextSettings:
    """Coverage for the exported ``TestsFlextSettings`` test surface."""

    def test_fetch_namespace_returns_registered_settings_singleton(self) -> None:
        class NamespaceSettings(FlextSettings):
            pass

        namespace = "tests_flext_settings_namespace"
        NamespaceSettings.reset_for_testing()
        FlextSettings.register_namespace(namespace, NamespaceSettings)

        try:
            resolved = FlextSettings.fetch_global().fetch_namespace(
                namespace,
                NamespaceSettings,
            )

            assert namespace in FlextSettings.registered_namespaces()
            assert isinstance(resolved, NamespaceSettings)
            assert resolved is NamespaceSettings.fetch_global()
        finally:
            NamespaceSettings.reset_for_testing()

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

    def test_fetch_global_validation_failure_does_not_poison_singleton(self) -> None:
        class RequiredSettings(FlextSettingsBase):
            required_value: str

        RequiredSettings.reset_for_testing()
        try:
            with pytest.raises(ValidationError):
                RequiredSettings.fetch_global()

            assert RequiredSettings._instance is None

            resolved = RequiredSettings.fetch_global(
                overrides={"required_value": "configured"},
            )

            assert resolved.required_value == "configured"
            assert RequiredSettings._instance is None
        finally:
            RequiredSettings.reset_for_testing()


__all__: t.MutableSequenceOf[str] = ["TestsFlextSettings"]
