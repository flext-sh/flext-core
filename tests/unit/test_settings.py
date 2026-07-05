"""Behavior contract for flext_core.FlextSettings — public API only.

Every test asserts observable behaviour of the public settings surface
(``fetch_global``/``update_global``/``clone``/``fetch_namespace``/namespace
registry/validation) — never private attributes, internal helpers, or
implementation structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from flext_core import FlextSettings, FlextSettingsBase

if TYPE_CHECKING:
    from tests.typings import t


class TestsFlextCoreSettings:
    """Public-contract behaviour for the exported settings facade."""

    def test_fetch_global_returns_stable_singleton(self) -> None:
        first = FlextSettings.fetch_global()
        second = FlextSettings.fetch_global()

        assert first is second

    def test_reset_for_testing_yields_a_fresh_singleton(self) -> None:
        first = FlextSettings.fetch_global()

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

    def test_update_global_without_overrides_returns_current_singleton(self) -> None:
        current = FlextSettings.fetch_global()

        assert FlextSettings.update_global() is current

    def test_update_global_rejects_unknown_field_typo(self) -> None:
        FlextSettings.fetch_global()

        with pytest.raises(ValueError, match=r"FlextSettings.*nonexistent_field"):
            FlextSettings.update_global(nonexistent_field=42)

    def test_validate_overrides_rejects_unknown_field(self) -> None:
        with pytest.raises(ValueError, match=r"FlextSettings.*typo_field"):
            FlextSettings.validate_overrides(typo_field="x")

    def test_validate_overrides_accepts_declared_fields(self) -> None:
        # Known fields must not raise — silent success is the contract.
        FlextSettings.validate_overrides(app_name="ok", debug=True)

    def test_clone_returns_isolated_snapshot_without_mutating_global(self) -> None:
        original = FlextSettings.fetch_global()
        original_app_name = original.app_name

        snapshot = original.clone(app_name="cloned-snapshot")

        assert snapshot is not original
        assert snapshot.app_name == "cloned-snapshot"
        assert original.app_name == original_app_name
        assert FlextSettings.fetch_global().app_name == original_app_name

    def test_clone_without_overrides_is_a_distinct_equal_copy(self) -> None:
        original = FlextSettings.fetch_global()

        copy = original.clone()

        assert copy is not original
        assert copy.model_dump() == original.model_dump()

    def test_fetch_global_with_overrides_does_not_mutate_singleton(self) -> None:
        singleton = FlextSettings.fetch_global()
        singleton_app_name = singleton.app_name

        derived = FlextSettings.fetch_global(overrides={"app_name": "derived"})

        assert derived is not singleton
        assert derived.app_name == "derived"
        assert FlextSettings.fetch_global().app_name == singleton_app_name

    def test_clone_for_injection_returns_none_for_default_path(self) -> None:
        assert FlextSettings.clone_for_injection(None) is None

    def test_clone_for_injection_clones_provided_instance(self) -> None:
        original = FlextSettings.fetch_global()

        cloned = FlextSettings.clone_for_injection(original)

        assert cloned is not None
        assert cloned is not original
        assert cloned.app_name == original.app_name

    def test_apply_override_reports_success_only_for_known_fields(self) -> None:
        settings = FlextSettings.fetch_global().clone()

        assert settings.apply_override("app_name", "renamed") is True
        assert settings.app_name == "renamed"
        assert settings.apply_override("no_such_field", "value") is False

    @pytest.mark.parametrize(
        "database_url",
        [
            "postgresql://localhost/db",
            "mysql://localhost/db",
            "sqlite:///local.db",
        ],
    )
    def test_valid_database_url_schemes_are_accepted(
        self,
        database_url: str,
    ) -> None:
        clone = FlextSettings.fetch_global().clone(database_url=database_url)

        assert clone.database_url == database_url

    def test_invalid_database_url_scheme_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FlextSettings.fetch_global().clone(database_url="ftp://host/db")

    def test_trace_without_debug_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FlextSettings.fetch_global().clone(trace=True, debug=False)

    def test_trace_with_debug_is_accepted(self) -> None:
        clone = FlextSettings.fetch_global().clone(trace=True, debug=True)

        assert clone.trace is True
        assert clone.debug is True

    def test_register_namespace_exposes_singleton_via_fetch_namespace(self) -> None:
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
            assert FlextSettings.resolve_namespace_settings(namespace) is (
                NamespaceSettings
            )
            assert isinstance(resolved, NamespaceSettings)
            assert resolved is NamespaceSettings.fetch_global()
        finally:
            NamespaceSettings.reset_for_testing()

    def test_fetch_namespace_rejects_unregistered_namespace(self) -> None:
        with pytest.raises(ValueError, match=r"not registered"):
            FlextSettings.fetch_global().fetch_namespace(
                "definitely_unregistered_namespace",
                FlextSettings,
            )

    def test_fetch_namespace_rejects_type_mismatch(self) -> None:
        class RegisteredSettings(FlextSettings):
            pass

        class OtherSettings(FlextSettings):
            pass

        namespace = "tests_flext_settings_mismatch"
        RegisteredSettings.reset_for_testing()
        FlextSettings.register_namespace(namespace, RegisteredSettings)

        try:
            with pytest.raises(TypeError):
                FlextSettings.fetch_global().fetch_namespace(
                    namespace,
                    OtherSettings,
                )
        finally:
            RegisteredSettings.reset_for_testing()

    def test_namespace_attribute_access_resolves_registered_settings(self) -> None:
        class AttrSettings(FlextSettings):
            pass

        namespace = "attrnamespace"
        AttrSettings.reset_for_testing()
        FlextSettings.register_namespace(namespace, AttrSettings)

        try:
            resolved = getattr(FlextSettings.fetch_global(), namespace)

            assert isinstance(resolved, AttrSettings)
        finally:
            AttrSettings.reset_for_testing()

    def test_unknown_namespace_attribute_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError):
            _ = FlextSettings.fetch_global().totally_unknown_namespace

    def test_for_context_applies_registered_overrides(self) -> None:
        context_id = "tests_flext_settings_ctx"
        FlextSettings.register_context_overrides(context_id, app_name="ctx-app")

        contextual = FlextSettings.for_context(context_id)

        assert contextual.app_name == "ctx-app"
        assert FlextSettings.fetch_global().app_name != "ctx-app"

    def test_for_context_without_overrides_returns_global(self) -> None:
        contextual = FlextSettings.for_context("unknown_context_no_overrides")

        assert contextual is FlextSettings.fetch_global()

    def test_fetch_global_validation_failure_does_not_poison_singleton(self) -> None:
        class RequiredSettings(FlextSettingsBase):
            required_value: str

        RequiredSettings.reset_for_testing()
        try:
            with pytest.raises(ValidationError):
                RequiredSettings.fetch_global()

            # Non-poisoning is observable: a later valid resolution succeeds
            # rather than replaying the cached failure.
            resolved = RequiredSettings.fetch_global(
                overrides={"required_value": "configured"},
            )

            assert resolved.required_value == "configured"

            valid = RequiredSettings.fetch_global(
                overrides={"required_value": "second"},
            )
            assert valid.required_value == "second"
        finally:
            RequiredSettings.reset_for_testing()


__all__: t.MutableSequenceOf[str] = ["TestsFlextCoreSettings"]
