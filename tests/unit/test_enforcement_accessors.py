"""Behavioral tests for the public enforcement contract (``u.check``).

Every test asserts observable behavior of the public API — the typed
``m.Report`` returned by ``u.check(target)`` and the public predicate
``FlextUtilitiesBeartypeEngine.has_nested_namespace`` — never internal
detection mechanics. A caller depends on: which classes/fields the checker
flags, the guidance carried on each violation, and which shapes are exempt.
"""

from __future__ import annotations

import pytest

from tests import m
from tests.unit._enforcement_support import make_class
from tests import u

_INHERITANCE_FRAGMENT = "must inherit FlextSettings"
_ACCESSOR_FRAGMENT = "accessor method"


class TestsFlextCoreEnforcementAccessors:
    """Public enforcement behavior: what ``u.check`` reports to a caller."""

    @pytest.mark.parametrize("prefix", ["get_user", "set_config", "is_ready"])
    def test_forbidden_accessor_prefix_is_flagged(self, prefix: str) -> None:
        # Arrange
        cls = make_class("FlextCoreAccessed", {prefix: lambda self: None})

        # Act
        report = u.check(cls)
        accessor = [v for v in report.violations if _ACCESSOR_FRAGMENT in v.message]

        # Assert — the offending method is named and remediation is offered.
        assert accessor, f"{prefix} should be flagged as a forbidden accessor"
        message = accessor[0].message
        assert f'"{prefix}"' in message
        assert "fetch_" in message or "computed_field" in message

    @pytest.mark.parametrize("prefix", ["fetch_remote", "resolve_ref", "compute_total"])
    def test_domain_verb_method_is_allowed(self, prefix: str) -> None:
        # Arrange
        cls = make_class("FlextCoreVerb", {prefix: lambda self: None})

        # Act
        messages = [
            v.message
            for v in u.check(cls).violations
            if _ACCESSOR_FRAGMENT in v.message
        ]

        # Assert
        assert not messages, f"{prefix} is a domain verb and must not be flagged"

    def test_accessor_violation_locates_the_owning_class(self) -> None:
        # Arrange
        cls = make_class("FlextCoreAccessedGet", {"get_user": lambda self: None})

        # Act
        accessor = next(
            v for v in u.check(cls).violations if _ACCESSOR_FRAGMENT in v.message
        )

        # Assert — public location fields point back at the target.
        assert accessor.qualname == "FlextCoreAccessedGet"
        assert accessor.message.startswith("FlextCoreAccessedGet.get_user")

    def test_bare_collection_field_is_flagged(self) -> None:
        # Arrange
        class _M(m.ArbitraryTypesModel):
            items: list[str] = m.Field(default_factory=list, description="d")

        # Act
        messages = [v.message for v in u.check(_M).violations]

        # Assert
        assert any("bare list" in msg for msg in messages)

    def test_missing_field_description_names_the_field(self) -> None:
        # Arrange
        class _M(m.ArbitraryTypesModel):
            undoc: str = "x"

        # Act
        messages = [v.message for v in u.check(_M).violations]

        # Assert
        assert any(
            'Field "undoc"' in msg and "missing description" in msg for msg in messages
        )

    def test_settings_named_class_must_inherit_flext_settings(self) -> None:
        # Arrange
        cls = make_class("FlextWorkerSettings", {})

        # Act
        inheritance = [
            v for v in u.check(cls).violations if _INHERITANCE_FRAGMENT in v.message
        ]

        # Assert — flagged, and tagged with the catalog rule id callers can filter on.
        assert inheritance
        assert inheritance[0].rule_id == "ENFORCE-042"

    def test_nested_settings_class_is_exempt_from_inheritance_rule(self) -> None:
        # Arrange — a real inner class inside a namespace container is metadata,
        # not a settings model, so it must not be forced to inherit FlextSettings.
        class FlextModelsSettings:
            class AutoSettings:
                pass

        inner = FlextModelsSettings.AutoSettings
        inner.__module__ = "flext_core._models.settings"

        # Act
        inheritance = [
            v for v in u.check(inner).violations if _INHERITANCE_FRAGMENT in v.message
        ]

        # Assert
        assert not inheritance

    def test_non_settings_class_is_not_flagged_for_inheritance(self) -> None:
        # Arrange
        cls = make_class("FlextCoreService", {})

        # Act
        inheritance = [
            v for v in u.check(cls).violations if _INHERITANCE_FRAGMENT in v.message
        ]

        # Assert
        assert not inheritance

    def test_clean_class_yields_an_empty_report(self) -> None:
        # Arrange
        cls = make_class("FlextCoreService", {"fetch_value": lambda self: None})

        # Act
        report = u.check(cls)

        # Assert — the Report public surface reports "nothing wrong".
        assert report.empty
        assert len(report) == 0
        assert not report
        assert list(report.messages) == []

    def test_direct_nested_class_is_a_namespace(self) -> None:
        # Arrange
        class _DirectHolder:
            class _SomeInner:
                pass

            class PublicInner:
                pass

        # Act / Assert
        assert u.has_nested_namespace(_DirectHolder)

    def test_inherited_nested_class_is_a_namespace(self) -> None:
        # Arrange
        class _Parent:
            class Nested:
                pass

        class _Empty(_Parent):
            pass

        # Act / Assert — inheritance still exposes the nested namespace.
        assert u.has_nested_namespace(_Empty)

    def test_plain_class_is_not_a_namespace(self) -> None:
        # Arrange
        class _Bare:
            x: int = 1

        # Act / Assert
        assert not u.has_nested_namespace(_Bare)
