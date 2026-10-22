"""Behavioral tests for the public enforcement contract (``u.check``).

Every test asserts observable behavior of the public API — the typed
``m.Report`` returned by ``u.check(target)`` and the public predicate
``FlextUtilitiesBeartypeEngine.has_nested_namespace`` — never internal
detection mechanics. A caller depends on: which classes/fields the checker
flags, the guidance carried on each violation, and which shapes are exempt.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import string

import pytest
from typing_extensions import TypeForm

from flext_core.utilities import FlextUtilitiesBeartypeEngine
from tests import m, tm, u
from tests.unit._enforcement_support import make_class, synthetic_method

_INHERITANCE_FRAGMENT = "must inherit FlextSettings"
_ACCESSOR_FRAGMENT = "accessor method"


def _inheritance_violations(cls: type) -> list[m.Violation]:
    """Return inheritance violations for *cls*."""
    return [v for v in u.check(cls).violations if _INHERITANCE_FRAGMENT in v.message]


class TestsFlextCoreEnforcementAccessors:
    """Public enforcement behavior: what ``u.check`` reports to a caller."""

    @staticmethod
    @pytest.mark.parametrize("prefix", ["get_user", "set_config", "is_ready"])
    def test_forbidden_accessor_prefix_is_flagged(prefix: str) -> None:
        # Arrange
        """Test forbidden accessor prefix is flagged."""
        cls = make_class("FlextCoreAccessed", {prefix: synthetic_method})

        # Act
        report = u.check(cls)
        accessor = [v for v in report.violations if _ACCESSOR_FRAGMENT in v.message]

        # Assert — the offending method is named and remediation is offered.
        tm.that(
            accessor,
            truth=True,
            msg=f"{prefix} should be flagged as a forbidden accessor",
        )
        message = accessor[0].message
        tm.that(f'"{prefix}"' in message, truth=True)
        tm.that("fetch_" in message or "computed_field" in message, truth=True)

    @staticmethod
    @pytest.mark.parametrize("prefix", ["fetch_remote", "resolve_ref", "compute_total"])
    def test_domain_verb_method_is_allowed(prefix: str) -> None:
        # Arrange
        """Test domain verb method is allowed."""
        cls = make_class("FlextCoreVerb", {prefix: synthetic_method})

        # Act
        messages = [
            v.message
            for v in u.check(cls).violations
            if _ACCESSOR_FRAGMENT in v.message
        ]

        # Assert
        tm.that(
            messages,
            truth=False,
            msg=f"{prefix} is a domain verb and must not be flagged",
        )

    @staticmethod
    def test_accessor_violation_locates_the_owning_class() -> None:
        # Arrange
        """Test accessor violation locates the owning class."""
        cls = make_class("FlextCoreAccessedGet", {"get_user": synthetic_method})

        # Act
        accessor = next(
            v for v in u.check(cls).violations if _ACCESSOR_FRAGMENT in v.message
        )

        # Assert — public location fields point back at the target.
        tm.that(accessor.qualname, eq="FlextCoreAccessedGet")
        tm.that(
            accessor.message.startswith("FlextCoreAccessedGet.get_user"), truth=True
        )

    @staticmethod
    def test_override_of_a_foreign_contract_keeps_its_name() -> None:
        # Arrange — stdlib string.Formatter declares get_value; a subclass
        # overriding it cannot rename a name the library calls.
        """Test override of a foreign contract keeps its name."""
        cls = type(
            "FlextCoreFormatter",
            (string.Formatter,),
            {"get_value": synthetic_method},
        )
        cls.__qualname__ = cls.__name__
        cls.__module__ = "flext_core.synthetic"

        # Act
        accessor = [
            v for v in u.check(cls).violations if _ACCESSOR_FRAGMENT in v.message
        ]

        # Assert
        tm.that(
            accessor,
            eq=[],
            msg="an override of a foreign contract is not an accessor",
        )

    @staticmethod
    def test_accessor_is_judged_once_at_the_introducing_class() -> None:
        # Arrange — the base introduces get_user; the subclass overrides it.
        """Test accessor is judged once at the introducing class."""
        base = make_class("FlextCoreAccessedBase", {"get_user": synthetic_method})
        child: type = type(
            "FlextCoreAccessedChild",
            (base,),
            {"get_user": synthetic_method},
        )
        child.__qualname__ = child.__name__
        child.__module__ = "flext_core.synthetic"

        # Act
        on_base = [
            v for v in u.check(base).violations if _ACCESSOR_FRAGMENT in v.message
        ]
        on_child = [
            v for v in u.check(child).violations if _ACCESSOR_FRAGMENT in v.message
        ]

        # Assert — reported where the name is introduced, never re-reported.
        tm.that(on_base, truth=True)
        tm.that(on_child, truth=False)

    @staticmethod
    def test_bare_collection_field_is_flagged() -> None:
        # Arrange
        """Test bare collection field is flagged."""

        class _M(m.ArbitraryTypesModel):
            items: list[str] = m.Field(
                default_factory=u.empty(TypeForm(list[str])),
                description="d",
            )

        # Act
        messages = [v.message for v in u.check(_M).violations]

        # Assert
        tm.that(any("bare list" in msg for msg in messages), truth=True)

    @staticmethod
    def test_missing_field_description_names_the_field() -> None:
        # Arrange
        """Test missing field description names the field."""

        class _M(m.ArbitraryTypesModel):
            undoc: str = "x"

        # Act
        messages = [v.message for v in u.check(_M).violations]

        # Assert
        tm.that(
            any(
                'Field "undoc"' in msg and "missing description" in msg
                for msg in messages
            ),
            truth=True,
        )

    @staticmethod
    def test_declared_settings_model_must_inherit_flext_settings() -> None:
        # Arrange — a class that DECLARES itself a pydantic-settings model but
        # bypasses the FlextSettings owner.
        """Test declared settings model must inherit flext settings."""
        cls = type("FlextWorkerSettings", (m.BaseSettings,), {})
        cls.__qualname__ = cls.__name__
        cls.__module__ = "flext_core.synthetic"

        # Act
        inheritance = _inheritance_violations(cls)

        # Assert — flagged, and tagged with the catalog rule id callers can filter on.
        tm.that(inheritance, truth=True)
        tm.that(inheritance[0].rule_id, eq="ENFORCE-042")

    @staticmethod
    def test_settings_named_plain_class_is_not_a_settings_target() -> None:
        # Arrange — a namespace holder whose name ends in "Settings" declares no
        # pydantic-settings base; the rule derives its target from the
        # declaration, never from the name.
        """Test settings named plain class is not a settings target."""
        cls = make_class("FlextWorkerSettings", {})

        # Act — plain class does not inherit FlextSettings
        inheritance = _inheritance_violations(cls)

        # Assert
        tm.that(inheritance, truth=False)

    @staticmethod
    def test_nested_settings_class_is_exempt_from_inheritance_rule() -> None:
        # Arrange — a real inner class inside a namespace container is metadata,
        # not a settings model, so it must not be forced to inherit FlextSettings.
        """Test nested settings class is exempt from inheritance rule."""

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
        tm.that(inheritance, truth=False)

    @staticmethod
    def test_non_settings_class_is_not_flagged_for_inheritance() -> None:
        # Arrange
        """Test non settings class is not flagged for inheritance."""
        cls = make_class("FlextCoreService", {})

        # Act
        inheritance = _inheritance_violations(cls)

        # Assert
        tm.that(inheritance, truth=False)

    @staticmethod
    def test_clean_class_yields_an_empty_report() -> None:
        # Arrange
        """Test clean class yields an empty report."""
        cls = make_class("FlextCoreService", {"fetch_value": synthetic_method})

        # Act
        report = u.check(cls)

        # Assert — the Report public surface reports "nothing wrong".
        tm.that(report.empty, truth=True)
        tm.that(len(report), eq=0)
        tm.that(report, truth=False)
        tm.that(list(report.messages), eq=[])

    @staticmethod
    def test_direct_nested_class_is_a_namespace() -> None:
        # Arrange
        """Test direct nested class is a namespace."""

        class _DirectHolder:
            class _SomeInner:
                pass

            class PublicInner:
                pass

        # Act / Assert
        tm.that(
            FlextUtilitiesBeartypeEngine.has_nested_namespace(_DirectHolder), truth=True
        )

    @staticmethod
    def test_inherited_nested_class_is_a_namespace() -> None:
        # Arrange
        """Test inherited nested class is a namespace."""

        class _Parent:
            class Nested:
                pass

        class _Empty(_Parent):
            pass

        # Act / Assert — inheritance still exposes the nested namespace.
        tm.that(FlextUtilitiesBeartypeEngine.has_nested_namespace(_Empty), truth=True)

    @staticmethod
    def test_plain_class_is_not_a_namespace() -> None:
        # Arrange
        """Test plain class is not a namespace."""

        class _Bare:
            x: int = 1

        # Act / Assert
        tm.that(FlextUtilitiesBeartypeEngine.has_nested_namespace(_Bare), truth=False)
