"""Public alias inspection distinguishes legal deferral from causal failures.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import warnings

import pytest

from flext_core import c, m, u
from tests.fixtures.type_aliases import TestsTypeAliasDeclarations


class TestsTypeAliasResolution:
    """Tests for ``TypeAliasResolution``."""

    @staticmethod
    def test_resolved_alias_keeps_its_runtime_value() -> None:
        """Test resolved alias keeps its runtime value."""
        alias = TestsTypeAliasDeclarations.LegalTypes.Resolved

        assert u.resolve_type_alias_value(alias) is alias.__value__
        assert not u.alias_contains_any(alias)

    @staticmethod
    def test_declared_static_only_alias_stays_lazy() -> None:
        """Test declared static only alias stays lazy."""
        alias = TestsTypeAliasDeclarations.LegalTypes.Deferred

        with pytest.raises(NameError) as raised:
            _ = alias.__value__

        assert raised.value.name == "DeferredPath"
        owner = TestsTypeAliasDeclarations.LegalTypes
        assert u.resolve_type_alias_value(alias, owner=owner) is None
        assert not u.alias_contains_any(alias, owner=owner)
        assert u.unwrap_type_alias(alias, owner=owner) is alias

    @staticmethod
    def test_report_accounts_for_each_deferred_alias_rule() -> None:
        """Test report accounts for each deferred alias rule."""
        target = TestsTypeAliasDeclarations.LegalTypes
        report = u.check(target, layer=c.EnforcementLayer.TYPES.lower())

        assert not report.complete
        assert report.deferred
        expected = {f"{target.__qualname__}.{name}" for name in ("Deferred", "Nested")}
        assert {item.location for item in report.deferred} == expected
        assert all(item.tag == "alias_any" for item in report.deferred)
        assert all(
            item.alias.unavailable_imports == ("DeferredPath",)
            for item in report.deferred
        )
        assert all(item.alias.module == target.__module__ for item in report.deferred)
        assert all(item.alias.line_number > 0 for item in report.deferred)
        assert all(
            item.alias.qualname.endswith(".Deferred") for item in report.deferred
        )

    @staticmethod
    def test_deferred_report_is_not_emitted_as_a_runtime_violation() -> None:
        """Test deferred report is not emitted as a runtime violation."""
        report = u.check(
            TestsTypeAliasDeclarations.LegalTypes,
            layer=c.EnforcementLayer.TYPES.lower(),
        )
        deferred = m.Report(deferred=report.deferred)

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            u.emit(deferred, mode=c.EnforcementMode.STRICT)

        assert not deferred.complete
        assert not deferred.violations
        assert not recorded
        assert deferred.model_dump()["complete"] is False

    @staticmethod
    def test_local_alias_does_not_inherit_a_name_based_exemption() -> None:
        """Test local alias does not inherit a name based exemption."""
        alias = TestsTypeAliasDeclarations.unbound()

        with pytest.raises(NameError) as raised:
            u.resolve_type_alias_value(alias)

        assert raised.value.name == "DeferredPath"

    @staticmethod
    def test_alias_without_owner_context_propagates_its_original_error() -> None:
        """Test alias without owner context propagates its original error."""
        alias = TestsTypeAliasDeclarations.LegalTypes.Deferred

        with pytest.raises(NameError) as raised:
            u.resolve_type_alias_value(alias)

        assert raised.value.name == "DeferredPath"

    @staticmethod
    def test_unrelated_owner_context_cannot_prove_a_deferral() -> None:
        """Test unrelated owner context cannot prove a deferral."""
        alias = TestsTypeAliasDeclarations.LegalTypes.Deferred

        with pytest.raises(NameError) as raised:
            u.resolve_type_alias_value(
                alias,
                owner=TestsTypeAliasDeclarations.InvalidTypes,
            )

        assert raised.value.name == "DeferredPath"

    @staticmethod
    def test_guarded_import_in_other_scope_does_not_hide_missing_name() -> None:
        """Test guarded import in other scope does not hide missing name."""
        with pytest.raises(NameError) as raised:
            u.alias_contains_any(TestsTypeAliasDeclarations.InvalidTypes.Invalid)

        assert raised.value.name == "Absent"

    @staticmethod
    def test_unexpected_alias_attribute_error_escapes_the_consumer() -> None:
        """Test unexpected alias attribute error escapes the consumer."""
        with pytest.raises(AttributeError) as raised:
            u.alias_contains_any(
                TestsTypeAliasDeclarations.InvalidAttributeTypes.Invalid,
            )

        assert raised.value.name == "Value"

    @staticmethod
    def test_check_does_not_turn_an_invalid_alias_into_an_empty_report() -> None:
        """Test check does not turn an invalid alias into an empty report."""
        with pytest.raises(NameError) as raised:
            u.check(
                TestsTypeAliasDeclarations.InvalidTypes,
                layer=c.EnforcementLayer.TYPES.lower(),
            )

        assert raised.value.name == "Absent"

    @staticmethod
    def test_complete_report_requires_no_deferred_inspection() -> None:
        """Test complete report requires no deferred inspection."""
        assert m.Report().complete

    @staticmethod
    def test_deferred_report_validates_and_round_trips_through_public_models() -> None:
        """Test deferred report validates and round trips through public models."""
        report = u.check(
            TestsTypeAliasDeclarations.LegalTypes,
            layer=c.EnforcementLayer.TYPES.lower(),
        )
        assert report.deferred
        restored = m.Report.model_validate_json(report.model_dump_json(round_trip=True))

        assert restored.deferred == report.deferred
        assert not restored.complete
        assert m.Report.model_json_schema()
