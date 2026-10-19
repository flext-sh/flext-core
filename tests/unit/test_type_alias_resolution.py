"""Public alias inspection distinguishes legal deferral from causal failures.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import warnings

import pytest
from typing_extensions import TypeForm

from tests import c, m, p, u
from tests.fixtures import type_aliases
from tests.fixtures.type_aliases import TestsFlextTypeAliasDeclarations


class TestsFlextTypeAliasResolution:
    """Tests for ``TypeAliasResolution``."""

    @staticmethod
    def test_registration_ports_validate_without_declaration_rebuild() -> None:
        """Specialized service aliases resolve during native model validation."""
        payload = m.ConfigMap(root={"bootstrap": "value"})
        service = m.ServiceRegistration(name="bootstrap", service=payload)
        assert service.service is payload

        def factory() -> p.ServiceValue:
            return payload

        registration = m.ServiceRegistrationSpec(
            services={"bootstrap": payload},
            factories={"factory": factory},
            resources={"resource": factory},
        )
        assert registration.services is not None
        assert registration.services["bootstrap"] is payload
        assert registration.factories is not None
        assert registration.factories["factory"]() is payload
        assert registration.resources is not None
        assert registration.resources["resource"]() is payload
        assert (
            m.FactoryRegistration(name="factory", factory=factory).factory() is payload
        )
        assert (
            m.ResourceRegistration(name="resource", factory=factory).factory()
            is payload
        )

    @staticmethod
    def test_model_and_logging_ports_resolve_at_validation_boundary() -> None:
        """Protocol-bearing input aliases remain runtime-valid Pydantic inputs."""
        payload = m.ConfigMap(root={"bootstrap": "value"})
        failure = m.ResultFailureSpec(error_data=payload)
        assert failure.error_data is payload
        event = u.type_adapter(TypeForm(p.LoggingEvent)).validate_python({
            "payload": payload,
            "event": "bootstrap",
        })
        assert event["payload"] is payload
        assert event["event"] == "bootstrap"
        overrides = u.type_adapter(
            TypeForm(p.SettingsOverridesMapping)
        ).validate_python({
            "payload": payload,
        })
        assert overrides["payload"] is payload

    @staticmethod
    def test_source_inspection_keeps_parser_and_alias_contracts_distinct() -> None:
        """Source inspection and universal parsing coexist on the public facade."""
        parsed = u.parse("42", int)
        assert parsed.success
        assert parsed.value == 42
        tree = u.parse_module(type_aliases)
        assert isinstance(tree, ast.Module)
        assert any(
            isinstance(node, ast.ClassDef)
            and node.name == TestsFlextTypeAliasDeclarations.__name__
            for node in tree.body
        )
        owner = TestsFlextTypeAliasDeclarations.LegalTypes
        deferred = u.deferred_alias(owner.Deferred, owner=owner)
        assert deferred is not None
        assert deferred.unavailable_imports == ("DeferredPath",)
        assert u.deferred_alias(owner.Resolved, owner=owner) is None
        assert u.deferred(owner.Deferred, owner=owner) == (deferred,)

    @staticmethod
    def test_resolved_alias_keeps_its_runtime_value() -> None:
        """Test resolved alias keeps its runtime value."""
        alias = TestsFlextTypeAliasDeclarations.LegalTypes.Resolved

        assert u.resolve_type_alias_value(alias) is alias.__value__
        assert not u.alias_contains_any(alias)

    @staticmethod
    def test_declared_static_only_alias_stays_lazy() -> None:
        """Test declared static only alias stays lazy."""
        alias = TestsFlextTypeAliasDeclarations.LegalTypes.Deferred

        with pytest.raises(NameError) as raised:
            _ = alias.__value__

        assert raised.value.name == "DeferredPath"
        owner = TestsFlextTypeAliasDeclarations.LegalTypes
        assert u.resolve_type_alias_value(alias, owner=owner) is None
        assert not u.alias_contains_any(alias, owner=owner)
        assert u.unwrap_type_alias(alias, owner=owner) is alias

    @staticmethod
    def test_report_accounts_for_each_deferred_alias_rule() -> None:
        """Test report accounts for each deferred alias rule."""
        target = TestsFlextTypeAliasDeclarations.LegalTypes
        report = u.check(target, layer=c.EnforcementLayer.TYPES.lower())

        assert not report.complete
        assert report.deferred
        expected = {
            f"{target.__qualname__}.{name}"
            for name in ("Deferred", "Nested", "Factory")
        }
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
            TestsFlextTypeAliasDeclarations.LegalTypes,
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
        alias = TestsFlextTypeAliasDeclarations.unbound()

        with pytest.raises(NameError) as raised:
            u.resolve_type_alias_value(alias, owner=m.ArbitraryTypesModel)

        assert raised.value.name == "DeferredPath"

    @staticmethod
    def test_alias_without_owner_context_propagates_its_original_error() -> None:
        """Test alias without owner context propagates its original error."""
        alias = TestsFlextTypeAliasDeclarations.LegalTypes.Deferred

        with pytest.raises(NameError) as raised:
            u.resolve_type_alias_value(alias)

        assert raised.value.name == "DeferredPath"

    @staticmethod
    def test_consumer_context_preserves_the_foreign_alias_declaration() -> None:
        """The consuming model is not the lexical owner of an imported alias."""
        alias = TestsFlextTypeAliasDeclarations.LegalTypes.Deferred

        class Consumer(m.ArbitraryTypesModel):
            value: int = m.Field(default=0, description="Consumer-owned field")
            factory: TestsFlextTypeAliasDeclarations.LegalTypes.Factory | None = (
                m.Field(
                    default=None, description="Foreign alias consumed by this model"
                )
            )

        deferred = u.deferred_alias(alias, owner=Consumer)
        assert deferred is not None
        assert deferred.module == type_aliases.__name__
        assert deferred.qualname == (
            f"{TestsFlextTypeAliasDeclarations.LegalTypes.__qualname__}.Deferred"
        )
        assert deferred.unavailable_imports == ("DeferredPath",)
        assert u.deferred(alias, owner=Consumer) == (deferred,)
        assert u.resolve_type_alias_value(alias, owner=Consumer) is None
        assert Consumer(value=1).value == 1
        report = u.check(Consumer)
        assert not report.complete
        assert any(item.alias == deferred for item in report.deferred)

    @staticmethod
    def test_equal_alias_names_do_not_replace_runtime_identity() -> None:
        """Only the exact alias binding can authenticate its defining AST path."""
        owner = TestsFlextTypeAliasDeclarations.OtherScope
        alias = owner.Deferred
        assert u.resolve_type_alias_value(alias, owner=m.ArbitraryTypesModel) is int
        assert u.deferred_alias(alias, owner=m.ArbitraryTypesModel) is None
        assert (
            u.deferred_alias(
                TestsFlextTypeAliasDeclarations.LegalTypes.Deferred,
                owner=owner,
            )
            is not None
        )

    @staticmethod
    def test_guarded_import_in_other_scope_does_not_hide_missing_name() -> None:
        """Test guarded import in other scope does not hide missing name."""
        with pytest.raises(NameError) as raised:
            u.alias_contains_any(TestsFlextTypeAliasDeclarations.InvalidTypes.Invalid)

        assert raised.value.name == "Absent"

    @staticmethod
    def test_unexpected_alias_attribute_error_escapes_the_consumer() -> None:
        """Test unexpected alias attribute error escapes the consumer."""
        with pytest.raises(AttributeError) as raised:
            u.alias_contains_any(
                TestsFlextTypeAliasDeclarations.InvalidAttributeTypes.Invalid,
                owner=m.ArbitraryTypesModel,
            )

        assert raised.value.name == "Value"

    @staticmethod
    def test_foreign_broken_alias_keeps_its_causal_name_error() -> None:
        """A consumer context cannot normalize an unguarded alias failure."""
        with pytest.raises(NameError) as raised:
            u.resolve_type_alias_value(
                TestsFlextTypeAliasDeclarations.InvalidTypes.Invalid,
                owner=m.ArbitraryTypesModel,
            )
        assert raised.value.name == "Absent"

    @staticmethod
    def test_check_does_not_turn_an_invalid_alias_into_an_empty_report() -> None:
        """Test check does not turn an invalid alias into an empty report."""
        with pytest.raises(NameError) as raised:
            u.check(
                TestsFlextTypeAliasDeclarations.InvalidTypes,
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
            TestsFlextTypeAliasDeclarations.LegalTypes,
            layer=c.EnforcementLayer.TYPES.lower(),
        )
        assert report.deferred
        restored = m.Report.model_validate_json(report.model_dump_json(round_trip=True))

        assert restored.deferred == report.deferred
        assert not restored.complete
        assert m.Report.model_json_schema()
