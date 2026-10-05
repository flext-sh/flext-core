"""Behavior contract for the typed enforcement source variants.

Exercises the PUBLIC surface of ``EnforcementSourceKind`` and every
``FlextModelsEnforcementSources`` discriminator model: default ``kind``
literals, required-field validation, model_dump round-trips, and the
``EnforcementRuleSpec`` discriminated-union dispatch that consumes them.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from tests.constants import c
from tests.models import m
from tests.typings import t


class TestsFlextModelsTestEnforcementSources:
    """Canonical namespace owner."""

    # One representative valid instance per surviving source variant, keyed by the
    # discriminator literal it must expose on the public ``kind`` field.
    _SOURCE_CASES: ClassVar[dict[str, m.BaseModel]] = {
        "flext_infra_rule": m.EnforcementInfraRuleSource(rule_ids=("ban-cast",)),
        "runtime_warning": m.EnforcementRuntimeWarningSource(
            category="FlextMroWarning",
        ),
        "beartype": m.EnforcementBeartypeSource(tag="no_module_compat_alias"),
        "code_smell": m.EnforcementCodeSmellSource(smell_tag="complex-method"),
    }

    class TestsFlextCoreEnforcementSources:
        """Behavior contract for surviving EnforcementSource variants."""

        # --- EnforcementSourceKind enum contract ---

        @staticmethod
        @pytest.mark.parametrize(
            ("member", "value"),
            [
                (c.EnforcementSourceKind.FLEXT_INFRA_RULE, "flext_infra_rule"),
                (c.EnforcementSourceKind.RUNTIME_WARNING, "runtime_warning"),
                (c.EnforcementSourceKind.BEARTYPE, "beartype"),
                (c.EnforcementSourceKind.CODE_SMELL, "code_smell"),
            ],
        )
        def test_source_kind_member_exposes_expected_value(
            member: c.EnforcementSourceKind,
            value: str,
        ) -> None:
            assert member.value == value

        @staticmethod
        def test_source_kind_has_exactly_the_surviving_members() -> None:
            assert {kind.value for kind in c.EnforcementSourceKind} == {
                "flext_infra_rule",
                "runtime_warning",
                "beartype",
                "code_smell",
            }

        @staticmethod
        def test_source_kind_dropped_the_minimal_ast_variant() -> None:
            assert "minimal_ast" not in {kind.value for kind in c.EnforcementSourceKind}

        # --- discriminator literals across every source model ---

        @staticmethod
        @pytest.mark.parametrize(
            ("expected_kind", "source"),
            list(_SOURCE_CASES.items()),
        )
        def test_source_model_exposes_matching_discriminator_literal(
            expected_kind: str,
            source: m.BaseModel,
        ) -> None:
            assert source.model_dump()["kind"] == expected_kind

        @staticmethod
        def test_every_source_kind_enum_value_has_a_source_model() -> None:
            model_kinds = {
                source.model_dump()["kind"]
                for source in TestsFlextModelsTestEnforcementSources._SOURCE_CASES.values()
            }
            assert model_kinds == {kind.value for kind in c.EnforcementSourceKind}

        # --- field-level public contract ---

        @staticmethod
        def test_infra_rule_source_keeps_declared_rule_ids() -> None:
            source = m.EnforcementInfraRuleSource(rule_ids=("ban-cast", "ban-any"))
            assert tuple(source.rule_ids) == ("ban-cast", "ban-any")

        @staticmethod
        def test_beartype_source_carries_runtime_tag() -> None:
            source = m.EnforcementBeartypeSource(tag="no_module_compat_alias")
            assert source.kind == "beartype"
            assert source.tag == "no_module_compat_alias"

        # --- validation error paths ---

        @staticmethod
        def test_infra_rule_source_rejects_empty_rule_ids() -> None:
            with pytest.raises(c.ValidationError):
                m.EnforcementInfraRuleSource(rule_ids=())

        @staticmethod
        def test_beartype_source_rejects_empty_tag() -> None:
            with pytest.raises(c.ValidationError):
                m.EnforcementBeartypeSource(tag="")

        @staticmethod
        @pytest.mark.parametrize(
            "factory",
            [
                m.EnforcementInfraRuleSource,
                m.EnforcementRuntimeWarningSource,
                m.EnforcementBeartypeSource,
                m.EnforcementCodeSmellSource,
            ],
        )
        def test_source_model_rejects_missing_required_field(
            factory: type[m.BaseModel],
        ) -> None:
            with pytest.raises(c.ValidationError):
                factory.model_validate({})

        @staticmethod
        def test_fix_action_rejects_kind_outside_literal_set() -> None:
            with pytest.raises(c.ValidationError):
                m.EnforcementFixAction.model_validate({
                    "kind": "not_a_fixer",
                    "target": "x",
                })

        @staticmethod
        def test_fix_action_defaults_safe_true_and_empty_params() -> None:
            action = m.EnforcementFixAction(kind="manual", target="remove_bypass")
            assert action.kind == "manual"
            assert action.target == "remove_bypass"
            assert action.safe is True
            assert dict(action.params) == {}

        # --- model_dump round-trip (public serialization contract) ---

        @staticmethod
        @pytest.mark.parametrize(
            ("expected_kind", "source"),
            list(_SOURCE_CASES.items()),
        )
        def test_source_model_dump_round_trips(
            expected_kind: str,
            source: m.BaseModel,
        ) -> None:
            dumped = source.model_dump()
            assert dumped["kind"] == expected_kind
            rebuilt = type(source).model_validate(dumped)
            assert rebuilt == source

        # --- discriminated-union dispatch through EnforcementRuleSpec ---

        @staticmethod
        @pytest.mark.parametrize(
            ("kind", "source_payload", "expected_type"),
            [
                (
                    "beartype",
                    {"kind": "beartype", "tag": "no_module_compat_alias"},
                    m.EnforcementBeartypeSource,
                ),
                (
                    "flext_infra_rule",
                    {"kind": "flext_infra_rule", "rule_ids": ["ban-cast"]},
                    m.EnforcementInfraRuleSource,
                ),
                (
                    "code_smell",
                    {"kind": "code_smell", "smell_tag": "complex-method"},
                    m.EnforcementCodeSmellSource,
                ),
            ],
        )
        def test_rule_spec_dispatches_source_by_discriminator(
            kind: str,
            source_payload: t.JsonMapping,
            expected_type: type[m.BaseModel],
        ) -> None:
            spec = m.EnforcementRuleSpec.model_validate({
                "id": "ENFORCE-001",
                "description": "d",
                "severity": "HIGH",
                "source": source_payload,
            })
            assert isinstance(spec.source, expected_type)
            assert spec.source.kind == kind

        @staticmethod
        @pytest.mark.parametrize(
            "retired_kind",
            ["minimal_ast", "ruff", "skill_pointer"],
        )
        def test_rule_spec_rejects_retired_source_discriminator(
            retired_kind: str,
        ) -> None:
            with pytest.raises(c.ValidationError):
                m.EnforcementRuleSpec.model_validate({
                    "id": "ENFORCE-001",
                    "description": "d",
                    "severity": "HIGH",
                    "source": {"kind": retired_kind},
                })
