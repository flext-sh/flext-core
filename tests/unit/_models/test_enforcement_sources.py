"""Behavior contract for the typed enforcement source variants.

Exercises the PUBLIC surface of ``EnforcementSourceKind`` and every
``FlextModelsEnforcementSources`` discriminator model: default ``kind``
literals, required-field validation, model_dump round-trips, and the
``EnforcementRuleSpec`` discriminated-union dispatch that consumes them.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from tests import c
from tests import m
from tests import p, t

# One representative valid instance per surviving source variant, keyed by the
# discriminator literal it must expose on the public ``kind`` field.
_SOURCE_CASES: dict[str, m.BaseModel] = {
    "flext_infra_detector": p.EnforcementInfraDetectorSource(
        violation_field="loose_symbol",
    ),
    "flext_tests_validator": p.EnforcementTestsValidatorSource(method="check_x"),
    "runtime_warning": p.EnforcementRuntimeWarningSource(category="FlextMroWarning"),
    "beartype": p.EnforcementBeartypeSource(
        predicate_kind=c.EnforcementPredicateKind.MODULE_ALIAS,
    ),
    "ruff": p.EnforcementRuffSource(rule_code="ANN401"),
    "skill_pointer": p.EnforcementSkillPointerSource(skill="pydantic-canonical"),
    "code_smell": p.EnforcementCodeSmellSource(smell_tag="complex-method"),
}


class TestsFlextCoreEnforcementSources:
    """Behavior contract for surviving EnforcementSource variants."""

    # --- EnforcementSourceKind enum contract ---

    @pytest.mark.parametrize(
        ("member", "value"),
        [
            (m.EnforcementSourceKind.FLEXT_INFRA_DETECTOR, "flext_infra_detector"),
            (m.EnforcementSourceKind.FLEXT_TESTS_VALIDATOR, "flext_tests_validator"),
            (m.EnforcementSourceKind.RUNTIME_WARNING, "runtime_warning"),
            (m.EnforcementSourceKind.BEARTYPE, "beartype"),
            (m.EnforcementSourceKind.CODE_SMELL, "code_smell"),
            (m.EnforcementSourceKind.RUFF, "ruff"),
            (m.EnforcementSourceKind.SKILL_POINTER, "skill_pointer"),
        ],
    )
    def test_source_kind_member_exposes_expected_value(
        self,
        member: p.EnforcementSourceKind,
        value: str,
    ) -> None:
        assert member.value == value

    def test_source_kind_has_exactly_the_surviving_members(self) -> None:
        assert {kind.value for kind in m.EnforcementSourceKind} == {
            "flext_infra_detector",
            "flext_tests_validator",
            "runtime_warning",
            "beartype",
            "code_smell",
            "ruff",
            "skill_pointer",
        }

    def test_source_kind_dropped_the_minimal_ast_variant(self) -> None:
        assert "minimal_ast" not in {kind.value for kind in m.EnforcementSourceKind}

    # --- discriminator literals across every source model ---

    @pytest.mark.parametrize(
        ("expected_kind", "source"),
        list(_SOURCE_CASES.items()),
    )
    def test_source_model_exposes_matching_discriminator_literal(
        self,
        expected_kind: str,
        source: p.BaseModel,
    ) -> None:
        assert source.model_dump()["kind"] == expected_kind

    def test_every_source_kind_enum_value_has_a_source_model(self) -> None:
        model_kinds = {source.model_dump()["kind"] for source in _SOURCE_CASES.values()}
        assert model_kinds == {kind.value for kind in m.EnforcementSourceKind}

    # --- field-level public contract ---

    def test_infra_detector_defaults_match_missing_to_false(self) -> None:
        source = m.EnforcementInfraDetectorSource(violation_field="loose_symbol")
        assert source.violation_field == "loose_symbol"
        assert source.match_missing is False

    def test_tests_validator_defaults_rule_ids_to_empty(self) -> None:
        source = m.EnforcementTestsValidatorSource(method="check_x")
        assert source.method == "check_x"
        assert tuple(source.rule_ids) == ()

    def test_skill_pointer_defaults_anchor_to_empty_string(self) -> None:
        source = m.EnforcementSkillPointerSource(skill="pydantic-canonical")
        assert source.skill == "pydantic-canonical"
        assert source.anchor == ""

    def test_beartype_source_carries_predicate_kind_enum(self) -> None:
        source = m.EnforcementBeartypeSource(
            predicate_kind=c.EnforcementPredicateKind.WRAPPER,
        )
        assert source.kind == "beartype"
        assert source.predicate_kind is c.EnforcementPredicateKind.WRAPPER

    # --- validation error paths ---

    def test_beartype_source_rejects_unknown_predicate_kind(self) -> None:
        with pytest.raises(c.ValidationError):
            m.EnforcementBeartypeSource.model_validate(
                {"predicate_kind": "not_a_kind"},
            )

    @pytest.mark.parametrize(
        ("factory", "payload"),
        [
            (m.EnforcementBeartypeSource, {}),
            (m.EnforcementRuffSource, {}),
            (m.EnforcementInfraDetectorSource, {}),
            (m.EnforcementTestsValidatorSource, {}),
            (m.EnforcementRuntimeWarningSource, {}),
            (m.EnforcementSkillPointerSource, {}),
            (m.EnforcementCodeSmellSource, {}),
        ],
    )
    def test_source_model_rejects_missing_required_field(
        self,
        factory: type[p.BaseModel],
        payload: t.JsonMapping,
    ) -> None:
        with pytest.raises(c.ValidationError):
            factory.model_validate(payload)

    def test_fix_action_rejects_kind_outside_literal_set(self) -> None:
        with pytest.raises(c.ValidationError):
            m.EnforcementFixAction.model_validate(
                {"kind": "not_a_fixer", "target": "x"},
            )

    def test_fix_action_defaults_safe_true_and_empty_params(self) -> None:
        action = m.EnforcementFixAction(kind="gate", target="loose-symbol-gate")
        assert action.kind == "gate"
        assert action.target == "loose-symbol-gate"
        assert action.safe is True
        assert dict(action.params) == {}

    # --- model_dump round-trip (public serialization contract) ---

    @pytest.mark.parametrize(
        ("expected_kind", "source"),
        list(_SOURCE_CASES.items()),
    )
    def test_source_model_dump_round_trips(
        self,
        expected_kind: str,
        source: p.BaseModel,
    ) -> None:
        dumped = source.model_dump()
        assert dumped["kind"] == expected_kind
        rebuilt = type(source).model_validate(dumped)
        assert rebuilt == source

    # --- discriminated-union dispatch through EnforcementRuleSpec ---

    @pytest.mark.parametrize(
        ("kind", "source_payload", "expected_type"),
        [
            (
                "beartype",
                {"kind": "beartype", "predicate_kind": "module_alias"},
                m.EnforcementBeartypeSource,
            ),
            (
                "ruff",
                {"kind": "ruff", "rule_code": "ANN401"},
                m.EnforcementRuffSource,
            ),
            (
                "flext_infra_detector",
                {"kind": "flext_infra_detector", "violation_field": "loose_symbol"},
                m.EnforcementInfraDetectorSource,
            ),
        ],
    )
    def test_rule_spec_dispatches_source_by_discriminator(
        self,
        kind: str,
        source_payload: t.JsonMapping,
        expected_type: type[p.BaseModel],
    ) -> None:
        spec = m.EnforcementRuleSpec.model_validate(
            {
                "id": "ENFORCE-001",
                "description": "d",
                "severity": "HIGH",
                "source": source_payload,
            },
        )
        assert isinstance(spec.source, expected_type)
        assert spec.source.kind == kind

    def test_rule_spec_rejects_unknown_source_discriminator(self) -> None:
        with pytest.raises(c.ValidationError):
            m.EnforcementRuleSpec.model_validate(
                {
                    "id": "ENFORCE-001",
                    "description": "d",
                    "severity": "HIGH",
                    "source": {"kind": "minimal_ast"},
                },
            )
