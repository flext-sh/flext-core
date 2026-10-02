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

from tests.constants import c
from tests.models import m
from tests.typings import t

# One representative valid instance per source variant, keyed by the
# discriminator literal it must expose on the public ``kind`` field.
_SOURCE_CASES: dict[str, m.BaseModel] = {
    "flext_infra_rule": m.EnforcementInfraRuleSource(rule_ids=("rule-a",)),
    "runtime_warning": m.EnforcementRuntimeWarningSource(category="FlextMroWarning"),
    "beartype": m.EnforcementBeartypeSource(tag="tag_a"),
    "code_smell": m.EnforcementCodeSmellSource(smell_tag="complex-method"),
}


class TestsFlextCoreEnforcementSources:
    """Behavior contract for the EnforcementSource variants."""

    @staticmethod
    def test_every_source_kind_enum_value_has_a_source_model() -> None:
        model_kinds = {source.model_dump()["kind"] for source in _SOURCE_CASES.values()}
        assert model_kinds == {kind.value for kind in c.EnforcementSourceKind}

    @pytest.mark.parametrize(("expected_kind", "source"), list(_SOURCE_CASES.items()))
    def test_source_model_dump_round_trips(
        self,
        expected_kind: str,
        source: m.BaseModel,
    ) -> None:
        dumped = source.model_dump()
        assert dumped["kind"] == expected_kind
        assert type(source).model_validate(dumped) == source

    @staticmethod
    def test_infra_rule_source_requires_at_least_one_rule_id() -> None:
        with pytest.raises(c.ValidationError):
            m.EnforcementInfraRuleSource(rule_ids=())

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
        self,
        factory: type[m.BaseModel],
    ) -> None:
        with pytest.raises(c.ValidationError):
            factory.model_validate({})

    @pytest.mark.parametrize(
        ("source_payload", "expected_type"),
        [
            ({"kind": "beartype", "tag": "tag_a"}, m.EnforcementBeartypeSource),
            (
                {"kind": "flext_infra_rule", "rule_ids": ["rule-a"]},
                m.EnforcementInfraRuleSource,
            ),
            (
                {"kind": "code_smell", "smell_tag": "smell_a"},
                m.EnforcementCodeSmellSource,
            ),
        ],
    )
    def test_rule_spec_dispatches_source_by_discriminator(
        self,
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

    @pytest.mark.parametrize(
        "retired_kind",
        [
            "minimal_ast",
            "flext_infra_detector",
            "flext_tests_validator",
            "ruff",
            "skill_pointer",
        ],
    )
    def test_rule_spec_rejects_retired_source_discriminators(
        self,
        retired_kind: str,
    ) -> None:
        with pytest.raises(c.ValidationError):
            m.EnforcementRuleSpec.model_validate({
                "id": "ENFORCE-001",
                "description": "d",
                "severity": "HIGH",
                "source": {"kind": retired_kind},
            })
