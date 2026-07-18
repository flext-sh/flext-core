"""Behavior contract for u.build_canonical_catalog() — cross-layer enforcement SSOT."""

from __future__ import annotations

import pytest

from tests import c
from tests import m
from tests import u


class TestsFlextEnforcementCatalog:
    """Behavior contract for u.build_canonical_catalog() shape, coverage, and construction."""

    def test_catalog_contains_at_least_one_rule(self) -> None:
        assert len(u.build_canonical_catalog().rules) > 0

    def test_catalog_is_frozen_and_rejects_mutation(self) -> None:
        with pytest.raises(c.ValidationError):
            u.build_canonical_catalog().rules = ()

    def test_catalog_version_is_monotonic_positive_integer(self) -> None:
        assert u.build_canonical_catalog().version >= 1

    def test_all_rule_ids_are_unique(self) -> None:
        ids = [rule.id for rule in u.build_canonical_catalog().rules]
        assert len(ids) == len(set(ids))

    def test_all_rule_ids_match_enforce_nnn_format(self) -> None:
        for rule in u.build_canonical_catalog().rules:
            assert c.PATTERN_ENFORCE_RULE_ID_RE.fullmatch(rule.id)

    def test_by_id_returns_rule_when_present_and_none_when_missing(self) -> None:
        first = u.build_canonical_catalog().rules[0]
        assert u.build_canonical_catalog().by_id(first.id) is first
        assert u.build_canonical_catalog().by_id("ENFORCE-999") is None

    def test_enabled_rules_are_a_subset_of_all_rules(self) -> None:
        enabled = u.build_canonical_catalog().enabled_rules()
        assert all(rule.enabled for rule in enabled)
        assert len(enabled) <= len(u.build_canonical_catalog().rules)

    def test_by_kind_returns_only_matching_source_kind(self) -> None:
        infra = u.build_canonical_catalog().by_kind(
            m.EnforcementSourceKind.FLEXT_INFRA_DETECTOR
        )
        assert all(
            rule.source.kind == m.EnforcementSourceKind.FLEXT_INFRA_DETECTOR.value
            for rule in infra
        )

    def test_catalog_covers_every_declared_source_kind(self) -> None:
        present = {rule.source.kind for rule in u.build_canonical_catalog().rules}
        expected = {member.value for member in m.EnforcementSourceKind}
        assert expected <= present

    def test_infra_detector_rules_match_declared_infra_rows(self) -> None:
        declared_fields = {row[2] for row in c.INFRA_DETECTOR_ROWS}
        infra = u.build_canonical_catalog().by_kind(
            m.EnforcementSourceKind.FLEXT_INFRA_DETECTOR
        )
        actual_fields: set[str] = set()
        for rule in infra:
            assert isinstance(rule.source, m.EnforcementInfraDetectorSource)
            actual_fields.add(rule.source.violation_field)

        assert len(infra) == len(c.INFRA_DETECTOR_ROWS)
        assert actual_fields == declared_fields

    def test_auto_fix_infra_detector_violation_fields_are_unique(self) -> None:
        """Each auto-fixable infra detector field maps to one catalog rule."""
        fields: list[str] = []
        for rule in u.build_canonical_catalog().by_kind(
            m.EnforcementSourceKind.FLEXT_INFRA_DETECTOR
        ):
            assert isinstance(rule.source, m.EnforcementInfraDetectorSource)
            if rule.fix_action is not None:
                fields.append(rule.source.violation_field)

        assert len(fields) == len(set(fields))

    def test_tests_validator_rules_cover_all_seven_public_dispatch_methods(
        self,
    ) -> None:
        validators = u.build_canonical_catalog().by_kind(
            m.EnforcementSourceKind.FLEXT_TESTS_VALIDATOR
        )
        assert len(validators) == 7
        methods: set[str] = set()
        for rule in validators:
            assert isinstance(rule.source, m.EnforcementTestsValidatorSource)
            methods.add(rule.source.method)
        assert methods == {
            "imports",
            "types",
            "bypass",
            "layer",
            "tests",
            "validate_config",
            "markdown",
        }

    def test_runtime_warning_category_references_flext_mro_violation(self) -> None:
        runtime = u.build_canonical_catalog().by_kind(
            m.EnforcementSourceKind.RUNTIME_WARNING
        )
        categories: set[str] = set()
        for rule in runtime:
            assert isinstance(rule.source, m.EnforcementRuntimeWarningSource)
            categories.add(rule.source.category)
        assert "flext_core._constants.enforcement.FlextMroViolation" in categories

    def test_fix_action_rules_match_declared_constants(self) -> None:
        catalog = u.build_canonical_catalog()
        rule_ids = {rule.id for rule in catalog.rules}

        assert set(c.ENFORCEMENT_FIX_ACTIONS) <= rule_ids

    def test_declared_fix_actions_materialize_in_catalog(self) -> None:
        catalog = u.build_canonical_catalog()
        materialized = {
            rule.id: rule.fix_action
            for rule in catalog.rules
            if rule.fix_action is not None
        }

        assert set(materialized) == set(c.ENFORCEMENT_FIX_ACTIONS)
        for rule_id, fix_action in materialized.items():
            declared = c.ENFORCEMENT_FIX_ACTIONS[rule_id]
            assert fix_action.kind == declared["kind"]
            assert fix_action.target == declared["target"]
            assert fix_action.params == declared["params"]
            assert fix_action.safe is declared["safe"]

    def test_compatibility_alias_import_uses_rope_rewriter(self) -> None:
        rule = u.build_canonical_catalog().by_id("ENFORCE-064")

        assert rule is not None
        assert rule.fix_action is not None
        assert rule.fix_action.kind == "rope"
        assert rule.fix_action.target == "rewrite_compatibility_alias"

    def test_rule_spec_construction_rejects_invalid_id_format(self) -> None:
        with pytest.raises(c.ValidationError):
            m.EnforcementRuleSpec(
                id="BAD-999",
                description="bad",
                severity=m.EnforcementRuleSeverity.HIGH,
                source=m.EnforcementRuffSource(rule_code="ANN401"),
            )

    def test_catalog_construction_rejects_duplicate_rule_ids(self) -> None:
        rule = m.EnforcementRuleSpec(
            id="ENFORCE-900",
            description="x",
            severity=m.EnforcementRuleSeverity.LOW,
            source=m.EnforcementRuffSource(rule_code="ANN401"),
        )
        with pytest.raises(c.ValidationError):
            m.EnforcementCatalog(rules=(rule, rule))

    def test_discriminated_union_routes_source_kind_by_payload(self) -> None:
        infra_rule = m.EnforcementRuleSpec(
            id="ENFORCE-901",
            description="x",
            severity=m.EnforcementRuleSeverity.HIGH,
            source=m.EnforcementInfraDetectorSource(violation_field="loose_objects"),
        )
        assert (
            infra_rule.source.kind == m.EnforcementSourceKind.FLEXT_INFRA_DETECTOR.value
        )

        ruff_rule = m.EnforcementRuleSpec(
            id="ENFORCE-902",
            description="x",
            severity=m.EnforcementRuleSeverity.LOW,
            source=m.EnforcementRuffSource(rule_code="PGH003"),
        )
        assert ruff_rule.source.kind == m.EnforcementSourceKind.RUFF.value

    def test_every_rule_severity_is_an_enum_member(self) -> None:
        for rule in u.build_canonical_catalog().rules:
            assert isinstance(rule.severity, m.EnforcementRuleSeverity)
