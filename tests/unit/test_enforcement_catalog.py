"""Behavior contract for c.ENFORCEMENT_CATALOG — cross-layer enforcement SSOT."""

from __future__ import annotations

import re

import pytest

from tests import c, m


class TestsFlextCoreEnforcementCatalog:
    """Behavior contract for c.ENFORCEMENT_CATALOG shape, coverage, and construction."""

    def test_catalog_contains_at_least_one_rule(self) -> None:
        assert len(c.ENFORCEMENT_CATALOG.rules) > 0

    def test_catalog_is_frozen_and_rejects_mutation(self) -> None:
        with pytest.raises(c.ValidationError):
            c.ENFORCEMENT_CATALOG.rules = ()

    def test_catalog_version_is_monotonic_positive_integer(self) -> None:
        assert c.ENFORCEMENT_CATALOG.version >= 1

    def test_all_rule_ids_are_unique(self) -> None:
        ids = [rule.id for rule in c.ENFORCEMENT_CATALOG.rules]
        assert len(ids) == len(set(ids))

    def test_all_rule_ids_match_enforce_nnn_format(self) -> None:
        for rule in c.ENFORCEMENT_CATALOG.rules:
            assert re.match(r"^ENFORCE-\d{3}$", rule.id)

    def test_by_id_returns_rule_when_present_and_none_when_missing(self) -> None:
        first = c.ENFORCEMENT_CATALOG.rules[0]
        assert c.ENFORCEMENT_CATALOG.by_id(first.id) is first
        assert c.ENFORCEMENT_CATALOG.by_id("ENFORCE-999") is None

    def test_enabled_rules_are_a_subset_of_all_rules(self) -> None:
        enabled = c.ENFORCEMENT_CATALOG.enabled_rules()
        assert all(rule.enabled for rule in enabled)
        assert len(enabled) <= len(c.ENFORCEMENT_CATALOG.rules)

    def test_by_kind_returns_only_matching_source_kind(self) -> None:
        infra = c.ENFORCEMENT_CATALOG.by_kind(
            m.EnforcementSourceKind.FLEXT_INFRA_DETECTOR,
        )
        assert all(
            rule.source.kind == m.EnforcementSourceKind.FLEXT_INFRA_DETECTOR.value
            for rule in infra
        )

    def test_catalog_covers_every_declared_source_kind(self) -> None:
        present = {rule.source.kind for rule in c.ENFORCEMENT_CATALOG.rules}
        expected = {member.value for member in m.EnforcementSourceKind}
        assert expected <= present

    def test_infra_detector_rules_match_project_enforcement_report_field_count(
        self,
    ) -> None:
        # ProjectEnforcementReport has 14 violation fields; catalog must match.
        infra = c.ENFORCEMENT_CATALOG.by_kind(
            m.EnforcementSourceKind.FLEXT_INFRA_DETECTOR,
        )
        assert len(infra) == 14

    def test_tests_validator_rules_cover_all_seven_public_dispatch_methods(
        self,
    ) -> None:
        validators = c.ENFORCEMENT_CATALOG.by_kind(
            m.EnforcementSourceKind.FLEXT_TESTS_VALIDATOR,
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
        runtime = c.ENFORCEMENT_CATALOG.by_kind(
            m.EnforcementSourceKind.RUNTIME_WARNING,
        )
        categories: set[str] = set()
        for rule in runtime:
            assert isinstance(rule.source, m.EnforcementRuntimeWarningSource)
            categories.add(rule.source.category)
        assert "flext_core._constants.enforcement.FlextMroViolation" in categories

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
        for rule in c.ENFORCEMENT_CATALOG.rules:
            assert isinstance(rule.severity, m.EnforcementRuleSeverity)
