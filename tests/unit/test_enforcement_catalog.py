"""Unit tests for the cross-layer enforcement catalog.

Covers the SSOT in ``flext_core.c.ENFORCEMENT_CATALOG`` and the model shapes
it relies on. These tests run with ``flext_core`` alone — they do NOT depend
on ``flext_infra`` or ``flext_tests``.
"""

from __future__ import annotations

import re

import pytest

from flext_core import (
    FlextConstantsEnforcement,
    FlextModelsEnforcement,
)

_me = FlextModelsEnforcement
_cat = FlextConstantsEnforcement.ENFORCEMENT_CATALOG


class TestEnforcementCatalogShape:
    """Catalog is a well-formed frozen Pydantic model."""

    def test_catalog_has_rules(self) -> None:
        assert len(_cat.rules) > 0, "catalog must contain at least one rule"

    def test_catalog_is_frozen(self) -> None:
        with pytest.raises(Exception):
            _cat.rules = ()

    def test_catalog_version_is_positive(self) -> None:
        assert _cat.version >= 1

    def test_all_ids_are_unique(self) -> None:
        ids = [rule.id for rule in _cat.rules]
        assert len(ids) == len(set(ids)), "duplicate rule IDs"

    def test_all_ids_match_regex(self) -> None:
        pattern = re.compile(r"^ENFORCE-\d{3}$")
        for rule in _cat.rules:
            assert pattern.match(rule.id), f"bad id: {rule.id!r}"

    def test_by_id_roundtrip(self) -> None:
        first = _cat.rules[0]
        assert _cat.by_id(first.id) is first
        assert _cat.by_id("ENFORCE-999") is None

    def test_enabled_rules_subset(self) -> None:
        enabled = _cat.enabled_rules()
        assert all(r.enabled for r in enabled)
        assert len(enabled) <= len(_cat.rules)

    def test_by_kind_filter(self) -> None:
        infra = _cat.by_kind(_me.EnforcementSourceKind.FLEXT_INFRA_DETECTOR)
        assert all(
            r.source.kind == _me.EnforcementSourceKind.FLEXT_INFRA_DETECTOR.value
            for r in infra
        )


class TestEnforcementCatalogCoverage:
    """Catalog covers every expected origin."""

    def test_has_all_source_kinds(self) -> None:
        present = {rule.source.kind for rule in _cat.rules}
        expected = {member.value for member in _me.EnforcementSourceKind}
        assert expected <= present, f"missing kinds: {expected - present}"

    def test_exactly_14_infra_detector_rules(self) -> None:
        """ProjectEnforcementReport has 14 violation fields; catalog matches."""
        infra = _cat.by_kind(_me.EnforcementSourceKind.FLEXT_INFRA_DETECTOR)
        assert len(infra) == 14

    def test_exactly_7_tests_validator_rules(self) -> None:
        """FlextTestsValidator exposes 7 public dispatch methods."""
        validators = _cat.by_kind(_me.EnforcementSourceKind.FLEXT_TESTS_VALIDATOR)
        assert len(validators) == 7
        methods = {getattr(r.source, "method", "") for r in validators}
        assert methods == {
            "imports",
            "types",
            "bypass",
            "layer",
            "tests",
            "validate_config",
            "markdown",
        }

    def test_runtime_warning_points_at_flext_mro_violation(self) -> None:
        runtime = _cat.by_kind(_me.EnforcementSourceKind.RUNTIME_WARNING)
        categories = {getattr(r.source, "category", "") for r in runtime}
        assert "flext_core._constants.enforcement.FlextMroViolation" in categories


class TestEnforcementRuleSpecConstruction:
    """Direct model construction preserves the catalog's invariants."""

    def test_bad_id_rejected(self) -> None:
        with pytest.raises(Exception):
            _me.EnforcementRuleSpec(
                id="BAD-999",
                description="bad",
                severity=_me.EnforcementRuleSeverity.HIGH,
                source=_me.EnforcementRuffSource(rule_code="ANN401"),
            )

    def test_duplicate_ids_rejected_in_catalog(self) -> None:
        rule = _me.EnforcementRuleSpec(
            id="ENFORCE-900",
            description="x",
            severity=_me.EnforcementRuleSeverity.LOW,
            source=_me.EnforcementRuffSource(rule_code="ANN401"),
        )
        with pytest.raises(Exception):
            _me.EnforcementCatalog(rules=(rule, rule))

    def test_discriminator_routes_to_right_source(self) -> None:
        infra_rule = _me.EnforcementRuleSpec(
            id="ENFORCE-901",
            description="x",
            severity=_me.EnforcementRuleSeverity.HIGH,
            source=_me.EnforcementInfraDetectorSource(
                violation_field="loose_objects",
            ),
        )
        assert infra_rule.source.kind == "flext_infra_detector"

        ruff_rule = _me.EnforcementRuleSpec(
            id="ENFORCE-902",
            description="x",
            severity=_me.EnforcementRuleSeverity.LOW,
            source=_me.EnforcementRuffSource(rule_code="PGH003"),
        )
        assert ruff_rule.source.kind == "ruff"

    def test_severity_is_enum_value(self) -> None:
        for rule in _cat.rules:
            assert isinstance(rule.severity, _me.EnforcementRuleSeverity)
