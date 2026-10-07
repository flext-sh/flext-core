"""Behavior contract for u.build_canonical_catalog() — cross-layer enforcement SSOT.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib

import pytest

from tests.constants import c
from tests.models import m
from tests.utilities import u


class TestsFlextEnforcementCatalog:
    """Behavior contract for ``u.build_canonical_catalog()``.

    Covers shape, coverage, and construction.
    """

    @staticmethod
    def test_catalog_contains_at_least_one_rule() -> None:
        """Test catalog contains at least one rule."""
        assert len(u.build_canonical_catalog().rules) > 0

    @staticmethod
    def test_catalog_is_frozen_and_rejects_mutation() -> None:
        """Test catalog is frozen and rejects mutation."""
        with pytest.raises(c.ValidationError):
            u.build_canonical_catalog().rules = ()

    @staticmethod
    def test_catalog_version_is_monotonic_positive_integer() -> None:
        """Test catalog version is monotonic positive integer."""
        assert u.build_canonical_catalog().version >= 1

    @staticmethod
    def test_all_rule_ids_are_unique() -> None:
        """Test all rule ids are unique."""
        ids = [rule.id for rule in u.build_canonical_catalog().rules]
        assert len(ids) == len(set(ids))

    @staticmethod
    def test_all_rule_ids_match_enforce_nnn_format() -> None:
        """Test all rule ids match enforce nnn format."""
        for rule in u.build_canonical_catalog().rules:
            assert c.PATTERN_ENFORCE_RULE_ID_RE.fullmatch(rule.id)

    @staticmethod
    def test_by_id_returns_rule_when_present_and_none_when_missing() -> None:
        """Test by id returns rule when present and none when missing."""
        first = u.build_canonical_catalog().rules[0]
        assert u.build_canonical_catalog().by_id(first.id) is first
        assert u.build_canonical_catalog().by_id("ENFORCE-999") is None

    @staticmethod
    def test_enabled_rules_are_a_subset_of_all_rules() -> None:
        """Test enabled rules are a subset of all rules."""
        enabled = u.build_canonical_catalog().enabled_rules()
        assert all(rule.enabled for rule in enabled)
        assert len(enabled) <= len(u.build_canonical_catalog().rules)

    @staticmethod
    def test_by_kind_returns_only_matching_source_kind() -> None:
        """Test by kind returns only matching source kind."""
        infra = u.build_canonical_catalog().by_kind(
            c.EnforcementSourceKind.FLEXT_INFRA_RULE,
        )
        assert infra
        assert all(
            rule.source.kind == c.EnforcementSourceKind.FLEXT_INFRA_RULE.value
            for rule in infra
        )

    @staticmethod
    def test_catalog_covers_every_declared_source_kind() -> None:
        """Test catalog covers every declared source kind."""
        present = {rule.source.kind for rule in u.build_canonical_catalog().rules}
        expected = {member.value for member in c.EnforcementSourceKind}
        assert expected <= present

    @staticmethod
    def test_runtime_warning_categories_resolve_to_warning_classes() -> None:
        """Test runtime warning categories resolve to warning classes."""
        runtime = u.build_canonical_catalog().by_kind(
            c.EnforcementSourceKind.RUNTIME_WARNING,
        )
        assert runtime
        for rule in runtime:
            assert isinstance(rule.source, m.EnforcementRuntimeWarningSource)
            module_name, _, class_name = rule.source.category.rpartition(".")
            category = getattr(importlib.import_module(module_name), class_name)
            assert issubclass(category, Warning)

    @staticmethod
    def test_fix_action_rules_match_declared_constants() -> None:
        """Test fix action rules match declared constants."""
        catalog = u.build_canonical_catalog()
        rule_ids = {rule.id for rule in catalog.rules}

        assert set(c.ENFORCEMENT_FIX_ACTIONS) <= rule_ids

    @staticmethod
    def test_declared_fix_actions_materialize_in_catalog() -> None:
        """Test declared fix actions materialize in catalog."""
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

    @staticmethod
    def test_rule_spec_construction_rejects_invalid_id_format() -> None:
        """Test rule spec construction rejects invalid id format."""
        with pytest.raises(c.ValidationError):
            m.EnforcementRuleSpec(
                id="BAD-999",
                description="bad",
                severity=c.EnforcementRuleSeverity.HIGH,
                source=m.EnforcementCodeSmellSource(smell_tag="complex-method"),
            )

    @staticmethod
    def test_catalog_construction_rejects_duplicate_rule_ids() -> None:
        """Test catalog construction rejects duplicate rule ids."""
        rule = m.EnforcementRuleSpec(
            id="ENFORCE-900",
            description="x",
            severity=c.EnforcementRuleSeverity.LOW,
            source=m.EnforcementCodeSmellSource(smell_tag="complex-method"),
        )
        with pytest.raises(c.ValidationError):
            m.EnforcementCatalog(rules=(rule, rule))

    @staticmethod
    def test_discriminated_union_routes_source_kind_by_payload() -> None:
        """Test discriminated union routes source kind by payload."""
        infra_rule = m.EnforcementRuleSpec(
            id="ENFORCE-901",
            description="x",
            severity=c.EnforcementRuleSeverity.HIGH,
            source=m.EnforcementInfraRuleSource(rule_ids=("ban-cast",)),
        )
        assert infra_rule.source.kind == c.EnforcementSourceKind.FLEXT_INFRA_RULE.value

        beartype_rule = m.EnforcementRuleSpec(
            id="ENFORCE-902",
            description="x",
            severity=c.EnforcementRuleSeverity.LOW,
            source=m.EnforcementBeartypeSource(tag="no_module_compat_alias"),
        )
        assert beartype_rule.source.kind == c.EnforcementSourceKind.BEARTYPE.value

    @staticmethod
    def test_every_rule_severity_is_an_enum_member() -> None:
        """Test every rule severity is an enum member."""
        for rule in u.build_canonical_catalog().rules:
            assert isinstance(rule.severity, c.EnforcementRuleSeverity)
