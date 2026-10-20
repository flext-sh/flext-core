"""Behavior contract for u.build_canonical_catalog() — cross-layer enforcement SSOT.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib

import pytest

from tests import c, m, tm, u


class TestsFlextEnforcementCatalog:
    """Behavior contract for ``u.build_canonical_catalog()``.

    Covers shape, coverage, and construction.
    """

    @staticmethod
    def test_catalog_contains_at_least_one_rule() -> None:
        """Test catalog contains at least one rule."""
        tm.that(len(u.build_canonical_catalog().rules) > 0, truth=True)

    @staticmethod
    def test_catalog_is_frozen_and_rejects_mutation() -> None:
        """Test catalog is frozen and rejects mutation."""
        field_name = "rules"
        with pytest.raises(c.ValidationError):
            setattr(u.build_canonical_catalog(), field_name, ())

    @staticmethod
    def test_catalog_version_is_monotonic_positive_integer() -> None:
        """Test catalog version is monotonic positive integer."""
        tm.that(u.build_canonical_catalog().version >= 1, truth=True)

    @staticmethod
    def test_all_rule_ids_are_unique() -> None:
        """Test all rule ids are unique."""
        ids = [rule.id for rule in u.build_canonical_catalog().rules]
        tm.that(len(ids), eq=len(set(ids)))

    @staticmethod
    def test_all_rule_ids_match_enforce_nnn_format() -> None:
        """Test all rule ids match enforce nnn format."""
        for rule in u.build_canonical_catalog().rules:
            tm.that(c.PATTERN_ENFORCE_RULE_ID_RE.fullmatch(rule.id), truth=True)

    @staticmethod
    def test_by_id_returns_rule_when_present_and_none_when_missing() -> None:
        """Test by id returns rule when present and none when missing."""
        first = u.build_canonical_catalog().rules[0]
        tm.that(u.build_canonical_catalog().by_id(first.id) is first, truth=True)
        tm.that(u.build_canonical_catalog().by_id("ENFORCE-999"), none=True)

    @staticmethod
    def test_enabled_rules_are_a_subset_of_all_rules() -> None:
        """Test enabled rules are a subset of all rules."""
        enabled = u.build_canonical_catalog().enabled_rules()
        tm.that(all(rule.enabled for rule in enabled), truth=True)
        tm.that(len(enabled) <= len(u.build_canonical_catalog().rules), truth=True)

    @staticmethod
    def test_by_kind_returns_only_matching_source_kind() -> None:
        """Test by kind returns only matching source kind."""
        infra = u.build_canonical_catalog().by_kind(
            c.EnforcementSourceKind.FLEXT_INFRA_RULE,
        )
        tm.that(infra, truth=True)
        tm.that(
            all(
                rule.source.kind == c.EnforcementSourceKind.FLEXT_INFRA_RULE.value
                for rule in infra
            ),
            truth=True,
        )

    @staticmethod
    def test_catalog_covers_every_declared_source_kind() -> None:
        """Test catalog covers every declared source kind."""
        present = {rule.source.kind for rule in u.build_canonical_catalog().rules}
        expected = {member.value for member in c.EnforcementSourceKind}
        tm.that(expected <= present, truth=True)

    @staticmethod
    def test_runtime_warning_categories_resolve_to_warning_classes() -> None:
        """Test runtime warning categories resolve to warning classes."""
        runtime = u.build_canonical_catalog().by_kind(
            c.EnforcementSourceKind.RUNTIME_WARNING,
        )
        tm.that(runtime, truth=True)
        for rule in runtime:
            source = rule.source
            tm.that(source, is_=m.EnforcementRuntimeWarningSource)
            if isinstance(source, m.EnforcementRuntimeWarningSource):
                module_name, _, class_name = source.category.rpartition(".")
                category = getattr(importlib.import_module(module_name), class_name)
                tm.that(category, where=lambda value: issubclass(value, Warning))

    @staticmethod
    def test_fix_action_rules_match_declared_constants() -> None:
        """Test fix action rules match declared constants."""
        catalog = u.build_canonical_catalog()
        rule_ids = {rule.id for rule in catalog.rules}

        tm.that(set(c.ENFORCEMENT_FIX_ACTIONS) <= rule_ids, truth=True)

    @staticmethod
    def test_declared_fix_actions_materialize_in_catalog() -> None:
        """Test declared fix actions materialize in catalog."""
        catalog = u.build_canonical_catalog()
        materialized = {
            rule.id: rule.fix_action
            for rule in catalog.rules
            if rule.fix_action is not None
        }

        tm.that(set(materialized), eq=set(c.ENFORCEMENT_FIX_ACTIONS))
        for rule_id, fix_action in materialized.items():
            declared = c.ENFORCEMENT_FIX_ACTIONS[rule_id]
            tm.that(fix_action.kind, eq=declared["kind"])
            tm.that(fix_action.target, eq=declared["target"])
            tm.that(fix_action.params, eq=declared["params"])
            tm.that(fix_action.safe is declared["safe"], truth=True)

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
        tm.that(
            infra_rule.source.kind, eq=c.EnforcementSourceKind.FLEXT_INFRA_RULE.value
        )

        beartype_rule = m.EnforcementRuleSpec(
            id="ENFORCE-902",
            description="x",
            severity=c.EnforcementRuleSeverity.LOW,
            source=m.EnforcementBeartypeSource(tag="no_module_compat_alias"),
        )
        tm.that(beartype_rule.source.kind, eq=c.EnforcementSourceKind.BEARTYPE.value)

    @staticmethod
    def test_every_rule_severity_is_an_enum_member() -> None:
        """Test every rule severity is an enum member."""
        for rule in u.build_canonical_catalog().rules:
            tm.that(isinstance(rule.severity, c.EnforcementRuleSeverity), truth=True)
