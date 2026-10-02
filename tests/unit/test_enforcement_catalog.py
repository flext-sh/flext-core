"""Behavior contract for u.build_canonical_catalog() — the catalog package data.

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
    """The catalog validated from package data is unique, typed and resolvable."""

    @staticmethod
    def test_catalog_is_loaded_once_and_frozen() -> None:
        """Test catalog is loaded once and frozen."""
        catalog = u.build_canonical_catalog()
        assert catalog is u.build_canonical_catalog()
        assert catalog.rules
        with pytest.raises(c.ValidationError):
            catalog.rules = ()

    @staticmethod
    def test_rule_ids_are_unique_and_match_the_id_grammar() -> None:
        """Test rule ids are unique and match the id grammar."""
        ids = [rule.id for rule in u.build_canonical_catalog().rules]
        assert len(ids) == len(set(ids))
        assert all(c.PATTERN_ENFORCE_RULE_ID_RE.fullmatch(rule_id) for rule_id in ids)

    @staticmethod
    def test_by_id_returns_the_rule_or_none() -> None:
        """Test by id returns the rule or none."""
        catalog = u.build_canonical_catalog()
        first = catalog.rules[0]
        assert catalog.by_id(first.id) is first
        assert catalog.by_id("ENFORCE-999") is None

    @pytest.mark.parametrize("kind", list(c.EnforcementSourceKind))
    def test_every_source_kind_is_present_and_filtered_by_kind(
        self,
        kind: c.EnforcementSourceKind,
    ) -> None:
        """Test every source kind is present and filtered by kind."""
        selected = u.build_canonical_catalog().by_kind(kind)
        assert selected
        assert all(rule.source.kind == kind.value for rule in selected)

    @staticmethod
    def test_beartype_rules_name_a_runtime_tag() -> None:
        """Test beartype rules name a runtime tag."""
        for rule in u.build_canonical_catalog().by_kind(
            c.EnforcementSourceKind.BEARTYPE,
        ):
            assert isinstance(rule.source, m.EnforcementBeartypeSource)
            assert rule.source.tag in c.ENFORCEMENT_TAG_CATEGORY

    @staticmethod
    def test_code_smell_rules_name_a_smell_tag() -> None:
        """Test code smell rules name a smell tag."""
        for rule in u.build_canonical_catalog().by_kind(
            c.EnforcementSourceKind.CODE_SMELL,
        ):
            assert isinstance(rule.source, m.EnforcementCodeSmellSource)
            assert rule.source.smell_tag in c.ENFORCEMENT_SMELL_TAGS

    @staticmethod
    def test_runtime_warning_categories_resolve_to_warning_classes() -> None:
        """Test runtime warning categories resolve to warning classes."""
        for rule in u.build_canonical_catalog().by_kind(
            c.EnforcementSourceKind.RUNTIME_WARNING,
        ):
            assert isinstance(rule.source, m.EnforcementRuntimeWarningSource)
            module_name, _, class_name = rule.source.category.rpartition(".")
            category = getattr(importlib.import_module(module_name), class_name)
            assert issubclass(category, Warning)

    @staticmethod
    def test_rule_spec_rejects_invalid_id_format() -> None:
        """Test rule spec rejects invalid id format."""
        with pytest.raises(c.ValidationError):
            m.EnforcementRuleSpec(
                id="BAD-999",
                description="bad",
                severity=c.EnforcementRuleSeverity.HIGH,
                source=m.EnforcementBeartypeSource(tag="tag_a"),
            )

    @staticmethod
    def test_catalog_rejects_duplicate_rule_ids() -> None:
        """Test catalog rejects duplicate rule ids."""
        rule = m.EnforcementRuleSpec(
            id="ENFORCE-900",
            description="x",
            severity=c.EnforcementRuleSeverity.LOW,
            source=m.EnforcementInfraRuleSource(rule_ids=("rule-a",)),
        )
        with pytest.raises(c.ValidationError):
            m.EnforcementCatalog(rules=(rule, rule))
