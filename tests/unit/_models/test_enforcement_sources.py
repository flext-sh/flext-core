"""Tests for new enforcement source kinds + ViolationReport (Lane A-CH Phase 0 Task 0.1).

Validates the schema additions that close the gap between detection-only rules
(documented via the existing 6 source variants) and rules that need:
- runtime beartype hook capture (EnforcementBeartypeSource),
- inline minimal-AST patterns with source-skip (EnforcementMinimalAstSource),
- typed Rope-backed fix dispatch (EnforcementRopeFixSource),
- typed ast-grep-backed fix dispatch (EnforcementAstGrepFixSource).

All additions are backward compatible: existing 38 catalog rules continue to
validate unchanged. Per AGENT_COORDINATION.md §4.1, A-CH owns these additions.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import c, m


class TestsFlextCoreModelsEnforcementSources:
    """Schema contract for enforcement source kinds + ViolationReport."""

    def test_beartype_source_minimal_fields(self) -> None:
        src = m.EnforcementBeartypeSource(hook="flext_runtime_violation")
        assert src.kind == "beartype"
        assert src.hook == "flext_runtime_violation"

    def test_beartype_source_rejects_extra_fields(self) -> None:
        with pytest.raises(c.ValidationError):
            m.EnforcementBeartypeSource(
                hook="x",
                bogus_field="y",  # type: ignore[call-arg]  # intentional invalid input for test
            )

    def test_minimal_ast_source_default_require_source_true(self) -> None:
        src = m.EnforcementMinimalAstSource(pattern="def $F($$$): ...")
        assert src.kind == "minimal_ast"
        assert src.pattern == "def $F($$$): ..."
        assert src.require_source is True

    def test_minimal_ast_source_can_disable_require_source(self) -> None:
        src = m.EnforcementMinimalAstSource(
            pattern="x = $Y",
            require_source=False,
        )
        assert src.require_source is False

    def test_rope_fix_source_accepts_known_kinds(self) -> None:
        for kind in (
            "rename",
            "move",
            "inline",
            "extract",
            "change_signature",
            "custom",
        ):
            src = m.EnforcementRopeFixSource(kind=kind, handler="h")
            assert src.kind == kind
            assert src.handler == "h"

    def test_rope_fix_source_rejects_unknown_kind(self) -> None:
        with pytest.raises(c.ValidationError):
            m.EnforcementRopeFixSource(
                kind="bogus",  # type: ignore[arg-type]  # intentional invalid input
                handler="x",
            )

    def test_astgrep_fix_source_links_skill_rule(self) -> None:
        src = m.EnforcementAstGrepFixSource(
            skill="flext-import-rules",
            rule_id="rewrite-pydantic-direct",
        )
        assert src.kind == "ast_grep_fix"
        assert src.skill == "flext-import-rules"
        assert src.rule_id == "rewrite-pydantic-direct"

    def test_enforcement_source_kind_includes_new_variants(self) -> None:
        kinds = {member.value for member in m.EnforcementSourceKind}
        assert "beartype" in kinds
        assert "minimal_ast" in kinds

    def test_rule_spec_fix_source_optional_default_none(self) -> None:
        spec = m.EnforcementRuleSpec(
            id="ENFORCE-001",
            description="placeholder for fix_source default check",
            severity=m.EnforcementRuleSeverity.LOW,
            source=m.EnforcementSkillPointerSource(skill="x"),
        )
        assert spec.fix_source is None

    def test_rule_spec_accepts_rope_fix_source(self) -> None:
        spec = m.EnforcementRuleSpec(
            id="ENFORCE-002",
            description="rope fix source attached",
            severity=m.EnforcementRuleSeverity.HIGH,
            source=m.EnforcementInfraDetectorSource(
                violation_field="loose_objects",
            ),
            fix_source=m.EnforcementRopeFixSource(
                kind="rename",
                handler="rename_handler",
            ),
        )
        assert spec.fix_source is not None
        assert spec.fix_source.kind == "rename"

    def test_rule_spec_accepts_astgrep_fix_source(self) -> None:
        spec = m.EnforcementRuleSpec(
            id="ENFORCE-003",
            description="ast-grep fix source attached",
            severity=m.EnforcementRuleSeverity.HIGH,
            source=m.EnforcementAstGrepSource(
                skill="flext-patterns",
                rule_id="some-rule",
            ),
            fix_source=m.EnforcementAstGrepFixSource(
                skill="flext-patterns",
                rule_id="some-rule",
            ),
        )
        assert spec.fix_source is not None
        assert spec.fix_source.kind == "ast_grep_fix"

    def test_violation_outcome_strenum_values(self) -> None:
        assert m.ViolationOutcome.HIT.value == "HIT"
        assert m.ViolationOutcome.MISS.value == "MISS"
        assert m.ViolationOutcome.SKIP.value == "SKIP"
        assert m.ViolationOutcome.HIT is not m.ViolationOutcome.MISS

    def test_violation_report_required_fields(self) -> None:
        report = m.ViolationReport(
            rule_id="ENFORCE-099",
            outcome=m.ViolationOutcome.HIT,
            file=Path("/tmp/example.py"),
            line=42,
            symbol="MyClass.method",
            payload={"detail": "matched"},
        )
        assert report.rule_id == "ENFORCE-099"
        assert report.outcome is m.ViolationOutcome.HIT
        assert report.file == Path("/tmp/example.py")
        assert report.line == 42
        assert report.symbol == "MyClass.method"
        assert report.payload["detail"] == "matched"

    def test_violation_report_optional_locator_fields(self) -> None:
        report = m.ViolationReport(
            rule_id="ENFORCE-099",
            outcome=m.ViolationOutcome.SKIP,
            payload={"reason": "source_unavailable"},
        )
        assert report.file is None
        assert report.line is None
        assert report.symbol is None

    def test_existing_38_rules_still_validate(self) -> None:
        """Regression: backward compatibility for the existing catalog."""
        catalog = c.ENFORCEMENT_CATALOG
        assert len(catalog.rules) >= 38
        for rule in catalog.rules:
            assert rule.fix_source is None  # existing rules have no fix_source
