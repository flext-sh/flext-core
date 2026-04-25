"""Behavior contract for the typed enforcement source variants and ViolationReport.

Covers the Phase 0 / Task 0.1 surface that does not require a catalog
consumer: ``EnforcementRopeFixSource`` / ``EnforcementAstGrepFixSource``
discriminator schemas, ``c.ViolationOutcome`` / ``ViolationReport``, and
the optional ``fix_source`` discriminated union on ``EnforcementRuleSpec``.

``EnforcementSourceKind.MINIMAL_AST`` and ``EnforcementMinimalAstSource``
land together with their first consumer rule in Task 1.1 — the catalog
completeness invariant requires every declared source kind to appear on
at least one ``EnforcementRuleSpec``.
"""

from __future__ import annotations

import pytest

from tests import c, m


class TestsFlextCoreModelsEnforcementSources:
    """Behavior contract for the new EnforcementSource variants + ViolationReport."""

    # --- EnforcementSourceKind ---

    def test_enforcement_source_kind_includes_beartype(self) -> None:
        assert m.EnforcementSourceKind.BEARTYPE == "beartype"

    def test_enforcement_source_kind_includes_minimal_ast(self) -> None:
        assert m.EnforcementSourceKind.MINIMAL_AST == "minimal_ast"

    # --- EnforcementMinimalAstSource ---

    def test_minimal_ast_source_default_require_source_true(self) -> None:
        src = m.EnforcementMinimalAstSource(pattern="$X = $Y")
        assert src.kind == "minimal_ast"
        assert src.pattern == "$X = $Y"
        assert src.require_source is True

    def test_minimal_ast_source_explicit_require_source_false(self) -> None:
        src = m.EnforcementMinimalAstSource(pattern="$X = $Y", require_source=False)
        assert src.require_source is False

    def test_minimal_ast_source_rejects_empty_pattern(self) -> None:
        with pytest.raises(c.ValidationError):
            m.EnforcementMinimalAstSource(pattern="")

    # --- EnforcementRopeFixSource ---

    def test_rope_fix_source_accepts_known_operation(self) -> None:
        src = m.EnforcementRopeFixSource(operation="rename", handler="rename_xyz")
        assert src.kind == "rope_fix"
        assert src.operation == "rename"
        assert src.handler == "rename_xyz"

    def test_rope_fix_source_rejects_unknown_operation(self) -> None:
        with pytest.raises(c.ValidationError):
            m.EnforcementRopeFixSource(operation="unknown", handler="x")

    def test_rope_fix_source_rejects_empty_handler(self) -> None:
        with pytest.raises(c.ValidationError):
            m.EnforcementRopeFixSource(operation="rename", handler="")

    # --- EnforcementAstGrepFixSource ---

    def test_astgrep_fix_source_links_skill_rule(self) -> None:
        src = m.EnforcementAstGrepFixSource(
            skill="flext-import-rules",
            rule_id="rewrite-pydantic-direct",
        )
        assert src.kind == "ast_grep_fix"
        assert src.skill == "flext-import-rules"
        assert src.rule_id == "rewrite-pydantic-direct"

    # --- ViolationOutcome ---

    def test_violation_outcome_strenum(self) -> None:
        assert c.ViolationOutcome.HIT == "HIT"
        assert c.ViolationOutcome.MISS == "MISS"
        assert c.ViolationOutcome.SKIP == "SKIP"
        assert c.ViolationOutcome.SKIP != c.ViolationOutcome.MISS

    # --- ViolationReport ---

    def test_violation_report_well_formed(self) -> None:
        report = m.ViolationReport(
            rule_id="ENFORCE-045",
            outcome=c.ViolationOutcome.HIT,
            file="x.py",
            line=42,
            symbol="Foo.bar",
        )
        assert report.rule_id == "ENFORCE-045"
        assert report.outcome == c.ViolationOutcome.HIT
        assert report.file == "x.py"
        assert report.line == 42
        assert report.symbol == "Foo.bar"
        assert report.payload == {}

    def test_violation_report_skip_with_no_location(self) -> None:
        report = m.ViolationReport(
            rule_id="ENFORCE-045",
            outcome=c.ViolationOutcome.SKIP,
        )
        assert report.outcome == c.ViolationOutcome.SKIP
        assert report.file == ""
        assert report.line == 0
        assert report.symbol == ""

    def test_violation_report_rejects_invalid_rule_id_pattern(self) -> None:
        with pytest.raises(c.ValidationError):
            m.ViolationReport(
                rule_id="not-an-enforce-id", outcome=c.ViolationOutcome.HIT
            )

    # --- EnforcementRuleSpec.fix_source ---

    def test_rule_spec_fix_source_optional_default_none(self) -> None:
        spec = m.EnforcementRuleSpec(
            id="ENFORCE-901",
            description="x",
            severity=m.EnforcementRuleSeverity.LOW,
            source=m.EnforcementRuffSource(rule_code="PGH003"),
        )
        assert spec.fix_source is None

    def test_rule_spec_fix_source_accepts_rope_variant(self) -> None:
        spec = m.EnforcementRuleSpec(
            id="ENFORCE-902",
            description="x",
            severity=m.EnforcementRuleSeverity.MEDIUM,
            source=m.EnforcementInfraDetectorSource(violation_field="loose_objects"),
            fix_source=m.EnforcementRopeFixSource(
                operation="custom", handler="rewrite_x"
            ),
        )
        assert spec.fix_source is not None
        assert spec.fix_source.kind == "rope_fix"

    def test_rule_spec_fix_source_accepts_astgrep_variant(self) -> None:
        spec = m.EnforcementRuleSpec(
            id="ENFORCE-903",
            description="x",
            severity=m.EnforcementRuleSeverity.MEDIUM,
            source=m.EnforcementInfraDetectorSource(
                violation_field="import_violations"
            ),
            fix_source=m.EnforcementAstGrepFixSource(
                skill="flext-import-rules", rule_id="rewrite-yaml-direct"
            ),
        )
        assert spec.fix_source is not None
        assert spec.fix_source.kind == "ast_grep_fix"
