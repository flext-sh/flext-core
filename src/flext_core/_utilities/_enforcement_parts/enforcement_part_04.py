"""Runtime enforcement engine MRO part."""

from __future__ import annotations

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me

from .enforcement_part_01 import (
    PREDICATE_BINDINGS,
)
from .enforcement_part_03 import (
    FlextUtilitiesEnforcement as FlextUtilitiesEnforcementPart03,
)


class FlextUtilitiesEnforcement(FlextUtilitiesEnforcementPart03):
    @classmethod
    def build_canonical_catalog(cls) -> me.EnforcementCatalog:
        """Build (cached) the canonical enforcement catalog from constants rows."""
        if cls._canonical_catalog is not None:
            return cls._canonical_catalog
        # Build all FLEXT_INFRA_DETECTOR rules from the compact data-table via a
        # single Pydantic v2 comprehension. ``EnforcementRuleSeverity`` is a
        # StrEnum so the severity string coerces directly.
        infra_specs = tuple(
            me.EnforcementRuleSpec(
                id=rid,
                description=desc,
                severity=me.EnforcementRuleSeverity(sev),
                source=me.EnforcementInfraDetectorSource(
                    violation_field=vf,
                    match_missing=mm,
                ),
                agents_md_anchor=anchor,
                skills=skills,
            )
            for rid, sev, vf, anchor, skills, mm, desc in c.INFRA_DETECTOR_ROWS
        )
        # Same comprehension applied to the BEARTYPE family.
        beartype_specs = tuple(
            me.EnforcementRuleSpec(
                id=rid,
                description=desc,
                severity=me.EnforcementRuleSeverity(sev),
                source=me.EnforcementBeartypeSource(
                    predicate_kind=PREDICATE_BINDINGS[tag][0],
                ),
                agents_md_anchor=anchor,
                skills=skills,
            )
            for rid, sev, tag, anchor, skills, desc in c.BEARTYPE_ROWS
        )
        # FLEXT_TESTS_VALIDATOR family — one ``tv.<method>`` per row.
        tests_validator_specs = tuple(
            me.EnforcementRuleSpec(
                id=rid,
                description=desc,
                severity=me.EnforcementRuleSeverity(sev),
                source=me.EnforcementTestsValidatorSource(
                    method=method,
                    rule_ids=rule_ids,
                ),
                skills=skills,
            )
            for rid, sev, method, rule_ids, skills, desc in c.TESTS_VALIDATOR_ROWS
        )
        # AST_GREP family — every row's ``skills`` mirrors its source skill.
        ast_grep_specs = tuple(
            me.EnforcementRuleSpec(
                id=rid,
                description=desc,
                severity=me.EnforcementRuleSeverity(sev),
                source=me.EnforcementAstGrepSource(
                    skill=skill,
                    rule_id=rule_id,
                ),
                skills=(skill,),
            )
            for rid, sev, skill, rule_id, desc in c.AST_GREP_ROWS
        )
        # SKILL_POINTER family — narrative entries, all ``enabled=False``.
        skill_pointer_specs = tuple(
            me.EnforcementRuleSpec(
                id=rid,
                description=desc,
                severity=me.EnforcementRuleSeverity(sev),
                source=me.EnforcementSkillPointerSource(
                    skill=src_skill,
                    anchor=src_anchor,
                ),
                agents_md_anchor=md_anchor,
                skills=skills,
                enabled=False,
            )
            for rid, sev, src_skill, src_anchor, md_anchor, skills, desc in c.SKILL_POINTER_ROWS
        )
        # RUFF family — 3 documentation-only rules sharing one ``notes`` line.
        ruff_notes = (
            "Dispatched by ruff via make lint; catalog entry is documentation-only."
        )
        ruff_specs = tuple(
            me.EnforcementRuleSpec(
                id=rid,
                description=desc,
                severity=me.EnforcementRuleSeverity(sev),
                source=me.EnforcementRuffSource(rule_code=rule_code),
                skills=skills,
                notes=ruff_notes,
            )
            for rid, sev, rule_code, skills, desc in c.RUFF_ROWS
        )

        cls._canonical_catalog = me.EnforcementCatalog(
            rules=(
                # --- FLEXT_INFRA_DETECTOR (14 rules, table-driven above) ---
                *infra_specs,
                # --- FLEXT_TESTS_VALIDATOR (7 rules, table-driven) ---
                *tests_validator_specs,
                # --- RUNTIME_WARNING (1 rule) ---
                me.EnforcementRuleSpec(
                    id="ENFORCE-022",
                    description=(
                        "FlextMroViolation emitted by the flext-core enforcement "
                        "engine at class-definition time."
                    ),
                    severity=me.EnforcementRuleSeverity.HIGH,
                    source=me.EnforcementRuntimeWarningSource(
                        category="flext_core._constants.enforcement.FlextMroViolation",
                    ),
                    skills=("flext-mro-namespace-rules", "pydantic-v2-governance"),
                ),
                # --- RUFF (3 rules, table-driven) ---
                *ruff_specs,
                # --- AST_GREP (8 rules, table-driven via AST_GREP_ROWS) ---
                *ast_grep_specs,
                # --- SKILL_POINTER (5 rules — narrative, all enabled=False) ---
                *skill_pointer_specs,
                # ENFORCE-039..044 + 045..055: 15 BEARTYPE rules built from the
                # ``BEARTYPE_ROWS`` data-table above; ENFORCE-040 (RUFF source)
                # is interleaved in the original ordering and stays inline below
                # to preserve the catalog's source-grouped narrative.
                *beartype_specs[:1],  # ENFORCE-039 (cast outside core)
                me.EnforcementRuleSpec(
                    id="ENFORCE-040",
                    description=(
                        "Linter ignore directive without inline justification "
                        "violates AGENTS.md §3.5 (Linter Zero Tolerance + "
                        "Suppressions)."
                    ),
                    severity=me.EnforcementRuleSeverity.MEDIUM,
                    source=me.EnforcementRuffSource(
                        rule_code="PGH003",
                    ),
                    agents_md_anchor="3-5-integrity",
                    skills=("flext-strict-typing", "flext-quality-gates"),
                ),
                *beartype_specs[1:],  # ENFORCE-041..055
                # ENFORCE-066+: plan §1.2 source-rule range (shifted from plan IDs
                # 053..065 because off-plan 045..053 already occupy the original
                # slots — IDs are flexible per AGENTS.md, semantic content matches
                # the plan exactly). First entry registers MINIMAL_AST in the
                # catalog (catalog completeness invariant).
                me.EnforcementRuleSpec(
                    id="ENFORCE-066",
                    description=(
                        "Module-level alias assignment (``LegacyName = NewName``) "
                        "where both sides are CapWords and the LHS is unreferenced"
                        " is a backwards-compat shim. Violates AGENTS.md §2.4 "
                        "(No Backward-Compat Aliases) — plan §1.2 row 053."
                    ),
                    severity=me.EnforcementRuleSeverity.MEDIUM,
                    source=me.EnforcementMinimalAstSource(
                        pattern="$X = $Y",
                        require_source=True,
                    ),
                    agents_md_anchor="2-4-no-backwards-compat-aliases",
                    skills=("flext-mro-namespace-rules",),
                ),
            ),
        )
        return cls._canonical_catalog


__all__: list[str] = ["FlextUtilitiesEnforcement"]
