"""Runtime enforcement engine MRO part."""

from __future__ import annotations

from typing import Literal, cast

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me

from .enforcement_part_01 import PREDICATE_BINDINGS
from .enforcement_part_03 import (
    FlextUtilitiesEnforcement as FlextUtilitiesEnforcementPart03,
)

from flext_core._typings.base import FlextTypingBase as t


class FlextUtilitiesEnforcement(FlextUtilitiesEnforcementPart03):
    @classmethod
    def _fix_action_for(cls, rule_id: str) -> me.EnforcementFixAction | None:
        """Return the catalog fix-action for ``rule_id`` when one exists."""
        raw = c.ENFORCEMENT_FIX_ACTIONS.get(rule_id)
        if raw is None:
            return None
        fix = cast("dict[str, t.JsonValue]", raw)
        return me.EnforcementFixAction(
            kind=cast("Literal['gate', 'manual', 'rope', 'transformer']", fix["kind"]),
            target=cast("str", fix["target"]),
            params=cast("t.JsonMapping", fix.get("params", {})),
            safe=cast("bool", fix.get("safe", True)),
        )

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
                    violation_field=vf, match_missing=mm
                ),
                agents_md_anchor=anchor,
                skills=skills,
                fix_action=cls._fix_action_for(rid),
                enabled=rid not in c.STAGED_INFRA_RULE_IDS,
            )
            for rid, sev, vf, anchor, skills, mm, desc in c.INFRA_DETECTOR_ROWS
        )
        # Same comprehension applied to the BEARTYPE family, including the
        # JSON-loaded function-parameters smell rule (ENFORCE-071).
        beartype_specs = tuple(
            me.EnforcementRuleSpec(
                id=rid,
                description=desc,
                severity=me.EnforcementRuleSeverity(sev),
                source=me.EnforcementBeartypeSource(
                    predicate_kind=PREDICATE_BINDINGS[tag][0]
                ),
                agents_md_anchor=anchor,
                skills=skills,
                fix_action=cls._fix_action_for(rid),
            )
            for rid, sev, tag, anchor, skills, desc in (
                *c.BEARTYPE_ROWS,
                *c.SMELL_BEARTYPE_ROWS,
            )
        )
        # CODE_SMELL family — qlty-detected smells documented in the catalog.
        code_smell_specs = tuple(
            me.EnforcementRuleSpec(
                id=rid,
                description=desc,
                severity=me.EnforcementRuleSeverity(sev),
                source=me.EnforcementCodeSmellSource(smell_tag=tag),
                agents_md_anchor=anchor,
                skills=skills,
                fix_action=cls._fix_action_for(rid),
            )
            for rid, sev, tag, anchor, skills, desc in c.SMELL_CODE_SMELL_ROWS
        )
        # FLEXT_TESTS_VALIDATOR family — one ``tv.<method>`` per row.
        tests_validator_specs = tuple(
            me.EnforcementRuleSpec(
                id=rid,
                description=desc,
                severity=me.EnforcementRuleSeverity(sev),
                source=me.EnforcementTestsValidatorSource(
                    method=method, rule_ids=rule_ids
                ),
                skills=skills,
                fix_action=cls._fix_action_for(rid),
            )
            for rid, sev, method, rule_ids, skills, desc in c.TESTS_VALIDATOR_ROWS
        )
        # SKILL_POINTER family — narrative entries, all ``enabled=False``.
        skill_pointer_specs = tuple(
            me.EnforcementRuleSpec(
                id=rid,
                description=desc,
                severity=me.EnforcementRuleSeverity(sev),
                source=me.EnforcementSkillPointerSource(
                    skill=src_skill, anchor=src_anchor
                ),
                agents_md_anchor=md_anchor,
                skills=skills,
                enabled=False,
                fix_action=cls._fix_action_for(rid),
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
                fix_action=cls._fix_action_for(rid),
            )
            for rid, sev, rule_code, skills, desc in c.RUFF_ROWS
        )

        cls._canonical_catalog = me.EnforcementCatalog(
            rules=(
                # --- FLEXT_INFRA_DETECTOR (22 rules, table-driven above) ---
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
                        category="flext_core._constants.enforcement.FlextMroViolation"
                    ),
                    skills=("flext-mro-namespace-rules", "pydantic-v2-governance"),
                    fix_action=cls._fix_action_for("ENFORCE-022"),
                ),
                # --- RUFF (3 rules, table-driven) ---
                *ruff_specs,
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
                    source=me.EnforcementRuffSource(rule_code="PGH003"),
                    agents_md_anchor="3-5-integrity",
                    skills=("flext-strict-typing", "flext-quality-gates"),
                    fix_action=cls._fix_action_for("ENFORCE-040"),
                ),
                *beartype_specs[1:],  # ENFORCE-041..071 (066+ runtime-enforced)
                # --- CODE_SMELL (7 rules, JSON-loaded via SMELL_CODE_SMELL_ROWS) ---
                *code_smell_specs,
            )
        )
        return cls._canonical_catalog


__all__: list[str] = ["FlextUtilitiesEnforcement"]
