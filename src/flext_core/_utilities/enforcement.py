"""Runtime enforcement engine — rules are DATA in c.*, predicates are METHODS in ub.*.

Every row in ``c.ENFORCEMENT_RULES`` is paired by naming convention with a
``check_<tag>`` staticmethod on ``FlextUtilitiesBeartypeEngine``. This module
holds only the dispatch engine + violation assembly; no rule strings, no
predicate bodies, no back-compat shims.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Iterator,
    Sequence,
)
from enum import EnumType
from typing import ClassVar

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine as ub
from flext_core._utilities.enforcement_collect import FlextUtilitiesEnforcementCollect


class FlextUtilitiesEnforcement(FlextUtilitiesEnforcementCollect):
    """Rule-driven runtime enforcement (static-only)."""

    @staticmethod
    def _apply_rule(
        tag: str,
        qualname: str,
        items: Iterator[tuple[str, tuple[object, ...]]],
    ) -> Sequence[me.Violation]:
        """Call ``ub.check_<tag>(*args)`` per item; emit violation on non-None."""
        predicate: Callable[..., t.StrMapping | None] = getattr(
            ub,
            f"check_{tag}",
        )
        return [
            FlextUtilitiesEnforcement._violation(tag, location, qualname, detail)
            for location, args in items
            if (detail := predicate(*args)) is not None
        ]

    # ------------------------------------------------------------------
    # Category-specific item iterators (feed _apply_rule)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Public query + emission API
    # ------------------------------------------------------------------

    @staticmethod
    def _items_for(
        target: type,
        tag: str,
        category: c.EnforcementCategory,
        effective_layer: str,
        rule_layer: str,
        *,
        is_model: bool,
    ) -> Iterator[tuple[str, tuple[object, ...]]]:
        """Return category-specific (location, args) pairs for one rule tag.

        This is the single category→iterator dispatch — ``check()`` runs
        every row in ``c.ENFORCEMENT_RULES`` through here and pipes the
        result into :meth:`_apply_rule`.
        """
        # Generic/Pydantic specializations (``Foo[Bar]``) are synthetic
        # runtime artifacts shared across all categories — skip universally.
        if "[" in target.__name__:
            return
        if category is c.EnforcementCategory.FIELD:
            if is_model and issubclass(target, mp.BaseModel):
                yield from FlextUtilitiesEnforcement._field_items(
                    target,
                    tag,
                )
            return
        if category is c.EnforcementCategory.MODEL_CLASS:
            if is_model:
                yield target.__qualname__, (target,)
            return
        if category is c.EnforcementCategory.ATTR:
            if rule_layer.lower() == effective_layer:
                yield from FlextUtilitiesEnforcement._attr_items(
                    target,
                    effective_layer,
                )
            return
        if category is c.EnforcementCategory.NAMESPACE:
            yield from FlextUtilitiesEnforcement._namespace_items(
                target,
                tag,
                effective_layer,
            )
            return
        if category is c.EnforcementCategory.PROTOCOL_TREE:
            if effective_layer != c.EnforcementLayer.PROTOCOLS.lower():
                return
            iterator = (
                FlextUtilitiesEnforcement._iter_effective
                if tag == "proto_inner_kind"
                else FlextUtilitiesEnforcement._iter_inner
            )

            def walk(node: type, path: str) -> Iterator[tuple[str, tuple[object, ...]]]:
                for name, value in iterator(node):
                    nested = f"{path}.{name}"
                    yield nested, (value,)
                    if ub.has_runtime_protocol_marker(value) or ub.has_nested_namespace(
                        value
                    ):
                        yield from walk(value, nested)

            yield from walk(target, target.__qualname__)

    @staticmethod
    def check(target: type, *, layer: str | None = None) -> me.Report:
        """Query all applicable rules and return a typed report (no emission).

        Every rule dispatches through the unified :meth:`_apply_rule` —
        no per-category engine duplication; item iterators live in the
        ``_*_items`` / :meth:`_items_for` helpers and vary only by tag.
        Attr-rule recursion is handled via ``c.ENFORCEMENT_RECURSIVE_TAGS``.
        """
        violations: list[me.Violation] = []
        effective_layer = layer or FlextUtilitiesEnforcement.detect_layer(target) or ""
        is_model = issubclass(target, mp.BaseModel)
        qn = target.__qualname__
        for tag, row in c.ENFORCEMENT_RULES.items():
            category, rule_layer, *_ = row
            items = FlextUtilitiesEnforcement._items_for(
                target,
                tag,
                category,
                effective_layer,
                str(rule_layer),
                is_model=is_model,
            )
            violations.extend(
                FlextUtilitiesEnforcement._apply_rule(
                    tag,
                    qn,
                    items,
                )
            )
            if (
                category is c.EnforcementCategory.ATTR
                and tag in c.ENFORCEMENT_RECURSIVE_TAGS
                and str(rule_layer).lower() == effective_layer
            ):
                for _name, inner in FlextUtilitiesEnforcement._iter_inner(target):
                    if isinstance(inner, EnumType):
                        continue
                    violations.extend(
                        FlextUtilitiesEnforcement.check(
                            inner,
                            layer=effective_layer,
                        ).violations
                    )
        return me.Report(violations=violations)

    @staticmethod
    def run(model_type: type[mp.BaseModel]) -> None:
        """Pydantic ``__pydantic_init_subclass__`` hook.

        Function-local classes (Python's ``<locals>`` qualname marker)
        are ephemeral fixtures — validated on demand via ``check()`` but
        never emitted during class construction.
        """
        if c.ENFORCEMENT_MODE is c.EnforcementMode.OFF:
            return
        if ub.defined_in_function_scope(model_type):
            return
        if FlextUtilitiesEnforcement._is_exempt(model_type):
            return
        report = FlextUtilitiesEnforcement.check(model_type)
        FlextUtilitiesEnforcement.emit(report)

    @staticmethod
    def run_layer(target: type, layer: str) -> None:
        """Namespace ``__init_subclass__`` hook — run layer + namespace rules.

        Function-local classes skip emission for the same reason as ``run``.
        """
        if c.ENFORCEMENT_NAMESPACE_MODE is c.EnforcementMode.OFF:
            return
        if ub.defined_in_function_scope(target):
            return
        if FlextUtilitiesEnforcement._is_exempt(target):
            return
        report = FlextUtilitiesEnforcement.check(target, layer=layer)
        FlextUtilitiesEnforcement.emit(
            report,
            mode=c.ENFORCEMENT_NAMESPACE_MODE,
        )

    _CANONICAL_CATALOG: ClassVar[me.EnforcementCatalog | None] = None

    @classmethod
    def build_canonical_catalog(cls) -> me.EnforcementCatalog:
        """Build (cached) the canonical enforcement catalog from constants rows."""
        if cls._CANONICAL_CATALOG is not None:
            return cls._CANONICAL_CATALOG
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
                source=me.EnforcementBeartypeSource(hook=hook),
                agents_md_anchor=anchor,
                skills=skills,
            )
            for rid, sev, hook, anchor, skills, desc in c.BEARTYPE_ROWS
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

        cls._CANONICAL_CATALOG = me.EnforcementCatalog(
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
        return cls._CANONICAL_CATALOG


__all__: list[str] = ["FlextUtilitiesEnforcement"]
