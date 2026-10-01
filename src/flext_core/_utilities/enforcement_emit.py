"""Enforcement emission primitives: violation assembly, emit, exemptions."""

from __future__ import annotations

import importlib.resources
import warnings
from types import MappingProxyType
from typing import ClassVar

from .._constants import _enforcement_data
from .._constants.enforcement import (
    FlextConstantsEnforcement as c,
    FlextMroViolation,
    FlextSmellViolation,
)
from .._models.enforcement import FlextModelsEnforcement as me
from .._typings.base import FlextTypingBase as t


class FlextUtilitiesEnforcementEmit:
    """Violation factory + warning/strict emission + exemption rules."""

    _canonical_catalog: ClassVar[me.EnforcementCatalog | None] = None
    _rules_by_tag: ClassVar[t.MappingKV[str, me.EnforcementRuleSpec] | None] = None

    @staticmethod
    def build_canonical_catalog() -> me.EnforcementCatalog:
        """Return the enforcement catalog validated from its package data."""
        owner = FlextUtilitiesEnforcementEmit
        if owner._canonical_catalog is None:
            owner._canonical_catalog = me.EnforcementCatalog.model_validate_json(
                importlib.resources
                .files(_enforcement_data)
                .joinpath(c.ENFORCEMENT_CATALOG_RESOURCE)
                .read_text(encoding="utf-8")
            )
        return owner._canonical_catalog

    @staticmethod
    def rules_by_tag() -> t.MappingKV[str, me.EnforcementRuleSpec]:
        """Return catalog rules keyed by their runtime predicate or smell tag."""
        owner = FlextUtilitiesEnforcementEmit
        if owner._rules_by_tag is None:
            owner._rules_by_tag = MappingProxyType({
                (
                    rule.source.tag
                    if isinstance(rule.source, me.EnforcementBeartypeSource)
                    else rule.source.smell_tag
                ): rule
                for rule in owner.build_canonical_catalog().rules
                if isinstance(
                    rule.source,
                    me.EnforcementBeartypeSource | me.EnforcementCodeSmellSource,
                )
            })
        return owner._rules_by_tag

    @staticmethod
    def _violation(
        tag: str,
        location: str,
        qualname: str,
        detail: t.StrMapping | None = None,
        category: c.EnforcementCategory | None = None,
    ) -> me.Violation:
        # Look up problem/fix templates from the canonical text mapping
        problem, fix = c.ENFORCEMENT_RULES_TEXT[tag]
        subs = detail or {}
        message = c.ENFORCEMENT_MSG_VIOLATION.format(
            location=location,
            problem=problem.format(**subs) if subs else problem,
            fix=fix.format(**subs) if subs else fix,
        )
        rule = FlextUtilitiesEnforcementEmit.rules_by_tag().get(tag)
        rule_id = rule.id if rule is not None else ""
        anchor = rule.agents_md_anchor if rule is not None else ""
        message = f"{message} [{rule_id}]" if rule_id else f"{message} [{tag}]"

        layer = "Model"
        if category is c.EnforcementCategory.ATTR:
            layer = c.ENFORCEMENT_TAG_LAYER.get(tag, "Attributes")
        elif category is c.EnforcementCategory.NAMESPACE:
            layer = "Namespace"
        elif category is c.EnforcementCategory.PROTOCOL_TREE:
            layer = "Protocols"
        elif (
            category is c.EnforcementCategory.MODEL_CLASS
            or category is c.EnforcementCategory.FIELD
        ):
            layer = "Model"

        return me.Violation(
            qualname=qualname,
            layer=layer,
            severity="HARD rules",
            rule_id=rule_id,
            agents_md_anchor=anchor,
            message=message,
        )

    @staticmethod
    def emit(report: me.Report, *, mode: c.EnforcementMode | None = None) -> None:
        """Emit violations as warnings (or raise TypeError in STRICT mode).

        Legal TYPE_CHECKING deferrals remain in ``report.deferred``; they are
        not runtime violations. Consumers claiming complete inspection must
        also require ``report.complete``.
        """
        if report.empty:
            return
        active = mode or c.ENFORCEMENT_MODE
        if active is c.EnforcementMode.OFF:
            return
        for v in report.violations:
            if v.rule_id and v.agents_md_anchor:
                fix_note = (
                    f"See AGENTS.md §{v.agents_md_anchor} and search for {v.rule_id}."
                )
            elif v.rule_id:
                fix_note = f"Search for enforcement rule {v.rule_id}."
            elif v.agents_md_anchor:
                fix_note = f"See AGENTS.md §{v.agents_md_anchor}."
            else:
                fix_note = f"See AGENTS.md § {v.layer} governance."

            msg = (
                f"\n{v.qualname} violates FLEXT {v.layer} {v.severity}:\n  - "
                f"{v.message}\n\nFix: {fix_note}"
            )
            category = (
                FlextSmellViolation
                if any(
                    rule.id == v.rule_id
                    for tag, rule in FlextUtilitiesEnforcementEmit.rules_by_tag().items()
                    if tag in c.ENFORCEMENT_SMELL_TAGS
                )
                else FlextMroViolation
            )
            warnings.warn(msg, category, stacklevel=4)
            if active is c.EnforcementMode.STRICT:
                raise TypeError(msg)

    @staticmethod
    def detect_layer(target: type) -> str | None:
        """Infer the facade layer from the class name.

        Matches the layer keyword (``Constants`` / ``Models`` / ``Protocols``
        / ``Types`` / ``Utilities``) only when it appears as a standalone
        PascalCase segment — at the end of the name or immediately followed
        by another capitalised word (e.g. ``FooConstantsSettings``).
        This prevents false positives such as ``BadConstants`` where the
        layer keyword is embedded inside a larger word.
        Generic-specialization brackets (``Foo[Bar]``) are stripped first
        so the search ignores type-parameter noise.
        """
        name = target.__name__.partition("[")[0]
        for suffix, layer in c.ENFORCEMENT_NAMESPACE_LAYER_MAP:
            idx = name.find(suffix)
            if idx == -1:
                continue
            end = idx + len(suffix)
            # Suffix must be at the very end, or followed by an uppercase
            # letter (start of next PascalCase word).
            if end == len(name) or (end < len(name) and name[end].isupper()):
                return layer
        return None


__all__: list[str] = ["FlextUtilitiesEnforcementEmit"]
