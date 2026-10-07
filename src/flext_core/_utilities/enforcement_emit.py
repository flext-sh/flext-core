"""Enforcement emission primitives: violation assembly, emit, exemptions.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib.resources
import warnings
from types import MappingProxyType
from typing import ClassVar

from flext_core._constants import _enforcement_data
from flext_core._constants.enforcement import (
    FlextConstantsEnforcement as c,
    FlextMroViolation,
)
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t


class FlextUtilitiesEnforcementEmit:
    """Violation factory + warning/strict emission + exemption rules."""

    _canonical_catalog: ClassVar[me.EnforcementCatalog | None] = None
    _rules_by_tag: ClassVar[t.MappingKV[str, me.EnforcementRuleSpec] | None] = None

    @classmethod
    def build_canonical_catalog(cls) -> me.EnforcementCatalog:
        """Return the enforcement catalog validated from its package data.

        Each rule carries the fix action declared for its id.

        Returns:
            The enforcement catalog validated from its package data.

        """
        if cls._canonical_catalog is None:
            catalog = me.EnforcementCatalog.model_validate_json(
                importlib.resources
                .files(_enforcement_data)
                .joinpath(c.ENFORCEMENT_CATALOG_RESOURCE)
                .read_text(encoding="utf-8"),
            )
            fix_actions = c.ENFORCEMENT_FIX_ACTIONS
            cls._canonical_catalog = catalog.model_copy(
                update={
                    "rules": tuple(
                        rule.model_copy(
                            update={
                                "fix_action": me.EnforcementFixAction.model_validate(
                                    fix_actions[rule.id],
                                ),
                            },
                        )
                        if rule.id in fix_actions
                        else rule
                        for rule in catalog.rules
                    ),
                },
            )
        return cls._canonical_catalog

    @classmethod
    def rules_by_tag(cls) -> t.MappingKV[str, me.EnforcementRuleSpec]:
        """Return catalog rules keyed by their runtime predicate or smell tag.

        Returns:
            Catalog rules keyed by their runtime predicate or smell tag.

        """
        if cls._rules_by_tag is None:
            cls._rules_by_tag = MappingProxyType({
                (
                    rule.source.tag
                    if isinstance(rule.source, me.EnforcementBeartypeSource)
                    else rule.source.smell_tag
                ): rule
                for rule in cls.build_canonical_catalog().rules
                if isinstance(
                    rule.source,
                    me.EnforcementBeartypeSource | me.EnforcementCodeSmellSource,
                )
            })
        return cls._rules_by_tag

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

        Raises:
            TypeError: If ``active is c.EnforcementMode.STRICT``.

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
            rules = FlextUtilitiesEnforcementEmit.rules_by_tag()
            category = (
                c.FlextSmellViolation
                if any(
                    rule.id == v.rule_id
                    for tag, rule in rules.items()
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

        Returns:
            The resulting ``str | None``.

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
