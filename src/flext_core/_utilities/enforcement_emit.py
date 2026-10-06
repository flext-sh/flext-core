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
    FlextConstantsEnforcement,
    FlextMroViolation,
)
from flext_core._models.enforcement import FlextModelsEnforcement
from flext_core._typings.base import FlextTypingBase


class FlextUtilitiesEnforcementEmit:
    """Violation factory + warning/strict emission + exemption rules."""

    _canonical_catalog: ClassVar[FlextModelsEnforcement.EnforcementCatalog | None] = (
        None
    )
    _rules_by_tag: ClassVar[
        FlextTypingBase.MappingKV[str, FlextModelsEnforcement.EnforcementRuleSpec]
        | None
    ] = None

    @classmethod
    def build_canonical_catalog(cls) -> FlextModelsEnforcement.EnforcementCatalog:
        """Return the enforcement catalog validated from its package data.

        Each rule carries the fix action declared for its id.

        Returns:
            The enforcement catalog validated from its package data.

        """
        if cls._canonical_catalog is None:
            catalog = FlextModelsEnforcement.EnforcementCatalog.model_validate_json(
                importlib.resources
                .files(_enforcement_data)
                .joinpath(FlextConstantsEnforcement.ENFORCEMENT_CATALOG_RESOURCE)
                .read_text(encoding="utf-8"),
            )
            fix_actions = FlextConstantsEnforcement.ENFORCEMENT_FIX_ACTIONS
            fix_action_type = FlextModelsEnforcement.EnforcementFixAction
            cls._canonical_catalog = catalog.model_copy(
                update={
                    "rules": tuple(
                        rule.model_copy(
                            update={
                                "fix_action": fix_action_type.model_validate(
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
    def rules_by_tag(
        cls,
    ) -> FlextTypingBase.MappingKV[str, FlextModelsEnforcement.EnforcementRuleSpec]:
        """Return catalog rules keyed by their runtime predicate or smell tag.

        Returns:
            Catalog rules keyed by their runtime predicate or smell tag.

        """
        if cls._rules_by_tag is None:
            cls._rules_by_tag = MappingProxyType({
                (
                    rule.source.tag
                    if isinstance(
                        rule.source,
                        FlextModelsEnforcement.EnforcementBeartypeSource,
                    )
                    else rule.source.smell_tag
                ): rule
                for rule in cls.build_canonical_catalog().rules
                if isinstance(
                    rule.source,
                    FlextModelsEnforcement.EnforcementBeartypeSource
                    | FlextModelsEnforcement.EnforcementCodeSmellSource,
                )
            })
        return cls._rules_by_tag

    @staticmethod
    def _violation(
        tag: str,
        location: str,
        qualname: str,
        detail: FlextTypingBase.StrMapping | None = None,
        category: FlextConstantsEnforcement.EnforcementCategory | None = None,
    ) -> FlextModelsEnforcement.Violation:
        # Look up problem/fix templates from the canonical text mapping
        problem, fix = FlextConstantsEnforcement.ENFORCEMENT_RULES_TEXT[tag]
        subs = detail or {}
        message = FlextConstantsEnforcement.ENFORCEMENT_MSG_VIOLATION.format(
            location=location,
            problem=problem.format(**subs) if subs else problem,
            fix=fix.format(**subs) if subs else fix,
        )
        rule = FlextUtilitiesEnforcementEmit.rules_by_tag().get(tag)
        rule_id = rule.id if rule is not None else ""
        anchor = rule.agents_md_anchor if rule is not None else ""
        message = f"{message} [{rule_id}]" if rule_id else f"{message} [{tag}]"

        layer = "Model"
        if category is FlextConstantsEnforcement.EnforcementCategory.ATTR:
            layer = FlextConstantsEnforcement.ENFORCEMENT_TAG_LAYER.get(
                tag,
                "Attributes",
            )
        elif category is FlextConstantsEnforcement.EnforcementCategory.NAMESPACE:
            layer = "Namespace"
        elif category is FlextConstantsEnforcement.EnforcementCategory.PROTOCOL_TREE:
            layer = "Protocols"
        elif (
            category is FlextConstantsEnforcement.EnforcementCategory.MODEL_CLASS
            or category is FlextConstantsEnforcement.EnforcementCategory.FIELD
        ):
            layer = "Model"

        return FlextModelsEnforcement.Violation(
            qualname=qualname,
            layer=layer,
            severity="HARD rules",
            rule_id=rule_id,
            agents_md_anchor=anchor,
            message=message,
        )

    @staticmethod
    def emit(
        report: FlextModelsEnforcement.Report,
        *,
        mode: FlextConstantsEnforcement.EnforcementMode | None = None,
    ) -> None:
        """Emit violations as warnings (or raise TypeError in STRICT mode).

        Legal TYPE_CHECKING deferrals remain in ``report.deferred``; they are
        not runtime violations. Consumers claiming complete inspection must
        also require ``report.complete``.

        Raises:
            TypeError: If ``active is c.EnforcementMode.STRICT``.

        """
        if report.empty:
            return
        active = mode or FlextConstantsEnforcement.ENFORCEMENT_MODE
        if active is FlextConstantsEnforcement.EnforcementMode.OFF:
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
            rules_by_tag = FlextUtilitiesEnforcementEmit.rules_by_tag()
            category = (
                FlextConstantsEnforcement.FlextSmellViolation
                if any(
                    rule.id == v.rule_id
                    for tag, rule in rules_by_tag.items()
                    if tag in FlextConstantsEnforcement.ENFORCEMENT_SMELL_TAGS
                )
                else FlextMroViolation
            )
            warnings.warn(msg, category, stacklevel=4)
            if active is FlextConstantsEnforcement.EnforcementMode.STRICT:
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
        for suffix, layer in FlextConstantsEnforcement.ENFORCEMENT_NAMESPACE_LAYER_MAP:
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
