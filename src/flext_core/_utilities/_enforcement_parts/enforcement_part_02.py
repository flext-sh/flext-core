"""Runtime enforcement engine MRO part.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import EnumType
from typing import ClassVar

from pydantic_settings import BaseSettings

from flext_core._constants import FlextConstantsEnforcement
from flext_core._models import FlextModelsEnforcement, FlextModelsPydantic
from flext_core._protocols import FlextProtocolsBase
from flext_core._utilities._enforcement_parts.enforcement_part_01 import (
    PREDICATE_BINDINGS,
)
from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine
from flext_core._utilities.enforcement_collect import FlextUtilitiesEnforcementCollect


class FlextUtilitiesEnforcement(FlextUtilitiesEnforcementCollect):
    """Rule-driven runtime enforcement (static-only)."""

    _MODEL_CONSTRUCTION_CATEGORIES: ClassVar[
        frozenset[FlextConstantsEnforcement.EnforcementCategory]
    ] = frozenset({
        FlextConstantsEnforcement.EnforcementCategory.FIELD,
        FlextConstantsEnforcement.EnforcementCategory.MODEL_CLASS,
    })

    @staticmethod
    def _apply_rule(
        _target: type,
        tag: str,
        qualname: str,
        items: Iterator[tuple[str, tuple[FlextProtocolsBase.AttributeProbe, ...]]],
        category: FlextConstantsEnforcement.EnforcementCategory,
    ) -> FlextModelsEnforcement.Report:
        """Apply a rule, separating proven deferrals from executed predicates.

        Every runtime tag carries its category and its predicate binding in the
        same data row, so a tag without a binding is a data defect and raises.

        Returns:
            The resulting ``me.Report``.

        """
        kind, params = PREDICATE_BINDINGS[tag]
        violations: list[FlextModelsEnforcement.Violation] = []
        deferred: list[FlextModelsEnforcement.DeferredInspection] = []
        for location, args in items:
            unavailable = FlextUtilitiesBeartypeEngine.deferred_aliases(
                params,
                _target,
                *args,
            )
            if unavailable:
                deferred.extend(
                    FlextModelsEnforcement.DeferredInspection(
                        tag=tag,
                        location=location,
                        alias=alias,
                    )
                    for alias in unavailable
                )
                continue
            detail = FlextUtilitiesBeartypeEngine.apply(kind, params, *args)
            if detail is not None:
                violations.append(
                    FlextUtilitiesEnforcement._violation(
                        tag,
                        location,
                        qualname,
                        detail,
                        category=category,
                    ),
                )
        return FlextModelsEnforcement.Report(violations=violations, deferred=deferred)

    @staticmethod
    def _items_for(
        target: type,
        tag: str,
        category: FlextConstantsEnforcement.EnforcementCategory,
        effective_layer: str,
    ) -> Iterator[tuple[str, tuple[FlextProtocolsBase.AttributeProbe, ...]]]:
        """Return category-specific (location, args) pairs for one rule tag.

        This is the single category→iterator dispatch — ``check()`` runs
        every row in ``c.ENFORCEMENT_RULES`` through here and pipes the
        result into :meth:`_apply_rule`.

        Yields:
            Each ``tuple[str, tuple[p.AttributeProbe, ...]]``.

        """
        # A class is a model by DECLARATION: the canonical FLEXT base or a
        # pydantic-settings base declared directly (which is exactly what the
        # settings-inheritance rule must see to report the bypass).
        is_model = issubclass(target, FlextModelsPydantic.BaseModel) or issubclass(
            target,
            BaseSettings,
        )
        if "[" in target.__name__:
            return
        yield from FlextUtilitiesEnforcement._category_items(
            target,
            tag,
            category,
            effective_layer,
            is_model=is_model,
        )

    @classmethod
    def _category_items(
        cls,
        target: type,
        tag: str,
        category: FlextConstantsEnforcement.EnforcementCategory,
        effective_layer: str,
        *,
        is_model: bool,
    ) -> Iterator[tuple[str, tuple[FlextProtocolsBase.AttributeProbe, ...]]]:
        """Resolve one enforcement category to its item iterator.

        Returns:
            The resulting item iterator.

        """
        items: Iterator[tuple[str, tuple[FlextProtocolsBase.AttributeProbe, ...]]] = (
            iter(())
        )
        if category is FlextConstantsEnforcement.EnforcementCategory.FIELD:
            # Field collection is uniform across every model base: settings
            # classes subclass BaseModel, so this check matches is_model while
            # handing _field_items exactly one type[BaseModel].
            if issubclass(target, FlextModelsPydantic.BaseModel):
                items = FlextUtilitiesEnforcement._field_items(target, tag)
        elif category is FlextConstantsEnforcement.EnforcementCategory.MODEL_CLASS:
            if is_model:
                items = iter(((target.__qualname__, (target,)),))
        elif category is FlextConstantsEnforcement.EnforcementCategory.ATTR:
            rule_layer = FlextConstantsEnforcement.ENFORCEMENT_TAG_LAYER.get(tag, "")
            if rule_layer.lower() == effective_layer:
                items = FlextUtilitiesEnforcement._attr_items(target, effective_layer)
        elif category is FlextConstantsEnforcement.EnforcementCategory.NAMESPACE:
            items = FlextUtilitiesEnforcement._namespace_items(
                target,
                tag,
                effective_layer,
            )
        elif (
            category is FlextConstantsEnforcement.EnforcementCategory.PROTOCOL_TREE
            and effective_layer
            == FlextConstantsEnforcement.EnforcementLayer.PROTOCOLS.lower()
        ):
            items = cls._walk_protocol_tree(target, tag, target.__qualname__)
        return items

    @classmethod
    def _walk_protocol_tree(
        cls,
        node: type,
        tag: str,
        path: str,
    ) -> Iterator[tuple[str, tuple[FlextProtocolsBase.AttributeProbe, ...]]]:
        """Walk one namespace value tree, yielding protocol/nested entries.

        Yields:
            Each ``tuple[str, tuple[p.AttributeProbe, ...]]``.

        """
        iterator = (
            FlextUtilitiesEnforcement._iter_effective
            if tag == "proto_inner_kind"
            else FlextUtilitiesEnforcement._iter_inner
        )
        for name, value in iterator(node):
            nested = f"{path}.{name}"
            yield nested, (value,)
            if FlextUtilitiesBeartypeEngine.has_runtime_protocol_marker(
                value,
            ) or FlextUtilitiesBeartypeEngine.has_nested_namespace(
                value,
            ):
                yield from cls._walk_protocol_tree(value, tag, nested)

    @staticmethod
    def _check(
        target: type,
        *,
        layer: str | None = None,
        categories: frozenset[FlextConstantsEnforcement.EnforcementCategory]
        | None = None,
    ) -> FlextModelsEnforcement.Report:
        """Query applicable rules and return a typed report (no emission).

        Every rule dispatches through the unified :meth:`_apply_rule` —
        no per-category engine duplication; item iterators live in the
        ``_*_items`` / :meth:`_items_for` helpers and vary only by tag.
        Attr-rule recursion is handled via ``c.ENFORCEMENT_RECURSIVE_TAGS``.

        Returns:
            The resulting ``me.Report``.

        """
        violations: list[FlextModelsEnforcement.Violation] = []
        deferred: list[FlextModelsEnforcement.DeferredInspection] = []
        effective_layer = layer or FlextUtilitiesEnforcement.detect_layer(target) or ""
        qn = target.__qualname__
        for tag, category in FlextConstantsEnforcement.ENFORCEMENT_TAG_CATEGORY.items():
            if categories is not None and category not in categories:
                continue
            rule_layer = FlextConstantsEnforcement.ENFORCEMENT_TAG_LAYER.get(tag, "")
            items = FlextUtilitiesEnforcement._items_for(
                target,
                tag,
                category,
                effective_layer,
            )
            report = FlextUtilitiesEnforcement._apply_rule(
                target,
                tag,
                qn,
                items,
                category,
            )
            violations.extend(report.violations)
            deferred.extend(report.deferred)
            if (
                category is FlextConstantsEnforcement.EnforcementCategory.ATTR
                and tag in FlextConstantsEnforcement.ENFORCEMENT_RECURSIVE_TAGS
                and rule_layer.lower() == effective_layer
            ):
                for _name, inner in FlextUtilitiesEnforcement._iter_inner(target):
                    if isinstance(
                        inner,
                        EnumType,
                    ) or not FlextUtilitiesBeartypeEngine.defined_inside(
                        inner,
                        target.__qualname__,
                    ):
                        continue
                    nested = FlextUtilitiesEnforcement.check(
                        inner,
                        layer=effective_layer,
                    )
                    violations.extend(nested.violations)
                    deferred.extend(nested.deferred)
        return FlextModelsEnforcement.Report(violations=violations, deferred=deferred)

    @staticmethod
    def check(
        target: type,
        *,
        layer: str | None = None,
    ) -> FlextModelsEnforcement.Report:
        """Query all applicable rules and return a typed report (no emission).

        Returns:
            The resulting ``me.Report``.

        """
        return FlextUtilitiesEnforcement._check(target, layer=layer)

    @staticmethod
    def check_model_construction(
        target: type[FlextModelsPydantic.BaseModel],
    ) -> FlextModelsEnforcement.Report:
        """Run only Pydantic construction rules for ``__pydantic_init_subclass__``.

        Returns:
            The resulting ``me.Report``.

        """
        return FlextUtilitiesEnforcement._check(
            target,
            categories=FlextUtilitiesEnforcement._MODEL_CONSTRUCTION_CATEGORIES,
        )


__all__: list[str] = ["FlextUtilitiesEnforcement"]
