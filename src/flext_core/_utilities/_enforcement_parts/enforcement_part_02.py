"""Runtime enforcement engine MRO part."""

from __future__ import annotations

from enum import EnumType
from typing import TYPE_CHECKING, ClassVar

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine as ub
from flext_core._utilities.enforcement_collect import FlextUtilitiesEnforcementCollect

from .enforcement_part_01 import (
    PREDICATE_BINDINGS,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
    )

    from flext_core._protocols.base import FlextProtocolsBase as p
    from flext_core._typings.base import FlextTypingBase as t


class FlextUtilitiesEnforcement(FlextUtilitiesEnforcementCollect):
    """Rule-driven runtime enforcement (static-only)."""

    _canonical_catalog: ClassVar[me.EnforcementCatalog | None] = None
    _MODEL_CONSTRUCTION_CATEGORIES: ClassVar[frozenset[c.EnforcementCategory]] = (
        frozenset({
            c.EnforcementCategory.FIELD,
            c.EnforcementCategory.MODEL_CLASS,
        })
    )

    @staticmethod
    def _apply_rule(
        _target: type,
        tag: str,
        qualname: str,
        items: Iterator[tuple[str, tuple[p.AttributeProbe, ...]]],
        category: c.EnforcementCategory,
    ) -> t.SequenceOf[me.Violation]:
        """Apply the visitor for ``tag`` to each item; emit violation on non-None.

        Catalog rules without a runtime predicate binding (static-only or
        beartype-driven entries keyed as ``ENFORCE-NNN``) are skipped gracefully.
        """
        binding = PREDICATE_BINDINGS.get(tag)
        if binding is None:
            return ()
        kind, params = binding
        return [
            FlextUtilitiesEnforcement._violation(
                tag,
                location,
                qualname,
                detail,
                category=category,
            )
            for location, args in items
            if (detail := ub.apply(kind, params, *args)) is not None
        ]

    @staticmethod
    def _items_for(
        target: type,
        tag: str,
        category: c.EnforcementCategory,
        effective_layer: str,
    ) -> Iterator[tuple[str, tuple[p.AttributeProbe, ...]]]:
        """Return category-specific (location, args) pairs for one rule tag.

        This is the single category→iterator dispatch — ``check()`` runs
        every row in ``c.ENFORCEMENT_RULES`` through here and pipes the
        result into :meth:`_apply_rule`.
        """
        is_model = issubclass(target, mp.BaseModel)
        rule_layer = c.ENFORCEMENT_TAG_LAYER.get(tag, "")
        if "[" in target.__name__:
            return

        def walk(
            node: type,
            path: str,
        ) -> Iterator[tuple[str, tuple[p.AttributeProbe, ...]]]:
            iterator = (
                FlextUtilitiesEnforcement._iter_effective
                if tag == "proto_inner_kind"
                else FlextUtilitiesEnforcement._iter_inner
            )
            for name, value in iterator(node):
                nested = f"{path}.{name}"
                yield nested, (value,)
                if ub.has_runtime_protocol_marker(value) or ub.has_nested_namespace(
                    value,
                ):
                    yield from walk(value, nested)

        items: Iterator[tuple[str, tuple[p.AttributeProbe, ...]]] = iter(())
        if category is c.EnforcementCategory.FIELD:
            if is_model:
                items = FlextUtilitiesEnforcement._field_items(target, tag)
        elif category is c.EnforcementCategory.MODEL_CLASS:
            if is_model:
                items = iter(((target.__qualname__, (target,)),))
        elif category is c.EnforcementCategory.ATTR:
            if rule_layer.lower() == effective_layer:
                items = FlextUtilitiesEnforcement._attr_items(
                    target,
                    effective_layer,
                )
        elif category is c.EnforcementCategory.NAMESPACE:
            items = FlextUtilitiesEnforcement._namespace_items(
                target,
                tag,
                effective_layer,
            )
        elif (
            category is c.EnforcementCategory.PROTOCOL_TREE
            and effective_layer == c.EnforcementLayer.PROTOCOLS.lower()
        ):
            items = walk(target, target.__qualname__)

        yield from items

    @staticmethod
    def _check(
        target: type,
        *,
        layer: str | None = None,
        categories: frozenset[c.EnforcementCategory] | None = None,
    ) -> me.Report:
        """Query applicable rules and return a typed report (no emission).

        Every rule dispatches through the unified :meth:`_apply_rule` —
        no per-category engine duplication; item iterators live in the
        ``_*_items`` / :meth:`_items_for` helpers and vary only by tag.
        Attr-rule recursion is handled via ``c.ENFORCEMENT_RECURSIVE_TAGS``.
        """
        violations: list[me.Violation] = []
        effective_layer = layer or FlextUtilitiesEnforcement.detect_layer(target) or ""
        qn = target.__qualname__
        for tag, category in c.ENFORCEMENT_TAG_CATEGORY.items():
            if categories is not None and category not in categories:
                continue
            rule_layer = c.ENFORCEMENT_TAG_LAYER.get(tag, "")
            items = FlextUtilitiesEnforcement._items_for(
                target,
                tag,
                category,
                effective_layer,
            )
            violations.extend(
                FlextUtilitiesEnforcement._apply_rule(
                    target,
                    tag,
                    qn,
                    items,
                    category,
                ),
            )
            if (
                category is c.EnforcementCategory.ATTR
                and tag in c.ENFORCEMENT_RECURSIVE_TAGS
                and rule_layer.lower() == effective_layer
            ):
                for _name, inner in FlextUtilitiesEnforcement._iter_inner(target):
                    if isinstance(inner, EnumType):
                        continue
                    violations.extend(
                        FlextUtilitiesEnforcement.check(
                            inner,
                            layer=effective_layer,
                        ).violations,
                    )
        return me.Report(violations=violations)

    @staticmethod
    def check(target: type, *, layer: str | None = None) -> me.Report:
        """Query all applicable rules and return a typed report (no emission)."""
        return FlextUtilitiesEnforcement._check(target, layer=layer)

    @staticmethod
    def check_model_construction(target: type[mp.BaseModel]) -> me.Report:
        """Run only Pydantic construction rules for ``__pydantic_init_subclass__``."""
        return FlextUtilitiesEnforcement._check(
            target,
            categories=FlextUtilitiesEnforcement._MODEL_CONSTRUCTION_CATEGORIES,
        )


__all__: list[str] = ["FlextUtilitiesEnforcement"]
