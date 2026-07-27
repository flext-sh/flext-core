"""Class placement, MRO, and protocol tree governance."""

from __future__ import annotations

from enum import EnumType

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities._beartype.helpers import FlextUtilitiesBeartypeHelpers as ubh

NO_VIOLATION: t.StrMapping | None = None
BARE_VIOLATION: t.StrMapping = {}
BINARY_ARITY: int = 2


class FlextUtilitiesBeartypeClassVisitor:
    """CLASS_PLACEMENT + PROTOCOL_TREE + MRO_SHAPE + LOOSE_SYMBOL visitors."""

    @staticmethod
    def v_class_placement(
        params: me.ClassPlacementParams, *args: type | str
    ) -> t.StrMapping | None:
        """CLASS_PLACEMENT — class-name / inner-class layer placement."""
        violation = NO_VIOLATION
        match args:
            case (value, layer) if (
                isinstance(value, type)
                and isinstance(layer, str)
                and layer in c.ENFORCEMENT_LAYER_ALLOWS
            ):
                allowed = c.ENFORCEMENT_LAYER_ALLOWS.get(layer, frozenset())
                forbidden_base_matches = (
                    ("StrEnum", isinstance(value, EnumType)),
                    ("Protocol", ubh.has_runtime_protocol_marker(value)),
                )
                if any(
                    base_name in params.forbidden_bases
                    and is_match
                    and base_name not in allowed
                    for base_name, is_match in forbidden_base_matches
                ):
                    violation = BARE_VIOLATION
            case (target,) if (
                isinstance(target, type)
                and params.max_nested_class_depth
                and "[" not in target.__name__
            ):
                deep = FlextUtilitiesBeartypeClassVisitor._deep_nested(
                    target, params.max_nested_class_depth
                )
                violation = {"qn": deep} if deep else NO_VIOLATION
            case (target, expected) if isinstance(target, type) and isinstance(
                expected, str
            ):
                if params.check_nested:
                    parts = target.__qualname__.split(".")
                    has_wrong_nested_prefix = all((
                        len(parts) >= c.ENFORCEMENT_NESTED_MRO_MIN_DEPTH,
                        not parts[0].startswith(expected),
                    ))
                    violation = (
                        {"expected": expected}
                        if has_wrong_nested_prefix
                        else NO_VIOLATION
                    )
                else:
                    violation = (
                        {"expected": expected, "actual": target.__name__}
                        if not target.__name__.startswith(expected)
                        else NO_VIOLATION
                    )
            case _:
                pass
        return violation

    @staticmethod
    def _deep_nested(node: type, budget: int) -> str | None:
        """Return qualname of first locally-defined non-Enum class past ``budget``."""
        for value in vars(node).values():
            if not (
                isinstance(value, type)
                and not isinstance(value, EnumType)
                and getattr(value, "__qualname__", "").startswith(
                    f"{node.__qualname__}."
                )
            ):
                continue
            if budget == 0:
                return value.__qualname__
            found = FlextUtilitiesBeartypeClassVisitor._deep_nested(value, budget - 1)
            if found:
                return found
        return None

    @staticmethod
    def v_protocol_tree(
        params: me.ProtocolTreeParams, value: type
    ) -> t.StrMapping | None:
        """PROTOCOL_TREE — inner-class kind + runtime_checkable governance."""
        if params.require_inner_kind_protocol_or_namespace:
            if (
                ubh.has_runtime_protocol_marker(value)
                or ubh.has_nested_namespace(value)
                or ubh.has_abstract_contract(value)
            ):
                pass
            else:
                return BARE_VIOLATION
        if (
            params.require_runtime_checkable
            and ubh.has_runtime_protocol_marker(value)
            and not getattr(value, "_is_runtime_protocol", False)
        ):
            return BARE_VIOLATION
        return NO_VIOLATION


__all__: list[str] = ["FlextUtilitiesBeartypeClassVisitor"]
