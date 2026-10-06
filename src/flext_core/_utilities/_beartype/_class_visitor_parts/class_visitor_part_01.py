"""Class placement, MRO, and protocol tree governance.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import EnumType

from flext_core._constants.enforcement import FlextConstantsEnforcement
from flext_core._models.enforcement import FlextModelsEnforcement
from flext_core._typings.base import FlextTypingBase
from flext_core._utilities._beartype.helpers import FlextUtilitiesBeartypeHelpers
from flext_core._utilities._beartype.module_source import (
    FlextUtilitiesBeartypeModuleSource,
)

NO_VIOLATION: FlextTypingBase.StrMapping | None = None
BARE_VIOLATION: FlextTypingBase.StrMapping = {}
BINARY_ARITY: int = 2


class FlextUtilitiesBeartypeClassVisitor:
    """CLASS_PLACEMENT + PROTOCOL_TREE + MRO_SHAPE + LOOSE_SYMBOL visitors."""

    @staticmethod
    def v_class_placement(
        params: FlextModelsEnforcement.ClassPlacementParams,
        *args: type | str,
    ) -> FlextTypingBase.StrMapping | None:
        """CLASS_PLACEMENT — class-name / inner-class layer placement.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        violation = NO_VIOLATION
        match args:
            case (value, layer) if (
                isinstance(value, type)
                and isinstance(layer, str)
                and layer in FlextConstantsEnforcement.ENFORCEMENT_LAYER_ALLOWS
            ):
                allowed = FlextConstantsEnforcement.ENFORCEMENT_LAYER_ALLOWS.get(
                    layer,
                    frozenset(),
                )
                forbidden_base_matches = (
                    ("StrEnum", isinstance(value, EnumType)),
                    (
                        "Protocol",
                        FlextUtilitiesBeartypeHelpers.has_runtime_protocol_marker(
                            value,
                        ),
                    ),
                )
                if any(
                    base_name in params.forbidden_bases
                    and is_match
                    and base_name not in allowed
                    for base_name, is_match in forbidden_base_matches
                ):
                    violation = BARE_VIOLATION
            case (root, nested) if (
                isinstance(root, type)
                and isinstance(nested, type)
                and params.max_nested_class_depth
            ):
                depth = nested.__qualname__.count(".") - root.__qualname__.count(".")
                violation = (
                    {"qn": nested.__qualname__}
                    if depth > params.max_nested_class_depth
                    else NO_VIOLATION
                )
            case (target, expected) if isinstance(target, type) and isinstance(
                expected,
                str,
            ):
                if params.check_nested:
                    parts = target.__qualname__.split(".")
                    has_wrong_nested_prefix = all((
                        len(parts)
                        >= FlextConstantsEnforcement.ENFORCEMENT_NESTED_MRO_MIN_DEPTH,
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
    def v_protocol_tree(
        params: FlextModelsEnforcement.ProtocolTreeParams,
        value: type,
    ) -> FlextTypingBase.StrMapping | None:
        """PROTOCOL_TREE — inner-class kind + runtime_checkable governance.

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        if params.require_inner_kind_protocol_or_namespace:
            if (
                FlextUtilitiesBeartypeHelpers.has_runtime_protocol_marker(value)
                or FlextUtilitiesBeartypeHelpers.has_nested_namespace(value)
                or FlextUtilitiesBeartypeHelpers.has_abstract_contract(value)
                or FlextUtilitiesBeartypeHelpers.has_protocol_ancestor(value)
            ):
                pass
            else:
                return BARE_VIOLATION
        if (
            params.require_runtime_checkable
            and FlextUtilitiesBeartypeHelpers.has_runtime_protocol_marker(value)
            and not FlextUtilitiesBeartypeModuleSource.declares_runtime_checkable(value)
        ):
            return BARE_VIOLATION
        return NO_VIOLATION


__all__: list[str] = ["FlextUtilitiesBeartypeClassVisitor"]
