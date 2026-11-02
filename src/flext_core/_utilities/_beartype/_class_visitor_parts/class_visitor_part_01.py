"""Class placement, MRO, and protocol tree governance.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import sys
from enum import EnumType

from flext_core import c
from flext_core._models import FlextModelsEnforcement, FlextModelsPydantic
from flext_core._typings import FlextTypingBase
from flext_core._utilities._beartype import (
    FlextUtilitiesBeartypeHelpers,
    FlextUtilitiesBeartypeModuleSource,
)

NO_VIOLATION: FlextTypingBase.StrMapping | None = None
BARE_VIOLATION: FlextTypingBase.StrMapping = {}
BINARY_ARITY: int = 2


class FlextUtilitiesBeartypeClassVisitor:
    """CLASS_PLACEMENT + PROTOCOL_TREE + MRO_SHAPE + LOOSE_SYMBOL visitors."""

    @staticmethod
    def model_state_service_first(
        target: type,
        package_name: str,
        alias_rows: tuple[tuple[str, str, str], ...],
    ) -> bool:
        """Recognize model-owned state precedence in the published service facade.

        Pydantic merges direct-base fields right to left. A project's declarative
        state therefore precedes its native service, unlike operational peers.

        Returns:
            Whether the first base owns declarations for the native service role.

        """
        if len(target.__bases__) != BINARY_ARITY or not any(
            alias == "s" and owner == target.__module__
            for alias, owner, _ in alias_rows
        ):
            return False
        state, service = target.__bases__
        model_owners = {
            f"{package_name}._{owner.rsplit('.', 1)[-1]}"
            for alias, owner, _ in alias_rows
            if alias == "m" and owner.startswith(f"{package_name}.")
        }
        if not any(
            state.__module__ == owner or state.__module__.startswith(f"{owner}.")
            for owner in model_owners
        ) or not issubclass(state, FlextModelsPydantic.BaseModel):
            return False
        # Read the loaded native role, not a class-name lookalike. Importing the
        # service while foundation models are being declared would create a cycle.
        native_package = FlextModelsPydantic.__module__.split(".", 1)[0]
        native_service = next(
            (
                vars(sys.modules[path]).get(alias)
                for alias, path, _ in FlextUtilitiesBeartypeHelpers.lazy_alias_suffixes(
                    native_package
                )
                if alias == "s" and path in sys.modules
            ),
            None,
        )
        if (
            not isinstance(native_service, type)
            or native_service not in service.__mro__
            or native_service in state.__mro__
            or not state.model_fields
        ):
            return False
        for owner in state.__mro__:
            if owner in FlextModelsPydantic.BaseModel.__mro__ or (
                owner in service.__mro__
                and owner.__module__.split(".", 1)[0] == native_package
            ):
                continue
            if not any(
                owner.__module__ == origin or owner.__module__.startswith(f"{origin}.")
                for origin in model_owners
            ) or not issubclass(owner, FlextModelsPydantic.BaseModel):
                return False
            decorators = owner.__pydantic_decorators__
            declarations = {
                *decorators.field_validators,
                *decorators.model_validators,
                *decorators.field_serializers,
                *decorators.model_serializers,
                *decorators.computed_fields,
            }
            if any(
                name not in declarations
                and (
                    inspect.isfunction(value)
                    or isinstance(value, (property, classmethod, staticmethod))
                )
                for name, value in vars(owner).items()
            ):
                return False
        return True

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
                and layer in c.ENFORCEMENT_LAYER_ALLOWS
            ):
                allowed = c.ENFORCEMENT_LAYER_ALLOWS.get(
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
