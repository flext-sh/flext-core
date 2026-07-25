"""Type introspection helpers for dispatcher handler compatibility.

The utilities here mirror the logic previously embedded in ``h``
to keep handler initialization lighter while still honoring the dispatcher
protocol expectations for message typing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, get_origin, get_type_hints

from flext_core import FlextConstants as c
from flext_core._protocols.result import FlextProtocolsResult as p
from flext_core._typings.base import FlextTypingBase as tb
from flext_core._typings.services import FlextTypesServices as ts
from flext_core.result import FlextResult as r

from .checker_part_01 import (
    FlextUtilitiesChecker as FlextUtilitiesCheckerPart01,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


class FlextUtilitiesChecker(FlextUtilitiesCheckerPart01):
    @classmethod
    def _extract_message_type_from_parameter(
        cls,
        parameter: inspect.Parameter,
        type_hints: tb.MappingKV[str, tb.TypeHintSpecifier | None],
        param_name: str,
    ) -> p.Result[tb.TypeHintSpecifier]:
        """Extract message type from parameter hints or signature annotation."""
        if param_name in type_hints:
            hint = type_hints[param_name]
            if hint is None:
                return r[tb.TypeHintSpecifier].fail(c.ERR_CHECKER_TYPE_HINT_NONE)
            return r[tb.TypeHintSpecifier].ok(
                hint if isinstance(hint, (str, type)) else str(hint),
            )
        annotation = parameter.annotation
        if annotation is inspect.Signature.empty:
            return r[tb.TypeHintSpecifier].fail(
                c.ERR_CHECKER_NO_ANNOTATION_OR_TYPE_HINT,
            )
        return r[tb.TypeHintSpecifier].ok(
            annotation if isinstance(annotation, (str, type)) else str(annotation),
        )

    @classmethod
    def _get_method_signature(
        cls,
        handle_method: Callable[..., ts.ModuleExport],
    ) -> p.Result[inspect.Signature]:
        """Extract signature from handle method, wrapping errors in Result."""
        try:
            return r[inspect.Signature].ok(inspect.signature(handle_method))
        except c.EXC_TYPE_VALIDATION:
            return r[inspect.Signature].fail(
                c.ERR_CHECKER_INVALID_HANDLE_METHOD_SIGNATURE,
            )

    @classmethod
    def _get_type_hints_safe(
        cls,
        handle_method: Callable[..., ts.ModuleExport],
        handler_class: type,
    ) -> tb.MappingKV[str, tb.TypeHintSpecifier | None]:
        """Safely extract type hints, returning empty dict on error."""
        try:
            return get_type_hints(
                handle_method,
                globalns=handle_method.__globals__,
                localns=dict(vars(handler_class)),
            )
        except (NameError, AttributeError, TypeError):
            return {}

    @classmethod
    def _handle_instance_check(
        cls,
        message_type: tb.TypeHintSpecifier,
        origin_type: tb.TypeHintSpecifier,
    ) -> bool:
        """Instance check for non-type objects; returns True on TypeError."""
        try:
            if isinstance(origin_type, type):
                return isinstance(message_type, origin_type) or cls._is_subclass_of(
                    message_type,
                    origin_type,
                )
            return True
        except TypeError:
            return True

    @classmethod
    def _handle_type_or_origin_check(
        cls,
        expected_type: tb.TypeHintSpecifier,
        message_type: tb.TypeHintSpecifier,
        origin_type: tb.TypeHintSpecifier,
        message_origin: tb.TypeHintSpecifier,
    ) -> bool:
        """Type checking for types or objects with __origin__."""
        try:
            if hasattr(message_type, "__origin__"):
                return message_origin is origin_type
            if isinstance(origin_type, type):
                return cls._is_subclass_of(message_type, origin_type)
            return message_type is expected_type
        except TypeError:
            return message_type is expected_type

    @classmethod
    def _evaluate_type_compatibility(
        cls,
        expected_type: tb.TypeHintSpecifier,
        message_type: ts.MessageTypeSpecifier,
    ) -> bool:
        """Evaluate compatibility between expected and actual message types."""
        object_check = cls._check_object_type_compatibility(expected_type)
        if object_check:
            return object_check
        origin_type = get_origin(expected_type) or expected_type
        message_origin = get_origin(message_type) or message_type
        dict_check = cls._check_dict_compatibility(
            expected_type,
            message_type,
            origin_type,
            message_origin,
        )
        if dict_check:
            return dict_check
        if isinstance(message_type, type) or hasattr(message_type, "__origin__"):
            return cls._handle_type_or_origin_check(
                expected_type,
                message_type,
                origin_type,
                message_origin,
            )
        return cls._handle_instance_check(message_type, origin_type)


__all__: list[str] = ["FlextUtilitiesChecker"]
