"""Type introspection helpers for dispatcher handler compatibility.

The utilities here mirror the logic previously embedded in ``h``
to keep handler initialization lighter while still honoring the dispatcher
protocol expectations for message typing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence
from typing import TYPE_CHECKING, get_args, get_origin

from pydantic import BaseModel

from flext_core import FlextConstants as c, FlextResult as r
from flext_core._typings.base import FlextTypingBase as tb

# mro-cqxy (ADR-011): runtime import for runtime-evaluated annotation
from flext_core._typings.services import FlextTypesServices as ts

from .checker_part_02 import FlextUtilitiesChecker as FlextUtilitiesCheckerPart02

if TYPE_CHECKING:
    from flext_core._protocols.base import FlextProtocolsBase as pb
    from flext_core._protocols.result import FlextProtocolsResult as p


class FlextUtilitiesChecker(FlextUtilitiesCheckerPart02):
    @classmethod
    def _extract_generic_message_types(
        cls, handler_class: type
    ) -> tb.SequenceOf[tb.TypeHintSpecifier]:
        """Extract message types from generic base annotations."""
        raw_bases: tb.VariadicTuple[tb.TypeHintSpecifier] | tuple[()] = getattr(
            handler_class, "__orig_bases__", ()
        )
        generic_bases: tb.VariadicTuple[tb.TypeHintSpecifier] = raw_bases
        message_types: MutableSequence[tb.TypeHintSpecifier] = [
            args[0]
            for base in generic_bases
            if (origin := get_origin(base)) is not None
            if getattr(origin, "__name__", "") in c.CHECKER_HANDLER_ORIGIN_NAMES
            if (args := get_args(base))
        ]
        if message_types:
            return message_types
        return [
            args[0]
            for parent_cls in handler_class.__bases__
            if isinstance(
                (meta := getattr(parent_cls, "__pydantic_generic_metadata__", None)),
                Mapping,
            )
            if (origin := meta.get("origin")) is not None
            if getattr(origin, "__name__", "") in c.CHECKER_HANDLER_ORIGIN_NAMES
            if (args := meta.get("args", ()))
        ]

    @classmethod
    def _extract_message_type_from_handle(
        cls, handler_class: type
    ) -> p.Result[tb.TypeHintSpecifier]:
        """Extract message type from handle method annotations when generics are absent."""
        if not hasattr(handler_class, c.MethodName.HANDLE):
            return r[tb.TypeHintSpecifier].fail(c.ERR_CHECKER_HANDLER_NO_HANDLE_METHOD)
        handle_method_raw: ts.GuardInput | None = getattr(
            handler_class, c.MethodName.HANDLE, None
        )
        if not cls._is_module_export_callable(handle_method_raw):
            return r[tb.TypeHintSpecifier].fail(
                c.ERR_CHECKER_HANDLER_HANDLE_NOT_CALLABLE
            )
        signature_result = cls._get_method_signature(handle_method_raw)
        if signature_result.failure:
            return r[tb.TypeHintSpecifier].from_failure(signature_result)
        signature = signature_result.unwrap()
        type_hints = cls._get_type_hints_safe(handle_method_raw, handler_class)
        for name, parameter in signature.parameters.items():
            if name == c.FRAME_SELF_KEY:
                continue
            return cls._extract_message_type_from_parameter(parameter, type_hints, name)
        return r[tb.TypeHintSpecifier].fail(c.ERR_CHECKER_NO_MESSAGE_PARAMETER)

    @classmethod
    def can_handle_message_type(
        cls,
        accepted_types: tb.VariadicTuple[tb.TypeHintSpecifier],
        message_type: ts.MessageTypeSpecifier | None,
    ) -> bool:
        """Check if handler can process this message type."""
        if not accepted_types or message_type is None:
            return False
        for expected_type in accepted_types:
            if cls._evaluate_type_compatibility(expected_type, message_type):
                return True
        return False

    @classmethod
    def compute_accepted_message_types(
        cls, handler_class: type
    ) -> tb.VariadicTuple[tb.TypeHintSpecifier]:
        """Compute message types accepted by a handler using cached introspection."""
        message_types: MutableSequence[tb.TypeHintSpecifier] = []
        generic_types = cls._extract_generic_message_types(handler_class)
        message_types.extend(generic_types)
        if not message_types:
            explicit_type_result = cls._extract_message_type_from_handle(handler_class)
            if explicit_type_result.success:
                message_types.append(explicit_type_result.unwrap())
        return tuple(message_types)

    @classmethod
    def resolve_message_route(cls, msg: pb.Routable | type[pb.Routable] | str) -> str:
        """Resolve route name from Routable attributes or string.

        Raises:
            TypeError: If message does not provide a valid route.

        """
        if isinstance(msg, str):
            return msg
        route_attrs = ("command_type", "query_type", "event_type")
        for attr in route_attrs:
            attr_val: tb.StrictStr | None = getattr(msg, attr, None)
            if isinstance(attr_val, str) and attr_val:
                return attr_val
        if isinstance(msg, type) and issubclass(msg, BaseModel):
            for attr in route_attrs:
                if attr in msg.model_fields:
                    field_info = msg.model_fields[attr]
                    default_val = field_info.default
                    if (
                        isinstance(default_val, str)
                        and default_val
                        and default_val != "PydanticUndefined"
                    ):
                        return default_val
        msg_type_error = f"Message {msg} does not provide a valid route via command_type, query_type, or event_type"
        raise TypeError(msg_type_error)


__all__: list[str] = ["FlextUtilitiesChecker"]
