"""Type introspection helpers for dispatcher handler compatibility.

The utilities here mirror the logic previously embedded in ``h``
to keep handler initialization lighter while still honoring the dispatcher
protocol expectations for message typing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableSequence
from typing import cast, get_args, get_origin

from pydantic import BaseModel

from flext_core import FlextResult, c, p
from flext_core._protocols import FlextProtocolsBase
from flext_core._typings import FlextTypesServices, FlextTypingBase
from flext_core._utilities._checker_parts import checker_part_02


class FlextUtilitiesChecker(checker_part_02.FlextUtilitiesChecker):
    @classmethod
    def _extract_generic_message_types(
        cls,
        handler_class: type,
    ) -> FlextTypingBase.SequenceOf[FlextTypingBase.TypeHintSpecifier]:
        """Extract message types from generic base annotations.

        Returns:
            The resulting sequence of type-hint specifiers.

        """
        raw_bases: (
            FlextTypingBase.VariadicTuple[FlextTypingBase.TypeHintSpecifier] | tuple[()]
        ) = getattr(
            handler_class,
            "__orig_bases__",
            (),
        )
        generic_bases: FlextTypingBase.VariadicTuple[
            FlextTypingBase.TypeHintSpecifier
        ] = raw_bases
        message_types: MutableSequence[FlextTypingBase.TypeHintSpecifier] = [
            args[0]
            for base in generic_bases
            if (origin := get_origin(base)) is not None
            if getattr(origin, "__name__", "") in c.CHECKER_HANDLER_ORIGIN_NAMES
            if (args := get_args(base))
        ]
        if message_types:
            return message_types
        metadata_types: MutableSequence[FlextTypingBase.TypeHintSpecifier] = []
        for parent_cls in handler_class.__bases__:
            metadata: (
                FlextTypingBase.MappingKV[
                    str,
                    FlextTypingBase.TypeHintSpecifier
                    | FlextTypingBase.VariadicTuple[FlextTypingBase.TypeHintSpecifier]
                    | None,
                ]
                | None
            ) = getattr(
                parent_cls,
                "__pydantic_generic_metadata__",
                None,
            )
            if not isinstance(metadata, Mapping):
                continue
            origin = metadata.get("origin")
            if getattr(origin, "__name__", "") not in c.CHECKER_HANDLER_ORIGIN_NAMES:
                continue
            args = metadata.get("args")
            if isinstance(args, tuple) and args:
                metadata_types.append(args[0])
        return metadata_types

    @classmethod
    def _extract_message_type_from_handle(
        cls,
        handler_class: type,
    ) -> p.Result[FlextTypingBase.TypeHintSpecifier]:
        """Extract message type from handle method annotations when generics are absent.

        Returns:
            The resulting ``p.Result[FlextTypingBase.TypeHintSpecifier]``.

        """
        if not hasattr(handler_class, c.MethodName.HANDLE):
            return FlextResult[FlextTypingBase.TypeHintSpecifier].fail(
                c.ERR_CHECKER_HANDLER_NO_HANDLE_METHOD
            )
        handle_method_raw: p.GuardInput | None = getattr(
            handler_class,
            c.MethodName.HANDLE,
            None,
        )
        if not cls._is_module_export_callable(handle_method_raw):
            return FlextResult[FlextTypingBase.TypeHintSpecifier].fail(
                c.ERR_CHECKER_HANDLER_HANDLE_NOT_CALLABLE,
            )
        handle_method = cast(
            "Callable[..., FlextTypesServices.ModuleExport]",
            handle_method_raw,
        )
        signature_result = cls._get_method_signature(handle_method)
        if signature_result.failure:
            return FlextResult[FlextTypingBase.TypeHintSpecifier].from_failure(
                signature_result
            )
        signature = signature_result.unwrap()
        type_hints = cls._get_type_hints_safe(handle_method, handler_class)
        for name, parameter in signature.parameters.items():
            if name == c.FRAME_SELF_KEY:
                continue
            return cls._extract_message_type_from_parameter(parameter, type_hints, name)
        return FlextResult[FlextTypingBase.TypeHintSpecifier].fail(
            c.ERR_CHECKER_NO_MESSAGE_PARAMETER
        )

    @classmethod
    def can_handle_message_type(
        cls,
        accepted_types: FlextTypingBase.VariadicTuple[
            FlextTypingBase.TypeHintSpecifier
        ],
        message_type: FlextTypesServices.MessageTypeSpecifier | None,
    ) -> bool:
        """Check if handler can process this message type.

        Returns:
            The resulting ``bool``.

        """
        if not accepted_types or message_type is None:
            return False
        for expected_type in accepted_types:
            if cls._evaluate_type_compatibility(expected_type, message_type):
                return True
        return False

    @classmethod
    def compute_accepted_message_types(
        cls,
        handler_class: type,
    ) -> FlextTypingBase.VariadicTuple[FlextTypingBase.TypeHintSpecifier]:
        """Compute message types accepted by a handler using cached introspection.

        Returns:
            The resulting tuple of type-hint specifiers.

        """
        message_types: MutableSequence[FlextTypingBase.TypeHintSpecifier] = []
        generic_types = cls._extract_generic_message_types(handler_class)
        message_types.extend(generic_types)
        if not message_types:
            explicit_type_result = cls._extract_message_type_from_handle(handler_class)
            if explicit_type_result.success:
                message_types.append(explicit_type_result.unwrap())
        return tuple(message_types)

    @classmethod
    def resolve_message_route(
        cls, msg: FlextProtocolsBase.Routable | type[FlextProtocolsBase.Routable] | str
    ) -> str:
        """Resolve route name from Routable attributes or string.

        Returns:
            The resulting ``str``.

        Raises:
            TypeError: If message does not provide a valid route.

        """
        if isinstance(msg, str):
            return msg
        route_attrs = ("command_type", "query_type", "event_type")
        for attr in route_attrs:
            attr_val: FlextTypingBase.StrictStr | None = getattr(msg, attr, None)
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
        msg_type_error = (
            f"Message {msg} does not provide a valid route via "
            "command_type, query_type, or event_type"
        )
        raise TypeError(msg_type_error)


__all__: list[str] = ["FlextUtilitiesChecker"]
