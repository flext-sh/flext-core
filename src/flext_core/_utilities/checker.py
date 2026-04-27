"""Type introspection helpers for dispatcher handler compatibility.

The utilities here mirror the logic previously embedded in ``h``
to keep handler initialization lighter while still honoring the dispatcher
protocol expectations for message typing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
from collections.abc import (
    Callable,
    Mapping,
    MutableSequence,
    Sequence,
)
from typing import TypeIs, get_args, get_origin, get_type_hints

from pydantic import BaseModel

from flext_core import FlextUtilitiesGuards, c, p, r, t


class FlextUtilitiesChecker:
    """Handler type checking utilities for h complexity reduction.

    Extracts type introspection and compatibility logic from h
    to simplify handler initialization and provide reusable type checking.
    """

    @staticmethod
    def _is_module_export_callable(
        value: Callable[..., t.ModuleExport] | t.GuardInput | None,
    ) -> TypeIs[Callable[..., t.ModuleExport]]:
        """Narrow value to a callable returning module exports.

        Excludes ``type`` objects (classes are callable but are not the
        bound/free functions we expect as handle methods).
        """
        return callable(value) and not isinstance(value, type)

    @staticmethod
    def _is_subclass_of(candidate: t.TypeHintSpecifier, parent: type) -> bool:
        """Safe subclass check that never raises TypeError."""
        return isinstance(candidate, type) and issubclass(candidate, parent)

    @classmethod
    def _is_dict_type(cls, candidate: t.TypeHintSpecifier) -> bool:
        """Check if candidate is dict or a subclass of dict."""
        return cls._is_subclass_of(candidate, dict)

    @classmethod
    def _check_dict_compatibility(
        cls,
        expected_type: t.TypeHintSpecifier,
        message_type: t.MessageTypeSpecifier,
        origin_type: t.TypeHintSpecifier,
        message_origin: t.TypeHintSpecifier,
    ) -> bool:
        """Check dict type compatibility between expected and message types."""
        origin_is_dict = cls._is_dict_type(origin_type)
        message_origin_is_dict = cls._is_dict_type(message_origin)
        if origin_is_dict and (
            message_origin_is_dict or cls._is_dict_type(message_type)
        ):
            return True
        return cls._is_dict_type(message_type) and (
            origin_is_dict or cls._is_dict_type(expected_type)
        )

    @classmethod
    def _check_object_type_compatibility(
        cls,
        expected_type: t.TypeHintSpecifier,
    ) -> bool:
        """Check if expected type is a canonical catch-all value contract."""
        return expected_type is t.JsonPayload

    @classmethod
    def _evaluate_type_compatibility(
        cls,
        expected_type: t.TypeHintSpecifier,
        message_type: t.MessageTypeSpecifier,
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

    @classmethod
    def _extract_generic_message_types(
        cls,
        handler_class: type,
    ) -> Sequence[t.TypeHintSpecifier]:
        """Extract message types from generic base annotations."""
        raw_bases: t.GuardInput = getattr(
            handler_class,
            "__orig_bases__",
            (),
        )
        generic_bases = (
            raw_bases if FlextUtilitiesGuards.object_tuple(raw_bases) else ()
        )
        message_types: MutableSequence[t.TypeHintSpecifier] = [
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
        cls,
        handler_class: type,
    ) -> p.Result[t.TypeHintSpecifier]:
        """Extract message type from handle method annotations when generics are absent."""
        if not hasattr(handler_class, c.MethodName.HANDLE):
            return r[t.TypeHintSpecifier].fail(
                c.ERR_CHECKER_HANDLER_NO_HANDLE_METHOD,
            )
        handle_method_raw: t.GuardInput | None = getattr(
            handler_class,
            c.MethodName.HANDLE,
            None,
        )
        if not cls._is_module_export_callable(handle_method_raw):
            return r[t.TypeHintSpecifier].fail(
                c.ERR_CHECKER_HANDLER_HANDLE_NOT_CALLABLE,
            )
        signature_result = cls._get_method_signature(handle_method_raw)
        if signature_result.failure:
            signature_error = signature_result.error or "Invalid handle signature"
            return r[t.TypeHintSpecifier].fail(signature_error)
        signature = signature_result.unwrap()
        type_hints = cls._get_type_hints_safe(handle_method_raw, handler_class)
        for name, parameter in signature.parameters.items():
            if name == c.FRAME_SELF_KEY:
                continue
            return cls._extract_message_type_from_parameter(parameter, type_hints, name)
        return r[t.TypeHintSpecifier].fail(c.ERR_CHECKER_NO_MESSAGE_PARAMETER)

    @classmethod
    def _extract_message_type_from_parameter(
        cls,
        parameter: inspect.Parameter,
        type_hints: Mapping[str, t.TypeHintSpecifier | None],
        param_name: str,
    ) -> p.Result[t.TypeHintSpecifier]:
        """Extract message type from parameter hints or signature annotation."""
        if param_name in type_hints:
            hint = type_hints[param_name]
            if hint is None:
                return r[t.TypeHintSpecifier].fail(c.ERR_CHECKER_TYPE_HINT_NONE)
            if isinstance(hint, str):
                return r[t.TypeHintSpecifier].ok(hint)
            if isinstance(hint, type):
                return r[t.TypeHintSpecifier].ok(hint)
            return r[t.TypeHintSpecifier].ok(str(hint))
        annotation = parameter.annotation
        if annotation is not inspect.Signature.empty:
            if isinstance(annotation, str):
                return r[t.TypeHintSpecifier].ok(annotation)
            if isinstance(annotation, type):
                return r[t.TypeHintSpecifier].ok(annotation)
            return r[t.TypeHintSpecifier].ok(str(annotation))
        return r[t.TypeHintSpecifier].fail(
            c.ERR_CHECKER_NO_ANNOTATION_OR_TYPE_HINT,
        )

    @classmethod
    def _get_method_signature(
        cls,
        handle_method: Callable[..., t.ModuleExport],
    ) -> p.Result[inspect.Signature]:
        """Extract signature from handle method, wrapping errors in Result."""
        try:
            return r[inspect.Signature].ok(inspect.signature(handle_method))
        except (TypeError, ValueError):
            return r[inspect.Signature].fail(
                c.ERR_CHECKER_INVALID_HANDLE_METHOD_SIGNATURE,
            )

    @classmethod
    def _get_type_hints_safe(
        cls,
        handle_method: Callable[..., t.ModuleExport],
        handler_class: type,
    ) -> Mapping[str, t.TypeHintSpecifier | None]:
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
        message_type: t.TypeHintSpecifier,
        origin_type: t.TypeHintSpecifier,
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
        expected_type: t.TypeHintSpecifier,
        message_type: t.TypeHintSpecifier,
        origin_type: t.TypeHintSpecifier,
        message_origin: t.TypeHintSpecifier,
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
    def can_handle_message_type(
        cls,
        accepted_types: t.VariadicTuple[t.TypeHintSpecifier],
        message_type: t.MessageTypeSpecifier | None,
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
        cls,
        handler_class: type,
    ) -> t.VariadicTuple[t.TypeHintSpecifier]:
        """Compute message types accepted by a handler using cached introspection."""
        message_types: MutableSequence[t.TypeHintSpecifier] = []
        generic_types = cls._extract_generic_message_types(handler_class)
        message_types.extend(generic_types)
        if not message_types:
            explicit_type_result = cls._extract_message_type_from_handle(handler_class)
            if explicit_type_result.success:
                message_types.append(explicit_type_result.unwrap())
        return tuple(message_types)

    @classmethod
    def resolve_message_route(cls, msg: p.Routable | type[p.Routable] | str) -> str:
        """Resolve route name from Routable attributes or string.

        Raises:
            TypeError: If message does not provide a valid route.

        """
        if isinstance(msg, str):
            return msg
        route_attrs = ("command_type", "query_type", "event_type")
        for attr in route_attrs:
            attr_val: t.StrictStr | None = getattr(msg, attr, None)
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


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesChecker"]
