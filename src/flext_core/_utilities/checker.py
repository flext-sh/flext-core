"""Type introspection helpers for dispatcher handler compatibility.

The utilities here mirror the logic previously embedded in ``h``
to keep handler initialization lighter while still honoring the dispatcher
protocol expectations for message typing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, MutableSequence, Sequence
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
        value: Callable[..., t.ModuleExport] | t.GuardInput,
    ) -> TypeIs[Callable[..., t.ModuleExport]]:
        """Check if value is a callable that returns module exports.

        Args:
            value: Value to check.

        Returns:
            True if value is callable, narrowed to Callable[..., ModuleExport].

        """
        return callable(value)

    @staticmethod
    def _is_subclass_of(candidate: t.TypeHintSpecifier, parent: type) -> bool:
        """Safe subclass check that never raises TypeError.

        Args:
            candidate: Potential subclass to check.
            parent: Parent class to check against.

        Returns:
            True if candidate is a class and subclass of parent, False otherwise.

        """
        return isinstance(candidate, type) and issubclass(candidate, parent)

    @classmethod
    def _is_dict_type(cls, candidate: t.TypeHintSpecifier) -> bool:
        """Check if candidate is dict or a subclass of dict.

        Args:
            candidate: Type to check.

        Returns:
            True if candidate is dict or dict subclass, False otherwise.

        """
        return cls._is_subclass_of(candidate, dict)

    @classmethod
    def _check_dict_compatibility(
        cls,
        expected_type: t.TypeHintSpecifier,
        message_type: t.MessageTypeSpecifier,
        origin_type: t.TypeHintSpecifier,
        message_origin: t.TypeHintSpecifier,
    ) -> bool:
        """Check dict type compatibility.

        Args:
            expected_type: Expected type
            message_type: Message type
            origin_type: Origin of expected type
            message_origin: Origin of message type

        Returns:
            True if dict compatible, False if not dict types

        """
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
        """Check if expected type is t.NormalizedValue (universal compatibility).

        Args:
            expected_type: Type to check for t.NormalizedValue compatibility

        Returns:
            True if t.NormalizedValue type (accepts everything), False otherwise

        """
        return expected_type is t.NormalizedValue or str(expected_type) == "typing.Any"

    @classmethod
    def _evaluate_type_compatibility(
        cls,
        expected_type: t.TypeHintSpecifier,
        message_type: t.MessageTypeSpecifier,
    ) -> bool:
        """Evaluate compatibility between expected and actual message types.

        Args:
            expected_type: Expected message type
            message_type: Actual message type

        Returns:
            True if types are compatible

        """
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
    ) -> Sequence[t.MessageTypeSpecifier]:
        """Extract message types from generic base annotations.

        Args:
            handler_class: Handler class to analyze

        Returns:
            List of message types from generic annotations

        """
        message_types: MutableSequence[t.MessageTypeSpecifier] = []
        raw_bases: t.GuardInput = getattr(
            handler_class,
            "__orig_bases__",
            (),
        )
        if FlextUtilitiesGuards.is_object_tuple(raw_bases):
            for base in raw_bases:
                origin = get_origin(base)
                if origin is not None and origin.__name__ in {"h", "FlextHandlers"}:
                    args = get_args(base)
                    if args:
                        message_types.append(args[0])
        if message_types:
            return message_types
        for parent_cls in handler_class.__bases__:
            meta = getattr(parent_cls, "__pydantic_generic_metadata__", None)
            if meta is None:
                continue
            origin = meta.get("origin")
            if origin is None:
                continue
            origin_name = getattr(origin, "__name__", "")
            if origin_name not in {"FlextHandlers", "h"}:
                continue
            args = meta.get("args", ())
            if args:
                message_types.append(args[0])
        return message_types

    @classmethod
    def _extract_message_type_from_handle(
        cls,
        handler_class: type,
    ) -> r[t.MessageTypeSpecifier]:
        """Extract message type from handle method annotations when generics are absent.

        Args:
            handler_class: Handler class to analyze

        Returns:
            r[t.MessageTypeSpecifier]: Success with message type or failure.

        """
        if not hasattr(handler_class, c.METHOD_HANDLE):
            return r[t.MessageTypeSpecifier].fail("Handler has no handle method")
        handle_method_raw: t.GuardInput = getattr(handler_class, c.METHOD_HANDLE, None)
        if not cls._is_module_export_callable(handle_method_raw):
            return r[t.MessageTypeSpecifier].fail(
                "Handler handle attribute is not callable",
            )
        handle_method: Callable[..., t.ModuleExport] = handle_method_raw
        signature_result = cls._get_method_signature(handle_method)
        if signature_result.is_failure:
            signature_error = signature_result.error or "Invalid handle signature"
            return r[t.MessageTypeSpecifier].fail(signature_error)
        signature = signature_result.unwrap()
        type_hints = cls._get_type_hints_safe(handle_method, handler_class)
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue
            return cls._extract_message_type_from_parameter(parameter, type_hints, name)
        return r[t.MessageTypeSpecifier].fail("No message parameter found in handle")

    @classmethod
    def _extract_message_type_from_parameter(
        cls,
        parameter: inspect.Parameter,
        type_hints: Mapping[str, t.TypeHintSpecifier | None],
        param_name: str,
    ) -> r[t.MessageTypeSpecifier]:
        """Extract message type from parameter hints or signature annotation.

        Args:
            parameter: Parameter to extract type from.
            type_hints: Type hints mapping for the function.
            param_name: Name of the parameter.

        Returns:
            r[MessageTypeSpecifier] with extracted type or failure message.

        """
        if param_name in type_hints:
            hint = type_hints[param_name]
            if hint is None:
                return r[t.MessageTypeSpecifier].fail("Type hint is None")
            if isinstance(hint, str):
                return r[t.MessageTypeSpecifier].ok(hint)
            if isinstance(hint, type):
                return r[t.MessageTypeSpecifier].ok(hint)
            return r[t.MessageTypeSpecifier].ok(str(hint))
        annotation = parameter.annotation
        if annotation is not inspect.Signature.empty:
            if isinstance(annotation, str):
                return r[t.MessageTypeSpecifier].ok(annotation)
            if isinstance(annotation, type):
                return r[t.MessageTypeSpecifier].ok(annotation)
            return r[t.MessageTypeSpecifier].ok(str(annotation))
        return r[t.MessageTypeSpecifier].fail(
            "No annotation or type hint for parameter",
        )

    @classmethod
    def _get_method_signature(
        cls,
        handle_method: Callable[..., t.ModuleExport],
    ) -> r[inspect.Signature]:
        """Extract signature from handle method.

        Args:
            handle_method: Method to extract signature from.

        Returns:
            r[Signature] with method signature or failure message.

        """
        try:
            return r[inspect.Signature].ok(inspect.signature(handle_method))
        except (TypeError, ValueError):
            return r[inspect.Signature].fail("Invalid handle method signature")

    @classmethod
    def _get_type_hints_safe(
        cls,
        handle_method: Callable[..., t.ModuleExport],
        handler_class: type,
    ) -> Mapping[str, t.TypeHintSpecifier | None]:
        """Safely extract type hints from handle method.

        Args:
            handle_method: Method to extract type hints from.
            handler_class: Handler class for local namespace context.

        Returns:
            Mapping of parameter names to type hints, empty dict on error.

        """
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
        """Handle instance checking for non-type objects.

        Args:
            message_type: Message type or instance to check.
            origin_type: Origin type to check against.

        Returns:
            True if instance or subclass check passes, True on TypeError.

        """
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
        """Handle type checking for types or objects with __origin__.

        Args:
            expected_type: Expected type.
            message_type: Message type or instance.
            origin_type: Origin of expected type.
            message_origin: Origin of message type.

        Returns:
            True if types are compatible or match, False otherwise.

        """
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
        accepted_types: tuple[t.MessageTypeSpecifier, ...],
        message_type: t.MessageTypeSpecifier,
    ) -> bool:
        """Check if handler can process this message type.

        Args:
            accepted_types: Types accepted by handler
            message_type: Type to check

        Returns:
            True if handler can process this message type

        """
        if not accepted_types:
            return False
        for expected_type in accepted_types:
            if cls._evaluate_type_compatibility(expected_type, message_type):
                return True
        return False

    @classmethod
    def compute_accepted_message_types(
        cls,
        handler_class: type,
    ) -> tuple[t.MessageTypeSpecifier, ...]:
        """Compute message types accepted by a handler using cached introspection.

        Args:
            handler_class: Handler class to analyze

        Returns:
            Tuple of accepted message types

        """
        message_types: MutableSequence[t.MessageTypeSpecifier] = []
        generic_types = cls._extract_generic_message_types(handler_class)
        message_types.extend(generic_types)
        if not message_types:
            explicit_type_result = cls._extract_message_type_from_handle(handler_class)
            if explicit_type_result.is_success:
                message_types.append(explicit_type_result.unwrap())
        return tuple(message_types)

    @classmethod
    def get_message_route(cls, msg: p.Routable | type[p.Routable] | str) -> str:
        """Resolve route name strictly from Routable attributes or string.

        Args:
            msg: Message instance, type, or explicit route string.

        Returns:
            str: Resolved route name.

        Raises:
            TypeError: If message does not provide a valid route.

        """
        if isinstance(msg, str):
            return msg
        route_attrs = ("command_type", "query_type", "event_type")
        for attr in route_attrs:
            attr_val: t.NormalizedValue | None = getattr(msg, attr, None)
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


__all__ = ["FlextUtilitiesChecker"]
