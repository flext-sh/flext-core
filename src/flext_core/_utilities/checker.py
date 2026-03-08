"""Type introspection helpers for dispatcher handler compatibility.

The utilities here mirror the logic previously embedded in ``h``
to keep handler initialization lighter while still honoring the dispatcher
protocol expectations for message typing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import get_origin, get_type_hints

from flext_core import FlextRuntime, c, p, r, t


class FlextUtilitiesChecker:
    """Handler type checking utilities for h complexity reduction.

    Extracts type introspection and compatibility logic from h
    to simplify handler initialization and provide reusable type checking.
    """

    @property
    def logger(self) -> p.Log.StructlogLogger:
        """Get structlog logger via FlextRuntime (infrastructure-level, no FlextLogger)."""
        return FlextRuntime.get_logger(__name__)

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
        origin_is_dict = isinstance(origin_type, type) and dict in origin_type.__mro__
        message_origin_is_dict = (
            isinstance(message_origin, type) and dict in message_origin.__mro__
        )

        # Handle dict-like compatibility for runtime classes and generic aliases.
        # If expected is dict-like, accept dict-derived message types.
        if origin_is_dict and (
            message_origin_is_dict
            or (isinstance(message_type, type) and dict in message_type.__mro__)
        ):
            return True

        # If message is dict-like, and expected is also dict-like.
        return (
            isinstance(message_type, type)
            and dict in message_type.__mro__
            and (
                origin_is_dict
                or (isinstance(expected_type, type) and dict in expected_type.__mro__)
            )
        )

    @classmethod
    def _check_object_type_compatibility(
        cls,
        expected_type: t.TypeHintSpecifier,
    ) -> bool:
        """Check if expected type is object (universal compatibility).

        Args:
            expected_type: Type to check for object compatibility

        Returns:
            True if object type (accepts everything), False otherwise

        """
        # object type should be compatible with everything
        return expected_type is object

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
        # Check object type compatibility (universal)
        object_check = cls._check_object_type_compatibility(expected_type)
        if object_check:
            return object_check

        # Get type origins
        origin_type = get_origin(expected_type) or expected_type
        message_origin = get_origin(message_type) or message_type

        # Check dict compatibility
        dict_check = cls._check_dict_compatibility(
            expected_type,
            message_type,
            origin_type,
            message_origin,
        )
        if dict_check:
            return dict_check

        # Check type or origin
        if isinstance(message_type, type) or hasattr(message_type, "__origin__"):
            return cls._handle_type_or_origin_check(
                expected_type,
                message_type,
                origin_type,
                message_origin,
            )

        # Check instance
        return cls._handle_instance_check(message_type, origin_type)

    @classmethod
    def _extract_generic_message_types(
        cls,
        handler_class: type,
    ) -> list[t.MessageTypeSpecifier]:
        """Extract message types from generic base annotations.

        Args:
            handler_class: Handler class to analyze

        Returns:
            List of message types from generic annotations

        """
        message_types: list[t.MessageTypeSpecifier] = []
        orig_bases = (
            handler_class.__orig_bases__
            if hasattr(handler_class, "__orig_bases__")
            else ()
        )
        for base in orig_bases:
            # Layer 0.5: Use FlextRuntime for type introspection
            origin = get_origin(base)
            # Check by name to avoid circular import
            # Note: origin is FlextHandlers class, not the alias 'h'
            if origin and origin.__name__ in {"h", "FlextHandlers"}:
                # Use FlextRuntime.extract_generic_args() from Layer 0.5 (defined in runtime.pyi stub)
                args = FlextRuntime.extract_generic_args(base)
                # Accept all type forms: plain types, generic aliases, and
                # and string type references. The _evaluate_type_compatibility method
                # handles all these forms correctly.
                # args[0] is never None - extract_generic_args returns tuple of types/strings
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
        if not hasattr(handler_class, c.Mixins.METHOD_HANDLE):
            return r[t.MessageTypeSpecifier].fail("Handler has no handle method")
        handle_method_raw = getattr(handler_class, c.Mixins.METHOD_HANDLE)
        if not callable(handle_method_raw):
            return r[t.MessageTypeSpecifier].fail(
                "Handler handle attribute is not callable"
            )
        handle_method: Callable[..., object] = handle_method_raw

        signature_result = cls._get_method_signature(handle_method)
        if signature_result.is_failure:
            signature_error = signature_result.error or "Invalid handle signature"
            return r[t.MessageTypeSpecifier].fail(
                signature_error,
            )
        signature = signature_result.value

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
        type_hints: Mapping[str, t.ContainerValue],
        param_name: str,
    ) -> r[t.MessageTypeSpecifier]:
        """Extract message type from parameter hints or annotation."""
        if param_name in type_hints:
            # Return the type hint directly (plain types, generic aliases, etc.)
            hint = type_hints[param_name]
            # Type narrowing: MessageTypeSpecifier = str | type[t.ContainerValue]
            # Check what hint is and return appropriately
            if hint is None:
                return r[t.MessageTypeSpecifier].fail("Type hint is None")
            # If hint is a string or a type, it's valid MessageTypeSpecifier
            if isinstance(hint, str):
                return r[t.MessageTypeSpecifier].ok(hint)
            if isinstance(hint, type):
                # Type narrowing: hint is type after type check
                # Return the type directly - type is a valid MessageTypeSpecifier component
                # The return value will be str (from annotation branch) or type object
                return r[t.MessageTypeSpecifier].ok(hint)
            # For other types (Sequence, Mapping), convert to string
            # string is valid MessageTypeSpecifier
            return r[t.MessageTypeSpecifier].ok(str(hint))

        annotation = parameter.annotation
        if annotation is not inspect.Signature.empty:
            # Type narrowing: annotation exists and is not empty
            # Annotation could be str, type, or a generic alias
            if isinstance(annotation, str):
                return r[t.MessageTypeSpecifier].ok(annotation)
            if isinstance(annotation, type):
                return r[t.MessageTypeSpecifier].ok(annotation)
            # For generic aliases and other types, convert to string representation
            return r[t.MessageTypeSpecifier].ok(str(annotation))

        return r[t.MessageTypeSpecifier].fail(
            "No annotation or type hint for parameter"
        )

    @classmethod
    def _get_method_signature(
        cls,
        handle_method: Callable[..., object],
    ) -> r[inspect.Signature]:
        """Extract signature from handle method."""
        try:
            # HandlerCallable is always callable per type definition
            # Pass directly - inspect.signature accepts any callable
            return r[inspect.Signature].ok(inspect.signature(handle_method))
        except (TypeError, ValueError):
            return r[inspect.Signature].fail("Invalid handle method signature")

    @classmethod
    def _get_type_hints_safe(
        cls,
        handle_method: Callable[..., object],
        handler_class: type,
    ) -> Mapping[str, t.ContainerValue]:
        """Safely extract type hints from handle method."""
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
            message_type: Message type to check
            origin_type: Origin type to check against

        Returns:
            True if instance check passes

        """
        try:
            if isinstance(origin_type, type):
                return isinstance(message_type, origin_type) or (
                    isinstance(message_type, type)
                    and origin_type in message_type.__mro__
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
            expected_type: Expected type
            message_type: Message type
            origin_type: Origin of expected type
            message_origin: Origin of message type

        Returns:
            True if types are compatible

        """
        try:
            if hasattr(message_type, "__origin__"):
                return message_origin is origin_type
            if isinstance(message_type, type) and isinstance(origin_type, type):
                return origin_type in message_type.__mro__
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
        message_types: list[t.MessageTypeSpecifier] = []
        generic_types = cls._extract_generic_message_types(handler_class)
        # Extend with extracted generic types
        message_types.extend(generic_types)

        if not message_types:
            explicit_type_result = cls._extract_message_type_from_handle(handler_class)
            if explicit_type_result.is_success:
                message_types.append(explicit_type_result.value)

        return tuple(message_types)


__all__ = [
    "FlextUtilitiesChecker",
]
