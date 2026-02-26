"""Type introspection helpers for dispatcher handler compatibility.

The utilities here mirror the logic previously embedded in ``h``
to keep handler initialization lighter while still honoring the dispatcher
protocol expectations for message typing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import cast, get_origin, get_type_hints

from flext_core.constants import c
from flext_core.protocols import p
from flext_core.runtime import FlextRuntime
from flext_core.typings import t


class FlextUtilitiesChecker:
    """Handler type checking utilities for h complexity reduction.

    Extracts type introspection and compatibility logic from h
    to simplify handler initialization and provide reusable type checking.
    """

    @property
    def logger(self) -> p.Log.StructlogLogger:
        """Get logger instance using FlextRuntime (avoids circular imports).

        Returns structlog logger instance with all logging methods (debug, info, warning, error, etc).
        Uses same structure/config as FlextLogger but without circular import.
        """
        return cast("p.Log.StructlogLogger", FlextRuntime.get_logger(__name__))

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
            explicit_type: t.MessageTypeSpecifier | None = (
                cls._extract_message_type_from_handle(handler_class)
            )
            if explicit_type is not None:
                message_types.append(explicit_type)

        return tuple(message_types)

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
    def _get_method_signature(
        cls,
        handle_method: t.HandlerCallable,
    ) -> inspect.Signature | None:
        """Extract signature from handle method."""
        try:
            # HandlerCallable is always callable per type definition
            # Pass directly - inspect.signature accepts any callable
            return inspect.signature(handle_method)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _get_type_hints_safe(
        cls,
        handle_method: t.HandlerCallable,
        handler_class: type,
    ) -> Mapping[str, t.GuardInputValue]:
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
    def _extract_message_type_from_parameter(
        cls,
        parameter: inspect.Parameter,
        type_hints: Mapping[str, t.GuardInputValue],
        param_name: str,
    ) -> t.MessageTypeSpecifier | None:
        """Extract message type from parameter hints or annotation."""
        if param_name in type_hints:
            # Return the type hint directly (plain types, generic aliases, etc.)
            hint = type_hints[param_name]
            # Type narrowing: MessageTypeSpecifier = str | type[t.GuardInputValue]
            # Check what hint is and return appropriately
            if hint is None:
                return None
            # If hint is a string or a type, it's valid MessageTypeSpecifier
            if isinstance(hint, str):
                return hint
            if isinstance(hint, type):
                # Type narrowing: hint is type after type check
                # Return the type directly - type is a valid MessageTypeSpecifier component
                # The return value will be str (from annotation branch) or type object
                return hint
            # For other types (Sequence, Mapping), convert to string
            # string is valid MessageTypeSpecifier
            return str(hint)

        annotation = parameter.annotation
        if annotation is not inspect.Signature.empty:
            # Type narrowing: annotation exists and is not empty
            # Annotation could be str, type, or a generic alias
            if isinstance(annotation, str):
                return annotation
            if isinstance(annotation, type):
                return annotation
            # For generic aliases and other types, convert to string representation
            return str(annotation)

        return None

    @classmethod
    def _extract_message_type_from_handle(
        cls,
        handler_class: type,
    ) -> t.MessageTypeSpecifier | None:
        """Extract message type from handle method annotations when generics are absent.

        Args:
            handler_class: Handler class to analyze

        Returns:
            Message type from handle method or None

        """
        if not hasattr(handler_class, c.Mixins.METHOD_HANDLE):
            return None
        handle_method_raw = getattr(handler_class, c.Mixins.METHOD_HANDLE)
        if not callable(handle_method_raw):
            return None
        handle_method: t.HandlerCallable = handle_method_raw

        signature = cls._get_method_signature(handle_method)
        if signature is None:
            return None

        type_hints = cls._get_type_hints_safe(handle_method, handler_class)

        for name, parameter in signature.parameters.items():
            if name == "self":
                continue

            return cls._extract_message_type_from_parameter(parameter, type_hints, name)

        return None

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
    def _check_object_type_compatibility(
        cls,
        expected_type: t.TypeOriginSpecifier,
    ) -> bool | None:
        """Check if expected type is object (universal compatibility).

        Args:
            expected_type: Type to check for object compatibility

        Returns:
            True if object type (accepts everything), None if not object type

        """
        # object type should be compatible with everything
        if expected_type is object:
            return True

        # object type by name should be compatible with everything
        if hasattr(expected_type, "__name__") and expected_type.__name__ == "object":
            return True

        return None  # Not object type - continue checking

    @classmethod
    def _check_dict_compatibility(
        cls,
        expected_type: t.TypeOriginSpecifier,
        message_type: t.MessageTypeSpecifier,
        origin_type: t.TypeOriginSpecifier,
        message_origin: t.TypeOriginSpecifier,
    ) -> bool | None:
        """Check dict type compatibility.

        Args:
            expected_type: Expected type
            message_type: Message type
            origin_type: Origin of expected type
            message_origin: Origin of message type

        Returns:
            True if dict compatible, None if not dict types

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
        if (
            isinstance(message_type, type)
            and dict in message_type.__mro__
            and (
                origin_is_dict
                or (isinstance(expected_type, type) and dict in expected_type.__mro__)
            )
        ):
            return True

        return None  # Not dict compatibility - continue checking

    @classmethod
    def _evaluate_type_compatibility(
        cls,
        expected_type: t.TypeOriginSpecifier,
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
        if object_check is not None:
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
        if dict_check is not None:
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
    def _handle_type_or_origin_check(
        cls,
        expected_type: t.TypeOriginSpecifier,
        message_type: t.TypeOriginSpecifier,
        origin_type: t.TypeOriginSpecifier,
        message_origin: t.TypeOriginSpecifier,
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
    def _handle_instance_check(
        cls,
        message_type: t.TypeOriginSpecifier,
        origin_type: t.TypeOriginSpecifier,
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


__all__ = [
    "FlextUtilitiesChecker",
]
