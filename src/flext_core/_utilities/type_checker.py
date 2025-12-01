"""Type introspection helpers for dispatcher handler compatibility.

The utilities here mirror the logic previously embedded in ``FlextHandlers``
to keep handler initialization lighter while still honoring the dispatcher
protocol expectations for message typing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import logging
from typing import cast, get_origin, get_type_hints

from flext_core.runtime import FlextRuntime, StructlogLogger
from flext_core.typings import FlextTypes

_logger = logging.getLogger(__name__)


class FlextUtilitiesTypeChecker:
    """Handler type checking utilities for FlextHandlers complexity reduction.

    Extracts type introspection and compatibility logic from FlextHandlers
    to simplify handler initialization and provide reusable type checking.
    """

    @property
    def logger(self) -> StructlogLogger:
        """Get logger instance using FlextRuntime (avoids circular imports).

        Returns structlog logger instance with all logging methods (debug, info, warning, error, etc).
        Uses same structure/config as FlextLogger but without circular import.
        """
        return FlextRuntime.get_logger(__name__)

    @classmethod
    def compute_accepted_message_types(
        cls,
        handler_class: type,
    ) -> tuple[FlextTypes.Utility.MessageTypeSpecifier, ...]:
        """Compute message types accepted by a handler using cached introspection.

        Args:
            handler_class: Handler class to analyze

        Returns:
            Tuple of accepted message types

        """
        message_types: list[FlextTypes.Utility.MessageTypeSpecifier] = []
        generic_types = cls._extract_generic_message_types(handler_class)
        # Extend with extracted generic types
        message_types.extend(generic_types)

        if not message_types:
            explicit_type: FlextTypes.Utility.MessageTypeSpecifier | None = (
                cls._extract_message_type_from_handle(handler_class)
            )
            if explicit_type is not None:
                message_types.append(explicit_type)

        return tuple(message_types)

    @classmethod
    def _extract_generic_message_types(
        cls,
        handler_class: type,
    ) -> list[FlextTypes.Utility.MessageTypeSpecifier]:
        """Extract message types from generic base annotations.

        Args:
            handler_class: Handler class to analyze

        Returns:
            List of message types from generic annotations

        """
        message_types: list[FlextTypes.Utility.MessageTypeSpecifier] = []
        for base in getattr(handler_class, "__orig_bases__", ()) or ():
            # Layer 0.5: Use FlextRuntime for type introspection
            origin = get_origin(base)
            # Check by name to avoid circular import
            if origin and origin.__name__ == "FlextHandlers":
                # Use FlextRuntime.extract_generic_args() from Layer 0.5
                args = FlextRuntime.extract_generic_args(base)
                # Accept all type forms: plain types, generic aliases (e.g., dict[str, FlextTypes.GeneralValueType]),
                # and string type references. The _evaluate_type_compatibility method
                # handles all these forms correctly.
                if args and args[0] is not None:
                    message_types.append(args[0])
        return message_types

    @classmethod
    def _get_method_signature(
        cls,
        handle_method: FlextTypes.Handler.HandlerCallable,
    ) -> inspect.Signature | None:
        """Extract signature from handle method."""
        try:
            # inspect.signature accepts any callable
            if not callable(handle_method):
                return None
            # Pass directly - inspect.signature accepts any callable
            return inspect.signature(handle_method)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _get_type_hints_safe(
        cls,
        handle_method: FlextTypes.Handler.HandlerCallable,
        handler_class: type,
    ) -> dict[str, FlextTypes.GeneralValueType]:
        """Safely extract type hints from handle method."""
        try:
            return get_type_hints(
                handle_method,
                globalns=getattr(handle_method, "__globals__", {}),
                localns=dict(vars(handler_class)),
            )
        except (NameError, AttributeError, TypeError):
            return {}

    @classmethod
    def _extract_message_type_from_parameter(
        cls,
        parameter: inspect.Parameter,
        type_hints: dict[str, FlextTypes.GeneralValueType],
        param_name: str,
    ) -> FlextTypes.Utility.MessageTypeSpecifier | None:
        """Extract message type from parameter hints or annotation."""
        if param_name in type_hints:
            # Return the type hint directly (plain types, generic aliases, etc.)
            hint = type_hints[param_name]
            # Type narrowing: MessageTypeSpecifier = str | type[GeneralValueType]
            # hint is GeneralValueType, but we need to check if it's a valid MessageTypeSpecifier
            if hint is None:
                return None
            # If hint is a string or a type, it's valid MessageTypeSpecifier
            if isinstance(hint, str):
                return hint
            if isinstance(hint, type):
                # Cast type to type[GeneralValueType] for MessageTypeSpecifier
                # MessageTypeSpecifier = str | type[GeneralValueType]
                return cast("FlextTypes.Utility.MessageTypeSpecifier", hint)
            # For other types (Sequence, Mapping), convert to string
            # string is valid MessageTypeSpecifier
            return str(hint)

        annotation = parameter.annotation
        if annotation is not inspect.Signature.empty:
            # Cast annotation to MessageTypeSpecifier (str | type[GeneralValueType])
            # annotation can be any type annotation, we cast it for type safety
            return cast("FlextTypes.Utility.MessageTypeSpecifier", annotation)

        return None

    @classmethod
    def _extract_message_type_from_handle(
        cls,
        handler_class: type,
    ) -> FlextTypes.Utility.MessageTypeSpecifier | None:
        """Extract message type from handle method annotations when generics are absent.

        Args:
            handler_class: Handler class to analyze

        Returns:
            Message type from handle method or None

        """
        handle_method = getattr(handler_class, "handle", None)
        if handle_method is None:
            return None

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
        accepted_types: tuple[FlextTypes.Utility.MessageTypeSpecifier, ...],
        message_type: FlextTypes.Utility.MessageTypeSpecifier,
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
        expected_type: FlextTypes.Utility.TypeOriginSpecifier,
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
        if (
            hasattr(expected_type, "__name__")
            and getattr(expected_type, "__name__", "") == "object"
        ):
            return True

        return None  # Not object type - continue checking

    @classmethod
    def _check_dict_compatibility(
        cls,
        expected_type: FlextTypes.Utility.TypeOriginSpecifier,
        message_type: FlextTypes.Utility.MessageTypeSpecifier,
        origin_type: FlextTypes.Utility.TypeOriginSpecifier,
        message_origin: FlextTypes.Utility.TypeOriginSpecifier,
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
        # Handle dict/dict[str, FlextTypes.GeneralValueType] compatibility
        # If expected is dict or dict[str, FlextTypes.GeneralValueType], accept dict instances
        if origin_type is dict and (
            message_origin is dict
            or (isinstance(message_type, type) and issubclass(message_type, dict))
        ):
            return True

        # If message is dict or dict[str, FlextTypes.GeneralValueType], and expected is also dict-like
        if (
            isinstance(message_type, type)
            and issubclass(message_type, dict)
            and (
                origin_type is dict
                or (isinstance(expected_type, type) and issubclass(expected_type, dict))
            )
        ):
            return True

        return None  # Not dict compatibility - continue checking

    @classmethod
    def _evaluate_type_compatibility(
        cls,
        expected_type: FlextTypes.Utility.TypeOriginSpecifier,
        message_type: FlextTypes.Utility.MessageTypeSpecifier,
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
        expected_type: FlextTypes.Utility.TypeOriginSpecifier,
        message_type: FlextTypes.Utility.TypeOriginSpecifier,
        origin_type: FlextTypes.Utility.TypeOriginSpecifier,
        message_origin: FlextTypes.Utility.TypeOriginSpecifier,
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
                return issubclass(message_type, origin_type)
            return message_type is expected_type
        except TypeError:
            return message_type is expected_type

    @classmethod
    def _handle_instance_check(
        cls,
        message_type: FlextTypes.Utility.TypeOriginSpecifier,
        origin_type: FlextTypes.Utility.TypeOriginSpecifier,
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
                return isinstance(message_type, origin_type)
            return True
        except TypeError:
            return True


__all__ = ["FlextUtilitiesTypeChecker"]
