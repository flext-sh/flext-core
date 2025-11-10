"""Reusable test handler fixtures - Eliminates 100+ lines of boilerplate.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable

from flext_core import FlextConstants, FlextHandlers, FlextModels, FlextResult


def create_test_handler(
    handler_id: str,
    handler_name: str | None = None,
    handler_type: FlextConstants.Cqrs.HandlerType = FlextConstants.Cqrs.HandlerType.COMMAND,
    process_fn: Callable[[object], FlextResult[object]] | None = None,
) -> FlextHandlers[object, object]:
    """Factory for creating test handlers - reduces massive boilerplate.

    Args:
        handler_id: Unique handler identifier
        handler_name: Display name (defaults to handler_id.title())
        handler_type: Handler type (COMMAND, QUERY, EVENT)
        process_fn: Optional custom processing function

    Returns:
        FlextHandlers instance ready for registration

    Examples:
        >>> # Simple identity handler
        >>> handler = create_test_handler("test_handler")

        >>> # Custom processing
        >>> def double_value(msg: object) -> FlextResult[object]:
        ...     return FlextResult[object].ok(int(msg) * 2)
        >>> handler = create_test_handler("doubler", process_fn=double_value)

    """

    class DynamicTestHandler(FlextHandlers[object, object]):
        def __init__(self) -> None:
            config = FlextModels.Cqrs.Handler(
                handler_id=handler_id,
                handler_name=handler_name or handler_id.replace("_", " ").title(),
                handler_type=handler_type,
                handler_mode=handler_type,
            )
            super().__init__(config=config)

        def handle(self, message: object) -> FlextResult[object]:
            if process_fn:
                return process_fn(message)
            return FlextResult[object].ok(f"Handled: {message}")

    return DynamicTestHandler()


def create_simple_handler(
    handler_id: str,
    result_value: object = "success",
) -> FlextHandlers[object, object]:
    """Create a simple handler that always returns the same value.

    Args:
        handler_id: Handler identifier
        result_value: Value to return

    Returns:
        Handler that always succeeds with result_value

    """

    def always_succeed(_msg: object) -> FlextResult[object]:
        return FlextResult[object].ok(result_value)

    return create_test_handler(handler_id, process_fn=always_succeed)


def create_failing_handler(
    handler_id: str,
    error_message: str = "Handler failed",
) -> FlextHandlers[object, object]:
    """Create a handler that always fails.

    Args:
        handler_id: Handler identifier
        error_message: Error message to return

    Returns:
        Handler that always fails with error_message

    """

    def always_fail(_msg: object) -> FlextResult[object]:
        return FlextResult[object].fail(error_message)

    return create_test_handler(handler_id, process_fn=always_fail)


def create_transform_handler(
    handler_id: str,
    transform_fn: Callable[[object], object],
) -> FlextHandlers[object, object]:
    """Create a handler that transforms messages.

    Args:
        handler_id: Handler identifier
        transform_fn: Transformation function

    Returns:
        Handler that transforms messages using transform_fn

    """

    def transform(msg: object) -> FlextResult[object]:
        try:
            result = transform_fn(msg)
            return FlextResult[object].ok(result)
        except Exception as e:
            return FlextResult[object].fail(str(e))

    return create_test_handler(handler_id, process_fn=transform)
