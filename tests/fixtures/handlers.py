"""Reusable test handler fixtures using advanced Python 3.13 patterns.

Provides comprehensive handler factories for testing CQRS patterns,
eliminating boilerplate code while ensuring maximum test coverage and edge cases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from flext_core import FlextConstants, FlextHandlers, FlextModels, FlextResult
from flext_core.typings import FlextTypes
from tests.helpers.constants import TestConstants


@dataclass(frozen=True, slots=True)
class HandlerTestCase:
    """Factory for handler test case configurations."""

    handler_id: str
    handler_name: str | None = None
    handler_type: FlextConstants.Cqrs.HandlerType = (
        FlextConstants.Cqrs.HandlerType.COMMAND
    )
    expected_result: FlextTypes.GeneralValueType | None = None
    should_fail: bool = False
    error_message: str | None = None
    description: str = field(default="", compare=False)

    def create_handler(
        self,
        process_fn: Callable[
            [FlextTypes.GeneralValueType],
            FlextResult[FlextTypes.GeneralValueType],
        ]
        | None = None,
    ) -> FlextHandlers[FlextTypes.GeneralValueType, FlextTypes.GeneralValueType]:
        """Create handler instance for this test case."""
        return create_test_handler(
            handler_id=self.handler_id,
            handler_name=self.handler_name,
            handler_type=self.handler_type,
            process_fn=process_fn,
        )


class HandlerFactories:
    """Centralized factories for test handlers."""

    @staticmethod
    def success_cases() -> list[HandlerTestCase]:
        """Generate success handler test cases."""
        return [
            HandlerTestCase(
                handler_id="success_command",
                handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
                expected_result="Handled: test",
                description="Command handler success",
            ),
            HandlerTestCase(
                handler_id="success_query",
                handler_type=FlextConstants.Cqrs.HandlerType.QUERY,
                expected_result="Handled: query",
                description="Query handler success",
            ),
            HandlerTestCase(
                handler_id="success_event",
                handler_type=FlextConstants.Cqrs.HandlerType.EVENT,
                expected_result="Handled: event",
                description="Event handler success",
            ),
        ]

    @staticmethod
    def failure_cases() -> list[HandlerTestCase]:
        """Generate failure handler test cases."""
        return [
            HandlerTestCase(
                handler_id="fail_command",
                handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
                should_fail=True,
                error_message="Command failed",
                description="Command handler failure",
            ),
            HandlerTestCase(
                handler_id="fail_query",
                handler_type=FlextConstants.Cqrs.HandlerType.QUERY,
                should_fail=True,
                error_message="Query failed",
                description="Query handler failure",
            ),
        ]


def create_test_handler(
    handler_id: str,
    handler_name: str | None = None,
    handler_type: FlextConstants.Cqrs.HandlerType = FlextConstants.Cqrs.HandlerType.COMMAND,
    process_fn: Callable[
        [FlextTypes.GeneralValueType],
        FlextResult[FlextTypes.GeneralValueType],
    ]
    | None = None,
) -> FlextHandlers[FlextTypes.GeneralValueType, FlextTypes.GeneralValueType]:
    """Factory for creating test handlers - reduces massive boilerplate.

    Following FLEXT standards: No lazy imports, proper type annotations,
    comprehensive edge case handling.

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
        >>> def double_value(
        ...     msg: FlextTypes.GeneralValueType,
        ... ) -> FlextResult[FlextTypes.GeneralValueType]:
        ...     return FlextResult[FlextTypes.GeneralValueType].ok(int(msg) * 2)
        >>> handler = create_test_handler("doubler", process_fn=double_value)

    """

    class DynamicTestHandler(
        FlextHandlers[FlextTypes.GeneralValueType, FlextTypes.GeneralValueType]
    ):
        """Dynamic test handler implementation."""

        def __init__(self) -> None:
            # Validate inputs
            if not handler_id:
                msg = "Handler ID cannot be empty"
                raise ValueError(msg)

            config = FlextModels.Cqrs.Handler(
                handler_id=handler_id,
                handler_name=handler_name or handler_id.replace("_", " ").title(),
                handler_type=handler_type,
                handler_mode=handler_type,
            )
            super().__init__(config=config)

        def handle(
            self, message: FlextTypes.GeneralValueType
        ) -> FlextResult[FlextTypes.GeneralValueType]:
            """Handle message with proper error handling."""
            try:
                if process_fn:
                    return process_fn(message)
                return FlextResult[FlextTypes.GeneralValueType].ok(
                    f"Handled: {message}"
                )
            except Exception as e:
                return FlextResult[FlextTypes.GeneralValueType].fail(
                    f"Handler error: {e}"
                )

    return DynamicTestHandler()


def create_simple_handler(
    handler_id: str,
    result_value: FlextTypes.GeneralValueType = TestConstants.Strings.BASIC_WORD,
) -> FlextHandlers[FlextTypes.GeneralValueType, FlextTypes.GeneralValueType]:
    """Create a simple handler that always returns the same value.

    Following FLEXT standards: Input validation, proper error handling,
    use of centralized constants.

    Args:
        handler_id: Handler identifier (must not be empty)
        result_value: Value to return

    Returns:
        Handler that always succeeds with result_value

    Raises:
        ValueError: If handler_id is empty

    """
    if not handler_id:
        msg = "Handler ID cannot be empty"
        raise ValueError(msg)

    def always_succeed(
        _msg: FlextTypes.GeneralValueType,
    ) -> FlextResult[FlextTypes.GeneralValueType]:
        """Always return success with configured value."""
        return FlextResult[FlextTypes.GeneralValueType].ok(result_value)

    return create_test_handler(handler_id, process_fn=always_succeed)


def create_failing_handler(
    handler_id: str,
    error_message: str = TestConstants.TestErrors.PROCESSING_ERROR,
) -> FlextHandlers[FlextTypes.GeneralValueType, FlextTypes.GeneralValueType]:
    """Create a handler that always fails.

    Following FLEXT standards: Input validation, use of centralized constants,
    comprehensive error handling.

    Args:
        handler_id: Handler identifier (must not be empty)
        error_message: Error message to return

    Returns:
        Handler that always fails with error_message

    Raises:
        ValueError: If handler_id is empty

    """
    if not handler_id:
        msg = "Handler ID cannot be empty"
        raise ValueError(msg)

    if not error_message:
        error_message = TestConstants.TestErrors.PROCESSING_ERROR

    def always_fail(
        _msg: FlextTypes.GeneralValueType,
    ) -> FlextResult[FlextTypes.GeneralValueType]:
        """Always return failure with configured error."""
        return FlextResult[FlextTypes.GeneralValueType].fail(error_message)

    return create_test_handler(handler_id, process_fn=always_fail)


def create_transform_handler(
    handler_id: str,
    transform_fn: Callable[[FlextTypes.GeneralValueType], FlextTypes.GeneralValueType],
) -> FlextHandlers[FlextTypes.GeneralValueType, FlextTypes.GeneralValueType]:
    """Create a handler that transforms messages.

    Following FLEXT standards: Input validation, proper exception handling,
    comprehensive edge case testing.

    Args:
        handler_id: Handler identifier (must not be empty)
        transform_fn: Transformation function (must be callable)

    Returns:
        Handler that transforms messages using transform_fn

    Raises:
        ValueError: If handler_id is empty or transform_fn is not callable

    """
    if not handler_id:
        msg = "Handler ID cannot be empty"
        raise ValueError(msg)

    if not callable(transform_fn):
        msg = "Transform function must be callable"
        raise ValueError(msg)

    def transform(
        msg: FlextTypes.GeneralValueType,
    ) -> FlextResult[FlextTypes.GeneralValueType]:
        """Transform message with proper error handling."""
        try:
            result = transform_fn(msg)
            return FlextResult[FlextTypes.GeneralValueType].ok(result)
        except Exception as e:
            return FlextResult[FlextTypes.GeneralValueType].fail(
                f"Transformation failed: {e}"
            )

    return create_test_handler(handler_id, process_fn=transform)
