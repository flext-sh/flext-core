"""Service base for flext-core tests.

Provides TestsFlextServiceBase, extending FlextTestsServiceBase with flext-core-specific
service functionality. All generic test service functionality comes from flext_tests.

Architecture:
- FlextTestsServiceBase (flext_tests) = Generic service base for all FLEXT projects
- TestsFlextServiceBase (tests/) = flext-core-specific service base extending FlextTestsServiceBase

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import override

from flext_core import (
    FlextConstants,
    FlextHandlers,
    FlextModels,
    FlextResult,
    FlextTypes,
    T,
)
from flext_tests import FlextTestsServiceBase
from tests.constants import TestsFlextConstants


class TestsFlextServiceBase(FlextTestsServiceBase[T]):
    """Service base for flext-core tests - extends FlextTestsServiceBase.

    Architecture: Extends FlextTestsServiceBase with flext-core-specific service
    functionality. All generic service functionality from FlextTestsServiceBase
    is available through inheritance.

    Rules:
    - NEVER redeclare functionality from FlextTestsServiceBase
    - Only flext-core-specific service functionality allowed
    - All generic service functionality comes from FlextTestsServiceBase
    """

    # NOTE: FlextTestsServiceBase extends FlextService and provides:
    # - Container integration
    # - Configuration management
    # - Logging setup
    # - Result wrapping
    # These are available through inheritance.

    class HandlerFactories:
        """Centralized factories for test handlers."""

        @staticmethod
        def success_cases() -> list[TestsFlextServiceBase.HandlerTestCase]:
            """Generate success handler test cases."""
            return [
                TestsFlextServiceBase.HandlerTestCase(
                    handler_id="success_command",
                    handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
                    expected_result="Handled: test",
                    description="Command handler success",
                ),
                TestsFlextServiceBase.HandlerTestCase(
                    handler_id="success_query",
                    handler_type=FlextConstants.Cqrs.HandlerType.QUERY,
                    expected_result="Handled: query",
                    description="Query handler success",
                ),
                TestsFlextServiceBase.HandlerTestCase(
                    handler_id="success_event",
                    handler_type=FlextConstants.Cqrs.HandlerType.EVENT,
                    expected_result="Handled: event",
                    description="Event handler success",
                ),
            ]

        @staticmethod
        def failure_cases() -> list[TestsFlextServiceBase.HandlerTestCase]:
            """Generate failure handler test cases."""
            return [
                TestsFlextServiceBase.HandlerTestCase(
                    handler_id="fail_command",
                    handler_type=FlextConstants.Cqrs.HandlerType.COMMAND,
                    should_fail=True,
                    error_message="Command failed",
                    description="Command handler failure",
                ),
                TestsFlextServiceBase.HandlerTestCase(
                    handler_id="fail_query",
                    handler_type=FlextConstants.Cqrs.HandlerType.QUERY,
                    should_fail=True,
                    error_message="Query failed",
                    description="Query handler failure",
                ),
            ]

    class Handlers:
        """Handler creation utilities for tests."""

        @staticmethod
        def create_test_handler(
            handler_id: str,
            handler_name: str | None = None,
            handler_type: FlextConstants.Cqrs.HandlerType = (
                FlextConstants.Cqrs.HandlerType.COMMAND
            ),
            process_fn: Callable[
                [FlextTypes.Container],
                FlextResult[FlextTypes.Container],
            ]
            | None = None,
        ) -> FlextHandlers[FlextTypes.Container, FlextTypes.Container]:
            """Factory for creating test handlers - reduces massive boilerplate.

            Args:
                handler_id: Unique handler identifier
                handler_name: Display name (defaults to handler_id.title())
                handler_type: Handler type (COMMAND, QUERY, EVENT)
                process_fn: Optional custom processing function

            Returns:
                FlextHandlers instance ready for registration

            """

            class DynamicTestHandler(
                FlextHandlers[FlextTypes.Container, FlextTypes.Container],
            ):
                """Dynamic test handler implementation."""

                def __init__(self) -> None:
                    if not handler_id:
                        msg = "Handler ID cannot be empty"
                        raise ValueError(msg)

                    config = FlextModels.Handler(
                        handler_id=handler_id,
                        handler_name=handler_name
                        or handler_id.replace("_", " ").title(),
                        handler_type=handler_type,
                        handler_mode=handler_type,
                    )
                    super().__init__(config=config)

                @override
                def handle(
                    self,
                    message: FlextTypes.Container,
                ) -> FlextResult[FlextTypes.Container]:
                    """Handle message with proper error handling."""
                    try:
                        if process_fn:
                            return process_fn(message)
                        return FlextResult[FlextTypes.Container].ok(
                            f"Handled: {message}",
                        )
                    except Exception as e:
                        return FlextResult[FlextTypes.Container].fail(
                            f"Handler error: {e}",
                        )

            return DynamicTestHandler()

        @staticmethod
        def create_simple_handler(
            handler_id: str,
            result_value: FlextTypes.Container = (
                TestsFlextConstants.Strings.BASIC_WORD
            ),
        ) -> FlextHandlers[FlextTypes.Container, FlextTypes.Container]:
            """Create a simple handler that always returns the same value.

            Args:
                handler_id: Handler identifier (must not be empty)
                result_value: Value to return

            Returns:
                Handler that always succeeds with result_value

            """
            if not handler_id:
                msg = "Handler ID cannot be empty"
                raise ValueError(msg)

            def always_succeed(
                _msg: FlextTypes.Container,
            ) -> FlextResult[FlextTypes.Container]:
                """Always return success with configured value."""
                return FlextResult[FlextTypes.Container].ok(result_value)

            return TestsFlextServiceBase.Handlers.create_test_handler(
                handler_id,
                process_fn=always_succeed,
            )

        @staticmethod
        def create_failing_handler(
            handler_id: str,
            error_message: str = TestsFlextConstants.TestErrors.PROCESSING_ERROR,
        ) -> FlextHandlers[FlextTypes.Container, FlextTypes.Container]:
            """Create a handler that always fails.

            Args:
                handler_id: Handler identifier (must not be empty)
                error_message: Error message to return

            Returns:
                Handler that always fails with error_message

            """
            if not handler_id:
                msg = "Handler ID cannot be empty"
                raise ValueError(msg)

            if not error_message:
                error_message = TestsFlextConstants.TestErrors.PROCESSING_ERROR

            def always_fail(
                _msg: FlextTypes.Container,
            ) -> FlextResult[FlextTypes.Container]:
                """Always return failure with configured error."""
                return FlextResult[FlextTypes.Container].fail(error_message)

            return TestsFlextServiceBase.Handlers.create_test_handler(
                handler_id,
                process_fn=always_fail,
            )

        @staticmethod
        def create_transform_handler(
            handler_id: str,
            transform_fn: Callable[
                [FlextTypes.Container],
                FlextTypes.Container,
            ],
        ) -> FlextHandlers[FlextTypes.Container, FlextTypes.Container]:
            """Create a handler that transforms messages.

            Args:
                handler_id: Handler identifier (must not be empty)
                transform_fn: Transformation function (must be callable)

            Returns:
                Handler that transforms messages using transform_fn

            """
            if not handler_id:
                msg = "Handler ID cannot be empty"
                raise ValueError(msg)

            def transform(
                msg: FlextTypes.Container,
            ) -> FlextResult[FlextTypes.Container]:
                """Transform message with proper error handling."""
                try:
                    result = transform_fn(msg)
                    return FlextResult[FlextTypes.Container].ok(result)
                except Exception as e:
                    return FlextResult[FlextTypes.Container].fail(
                        f"Transformation failed: {e}",
                    )

            return TestsFlextServiceBase.Handlers.create_test_handler(
                handler_id,
                process_fn=transform,
            )


__all__ = [
    "TestsFlextServiceBase",
]
