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
from typing import TYPE_CHECKING, Annotated, override

from flext_tests import FlextTestsServiceBase
from pydantic import BaseModel, ConfigDict, Field

from flext_core import T, h, r

from tests import , m, t


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

    class HandlerTestCase(BaseModel):
        """Factory for handler test case configurations."""

        model_config = ConfigDict(frozen=True)

        handler_id: Annotated[
            str, Field(description="Unique handler identifier for test case")
        ]
        handler_name: Annotated[
            str | None,
            Field(default=None, description="Optional display name for handler"),
        ] = None
        handler_type: Annotated[
            c.Cqrs.HandlerType,
            Field(
                default=c.Cqrs.HandlerType.COMMAND,
                description="Handler type used for test case configuration",
            ),
        ] = c.Cqrs.HandlerType.COMMAND
        expected_result: Annotated[
            t.Container | None,
            Field(
                default=None,
                description="Expected handler result when execution succeeds",
            ),
        ] = None
        should_fail: Annotated[
            bool, Field(default=False, description="Whether test case expects failure")
        ] = False
        error_message: Annotated[
            str | None,
            Field(default=None, description="Expected error message for failures"),
        ] = None
        description: Annotated[
            str, Field(default="", description="Human-readable test case description")
        ] = ""

        def create_handler(
            self,
            process_fn: Callable[
                [t.Container],
                r[t.Container],
            ]
            | None = None,
        ) -> h[t.Container, t.Container]:
            """Create handler instance for this test case."""
            return TestsFlextServiceBase.Handlers.create_test_handler(
                handler_id=self.handler_id,
                handler_name=self.handler_name,
                handler_type=self.handler_type,
                process_fn=process_fn,
            )

    class HandlerFactories:
        """Centralized factories for test handlers."""

        @staticmethod
        def success_cases() -> list[TestsFlextServiceBase.HandlerTestCase]:
            """Generate success handler test cases."""
            return [
                TestsFlextServiceBase.HandlerTestCase(
                    handler_id="success_command",
                    handler_type=c.Cqrs.HandlerType.COMMAND,
                    expected_result="Handled: test",
                    description="Command handler success",
                ),
                TestsFlextServiceBase.HandlerTestCase(
                    handler_id="success_query",
                    handler_type=c.Cqrs.HandlerType.QUERY,
                    expected_result="Handled: query",
                    description="Query handler success",
                ),
                TestsFlextServiceBase.HandlerTestCase(
                    handler_id="success_event",
                    handler_type=c.Cqrs.HandlerType.EVENT,
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
                    handler_type=c.Cqrs.HandlerType.COMMAND,
                    should_fail=True,
                    error_message="Command failed",
                    description="Command handler failure",
                ),
                TestsFlextServiceBase.HandlerTestCase(
                    handler_id="fail_query",
                    handler_type=c.Cqrs.HandlerType.QUERY,
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
            handler_type: c.Cqrs.HandlerType = c.Cqrs.HandlerType.COMMAND,
            process_fn: Callable[
                [t.Container],
                r[t.Container],
            ]
            | None = None,
        ) -> h[t.Container, t.Container]:
            """Factory for creating test handlers - reduces massive boilerplate.

            Args:
                handler_id: Unique handler identifier
                handler_name: Display name (defaults to handler_id.title())
                handler_type: Handler type (COMMAND, QUERY, EVENT)
                process_fn: Optional custom processing function

            Returns:
                h instance ready for registration

            """

            class DynamicTestHandler(
                h[t.Container, t.Container],
            ):
                """Dynamic test handler implementation."""

                def __init__(self) -> None:
                    if not handler_id:
                        msg = "Handler ID cannot be empty"
                        raise ValueError(msg)
                    config = m.Handler(
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
                    message: t.Container,
                ) -> r[t.Container]:
                    """Handle message with proper error handling."""
                    try:
                        if process_fn:
                            return process_fn(message)
                        return r[t.Container].ok(
                            f"Handled: {message}",
                        )
                    except Exception as e:
                        return r[t.Container].fail(
                            f"Handler error: {e}",
                        )

            return DynamicTestHandler()

        @staticmethod
        def create_simple_handler(
            handler_id: str,
            result_value: t.Container = c.Strings.BASIC_WORD,
        ) -> h[t.Container, t.Container]:
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
                _msg: t.Container,
            ) -> r[t.Container]:
                """Always return success with configured value."""
                return r[t.Container].ok(result_value)

            return TestsFlextServiceBase.Handlers.create_test_handler(
                handler_id,
                process_fn=always_succeed,
            )

        @staticmethod
        def create_failing_handler(
            handler_id: str,
            error_message: str = c.TestErrors.PROCESSING_ERROR,
        ) -> h[t.Container, t.Container]:
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
                error_message = c.TestErrors.PROCESSING_ERROR

            def always_fail(
                _msg: t.Container,
            ) -> r[t.Container]:
                """Always return failure with configured error."""
                return r[t.Container].fail(error_message)

            return TestsFlextServiceBase.Handlers.create_test_handler(
                handler_id,
                process_fn=always_fail,
            )

        @staticmethod
        def create_transform_handler(
            handler_id: str,
            transform_fn: Callable[[t.Container], t.Container],
        ) -> h[t.Container, t.Container]:
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
                msg: t.Container,
            ) -> r[t.Container]:
                """Transform message with proper error handling."""
                try:
                    result = transform_fn(msg)
                    return r[t.Container].ok(result)
                except Exception as e:
                    return r[t.Container].fail(
                        f"Transformation failed: {e}",
                    )

            return TestsFlextServiceBase.Handlers.create_test_handler(
                handler_id,
                process_fn=transform,
            )


__all__ = ["TestsFlextServiceBase"]
