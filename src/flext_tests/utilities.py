"""Test utilities for FLEXT ecosystem.

Provides utility functions for testing FLEXT components.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from contextlib import contextmanager
from typing import TypeVar

from flext_core import FlextResult
from flext_tests.matchers import FlextTestsMatchers

TResult = TypeVar("TResult")


class FlextTestsUtilities:
    """Test utilities for FLEXT ecosystem.

    Provides helper functions and utilities for testing FLEXT components.
    """

    @staticmethod
    def create_test_result(
        *,
        success: bool = True,
        data: object | None = None,
        error: str | None = None,
    ) -> FlextResult[object]:
        """Create a test FlextResult.

        Args:
            success: Whether the result should be successful
            data: Success data (must not be None for success)
            error: Error message for failure results

        Returns:
            FlextResult instance

        """
        if success:
            # Fast fail: None is not a valid success value
            if data is None:
                # Use empty dict as default test data
                return FlextResult[object].ok({})
            return FlextResult[object].ok(data)
        return FlextResult[object].fail(error or "Test error")

    @staticmethod
    def functional_service(
        service_type: str = "api",
        **config: str | int | bool,
    ) -> dict[str, str | int | bool]:
        """Create a functional service configuration for testing.

        Args:
            service_type: Type of service
            **config: Service configuration overrides

        Returns:
            Service configuration dictionary

        """
        base_config: dict[str, str | int | bool] = {
            "type": service_type,
            "name": f"functional_{service_type}_service",
            "enabled": True,
            "host": "localhost",
            "port": 8000,
            "timeout": 30,
            "retries": 3,
        }
        base_config.update(config)
        return base_config

    @staticmethod
    @contextmanager
    def test_context(
        target: object,
        attribute: str,
        new_value: object,
    ) -> Generator[None]:
        """Context manager for temporarily changing object attributes.

        Args:
            target: Object to modify
            attribute: Attribute name to change
            new_value: New value for the attribute

        Yields:
            None

        """
        original_value = getattr(target, attribute, None)
        attribute_existed = hasattr(target, attribute)
        setattr(target, attribute, new_value)

        try:
            yield
        finally:
            if attribute_existed:
                setattr(target, attribute, original_value)
            else:
                # Attribute didn't exist originally, remove it
                delattr(target, attribute)

    class TestUtilities:
        """Nested class with additional test utilities."""

        @staticmethod
        def assert_result_success(result: FlextResult[TResult]) -> None:
            """Assert that a FlextResult is successful.

            Args:
                result: FlextResult to check

            Raises:
                AssertionError: If result is not successful

            """
            assert result.is_success, f"Expected success result, got: {result}"

        @staticmethod
        def assert_result_failure(result: FlextResult[TResult]) -> None:
            """Assert that a FlextResult is a failure.

            Args:
                result: FlextResult to check

            Raises:
                AssertionError: If result is not a failure

            """
            assert result.is_failure, f"Expected failure result, got: {result}"

        @staticmethod
        def create_test_service(**methods: object) -> object:
            """Create a test service with specified methods.

            Args:
                **methods: Method implementations for the service

            Returns:
                Test service instance with specified methods

            """

            # Create a real service class dynamically
            class TestService:
                """Real test service implementation."""

                def __init__(self, **method_impls: object) -> None:
                    """Initialize test service with method implementations."""
                    for method_name, implementation in method_impls.items():
                        setattr(self, method_name, implementation)

            return TestService(**methods)

        @staticmethod
        def generate_test_id(prefix: str = "test") -> str:
            """Generate a unique test identifier.

            Args:
                prefix: Prefix for the identifier

            Returns:
                Unique test identifier

            """
            return f"{prefix}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def create_test_data(
        size: int = 10,
        prefix: str = "test",
        data_type: str = "generic",
    ) -> dict[str, object]:
        """Create test data dictionary.

        Args:
            size: Size of the data
            prefix: Prefix for keys
            data_type: Type of data to create

        Returns:
            Test data dictionary

        """
        data: dict[str, object] = {
            "id": str(uuid.uuid4()),
            "name": f"{prefix}_{data_type}",
            "size": size,
            "created_at": "2025-01-01T00:00:00Z",
        }

        if data_type == "user":
            data.update({
                "email": f"{prefix}@example.com",
                "active": True,
            })
        elif data_type == "config":
            data.update({
                "enabled": True,
                "timeout": 30,
            })

        return data

    @staticmethod
    def create_api_response(
        *,
        success: bool = True,
        data: object | None = None,
        error_message: str | None = None,
    ) -> dict[str, object]:
        """Create API response test data.

        Args:
            success: Whether the response should be successful
            data: Response data
            error_message: Error message for failed responses

        Returns:
            API response dictionary

        """
        response: dict[str, object] = {
            "status": "success" if success else "error",
            "timestamp": "2025-01-01T00:00:00Z",
            "request_id": str(uuid.uuid4()),
        }

        if success:
            response["data"] = data
        else:
            response["error"] = {
                "code": "TEST_ERROR",
                "message": error_message or "Test error",
            }

        return response

    @classmethod
    def utilities(cls) -> FlextTestsUtilities:
        """Get utilities instance."""
        return cls()

    @classmethod
    def assertion(cls) -> object:
        """Get assertion instance (for compatibility - returns matchers instance)."""
        return FlextTestsMatchers()

    class ResultHelpers:
        """Helpers for FlextResult testing."""

        @staticmethod
        def create_success_result(value: object) -> FlextResult[object]:
            """Create a successful FlextResult with given value."""
            return FlextResult[object].ok(value)

        @staticmethod
        def create_failure_result(
            error: str,
            error_code: str | None = None,
        ) -> FlextResult[object]:
            """Create a failed FlextResult with given error."""
            return FlextResult[object].fail(error, error_code=error_code)

        @staticmethod
        def assert_success_with_value(
            result: FlextResult[object],
            expected_value: object,
        ) -> None:
            """Assert result is success and has expected value."""
            assert result.is_success, f"Expected success, got failure: {result.error}"
            assert result.value == expected_value

        @staticmethod
        def assert_failure_with_error(
            result: FlextResult[object],
            expected_error: str | None = None,
        ) -> None:
            """Assert result is failure and has expected error."""
            assert result.is_failure, f"Expected failure, got success: {result.value}"
            if expected_error:
                assert result.error is not None
                assert expected_error in result.error

        @staticmethod
        def create_test_cases(
            success_cases: list[tuple[object, object]],
            failure_cases: list[tuple[str, str | None]],
        ) -> list[tuple[object, bool, object | None, str | None]]:
            """Create parametrized test cases for Result testing.

            Args:
                success_cases: List of (value, expected_value) tuples
                failure_cases: List of (error, error_code) tuples

            Returns:
                List of (result, is_success, expected_value, expected_error) tuples

            """
            cases: list[tuple[object, bool, object | None, str | None]] = []
            for value, expected in success_cases:
                result = FlextResult[object].ok(value)
                cases.append((result, True, expected, None))
            for error, error_code in failure_cases:
                result = FlextResult[object].fail(error, error_code=error_code)
                cases.append((result, False, None, error))
            return cases
