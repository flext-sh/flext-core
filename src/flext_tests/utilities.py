"""Test utilities for FLEXT ecosystem.

Provides utility functions for testing FLEXT components.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import MagicMock

from flext_core import FlextCore


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
    ) -> FlextCore.Result[object]:
        """Create a test FlextCore.Result.

        Args:
            success: Whether the result should be successful
            data: Success data
            error: Error message for failure results

        Returns:
            FlextCore.Result instance

        """
        if success:
            return FlextCore.Result[object].ok(data)
        return FlextCore.Result[object].fail(error or "Test error")

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
        **options: FlextCore.Types.Dict,
    ) -> Generator[None]:
        """Context manager for temporarily changing object attributes.

        Args:
            target: Object to modify
            attribute: Attribute name to change
            new_value: New value for the attribute
            **options: Additional options

        Yields:
            None

        """
        original_value = getattr(target, attribute, None)
        setattr(target, attribute, new_value)

        try:
            yield
        finally:
            if original_value is not None:
                setattr(target, attribute, original_value)
            elif options.get("delete_after"):
                delattr(target, attribute)

    class TestUtilities:
        """Nested class with additional test utilities."""

        @staticmethod
        def assert_result_success(result: FlextCore.Result[object]) -> None:
            """Assert that a FlextCore.Result is successful.

            Args:
                result: FlextCore.Result to check

            Raises:
                AssertionError: If result is not successful

            """
            assert result.is_success, f"Expected success result, got: {result}"

        @staticmethod
        def assert_result_failure(result: FlextCore.Result[object]) -> None:
            """Assert that a FlextCore.Result is a failure.

            Args:
                result: FlextCore.Result to check

            Raises:
                AssertionError: If result is not a failure

            """
            assert result.is_failure, f"Expected failure result, got: {result}"

        @staticmethod
        def create_mock_service(**methods: FlextCore.Types.Dict) -> MagicMock:
            """Create a mock service with specified methods.

            Args:
                **methods: Method implementations for the mock

            Returns:
                Configured MagicMock instance

            """
            mock = MagicMock()
            for method_name, implementation in methods.items():
                setattr(mock, method_name, implementation)
            return mock

        @staticmethod
        def generate_test_id(prefix: str = "test") -> str:
            """Generate a unique test identifier.

            Args:
                prefix: Prefix for the identifier

            Returns:
                Unique test identifier

            """
            return f"{prefix}_{uuid.uuid4().hex[:8]}"
