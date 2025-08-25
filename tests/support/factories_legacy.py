# ruff: noqa: ANN401
"""Unified factories for flext-core tests using factory_boy and pytest ecosystem.

Advanced factory patterns with comprehensive test data generation using:
- factory_boy for object factories
- pytest-benchmark for performance data
- faker for realistic data generation
- Pydantic models for type safety

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Callable
from typing import Any

from flext_core import (
    FlextFieldType,
    FlextResult,
    FlextTypes,
)

JsonDict = FlextTypes.Core.JsonDict


class TestDataFactory:
    """Central factory for generating consistent test data across all tests.

    Implements Factory pattern with Builder-style customization for creating
    test objects with sensible defaults and type safety.
    """

    @staticmethod
    def create_test_user_data(**overrides: object) -> JsonDict:
        """Create test user data with sensible defaults.

        Args:
            **overrides: Custom values to override defaults

        Returns:
            Dictionary with user test data

        """
        defaults = {
            "id": str(uuid.uuid4()),
            "name": "Test User",
            "email": "test@example.com",
            "age": 25,
            "is_active": True,
            "metadata": {"department": "engineering", "level": "senior"},
        }
        return {**defaults, **overrides}

    @staticmethod
    def create_test_config_data(**overrides: object) -> JsonDict:
        """Create test configuration data with sensible defaults.

        Args:
            **overrides: Custom values to override defaults

        Returns:
            Dictionary with configuration test data

        """
        defaults = {
            "database_url": "postgresql://localhost/test",
            "log_level": "DEBUG",
            "debug": True,
            "timeout": 30,
            "max_connections": 100,
            "features": ["auth", "cache", "metrics"],
        }
        return {**defaults, **overrides}

    @staticmethod
    def create_test_field_data(
        field_type: str = "string",
        **overrides: object,
    ) -> JsonDict:
        """Create test field data with sensible defaults.

        Args:
            field_type: Type of field to create
            **overrides: Custom values to override defaults

        Returns:
            Dictionary with field test data

        """
        field_id = str(uuid.uuid4())
        base_defaults = {
            "field_id": field_id,
            "field_name": f"test_field_{field_id[:8]}",
            "field_type": field_type,
            "required": True,
            "description": f"Test {field_type} field",
        }

        type_specific_defaults = {
            "string": {
                "min_length": 1,
                "max_length": 100,
                "pattern": r"^[a-zA-Z0-9_]+$",
            },
            "integer": {
                "min_value": 0,
                "max_value": 1000,
            },
            "boolean": {
                "default_value": False,
            },
        }

        defaults = {
            **base_defaults,
            **type_specific_defaults.get(field_type, {}),
        }
        return {**defaults, **overrides}

    @staticmethod
    def create_test_service_data(**overrides: object) -> JsonDict:
        """Create test service data with sensible defaults.

        Args:
            **overrides: Custom values to override defaults

        Returns:
            Dictionary with service test data

        """
        defaults = {
            "name": "test_service",
            "version": "1.0.0",
            "config": {"host": "localhost", "port": 8080},
            "dependencies": ["database", "cache"],
            "healthy": True,
        }
        return {**defaults, **overrides}

    @staticmethod
    def create_test_payload_data(**overrides: object) -> JsonDict:
        """Create test payload data with sensible defaults.

        Args:
            **overrides: Custom values to override defaults

        Returns:
            Dictionary with payload test data

        """
        defaults = {
            "message_id": str(uuid.uuid4()),
            "type": "test_message",
            "data": {"action": "test", "value": 42},
            "timestamp": "2024-01-01T00:00:00Z",
            "source": "test_system",
        }
        return {**defaults, **overrides}

    @staticmethod
    def create_test_validation_data(**overrides: object) -> JsonDict:
        """Create test validation data with sensible defaults.

        Args:
            **overrides: Custom values to override defaults

        Returns:
            Dictionary with validation test data

        """
        defaults = {
            "rules": [
                {"field": "email", "type": "email"},
                {"field": "age", "type": "integer", "min": 0, "max": 120},
            ],
            "data": {"email": "test@example.com", "age": 25},
            "expected_valid": True,
        }
        return {**defaults, **overrides}

    @classmethod
    def create_test_result_success(
        cls,
        data: object = "test_data",
    ) -> FlextResult[Any]:
        """Create successful FlextResult for testing.

        Args:
            data: Data to include in successful result

        Returns:
            FlextResult with success state

        """
        return FlextResult[Any].ok(data)

    @classmethod
    def create_test_result_failure(
        cls,
        error: str = "test_error",
        error_code: str = "TEST_ERROR",
    ) -> FlextResult[Any]:
        """Create failed FlextResult for testing.

        Args:
            error: Error message
            error_code: Error code

        Returns:
            FlextResult with failure state

        """
        return FlextResult[Any].fail(error, error_code=error_code)

    @staticmethod
    def create_mock_validator(
        *,
        should_pass: bool = True,
    ) -> Callable[[Any], FlextResult[Any]]:
        """Create mock validator function for testing.

        Args:
            should_pass: Whether validator should pass or fail

        Returns:
            Mock validator function

        """

        def validator(value: object) -> FlextResult[Any]:
            if should_pass:
                return FlextResult[Any].ok(value)
            return FlextResult[Any].fail(f"Validation failed for: {value}")

        return validator

    @staticmethod
    def create_test_json_file_content(**overrides: object) -> str:
        """Create JSON file content for testing.

        Args:
            **overrides: Custom values to override defaults

        Returns:
            JSON string for file testing

        """
        defaults = {
            "name": "test_config",
            "version": "1.0.0",
            "settings": {
                "debug": True,
                "timeout": 30,
            },
        }
        data = {**defaults, **overrides}
        return json.dumps(data, indent=2)

    @staticmethod
    def create_field_types_matrix() -> list[
        tuple[FlextFieldType, list[Any], list[bool]]
    ]:
        """Create matrix of field types with test values and expected validity.

        Returns:
            List of tuples (field_type, test_values, expected_valid)

        """
        return [
            (
                FlextFieldType.STRING,
                ["valid_string", "", "a" * 1000, 123, None],
                [True, True, False, False, False],
            ),
            (
                FlextFieldType.INTEGER,
                [42, 0, -10, "123", "invalid", None],
                [True, True, True, False, False, False],
            ),
            (
                FlextFieldType.BOOLEAN,
                [True, False, "true", "false", 1, 0, "invalid", None],
                [True, True, False, False, False, False, False, False],
            ),
            (
                FlextFieldType.FLOAT,
                [math.pi, 0.0, -2.5, "3.14", "invalid", None],
                [True, True, True, False, False, False],
            ),
        ]

    @staticmethod
    def create_edge_case_values() -> dict[str, list[Any]]:
        """Create edge case values for comprehensive testing.

        Returns:
            Dictionary mapping edge case types to test values

        """
        return {
            "empty": ["", [], {}, None],
            "large": [
                "x" * 10000,
                list(range(1000)),
                {f"key_{i}": i for i in range(100)},
            ],
            "unicode": ["🚀", "测试", "مرحبا", "🔥🎯"],
            "special_chars": ["!@#$%^&*()", "\n\t\r", "\\", "'\""],
            "numeric_edge": [0, -1, 999999999, 1e-10, float("inf")],
            "boolean_like": ["true", "false", "yes", "no", "1", "0"],
        }


class ServiceFactory:
    """Factory for creating test services and dependencies.

    Specialized factory for service-related test objects following
    Dependency Injection and Service Locator patterns.
    """

    @staticmethod
    def create_mock_database_service() -> dict[str, Any]:
        """Create mock database service for testing."""
        return {
            "type": "database",
            "connection_string": "postgresql://test:test@localhost/test",
            "pool_size": 10,
            "timeout": 30,
            "connected": True,
        }

    @staticmethod
    def create_mock_cache_service() -> dict[str, Any]:
        """Create mock cache service for testing."""
        return {
            "type": "cache",
            "host": "localhost",
            "port": 6379,
            "ttl": 3600,
            "connected": True,
        }

    @staticmethod
    def create_mock_logger_service() -> dict[str, Any]:
        """Create mock logger service for testing."""
        return {
            "type": "logger",
            "level": "DEBUG",
            "format": "json",
            "handlers": ["console", "file"],
        }


# Convenience aliases for common patterns
create_user = TestDataFactory.create_test_user_data
create_config = TestDataFactory.create_test_config_data
create_field = TestDataFactory.create_test_field_data
create_success = TestDataFactory.create_test_result_success
create_failure = TestDataFactory.create_test_result_failure
