"""Test matchers and assertions for FLEXT ecosystem tests.

Provides custom pytest-compatible matchers and assertion helpers for validating
FlextResult patterns, data structures, and common test scenarios. Includes
builder pattern for test datasets and validation utilities.

Scope: Custom assertion methods for FlextResult success/failure validation,
dictionary/list containment checks, email format validation, configuration
validation, and test data builders using Models. Supports method chaining
and reusable validation helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import re
from typing import Self, TypeVar

from flext_core import FlextResult, FlextUtilities

T_co = TypeVar("T_co", covariant=True)
TKey = TypeVar("TKey")
TValue = TypeVar("TValue")
TConfigValue = TypeVar("TConfigValue", bound=object)


class FlextTestsMatchers:
    """Custom test matchers for FLEXT ecosystem.

    Provides pytest-compatible matchers for common FLEXT patterns.
    """

    @staticmethod
    def assert_success(
        result: FlextResult[T_co],
        error_msg: str | None = None,
    ) -> T_co:
        """Assert result is success and return unwrapped value.

        Args:
            result: FlextResult to check
            error_msg: Optional custom error message

        Returns:
            Unwrapped value from result

        Raises:
            AssertionError: If result is failure

        """
        if not result.is_success:
            msg = error_msg or f"Expected success but got failure: {result.error}"
            raise AssertionError(msg)
        return result.unwrap()

    @staticmethod
    def assert_failure(
        result: FlextResult[T_co],
        expected_error: str | None = None,
    ) -> str:
        """Assert result is failure and return error message.

        Args:
            result: FlextResult to check
            expected_error: Optional expected error substring

        Returns:
            Error message from result

        Raises:
            AssertionError: If result is success

        """
        if result.is_success:
            msg = f"Expected failure but got success: {result.unwrap()}"
            raise AssertionError(msg)
        error = result.error
        if error is None:
            msg = "Expected error but got None"
            raise AssertionError(msg)
        if expected_error and expected_error not in error:
            msg = f"Expected error containing '{expected_error}' but got: {error}"
            raise AssertionError(msg)
        return error

    class TestDataBuilder:
        """Builder for test datasets."""

        def __init__(self) -> None:
            """Initialize test data builder."""
            self._data: dict[str, object] = {}

        def with_users(self, count: int = 5) -> Self:
            """Add users to dataset."""
            self._data["users"] = [
                {
                    "id": f"USER-{i}",
                    "name": f"User {i}",
                    "email": f"user{i}@example.com",
                    "age": 20 + i,
                }
                for i in range(count)
            ]
            return self

        def with_configs(
            self,
            *,
            production: bool = False,
        ) -> Self:
            """Add configuration to dataset."""
            self._data["configs"] = {
                "environment": "production" if production else "development",
                "debug": not production,
                "database_url": "postgresql://localhost/testdb",
                "api_timeout": 30,
                "max_connections": 10,
            }
            return self

        def with_validation_fields(
            self,
            count: int = 5,
        ) -> Self:
            """Add validation fields to dataset."""
            self._data["validation_fields"] = {
                "valid_emails": [f"user{i}@example.com" for i in range(count)],
                "invalid_emails": ["invalid", "no-at-sign.com", ""],
                "valid_hostnames": ["example.com", "localhost"],
                "invalid_hostnames": ["invalid..hostname", ""],
            }
            return self

        def build(self) -> dict[str, object]:
            """Build the dataset."""
            return dict[str, object](self._data)

    def assert_true(self, condition: bool, message: str | None = None) -> None:
        """Assert that a condition is true.

        Args:
            condition: Condition to check
            message: Custom error message

        Raises:
            AssertionError: If condition is not true

        """
        assert condition, message or "Assertion failed: condition is not true"

    def assert_false(self, condition: bool, message: str | None = None) -> None:
        """Assert that a condition is false.

        Args:
            condition: Condition to check
            message: Custom error message

        Raises:
            AssertionError: If condition is not false

        """
        assert not condition, message or "Assertion failed: condition is not false"

    def assert_is_none(self, value: object, message: str | None = None) -> None:
        """Assert that a value is None.

        Args:
            value: Value to check
            message: Custom error message

        Raises:
            AssertionError: If value is not None

        """
        assert value is None, message or f"Expected None, got {value}"

    def assert_is_not_none(self, value: object, message: str | None = None) -> None:
        """Assert that a value is not None.

        Args:
            value: Value to check
            message: Custom error message

        Raises:
            AssertionError: If value is None

        """
        assert value is not None, message or f"Expected not None, got {value}"

    def assert_result_success(
        self,
        result: FlextResult[T_co],
        message: str | None = None,
    ) -> None:
        """Assert that a FlextResult is successful.

        Args:
            result: FlextResult to check
            message: Custom error message

        Raises:
            AssertionError: If result is not successful

        """
        assert result.is_success, message or f"Expected success result, got: {result}"

    @staticmethod
    def assert_result_failure(
        result: FlextResult[T_co],
        expected_error: str | None = None,
        message: str | None = None,
    ) -> None:
        """Assert that a FlextResult is a failure.

        Args:
            result: FlextResult to check
            expected_error: Expected error message substring
            message: Custom error message

        Raises:
            AssertionError: If result is not a failure or error doesn't match

        """
        assert result.is_failure, message or f"Expected failure result, got: {result}"

        if expected_error:
            error_str = str(result.error) if result.error else ""
            assert expected_error in error_str, (
                f"Expected error containing '{expected_error}', got: '{error_str}'"
            )

    @staticmethod
    def assert_dict_contains(
        data: dict[TKey, TValue],
        expected: dict[TKey, TValue],
        message: str | None = None,
    ) -> None:
        """Assert that a dictionary contains expected key-value pairs.

        Args:
            data: Dictionary to check
            expected: Expected key-value pairs
            message: Custom error message

        Raises:
            AssertionError: If dictionary doesn't contain expected pairs

        """
        for key, expected_value in expected.items():
            assert key in data, message or f"Key '{key}' not found in data"
            assert data[key] == expected_value, (
                message or f"Key '{key}': expected {expected_value}, got {data[key]}"
            )

    @staticmethod
    def assert_list_contains(
        items: list[TValue],
        expected_item: TValue,
        message: str | None = None,
    ) -> None:
        """Assert that a list contains an expected item.

        Args:
            items: List to check
            expected_item: Item that should be in the list
            message: Custom error message

        Raises:
            AssertionError: If item is not in the list

        """
        assert expected_item in items, (
            message or f"Expected item '{expected_item}' not found in list"
        )

    @staticmethod
    def assert_valid_email(email: str, message: str | None = None) -> None:
        """Assert that a string is a valid email format.

        Args:
            email: Email string to validate
            message: Custom error message

        Raises:
            AssertionError: If email format is invalid

        """
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        assert re.match(email_pattern, email), (
            message or f"Invalid email format: '{email}'"
        )

    @staticmethod
    def assert_config_valid(
        config: dict[str, TConfigValue],
        message: str | None = None,
    ) -> None:
        """Assert that a configuration dictionary is valid.

        Args:
            config: Configuration dictionary to validate
            message: Custom error message

        Raises:
            AssertionError: If configuration is invalid

        """
        required_keys = ["service_type", "environment"]
        for key in required_keys:
            assert key in config, message or f"Required config key '{key}' missing"

        timeout = config.get("timeout", 0)
        assert isinstance(timeout, int) and timeout > 0, (
            message or "Config timeout must be positive integer"
        )

    @staticmethod
    def validate_required_string(value: str, field_name: str) -> str:
        """Generic string validation for tests.

        Args:
            value: String to validate
            field_name: Field name for error messages

        Returns:
            Validated string

        Raises:
            ValueError: If validation fails

        """
        if not value or not value.strip():
            msg = f"{field_name} cannot be empty"
            raise ValueError(msg)
        return value.strip()

    @staticmethod
    def validate_enum(value: str, allowed: set[str], field_name: str) -> str:
        """Generic enum validation for tests.

        Args:
            value: Value to validate
            allowed: Set of allowed values
            field_name: Field name for error messages

        Returns:
            Validated value

        Raises:
            ValueError: If validation fails

        """
        if value not in allowed:
            msg = f"Invalid {field_name}: {value}"
            raise ValueError(msg)
        return value

    @staticmethod
    def validate_list_not_empty(value: list[object], field_name: str) -> list[object]:
        """Generic list validation for tests.

        Args:
            value: List to validate
            field_name: Field name for error messages

        Returns:
            Validated list

        Raises:
            ValueError: If validation fails

        """
        if not FlextUtilities.TypeGuards.is_list_non_empty(value):
            msg = f"{field_name} cannot be empty"
            raise ValueError(msg)
        return value
