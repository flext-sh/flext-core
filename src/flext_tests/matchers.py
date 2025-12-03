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
from collections.abc import Mapping
from typing import Self, TypeVar

from flext_core import FlextResult, u
from flext_core.typings import T_co, t
from flext_tests.typings import FlextTestsTypings

TKey = TypeVar("TKey")
TValue = TypeVar("TValue")
TConfigValue = TypeVar("TConfigValue", bound=t.GeneralValueType)


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

    @staticmethod
    def assert_true(condition: bool, message: str | None = None) -> None:
        """Assert that a condition is true.

        Args:
            condition: Condition to check
            message: Custom error message

        Raises:
            AssertionError: If condition is not true

        """
        assert condition, message or "Assertion failed: condition is not true"

    @staticmethod
    def assert_false(condition: bool, message: str | None = None) -> None:
        """Assert that a condition is false.

        Args:
            condition: Condition to check
            message: Custom error message

        Raises:
            AssertionError: If condition is not false

        """
        assert not condition, message or "Assertion failed: condition is not false"

    @staticmethod
    def assert_is_none(value: object, message: str | None = None) -> None:
        """Assert that a value is None.

        Args:
            value: Value to check
            message: Custom error message

        Raises:
            AssertionError: If value is not None

        """
        assert value is None, message or f"Expected None, got {value}"

    @staticmethod
    def assert_is_not_none(value: object, message: str | None = None) -> None:
        """Assert that a value is not None.

        Args:
            value: Value to check
            message: Custom error message

        Raises:
            AssertionError: If value is None

        """
        assert value is not None, message or f"Expected not None, got {value}"

    @staticmethod
    def assert_result_success(
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

        timeout = u.get(config, "timeout", default=0) or 0
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
    def validate_list_not_empty(
        value: list[FlextTestsTypings.TestResultValue], field_name: str
    ) -> list[FlextTestsTypings.TestResultValue]:
        """Generic list validation for tests.

        Args:
            value: List to validate
            field_name: Field name for error messages

        Returns:
            Validated list

        Raises:
            ValueError: If validation fails

        """
        if not u.TypeGuards.is_list_non_empty(value):
            msg = f"{field_name} cannot be empty"
            raise ValueError(msg)
        return value

    class TestDataBuilder:
        """Builder for test datasets with fluent API.

        Provides method chaining for building complex test data structures
        with users, configs, validation fields, and other test entities.
        """

        def __init__(self) -> None:
            """Initialize empty builder."""
            self._data: dict[str, t.GeneralValueType] = {}

        def with_users(self, count: int = 5) -> Self:
            """Add users to the dataset.

            Args:
                count: Number of users to generate

            Returns:
                Self for method chaining

            """
            users: list[Mapping[str, t.ScalarValue]] = []
            for i in range(count):
                user: Mapping[str, t.ScalarValue] = {
                    "id": f"USER-{i}",
                    "name": f"User {i}",
                    "email": f"user{i}@example.com",
                    "age": 20 + i,
                }
                users.append(user)
            self._data["users"] = users
            return self

        def with_configs(self, *, production: bool = False) -> Self:
            """Add configuration to the dataset.

            Args:
                production: Whether to use production settings

            Returns:
                Self for method chaining

            """
            configs: Mapping[str, t.ScalarValue] = {
                "database_url": "postgresql://localhost/testdb",
                "api_timeout": 30,
                "debug": not production,
                "environment": "production" if production else "development",
                "max_connections": 10,
            }
            self._data["configs"] = configs
            return self

        def with_validation_fields(self, count: int = 5) -> Self:
            """Add validation fields to the dataset.

            Args:
                count: Number of valid emails to generate (default: 5)

            Returns:
                Self for method chaining

            """
            # Generate valid emails based on count
            valid_emails = [f"user{i}@example.com" for i in range(count)]

            # Static invalid emails for validation testing
            invalid_emails = ["invalid", "missing-at-symbol", "also@invalid"]

            # Static valid hostnames
            valid_hostnames = ["example.com", "localhost"]

            self._data["validation_fields"] = {
                "valid_emails": valid_emails,
                "invalid_emails": invalid_emails,
                "valid_hostnames": valid_hostnames,
            }
            return self

        def build(self) -> dict[str, t.GeneralValueType]:
            """Build and return the dataset.

            Returns:
                Built dataset dictionary

            """
            return self._data.copy()
