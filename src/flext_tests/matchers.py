"""Test matchers and assertions for FLEXT ecosystem.

Provides custom pytest matchers and assertion helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import re
from typing import (
    TypeVar,
)

from flext_core import FlextResult, FlextTypes

T_co = TypeVar("T_co", covariant=True)
TKey = TypeVar("TKey")
TValue = TypeVar("TValue")
TConfigValue = TypeVar("TConfigValue", bound=object)


class DataBuilder:
    """Builder for test datasets (renamed to avoid pytest collection)."""

    def __init__(self) -> None:
        """Initialize test data builder."""
        super().__init__()
        self._data: FlextTypes.Dict = {}

    def with_users(self, count: int = 5) -> DataBuilder:
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

    def with_configs(self, *, production: bool = False) -> DataBuilder:
        """Add configuration to dataset."""
        self._data["configs"] = {
            "environment": "production" if production else "development",
            "debug": not production,
            "database_url": "postgresql://localhost/testdb",
            "api_timeout": 30,
            "max_connections": 10,
        }
        return self

    def with_validation_fields(self, count: int = 5) -> DataBuilder:
        """Add validation fields to dataset."""
        self._data["validation_fields"] = {
            "valid_emails": [f"user{i}@example.com" for i in range(count)],
            "invalid_emails": ["invalid", "no-at-sign.com", ""],
            "valid_hostnames": ["example.com", "localhost"],
            "invalid_hostnames": ["invalid..hostname", ""],
        }
        return self

    def build(self) -> FlextTypes.Dict:
        """Build the dataset."""
        return dict[str, object](self._data)


class FlextTestsMatchers:
    """Custom test matchers for FLEXT ecosystem.

    Provides pytest-compatible matchers for common FLEXT patterns.
    """

    class TestDataBuilder:
        """Builder for test datasets."""

        def __init__(self) -> None:
            """Initialize test data builder."""
            super().__init__()
            self._data: FlextTypes.Dict = {}

        def with_users(self, count: int = 5) -> FlextTestsMatchers.TestDataBuilder:
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
            self, *, production: bool = False
        ) -> FlextTestsMatchers.TestDataBuilder:
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
            self, count: int = 5
        ) -> FlextTestsMatchers.TestDataBuilder:
            """Add validation fields to dataset."""
            self._data["validation_fields"] = {
                "valid_emails": [f"user{i}@example.com" for i in range(count)],
                "invalid_emails": ["invalid", "no-at-sign.com", ""],
                "valid_hostnames": ["example.com", "localhost"],
                "invalid_hostnames": ["invalid..hostname", ""],
            }
            return self

        def build(self) -> FlextTypes.Dict:
            """Build the dataset."""
            return dict[str, object](self._data)

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
        config: dict[str, TConfigValue], message: str | None = None
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
