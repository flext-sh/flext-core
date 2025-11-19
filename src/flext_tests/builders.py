"""Test data builders for FLEXT ecosystem.

Provides builder pattern for creating complex test data structures.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations


class FlextTestsBuilders:
    """Test data builders using builder pattern.

    Provides fluent interface for building complex test data structures.
    """

    def __init__(self) -> None:
        """Initialize test data builder."""
        self._data: dict[str, object] = {}

    def with_users(self, count: int = 5) -> FlextTestsBuilders:
        """Add users to dataset.

        Args:
            count: Number of users to create

        Returns:
            Self for method chaining

        """
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
    ) -> FlextTestsBuilders:
        """Add configuration to dataset.

        Args:
            production: Whether to use production config

        Returns:
            Self for method chaining

        """
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
    ) -> FlextTestsBuilders:
        """Add validation fields to dataset.

        Args:
            count: Number of validation fields

        Returns:
            Self for method chaining

        """
        self._data["validation_fields"] = {
            "valid_emails": [f"user{i}@example.com" for i in range(count)],
            "invalid_emails": ["invalid", "no-at-sign.com", ""],
            "valid_hostnames": ["example.com", "localhost"],
            "invalid_hostnames": ["invalid..hostname", ""],
        }
        return self

    def build(self) -> dict[str, object]:
        """Build the dataset.

        Returns:
            Built dataset dictionary

        """
        return dict(self._data)

    def reset(self) -> FlextTestsBuilders:
        """Reset builder state.

        Returns:
            Self for method chaining

        """
        self._data = {}
        return self
