"""Test data builders for FLEXT ecosystem tests.

Provides builder pattern implementation for creating complex test data structures
with fluent interface for method chaining. Supports building datasets with users,
configurations, and validation fields using Models and factories.

Scope: Builder class for constructing test datasets with fluent interface,
supporting method chaining for complex data assembly. Integrates with
FlextTestsFactories for consistent test data generation using Models.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import Self

from flext_tests.factories import FlextTestsFactories


class FlextTestsBuilders:
    """Test data builders using builder pattern.

    Provides fluent interface for building complex test data structures.
    """

    def __init__(self) -> None:
        """Initialize test data builder."""
        self._data: dict[str, object] = {}

    def with_users(self, count: int = 5) -> Self:
        """Add users to dataset using factories.

        Args:
            count: Number of users to create

        Returns:
            Self for method chaining

        """
        users = FlextTestsFactories.batch_users(count)
        self._data["users"] = [
            {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "active": user.active,
            }
            for user in users
        ]
        return self

    def with_configs(self, *, production: bool = False) -> Self:
        """Add configuration to dataset using factories.

        Args:
            production: Whether to use production config

        Returns:
            Self for method chaining

        """
        config = FlextTestsFactories.create_config(
            service_type="api",
            environment="production" if production else "development",
            debug=not production,
            timeout=30,
        )
        self._data["configs"] = {
            "service_type": config.service_type,
            "environment": config.environment,
            "debug": config.debug,
            "log_level": config.log_level,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
            "database_url": "postgresql://localhost/testdb",
            "max_connections": 10,
        }
        return self

    def with_validation_fields(self, count: int = 5) -> Self:
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

    def reset(self) -> Self:
        """Reset builder state.

        Returns:
            Self for method chaining

        """
        self._data = {}
        return self
