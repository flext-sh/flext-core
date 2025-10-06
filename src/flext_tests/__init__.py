"""FLEXT Testing Framework.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

Testing utilities and fixtures for FLEXT ecosystem projects.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from flext_core import FlextResult, FlextTypes


class FlextTestsDomains:
    """FLEXT testing domains for test organization."""

    LDAP = "ldap"
    LDIF = "ldif"
    API = "api"
    CLI = "cli"
    DATABASE = "database"
    MIGRATION = "migration"


class FlextTestsMatchers:
    """Test data factories and matchers for FLEXT examples.

    Minimal implementation to support example scenarios.
    """

    class TestDataBuilder:
        """Builder for test datasets."""

        def __init__(self) -> None:
            self._data: dict[str, object] = {}

        def with_users(self, count: int = 5) -> FlextTestsMatchers.TestDataBuilder:
            """Add users to dataset."""
            from flext_core import FlextTypes

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
            self, production: bool = False
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

        def build(self) -> dict[str, object]:
            """Build the dataset."""
            return dict(self._data)

    @staticmethod
    def create_validation_test_data() -> dict[str, object]:
        """Create validation test data."""
        return {
            "valid_emails": [
                "user@example.com",
                "admin@test.com",
                "info@company.org",
            ],
            "invalid_emails": ["invalid", "no-at-sign.com", "@example.com", ""],
            "valid_hostnames": ["example.com", "localhost", "server.local"],
            "invalid_hostnames": ["invalid..hostname", "", "bad host"],
        }

    @staticmethod
    def create_realistic_test_data() -> dict[str, object]:
        """Create realistic integration test data."""
        return {
            "order": {
                "customer_id": f"CUST-{uuid4().hex[:8]}",
                "order_id": f"ORD-{uuid4().hex[:8]}",
                "total": 199.98,
                "items": [
                    {
                        "product_id": "PROD-001",
                        "name": "Product 1",
                        "price": Decimal("99.99"),
                        "quantity": 1,
                    },
                    {
                        "product_id": "PROD-002",
                        "name": "Product 2",
                        "price": Decimal("99.99"),
                        "quantity": 1,
                    },
                ],
            },
            "api_response": {
                "request_id": str(uuid4()),
                "status": "success",
                "data": {"message": "API call successful"},
            },
            "user_registration": {
                "user_id": f"USER-{uuid4().hex[:8]}",
                "email": "newuser@example.com",
                "created_at": "2025-01-01T00:00:00Z",
            },
        }

    class UserFactory:
        """Factory for creating test users."""

        @staticmethod
        def create(**overrides: object) -> dict[str, object]:
            """Create a single user."""
            default = {
                "id": f"USER-{uuid4().hex[:8]}",
                "name": "Test User",
                "email": "test@example.com",
                "age": 30,
            }
            return {**default, **overrides}

        @staticmethod
        def create_batch(count: int) -> list[dict[str, object]]:
            """Create multiple users."""
            return [
                {
                    "id": f"USER-{i}",
                    "name": f"User {i}",
                    "email": f"user{i}@example.com",
                    "age": 20 + i,
                }
                for i in range(count)
            ]

    class ConfigFactory:
        """Factory for creating test configurations."""

        @staticmethod
        def create(**overrides: object) -> dict[str, object]:
            """Create development configuration."""
            default = {
                "environment": "development",
                "debug": True,
                "log_level": "DEBUG",
                "database_url": "postgresql://localhost/devdb",
                "api_timeout": 30,
                "max_connections": 10,
            }
            return {**default, **overrides}

        @staticmethod
        def production_config(**overrides: object) -> dict[str, object]:
            """Create production configuration."""
            default = {
                "environment": "production",
                "debug": False,
                "log_level": "INFO",
                "database_url": "postgresql://prodhost/proddb",
                "api_timeout": 60,
                "max_connections": 100,
            }
            return {**default, **overrides}

    class ResultFactory:
        """Factory for creating FlextResult instances."""

        @staticmethod
        def success_result(data: object | None = None) -> object:
            """Create a successful result."""
            from flext_core import FlextResult

            if data is None:
                data = {"status": "success"}
            return FlextResult[object].ok(data)

        @staticmethod
        def failure_result(message: str = "Operation failed") -> object:
            """Create a failed result."""
            from flext_core import FlextResult

            return FlextResult[object].fail(message)

        @staticmethod
        def user_result(success: bool = True) -> object:
            """Create a user-specific result."""
            from flext_core import FlextResult

            if success:
                user_data = FlextTestsMatchers.UserFactory.create()
                return FlextResult[dict[str, object]].ok(user_data)
            return FlextResult[dict[str, object]].fail("User not found")


__version__ = "0.9.9"

__all__ = ["FlextResult", "FlextTestsDomains", "FlextTestsMatchers", "FlextTypes"]
