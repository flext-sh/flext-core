# ruff: noqa: ANN401
"""Domain factories for test data generation.

Simple factory pattern without factory-boy complications.
Uses FlextResult patterns and realistic data for testing.
"""

from __future__ import annotations

import random
import uuid
from decimal import Decimal
from typing import Any

from flext_core import FlextResult, FlextTypes

JsonDict = FlextTypes.Core.JsonDict


class FlextResultFactory:
    """Factory for creating FlextResult objects with realistic scenarios."""

    @staticmethod
    def success_result(data: object = None) -> FlextResult[Any]:
        """Create successful result."""
        return FlextResult[Any].ok(
            data or {"status": "success", "id": str(uuid.uuid4())}
        )

    @staticmethod
    def create_success(data: object = None) -> FlextResult[Any]:
        """Create successful result (alias for success_result)."""
        return FlextResultFactory.success_result(data)

    @staticmethod
    def create_failure(
        error: str = "Operation failed", error_code: str | None = None
    ) -> FlextResult[Any]:
        """Create failed result with optional error code."""
        return FlextResult[Any].fail(error, error_code=error_code)

    @staticmethod
    def failed_result(error: str = "Operation failed") -> FlextResult[Any]:
        """Create failed result."""
        return FlextResult[Any].fail(error, error_code="OPERATION_ERROR")

    @staticmethod
    def validation_error(field: str = "unknown", value: object = None) -> FlextResult[Any]:
        """Create validation failure FlextResult."""
        return FlextResult[Any].fail(
            f"Validation failed for field '{field}'",
            error_code="VALIDATION_ERROR",
            error_data={"field": field, "value": value},
        )


class UserDataFactory:
    """Factory for creating realistic user data."""

    @staticmethod
    def create(**overrides: object) -> dict[str, Any]:
        """Create user data dict."""
        data = {
            "id": str(uuid.uuid4()),
            "name": f"User {random.randint(100, 999)}",  # noqa: S311
            "email": f"user{random.randint(1, 1000)}@example.com",  # noqa: S311
            "age": random.randint(18, 65),  # noqa: S311
            "active": True,
            "created_at": "2024-01-01T00:00:00Z",
        }
        data.update(overrides)
        return data

    @staticmethod
    def build(**overrides: object) -> dict[str, Any]:
        """Build user data dict (alias for create)."""
        return UserDataFactory.create(**overrides)

    @staticmethod
    def batch(count: int = 5) -> list[dict[str, Any]]:
        """Create batch of user data."""
        return [UserDataFactory.create() for _ in range(count)]


class ConfigurationFactory:
    """Factory for creating realistic configuration data."""

    @staticmethod
    def create(**overrides: object) -> dict[str, Any]:
        """Create configuration data dict."""
        data = {
            "database_url": "postgresql://localhost:5432/test",
            "log_level": "INFO",
            "debug": False,
            "cache_enabled": True,
            "api_timeout": 30,
            "max_connections": 10,
        }
        data.update(overrides)
        return data


class ServiceDataFactory:
    """Factory for creating realistic service data."""

    @staticmethod
    def create(**overrides: object) -> dict[str, Any]:
        """Create service data dict."""
        data = {
            "name": f"test_service_{random.randint(1, 100)}",  # noqa: S311
            "version": f"1.{random.randint(0, 10)}.{random.randint(0, 50)}",  # noqa: S311
            "port": random.randint(8000, 9000),  # noqa: S311
            "health_check_path": "/health",
            "dependencies": [],
        }
        data.update(overrides)
        return data


class PayloadDataFactory:
    """Factory for creating realistic message/payload data."""

    @staticmethod
    def create(**overrides: object) -> dict[str, Any]:
        """Create payload data dict."""
        data = {
            "message_id": str(uuid.uuid4()),
            "type": "user_created",
            "timestamp": "2024-01-01T00:00:00Z",
            "data": {"user_id": str(uuid.uuid4())},
            "metadata": {"source": "test", "version": "1.0"},
        }
        data.update(overrides)
        return data


class ValidationTestCases:
    """Validation test case data factory."""

    @staticmethod
    def valid_email_cases() -> list[str]:
        """Return valid email test cases."""
        return [
            "user@example.com",
            "test.user@domain.co.uk",
            "user+tag@example.org",
            "123@numbers.com",
        ]

    @staticmethod
    def invalid_email_cases() -> list[str]:
        """Return invalid email test cases."""
        return [
            "not-an-email",
            "@missing-local.com",
            "missing-at-sign.com",
            "user@.com",
            "user@domain.",
        ]

    @staticmethod
    def valid_ages() -> list[int]:
        """Return valid age test cases."""
        return [18, 25, 35, 45, 65, 99]

    @staticmethod
    def invalid_ages() -> list[int]:
        """Return invalid age test cases."""
        return [-1, 0, 17, 150, 999]


class RealisticData:
    """Factory for realistic test data scenarios."""

    @staticmethod
    def user_registration_data() -> dict[str, Any]:
        """Create realistic user registration data."""
        return {
            "name": f"John Doe {random.randint(1, 100)}",  # noqa: S311
            "email": f"john.doe{random.randint(1, 1000)}@company.com",  # noqa: S311
            "age": random.randint(18, 65),  # noqa: S311
            "phone": f"+1-{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",  # noqa: S311
            "address": {
                "street": f"{random.randint(1, 999)} Main St",  # noqa: S311
                "city": "Test City",
                "state": "TC",
                "zip": f"{random.randint(10000, 99999)}",  # noqa: S311
            },
        }

    @staticmethod
    def order_data() -> dict[str, Any]:
        """Create realistic order data."""
        return {
            "order_id": str(uuid.uuid4()),
            "customer_id": str(uuid.uuid4()),
            "items": [
                {
                    "product_id": str(uuid.uuid4()),
                    "name": f"Product {random.randint(1, 100)}",  # noqa: S311
                    "quantity": random.randint(1, 5),  # noqa: S311
                    "price": Decimal(str(random.uniform(10.0, 100.0))).quantize(  # noqa: S311
                        Decimal("0.01")
                    ),
                }
            ],
            "total": Decimal("50.00"),
            "status": "pending",
        }

    @staticmethod
    def api_response_data() -> dict[str, Any]:
        """Create realistic API response data."""
        return {
            "success": True,
            "data": {"id": str(uuid.uuid4()), "status": "processed"},
            "message": "Operation completed successfully",
            "timestamp": "2024-01-01T00:00:00Z",
            "request_id": str(uuid.uuid4()),
        }


# Export all factories
__all__ = [
    "ConfigurationFactory",
    "FlextResultFactory",
    "PayloadDataFactory",
    "RealisticData",
    "ServiceDataFactory",
    "UserDataFactory",
    "ValidationTestCases",
]
