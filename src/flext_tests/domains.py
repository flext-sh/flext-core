"""Domain factories for test data generation.

Simple factory pattern without factory-boy complications.
Uses FlextResult patterns and realistic data for testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import random
import uuid
from decimal import Decimal

from flext_core import FlextTypes

JsonDict = FlextTypes.Core.JsonObject
JsonValue = FlextTypes.Core.JsonValue


class UserDataFactory:
    """Factory for creating realistic user data."""

    @staticmethod
    def create(**overrides: object) -> FlextTypes.Core.Dict:
        """Create user data dict."""
        data: FlextTypes.Core.Dict = {
            "id": str(uuid.uuid4()),
            "name": f"User {random.randint(100, 999)}",
            "email": f"user{random.randint(1, 1000)}@example.com",
            "age": random.randint(18, 65),
            "active": True,
            "created_at": "2024-01-01T00:00:00Z",
        }
        data.update(overrides)
        return data

    @staticmethod
    def build(**overrides: object) -> FlextTypes.Core.Dict:
        """Build user data dict (alias for create)."""
        return UserDataFactory.create(**overrides)

    @staticmethod
    def batch(count: int = 5) -> list[FlextTypes.Core.Dict]:
        """Create batch of user data."""
        return [UserDataFactory.create() for _ in range(count)]


class SimpleConfigurationFactory:
    """Factory for creating realistic configuration data."""

    @staticmethod
    def create(**overrides: object) -> FlextTypes.Core.Dict:
        """Create configuration data dict."""
        data: FlextTypes.Core.Dict = {
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
    def create(**overrides: object) -> FlextTypes.Core.Dict:
        """Create service data dict."""
        data: FlextTypes.Core.Dict = {
            "name": f"test_service_{random.randint(1, 100)}",
            "version": f"1.{random.randint(0, 10)}.{random.randint(0, 50)}",
            "port": random.randint(8000, 9000),
            "health_check_path": "/health",
            "dependencies": [],
        }
        data.update(overrides)
        return data


class PayloadDataFactory:
    """Factory for creating realistic message/payload data."""

    @staticmethod
    def create(**overrides: object) -> FlextTypes.Core.Dict:
        """Create payload data dict."""
        data: FlextTypes.Core.Dict = {
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
    def valid_email_cases() -> FlextTypes.Core.StringList:
        """Return valid email test cases."""
        return [
            "user@example.com",
            "test.user@domain.co.uk",
            "user+tag@example.org",
            "123@numbers.com",
        ]

    @staticmethod
    def invalid_email_cases() -> FlextTypes.Core.StringList:
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
    def user_registration_data() -> FlextTypes.Core.Dict:
        """Create realistic user registration data."""
        return {
            "name": f"John Doe {random.randint(1, 100)}",
            "email": f"john.doe{random.randint(1, 1000)}@company.com",
            "age": random.randint(18, 65),
            "phone": f"+1-{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            "address": {
                "street": f"{random.randint(1, 999)} Main St",
                "city": "Test City",
                "state": "TC",
                "zip": f"{random.randint(10000, 99999)}",
            },
        }

    @staticmethod
    def order_data() -> FlextTypes.Core.Dict:
        """Create realistic order data."""
        return {
            "order_id": str(uuid.uuid4()),
            "customer_id": str(uuid.uuid4()),
            "items": [
                {
                    "product_id": str(uuid.uuid4()),
                    "name": f"Product {random.randint(1, 100)}",
                    "quantity": random.randint(1, 5),
                    "price": Decimal(str(random.uniform(10.0, 100.0))).quantize(
                        Decimal("0.01"),
                    ),
                },
            ],
            "total": Decimal("50.00"),
            "status": "pending",
        }

    @staticmethod
    def api_response_data() -> FlextTypes.Core.Dict:
        """Create realistic API response data."""
        return {
            "success": True,
            "data": {"id": str(uuid.uuid4()), "status": "processed"},
            "message": "Operation completed successfully",
            "timestamp": "2024-01-01T00:00:00Z",
            "request_id": str(uuid.uuid4()),
        }


# Main unified class
class FlextTestsDomains:
    """Unified domain test data factories for FLEXT ecosystem.

    Consolidates all domain data factory patterns into a single class interface.
    """

    # Delegate to existing implementations
    UserData = UserDataFactory
    Configuration = SimpleConfigurationFactory
    ServiceData = ServiceDataFactory
    PayloadData = PayloadDataFactory
    Validation = ValidationTestCases
    Realistic = RealisticData


# Export all factories
__all__ = [
    "FlextTestsDomains",
    "PayloadDataFactory",
    "RealisticData",
    "ServiceDataFactory",
    "SimpleConfigurationFactory",
    "UserDataFactory",
    "ValidationTestCases",
]
