"""Domain factories for test data generation.

Simple factory pattern without factory-boy complications.
Uses FlextResult patterns and realistic data for testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime
from decimal import Decimal
from typing import ClassVar

from flext_core import FlextModels, FlextTypes


class FlextTestsDomains:
    """Unified domain test data factories for FLEXT ecosystem.

    Consolidates all domain data factory patterns into a single class interface.
    Simple factory pattern without factory-boy complications.
    """

    class RepositoryError(Exception):
        """Custom exception for repository operations."""

    class TestUser(FlextModels.Config):
        """Test user model for factory testing."""

        id: str
        name: str
        email: str
        age: int
        is_active: bool
        created_at: datetime
        metadata: FlextTypes.Core.Dict

    class TestConfig(FlextModels.Config):
        """Test configuration model for factory testing."""

        database_url: str
        log_level: str = "INFO"  # Override base class field with default
        debug: bool = False  # Override base class field with default
        timeout: int
        max_connections: int
        features: FlextTypes.Core.StringList

    class TestField(FlextModels.Config):
        """Test field model for factory testing."""

        field_id: str
        field_name: str
        field_type: str
        required: bool
        description: str
        min_length: int | None = None
        max_length: int | None = None
        min_value: int | None = None
        max_value: int | None = None
        default_value: object = None
        pattern: str | None = None

    class BaseTestEntity(FlextModels.Config):
        """Base test entity for domain testing."""

        id: str
        name: str
        created_at: datetime
        updated_at: datetime
        version: int = 1
        metadata: ClassVar[FlextTypes.Core.Dict] = {}

    class BaseTestValueObject(FlextModels.Config):
        """Base test value object for domain testing."""

        value: str
        description: str
        category: str
        tags: ClassVar[FlextTypes.Core.StringList] = []

    # === User Data Factory ===

    @staticmethod
    def create_user(**overrides: object) -> FlextTypes.Core.Dict:
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
    def build_user(**overrides: object) -> FlextTypes.Core.Dict:
        """Build user data dict (alias for create_user)."""
        return FlextTestsDomains.create_user(**overrides)

    @staticmethod
    def batch_users(count: int = 5) -> list[FlextTypes.Core.Dict]:
        """Create batch of user data."""
        return [FlextTestsDomains.create_user() for _ in range(count)]

    # === Configuration Data Factory ===

    @staticmethod
    def create_configuration(**overrides: object) -> FlextTypes.Core.Dict:
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

    # === Service Data Factory ===

    @staticmethod
    def create_service(**overrides: object) -> FlextTypes.Core.Dict:
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

    # === Payload Data Factory ===

    @staticmethod
    def create_payload(**overrides: object) -> FlextTypes.Core.Dict:
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

    # === Validation Test Cases ===

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

    # === Realistic Data Factory ===

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

    # === Convenience Inner Classes ===

    class UserData:
        """User data factory methods."""

        @staticmethod
        def create(**overrides: object) -> FlextTypes.Core.Dict:
            """Create user data dict."""
            return FlextTestsDomains.create_user(**overrides)

        @staticmethod
        def build(**overrides: object) -> FlextTypes.Core.Dict:
            """Build user data dict (alias for create)."""
            return FlextTestsDomains.build_user(**overrides)

        @staticmethod
        def batch(count: int = 5) -> list[FlextTypes.Core.Dict]:
            """Create batch of user data."""
            return FlextTestsDomains.batch_users(count)

    class Configuration:
        """Configuration data factory methods."""

        @staticmethod
        def create(**overrides: object) -> FlextTypes.Core.Dict:
            """Create configuration data dict."""
            return FlextTestsDomains.create_configuration(**overrides)

    class ServiceData:
        """Service data factory methods."""

        @staticmethod
        def create(**overrides: object) -> FlextTypes.Core.Dict:
            """Create service data dict."""
            return FlextTestsDomains.create_service(**overrides)

    class PayloadData:
        """Payload data factory methods."""

        @staticmethod
        def create(**overrides: object) -> FlextTypes.Core.Dict:
            """Create payload data dict."""
            return FlextTestsDomains.create_payload(**overrides)

    class Validation:
        """Validation test cases."""

        @staticmethod
        def valid_email_cases() -> FlextTypes.Core.StringList:
            """Return valid email test cases."""
            return FlextTestsDomains.valid_email_cases()

        @staticmethod
        def invalid_email_cases() -> FlextTypes.Core.StringList:
            """Return invalid email test cases."""
            return FlextTestsDomains.invalid_email_cases()

        @staticmethod
        def valid_ages() -> list[int]:
            """Return valid age test cases."""
            return FlextTestsDomains.valid_ages()

        @staticmethod
        def invalid_ages() -> list[int]:
            """Return invalid age test cases."""
            return FlextTestsDomains.invalid_ages()

    class Realistic:
        """Realistic test data scenarios."""

        @staticmethod
        def user_registration_data() -> FlextTypes.Core.Dict:
            """Create realistic user registration data."""
            return FlextTestsDomains.user_registration_data()

        @staticmethod
        def order_data() -> FlextTypes.Core.Dict:
            """Create realistic order data."""
            return FlextTestsDomains.order_data()

        @staticmethod
        def api_response_data() -> FlextTypes.Core.Dict:
            """Create realistic API response data."""
            return FlextTestsDomains.api_response_data()


# === REMOVED COMPATIBILITY ALIASES AND FACADES ===
# Legacy compatibility removed as per user request
# All compatibility facades, aliases and protocol facades have been commented out
# Only FlextTestsDomains class is now exported

# Main class alias for backward compatibility - REMOVED
# FlextTestsDomain = FlextTestsDomains

# Legacy UserDataFactory class - REMOVED (commented out)
# class UserDataFactory:
#     """Compatibility facade for UserDataFactory - use FlextTestsDomains instead."""
#     ... all methods commented out

# Legacy SimpleConfigurationFactory class - REMOVED (commented out)
# class SimpleConfigurationFactory:
#     """Compatibility facade for SimpleConfigurationFactory - use FlextTestsDomains instead."""
#     ... all methods commented out

# Legacy ServiceDataFactory class - REMOVED (commented out)
# class ServiceDataFactory:
#     """Compatibility facade for ServiceDataFactory - use FlextTestsDomains instead."""
#     ... all methods commented out

# Legacy PayloadDataFactory class - REMOVED (commented out)
# class PayloadDataFactory:
#     """Compatibility facade for PayloadDataFactory - use FlextTestsDomains instead."""
#     ... all methods commented out

# Legacy ValidationTestCases class - REMOVED (commented out)
# class ValidationTestCases:
#     """Compatibility facade for ValidationTestCases - use FlextTestsDomains instead."""
#     ... all methods commented out

# Legacy RealisticData class - REMOVED (commented out)
# class RealisticData:
#     """Compatibility facade for RealisticData - use FlextTestsDomains instead."""
#     ... all methods commented out

# Export only the unified class
__all__ = [
    "FlextTestsDomains",
]
