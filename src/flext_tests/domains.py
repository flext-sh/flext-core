"""Test domain objects and fixtures for FLEXT ecosystem tests.

Provides reusable domain objects, test data structures, and fixtures for
domain-specific testing scenarios. Includes payloads, API responses,
validation test cases, and domain result helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import uuid

from flext_tests.factories import tt
from flext_tests.typings import t


class FlextTestsDomains:
    """Test domain objects and fixtures.

    Provides common test data and domain objects used across FLEXT test suites.
    """

    class TestDomainResult:
        """Simple domain result for testing services.

        Implements ResultLike protocol for compatibility with FlextResult operations.
        """

        __test__ = False  # Not a test class, just a helper class

        def __init__(self, value: str) -> None:
            """Initialize domain result."""
            super().__init__()
            self.value = value

        @property
        def is_success(self) -> bool:
            """Check if success."""
            return True

        @property
        def is_failure(self) -> bool:
            """Check if failure."""
            return False

        @property
        def error(self) -> str | None:
            """Get error."""
            return None

        def unwrap(self) -> FlextTestsDomains.TestDomainResult:
            """Unwrap the result value."""
            return self

    @staticmethod
    def create_configuration(
        service_type: str = "api",
        environment: str = "test",
        **overrides: t.GeneralValueType,
    ) -> dict[str, t.GeneralValueType]:
        """Create test configuration data using factories.

        Args:
            service_type: Type of service configuration
            environment: Environment setting
            **overrides: Additional configuration overrides

        Returns:
            Configuration dictionary

        """
        config_result = tt.model(
            "config",
            service_type=service_type,
            environment=environment,
        )
        # Extract attributes using getattr with defaults for type safety
        # Use dict[str, t.GeneralValueType] directly instead of dict[str, t.GeneralValueType]
        base_config: dict[str, t.GeneralValueType] = {
            "service_type": getattr(config_result, "service_type", service_type),
            "environment": getattr(config_result, "environment", environment),
            "debug": getattr(config_result, "debug", False),
            "log_level": getattr(config_result, "log_level", "INFO"),
            "timeout": getattr(config_result, "timeout", 30.0),
            "max_retries": getattr(config_result, "max_retries", 3),
            "namespace": f"test_{service_type}_{uuid.uuid4().hex[:8]}",
            "storage_backend": "memory",
            "enable_caching": True,
            "cache_ttl": 300,
        }
        base_config.update(overrides)
        return base_config

    @staticmethod
    def create_payload(
        data_type: str = "user",
        **custom_fields: t.GeneralValueType,
    ) -> dict[str, t.GeneralValueType]:
        """Create test payload data.

        Args:
            data_type: Type of data to create
            **custom_fields: Custom field overrides

        Returns:
            Payload dictionary

        """
        payloads: t.StringConfigurationDictDict = {
            "user": {
                "id": str(uuid.uuid4()),
                "name": "Test User",
                "email": "test@example.com",
                "active": True,
            },
            "order": {
                "order_id": str(uuid.uuid4()),
                "user_id": str(uuid.uuid4()),
                "amount": 99.99,
                "currency": "USD",
                "status": "pending",
            },
            "api_request": {
                "method": "GET",
                "url": "/api/test",
                "headers": {"Content-Type": "application/json"},
                "body": None,
            },
        }

        payload = dict(payloads.get(data_type, {}))
        payload.update(custom_fields)
        return payload

    @staticmethod
    def api_response_data(
        status: str = "success",
        *,
        include_data: bool | None = None,
        **custom_fields: t.GeneralValueType,
    ) -> dict[str, t.GeneralValueType]:
        """Create API response test data.

        Args:
            status: Response status
            include_data: Whether to include data field
            **custom_fields: Custom response fields

        Returns:
            API response dictionary

        """
        response: dict[str, t.GeneralValueType] = {
            "status": status,
            "timestamp": "2025-01-01T00:00:00Z",
            "request_id": str(uuid.uuid4()),
        }

        if include_data:
            response["data"] = {"test": "data"}

        if status == "error":
            response["error"] = {
                "code": "TEST_ERROR",
                "message": "Test error message",
            }

        response.update(custom_fields)
        return response

    @staticmethod
    def valid_email_cases() -> list[tuple[str, bool]]:
        """Get valid email test cases.

        Returns:
            List of (email, is_valid) tuples

        """
        return [
            ("test@example.com", True),
            ("user.name@domain.co.uk", True),
            ("test+tag@example.com", True),
            ("invalid-email", False),
            ("@example.com", False),
            ("test@", False),
            ("", False),
        ]

    @staticmethod
    def create_service(
        service_type: str = "api",
        **config: t.GeneralValueType,
    ) -> dict[str, t.GeneralValueType]:
        """Create test service configuration.

        Args:
            service_type: Type of service
            **config: Service configuration

        Returns:
            Service configuration dictionary

        """
        base_service: dict[str, t.GeneralValueType] = {
            "type": service_type,
            "name": f"test_{service_type}_service",
            "enabled": True,
            "config": FlextTestsDomains.create_configuration(service_type=service_type),
        }
        base_service.update(config)
        return base_service

    @staticmethod
    def create_user(**overrides: str | bool) -> dict[str, str | bool]:
        """Create test user data using factories.

        Args:
            **overrides: User field overrides

        Returns:
            User data dictionary

        """
        first_name = str(overrides.get("first_name", "Test"))
        last_name = str(overrides.get("last_name", "User"))
        email = str(overrides.get("email", "test@example.com"))

        user_model_result = tt.model(
            "user",
            name=f"{first_name} {last_name}",
            email=email,
        )
        # Type narrowing: tt.model("user") returns m.Tests.Factory.User
        # Extract attributes safely using getattr
        user: dict[str, str | bool] = {
            "id": getattr(user_model_result, "id", ""),
            "username": str(overrides.get("username", "testuser")),
            "email": getattr(user_model_result, "email", email),
            "first_name": first_name,
            "last_name": last_name,
            "active": getattr(user_model_result, "active", True),
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
        }
        user.update(overrides)
        return user

    @staticmethod
    def batch_users(
        count: int = 5,
        **user_overrides: str | bool,
    ) -> list[dict[str, str | bool]]:
        """Create a batch of test users.

        Args:
            count: Number of users to create
            **user_overrides: Common overrides for all users

        Returns:
            List of user dictionaries

        """
        users = []
        for i in range(count):
            user_overrides_copy = user_overrides.copy()
            user_overrides_copy["username"] = f"testuser{i}"
            user_overrides_copy["email"] = f"testuser{i}@example.com"
            users.append(FlextTestsDomains.create_user(**user_overrides_copy))
        return users

    @staticmethod
    def invalid_email_cases() -> list[tuple[str, bool]]:
        """Get invalid email test cases.

        Returns:
            List of (email, is_valid) tuples - all marked as invalid

        """
        return [
            ("invalid-email", False),
            ("@example.com", False),
            ("test@", False),
            ("", False),
            ("test@.com", False),
            ("test..test@example.com", False),
            ("test@example..com", False),
        ]

    @staticmethod
    def valid_ages() -> list[int]:
        """Get valid age test cases.

        Returns:
            List of valid ages

        """
        return [18, 25, 30, 45, 65, 80, 99]

    @staticmethod
    def invalid_ages() -> list[int]:
        """Get invalid age test cases.

        Returns:
            List of invalid ages

        """
        return [-5, 0, 17, 151, 200]


__all__ = ["FlextTestsDomains"]
