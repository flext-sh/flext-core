"""Test domain objects and fixtures for FLEXT ecosystem tests.

Provides reusable domain objects, test data structures, and fixtures for
domain-specific testing scenarios. Includes payloads, API responses,
validation test cases, and domain result helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, MutableMapping

from flext_tests import t, tt


class FlextTestsDomains:
    """Test domain objects and fixtures.

    Provides common test data and domain objects used across FLEXT test suites.
    """

    class TestDomainResult:
        """Simple domain result for testing services.

        Implements ResultLike protocol for compatibility with r operations.
        """

        __test__ = False

        def __init__(self, value: str) -> None:
            """Initialize domain result."""
            super().__init__()
            self.value = value

        @property
        def error(self) -> str | None:
            """Get error."""
            return None

        @property
        def is_failure(self) -> bool:
            """Check if failure."""
            return False

        @property
        def is_success(self) -> bool:
            """Check if success."""
            return True

        def unwrap(self) -> FlextTestsDomains.TestDomainResult:
            """Unwrap the result value."""
            return self

    @staticmethod
    def api_response_data(
        status: str = "success",
        *,
        include_data: bool | None = None,
        **custom_fields: t.Tests.object,
    ) -> MutableMapping[str, t.Tests.object]:
        """Create API response test data.

        Args:
            status: Response status
            include_data: Whether to include data field
            **custom_fields: Custom response fields

        Returns:
            API response dictionary

        """
        response: MutableMapping[str, t.Tests.object] = {
            "status": status,
            "timestamp": "2025-01-01T00:00:00Z",
            "request_id": str(uuid.uuid4()),
        }
        if include_data:
            response["data"] = {"test": "data"}
        if status == "error":
            response["error"] = {"code": "TEST_ERROR", "message": "Test error message"}
        response.update(custom_fields)
        return response

    @staticmethod
    def batch_users(
        count: int = 5, **user_overrides: str | bool
    ) -> list[MutableMapping[str, str | bool]]:
        """Create a batch of test users.

        Args:
            count: Number of users to create
            **user_overrides: Common overrides for all users

        Returns:
            List of user dictionaries

        """
        users: list[MutableMapping[str, str | bool]] = []
        for i in range(count):
            user_overrides_copy: dict[str, str | bool] = dict(user_overrides)
            user_overrides_copy["username"] = f"testuser{i}"
            user_overrides_copy["email"] = f"testuser{i}@example.com"
            user_data = FlextTestsDomains.create_user(**user_overrides_copy)
            users.append(user_data)
        return users

    @staticmethod
    def create_configuration(
        service_type: str = "api",
        environment: str = "test",
        **overrides: t.Tests.object,
    ) -> MutableMapping[str, t.Tests.object]:
        """Create test configuration data using factories.

        Args:
            service_type: Type of service configuration
            environment: Environment setting
            **overrides: Additional configuration overrides

        Returns:
            Configuration dictionary

        """
        config_result = tt.model(
            "config", service_type=service_type, environment=environment
        )
        base_config: MutableMapping[str, t.Tests.object] = {
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
        data_type: str = "user", **custom_fields: t.Tests.object
    ) -> MutableMapping[str, t.Tests.object]:
        """Create test payload data.

        Args:
            data_type: Type of data to create
            **custom_fields: Custom field overrides

        Returns:
            Payload dictionary

        """
        payloads: MutableMapping[str, Mapping[str, t.Tests.object]] = {
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
        payload = dict(payloads.get(data_type, {}).items())
        payload.update(custom_fields)
        return payload

    @staticmethod
    def create_service(
        service_type: str = "api", **config: t.Tests.object
    ) -> MutableMapping[str, t.Tests.object]:
        """Create test service configuration.

        Args:
            service_type: Type of service
            **config: Service configuration

        Returns:
            Service configuration dictionary

        """
        base_service: MutableMapping[str, t.Tests.object] = {
            "type": service_type,
            "name": f"test_{service_type}_service",
            "enabled": True,
            "config": FlextTestsDomains.create_configuration(service_type=service_type),
        }
        base_service.update(config)
        return base_service

    @staticmethod
    def create_user(**overrides: str | bool) -> MutableMapping[str, str | bool]:
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
            "user", name=f"{first_name} {last_name}", email=email
        )
        user: MutableMapping[str, str | bool] = {
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
    def invalid_ages() -> list[int]:
        """Get invalid age test cases.

        Returns:
            List of invalid ages

        """
        return [-5, 0, 17, 151, 200]

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


td = FlextTestsDomains

__all__ = ["FlextTestsDomains", "td"]
