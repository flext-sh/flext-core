"""Test domain objects and fixtures for FLEXT ecosystem.

Provides reusable domain objects, test data, and fixtures.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import uuid

from flext_core import FlextCore


class FlextTestsDomains:
    """Test domain objects and fixtures.

    Provides common test data and domain objects used across FLEXT test suites.
    """

    @staticmethod
    def create_configuration(
        service_type: str = "api",
        environment: str = "test",
        **overrides: FlextCore.Types.Dict,
    ) -> FlextCore.Types.Dict:
        """Create test configuration data.

        Args:
            service_type: Type of service configuration
            environment: Environment setting
            **overrides: Additional configuration overrides

        Returns:
            Configuration dictionary

        """
        base_config: FlextCore.Types.Dict = {
            "service_type": service_type,
            "environment": environment,
            "debug": True,
            "log_level": "DEBUG",
            "timeout": 30,
            "max_retries": 3,
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
        **custom_fields: FlextCore.Types.Dict,
    ) -> FlextCore.Types.Dict:
        """Create test payload data.

        Args:
            data_type: Type of data to create
            **custom_fields: Custom field overrides

        Returns:
            Payload dictionary

        """
        payloads: dict[str, FlextCore.Types.Dict] = {
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

        payload = payloads.get(data_type, {})
        payload.update(custom_fields)
        return payload

    @staticmethod
    def api_response_data(
        status: str = "success",
        include_data: bool | None = None,
        **custom_fields: FlextCore.Types.Dict,
    ) -> FlextCore.Types.Dict:
        """Create API response test data.

        Args:
            status: Response status
            include_data: Whether to include data field
            **custom_fields: Custom response fields

        Returns:
            API response dictionary

        """
        response: FlextCore.Types.Dict = {
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
        **config: FlextCore.Types.Dict,
    ) -> FlextCore.Types.Dict:
        """Create test service configuration.

        Args:
            service_type: Type of service
            **config: Service configuration

        Returns:
            Service configuration dictionary

        """
        base_service: FlextCore.Types.Dict = {
            "type": service_type,
            "name": f"test_{service_type}_service",
            "enabled": True,
            "config": FlextTestsDomains.create_configuration(service_type=service_type),
        }
        base_service.update(config)
        return base_service

    @staticmethod
    def create_user(**overrides: str | bool) -> dict[str, str | bool]:
        """Create test user data.

        Args:
            **overrides: User field overrides

        Returns:
            User data dictionary

        """
        user: dict[str, str | bool] = {
            "id": str(uuid.uuid4()),
            "username": "testuser",
            "email": "test@example.com",
            "first_name": "Test",
            "last_name": "User",
            "active": True,
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
        }
        user.update(overrides)
        return user
