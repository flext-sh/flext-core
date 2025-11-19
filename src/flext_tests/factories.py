"""Test data factories for FLEXT ecosystem.

Provides factory pattern for creating test objects.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import uuid


class FlextTestsFactories:
    """Test data factories using factory pattern.

    Provides factory methods for creating test objects.
    """

    @staticmethod
    def create_user(
        user_id: str | None = None,
        name: str | None = None,
        email: str | None = None,
        **overrides: object,
    ) -> dict[str, object]:
        """Create a test user.

        Args:
            user_id: Optional user ID
            name: Optional user name
            email: Optional user email
            **overrides: Additional field overrides

        Returns:
            User dictionary

        """
        user: dict[str, object] = {
            "id": user_id or str(uuid.uuid4()),
            "name": name or "Test User",
            "email": email or f"user_{uuid.uuid4().hex[:8]}@example.com",
            "active": True,
        }
        user.update(overrides)
        return user

    @staticmethod
    def create_config(
        service_type: str = "api",
        environment: str = "test",
        **overrides: object,
    ) -> dict[str, object]:
        """Create a test configuration.

        Args:
            service_type: Type of service
            environment: Environment name
            **overrides: Additional field overrides

        Returns:
            Configuration dictionary

        """
        config: dict[str, object] = {
            "service_type": service_type,
            "environment": environment,
            "debug": True,
            "log_level": "DEBUG",
            "timeout": 30,
            "max_retries": 3,
        }
        config.update(overrides)
        return config

    @staticmethod
    def create_service(
        service_type: str = "api",
        service_id: str | None = None,
        **overrides: object,
    ) -> dict[str, object]:
        """Create a test service.

        Args:
            service_type: Type of service
            service_id: Optional service ID
            **overrides: Additional field overrides

        Returns:
            Service dictionary

        """
        service: dict[str, object] = {
            "id": service_id or str(uuid.uuid4()),
            "type": service_type,
            "name": f"Test {service_type} Service",
            "status": "active",
        }
        service.update(overrides)
        return service

    @staticmethod
    def batch_users(count: int = 5) -> list[dict[str, object]]:
        """Create a batch of test users.

        Args:
            count: Number of users to create

        Returns:
            List of user dictionaries

        """
        return [
            FlextTestsFactories.create_user(
                name=f"User {i}",
                email=f"user{i}@example.com",
            )
            for i in range(count)
        ]
