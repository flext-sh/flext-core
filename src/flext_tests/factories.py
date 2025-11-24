"""Test data factories for FLEXT ecosystem.

Provides factory pattern for creating test objects.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import operator
import uuid
from collections.abc import Callable
from typing import Never

from flext_core import FlextResult, FlextService


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

    @staticmethod
    def create_test_operation(
        operation_type: str = "simple",
        **overrides: object,
    ) -> Callable[..., object]:
        """Create a test operation callable.

        Args:
            operation_type: Type of operation ('simple', 'add', 'format', 'error')
            **overrides: Additional configuration

        Returns:
            Test operation callable

        """
        if operation_type == "simple":
            return lambda **kwargs: "success"
        if operation_type == "add":
            return operator.add
        if operation_type == "format":
            return lambda name, value=10: f"{name}: {value}"
        if operation_type == "error":

            def error_op(**kwargs: object) -> Never:
                msg = overrides.get("error_message", "Test error")
                raise ValueError(msg)

            return error_op
        if operation_type == "type_error":

            def type_error_op(**kwargs: object) -> Never:
                msg = overrides.get("error_message", "Wrong type")
                raise TypeError(msg)

            return type_error_op
        return lambda **kwargs: f"unknown operation: {operation_type}"

    @staticmethod
    def create_test_service(
        service_type: str = "test",
        **overrides: object,
    ) -> type:
        """Create a test service class.

        Args:
            service_type: Type of service to create
            **overrides: Additional attributes for the service

        Returns:
            Test service class

        """

        class TestService(FlextService[object]):
            """Generic test service."""

            name: str | None = None
            amount: int | None = None
            enabled: bool | None = None

            def __init__(self, **data: object) -> None:
                super().__init__(**data)
                # Store overrides for use in methods
                self._overrides = overrides

            def execute(self, **kwargs: object) -> FlextResult[object]:
                """Execute test operation."""
                if service_type == "user":
                    user_id = "default_123" if overrides.get("default") else "test_123"
                    return FlextResult[object].ok({
                        "user_id": user_id,
                        "email": "test@example.com",
                    })
                if service_type == "complex":
                    # Simulate complex service with validation
                    if (
                        hasattr(self, "name")
                        and self.name is not None
                        and not self.name
                    ):
                        return FlextResult.fail("Name is required")
                    if (
                        hasattr(self, "amount")
                        and self.amount is not None
                        and self.amount < 0
                    ):
                        return FlextResult.fail("Amount must be non-negative")
                    is_disabled_with_amount = (
                        hasattr(self, "enabled")
                        and self.enabled is not None
                        and not self.enabled
                        and hasattr(self, "amount")
                        and self.amount is not None
                        and self.amount > 0
                    )
                    if is_disabled_with_amount:
                        return FlextResult.fail("Cannot have amount when disabled")
                    return FlextResult[object].ok({"result": "success"})
                return FlextResult[object].ok({"service_type": service_type})

            def validate_business_rules(self) -> FlextResult[bool]:
                """Validate business rules for complex service."""
                if service_type == "complex":
                    if (
                        hasattr(self, "name")
                        and self.name is not None
                        and not self.name
                    ):
                        return FlextResult[bool].fail("Name is required")
                    if (
                        hasattr(self, "amount")
                        and self.amount is not None
                        and self.amount < 0
                    ):
                        return FlextResult[bool].fail("Value must be non-negative")
                    is_disabled_with_amount = (
                        hasattr(self, "enabled")
                        and self.enabled is not None
                        and not self.enabled
                        and hasattr(self, "amount")
                        and self.amount is not None
                        and self.amount > 0
                    )
                    if is_disabled_with_amount:
                        return FlextResult[bool].fail(
                            "Cannot have amount when disabled"
                        )
                    return FlextResult[bool].ok(True)
                return super().validate_business_rules()

            def validate_config(self) -> FlextResult[bool]:
                """Validate config for complex service."""
                if service_type == "complex":
                    if (
                        hasattr(self, "name")
                        and self.name is not None
                        and len(self.name) > 50
                    ):
                        return FlextResult[bool].fail("Name too long")
                    if (
                        hasattr(self, "amount")
                        and self.amount is not None
                        and self.amount > 1000
                    ):
                        return FlextResult[bool].fail("Value too large")
                    return FlextResult[bool].ok(True)
                return super().validate_config()

        return TestService
