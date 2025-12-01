"""Test data factories for FLEXT ecosystem tests.

Provides factory pattern implementation for creating test objects using Models.
Factories generate validated test data with sensible defaults and support
field overrides for test scenarios.

Scope: Factory methods for creating test users, configurations, services,
and test operation callables. Includes Models for User, Config, and Service
test data structures. Supports batch generation and test service class creation
with validation logic.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Never

from pydantic import PrivateAttr

from flext_core import FlextResult, FlextService, FlextUtilities
from flext_core._models.entity import FlextModelsEntity
from flext_core.typings import FlextTypes
from flext_tests.typings import FlextTestsTypings


class FlextTestsFactories:
    """Test data factories using factory pattern.

    Provides factory methods for creating test objects using Models.
    """

    class User(FlextModelsEntity.Value):
        """Test user model."""

        id: str
        name: str
        email: str
        active: bool = True

    class Config(FlextModelsEntity.Value):
        """Test configuration model."""

        service_type: str = "api"
        environment: str = "test"
        debug: bool = True
        log_level: str = "DEBUG"
        timeout: int = 30
        max_retries: int = 3

    class Service(FlextModelsEntity.Value):
        """Test service model."""

        id: str
        type: str = "api"
        name: str = ""
        status: str = "active"

    @staticmethod
    def create_user(
        user_id: str | None = None,
        name: str | None = None,
        email: str | None = None,
        **overrides: FlextTestsTypings.TestResultValue,
    ) -> FlextTestsFactories.User:
        """Create a test user.

        Args:
            user_id: Optional user ID
            name: Optional user name
            email: Optional user email
            **overrides: Additional field overrides

        Returns:
            User model instance

        """
        user_data: dict[str, FlextTypes.GeneralValueType] = {
            "id": user_id or str(uuid.uuid4()),
            "name": name or "Test User",
            "email": email
            or f"user_{FlextUtilities.Generators.generate_short_id()}@example.com",
            "active": True,
        }
        user_data.update(overrides)
        return FlextTestsFactories.User.model_validate(user_data)

    @staticmethod
    def create_config(
        service_type: str = "api",
        environment: str = "test",
        **overrides: FlextTestsTypings.TestResultValue,
    ) -> FlextTestsFactories.Config:
        """Create a test configuration.

        Args:
            service_type: Type of service
            environment: Environment name
            **overrides: Additional field overrides

        Returns:
            Config model instance

        """
        config_data: dict[str, FlextTestsTypings.TestResultValue] = {
            "service_type": service_type,
            "environment": environment,
        }
        config_data.update(overrides)
        return FlextTestsFactories.Config.model_validate(config_data)

    @staticmethod
    def create_service(
        service_type: str = "api",
        service_id: str | None = None,
        **overrides: FlextTestsTypings.TestResultValue,
    ) -> FlextTestsFactories.Service:
        """Create a test service.

        Args:
            service_type: Type of service
            service_id: Optional service ID
            **overrides: Additional field overrides

        Returns:
            Service model instance

        """
        service_data: dict[str, FlextTestsTypings.TestResultValue] = {
            "id": service_id or str(uuid.uuid4()),
            "type": service_type,
            "status": "active",
        }
        if "name" not in overrides:
            service_data["name"] = f"Test {service_type} Service"
        # Update with compatible types
        service_data.update(dict(overrides.items()))
        return FlextTestsFactories.Service.model_validate(service_data)

    @staticmethod
    def batch_users(count: int = 5) -> list[FlextTestsFactories.User]:
        """Create a batch of test users.

        Args:
            count: Number of users to create

        Returns:
            List of user model instances

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
        **overrides: FlextTestsTypings.TestResultValue,
    ) -> Callable[..., FlextTestsTypings.TestResultValue]:
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
            # operator.add takes two arguments, wrap it
            def add_op(
                a: FlextTestsTypings.TestResultValue,
                b: FlextTestsTypings.TestResultValue,
            ) -> FlextTestsTypings.TestResultValue:
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    return a + b
                return str(a) + str(b)

            return add_op
        if operation_type == "format":

            def format_op(name: str, value: int = 10) -> str:
                return f"{name}: {value}"

            return format_op
        if operation_type == "error":

            def error_op(**kwargs: FlextTestsTypings.TestResultValue) -> Never:
                msg = overrides.get("error_message", "Test error")
                if isinstance(msg, str):
                    raise ValueError(msg)
                msg = "Test error"
                raise ValueError(msg)

            return error_op
        if operation_type == "type_error":

            def type_error_op(**kwargs: FlextTestsTypings.TestResultValue) -> Never:
                msg = overrides.get("error_message", "Wrong type")
                if isinstance(msg, str):
                    raise TypeError(msg)
                msg = "Wrong type"
                raise TypeError(msg)

            return type_error_op
        return lambda **kwargs: f"unknown operation: {operation_type}"

    @staticmethod
    def create_test_service(
        service_type: str = "test",
        **overrides: FlextTestsTypings.TestResultValue,
    ) -> type:
        """Create a test service class.

        Args:
            service_type: Type of service to create
            **overrides: Additional attributes for the service

        Returns:
            Test service class

        """

        class TestService(FlextService[FlextTestsTypings.TestResultValue]):
            """Generic test service."""

            name: str | None = None
            amount: int | None = None
            enabled: bool | None = None
            _overrides: dict[str, FlextTestsTypings.TestResultValue] = PrivateAttr(
                default_factory=dict
            )

            def __init__(
                self,
                **data: (
                    FlextTypes.ScalarValue
                    | Sequence[FlextTypes.ScalarValue]
                    | Mapping[str, FlextTypes.ScalarValue]
                ),
            ) -> None:
                super().__init__(**data)
                self._overrides.update(overrides)

            def _validate_name_not_empty(self) -> FlextResult[bool]:
                """Validate name is not empty."""
                if self.name is not None and not self.name:
                    return FlextResult[bool].fail("Name is required")
                return FlextResult[bool].ok(True)

            def _validate_amount_non_negative(self) -> FlextResult[bool]:
                """Validate amount is non-negative."""
                if self.amount is not None and self.amount < 0:
                    return FlextResult[bool].fail("Amount must be non-negative")
                return FlextResult[bool].ok(True)

            def _validate_disabled_without_amount(self) -> FlextResult[bool]:
                """Validate disabled service doesn't have amount."""
                if (
                    self.enabled is not None
                    and not self.enabled
                    and self.amount is not None
                    and self.amount > 0
                ):
                    return FlextResult[bool].fail("Cannot have amount when disabled")
                return FlextResult[bool].ok(True)

            def _validate_business_rules_complex(self) -> FlextResult[bool]:
                """Validate business rules for complex service."""
                name_result = self._validate_name_not_empty()
                if name_result.is_failure:
                    return name_result

                amount_result = self._validate_amount_non_negative()
                if amount_result.is_failure:
                    return amount_result

                disabled_result = self._validate_disabled_without_amount()
                if disabled_result.is_failure:
                    return disabled_result

                return FlextResult[bool].ok(True)

            def execute(
                self, **_kwargs: FlextTestsTypings.TestResultValue
            ) -> FlextResult[FlextTestsTypings.TestResultValue]:
                """Execute test operation."""
                if service_type == "user":
                    user_id = "default_123" if overrides.get("default") else "test_123"
                    # Create dict compatible with TestResultValue
                    # dict[str, ...] is a Mapping, which is part of TestResultValue
                    user_data: FlextTestsTypings.TestResultValue = {
                        "user_id": user_id,
                        "email": "test@example.com",
                    }
                    return FlextResult[FlextTestsTypings.TestResultValue].ok(user_data)
                if service_type == "complex":
                    validation = self._validate_business_rules_complex()
                    if validation.is_failure:
                        return FlextResult[FlextTestsTypings.TestResultValue].fail(
                            validation.error or "Validation failed",
                        )
                    result_data: FlextTestsTypings.TestResultValue = {
                        "result": "success"
                    }
                    return FlextResult[FlextTestsTypings.TestResultValue].ok(
                        result_data
                    )
                service_data: FlextTestsTypings.TestResultValue = {
                    "service_type": service_type
                }
                return FlextResult[FlextTestsTypings.TestResultValue].ok(service_data)

            def validate_business_rules(self) -> FlextResult[bool]:
                """Validate business rules for complex service."""
                if service_type == "complex":
                    return self._validate_business_rules_complex()
                return super().validate_business_rules()

            def validate_config(self) -> FlextResult[bool]:
                """Validate config for complex service."""
                if service_type == "complex":
                    if self.name is not None and len(self.name) > 50:
                        return FlextResult[bool].fail("Name too long")
                    if self.amount is not None and self.amount > 1000:
                        return FlextResult[bool].fail("Value too large")
                    return FlextResult[bool].ok(True)
                return FlextResult[bool].ok(True)

        return TestService
