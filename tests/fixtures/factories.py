"""Factory-boy factories for test data generation.

Provides comprehensive factories for all test models using factory-boy patterns.
Reduces boilerplate code and enables dynamic test data generation with advanced
Python 3.13 features like pattern matching and type annotations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Generic, TypeVar

import factory
from factory import Faker
from flext_core import FlextModels, FlextResult, FlextService

from tests.fixtures.constants import TestConstants

# =========================================================================
# Type Variables for Generic Factories
# =========================================================================

T = TypeVar("T")
ServiceT = TypeVar("ServiceT", bound=FlextService)

# =========================================================================
# Test Models
# =========================================================================


class User(FlextModels.Entity):
    """Test user entity."""

    user_id: str
    name: str
    email: str
    is_active: bool = True


class ServiceTestType(StrEnum):
    """Service test types."""

    GET_USER = "get_user"
    VALIDATE = "validate"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class ServiceTestCase:
    """Test case data container (not a test class)."""

    service_type: ServiceTestType
    input_value: str
    expected_success: bool = True
    expected_error: str | None = None
    extra_param: int = TestConstants.Validation.MIN_LENGTH_DEFAULT
    description: str = field(default="", compare=False)


# =========================================================================
# Service Implementations
# =========================================================================


class GetUserService(FlextService[User]):
    """Service to get a user by ID."""

    user_id: str

    def execute(self, **_kwargs: object) -> FlextResult[User]:
        """Get user by ID."""
        return FlextResult.ok(
            User(
                user_id=self.user_id,
                name=f"{TestConstants.Services.DEFAULT_USER_NAME_PREFIX}{self.user_id}",
                email=f"user{self.user_id}{TestConstants.Services.DEFAULT_EMAIL_DOMAIN}",
            ),
        )


class ValidatingService(FlextService[str]):
    """Service with validation."""

    value_input: str
    min_length: int = TestConstants.Validation.MIN_LENGTH_DEFAULT

    def execute(self, **_kwargs: object) -> FlextResult[str]:
        """Validate and return value."""
        if len(self.value_input) < self.min_length:
            return FlextResult.fail(
                f"Value must be at least {self.min_length} characters",
            )
        return FlextResult.ok(self.value_input.upper())


class FailingService(FlextService[str]):
    """Service that always fails."""

    error_message: str = TestConstants.Services.DEFAULT_ERROR_MESSAGE

    def execute(self, **_kwargs: object) -> FlextResult[str]:
        """Always fails."""
        return FlextResult.fail(self.error_message)


# Auto-execute variants
class GetUserServiceAuto(GetUserService):
    """Auto-executing GetUserService."""

    auto_execute: ClassVar[bool] = True


class ValidatingServiceAuto(ValidatingService):
    """Auto-executing ValidatingService."""

    auto_execute: ClassVar[bool] = True


class FailingServiceAuto(FailingService):
    """Auto-executing FailingService."""

    auto_execute: ClassVar[bool] = True


# =========================================================================
# Factory-Boy Factories
# =========================================================================


class UserFactory(factory.Factory):
    """Factory for User entities."""

    class Meta:
        model = User

    user_id = factory.Sequence(lambda n: f"user_{n:03d}")
    name = Faker("name")
    email = factory.LazyAttribute(lambda obj: f"{obj.user_id}@example.com")
    is_active = True


class GetUserServiceFactory(factory.Factory):
    """Factory for GetUserService."""

    class Meta:
        model = GetUserService

    user_id = factory.Sequence(lambda n: f"user_{n:03d}")


class ValidatingServiceFactory(factory.Factory):
    """Factory for ValidatingService."""

    class Meta:
        model = ValidatingService

    value_input = Faker("word")
    min_length = TestConstants.Validation.MIN_LENGTH_DEFAULT


class FailingServiceFactory(factory.Factory):
    """Factory for FailingService."""

    class Meta:
        model = FailingService

    error_message = TestConstants.Services.DEFAULT_ERROR_MESSAGE


class GetUserServiceAutoFactory(factory.Factory):
    """Factory for GetUserServiceAuto."""

    class Meta:
        model = GetUserServiceAuto

    user_id = factory.Sequence(lambda n: f"user_{n:03d}")


class ValidatingServiceAutoFactory(factory.Factory):
    """Factory for ValidatingServiceAuto."""

    class Meta:
        model = ValidatingServiceAuto

    value_input = Faker("word")
    min_length = TestConstants.Validation.MIN_LENGTH_DEFAULT


class FailingServiceAutoFactory(factory.Factory):
    """Factory for FailingServiceAuto."""

    class Meta:
        model = FailingServiceAuto

    error_message = TestConstants.Services.DEFAULT_ERROR_MESSAGE


class ServiceTestCaseFactory(factory.Factory):
    """Factory for ServiceTestCase."""

    class Meta:
        model = ServiceTestCase

    service_type = factory.Iterator(ServiceTestType)
    input_value = Faker("word")
    expected_success = True
    expected_error = None
    extra_param = TestConstants.Validation.MIN_LENGTH_DEFAULT
    description = factory.LazyAttribute(
        lambda obj: f"Test case for {obj.service_type} with {obj.input_value}",
    )


# =========================================================================
# Advanced Factory Patterns with Python 3.13
# =========================================================================


class ServiceFactoryRegistry(Generic[ServiceT]):
    """Registry for service factories using pattern matching."""

    _factories: ClassVar[dict[ServiceTestType, type[factory.Factory]]] = {
        ServiceTestType.GET_USER: GetUserServiceFactory,
        ServiceTestType.VALIDATE: ValidatingServiceFactory,
        ServiceTestType.FAIL: FailingServiceFactory,
    }

    @classmethod
    def create_service(
        cls, case: ServiceTestCase,
    ) -> FlextService[User] | FlextService[str]:
        """Create appropriate service based on case type using pattern matching."""
        factory_class = cls._factories.get(case.service_type)
        if not factory_class:
            msg = f"No factory for service type: {case.service_type}"
            raise ValueError(msg)

        # Create service instance using factory
        service: FlextService[User] | FlextService[str]

        match case.service_type:
            case ServiceTestType.GET_USER:
                service = factory_class.build(user_id=case.input_value)
            case ServiceTestType.VALIDATE:
                service = factory_class.build(
                    value_input=case.input_value, min_length=case.extra_param,
                )
            case ServiceTestType.FAIL:
                service = factory_class.build(error_message=case.input_value)
            case _ as unreachable:
                msg = f"Unreachable case: {unreachable}"
                raise RuntimeError(msg)

        return service


# =========================================================================
# Test Data Generators
# =========================================================================


class TestDataGenerators:
    """Advanced test data generators using comprehensions and patterns."""

    @staticmethod
    def generate_user_success_cases(count: int = 3) -> list[ServiceTestCase]:
        """Generate successful user service test cases."""
        return [
            ServiceTestCase(
                service_type=ServiceTestType.GET_USER,
                input_value=str(i * 100 + 1),  # "1", "101", "201" etc.
                description=f"Valid user ID {i}",
            )
            for i in range(1, count + 1)
        ]

    @staticmethod
    def generate_validation_success_cases(count: int = 2) -> list[ServiceTestCase]:
        """Generate successful validation test cases."""
        return [
            ServiceTestCase(
                service_type=ServiceTestType.VALIDATE,
                input_value=f"value_{i}",
                description=f"Valid input {i}",
            )
            for i in range(1, count + 1)
        ] + [
            ServiceTestCase(
                service_type=ServiceTestType.VALIDATE,
                input_value="test",
                extra_param=2,
                description="Custom min length",
            ),
        ]

    @staticmethod
    def generate_validation_failure_cases() -> list[ServiceTestCase]:
        """Generate validation failure test cases."""
        return [
            ServiceTestCase(
                service_type=ServiceTestType.VALIDATE,
                input_value="ab",
                expected_success=False,
                expected_error="must be at least 3 characters",
                description="Too short input",
            ),
            ServiceTestCase(
                service_type=ServiceTestType.VALIDATE,
                input_value="x",
                expected_success=False,
                expected_error="must be at least 5 characters",
                extra_param=5,
                description="Custom length requirement",
            ),
        ]


# =========================================================================
# Unified Test Cases Factory
# =========================================================================


class ServiceTestCases:
    """Unified factory for all test cases using advanced patterns."""

    # Success cases for GetUserService - dynamically generated
    USER_SUCCESS: ClassVar[list[ServiceTestCase]] = (
        TestDataGenerators.generate_user_success_cases()
    )

    # Validation success cases - dynamically generated
    VALIDATE_SUCCESS: ClassVar[list[ServiceTestCase]] = (
        TestDataGenerators.generate_validation_success_cases()
    )

    # Validation failure cases - dynamically generated
    VALIDATE_FAILURE: ClassVar[list[ServiceTestCase]] = (
        TestDataGenerators.generate_validation_failure_cases()
    )

    @staticmethod
    def create_service(
        case: ServiceTestCase,
    ) -> FlextService[User] | FlextService[str]:
        """Create appropriate service based on case type."""
        return ServiceFactoryRegistry.create_service(case)
