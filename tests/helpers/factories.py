"""Test factories for test data generation.

Provides comprehensive factories for all test models using native Python patterns.
Replaces factory-boy with pure Python implementations for full type checker compliance.
Uses dataclasses and factory functions with Python 3.13 features.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from itertools import count
from typing import ClassVar, override

from flext_core import FlextResult, FlextService, m, t

from ..constants import TestsFlextConstants
from ..models import TestsFlextModels

User = TestsFlextModels.Tests.User
ServiceTestType = TestsFlextModels.Tests.ServiceTestType
ServiceTestCase = TestsFlextModels.Tests.ServiceTestCase


class GetUserService(FlextService[User]):
    """Service to get a user by ID."""

    user_id: str

    @override
    def execute(self) -> FlextResult[User]:
        """Get user by ID."""
        return FlextResult.ok(
            User(
                user_id=self.user_id,
                name=f"{TestsFlextConstants.Services.DEFAULT_USER_NAME_PREFIX}{self.user_id}",
                email=f"user{self.user_id}{TestsFlextConstants.Services.DEFAULT_EMAIL_DOMAIN}",
            ),
        )


class ValidatingService(FlextService[str]):
    """Service with validation."""

    value_input: str
    min_length: int = TestsFlextConstants.TestValidation.MIN_LENGTH_DEFAULT

    @override
    def execute(self) -> FlextResult[str]:
        """Validate and return value."""
        if len(self.value_input) < self.min_length:
            return FlextResult[str].fail(
                f"Value must be at least {self.min_length} characters",
            )
        return FlextResult[str].ok(self.value_input.upper())


class FailingService(FlextService[str]):
    """Service that always fails."""

    error_message: str = TestsFlextConstants.Services.DEFAULT_ERROR_MESSAGE

    @override
    def execute(self) -> FlextResult[str]:
        """Always fails."""
        return FlextResult[str].fail(self.error_message)


class GetUserServiceAuto(GetUserService):
    """Auto-executing GetUserService."""

    auto_execute: ClassVar[bool] = True


class ValidatingServiceAuto(ValidatingService):
    """Auto-executing ValidatingService."""

    auto_execute: ClassVar[bool] = True


class FailingServiceAuto(FailingService):
    """Auto-executing FailingService."""

    auto_execute: ClassVar[bool] = True


class UserFactory:
    """Factory for User entities using native Python patterns."""

    _counter: ClassVar[count[int]] = count(1)
    _names: ClassVar[list[str]] = [
        "Alice Johnson",
        "Bob Smith",
        "Carol Williams",
        "David Brown",
        "Eve Davis",
    ]
    _name_index: ClassVar[int] = 0

    @classmethod
    def _next_name(cls) -> str:
        """Get next name from rotation."""
        name = cls._names[cls._name_index % len(cls._names)]
        cls._name_index += 1
        return name

    @classmethod
    def build(
        cls,
        *,
        user_id: str | None = None,
        name: str | None = None,
        email: str | None = None,
        is_active: bool = True,
    ) -> User:
        """Build a User instance with optional overrides."""
        n = next(cls._counter)
        actual_user_id = user_id if user_id is not None else f"user_{n:03d}"
        actual_name = name if name is not None else cls._next_name()
        actual_email = email if email is not None else f"{actual_user_id}@example.com"
        return User(
            user_id=actual_user_id,
            name=actual_name,
            email=actual_email,
            is_active=is_active,
        )

    @classmethod
    def build_batch(cls, size: int) -> list[User]:
        """Build multiple User instances with auto-generated values."""
        return [cls.build() for _ in range(size)]

    @classmethod
    def reset(cls) -> None:
        """Reset factory state for test isolation."""
        cls._counter = count(1)
        cls._name_index = 0


class GetUserServiceFactory:
    """Factory for GetUserService."""

    _counter: ClassVar[count[int]] = count(1)

    @classmethod
    def build(cls, *, user_id: str | None = None) -> GetUserService:
        """Build a GetUserService instance."""
        n = next(cls._counter)
        actual_user_id = user_id if user_id is not None else f"user_{n:03d}"
        return GetUserService.model_construct(user_id=actual_user_id)

    @classmethod
    def build_batch(cls, size: int) -> list[GetUserService]:
        """Build multiple GetUserService instances with auto-generated values."""
        return [cls.build() for _ in range(size)]

    @classmethod
    def reset(cls) -> None:
        """Reset factory state."""
        cls._counter = count(1)


class ValidatingServiceFactory:
    """Factory for ValidatingService."""

    _words: ClassVar[list[str]] = ["alpha", "bravo", "charlie", "delta", "echo"]
    _word_index: ClassVar[int] = 0

    @classmethod
    def _next_word(cls) -> str:
        """Get next word from rotation."""
        word = cls._words[cls._word_index % len(cls._words)]
        cls._word_index += 1
        return word

    @classmethod
    def build(
        cls,
        *,
        value_input: str | None = None,
        min_length: int = TestsFlextConstants.TestValidation.MIN_LENGTH_DEFAULT,
    ) -> ValidatingService:
        """Build a ValidatingService instance."""
        actual_value = value_input if value_input is not None else cls._next_word()
        return ValidatingService.model_construct(
            value_input=actual_value,
            min_length=min_length,
        )

    @classmethod
    def build_batch(cls, size: int) -> list[ValidatingService]:
        """Build multiple ValidatingService instances with auto-generated values."""
        return [cls.build() for _ in range(size)]

    @classmethod
    def reset(cls) -> None:
        """Reset factory state."""
        cls._word_index = 0


class FailingServiceFactory:
    """Factory for FailingService."""

    @classmethod
    def build(
        cls,
        *,
        error_message: str = TestsFlextConstants.Services.DEFAULT_ERROR_MESSAGE,
    ) -> FailingService:
        """Build a FailingService instance."""
        return FailingService.model_construct(error_message=error_message)

    @classmethod
    def build_batch(cls, size: int) -> list[FailingService]:
        """Build multiple FailingService instances with default error message."""
        return [cls.build() for _ in range(size)]


class GetUserServiceAutoFactory:
    """Factory for GetUserServiceAuto."""

    _counter: ClassVar[count[int]] = count(1)

    @classmethod
    def build(cls, *, user_id: str | None = None) -> GetUserServiceAuto:
        """Build a GetUserServiceAuto instance."""
        n = next(cls._counter)
        actual_user_id = user_id if user_id is not None else f"user_{n:03d}"
        return GetUserServiceAuto.model_construct(user_id=actual_user_id)

    @classmethod
    def build_batch(cls, size: int) -> list[GetUserServiceAuto]:
        """Build multiple GetUserServiceAuto instances with auto-generated values."""
        return [cls.build() for _ in range(size)]

    @classmethod
    def reset(cls) -> None:
        """Reset factory state."""
        cls._counter = count(1)


class ValidatingServiceAutoFactory:
    """Factory for ValidatingServiceAuto."""

    _words: ClassVar[list[str]] = ["alpha", "bravo", "charlie", "delta", "echo"]
    _word_index: ClassVar[int] = 0

    @classmethod
    def _next_word(cls) -> str:
        """Get next word from rotation."""
        word = cls._words[cls._word_index % len(cls._words)]
        cls._word_index += 1
        return word

    @classmethod
    def build(
        cls,
        *,
        value_input: str | None = None,
        min_length: int = TestsFlextConstants.TestValidation.MIN_LENGTH_DEFAULT,
    ) -> ValidatingServiceAuto:
        """Build a ValidatingServiceAuto instance."""
        actual_value = value_input if value_input is not None else cls._next_word()
        return ValidatingServiceAuto.model_construct(
            value_input=actual_value,
            min_length=min_length,
        )

    @classmethod
    def build_batch(cls, size: int) -> list[ValidatingServiceAuto]:
        """Build multiple ValidatingServiceAuto instances with auto-generated values."""
        return [cls.build() for _ in range(size)]

    @classmethod
    def reset(cls) -> None:
        """Reset factory state."""
        cls._word_index = 0


class FailingServiceAutoFactory:
    """Factory for FailingServiceAuto."""

    @classmethod
    def build(
        cls,
        *,
        error_message: str = TestsFlextConstants.Services.DEFAULT_ERROR_MESSAGE,
    ) -> FailingServiceAuto:
        """Build a FailingServiceAuto instance."""
        return FailingServiceAuto.model_construct(error_message=error_message)

    @classmethod
    def build_batch(cls, size: int) -> list[FailingServiceAuto]:
        """Build multiple FailingServiceAuto instances with default error message."""
        return [cls.build() for _ in range(size)]


class ServiceTestCaseFactory:
    """Factory for ServiceTestCase."""

    _service_types: ClassVar[list[ServiceTestType]] = [
        ServiceTestType.GET_USER,
        ServiceTestType.VALIDATE,
        ServiceTestType.FAIL,
    ]
    _type_index: ClassVar[int] = 0
    _words: ClassVar[list[str]] = ["test", "sample", "example", "demo", "data"]
    _word_index: ClassVar[int] = 0

    @classmethod
    def _next_type(cls) -> ServiceTestType:
        """Get next service type from rotation."""
        service_type = cls._service_types[cls._type_index % len(cls._service_types)]
        cls._type_index += 1
        return service_type

    @classmethod
    def _next_word(cls) -> str:
        """Get next word from rotation."""
        word = cls._words[cls._word_index % len(cls._words)]
        cls._word_index += 1
        return word

    @classmethod
    def build(
        cls,
        *,
        service_type: ServiceTestType | None = None,
        input_value: str | None = None,
        expected_success: bool = True,
        expected_error: str | None = None,
        extra_param: int = TestsFlextConstants.TestValidation.MIN_LENGTH_DEFAULT,
        description: str | None = None,
    ) -> ServiceTestCase:
        """Build a ServiceTestCase instance."""
        actual_type = service_type if service_type is not None else cls._next_type()
        actual_input = input_value if input_value is not None else cls._next_word()
        actual_description = (
            description
            if description is not None
            else f"Test case for {actual_type} with {actual_input}"
        )
        return ServiceTestCase(
            service_type=actual_type,
            input_value=actual_input,
            expected_success=expected_success,
            expected_error=expected_error,
            extra_param=extra_param,
            description=actual_description,
        )

    @classmethod
    def build_batch(cls, size: int) -> list[ServiceTestCase]:
        """Build multiple ServiceTestCase instances with auto-generated values."""
        return [cls.build() for _ in range(size)]

    @classmethod
    def reset(cls) -> None:
        """Reset factory state."""
        cls._type_index = 0
        cls._word_index = 0


class ServiceFactoryRegistry:
    """Registry for service factories using pattern matching."""

    _factories: ClassVar[
        dict[
            ServiceTestType,
            type[
                GetUserServiceFactory | ValidatingServiceFactory | FailingServiceFactory
            ],
        ]
    ] = {
        ServiceTestType.GET_USER: GetUserServiceFactory,
        ServiceTestType.VALIDATE: ValidatingServiceFactory,
        ServiceTestType.FAIL: FailingServiceFactory,
    }

    @classmethod
    def create_service(
        cls,
        case: ServiceTestCase,
    ) -> FlextService[User] | FlextService[str]:
        """Create appropriate service based on case type using pattern matching."""
        service: FlextService[User] | FlextService[str]
        match case.service_type:
            case ServiceTestType.GET_USER:
                service = GetUserServiceFactory.build(user_id=case.input_value)
            case ServiceTestType.VALIDATE:
                service = ValidatingServiceFactory.build(
                    value_input=case.input_value,
                    min_length=case.extra_param,
                )
            case ServiceTestType.FAIL:
                service = FailingServiceFactory.build(error_message=case.input_value)
            case _:
                msg = f"Unsupported service type: {case.service_type}"
                raise ValueError(msg)
        return service


class TestDataGenerators:
    """Advanced test data generators using comprehensions and patterns."""

    @staticmethod
    def generate_user_success_cases(num_cases: int = 3) -> list[ServiceTestCase]:
        """Generate successful user service test cases."""
        return [
            ServiceTestCase(
                service_type=ServiceTestType.GET_USER,
                input_value=str(i * 100 + 1),
                description=f"Valid user ID {i}",
            )
            for i in range(1, num_cases + 1)
        ]

    @staticmethod
    def generate_validation_success_cases(num_cases: int = 2) -> list[ServiceTestCase]:
        """Generate successful validation test cases."""
        return [
            ServiceTestCase(
                service_type=ServiceTestType.VALIDATE,
                input_value=f"value_{i}",
                description=f"Valid input {i}",
            )
            for i in range(1, num_cases + 1)
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


class ServiceTestCases:
    """Unified factory for all test cases using advanced patterns."""

    USER_SUCCESS: ClassVar[list[ServiceTestCase]] = (
        TestDataGenerators.generate_user_success_cases()
    )
    VALIDATE_SUCCESS: ClassVar[list[ServiceTestCase]] = (
        TestDataGenerators.generate_validation_success_cases()
    )
    VALIDATE_FAILURE: ClassVar[list[ServiceTestCase]] = (
        TestDataGenerators.generate_validation_failure_cases()
    )

    @staticmethod
    def create_service(case: ServiceTestCase) -> FlextService[User] | FlextService[str]:
        """Create appropriate service based on case type."""
        return ServiceFactoryRegistry.create_service(case)


class GenericModelFactory:
    """Factories for generic reusable models (Value, Snapshot, Progress)."""

    @staticmethod
    def operation_context(source: str | None = None) -> m.OperationContext:
        """Create OperationContext value object."""
        return m.OperationContext(source=source)

    @staticmethod
    def service_snapshot(
        name: str,
        version: str | None = None,
        status: str = "active",
    ) -> m.Service:
        """Create ServiceSnapshot."""
        return m.Service(name=name, version=version, status=status)

    @staticmethod
    def configuration_snapshot(
        config: dict[str, t.ContainerValue] | None = None,
        source: str | None = None,
        environment: str | None = None,
    ) -> m.Configuration:
        """Create ConfigurationSnapshot."""
        return m.Configuration(
            config=m.Dict.model_validate(config or {}),
            source=source,
            environment=environment,
        )

    @staticmethod
    def health_status(
        *,
        healthy: bool = True,
        checks: dict[str, bool] | None = None,
    ) -> m.Health:
        """Create HealthStatus."""
        return m.Health(healthy=healthy, checks=m.Dict.model_validate(checks or {}))

    @staticmethod
    def operation_progress(
        success: int = 0,
        failure: int = 0,
        skipped: int = 0,
    ) -> m.Operation:
        """Create OperationProgress."""
        return m.Operation(
            success_count=success,
            failure_count=failure,
            skipped_count=skipped,
        )

    @staticmethod
    def conversion_progress() -> m.Conversion:
        """Create ConversionProgress."""
        return m.Conversion()


def reset_all_factories() -> None:
    """Reset all factory states for test isolation."""
    UserFactory.reset()
    GetUserServiceFactory.reset()
    ValidatingServiceFactory.reset()
    GetUserServiceAutoFactory.reset()
    ValidatingServiceAutoFactory.reset()
    ServiceTestCaseFactory.reset()
