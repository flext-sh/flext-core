"""Test factories for test data generation.

Provides comprehensive factories for all test models using native Python patterns.
Replaces factory-boy with pure Python implementations for full type checker compliance.
Uses dataclasses and factory functions with Python 3.13 features.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from itertools import count
from typing import ClassVar, cast, override

from pydantic import BaseModel

from flext_core import r, s, t

from .. import c, m

User = m.Tests.User
"""Type alias for backward-compatible import: ``from .helpers.factories import User``."""

ServiceTestCase = m.ServiceTestCase
"""Type alias for backward-compatible import: ``from .helpers.factories import ServiceTestCase``."""


class GetUserService(s[m.Tests.User]):
    """Service to get a user by ID."""

    user_id: str

    @override
    def execute(self) -> r[m.Tests.User]:
        """Get user by ID."""
        return r[m.Tests.User].ok(
            m.Tests.User(
                id=self.user_id,
                name=f"{c.Services.DEFAULT_USER_NAME_PREFIX}{self.user_id}",
                email=f"user{self.user_id}{c.Services.DEFAULT_EMAIL_DOMAIN}",
            ),
        )


class ValidatingService(s[str]):
    """Service with validation."""

    value_input: str
    min_length: int = c.TestValidation.MIN_LENGTH_DEFAULT

    @override
    def execute(self) -> r[str]:
        """Validate and return value."""
        if len(self.value_input) < self.min_length:
            return r[str].fail(
                f"Value must be at least {self.min_length} characters",
            )
        return r[str].ok(self.value_input.upper())


class FailingService(s[str]):
    """Service that always fails."""

    error_message: str = c.Services.DEFAULT_ERROR_MESSAGE

    @override
    def execute(self) -> r[str]:
        """Always fails."""
        return r[str].fail(self.error_message)


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
    """Factory for m.Tests.User entities using native Python patterns."""

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
    ) -> m.Tests.User:
        """Build a m.Tests.User instance with optional overrides."""
        n = next(cls._counter)
        actual_user_id = user_id if user_id is not None else f"user_{n:03d}"
        actual_name = name if name is not None else cls._next_name()
        actual_email = email if email is not None else f"{actual_user_id}@example.com"
        return m.Tests.User(
            id=actual_user_id,
            name=actual_name,
            email=actual_email,
            active=is_active,
        )

    @classmethod
    def build_batch(cls, size: int) -> list[m.Tests.User]:
        """Build multiple m.Tests.User instances with auto-generated values."""
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
        min_length: int = c.TestValidation.MIN_LENGTH_DEFAULT,
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
        error_message: str = c.Services.DEFAULT_ERROR_MESSAGE,
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
        min_length: int = c.TestValidation.MIN_LENGTH_DEFAULT,
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
        error_message: str = c.Services.DEFAULT_ERROR_MESSAGE,
    ) -> FailingServiceAuto:
        """Build a FailingServiceAuto instance."""
        return FailingServiceAuto.model_construct(error_message=error_message)

    @classmethod
    def build_batch(cls, size: int) -> list[FailingServiceAuto]:
        """Build multiple FailingServiceAuto instances with default error message."""
        return [cls.build() for _ in range(size)]


class ServiceTestCaseFactory:
    """Factory for m.ServiceTestCase."""

    _service_types: ClassVar[list[m.ServiceTestType]] = [
        m.ServiceTestType.GET_USER,
        m.ServiceTestType.VALIDATE,
        m.ServiceTestType.FAIL,
    ]
    _type_index: ClassVar[int] = 0
    _words: ClassVar[list[str]] = ["test", "sample", "example", "demo", "data"]
    _word_index: ClassVar[int] = 0

    @classmethod
    def _next_type(cls) -> m.ServiceTestType:
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
        service_type: m.ServiceTestType | None = None,
        input_value: str | None = None,
        expected_success: bool = True,
        expected_error: str | None = None,
        extra_param: int = c.TestValidation.MIN_LENGTH_DEFAULT,
        description: str | None = None,
    ) -> m.ServiceTestCase:
        """Build a m.ServiceTestCase instance."""
        actual_type = service_type if service_type is not None else cls._next_type()
        actual_input = input_value if input_value is not None else cls._next_word()
        actual_description = (
            description
            if description is not None
            else f"Test case for {actual_type} with {actual_input}"
        )
        return m.ServiceTestCase(
            service_type=actual_type,
            input_value=actual_input,
            expected_success=expected_success,
            expected_error=expected_error,
            extra_param=extra_param,
            description=actual_description,
        )

    @classmethod
    def build_batch(cls, size: int) -> list[m.ServiceTestCase]:
        """Build multiple m.ServiceTestCase instances with auto-generated values."""
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
            m.ServiceTestType,
            type[
                GetUserServiceFactory | ValidatingServiceFactory | FailingServiceFactory
            ],
        ]
    ] = {
        m.ServiceTestType.GET_USER: GetUserServiceFactory,
        m.ServiceTestType.VALIDATE: ValidatingServiceFactory,
        m.ServiceTestType.FAIL: FailingServiceFactory,
    }

    @classmethod
    def create_service(
        cls,
        case: m.ServiceTestCase,
    ) -> s[m.Tests.User] | s[str]:
        """Create appropriate service based on case type using pattern matching."""
        service: s[m.Tests.User] | s[str]
        match case.service_type:
            case m.ServiceTestType.GET_USER:
                service = GetUserServiceFactory.build(user_id=case.input_value)
            case m.ServiceTestType.VALIDATE:
                service = ValidatingServiceFactory.build(
                    value_input=case.input_value,
                    min_length=case.extra_param,
                )
            case m.ServiceTestType.FAIL:
                service = FailingServiceFactory.build(error_message=case.input_value)
            case _:
                msg = f"Unsupported service type: {case.service_type}"
                raise ValueError(msg)
        return service


class TestDataGenerators:
    """Advanced test data generators using comprehensions and patterns."""

    @staticmethod
    def generate_user_success_cases(
        num_cases: int = 3,
    ) -> list[m.ServiceTestCase]:
        """Generate successful user service test cases."""
        return [
            m.ServiceTestCase(
                service_type=m.ServiceTestType.GET_USER,
                input_value=str(i * 100 + 1),
                description=f"Valid user ID {i}",
            )
            for i in range(1, num_cases + 1)
        ]

    @staticmethod
    def generate_validation_success_cases(
        num_cases: int = 2,
    ) -> list[m.ServiceTestCase]:
        """Generate successful validation test cases."""
        return [
            m.ServiceTestCase(
                service_type=m.ServiceTestType.VALIDATE,
                input_value=f"value_{i}",
                description=f"Valid input {i}",
            )
            for i in range(1, num_cases + 1)
        ] + [
            m.ServiceTestCase(
                service_type=m.ServiceTestType.VALIDATE,
                input_value="test",
                extra_param=2,
                description="Custom min length",
            ),
        ]

    @staticmethod
    def generate_validation_failure_cases() -> list[m.ServiceTestCase]:
        """Generate validation failure test cases."""
        return [
            m.ServiceTestCase(
                service_type=m.ServiceTestType.VALIDATE,
                input_value="ab",
                expected_success=False,
                expected_error="must be at least 3 characters",
                description="Too short input",
            ),
            m.ServiceTestCase(
                service_type=m.ServiceTestType.VALIDATE,
                input_value="x",
                expected_success=False,
                expected_error="must be at least 5 characters",
                extra_param=5,
                description="Custom length requirement",
            ),
        ]


class ServiceTestCases:
    """Unified factory for all test cases using advanced patterns."""

    USER_SUCCESS: ClassVar[list[m.ServiceTestCase]] = (
        TestDataGenerators.generate_user_success_cases()
    )
    VALIDATE_SUCCESS: ClassVar[list[m.ServiceTestCase]] = (
        TestDataGenerators.generate_validation_success_cases()
    )
    VALIDATE_FAILURE: ClassVar[list[m.ServiceTestCase]] = (
        TestDataGenerators.generate_validation_failure_cases()
    )

    @staticmethod
    def create_service(
        case: m.ServiceTestCase,
    ) -> s[m.Tests.User] | s[str]:
        """Create appropriate service based on case type."""
        return ServiceFactoryRegistry.create_service(case)


class GenericModelFactory:
    """Factories for generic reusable models (Value, Snapshot, Progress)."""

    @staticmethod
    def operation_context(source: str | None = None) -> m.OperationContext:
        """Create OperationContext value object."""
        return m.OperationContext.model_validate({"source": source})

    @staticmethod
    def service_snapshot(
        name: str,
        version: str | None = None,
        status: str = "active",
    ) -> m.Service:
        """Create ServiceSnapshot."""
        return m.Service(
            name=name,
            version=version,
            status=status,
            metadata=t.Dict({}),
        )

    @staticmethod
    def configuration_snapshot(
        config: dict[str, t.NormalizedValue] | None = None,
        source: str | None = None,
        environment: str | None = None,
    ) -> m.Configuration:
        """Create ConfigurationSnapshot."""
        config_root = cast(
            "dict[str, t.NormalizedValue | BaseModel]",
            dict(config) if config else {},
        )
        return m.Configuration.model_validate({
            "config": t.Dict(config_root),
            "source": source,
            "environment": environment,
        })

    @staticmethod
    def health_status(
        *,
        healthy: bool = True,
        checks: dict[str, bool] | None = None,
    ) -> m.Health:
        """Create HealthStatus."""
        return m.Health.model_validate({
            "healthy": healthy,
            "checks": t.Dict({
                str(key): value for key, value in (checks or {}).items()
            }),
        })

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
            metadata=t.Dict({}),
        )

    @staticmethod
    def conversion_progress() -> m.Conversion:
        """Create ConversionProgress."""
        return m.Conversion(
            converted=[],
            errors=[],
            warnings=[],
            skipped=[],
            metadata=t.Dict({}),
        )


def reset_all_factories() -> None:
    """Reset all factory states for test isolation."""
    UserFactory.reset()
    GetUserServiceFactory.reset()
    ValidatingServiceFactory.reset()
    GetUserServiceAutoFactory.reset()
    ValidatingServiceAutoFactory.reset()
    ServiceTestCaseFactory.reset()
