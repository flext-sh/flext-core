"""Unified factories for flext-core tests using factory_boy and pytest ecosystem.

Advanced factory patterns with comprehensive test data generation using:
- factory_boy for object factories
- pytest-benchmark for performance data
- faker for realistic data generation
- Pydantic models for type safety

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import uuid
from typing import Any, override

import factory
import factory.fuzzy
from factory.base import DictFactory, Factory
from factory.declarations import (
    LazyAttribute,
    LazyFunction,
    Sequence,
    SubFactory,
    Trait,
)
from factory.faker import Faker

from flext_core import FlextResult
from flext_core.exceptions import FlextError, FlextValidationError
from flext_core.models import FlextModel, FlextValue
from flext_core.root_models import FlextMetadata, FlextVersion
from flext_core.typings import FlextTypes

JsonDict = FlextTypes.Core.JsonDict


class BaseTestEntity(FlextModel):
    """Base entity for testing with proper FlextCore integration."""

    # Using proper FlextCore types for compatibility
    name: str = "test_entity"
    description: str = ""
    active: bool = True
    version: FlextVersion = FlextVersion(1)
    metadata: FlextMetadata = FlextMetadata({})

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for test entity."""
        if not self.name.strip():
            return FlextResult[None].fail(
                "Entity name cannot be empty",
                error_code="INVALID_NAME",
            )
        return FlextResult[None].ok(None)


class BaseTestValueObject(FlextValue):
    """Base value object for testing with proper FlextCore integration."""

    value: str = "test_value"

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for test value object."""
        if not self.value:
            return FlextResult[None].fail(
                "Value cannot be empty",
                error_code="INVALID_VALUE",
            )
        return FlextResult[None].ok(None)


class UserDataFactory(DictFactory):  # type: ignore[misc]  # Factory inheritance patterns
    """Factory for generating user test data with realistic attributes."""

    class Meta:  # type: ignore[misc]  # Factory meta compatibility
        model = dict

    id = LazyFunction(lambda: str(uuid.uuid4()))
    username = Sequence(lambda n: f"user_{n}")  # type: ignore[arg-type]  # Factory sequence patterns
    email = LazyAttribute(lambda obj: f"{obj.username}@example.com")  # type: ignore[arg-type]  # Factory lazy patterns
    first_name = Faker("first_name")
    last_name = Faker("last_name")
    age = factory.fuzzy.FuzzyInteger(18, 80)
    is_active = True
    created_at = Faker("date_time_this_year")

    # Traits for different user types
    class Params:
        is_admin = Trait(
            is_active=True,
            permissions=["read", "write", "admin"],
            role="administrator",
        )
        is_guest = Trait(
            is_active=True,
            permissions=["read"],
            role="guest",
        )
        is_inactive = Trait(
            is_active=False,
            deactivated_at=Faker("date_time_this_month"),
        )

    # Complex nested data
    profile = LazyFunction(
        lambda: {
            "bio": "Sample bio text",
            "location": "Sample City",
            "avatar_url": "https://example.com/avatar.jpg",
        }
    )

    preferences = LazyFunction(
        lambda: {
            "theme": factory.fuzzy.FuzzyChoice(["light", "dark", "auto"]).fuzz(),
            "notifications": {
                "email": True,
                "push": False,
                "sms": False,
            },
            "language": "en_US",
            "timezone": "UTC",
        }
    )


class ConfigurationFactory(DictFactory):  # type: ignore[misc]  # Factory inheritance patterns
    """Factory for generating configuration test data."""

    class Meta:  # type: ignore[misc]  # Factory meta compatibility
        model = dict

    name = Sequence(lambda n: f"config_{n}")  # type: ignore[arg-type]  # Factory sequence patterns
    version = "1.0.0"
    debug = False

    # Database configuration with sub-factory
    database = SubFactory("tests.support.factories.DatabaseConfigFactory")

    # Logging configuration
    logging = LazyFunction(
        lambda: {
            "level": factory.fuzzy.FuzzyChoice(
                [
                    "DEBUG",
                    "INFO",
                    "WARNING",
                    "ERROR",
                ]
            ).fuzz(),
            "format": "json",
            "handlers": ["console", "file"],
            "file_path": "/var/log/app.log",
        }
    )

    # Feature flags
    features = LazyFunction(
        lambda: {
            "cache_enabled": factory.fuzzy.FuzzyChoice([True, False]).fuzz(),
            "metrics_enabled": True,
            "auth_required": True,
            "rate_limiting": factory.fuzzy.FuzzyChoice([True, False]).fuzz(),
        }
    )

    class Params:
        production = Trait(
            debug=False,
            logging__level="INFO",
            features__cache_enabled=True,
        )
        development = Trait(
            debug=True,
            logging__level="DEBUG",
            features__rate_limiting=False,
        )


class DatabaseConfigFactory(DictFactory):  # type: ignore[misc]  # Factory inheritance patterns
    """Factory for database configuration data."""

    class Meta:  # type: ignore[misc]  # Factory meta compatibility
        model = dict

    host = "localhost"
    port = factory.fuzzy.FuzzyInteger(3000, 9999)
    database = Sequence(lambda n: f"testdb_{n}")  # type: ignore[arg-type]  # Factory sequence patterns
    username = "testuser"
    password = "testpass"
    pool_size = factory.fuzzy.FuzzyInteger(5, 20)
    timeout = 30
    ssl_enabled = False

    # Build URL from components
    url = LazyAttribute(
        lambda obj: f"postgresql://{obj.username}:{obj.password}@{obj.host}:{obj.port}/{obj.database}",  # type: ignore[arg-type]  # Factory lazy patterns
    )


class ServiceDefinitionFactory(DictFactory):  # type: ignore[misc]  # Factory inheritance patterns
    """Factory for service definition data."""

    class Meta:  # type: ignore[misc]  # Factory meta compatibility
        model = dict

    name = Sequence(lambda n: f"service_{n}")  # type: ignore[arg-type]  # Factory sequence patterns
    version = factory.fuzzy.FuzzyChoice(["1.0.0", "1.1.0", "2.0.0"])
    description = Faker("sentence", nb_words=8)

    endpoints = LazyFunction(
        lambda: [
            {"path": "/health", "method": "GET", "description": "Health check"},
            {"path": "/api/v1/users", "method": "GET", "description": "List users"},
            {"path": "/api/v1/users", "method": "POST", "description": "Create user"},
        ]
    )

    dependencies = LazyFunction(
        lambda: [
            {"name": "database", "version": ">=1.0.0", "required": True},
            {"name": "cache", "version": ">=2.0.0", "required": False},
        ]
    )

    health_check = LazyFunction(
        lambda: {
            "endpoint": "/health",
            "interval": 30,
            "timeout": 5,
            "retries": 3,
        }
    )


class FlextResultFactory:
    """Factory for FlextResult objects."""

    @staticmethod
    def create_success(data: Any = "test_data") -> FlextResult[Any]:
        """Create successful FlextResult."""
        return FlextResult[Any].ok(data)

    @staticmethod
    def create_failure(
        error: str = "Test error",
        error_code: str = "TEST_ERROR",
    ) -> FlextResult[Any]:
        """Create failed FlextResult."""
        return FlextResult[Any].fail(error, error_code=error_code)

    @staticmethod
    def create_validation_failure(field: str = "test_field") -> FlextResult[Any]:
        """Create validation failure FlextResult."""
        return FlextResult[Any].fail(
            f"Validation failed for field: {field}",
            error_code="VALIDATION_ERROR",
        )


class ExceptionFactory:
    """Factory for creating various exception types."""

    @staticmethod
    def create_domain_error(
        message: str = "Domain error occurred",
        error_code: str = "DOMAIN_ERROR",
    ) -> FlextError:  # type: ignore[return-value]  # FlextError from dynamic generation
        """Create FlextError."""
        return FlextError(message, error_code=error_code)  # type: ignore[call-arg]  # Dynamic exception args

    @staticmethod
    def create_validation_error(
        message: str = "Validation failed",
        error_code: str = "VALIDATION_ERROR",
    ) -> FlextValidationError:  # type: ignore[return-value]  # FlextValidationError from dynamic generation
        """Create FlextValidationError."""
        return FlextValidationError(message, error_code=error_code)  # type: ignore[call-arg]  # Dynamic exception args


class TestEntityFactory(Factory[BaseTestEntity]):
    """Factory for creating FlextEntity test objects."""

    class Meta:  # type: ignore[misc]  # Factory meta compatibility
        model = BaseTestEntity

    name = Sequence(lambda n: f"test_entity_{n}")  # type: ignore[arg-type]  # Factory sequence patterns
    description = Faker("sentence", nb_words=6)
    active = True
    version = LazyFunction(
        lambda: FlextVersion(factory.fuzzy.FuzzyInteger(1, 10).fuzz())
    )

    metadata = LazyFunction(
        lambda: FlextMetadata(
            {
                "created_by": "test_user",
                "tags": ["test", "automated"],
                "priority": factory.fuzzy.FuzzyChoice(["low", "medium", "high"]).fuzz(),
            }
        )
    )

    class Params:
        inactive = Trait(
            active=False,
            metadata=LazyFunction(lambda: FlextMetadata({"status": "deactivated"})),
        )
        high_priority = Trait(
            metadata=LazyFunction(
                lambda: FlextMetadata(
                    {
                        "priority": "high",
                        "urgent": True,
                    }
                )
            ),
        )


class TestValueObjectFactory(Factory[BaseTestValueObject]):
    """Factory for creating FlextValue test objects."""

    class Meta:  # type: ignore[misc]  # Factory meta compatibility
        model = BaseTestValueObject

    value = Sequence(lambda n: f"test_value_{n}")  # type: ignore[arg-type]  # Factory sequence patterns

    class Params:
        empty = Trait(value="")
        long = Trait(value=LazyFunction(lambda: "x" * 1000))
        unicode = Trait(value="🚀 Test Value 测试 مرحبا")


# Batch creation utilities
class BatchFactory:
    """Utility for creating batches of test objects."""

    @staticmethod
    def create_user_batch(count: int = 5, **kwargs: Any) -> list[JsonDict]:
        """Create batch of users."""
        return UserDataFactory.create_batch(count, **kwargs)

    @staticmethod
    def create_config_batch(count: int = 3, **kwargs: Any) -> list[JsonDict]:
        """Create batch of configurations."""
        return ConfigurationFactory.create_batch(count, **kwargs)

    @staticmethod
    def create_entity_batch(count: int = 5, **kwargs: Any) -> list[BaseTestEntity]:
        """Create batch of entities."""
        return TestEntityFactory.create_batch(count, **kwargs)

    @staticmethod
    def create_mixed_results(
        success_count: int = 3, failure_count: int = 2
    ) -> list[FlextResult[Any]]:
        """Create mixed success/failure results."""
        results: list[FlextResult[Any]] = [
            FlextResultFactory.create_success(
                data=f"success_data_{uuid.uuid4()}",
            )
            for _ in range(success_count)
        ]

        results.extend(
            FlextResultFactory.create_failure(
                error=f"Error {i + 1}",
                error_code=f"ERR_{i + 1:03d}",
            )
            for i in range(failure_count)
        )

        return results


# Convenience aliases for common patterns
create_user = UserDataFactory
create_config = ConfigurationFactory
create_db_config = DatabaseConfigFactory
create_service = ServiceDefinitionFactory
create_entity = TestEntityFactory
create_value_object = TestValueObjectFactory
create_batch = BatchFactory

# Result creators
success_result = FlextResultFactory.create_success
failure_result = FlextResultFactory.create_failure
validation_failure = FlextResultFactory.create_validation_failure

# Exception creators
domain_error = ExceptionFactory.create_domain_error
validation_error = ExceptionFactory.create_validation_error


__all__ = [
    # Test base classes
    "BaseTestEntity",
    "BaseTestValueObject",
    "BatchFactory",
    "ConfigurationFactory",
    "DatabaseConfigFactory",
    "ExceptionFactory",
    "FlextResultFactory",
    "ServiceDefinitionFactory",
    "TestEntityFactory",
    "TestValueObjectFactory",
    # Main factories
    "UserDataFactory",
    "create_batch",
    "create_config",
    "create_db_config",
    "create_entity",
    "create_service",
    # Convenience aliases
    "create_user",
    "create_value_object",
    "domain_error",
    "failure_result",
    "success_result",
    "validation_error",
    "validation_failure",
]
