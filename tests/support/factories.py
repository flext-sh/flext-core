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
from typing import override

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

from flext_core import (
    FlextExceptions,
    FlextModels,
    FlextResult,
    FlextTypes,
)

JsonDict = FlextTypes.Core.JsonObject


class BaseTestEntity(FlextModels.Entity):
    """Base entity for testing with proper FlextCore integration."""

    # Using proper FlextCore types for compatibility
    name: str = "test_entity"
    description: str = ""
    active: bool = True
    # Note: version field inherited from Entity base class (int type)
    metadata: FlextModels.Metadata = FlextModels.Metadata(root={})

    @override
    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for test entity."""
        if not self.name.strip():
            return FlextResult[None].fail(
                "Entity name cannot be empty",
                error_code="INVALID_NAME",
            )
        return FlextResult[None].ok(None)


class BaseTestValueObject(FlextModels.Value):
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


class UserDataFactory(DictFactory[dict[str, object]]):
    """Factory for generating user test data with realistic attributes."""

    class Meta:
        """Factory meta compatibility."""

        model = dict

    id = LazyFunction(lambda: str(uuid.uuid4()))
    username = Sequence(lambda n: f"user_{n}")
    email: LazyAttribute[object, str] = LazyAttribute(
        lambda obj: f"{obj.username}@example.com"
    )
    first_name: Faker[str, str] = Faker("first_name")
    last_name: Faker[str, str] = Faker("last_name")
    age = factory.fuzzy.FuzzyInteger(18, 80)
    is_active = True
    created_at: Faker[str, str] = Faker("date_time_this_year")

    # Traits for different user types
    class Params:
        """User factory parameters for different user types."""

        is_REDACTED_LDAP_BIND_PASSWORD: Trait[dict[str, object]] = Trait(
            is_active=True,
            permissions=["read", "write", "REDACTED_LDAP_BIND_PASSWORD"],
            role="REDACTED_LDAP_BIND_PASSWORDistrator",
        )
        is_guest: Trait[dict[str, object]] = Trait(
            is_active=True,
            permissions=["read"],
            role="guest",
        )
        is_inactive: Trait[dict[str, object]] = Trait(
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

    preferences: LazyFunction[dict[str, object]] = LazyFunction(
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


class ConfigurationFactory(DictFactory[dict[str, object]]):
    """Factory for generating configuration test data."""

    class Meta:
        """Factory meta compatibility."""

        model = dict

    name = Sequence(lambda n: f"config_{n}")
    version = "1.0.0"
    debug = False

    # Database configuration with sub-factory
    database: SubFactory[dict[str, object], dict[str, object]] = SubFactory(
        "tests.support.factories.DatabaseConfigFactory"
    )

    # Logging configuration
    logging: LazyFunction[dict[str, object]] = LazyFunction(
        lambda: {
            "level": factory.fuzzy.FuzzyChoice([
                "DEBUG",
                "INFO",
                "WARNING",
                "ERROR",
            ]).fuzz(),
            "format": "json",
            "handlers": ["console", "file"],
            "file_path": "/var/log/app.log",
        }
    )

    # Feature flags
    features: LazyFunction[dict[str, object]] = LazyFunction(
        lambda: {
            "cache_enabled": factory.fuzzy.FuzzyChoice([True, False]).fuzz(),
            "metrics_enabled": True,
            "auth_required": True,
            "rate_limiting": factory.fuzzy.FuzzyChoice([True, False]).fuzz(),
        }
    )

    class Params:
        """Configuration factory parameters for different environments."""

        production: Trait[dict[str, object]] = Trait(
            debug=False,
            logging__level="INFO",
            features__cache_enabled=True,
        )
        development: Trait[dict[str, object]] = Trait(
            debug=True,
            logging__level="DEBUG",
            features__rate_limiting=False,
        )


class DatabaseConfigFactory(DictFactory[dict[str, object]]):
    """Factory for database configuration data."""

    class Meta:
        """Factory meta compatibility."""

        model = dict

    host = "localhost"
    port = factory.fuzzy.FuzzyInteger(3000, 9999)
    database = Sequence(lambda n: f"testdb_{n}")
    username = "testuser"
    password = "testpass"
    pool_size = factory.fuzzy.FuzzyInteger(5, 20)
    timeout = 30
    ssl_enabled = False

    # Build URL from components
    url: LazyAttribute[object, str] = LazyAttribute(
        lambda obj: f"postgresql://{obj.username}:{obj.password}@{obj.host}:{obj.port}/{obj.database}",
    )


class ServiceDefinitionFactory(DictFactory[dict[str, object]]):
    """Factory for service definition data."""

    class Meta:
        """Factory meta compatibility."""

        model = dict

    name = Sequence(lambda n: f"service_{n}")
    version: factory.fuzzy.FuzzyChoice[str, str] = factory.fuzzy.FuzzyChoice([
        "1.0.0",
        "1.1.0",
        "2.0.0",
    ])
    description: Faker[str, str] = Faker("sentence", nb_words=8)

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
    def create_success(data: object = "test_data") -> FlextResult[object]:
        """Create successful FlextResult."""
        return FlextResult[object].ok(data)

    @staticmethod
    def create_failure(
        error: str = "Test error",
        error_code: str = "TEST_ERROR",
    ) -> FlextResult[object]:
        """Create failed FlextResult."""
        return FlextResult[object].fail(error, error_code=error_code)

    @staticmethod
    def create_validation_failure(field: str = "test_field") -> FlextResult[object]:
        """Create validation failure FlextResult."""
        return FlextResult[object].fail(
            f"Validation failed for field: {field}",
            error_code="VALIDATION_ERROR",
        )


class ExceptionFactory:
    """Factory for creating various exception types."""

    @staticmethod
    def create_domain_error(
        message: str = "Domain error occurred",
        error_code: str = "DOMAIN_ERROR",
    ) -> Exception:
        """Create FlextExceptions."""
        return FlextExceptions.Error(message, error_code=error_code)

    @staticmethod
    def create_validation_error(
        message: str = "Validation failed",
        error_code: str = "VALIDATION_ERROR",
    ) -> Exception:
        """Create FlextExceptions."""
        return FlextExceptions.Error(message, error_code=error_code)


class TestEntityFactory(Factory[BaseTestEntity]):
    """Factory for creating FlextEntity test objects."""

    class Meta:
        """Factory meta compatibility."""

        model = BaseTestEntity

    name = Sequence(lambda n: f"test_entity_{n}")
    description: Faker[str, str] = Faker("sentence", nb_words=6)
    active = True
    version = LazyFunction(
        lambda: FlextModels.Version(root=factory.fuzzy.FuzzyInteger(1, 10).fuzz())
    )

    metadata = LazyFunction(
        lambda: FlextModels.Metadata(
            root={
                "created_by": "test_user",
                "tags": "test,automated",  # Convert list to string for Metadata compatibility
                "priority": factory.fuzzy.FuzzyChoice(["low", "medium", "high"]).fuzz(),
            }
        )
    )

    class Params:
        """Test entity factory parameters for different entity states."""

        inactive: Trait[BaseTestEntity] = Trait(
            active=False,
            metadata=LazyFunction(
                lambda: FlextModels.Metadata(root={"status": "deactivated"})
            ),
        )
        high_priority: Trait[BaseTestEntity] = Trait(
            metadata=LazyFunction(
                lambda: FlextModels.Metadata(
                    root={
                        "priority": "high",
                        "urgent": "true",  # Convert bool to string for Metadata compatibility
                    }
                )
            ),
        )


class TestValueObjectFactory(Factory[BaseTestValueObject]):
    """Factory for creating FlextValue test objects."""

    class Meta:
        """Factory meta compatibility."""

        model = BaseTestValueObject

    value = Sequence(lambda n: f"test_value_{n}")

    class Params:
        """Test value object parameters."""

        empty: Trait[BaseTestValueObject] = Trait(value="")
        long: Trait[BaseTestValueObject] = Trait(value=LazyFunction(lambda: "x" * 1000))
        unicode: Trait[BaseTestValueObject] = Trait(value="🚀 Test Value 测试 مرحبا")


# Batch creation utilities
class BatchFactory:
    """Utility for creating batches of test objects."""

    @staticmethod
    def create_user_batch(count: int = 5, **kwargs: object) -> list[dict[str, object]]:
        """Create batch of users."""
        return UserDataFactory.create_batch(count, **kwargs)

    @staticmethod
    def create_config_batch(
        count: int = 3, **kwargs: object
    ) -> list[dict[str, object]]:
        """Create batch of configurations."""
        return ConfigurationFactory.create_batch(count, **kwargs)

    @staticmethod
    def create_entity_batch(count: int = 5, **kwargs: object) -> list[BaseTestEntity]:
        """Create batch of entities."""
        return TestEntityFactory.create_batch(count, **kwargs)

    @staticmethod
    def create_mixed_results(
        success_count: int = 3, failure_count: int = 2
    ) -> list[FlextResult[object]]:
        """Create mixed success/failure results."""
        results: list[FlextResult[object]] = [
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
