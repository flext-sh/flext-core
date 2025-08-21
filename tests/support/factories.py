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
from typing import Any

import factory
import factory.fuzzy
from factory import Faker, Sequence, Trait

from flext_core import FlextResult
from flext_core.exceptions import FlextError, FlextValidationError
from flext_core.models import FlextEntity, FlextValue
from flext_core.typings import FlextTypes

JsonDict = FlextTypes.Core.JsonDict


class BaseTestEntity(FlextEntity):
    """Base entity for testing with proper FlextCore integration."""

    # Using proper field definitions following flext-core patterns
    name: str = "test_entity"
    description: str = ""
    active: bool = True
    version: int = 1
    metadata: dict[str, Any] = factory.LazyFunction(dict)

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate domain rules for test entity."""
        if not self.name.strip():
            return FlextResult[None].fail(
                "Entity name cannot be empty",
                error_code="INVALID_NAME"
            )
        return FlextResult[None].ok(None)


class BaseTestValueObject(FlextValue):
    """Base value object for testing with proper FlextCore integration."""

    value: str = "test_value"

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate domain rules for test value object."""
        if not self.value:
            return FlextResult[None].fail(
                "Value cannot be empty",
                error_code="INVALID_VALUE"
            )
        return FlextResult[None].ok(None)


class UserDataFactory(factory.DictFactory):
    """Factory for generating user test data with realistic attributes."""

    class Meta:
        model = dict

    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    username = Sequence(lambda n: f"user_{n}")
    email = factory.LazyAttribute(lambda obj: f"{obj['username']}@example.com")
    first_name = Faker("first_name")
    last_name = Faker("last_name")
    age = factory.fuzzy.FuzzyInteger(18, 80)
    is_active = True
    created_at = Faker("date_time_this_year")

    # Traits for different user types
    class Params:
        is_REDACTED_LDAP_BIND_PASSWORD = Trait(
            is_active=True,
            permissions=["read", "write", "REDACTED_LDAP_BIND_PASSWORD"],
            role="REDACTED_LDAP_BIND_PASSWORDistrator"
        )
        is_guest = Trait(
            is_active=True,
            permissions=["read"],
            role="guest"
        )
        is_inactive = Trait(
            is_active=False,
            deactivated_at=Faker("date_time_this_month")
        )

    # Complex nested data
    profile = factory.LazyFunction(lambda: {
        "bio": Faker("text", max_nb_chars=200).generate(),
        "location": Faker("city").generate(),
        "avatar_url": Faker("image_url").generate(),
    })

    preferences = factory.LazyFunction(lambda: {
        "theme": factory.fuzzy.FuzzyChoice(["light", "dark", "auto"]).fuzz(),
        "notifications": {
            "email": True,
            "push": False,
            "sms": False,
        },
        "language": "en_US",
        "timezone": Faker("timezone").generate(),
    })


class ConfigurationFactory(factory.DictFactory):
    """Factory for generating configuration test data."""

    class Meta:
        model = dict

    name = Sequence(lambda n: f"config_{n}")
    version = "1.0.0"
    debug = False

    # Database configuration with sub-factory
    database = factory.SubFactory("tests.support.factories.DatabaseConfigFactory")

    # Logging configuration
    logging = factory.LazyFunction(lambda: {
        "level": factory.fuzzy.FuzzyChoice(["DEBUG", "INFO", "WARNING", "ERROR"]).fuzz(),
        "format": "json",
        "handlers": ["console", "file"],
        "file_path": "/var/log/app.log",
    })

    # Feature flags
    features = factory.LazyFunction(lambda: {
        "cache_enabled": factory.fuzzy.FuzzyChoice([True, False]).fuzz(),
        "metrics_enabled": True,
        "auth_required": True,
        "rate_limiting": factory.fuzzy.FuzzyChoice([True, False]).fuzz(),
    })

    class Params:
        production = Trait(
            debug=False,
            logging__level="INFO",
            features__cache_enabled=True
        )
        development = Trait(
            debug=True,
            logging__level="DEBUG",
            features__rate_limiting=False
        )


class DatabaseConfigFactory(factory.DictFactory):
    """Factory for database configuration data."""

    class Meta:
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
    url = factory.LazyAttribute(
        lambda obj: f"postgresql://{obj['username']}:{obj['password']}@{obj['host']}:{obj['port']}/{obj['database']}"
    )


class ServiceDefinitionFactory(factory.DictFactory):
    """Factory for service definition data."""

    class Meta:
        model = dict

    name = Sequence(lambda n: f"service_{n}")
    version = factory.fuzzy.FuzzyChoice(["1.0.0", "1.1.0", "2.0.0"])
    description = Faker("sentence", nb_words=8)

    endpoints = factory.LazyFunction(lambda: [
        {"path": "/health", "method": "GET", "description": "Health check"},
        {"path": "/api/v1/users", "method": "GET", "description": "List users"},
        {"path": "/api/v1/users", "method": "POST", "description": "Create user"},
    ])

    dependencies = factory.LazyFunction(lambda: [
        {"name": "database", "version": ">=1.0.0", "required": True},
        {"name": "cache", "version": ">=2.0.0", "required": False},
    ])

    health_check = factory.LazyFunction(lambda: {
        "endpoint": "/health",
        "interval": 30,
        "timeout": 5,
        "retries": 3,
    })


class FlextResultFactory(factory.Factory):
    """Factory for FlextResult objects."""

    class Meta:
        model = FlextResult
        abstract = True

    @classmethod
    def create_success(cls, data: Any = "test_data") -> FlextResult[Any]:
        """Create successful FlextResult."""
        return FlextResult[Any].ok(data)

    @classmethod
    def create_failure(
        cls,
        error: str = "Test error",
        error_code: str = "TEST_ERROR"
    ) -> FlextResult[Any]:
        """Create failed FlextResult."""
        return FlextResult[Any].fail(error, error_code=error_code)

    @classmethod
    def create_validation_failure(cls, field: str = "test_field") -> FlextResult[Any]:
        """Create validation failure FlextResult."""
        return FlextResult[Any].fail(
            f"Validation failed for field: {field}",
            error_code="VALIDATION_ERROR"
        )


class ExceptionFactory(factory.Factory):
    """Factory for creating various exception types."""

    class Meta:
        model = FlextError
        abstract = True

    @classmethod
    def create_domain_error(
        cls,
        message: str = "Domain error occurred",
        error_code: str = "DOMAIN_ERROR"
    ) -> FlextError:
        """Create FlextError."""
        return FlextError(message, error_code=error_code)

    @classmethod
    def create_validation_error(
        cls,
        message: str = "Validation failed",
        error_code: str = "VALIDATION_ERROR"
    ) -> FlextValidationError:
        """Create FlextValidationError."""
        return FlextValidationError(message, error_code=error_code)


class TestEntityFactory(factory.Factory):
    """Factory for creating FlextEntity test objects."""

    class Meta:
        model = BaseTestEntity

    name = Sequence(lambda n: f"test_entity_{n}")
    description = Faker("sentence", nb_words=6)
    active = True
    version = factory.fuzzy.FuzzyInteger(1, 10)

    metadata = factory.LazyFunction(lambda: {
        "created_by": "test_user",
        "tags": ["test", "automated"],
        "priority": factory.fuzzy.FuzzyChoice(["low", "medium", "high"]).fuzz(),
    })

    class Params:
        inactive = Trait(
            active=False,
            metadata__status="deactivated"
        )
        high_priority = Trait(
            metadata__priority="high",
            metadata__urgent=True
        )


class TestValueObjectFactory(factory.Factory):
    """Factory for creating FlextValue test objects."""

    class Meta:
        model = BaseTestValueObject

    value = Sequence(lambda n: f"test_value_{n}")

    class Params:
        empty = Trait(value="")
        long = Trait(value=factory.LazyFunction(lambda: "x" * 1000))
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
    def create_mixed_results(success_count: int = 3, failure_count: int = 2) -> list[FlextResult[Any]]:
        """Create mixed success/failure results."""
        results = [FlextResultFactory.create_success(
                data=Faker("sentence").generate()
            ) for _ in range(success_count)]

        results.extend(FlextResultFactory.create_failure(
                error=f"Error {i + 1}",
                error_code=f"ERR_{i + 1:03d}"
            ) for i in range(failure_count))

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
