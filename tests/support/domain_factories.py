"""Advanced domain factories using factory_boy for comprehensive test data generation.

Leverages factory_boy's powerful features including SubFactory, LazyAttribute,
LazyFunction, Sequence, Faker, and Trait for realistic test data creation.
"""

from __future__ import annotations

import random
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

import factory
from factory import Faker, LazyAttribute, LazyFunction, SubFactory, Trait
from factory.fuzzy import FuzzyChoice, FuzzyDecimal, FuzzyInteger

from flext_core import FlextContainer, FlextResult
from flext_core.fields import FlextFieldCore, FlextFields
from flext_core.typings import FlextTypes

JsonDict = FlextTypes.Core.JsonDict


class FlextResultFactory(factory.Factory):
    """Factory for creating FlextResult objects with realistic scenarios."""

    class Meta:
        model = dict  # We'll create the actual FlextResult in _create

    # Base attributes for result creation
    success = True
    data = Faker("pydict", nb_elements=3, value_types=["str", "int", "bool"])
    error = None
    error_code = None
    error_data = None

    class Params:
        # Traits for different result types
        successful = Trait(
            success=True,
            error=None,
            error_code=None,
        )

        failed = Trait(
            success=False,
            data=None,
            error=Faker("sentence"),
            error_code=FuzzyChoice(["VALIDATION_ERROR", "NOT_FOUND", "INTERNAL_ERROR"]),
        )

        with_validation_error = Trait(
            success=False,
            data=None,
            error="Validation failed",
            error_code="VALIDATION_ERROR",
            error_data={"field": Faker("word"), "value": Faker("word")},
        )

        with_not_found_error = Trait(
            success=False,
            data=None,
            error=LazyAttribute(lambda obj: f"Resource '{Faker('word')}' not found"),
            error_code="NOT_FOUND",
        )

    @classmethod
    def _create(cls, model_class: type, **kwargs: Any) -> FlextResult[Any]:
        """Create actual FlextResult instance."""
        if kwargs.get("success", True):
            return FlextResult[Any].ok(kwargs.get("data"))
        return FlextResult[Any].fail(
            kwargs.get("error", "Unknown error"),
            error_code=kwargs.get("error_code"),
            error_data=kwargs.get("error_data"),
        )

    @classmethod
    def create_success(cls, data: Any = None) -> FlextResult[Any]:
        """Create successful FlextResult."""
        return FlextResult[Any].ok(data)

    @classmethod
    def create_failure(cls, error: str, error_code: str | None = None) -> FlextResult[Any]:
        """Create failed FlextResult."""
        return FlextResult[Any].fail(error, error_code=error_code)

    @classmethod
    def create_validation_failure(cls, field: str, value: Any = None) -> FlextResult[Any]:
        """Create validation failure FlextResult."""
        return FlextResult[Any].fail(
            f"Validation failed for field '{field}'",
            error_code="VALIDATION_ERROR",
            error_data={"field": field, "value": value},
        )

    @classmethod
    def create(cls, **kwargs: Any) -> FlextResult[Any]:
        """Create FlextResult using factory_boy patterns."""
        return cls._create(dict, **kwargs)


class UserDataFactory(factory.Factory):
    """Factory for creating realistic user data using Faker."""

    class Meta:
        model = dict

    id = LazyFunction(lambda: str(uuid.uuid4()))
    name = Faker("name")
    email = Faker("email")
    age = FuzzyInteger(18, 80)
    is_active = Faker("boolean", chance_of_getting_true=80)
    created_at = Faker("date_time_this_year", tzinfo=None)
    updated_at = LazyAttribute(lambda obj: obj.created_at)

    # Profile information
    first_name = Faker("first_name")
    last_name = Faker("last_name")
    phone = Faker("phone_number")
    address = SubFactory("tests.support.domain_factories.AddressFactory")

    # Professional information
    job_title = Faker("job")
    company = Faker("company")
    department = FuzzyChoice(["engineering", "marketing", "sales", "hr", "finance"])
    salary = FuzzyDecimal(30000, 150000, precision=2)

    # Metadata
    metadata = factory.LazyFunction(lambda: {
        "preferences": {
            "theme": factory.Faker("random_element", elements=["light", "dark"]),
            "language": factory.Faker("language_code"),
            "timezone": factory.Faker("timezone"),
        },
        "permissions": factory.Faker("random_elements",
                                   elements=["read", "write", "REDACTED_LDAP_BIND_PASSWORD", "delete"],
                                   length=factory.Faker("random_int", min=1, max=4),
                                   unique=True),
        "last_login": factory.LazyFunction(lambda: datetime.now().isoformat()),
    })

    class Params:
        # Traits for different user types
        REDACTED_LDAP_BIND_PASSWORD = Trait(
            is_active=True,
            metadata=factory.LazyFunction(lambda: {
                "role": "REDACTED_LDAP_BIND_PASSWORD",
                "permissions": ["read", "write", "REDACTED_LDAP_BIND_PASSWORD", "delete"],
                "access_level": "full",
            }),
        )

        inactive = Trait(
            is_active=False,
            metadata=factory.LazyFunction(lambda: {
                "deactivated_at": factory.Faker("date_time_this_year").isoformat(),
                "reason": "voluntary",
            }),
        )

        premium = Trait(
            metadata=factory.LazyFunction(lambda: {
                "subscription": "premium",
                "billing_cycle": "monthly",
                "features": ["advanced_analytics", "priority_support"],
            }),
        )


class AddressFactory(factory.Factory):
    """Factory for creating realistic address data."""

    class Meta:
        model = dict

    street = Faker("street_address")
    city = Faker("city")
    state = Faker("state")
    postal_code = Faker("postcode")
    country = Faker("country")
    latitude = Faker("latitude")
    longitude = Faker("longitude")


class ConfigurationFactory(factory.Factory):
    """Factory for creating realistic configuration data."""

    class Meta:
        model = dict

    # Database configuration
    database_url = LazyAttribute(
        lambda obj: f"postgresql://{Faker('user_name')}:"
                   f"{Faker('password')}@"
                   f"{Faker('hostname')}:5432/"
                   f"{Faker('word')}_db"
    )

    # Server configuration
    host = Faker("ipv4")
    port = FuzzyInteger(8000, 9000)
    debug = Faker("boolean", chance_of_getting_true=30)

    # Logging configuration
    log_level = FuzzyChoice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    log_format = FuzzyChoice(["json", "text", "structured"])

    # Cache configuration
    cache_backend = FuzzyChoice(["redis", "memcached", "memory"])
    cache_url = LazyAttribute(
        lambda obj: f"redis://{Faker('hostname')}:6379/0"
        if obj.cache_backend == "redis"
        else f"memcached://{Faker('hostname')}:11211"
    )

    # Feature flags
    features = factory.LazyFunction(lambda: {
        "authentication": factory.Faker("boolean"),
        "rate_limiting": factory.Faker("boolean"),
        "metrics": factory.Faker("boolean"),
        "caching": factory.Faker("boolean"),
        "async_processing": factory.Faker("boolean"),
    })

    # Performance settings
    max_connections = FuzzyInteger(50, 500)
    timeout = FuzzyInteger(10, 60)
    retry_attempts = FuzzyInteger(1, 5)

    class Params:
        # Environment-specific traits
        development = Trait(
            debug=True,
            log_level="DEBUG",
            max_connections=10,
        )

        production = Trait(
            debug=False,
            log_level="INFO",
            max_connections=200,
            timeout=30,
        )

        testing = Trait(
            debug=True,
            log_level="DEBUG",
            database_url="sqlite:///:memory:",
            cache_backend="memory",
        )


class FieldDataFactory(factory.Factory):
    """Factory for creating realistic field configuration data."""

    class Meta:
        model = dict

    field_id = LazyFunction(lambda: str(uuid.uuid4()))
    field_name = LazyAttribute(lambda obj: f"field_{Faker('word')}")
    field_type = FuzzyChoice(["string", "integer", "boolean", "float"])
    required = Faker("boolean", chance_of_getting_true=70)
    description = Faker("sentence")

    # Validation rules
    tags = factory.LazyFunction(lambda:
        factory.Faker("random_elements",
                     elements=["validation", "required", "system", "user", "REDACTED_LDAP_BIND_PASSWORD"],
                     length=factory.Faker("random_int", min=0, max=3),
                     unique=True)
    )

    class Params:
        # Field type specific traits
        string_field = Trait(
            field_type="string",
            min_length=FuzzyInteger(1, 10),
            max_length=FuzzyInteger(50, 200),
            pattern=FuzzyChoice([r"^[a-zA-Z0-9_]+$", r"^\w+@\w+\.\w+$", r"^\d{3}-\d{2}-\d{4}$"]),
            allowed_values=factory.LazyFunction(lambda:
                factory.Faker("random_elements",
                             elements=["option1", "option2", "option3", "option4"],
                             length=factory.Faker("random_int", min=0, max=4),
                             unique=True)
            ),
        )

        integer_field = Trait(
            field_type="integer",
            min_value=FuzzyInteger(-100, 0),
            max_value=FuzzyInteger(100, 1000),
            default_value=FuzzyInteger(0, 50),
        )

        boolean_field = Trait(
            field_type="boolean",
            default_value=Faker("boolean"),
        )

        email_field = Trait(
            field_type="string",
            pattern=r"^\w+@\w+\.\w+$",
            description="Valid email address",
            tags=["email", "contact", "required"],
        )


class FlextFieldFactory(factory.Factory):
    """Factory for creating actual FlextFieldCore objects."""

    class Meta:
        model = dict  # We'll create the actual field in _create

    field_id = LazyFunction(lambda: str(uuid.uuid4()))
    field_name = LazyAttribute(lambda obj: f"test_field_{obj.field_id[:8]}")
    field_type = "string"
    required = True
    description = Faker("sentence")

    @classmethod
    def _create(cls, model_class: type, **kwargs: Any) -> FlextFieldCore:
        """Create actual FlextFieldCore instance."""
        field_type = kwargs.pop("field_type", "string")
        field_id = kwargs.pop("field_id")
        field_name = kwargs.pop("field_name")

        if field_type == "string":
            return FlextFields.create_string_field(field_id, field_name, **kwargs)
        if field_type == "integer":
            return FlextFields.create_integer_field(field_id, field_name, **kwargs)
        if field_type == "boolean":
            return FlextFields.create_boolean_field(field_id, field_name, **kwargs)
        msg = f"Unsupported field type: {field_type}"
        raise ValueError(msg)

    class Params:
        string_constraints = Trait(
            field_type="string",
            min_length=FuzzyInteger(1, 5),
            max_length=FuzzyInteger(10, 100),
            pattern=r"^[a-zA-Z0-9_]+$",
        )

        integer_constraints = Trait(
            field_type="integer",
            min_value=FuzzyInteger(0, 10),
            max_value=FuzzyInteger(100, 1000),
        )


class ServiceDataFactory(factory.Factory):
    """Factory for creating realistic service data."""

    class Meta:
        model = dict

    name = LazyAttribute(lambda obj: f"{Faker('word')}_service")
    version = factory.LazyFunction(lambda: f"1.{random.randint(0, 10)}.{random.randint(0, 50)}")
    description = Faker("sentence")

    # Service configuration
    config = factory.LazyFunction(lambda: {
        "host": factory.Faker("hostname"),
        "port": factory.Faker("port_number"),
        "protocol": factory.Faker("random_element", elements=["http", "https", "tcp"]),
        "timeout": factory.Faker("random_int", min=5, max=60),
    })

    # Dependencies
    dependencies = factory.LazyFunction(lambda:
        factory.Faker("random_elements",
                     elements=["database", "cache", "queue", "auth", "logging", "metrics"],
                     length=factory.Faker("random_int", min=1, max=4),
                     unique=True)
    )

    # Health status
    healthy = Faker("boolean", chance_of_getting_true=90)
    status = LazyAttribute(lambda obj: "running" if obj.healthy else "error")

    # Metrics
    metrics = factory.LazyFunction(lambda: {
        "requests_per_second": factory.Faker("random_int", min=10, max=1000),
        "error_rate": factory.Faker("pyfloat", min_value=0, max_value=0.1, right_digits=3),
        "response_time_ms": factory.Faker("random_int", min=50, max=500),
        "uptime_seconds": factory.Faker("random_int", min=3600, max=86400),
    })

    class Params:
        database_service = Trait(
            name="database_service",
            config=factory.LazyFunction(lambda: {
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "pool_size": 10,
                "ssl_mode": "prefer",
            }),
            dependencies=[],
        )

        cache_service = Trait(
            name="cache_service",
            config=factory.LazyFunction(lambda: {
                "host": "localhost",
                "port": 6379,
                "db": 0,
                "ttl": 3600,
            }),
            dependencies=["database"],
        )

        api_service = Trait(
            name="api_service",
            config=factory.LazyFunction(lambda: {
                "host": "0.0.0.0",
                "port": 8080,
                "workers": 4,
                "max_requests": 1000,
            }),
            dependencies=["database", "cache", "auth"],
        )


class PayloadDataFactory(factory.Factory):
    """Factory for creating realistic message/payload data."""

    class Meta:
        model = dict

    message_id = LazyFunction(lambda: str(uuid.uuid4()))
    type = FuzzyChoice(["user_created", "order_placed", "payment_processed", "notification_sent"])
    source = Faker("word")
    timestamp = Faker("date_time_this_month", tzinfo=None)

    # Payload data
    data = factory.LazyFunction(lambda: {
        "entity_id": str(uuid.uuid4()),
        "action": factory.Faker("random_element",
                               elements=["create", "update", "delete", "activate"]),
        "metadata": {
            "user_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
            "ip_address": factory.Faker("ipv4"),
        },
    })

    # Headers for message routing
    headers = factory.LazyFunction(lambda: {
        "content-type": "application/json",
        "correlation-id": str(uuid.uuid4()),
        "retry-count": "0",
        "priority": factory.Faker("random_element", elements=["low", "normal", "high"]),
    })

    class Params:
        user_event = Trait(
            type="user_created",
            data=factory.LazyFunction(lambda: {
                "user_id": str(uuid.uuid4()),
                "email": factory.Faker("email"),
                "name": factory.Faker("name"),
                "registration_source": "web",
            }),
        )

        order_event = Trait(
            type="order_placed",
            data=factory.LazyFunction(lambda: {
                "order_id": str(uuid.uuid4()),
                "customer_id": str(uuid.uuid4()),
                "total_amount": factory.Faker("pydecimal", left_digits=3, right_digits=2),
                "currency": "USD",
                "items": [
                    {
                        "product_id": str(uuid.uuid4()),
                        "quantity": factory.Faker("random_int", min=1, max=5),
                        "price": factory.Faker("pydecimal", left_digits=2, right_digits=2),
                    }
                    for _ in range(factory.Faker("random_int", min=1, max=3))
                ],
            }),
        )


class ContainerFactory(factory.Factory):
    """Factory for creating populated FlextContainer instances."""

    class Meta:
        model = dict  # We'll create actual container in _create

    @classmethod
    def _create(cls, model_class: type, **kwargs: Any) -> FlextContainer:
        """Create actual FlextContainer with services."""
        container = FlextContainer()

        # Register some default services
        services = kwargs.get("services", {})
        for name, service in services.items():
            container.register(name, service)

        # Register some default factories
        factories = kwargs.get("factories", {})
        for name, factory_func in factories.items():
            container.register_factory(name, factory_func)

        return container

    class Params:
        # Pre-configured container types
        with_basic_services = Trait(
            services={
                "database": ServiceDataFactory.build(database_service=True),
                "cache": ServiceDataFactory.build(cache_service=True),
            },
        )

        with_all_services = Trait(
            services={
                "database": ServiceDataFactory.build(database_service=True),
                "cache": ServiceDataFactory.build(cache_service=True),
                "api": ServiceDataFactory.build(api_service=True),
            },
            factories={
                "user_factory": UserDataFactory.build,
                "config_factory": lambda: ConfigurationFactory.build(testing=True),
            },
        )


# =============================================================================
# BATCH FACTORIES - For creating multiple related objects
# =============================================================================

class BatchFactory:
    """Factory for creating batches of related test objects."""

    @staticmethod
    def create_user_batch(
        count: int = 10,
        **traits: Any,
    ) -> list[dict[str, Any]]:
        """Create a batch of users with optional traits."""
        return UserDataFactory.build_batch(count, **traits)

    @staticmethod
    def create_field_batch(
        count: int = 5,
        field_types: Sequence[str] | None = None,
    ) -> list[FlextFieldCore]:
        """Create a batch of fields with different types."""
        if field_types is None:
            field_types = ["string", "integer", "boolean"]

        fields = []
        for i in range(count):
            field_type = field_types[i % len(field_types)]
            field = FlextFieldFactory.build(field_type=field_type)
            fields.append(field)

        return fields

    @staticmethod
    def create_service_ecosystem() -> dict[str, Any]:
        """Create a complete service ecosystem for testing."""
        return {
            "database": ServiceDataFactory.build(database_service=True),
            "cache": ServiceDataFactory.build(cache_service=True),
            "api": ServiceDataFactory.build(api_service=True),
            "users": UserDataFactory.build_batch(5),
            "config": ConfigurationFactory.build(production=True),
        }


# Export all factories
__all__ = [
    "AddressFactory",
    "BatchFactory",
    "ConfigurationFactory",
    "ContainerFactory",
    "FieldDataFactory",
    "FlextFieldFactory",
    "FlextResultFactory",
    "PayloadDataFactory",
    "ServiceDataFactory",
    "UserDataFactory",
]
