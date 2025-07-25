"""Enhanced Usage Examples for FLEXT Core Improvements.

This file demonstrates how the improved APIs dramatically reduce boilerplate
code and make common patterns much more intuitive and powerful.
"""

from __future__ import annotations

import contextlib
from datetime import UTC
from datetime import datetime
from typing import Any

from flext_core import FlextConfigBuilder
from flext_core import FlextContainer
from flext_core import FlextEntity
from flext_core import FlextResult
from flext_core import FlextServiceBuilder
from flext_core import create_factory
from flext_core import create_singleton_factory
from flext_core import validate

# Demonstrate the enhanced APIs
from flext_core import validate_choice
from flext_core import validate_email
from flext_core import validate_number
from flext_core import validate_string

# Example 1: Powerful Result Chaining (BEFORE vs AFTER)


def example_old_result_pattern() -> None:
    """Old way: verbose error handling."""

    def validate_user_old(data: dict[str, Any]) -> FlextResult[dict[str, Any]]:
        # Check email
        email = data.get("email")
        if not email:
            return FlextResult.fail("Email required")
        if "@" not in email:
            return FlextResult.fail("Invalid email")

        # Check age
        age = data.get("age")
        if age is None:
            return FlextResult.fail("Age required")
        if age < 18:
            return FlextResult.fail("Must be 18+")

        return FlextResult.ok({"email": email, "age": age})

    # Usage requires lots of if/else checking
    result = validate_user_old({"email": "test@example.com", "age": 25})
    if result.is_success:
        pass
    else:
        pass


def example_new_result_pattern() -> None:
    """New way: fluent functional chaining."""

    def validate_user_new(data: dict[str, Any]) -> FlextResult[dict[str, Any]]:
        return (
            FlextResult.ok(data)
            .map(lambda d: d.get("email", ""))
            .then(lambda email: validate_email(email))
            .zip_with(
                FlextResult.ok(data.get("age", 0)),
                lambda email, age: {"email": email, "age": age},
            )
            .filter(lambda user: user["age"] >= 18)
            .recover_with(
                lambda _: FlextResult.ok({"email": "default@example.com", "age": 18}),
            )
        )

    # Usage is clean and handles all error cases
    (
        validate_user_new({"email": "test@example.com", "age": 25})
        .tap(lambda user: print(f"Processing user: {user['email']}"))
        .map(lambda user: f"Welcome {user['email']}!")
    )


# Example 2: Configuration Building (BEFORE vs AFTER)


def example_old_config_building() -> None:
    """Old way: manual dictionary building with validation."""

    def build_database_config_old(env: str) -> dict[str, Any]:
        config = {}

        # Required settings
        if env == "production":
            config["host"] = "prod-db.company.com"
            config["port"] = 5432
        elif env == "staging":
            config["host"] = "staging-db.company.com"
            config["port"] = 5432
        else:
            config["host"] = "localhost"
            config["port"] = 5433

        # Optional settings with defaults
        config["pool_size"] = 10
        config["timeout"] = 30
        config["ssl"] = env == "production"

        # Validation
        required_keys = ["host", "port"]
        for key in required_keys:
            if key not in config:
                msg = f"Missing required config: {key}"
                raise ValueError(msg)

        return config

    with contextlib.suppress(ValueError):
        build_database_config_old("production")


def example_new_config_building() -> None:
    """New way: fluent builder with validation."""

    def build_database_config_new(env: str) -> FlextResult[dict[str, Any]]:
        builder = FlextConfigBuilder()

        # Environment-specific settings
        if env == "production":
            builder.set_required("host", "prod-db.company.com")
        elif env == "staging":
            builder.set_required("host", "staging-db.company.com")
        else:
            builder.set_required("host", "localhost")

        return (
            builder.set_required("port", 5432 if env != "development" else 5433)
            .set_default("pool_size", 10)
            .set_default("timeout", 30)
            .set("ssl", env == "production")
            .set_if_not_none("password", None)  # Only set if provided
            .merge({"created_at": datetime.now(UTC).isoformat()})
            .build()
        )

    # Usage with automatic validation
    result = build_database_config_new("production")
    if result.is_success:
        pass
    else:
        pass


# Example 3: Service Registration (BEFORE vs AFTER)


class DatabaseService:
    """Example database service."""

    def __init__(self, host: str, port: int) -> None:
        """Initialize database service."""
        self.host = host
        self.port = port

    def connect(self) -> str:
        """Connect to database."""
        return f"Connected to {self.host}:{self.port}"


class CacheService:
    """Example cache service."""

    def __init__(self, ttl: int = 3600) -> None:
        """Initialize cache service."""
        self.ttl = ttl

    def get(self, key: str) -> str | None:
        """Get cached value."""
        return f"cached_{key}"


def example_old_service_registration() -> None:
    """Old way: manual service registration.

    Register services in container.
    """
    container = FlextContainer()

    # Manual registration with error checking
    db_service = DatabaseService("localhost", 5432)
    result = container.register("database", db_service)
    if not result.is_success:
        return

    # Factory registration
    def cache_factory() -> CacheService:
        return CacheService(ttl=7200)

    result = container.register_singleton("cache", cache_factory)
    if not result.is_success:
        return

    # Manual service retrieval
    db_result = container.get("database")
    if db_result.is_success:
        pass
    else:
        pass


def example_new_service_registration() -> None:
    """New way: builder pattern with bulk operations."""

    def create_database() -> DatabaseService:
        return DatabaseService("localhost", 5432)

    def create_cache() -> CacheService:
        return CacheService(ttl=7200)

    # Build service configuration with validation
    config_result = (
        FlextServiceBuilder()
        .add_factory("database", create_database, singleton=True)
        .add_factory("cache", create_cache, singleton=True)
        .add_services(
            logger=print,  # Simple service
            metrics=lambda: "metrics_service",
        )
        .build()
    )

    if config_result.is_failure:
        return

    # Apply configuration to container
    container = FlextContainer()
    config = config_result.data

    # Register all services from config
    for name, service in config["services"].items():
        container.register(name, service)

    for name, factory in config["factories"].items():
        singleton = name in config["singletons"]
        if singleton:
            container.register_singleton(name, factory)
        else:
            container.register(name, factory)

    # Use enhanced container methods
    container.get_or_fail("database")

    # Check service health
    container.service_health_check()


# Example 4: Validation Chains (BEFORE vs AFTER)


def example_old_validation() -> None:
    """Old way: manual validation with lots of if/else."""

    def validate_product_old(data: dict[str, Any]) -> FlextResult[dict[str, Any]]:
        name = data.get("name", "")
        if not name:
            return FlextResult.fail("Product name required")
        if len(name) < 3:
            return FlextResult.fail("Product name too short")
        if len(name) > 50:
            return FlextResult.fail("Product name too long")

        price = data.get("price")
        if price is None:
            return FlextResult.fail("Price required")
        if not isinstance(price, (int, float)):
            return FlextResult.fail("Price must be numeric")
        if price <= 0:
            return FlextResult.fail("Price must be positive")
        if price > 10000:
            return FlextResult.fail("Price too high")

        category = data.get("category", "")
        valid_categories = ["electronics", "books", "clothing", "toys"]
        if category not in valid_categories:
            return FlextResult.fail(
                f"Invalid category. Must be one of: {valid_categories}",
            )

        return FlextResult.ok(
            {
                "name": name,
                "price": price,
                "category": category,
            },
        )

    validate_product_old(
        {
            "name": "Test Product",
            "price": 29.99,
            "category": "electronics",
        },
    )


def example_new_validation() -> None:
    """New way: fluent validation chains."""

    def validate_product_new(data: dict[str, Any]) -> FlextResult[dict[str, Any]]:
        # Validate each field with chaining
        name_result = validate_string(
            data.get("name", ""),
            min_length=3,
            max_length=50,
        )

        price_result = validate_number(
            data.get("price"),
            min_value=0.01,
            max_value=10000,
        )

        category_result = validate_choice(
            data.get("category", ""),
            ["electronics", "books", "clothing", "toys"],
        )

        # Combine all validations
        return FlextResult.combine(name_result, price_result, category_result).map(
            lambda values: {
                "name": values[0],
                "price": values[1],
                "category": values[2],
            },
        )

    # Alternative: single validation chain
    def validate_product_chain(data: dict[str, Any]) -> FlextResult[str]:
        return (
            validate(data.get("name", ""))
            .validate_with(
                lambda name: validate_string(name, min_length=3, max_length=50),
            )
            .map(lambda name: f"Product: {name}")
        )

    result = validate_product_new(
        {
            "name": "Test Product",
            "price": 29.99,
            "category": "electronics",
        },
    )

    if result.is_success:
        pass


# Example 5: Enhanced Domain Entities (BEFORE vs AFTER)


class User(FlextEntity):
    """Enhanced user entity with new functionality."""

    name: str
    email: str
    is_active: bool = False
    login_count: int = 0

    def validate_domain_rules(self) -> None:
        """Validate business rules."""
        if "@" not in self.email:
            msg = "Invalid email format"
            raise ValueError(msg)
        if self.is_active and not self.name.strip():
            msg = "Active users must have names"
            raise ValueError(msg)


def example_old_entity_usage() -> None:
    """Old way: manual entity management."""
    # Create user
    user = User(
        name="John Doe",
        email="john@example.com",
    )

    # Manual version management for updates
    User(
        id=user.id,
        name="John Smith",  # Changed name
        email=user.email,
        is_active=user.is_active,
        login_count=user.login_count,
        created_at=user.created_at,
        version=user.version + 1,  # Manual increment
    )


def example_new_entity_usage() -> None:
    """New way: enhanced entity methods."""
    # Create user
    user = User(
        name="John Doe",
        email="john@example.com",
    )

    # Easy entity updates with automatic versioning
    updated_user = user.copy_with(
        name="John Smith",
        is_active=True,
        login_count=user.login_count + 1,
    )

    # Version comparison

    # Age calculation

    # Different serialization options

    # Easy reconstruction
    data = updated_user.to_dict_with_metadata()
    User.from_dict(data)


# Example 6: Factory Patterns (BEFORE vs AFTER)


def example_old_factory_pattern() -> None:
    """Old way: manual factory implementation."""

    class DatabaseFactory:
        def __init__(self, config: dict[str, Any]) -> None:
            self.config = config
            self._instance = None
            self._created = False

        def create(self) -> DatabaseService:
            try:
                return DatabaseService(
                    self.config["host"],
                    self.config["port"],
                )
            except Exception as e:
                msg = f"Failed to create database: {e}"
                raise ValueError(msg)

        def get_singleton(self) -> DatabaseService:
            if not self._created:
                self._instance = self.create()
                self._created = True
            return self._instance

    config = {"host": "localhost", "port": 5432}
    factory = DatabaseFactory(config)

    with contextlib.suppress(ValueError):
        factory.get_singleton()


def example_new_factory_pattern() -> None:
    """New way: FlextFactory with error handling."""

    def create_database(host: str = "localhost", port: int = 5432) -> DatabaseService:
        return DatabaseService(host, port)

    # Create factory with built-in error handling
    factory = create_factory(create_database)

    # Create multiple instances with tracking
    result = factory.create_many(3, host="db-cluster", port=5432)

    if result.is_success:
        pass
    else:
        pass

    # Singleton factory for shared resources

    singleton_factory = create_singleton_factory(
        lambda: CacheService(ttl=3600),
    )

    # Always returns same instance
    singleton_factory.get_instance().unwrap()
    singleton_factory.get_instance().unwrap()


# Run all examples
def main() -> None:
    """Run all enhancement examples."""
    example_old_result_pattern()
    example_new_result_pattern()

    example_old_config_building()
    example_new_config_building()

    example_old_service_registration()
    example_new_service_registration()

    example_old_validation()
    example_new_validation()

    example_old_entity_usage()
    example_new_entity_usage()

    example_old_factory_pattern()
    example_new_factory_pattern()


if __name__ == "__main__":
    main()
