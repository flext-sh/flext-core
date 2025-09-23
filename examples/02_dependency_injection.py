#!/usr/bin/env python3
"""02 - FlextContainer Fundamentals: Complete Dependency Injection.

This example demonstrates the COMPLETE FlextContainer[T] API - the foundation
for dependency injection across the entire FLEXT ecosystem. FlextContainer provides
type-safe service registration, resolution, and lifecycle management.

Key Concepts Demonstrated:
- Service registration: register(), register_factory(), batch_register()
- Service resolution: get(), get_typed(), get_or_create()
- Auto-wiring: auto_wire() with dependency resolution
- Container management: clear(), has(), list_services()
- Global singleton: get_global(), register_global()
- Configuration: configure(), configure_container()
- Service lifecycles: singleton vs factory patterns

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import warnings
from typing import Protocol

from flext_core import (
    FlextConstants,
    FlextContainer,
    FlextLogger,
    FlextModels,
    FlextResult,
    FlextService,
)

# ========== SERVICE INTERFACES (PROTOCOLS) ==========


class DatabaseServiceProtocol(Protocol):
    """Protocol defining database service interface."""

    def connect(self) -> FlextResult[None]:
        """Connect to database."""
        ...

    def query(self, sql: str) -> FlextResult[list[dict[str, object]]]:
        """Execute query."""
        ...


class CacheServiceProtocol(Protocol):
    """Protocol defining cache service interface."""

    def get(self, key: str) -> FlextResult[object]:
        """Get value from cache."""
        ...

    def set(self, key: str, value: object) -> FlextResult[None]:
        """Set value in cache."""
        ...


class EmailServiceProtocol(Protocol):
    """Protocol defining email service interface."""

    def send(self, to: str, subject: str, body: str) -> FlextResult[None]:
        """Send email."""
        ...


# ========== SERVICE IMPLEMENTATIONS ==========


class DatabaseService:
    """Concrete database service implementation."""

    def __init__(self, connection_string: str = "sqlite:///:memory:") -> None:
        """Initialize with connection string."""
        self._connection_string = connection_string
        self._connected = False
        self._logger = FlextLogger(__name__)

    def connect(self) -> FlextResult[None]:
        """Connect to database."""
        if self._connected:
            return FlextResult[None].fail("Already connected")
        self._connected = True
        self._logger.info(f"Connected to {self._connection_string}")
        return FlextResult[None].ok(None)

    def query(self, sql: str) -> FlextResult[list[dict[str, object]]]:
        """Execute query."""
        if not self._connected:
            return FlextResult[list[dict[str, object]]].fail("Not connected")
        # Simulate query execution using the sql parameter
        self._logger.debug(f"Executing query: {sql[:50]}...")  # Log first 50 chars
        # Simulate different responses based on query
        if "users" in sql.lower():
            return FlextResult[list[dict[str, object]]].ok([{"id": 1, "name": "Test"}])
        return FlextResult[list[dict[str, object]]].ok([])


class CacheService:
    """Concrete cache service implementation."""

    def __init__(self) -> None:
        """Initialize cache."""
        self._cache: dict[str, object] = {}
        self._logger = FlextLogger(__name__)

    def get(self, key: str) -> FlextResult[object]:
        """Get value from cache."""
        if key in self._cache:
            self._logger.debug(f"Cache hit: {key}")
            return FlextResult[object].ok(self._cache[key])
        self._logger.debug(f"Cache miss: {key}")
        return FlextResult[object].fail(f"Key not found: {key}")

    def set(self, key: str, value: object) -> FlextResult[None]:
        """Set value in cache."""
        self._cache[key] = value
        self._logger.debug(f"Cache set: {key}")
        return FlextResult[None].ok(None)


class EmailService:
    """Concrete email service implementation."""

    def __init__(self, smtp_host: str = FlextConstants.Platform.DEFAULT_HOST) -> None:
        """Initialize with SMTP host."""
        self._smtp_host = smtp_host
        self._logger = FlextLogger(__name__)

    def send(self, to: str, subject: str, body: str) -> FlextResult[None]:
        """Send email (simulated)."""
        self._logger.info(f"Email sent to {to}: {subject}")
        self._logger.debug(
            f"Email body: {body[:100]}..."
        )  # Log first 100 chars of body
        # Simulate email sending with the body content
        return FlextResult[None].ok(None)


# ========== DOMAIN MODELS ==========


class User(FlextModels.Entity):
    """User entity with domain logic."""

    name: str
    email: str
    age: int
    is_active: bool = True


class UserRepository(FlextService[User]):
    """Repository pattern for User entities."""

    def __init__(self, database: DatabaseServiceProtocol, **data: object) -> None:
        """Initialize with database dependency."""
        super().__init__(**data)
        self._database = database
        self._logger = FlextLogger(__name__)

    def execute(self) -> FlextResult[User]:
        """Execute the main domain operation - find default user."""
        return self.find_by_id("default_user")

    def find_by_id(self, user_id: str) -> FlextResult[User]:
        """Find user by ID."""
        # Use parameterized query (simulated) to avoid SQL injection
        query = "SELECT * FROM users WHERE id = :user_id"  # nosec B608
        result = self._database.query(query)
        if result.is_failure:
            return FlextResult[User].fail(f"Database error: {result.error}")

        data = result.unwrap()
        if not data:
            return FlextResult[User].fail(f"User not found: {user_id}")

        # Simulate user creation from data
        user = User(
            id="user_1",
            name="John Doe",
            email="john@example.com",
            age=30,
            domain_events=[],
        )
        return FlextResult[User].ok(user)

    def save(self, user: User) -> FlextResult[None]:
        """Save user to database."""
        self._logger.info(f"Saving user: {user.id}")
        # Simulate save operation
        return FlextResult[None].ok(None)


# ========== COMPREHENSIVE CONTAINER SERVICE ==========


class ComprehensiveDIService(FlextService[User]):
    """Service demonstrating ALL FlextContainer patterns and methods."""

    def __init__(self, **data: object) -> None:
        """Initialize with FlextContainer."""
        super().__init__(**data)
        self._container = FlextContainer.get_global()
        self._logger = FlextLogger(__name__)

    def execute(self) -> FlextResult[User]:
        """Execute the service - required by FlextService."""
        return FlextResult[User].ok(
            User(
                id="user_1",
                name="Demo User",
                email="demo@example.com",
                age=25,
                domain_events=[],
            )
        )

    # ========== BASIC REGISTRATION ==========

    def demonstrate_basic_registration(self) -> None:
        """Show basic service registration patterns."""
        print("\n=== Basic Service Registration ===")

        # Clear container for clean demonstration
        self._container.clear()
        print("✅ Container cleared")

        # Register singleton service
        db_service = DatabaseService(
            f"postgresql://{FlextConstants.Platform.DEFAULT_HOST}/mydb"
        )
        result = self._container.register("database", db_service)
        print(f"Register singleton: {result.is_success}")

        # Register factory (creates new instance each time)
        result = self._container.register_factory("cache", CacheService)
        print(f"Register factory: {result.is_success}")

        # Check if services are registered
        has_db = self._container.has("database")
        has_cache = self._container.has("cache")
        print(f"Has database: {has_db}, Has cache: {has_cache}")

        # Get service count
        count = self._container.get_service_count()
        print(f"Service count: {count}")

    # ========== SERVICE RESOLUTION ==========

    def demonstrate_service_resolution(self) -> None:
        """Show all ways to resolve services."""
        print("\n=== Service Resolution ===")

        # Basic get (returns FlextResult)
        db_result = self._container.get("database")
        if db_result.is_success:
            db = db_result.unwrap()
            print(f"✅ Got database: {type(db).__name__}")
        else:
            print(f"❌ Failed to get database: {db_result.error}")

        # Type-safe get with validation
        typed_result = self._container.get_typed("database", DatabaseService)
        if typed_result.is_success:
            print(f"✅ Got typed database: {type(typed_result.unwrap()).__name__}")
        else:
            print(f"❌ Type validation failed: {typed_result.error}")

        # Get or create with factory
        email_result = self._container.get_or_create(
            "email", lambda: EmailService("smtp.gmail.com")
        )
        print(f"Get or create email: {email_result.is_success}")

    # ========== BATCH OPERATIONS ==========

    def demonstrate_batch_operations(self) -> None:
        """Show batch registration patterns."""
        print("\n=== Batch Operations ===")

        services: dict[str, object] = {
            "logger": FlextLogger("batch_example"),
            "config": {"debug": True, "timeout": 30},
            "metrics": {"requests": 0, "errors": 0},
        }

        result = self._container.batch_register(services)
        if result.is_success:
            # batch_register may return None for success
            print(f"✅ Batch registered {len(services)} services")
        else:
            print(f"❌ Batch registration failed: {result.error}")

        # List all services
        services_result = self._container.list_services()
        if services_result.is_success:
            services_list = services_result.unwrap()
            print(f"All services: {services_list}")
        else:
            print(f"❌ Failed to list services: {services_result.error}")

    # ========== AUTO-WIRING ==========

    def demonstrate_auto_wiring(self) -> None:
        """Show dependency auto-wiring."""
        print("\n=== Auto-Wiring ===")

        # Register dependencies first
        self._container.register("database", DatabaseService())

        # Auto-wire UserRepository (resolves database dependency)
        result = self._container.auto_wire(UserRepository)
        if result.is_success:
            repo = result.unwrap()
            print(f"✅ Auto-wired: {type(repo).__name__}")

            # Test the auto-wired service
            db_result = self._container.get_typed("database", DatabaseService)
            if db_result.is_success:
                db = db_result.unwrap()
                db.connect()  # Connect database first

                user_result = repo.find_by_id("user_1")
                if user_result.is_success:
                    user = user_result.unwrap()
                    print(f"   Found user: {user.name}")
        else:
            print(f"❌ Auto-wire failed: {result.error}")

    # ========== CONFIGURATION ==========

    def demonstrate_configuration(self) -> None:
        """Show container configuration."""
        print("\n=== Container Configuration ===")

        # Configure container with settings
        config: dict[str, object] = {
            "services": {
                "database": {
                    "connection_string": "postgresql://prod/db",
                    "pool_size": 10,
                },
                "cache": {
                    "ttl": 3600,
                    "max_size": 1000,
                },
            },
            "auto_wire": {
                "enabled": True,
                "scan_packages": ["flext_core"],
            },
        }

        result = self._container.configure_container(config)
        print(f"Container configuration: {result.is_success}")

        # Get configuration info
        info = self._container.get_info()
        print(f"Container info: {info}")

    # ========== SERVICE LIFECYCLE ==========

    def demonstrate_service_lifecycle(self) -> None:
        """Show singleton vs factory lifecycles."""
        print("\n=== Service Lifecycles ===")

        # Singleton: same instance every time
        self._container.register("singleton_cache", CacheService())

        cache1_result = self._container.get("singleton_cache")
        cache2_result = self._container.get("singleton_cache")

        if cache1_result.is_success and cache2_result.is_success:
            cache1 = cache1_result.unwrap()
            cache2 = cache2_result.unwrap()
            print(f"Singleton same instance: {cache1 is cache2}")

        # Factory: new instance every time
        self._container.register_factory("factory_cache", CacheService)

        cache3_result = self._container.get("factory_cache")
        cache4_result = self._container.get("factory_cache")

        if cache3_result.is_success and cache4_result.is_success:
            cache3 = cache3_result.unwrap()
            cache4 = cache4_result.unwrap()
            print(f"Factory new instances: {cache3 is not cache4}")

    # ========== GLOBAL CONTAINER ==========

    def demonstrate_global_container(self) -> None:
        """Show global container patterns."""
        print("\n=== Global Container Patterns ===")

        # Get global singleton
        global_container = FlextContainer.get_global()
        print(f"Global container: {type(global_container).__name__}")

        # Register globally
        result = FlextContainer.register_global("global_service", {"data": "global"})
        print(f"Register global: {result.is_success}")

        # Get from any container instance
        new_container = FlextContainer()
        global_result = new_container.get("global_service")
        if global_result.is_success:
            print("✅ Got global service from new container")
        else:
            print(f"❌ Global service not accessible: {global_result.error}")

    # ========== ERROR HANDLING ==========

    def demonstrate_error_handling(self) -> None:
        """Show error handling patterns."""
        print("\n=== Error Handling ===")

        # Try to get non-existent service
        result = self._container.get("non_existent")
        if result.is_failure:
            print(f"✅ Correct failure for non-existent: {result.error}")

        # Try to register with invalid name
        register_result = self._container.register("", DatabaseService())
        if register_result.is_failure:
            print(f"✅ Correct failure for empty name: {register_result.error}")

        # Type mismatch in get_typed
        self._container.register("wrong_type", CacheService())
        result = self._container.get_typed("wrong_type", DatabaseService)
        if result.is_failure:
            print(f"✅ Correct failure for type mismatch: {result.error}")

    # ========== DEPRECATED PATTERNS ==========

    def demonstrate_deprecated_patterns(self) -> None:
        """Show deprecated patterns with warnings."""
        print("\n=== ⚠️ DEPRECATED PATTERNS ===")

        # OLD: Manual singleton pattern (DEPRECATED)
        warnings.warn(
            "Manual singleton pattern is DEPRECATED! Use FlextContainer.register().",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (manual singleton):")
        print("class DatabaseService:")
        print("    _instance = None")
        print("    def __new__(cls):")
        print("        if cls._instance is None:")
        print("            cls._instance = super().__new__(cls)")
        print("        return cls._instance")

        print("\n✅ CORRECT WAY (FlextContainer):")
        print("container.register('database', DatabaseService())")

        # OLD: Service locator anti-pattern (DEPRECATED)
        warnings.warn(
            "Service locator is an ANTI-PATTERN! Use dependency injection.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (service locator):")
        print("class UserService:")
        print("    def get_user(self):")
        print("        db = ServiceLocator.get('database')  # Anti-pattern!")

        print("\n✅ CORRECT WAY (dependency injection):")
        print("class UserService:")
        print("    def __init__(self, database: DatabaseService):")
        print("        self._database = database  # Injected dependency")

        # OLD: Global variables (DEPRECATED)
        warnings.warn(
            "Global variables are DEPRECATED! Use FlextContainer for state.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (global variables):")
        print("DATABASE = None  # Global variable")
        print("CACHE = None     # Global variable")

        print("\n✅ CORRECT WAY (FlextContainer):")
        print("container = FlextContainer.get_global()")
        print("container.register('database', DatabaseService())")


def main() -> None:
    """Main entry point demonstrating all FlextContainer capabilities."""
    service = ComprehensiveDIService()

    print("=" * 60)
    print("FLEXTCONTAINER COMPLETE API DEMONSTRATION")
    print("Foundation for Dependency Injection in FLEXT Ecosystem")
    print("=" * 60)

    # Core patterns
    service.demonstrate_basic_registration()
    service.demonstrate_service_resolution()
    service.demonstrate_batch_operations()

    # Advanced patterns
    service.demonstrate_auto_wiring()
    service.demonstrate_configuration()
    service.demonstrate_service_lifecycle()

    # Professional patterns
    service.demonstrate_global_container()
    service.demonstrate_error_handling()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    print("\n" + "=" * 60)
    print("✅ ALL FlextContainer methods demonstrated!")
    print("🎯 Next: See 03_models_basics.py for FlextModels patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
