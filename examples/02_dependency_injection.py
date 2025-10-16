#!/usr/bin/env python3
"""02 - FlextContainer Fundamentals: Complete Dependency Injection.

This example demonstrates the COMPLETE FlextContainer[T] API - the foundation
for dependency injection across the entire FLEXT ecosystem. FlextContainer provides
type-safe service registration, resolution, and lifecycle management with railway-oriented
error handling and SOLID principles.

Key Concepts Demonstrated:
- Service registration: register(), register_factory(), batch_register()
- Service resolution: get(), get_typed(), get_or_create()
- Auto-wiring: auto_wire() with dependency resolution
- Container management: clear(), has(), list_services()
- Global singleton: get_global(), register_global()
- Configuration: configure(), configure_container()
- Service lifecycles: singleton vs factory patterns
- Error handling: Railway patterns with FlextResult[T]
- Type safety: Full generic type support with Pydantic 2

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import time
import warnings
from copy import deepcopy
from datetime import UTC, datetime
from typing import ClassVar, Protocol, cast, runtime_checkable

from flext_core import (
    FlextConfig,
    FlextConstants,
    FlextContainer,
    FlextLogger,
    FlextModels,
    FlextResult,
    FlextService,
    FlextTypes,
)


class DemoScenarios:
    """Inline scenario helpers for dependency injection demonstrations."""

    _DATASET: ClassVar[FlextTypes.Dict] = {
        "users": [
            {
                "id": 1,
                "name": "Alice Example",
                "email": "alice@example.com",
                "age": 30,
            },
            {
                "id": 2,
                "name": "Bob Example",
                "email": "bob@example.com",
                "age": 28,
            },
        ],
        "services": ["database", "cache", "email"],
    }

    _CONFIG: ClassVar[FlextTypes.Dict] = {
        "database_url": "sqlite:///:memory:",
        "api_timeout": 30,
        "retry": 3,
    }

    _PROD_CONFIG: ClassVar[FlextTypes.Dict] = {
        "database_url": "postgresql://prod-db/flext",
        "api_timeout": 20,
        "retry": 5,
    }

    _PAYLOAD: ClassVar[FlextTypes.Dict] = {
        "event": "user_registered",
        "user_id": "usr-123",
        "metadata": {"source": "examples", "version": "1.0"},
    }

    @staticmethod
    def dataset() -> FlextTypes.Dict:
        """Get a copy of the demo dataset for dependency injection examples."""
        return deepcopy(DemoScenarios._DATASET)

    @staticmethod
    def config(*, production: bool = False) -> FlextTypes.Dict:
        """Get a copy of the demo configuration (production or development)."""
        base = DemoScenarios._PROD_CONFIG if production else DemoScenarios._CONFIG
        return deepcopy(base)

    @staticmethod
    def user(**overrides: object) -> FlextTypes.Dict:
        """Get a demo user object with optional overrides."""
        dataset: FlextTypes.Dict = DemoScenarios._DATASET
        users_raw = dataset["users"]
        users_list: list[FlextTypes.Dict] = cast("list[FlextTypes.Dict]", users_raw)
        user = deepcopy(users_list[0])
        user.update(overrides)
        return user

    @staticmethod
    def service_batch(logger_name: str = "example_batch") -> FlextTypes.Dict:
        """Get a demo service batch configuration."""
        return {
            "logger": FlextLogger(logger_name),
            "config": DemoScenarios.config(),
            "metrics": {"requests": 0, "errors": 0},
        }

    @staticmethod
    def payload(**overrides: object) -> FlextTypes.Dict:
        """Get a demo payload with optional overrides."""
        payload = deepcopy(DemoScenarios._PAYLOAD)
        payload.update(overrides)
        return payload


# ========== SERVICE INTERFACES (PROTOCOLS) ==========


class DatabaseServiceProtocol(Protocol):
    """Protocol defining database service interface."""

    def connect(self) -> FlextResult[None]:
        """Connect to database."""
        ...

    def query(self, sql: str) -> FlextResult[list[FlextTypes.Dict]]:
        """Execute query."""
        ...

    def query_with_params(
        self,
        sql: str,
        params: FlextTypes.List,
        command_type: type[object],
    ) -> FlextResult[list[FlextTypes.Dict]]:
        """Execute parameterized query to prevent SQL injection."""
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


@runtime_checkable
class HasGetStats(Protocol):
    """Protocol for objects with get_stats method."""

    def get_stats(self) -> FlextTypes.Dict:
        """Return statistics dictionary."""
        ...


@runtime_checkable
class HasCacheSet(Protocol):
    """Protocol for cache objects with set method."""

    def set(self, key: str, value: object) -> FlextResult[None]:
        """Set value in cache."""
        ...


@runtime_checkable
class HasCacheGet(Protocol):
    """Protocol for cache objects with get method."""

    def get(self, key: str) -> FlextResult[object]:
        """Get value from cache."""
        ...


# ========== SERVICE IMPLEMENTATIONS ==========


class DatabaseService(FlextService[None]):
    """Concrete database service implementation with enhanced error handling."""

    def __init__(self, connection_string: str = "sqlite:///:memory:") -> None:
        """Initialize with connection string using FLEXT patterns."""
        super().__init__()
        self._connection_string = connection_string
        self.connected = False
        self._query_count = 0

    def execute(self) -> FlextResult[None]:
        """Execute the database service."""
        return FlextResult[None].ok(None)

    def connect(self) -> FlextResult[None]:
        """Connect to database with state validation."""
        if self.connected:
            return FlextResult[None].fail(
                "Database already connected",
                error_code=FlextConstants.Errors.ALREADY_EXISTS,
                error_data={"connection_string": self._connection_string},
            )

        # Simulate connection attempt
        self.connected = True
        self.logger.info(
            "Database connection established",
            extra={"connection_string": self._connection_string},
        )
        return FlextResult[None].ok(None)

    def query(self, sql: str) -> FlextResult[list[FlextTypes.Dict]]:
        """Execute query with comprehensive error handling."""
        if not self.connected:
            return FlextResult[list[FlextTypes.Dict]].fail(
                "Database not connected",
                error_code=FlextConstants.Errors.CONNECTION_ERROR,
                error_data={"connection_string": self._connection_string},
            )

        if not sql or not sql.strip():
            return FlextResult[list[FlextTypes.Dict]].fail(
                "SQL query cannot be empty",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Simulate query execution with metrics
        self._query_count += 1
        query_preview = sql[: FlextConstants.Validation.PREVIEW_LENGTH] + (
            "..." if len(sql) > FlextConstants.Validation.PREVIEW_LENGTH else ""
        )
        self.logger.debug(f"Executing query #{self._query_count}: {query_preview}")

        # Simulate different responses based on query with enhanced data
        if "users" in sql.lower():
            return FlextResult[list[FlextTypes.Dict]].ok([
                {"id": 1, "name": "John Doe", "email": "john@example.com"},
                {"id": 2, "name": "Jane Smith", "email": "jane@example.com"},
            ])
        if "count" in sql.lower():
            return FlextResult[list[FlextTypes.Dict]].ok([{"count": 42}])

        return FlextResult[list[FlextTypes.Dict]].ok([])

    def query_with_params(
        self,
        sql: str,
        params: FlextTypes.List,
        command_type: type[object],
    ) -> FlextResult[list[FlextTypes.Dict]]:
        """Execute parameterized query to prevent SQL injection."""
        if not self.connected:
            return FlextResult[list[FlextTypes.Dict]].fail(
                "Database not connected",
                error_code=FlextConstants.Errors.CONNECTION_ERROR,
                error_data={"connection_string": self._connection_string},
            )

        if not sql or not sql.strip():
            return FlextResult[list[FlextTypes.Dict]].fail(
                "SQL query cannot be empty",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Parameters are already validated as FlextTypes.List type in the protocol

        # Validate command type
        valid_command_types = (FlextTypes.Dict, FlextTypes.List, tuple)
        if command_type not in valid_command_types:
            return FlextResult[list[FlextTypes.Dict]].fail(
                "Invalid command type for query result",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Simulate parameterized query execution with metrics
        self._query_count += 1
        query_preview = sql[: FlextConstants.Validation.PREVIEW_LENGTH] + (
            "..." if len(sql) > FlextConstants.Validation.PREVIEW_LENGTH else ""
        )
        param_preview = str(params)[: FlextConstants.Validation.PREVIEW_LENGTH] + (
            "..." if len(str(params)) > FlextConstants.Validation.PREVIEW_LENGTH else ""
        )
        self.logger.debug(
            f"Executing parameterized query #{self._query_count}: {query_preview} with params: {param_preview}",
        )

        # Simulate different responses based on query with enhanced data
        if "users" in sql.lower():
            user_data: list[FlextTypes.Dict] = [
                {"id": 1, "name": "John Doe", "email": "john@example.com"},
                {"id": 2, "name": "Jane Smith", "email": "jane@example.com"},
            ]
            return FlextResult[list[FlextTypes.Dict]].ok(user_data)
        if "count" in sql.lower():
            count_data: FlextTypes.Dict = {"count": 42}
            return FlextResult[list[FlextTypes.Dict]].ok([count_data])

        return FlextResult[list[FlextTypes.Dict]].ok([])

    def get_stats(self) -> FlextTypes.Dict:
        """Get database service statistics."""
        return {
            "connection_string": self._connection_string,
            "connected": self.connected,
            "queries_executed": self._query_count,
        }


class CacheService(FlextService[object]):
    """Concrete cache service implementation with advanced FLEXT patterns."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600) -> None:
        """Initialize cache with size and TTL limits."""
        super().__init__()
        self._cache: FlextTypes.Dict = {}
        self._metadata: FlextTypes.NestedDict = {}
        self.max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0

    def execute(self) -> FlextResult[object]:
        """Execute the cache service."""
        return FlextResult[object].ok(None)

    def get(self, key: str) -> FlextResult[object]:
        """Get value from cache with TTL validation."""
        if not key or not key.strip():
            return FlextResult[object].fail(
                "Cache key cannot be empty",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Check if key exists
        if key not in self._cache:
            self._misses += 1
            self.logger.debug("Cache miss: %s", key)
            return FlextResult[object].fail(
                f"Key not found: {key}",
                error_code=FlextConstants.Errors.NOT_FOUND,
            )

        # Check TTL if metadata exists
        if key in self._metadata:
            metadata = self._metadata[key]
            current_time = time.time()
            created_at_raw = metadata.get("created_at", 0)
            created_at = 0.0
            if created_at_raw is not None:
                try:
                    # Safe conversion with type checking
                    if isinstance(created_at_raw, (int, float, str)):
                        created_at = float(created_at_raw)
                    else:
                        created_at = 0.0
                except (TypeError, ValueError):
                    created_at = 0.0

            if current_time - created_at > self._ttl_seconds:
                # Expired - remove from cache
                del self._cache[key]
                del self._metadata[key]
                self._misses += 1
                self.logger.debug("Cache expired: %s", key)
                return FlextResult[object].fail(
                    f"Key expired: {key}",
                    error_code=FlextConstants.Errors.OPERATION_ERROR,
                )

        # Valid hit
        self._hits += 1
        self.logger.debug("Cache hit: %s", key)
        return FlextResult[object].ok(self._cache[key])

    def set(
        self,
        key: str,
        value: object,
        ttl_seconds: int | None = None,
    ) -> FlextResult[None]:
        """Set value in cache with optional TTL override."""
        if not key or not key.strip():
            return FlextResult[None].fail(
                "Cache key cannot be empty",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        if value is None:
            return FlextResult[None].fail(
                "Cache value cannot be None",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Check size limit and evict if necessary
        if len(self._cache) >= self.max_size:
            self._evict_oldest()

        # Set the value
        self._cache[key] = value
        self._metadata[key] = {
            "created_at": time.time(),
            "ttl_seconds": ttl_seconds or self._ttl_seconds,
        }

        self.logger.debug(
            f"Cache set: {key} (TTL: {ttl_seconds or self._ttl_seconds}s)",
        )
        return FlextResult[None].ok(None)

    def _evict_oldest(self) -> None:
        """Evict the oldest cache entry using LRU strategy."""
        if not self._metadata:
            return

        # Find oldest entry
        def get_timestamp(key: str) -> float:
            metadata = self._metadata[key]
            created_at_raw = metadata.get("created_at", 0)
            if created_at_raw is not None:
                try:
                    # Safe conversion with type checking
                    if isinstance(created_at_raw, (int, float, str)):
                        return float(created_at_raw)
                    return 0.0
                except (TypeError, ValueError):
                    return 0.0
            return 0.0

        oldest_key = min(self._metadata.keys(), key=get_timestamp)

        del self._cache[oldest_key]
        del self._metadata[oldest_key]
        self.logger.debug("Cache evicted: %s", oldest_key)

    def get_stats(self) -> FlextTypes.Dict:
        """Get cache service statistics."""
        return {
            "max_size": self.max_size,
            "current_size": len(self._cache),
            "ttl_seconds": self._ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / (self._hits + self._misses))
            if (self._hits + self._misses) > 0
            else 0.0,
        }


class EmailService(FlextService[None]):
    """Concrete email service implementation."""

    def __init__(self, smtp_host: str = FlextConstants.Platform.DEFAULT_HOST) -> None:
        """Initialize with SMTP host."""
        super().__init__()
        self._smtp_host = smtp_host

    def execute(self) -> FlextResult[None]:
        """Execute the email service."""
        return FlextResult[None].ok(None)

    def send(self, to: str, subject: str, body: str) -> FlextResult[None]:
        """Send email (simulated)."""
        self.logger.info("Email sent to %s: %s", to, subject)
        _ = body
        return FlextResult[None].ok(None)


# ========== DOMAIN MODELS ==========


class User(FlextModels.Entity):
    """User entity with domain logic."""

    name: str
    email: str
    age: int
    is_active: bool = True


class UserRepository(FlextService[User]):
    """Repository pattern for User entities with enhanced domain logic."""

    def __init__(self) -> None:
        """Initialize with database dependency using FLEXT patterns."""
        super().__init__()
        container = FlextContainer.get_global()
        self._database = container.get("database").unwrap_or(None)
        self._cache: dict[str, User] = {}

    def execute(self) -> FlextResult[User]:
        """Execute the main domain operation - find default user."""
        if self._database is None:
            return FlextResult[User].fail(
                "Database not available",
                error_code=FlextConstants.Errors.EXTERNAL_SERVICE_ERROR,
            )

        # Simulate database query
        user = User(
            id="default_user",
            name="Default User",
            email="default@example.com",
            age=0,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            version=1,
            domain_events=[],
        )
        self._cache[user.id] = user
        self.logger.info("User loaded from database: %s", user.id)
        return FlextResult[User].ok(user)

    def save(self, user: User) -> FlextResult[None]:
        """Save user to database with validation."""
        # Validate user before saving
        validation_result = self._validate_user(user)
        if validation_result.is_failure:
            return validation_result

        # Simulate save operation with enhanced logging
        self.logger.info(
            f"Saving user to database: {user.id}",
            extra={"user_email": user.email, "user_age": user.age},
        )

        # Update cache
        self._cache[user.id] = user

        return FlextResult[None].ok(None)

    def find_by_id(self, user_id: str) -> FlextResult[User]:
        """Find user by ID from cache or database."""
        # Check cache first
        if user_id in self._cache:
            return FlextResult[User].ok(self._cache[user_id])

        # Simulate database lookup
        if user_id == "default_user":
            user = User(
                id=user_id,
                name="Default User",
                email="default@example.com",
                age=0,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                version=1,
                domain_events=[],
            )
            self._cache[user_id] = user
            return FlextResult[User].ok(user)

        return FlextResult[User].fail(
            f"User not found: {user_id}",
            error_code=FlextConstants.Errors.NOT_FOUND,
        )

    def _validate_user(self, user: User) -> FlextResult[None]:
        """Validate user data before saving."""
        if not user.name or not user.name.strip():
            return FlextResult[None].fail(
                "User name cannot be empty",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        if not user.email or "@" not in user.email:
            return FlextResult[None].fail(
                "User email must be valid",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        if (
            user.age < FlextConstants.Validation.MIN_AGE
            or user.age > FlextConstants.Validation.MAX_AGE
        ):
            return FlextResult[None].fail(
                f"User age must be between {FlextConstants.Validation.MIN_AGE} and {FlextConstants.Validation.MAX_AGE}",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        return FlextResult[None].ok(None)

    def get_stats(self) -> FlextTypes.Dict:
        """Get repository statistics."""
        return {
            "cached_users": len(self._cache),
            "database_service_type": type(self._database).__name__,
        }


# ========== COMPREHENSIVE CONTAINER SERVICE ==========


class ComprehensiveDIService(FlextService[User]):
    """Service demonstrating ALL FlextContainer patterns with FlextMixins infrastructure.

    This service inherits from FlextService to demonstrate:
    - Inherited container property (FlextContainer singleton for DI)
    - Inherited logger property (FlextLogger with service context)
    - Inherited context property (FlextContext for request tracking)
    - Inherited config property (FlextConfig with settings)
    - Inherited metrics property (FlextMetrics for observability)

    The focus is on demonstrating FlextContainer DI patterns while leveraging
    the complete FlextMixins infrastructure for service orchestration.
    """

    def __init__(self) -> None:
        """Initialize with inherited FlextMixins infrastructure.

        Note: No manual logger or container initialization needed!
        All infrastructure is inherited from FlextService base class:
        - self.logger: FlextLogger with service context
        - self.container: FlextContainer global singleton
        - self.context: FlextContext for request tracking
        - self.config: FlextConfig with application settings
        - self.metrics: FlextMetrics for observability
        """
        super().__init__()
        self._scenarios = DemoScenarios()
        self._dataset = self._scenarios.dataset()
        self._config_data = self._scenarios.config()

        # Demonstrate inherited logger (no manual instantiation needed!)
        self.logger.info(
            "ComprehensiveDIService initialized with inherited infrastructure",
            extra={
                "dataset_keys": list(self._dataset.keys()),
                "config_keys": list(self._config_data.keys()),
                "service_type": "FlextContainer DI demonstration",
            },
        )

    def execute(self) -> FlextResult[User]:
        """Execute with automatic error handling and monitoring.

        This method satisfies the FlextService abstract interface while
        demonstrating FlextContainer DI patterns. Uses inherited infrastructure:
        - self.container for service resolution and dependency injection
        - self.logger for structured logging throughout execution
        - self.context for request tracking (if needed)

        Returns:
            FlextResult containing User entity from DI container operations

        """
        return self._execute_with_container_patterns()

    def _execute_with_container_patterns(self) -> FlextResult[User]:
        """Execute using advanced container patterns."""
        db_result = self.container.get("database")
        if db_result.is_failure:
            return FlextResult[User].fail(
                f"Cannot access database: {db_result.error}",
                error_code=FlextConstants.Errors.EXTERNAL_SERVICE_ERROR,
            )

        cache_result = self.container.get("cache")
        if cache_result.is_failure:
            return FlextResult[User].fail(
                f"Cannot access cache: {cache_result.error}",
                error_code=FlextConstants.Errors.EXTERNAL_SERVICE_ERROR,
            )

        user_source = self._scenarios.user()
        age_raw = user_source.get("age", 0)
        user = User(
            id=str(user_source["id"]),
            name=str(user_source["name"]),
            email=str(user_source["email"]),
            age=int(age_raw) if isinstance(age_raw, (int, str)) else 0,
            created_at=datetime.now(UTC),
            updated_at=None,
            version=1,
        )

        self._log_service_statistics()

        return FlextResult[User].ok(user)

    def _log_service_statistics(self) -> None:
        """Log comprehensive service statistics."""
        try:
            # Get service stats from registered services
            db_stats = None
            cache_stats = None

            db_result = self.container.get("database")
            if db_result.is_success:
                db = db_result.unwrap()
                if isinstance(db, HasGetStats):
                    db_stats = db.get_stats()

            cache_result = self.container.get("cache")
            if cache_result.is_success:
                cache = cache_result.unwrap()
                if isinstance(cache, HasGetStats):
                    cache_stats = cache.get_stats()

            # Log comprehensive statistics
            service_count = len(
                set(self.container.services.keys())
                | set(self.container.factories.keys())
            )
            self.logger.info(
                "Service execution statistics",
                extra={
                    "database_stats": db_stats,
                    "cache_stats": cache_stats,
                    "container_service_count": service_count,
                },
            )
        except Exception as e:
            self.logger.warning("Failed to log statistics: %s", e)

    def get_service_stats(self) -> FlextTypes.Dict:
        """Get comprehensive service statistics."""
        service_count = len(
            set(self.container.services.keys()) | set(self.container.factories.keys())
        )
        return {
            "container_service_count": service_count,
        }

    # ========== BASIC REGISTRATION ==========

    def demonstrate_basic_registration(self) -> None:
        """Show basic service registration patterns."""
        print("\n=== Basic Service Registration ===")

        self.container.clear()
        print("✅ Container cleared")

        connection = str(self._config_data.get("database_url", "sqlite:///:memory:"))
        db_service = DatabaseService(connection)
        result = self.container.register("database", db_service)
        print(f"Register singleton: {result.is_success}")

        result = self.container.register_factory("cache", CacheService)
        print(f"Register factory: {result.is_success}")

        has_db = self.container.has("database")
        has_cache = self.container.has("cache")
        print(f"Has database: {has_db}, Has cache: {has_cache}")

        service_count = len(
            set(self.container.services.keys()) | set(self.container.factories.keys())
        )
        print(f"Service count: {service_count}")

    # ========== SERVICE RESOLUTION ==========

    def demonstrate_service_resolution(self) -> None:
        """Show all ways to resolve services with enhanced monitoring."""
        print("\n=== Service Resolution ===")

        dataset = self._dataset
        users_list = cast("FlextTypes.List", dataset.get("users", []))
        sample_user = cast("FlextTypes.Dict", users_list[0] if users_list else {})

        db_result = self.container.get("database")
        if db_result.is_success:
            db = db_result.unwrap()
            print(f"✅ Got database: {type(db).__name__}")
            if isinstance(db, HasGetStats):
                stats_method = db.get_stats
                stats = stats_method()
                print(
                    f"   Database stats: {stats.get('queries_executed', 0)} queries executed",
                )
        else:
            print(f"❌ Failed to get database: {db_result.error}")

        typed_result: FlextResult[object] = self.container.get_typed(
            "database", DatabaseService
        )
        if typed_result.is_success:
            db_typed = cast("DatabaseService", typed_result.unwrap())
            print(f"✅ Got typed database: {type(db_typed).__name__}")
            # Safe access to database stats
            stats = db_typed.get_stats()
            print(
                f"   Database stats: {stats.get('queries_executed', 0)} queries executed",
            )
        else:
            print(f"❌ Type validation failed: {typed_result.error}")

        user_dict = sample_user
        email_result = self.container.get_or_create(
            "email",
            lambda: EmailService(str(user_dict.get("email", "smtp@example.com"))),
        )
        print(f"Get or create email: {email_result.is_success}")

        cache_result = self.container.get("cache")
        if cache_result.is_success:
            cache = cache_result.unwrap()
            user_dict = sample_user
            cache_key = user_dict.get("id", "test_user")
            if isinstance(cache, HasCacheSet):
                set_method = cache.set
                set_method(str(cache_key), sample_user)
            if isinstance(cache, HasCacheGet):
                get_method = cache.get
                get_result = get_method(str(cache_key))
            else:
                get_result = FlextResult[object].fail("Cache get method not available")
            if get_result.is_success:
                cached_data = get_result.unwrap()
                if isinstance(cached_data, dict):
                    cached_dict = cast("FlextTypes.Dict", cached_data)
                    print(f"✅ Cache test: {cached_dict.get('email')}")

            cache_stats_func: object = getattr(cache, "get_stats", None)
            stats_raw: object = cache_stats_func() if callable(cache_stats_func) else {}
            stats_dict: FlextTypes.Dict = (
                cast("FlextTypes.Dict", stats_raw)
                if isinstance(stats_raw, dict)
                else {}
            )
            print(
                f"   Cache stats: {stats_dict.get('current_size', 0)} items, {stats_dict.get('hits', 0)} hits, {stats_dict.get('misses', 0)} misses"
            )

    # ========== BATCH OPERATIONS ==========

    def demonstrate_batch_operations(self) -> None:
        """Show batch registration patterns."""
        print("\n=== Batch Operations ===")

        services = self._scenarios.service_batch("container_batch")

        result = self.container.batch_register(services)
        if result.is_success:
            print(f"✅ Batch registered {len(services)} services")
        else:
            print(f"❌ Batch registration failed: {result.error}")

        services_result = self.container.list_services()
        if services_result.is_success:
            services_list = services_result.unwrap()
            print(f"All services: {services_list}")
        else:
            print(f"❌ Failed to FlextTypes.List services: {services_result.error}")

    # ========== AUTO-WIRING ==========

    def demonstrate_auto_wiring(self) -> None:
        """Show dependency auto-wiring."""
        print("\n=== Auto-Wiring ===")

        self.container.register("database", DatabaseService())

        result = self.container.auto_wire(UserRepository)
        if result.is_success:
            repo = cast("UserRepository", result.unwrap())
            print(f"✅ Auto-wired: {type(repo).__name__}")

            db_result: FlextResult[object] = self.container.get_typed(
                "database", DatabaseService
            )
            if db_result.is_success:
                db = cast("DatabaseService", db_result.unwrap())
                db.connect()

                users_list_result = self._dataset.get(
                    "users", cast("list[FlextTypes.Dict]", [{}])
                )
                users_list: FlextTypes.List = cast("FlextTypes.List", users_list_result)
                user_id = str(
                    cast("FlextTypes.Dict", users_list[0]).get("id", "user_1")
                )
                user_result = repo.find_by_id(user_id)
                if user_result.is_success:
                    user = user_result.unwrap()
                    print(f"   Found user: {user.name}")
        else:
            print(f"❌ Auto-wire failed: {result.error}")

    # ========== CONFIGURATION ==========

    def demonstrate_configuration(self) -> None:
        """Show container configuration."""
        print("\n=== Container Configuration ===")

        production_config = self._scenarios.config(production=True)
        config: FlextTypes.Dict = {
            "services": {
                "database": {
                    "connection_string": production_config.get(
                        "database_url",
                        "postgresql://prod/db",
                    ),
                    "pool_size": production_config.get("max_connections", 10),
                },
                "cache": {
                    "ttl": self._config_data.get("api_timeout", 30),
                    "max_size": production_config.get("max_connections", 1000),
                },
            },
            "auto_wire": {
                "enabled": True,
                "scan_packages": ["flext_core"],
            },
        }

        result = self.container.configure_container(config)
        print(f"Container configuration: {result.is_success}")

        # Build info manually since get_info() is removed
        service_count = len(
            set(self.container.services.keys()) | set(self.container.factories.keys())
        )
        info = {
            "service_count": service_count,
            "direct_services": len(self.container.services),
            "factories": len(self.container.factories),
        }
        print(f"Container info: {info}")

    # ========== SERVICE LIFECYCLE ==========

    def demonstrate_service_lifecycle(self) -> None:
        """Show singleton vs factory lifecycles."""
        print("\n=== Service Lifecycles ===")

        # Singleton: same instance every time
        self.container.register("singleton_cache", CacheService())

        cache1_result = self.container.get("singleton_cache")
        cache2_result = self.container.get("singleton_cache")

        if cache1_result.is_success and cache2_result.is_success:
            cache1 = cache1_result.unwrap()
            cache2 = cache2_result.unwrap()
            print(f"Singleton same instance: {cache1 is cache2}")

        # Factory: Note - current implementation caches factory results
        # This demonstrates the current behavior, not ideal factory pattern
        self.container.register_factory("factory_cache", CacheService)

        cache3_result = self.container.get("factory_cache")
        cache4_result = self.container.get("factory_cache")

        if cache3_result.is_success and cache4_result.is_success:
            cache3 = cache3_result.unwrap()
            cache4 = cache4_result.unwrap()
            print(f"Factory instances (cached): {cache3 is cache4}")
            print("   Note: Current implementation caches factory results")

    # ========== GLOBAL CONTAINER ==========

    def demonstrate_global_container(self) -> None:
        """Show global container patterns."""
        print("\n=== Global Container Patterns ===")

        global_container = FlextContainer.get_global()
        print(f"Global container: {type(global_container).__name__}")

        global_payload = self._scenarios.payload(type="global_service")
        result = global_container.register("global_service", global_payload)
        print(f"Register global: {result.is_success}")

        global_result = global_container.get("global_service")
        if global_result.is_success:
            print("✅ Got global service from global container")
        else:
            print(f"❌ Global service not accessible: {global_result.error}")

        # Note: Global services are only accessible from the global container instance

    # ========== ERROR HANDLING ==========

    def demonstrate_error_handling(self) -> None:
        """Show error handling patterns."""
        print("\n=== Error Handling ===")

        # Try to get non-existent service
        result = self.container.get("non_existent")
        if result.is_failure:
            print(f"✅ Correct failure for non-existent: {result.error}")

        # Try to register with invalid name
        register_result = self.container.register("", DatabaseService())
        if register_result.is_failure:
            print(f"✅ Correct failure for empty name: {register_result.error}")

        # Type mismatch in get_typed
        self.container.register("wrong_type", CacheService())
        result = self.container.get_typed("wrong_type", DatabaseService)
        if result.is_failure:
            print(f"✅ Correct failure for type mismatch: {result.error}")

    # ========== NEW FlextResult METHODS (v0.9.9+) ==========

    def demonstrate_from_callable(self) -> None:
        """Show from_callable for safe service initialization."""
        print("\n=== from_callable(): Safe Service Initialization ===")

        # Safe database connection initialization
        def risky_db_connect() -> DatabaseService:
            db = DatabaseService("postgresql://invalid-host/db")
            connect_result = db.connect()
            if connect_result.is_failure:
                msg = "Database connection failed"
                raise RuntimeError(msg)
            return db

        db_result = FlextResult[DatabaseService].from_callable(risky_db_connect)
        if db_result.is_failure:
            print(f"✅ Caught connection error safely: {db_result.error}")
        else:
            print("Database connected successfully")

        # Safe cache initialization
        def safe_cache_init() -> CacheService:
            return CacheService(max_size=100, ttl_seconds=300)

        cache_result = FlextResult[CacheService].from_callable(safe_cache_init)
        if cache_result.is_success:
            cache = cache_result.unwrap()
            print(f"✅ Cache initialized: max_size={cache.max_size}")

    def demonstrate_flow_through(self) -> None:
        """Show pipeline composition for service initialization."""
        print("\n=== flow_through(): Service Initialization Pipeline ===")

        def connect_database(db: DatabaseService) -> FlextResult[DatabaseService]:
            """Connect to database."""
            result = db.connect()
            if result.is_failure:
                return FlextResult[DatabaseService].fail(
                    f"Connection failed: {result.error}"
                )
            return FlextResult[DatabaseService].ok(db)

        def validate_database(db: DatabaseService) -> FlextResult[DatabaseService]:
            """Validate database is ready."""
            if not db.connected:
                return FlextResult[DatabaseService].fail("Database not connected")
            return FlextResult[DatabaseService].ok(db)

        # Pipeline: create → connect → validate
        result = (
            FlextResult[DatabaseService]
            .ok(DatabaseService())
            .flow_through(
                connect_database,
                validate_database,
            )
        )

        if result.is_success:
            db = result.unwrap()
            print(f"✅ Service pipeline success: connected={db.connected}")
        else:
            print(f"Pipeline failure: {result.error}")

    def demonstrate_lash(self) -> None:
        """Show error recovery with fallback services."""
        print("\n=== lash(): Service Fallback Pattern ===")

        # Primary service that fails
        def try_primary_database() -> FlextResult[str]:
            """Try to get data from primary database."""
            return FlextResult[str].fail("Primary database unavailable")

        # Recovery function using cache
        def recover_with_cache(error: str) -> FlextResult[str]:
            """Recover by using cache service."""
            print(f"  Recovering from: {error}")
            cache = CacheService()
            cache.set("fallback_key", "cached_data")
            cached_result = cache.get("fallback_key")
            if cached_result.is_success:
                data = cached_result.unwrap()
                return FlextResult[str].ok(str(data))
            return FlextResult[str].fail("Cache also unavailable")

        result = try_primary_database().lash(recover_with_cache)
        if result.is_success:
            print(f"✅ Recovered with fallback: {result.unwrap()}")

        # Success case - no recovery needed
        def successful_operation() -> FlextResult[str]:
            return FlextResult[str].ok("Primary success")

        result = successful_operation().lash(recover_with_cache)
        print(f"Primary success (no recovery): {result.unwrap()}")

    def demonstrate_alt(self) -> None:
        """Show fallback pattern for service resolution."""
        print("\n=== alt(): Service Resolution Fallback ===")

        # Try primary service, use fallback if fails
        primary = self.container.get("non_existent_service")
        fallback = FlextResult[object].ok(CacheService())

        result = primary.alt(fallback)
        if result.is_success:
            service = result.unwrap()
            print(f"✅ Got fallback service: {type(service).__name__}")

        # Chain multiple fallbacks
        first = self.container.get("service1")
        second = self.container.get("service2")
        third = FlextResult[object].ok(EmailService())

        result = first.alt(second).alt(third)
        if result.is_success:
            service = result.unwrap()
            print(f"Fallback chain: {type(service).__name__}")

    def demonstrate_value_or_call(self) -> None:
        """Show lazy default evaluation for expensive service creation."""
        print("\n=== value_or_call(): Lazy Service Creation ===")

        expensive_created = False

        def expensive_service_creation() -> EmailService:
            """Expensive service initialization."""
            nonlocal expensive_created
            expensive_created = True
            print("  Creating expensive EmailService...")
            return EmailService("expensive-smtp.example.com")

        # Success case - expensive creation NOT called
        success = self.container.get("database")
        if success.is_success:
            service = success.value_or_call(expensive_service_creation)
            print(
                f"✅ Success: {type(service).__name__}, expensive_created={expensive_created}"
            )

        # Failure case - expensive creation IS called
        expensive_created = False
        failure = self.container.get("non_existent_service")
        service = failure.value_or_call(expensive_service_creation)
        print(
            f"Fallback: {type(service).__name__}, expensive_created={expensive_created}"
        )

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


def demonstrate_flext_di_access() -> None:
    """Demonstrate FlextConstants unified access to dependency injection.

    Shows how FlextConstants provides convenient access to the DI container
    alongside other flext-core components.
    """
    print("\n" + "=" * 60)
    print("FLEXT UNIFIED DI ACCESS")
    print("Modern pattern for dependency injection with FlextConstants")
    print("=" * 60)
    print("Modern pattern for dependency injection with FlextConstants")
    print("=" * 60)

    # 1. Access container through FlextConstants instance
    print("\n=== 1. Container Access Through FlextConstants ===")
    container = FlextContainer()
    print(f"  ✅ Container accessed: {type(container).__name__}")

    # 2. Register services through FlextConstants container
    print("\n=== 2. Service Registration ===")
    logger = FlextLogger("di-demo-service")
    logger.info("Registering services via FlextConstants container")

    container.register("demo_logger", logger)
    print("  ✅ Logger registered in container")

    # 3. Access configuration alongside container
    print("\n=== 3. Combined Configuration and DI ===")
    config = FlextConfig()
    print(f"  ✅ Config access: log_level = {config.log_level}")
    services_result = container.list_services()
    service_count = len(services_result.unwrap()) if services_result.is_success else 0
    print(f"  ✅ Container access: {service_count} services")

    # 4. Setup complete infrastructure with DI
    print("\n=== 4. Infrastructure Setup with DI ===")
    setup_result = FlextContainer.create_module_utilities("di-demo-service")

    if setup_result.is_success:
        infra = setup_result.unwrap()
        infra_container = infra["container"]
        infra_logger = infra["logger"]

        print("  ✅ Infrastructure initialized with DI:")
        print(f"     - Container: {type(infra_container).__name__}")
        print(f"     - Logger: {type(infra_logger).__name__}")
        if hasattr(infra_container, "list_services"):
            # Type assertion for container with list_services method
            container_with_services = cast("FlextContainer", infra_container)
            infra_services_result = container_with_services.list_services()
            infra_service_count = (
                len(infra_services_result.unwrap())
                if infra_services_result.is_success
                else 0
            )
        else:
            infra_service_count = 0
        print(f"     - Services: {infra_service_count}")

    # 5. Direct class access for type-safe operations
    print("\n=== 5. Type-Safe DI Operations ===")
    direct_container = FlextContainer.get_global()
    result = FlextResult[str].ok("Service initialized")

    print(f"  ✅ Direct container access: {type(direct_container).__name__}")
    print(f"  ✅ Result pattern: {result.value}")

    print("\n" + "=" * 60)
    print("✅ FlextConstants DI demonstration complete!")
    print("Benefits: Unified access, lazy loading, integrated patterns")
    print("=" * 60)


def main() -> None:
    """Main entry point demonstrating all FlextContainer capabilities."""
    service = ComprehensiveDIService()

    print("=" * 60)
    print("FLEXTCONTAINER COMPLETE API DEMONSTRATION")
    print("Foundation for Dependency Injection in FLEXT Ecosystem")
    print("=" * 60)

    # Core DI patterns
    service.demonstrate_basic_registration()
    service.demonstrate_service_resolution()
    service.demonstrate_batch_operations()

    # Advanced patterns
    service.demonstrate_auto_wiring()
    service.demonstrate_configuration()
    service.demonstrate_service_lifecycle()

    # Global container
    service.demonstrate_global_container()
    service.demonstrate_error_handling()

    # New FlextResult methods (v0.9.9+)
    service.demonstrate_from_callable()
    service.demonstrate_flow_through()
    service.demonstrate_lash()
    service.demonstrate_alt()
    service.demonstrate_value_or_call()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    # Modern FlextConstants pattern demonstration
    demonstrate_flext_di_access()

    print("\n" + "=" * 60)
    print("✅ FlextContainer dependency injection demonstration complete!")
    print(
        "✨ Including new v0.9.9+ methods: from_callable, flow_through, lash, alt, value_or_call"
    )
    print("🎯 Next: See 03_models_basics.py for FlextModels patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
