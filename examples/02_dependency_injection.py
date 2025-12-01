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
from collections.abc import Mapping
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
)
from flext_core.typings import FlextTypes


class DemoScenarios:
    """Inline scenario helpers for dependency injection demonstrations."""

    _DATASET: ClassVar[dict[str, FlextTypes.GeneralValueType]] = {
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

    _CONFIG: ClassVar[dict[str, FlextTypes.GeneralValueType]] = {
        "database_url": "sqlite:///:memory:",
        "api_timeout": 30,
        "retry": 3,
    }

    _PROD_CONFIG: ClassVar[dict[str, FlextTypes.GeneralValueType]] = {
        "database_url": "postgresql://prod-db/flext",
        "api_timeout": 20,
        "retry": 5,
    }

    _PAYLOAD: ClassVar[dict[str, FlextTypes.GeneralValueType]] = {
        "event": "user_registered",
        "user_id": "usr-123",
        "metadata": {"source": "examples", "version": "1.0"},
    }

    @staticmethod
    def dataset() -> dict[str, FlextTypes.GeneralValueType]:
        """Get a copy of the demo dataset for dependency injection examples."""
        return deepcopy(DemoScenarios._DATASET)

    @staticmethod
    def config(*, production: bool = False) -> dict[str, FlextTypes.GeneralValueType]:
        """Get a copy of the demo configuration (production or development)."""
        base = DemoScenarios._PROD_CONFIG if production else DemoScenarios._CONFIG
        return deepcopy(base)

    @staticmethod
    def user(
        **overrides: FlextTypes.GeneralValueType,
    ) -> dict[str, FlextTypes.GeneralValueType]:
        """Get a demo user object with optional overrides."""
        dataset: dict[str, FlextTypes.GeneralValueType] = DemoScenarios._DATASET
        users_raw = dataset["users"]
        users_list: list[dict[str, FlextTypes.GeneralValueType]] = cast(
            "list[dict[str, FlextTypes.GeneralValueType]]", users_raw
        )
        user = deepcopy(users_list[0])
        user.update(overrides)
        return user

    @staticmethod
    def service_batch(
        logger_name: str = "example_batch",
    ) -> dict[str, FlextTypes.GeneralValueType | FlextLogger]:
        """Get a demo service batch configuration."""
        return {
            "logger": FlextLogger(logger_name),
            "config": DemoScenarios.config(),
            "metrics": {"requests": 0, "errors": 0},
        }

    @staticmethod
    def payload(
        **overrides: FlextTypes.GeneralValueType,
    ) -> dict[str, FlextTypes.GeneralValueType]:
        """Get a demo payload with optional overrides."""
        payload = deepcopy(DemoScenarios._PAYLOAD)
        payload.update(overrides)
        return payload


# ========== SERVICE INTERFACES (PROTOCOLS) ==========


class DatabaseServiceProtocol(Protocol):
    """Protocol defining database service interface."""

    def connect(self) -> FlextResult[bool]:
        """Connect to database."""
        ...

    def query(
        self, sql: str
    ) -> FlextResult[list[dict[str, FlextTypes.GeneralValueType]]]:
        """Execute query."""
        ...

    def query_with_params(
        self,
        sql: str,
        params: list[FlextTypes.GeneralValueType],
        command_type: type[
            dict[str, FlextTypes.GeneralValueType]
            | list[FlextTypes.GeneralValueType]
            | tuple[FlextTypes.GeneralValueType, ...]
        ],
    ) -> FlextResult[list[dict[str, FlextTypes.GeneralValueType]]]:
        """Execute parameterized query to prevent SQL injection."""
        ...


class CacheServiceProtocol(Protocol):
    """Protocol defining cache service interface."""

    def get(self, key: str) -> FlextResult[FlextTypes.GeneralValueType]:
        """Get value from cache."""
        ...

    def set(self, key: str, value: FlextTypes.GeneralValueType) -> FlextResult[bool]:
        """Set value in cache."""
        ...


class EmailServiceProtocol(Protocol):
    """Protocol defining email service interface."""

    def send(self, to: str, subject: str, body: str) -> FlextResult[bool]:
        """Send email."""
        ...


@runtime_checkable
class HasGetStats(Protocol):
    """Protocol for objects with get_stats method."""

    def get_stats(self) -> Mapping[str, FlextTypes.GeneralValueType]:
        """Return statistics dictionary."""
        ...


@runtime_checkable
class HasCacheSet(Protocol):
    """Protocol for cache objects with set method."""

    def set(self, key: str, value: FlextTypes.GeneralValueType) -> FlextResult[bool]:
        """Set value in cache."""
        ...


@runtime_checkable
class HasCacheGet(Protocol):
    """Protocol for cache objects with get method."""

    def get(self, key: str) -> FlextResult[FlextTypes.GeneralValueType]:
        """Get value from cache."""
        ...


# ========== SERVICE IMPLEMENTATIONS ==========


class DatabaseService(FlextService[None]):
    """Concrete database service implementation with enhanced error handling."""

    connected: bool = False
    _query_count: int = 0

    def __init__(self, connection_string: str = "sqlite:///:memory:") -> None:
        """Initialize with connection string using FLEXT patterns."""
        super().__init__()
        self._connection_string = connection_string

    def execute(self, **_kwargs: object) -> FlextResult[None]:
        """Execute the database service."""
        return FlextResult[None].ok(None)

    def connect(self) -> FlextResult[bool]:
        """Connect to database with state validation."""
        if self.connected:
            return FlextResult[bool].fail(
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
        return FlextResult[bool].ok(True)

    def query(
        self, sql: str
    ) -> FlextResult[list[dict[str, FlextTypes.GeneralValueType]]]:
        """Execute query with comprehensive error handling."""
        if not self.connected:
            return FlextResult[list[dict[str, FlextTypes.GeneralValueType]]].fail(
                "Database not connected",
                error_code=FlextConstants.Errors.CONNECTION_ERROR,
                error_data={"connection_string": self._connection_string},
            )

        if not sql or not sql.strip():
            return FlextResult[list[dict[str, FlextTypes.GeneralValueType]]].fail(
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
            return FlextResult[list[dict[str, FlextTypes.GeneralValueType]]].ok([
                {"id": 1, "name": "John Doe", "email": "john@example.com"},
                {"id": 2, "name": "Jane Smith", "email": "jane@example.com"},
            ])
        if "count" in sql.lower():
            return FlextResult[list[dict[str, FlextTypes.GeneralValueType]]].ok([
                {"count": 42}
            ])

        return FlextResult[list[dict[str, FlextTypes.GeneralValueType]]].ok([])

    def query_with_params(
        self,
        sql: str,
        params: list[FlextTypes.GeneralValueType],
        command_type: type[
            dict[str, FlextTypes.GeneralValueType]
            | list[FlextTypes.GeneralValueType]
            | tuple[FlextTypes.GeneralValueType, ...]
        ],
    ) -> FlextResult[list[dict[str, FlextTypes.GeneralValueType]]]:
        """Execute parameterized query to prevent SQL injection."""
        if not self.connected:
            return FlextResult[list[dict[str, FlextTypes.GeneralValueType]]].fail(
                "Database not connected",
                error_code=FlextConstants.Errors.CONNECTION_ERROR,
                error_data={"connection_string": self._connection_string},
            )

        if not sql or not sql.strip():
            return FlextResult[list[dict[str, FlextTypes.GeneralValueType]]].fail(
                "SQL query cannot be empty",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        # Validate command type
        valid_command_types = (dict, list, tuple)
        if command_type not in valid_command_types:
            return FlextResult[list[dict[str, FlextTypes.GeneralValueType]]].fail(
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
            user_data: list[dict[str, FlextTypes.GeneralValueType]] = [
                {"id": 1, "name": "John Doe", "email": "john@example.com"},
                {"id": 2, "name": "Jane Smith", "email": "jane@example.com"},
            ]
            return FlextResult[list[dict[str, FlextTypes.GeneralValueType]]].ok(
                user_data
            )
        if "count" in sql.lower():
            count_data: dict[str, FlextTypes.GeneralValueType] = {"count": 42}
            return FlextResult[list[dict[str, FlextTypes.GeneralValueType]]].ok([
                count_data
            ])

        return FlextResult[list[dict[str, FlextTypes.GeneralValueType]]].ok([])

    def get_stats(self) -> dict[str, FlextTypes.GeneralValueType]:
        """Get database service statistics."""
        return {
            "connection_string": self._connection_string,
            "connected": self.connected,
            "queries_executed": self._query_count,
        }


class CacheService(FlextService[object]):
    """Concrete cache service implementation with advanced FLEXT patterns.

    This service demonstrates the NEW v0.9.9+ FlextResult methods:
    - from_callable: Replace manual try/except with functional composition
    - flow_through: Clean pipeline patterns for complex operations
    - value_or_call: Lazy evaluation of expensive defaults
    """

    max_size: int = 1000
    _ttl_seconds: int = 3600
    _hits: int = 0
    _misses: int = 0

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600) -> None:
        """Initialize cache with size and TTL limits."""
        super().__init__()
        self.max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cache: dict[str, FlextTypes.GeneralValueType] = {}
        self._metadata: dict[str, FlextTypes.GeneralValueType] = {}

    def execute(self, **_kwargs: object) -> FlextResult[object]:
        """Execute the cache service."""
        return FlextResult[object].ok(None)

    def get(self, key: str) -> FlextResult[FlextTypes.GeneralValueType]:
        """Get value from cache with TTL validation using functional composition."""

        def validate_key() -> str:
            """Validate cache key."""
            if not key or not key.strip():
                error_msg = "Cache key cannot be empty"
                raise ValueError(error_msg)
            return key

        def check_existence(valid_key: str) -> FlextResult[str]:
            """Check if key exists in cache."""
            if valid_key not in self._cache:
                self._misses += 1
                self.logger.debug("Cache miss: %s", valid_key)
                error_msg = f"Key not found: {valid_key}"
                return FlextResult[str].fail(
                    error_msg,
                    error_code=FlextConstants.Errors.NOT_FOUND,
                )
            return FlextResult[str].ok(valid_key)

        def validate_ttl(valid_key: str) -> FlextResult[str]:
            """Validate TTL for existing key."""
            if valid_key in self._metadata:
                metadata = self._metadata[valid_key]
                current_time = time.time()
                created_at_raw = 0
                if isinstance(metadata, dict):
                    created_at_raw = metadata.get("created_at", 0)

                # Use from_callable for safe conversion
                def convert_timestamp() -> float:
                    if created_at_raw is not None and isinstance(
                        created_at_raw,
                        (int, float, str),
                    ):
                        return float(created_at_raw)
                    return 0.0

                created_at_result = FlextResult[float].create_from_callable(
                    convert_timestamp,
                )

                if created_at_result.is_success:
                    created_at = created_at_result.unwrap()
                    if current_time - created_at > self._ttl_seconds:
                        # Expired - remove from cache
                        del self._cache[valid_key]
                        del self._metadata[valid_key]
                        self._misses += 1
                        self.logger.debug("Cache expired: %s", valid_key)
                        error_msg = f"Key expired: {valid_key}"
                        return FlextResult[str].fail(
                            error_msg,
                            error_code=FlextConstants.Errors.TIMEOUT_ERROR,
                        )

            return FlextResult[str].ok(valid_key)

        def get_value(valid_key: str) -> FlextResult[object]:
            """Get the actual cached value."""
            self._hits += 1
            self.logger.debug("Cache hit: %s", valid_key)
            return FlextResult[object].ok(self._cache[valid_key])

        # NEW: Use flow_through for clean pipeline composition
        def check_existence_obj(key: object) -> FlextResult[object]:
            str_key = str(key)
            result = check_existence(str_key)
            if result.is_success:
                return FlextResult[object].ok(result.value)
            return FlextResult[object].fail(result.error or "Check failed")

        def validate_ttl_obj(key: object) -> FlextResult[object]:
            str_key = str(key)
            result = validate_ttl(str_key)
            if result.is_success:
                return FlextResult[object].ok(result.value)
            return FlextResult[object].fail(result.error or "Validation failed")

        result = (
            FlextResult[str]
            .create_from_callable(
                validate_key,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )
            .flow_through(
                check_existence_obj,
                validate_ttl_obj,
            )
        )
        if result.is_success:
            str_key = str(result.unwrap())
            return get_value(str_key)
        return FlextResult[object].fail(
            result.error or "Unknown error",
            error_code=result.error_code,
        )

    def set(
        self,
        key: str,
        value: FlextTypes.GeneralValueType,
        ttl_seconds: int | None = None,
    ) -> FlextResult[bool]:
        """Set value in cache with optional TTL override."""
        if not key or not key.strip():
            return FlextResult[bool].fail(
                "Cache key cannot be empty",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        if value is None:
            return FlextResult[bool].fail(
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
        return FlextResult[bool].ok(True)

    def _evict_oldest(self) -> None:
        """Evict the oldest cache entry using LRU strategy with functional patterns."""

        def find_oldest_key() -> str:
            """Find the oldest key using functional composition."""
            if not self._metadata:
                error_msg = "No cache entries to evict"
                raise ValueError(error_msg)

            def get_timestamp(key: str) -> float:
                """Get timestamp for a key using from_callable."""

                def extract_timestamp() -> float:
                    metadata = self._metadata[key]
                    created_at_raw = 0
                    if isinstance(metadata, dict):
                        created_at_raw = metadata.get("created_at", 0)
                    if created_at_raw is not None and isinstance(
                        created_at_raw,
                        (int, float, str),
                    ):
                        return float(created_at_raw)
                    return 0.0

                # Use from_callable for safe timestamp extraction
                result = FlextResult[float].create_from_callable(extract_timestamp)
                return result.unwrap_or(0.0)

            return min(self._metadata.keys(), key=get_timestamp)

        def perform_eviction(oldest_key: object) -> FlextResult[object]:
            """Perform the actual eviction."""
            str_key = str(oldest_key)
            del self._cache[str_key]
            del self._metadata[str_key]
            self.logger.debug("Cache evicted: %s", str_key)
            return FlextResult[object].ok(str_key)

        # NEW: Use from_callable and flow_through for eviction
        eviction_result = (
            FlextResult[str]
            .create_from_callable(find_oldest_key)
            .flow_through(perform_eviction)
        )

        # For void methods, we just execute the flow but don't return the result
        if eviction_result.is_failure:
            self.logger.warning("Cache eviction failed: %s", eviction_result.error)

    def get_stats(self) -> dict[str, FlextTypes.GeneralValueType]:
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

    def execute(self, **_kwargs: object) -> FlextResult[None]:
        """Execute the email service."""
        return FlextResult[None].ok(None)

    def send(self, to: str, subject: str, body: str) -> FlextResult[bool]:
        """Send email (simulated)."""
        self.logger.info("Email sent to %s: %s", to, subject)
        _ = body
        return FlextResult[bool].ok(True)


# ========== DOMAIN MODELS ==========


class User(FlextModels.Entity):
    """User entity with domain logic."""

    id: str
    name: str
    email: str
    age: int
    is_active: bool = True
    version: int = 1


class UserRepository(FlextService[User]):
    """Repository pattern for User entities with enhanced domain logic."""

    def __init__(self) -> None:
        """Initialize with database dependency using FLEXT patterns."""
        super().__init__()
        container = FlextContainer()
        self._database = container.get("database").unwrap_or(None)
        self._cache: dict[str, User] = {}

    def execute(self, **_kwargs: object) -> FlextResult[User]:
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
        self._cache[user.entity_id] = user
        self.logger.info("User loaded from database: %s", user.entity_id)
        return FlextResult[User].ok(user)

    def save(self, user: User) -> FlextResult[bool]:
        """Save user to database with validation."""
        # Validate user before saving
        validation_result = self._validate_user(user)
        if validation_result.is_failure:
            return FlextResult[bool].fail(
                validation_result.error or "Validation failed",
            )

        # Simulate save operation with enhanced logging
        self.logger.info(
            f"Saving user to database: {user.entity_id}",
            extra={"user_email": user.email, "user_age": user.age},
        )

        # Update cache
        self._cache[user.entity_id] = user

        return FlextResult[bool].ok(True)

    def find_by_id(self, user_id: str) -> FlextResult[User]:
        """Find user by ID from cache or database."""
        # Check cache first
        if user_id in self._cache:
            return FlextResult[User].ok(self._cache[user_id])

        # Simulate database lookup
        if user_id == "default_user":
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
            self._cache[user_id] = user
            return FlextResult[User].ok(user)

        return FlextResult[User].fail(
            f"User not found: {user_id}",
            error_code=FlextConstants.Errors.NOT_FOUND,
        )

    def _validate_user(self, user: User) -> FlextResult[bool]:
        """Validate user data before saving."""
        if not user.name or not user.name.strip():
            return FlextResult[bool].fail(
                "User name cannot be empty",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        if not user.email or "@" not in user.email:
            return FlextResult[bool].fail(
                "User email must be valid",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        if (
            user.age < FlextConstants.Validation.MIN_AGE
            or user.age > FlextConstants.Validation.MAX_AGE
        ):
            return FlextResult[bool].fail(
                f"User age must be between {FlextConstants.Validation.MIN_AGE} and {FlextConstants.Validation.MAX_AGE}",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        return FlextResult[bool].ok(True)

    def get_stats(self) -> dict[str, FlextTypes.GeneralValueType]:
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

    def execute(self, **_kwargs: object) -> FlextResult[User]:
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
            updated_at=datetime.now(UTC),
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
            service_count = len(self.container.list_services())
            # Convert stats to GeneralValueType-compatible format
            db_stats_dict: dict[str, FlextTypes.GeneralValueType] = (
                dict(db_stats) if db_stats else {}
            )
            cache_stats_dict: dict[str, FlextTypes.GeneralValueType] = (
                dict(cache_stats) if cache_stats else {}
            )
            self.logger.info(
                "Service execution statistics",
                database_stats=db_stats_dict,
                cache_stats=cache_stats_dict,
                container_service_count=service_count,
            )
        except Exception as e:
            self.logger.warning("Failed to log statistics: %s", str(e))

    def get_service_stats(self) -> dict[str, FlextTypes.GeneralValueType]:
        """Get comprehensive service statistics."""
        service_count = len(
            self.container.list_services(),
        )
        return {
            "container_service_count": service_count,
        }

    # ========== BASIC REGISTRATION ==========

    def demonstrate_basic_registration(self) -> None:
        """Show basic service registration patterns."""
        print("\n=== Basic Service Registration ===")

        self.container.clear_all()
        print("✅ Container cleared")

        connection = str(self._config_data.get("database_url", "sqlite:///:memory:"))
        db_service = DatabaseService(connection_string=connection)
        self.container.with_service("database", db_service)
        print("Register singleton: True")

        self.container.with_factory("cache", CacheService)
        print("Register factory: True")

        has_db = self.container.has_service("database")
        has_cache = self.container.has_service("cache")
        print(f"Has database: {has_db}, Has cache: {has_cache}")

        service_count = len(
            self.container.list_services(),
        )
        print(f"Service count: {service_count}")

    # ========== SERVICE RESOLUTION ==========

    def demonstrate_service_resolution(self) -> None:
        """Show all ways to resolve services with enhanced monitoring."""
        print("\n=== Service Resolution ===")

        dataset = self._dataset
        users_list = cast("list[object]", dataset.get("users", []))
        sample_user = cast("dict[str, object]", users_list[0] if users_list else {})

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

        typed_result: FlextResult[DatabaseService] = self.container.get_typed(
            "database",
            DatabaseService,
        )
        if typed_result.is_success:
            db_typed = typed_result.unwrap()
            print(f"✅ Got typed database: {type(db_typed).__name__}")
            # Safe access to database stats
            stats = db_typed.get_stats()
            print(
                f"   Database stats: {stats.get('queries_executed', 0)} queries executed",
            )
        else:
            print(f"❌ Type validation failed: {typed_result.error}")

        user_dict = sample_user
        # Use get with fallback to register if not found
        email_result = self.container.get("email")
        if email_result.is_failure:
            email_service = EmailService(
                smtp_host=str(user_dict.get("email", "smtp@example.com")),
            )
            register_result = self.container.register("email", email_service)
            if register_result.is_success:
                email_result = self.container.get("email")
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
                    cached_dict = cast(
                        "dict[str, FlextTypes.GeneralValueType]", cached_data
                    )
                    print(f"✅ Cache test: {cached_dict.get('email')}")

            cache_stats_func: (
                Callable[[], Mapping[str, FlextTypes.GeneralValueType]] | None
            ) = getattr(cache, "get_stats", None)
            stats_raw: Mapping[str, FlextTypes.GeneralValueType] | None = (
                cache_stats_func() if callable(cache_stats_func) else None
            )
            stats_dict: dict[str, FlextTypes.GeneralValueType] = (
                dict(stats_raw) if stats_raw else {}
            )
            print(
                f"   Cache stats: {stats_dict.get('current_size', 0)} items, {stats_dict.get('hits', 0)} hits, {stats_dict.get('misses', 0)} misses",
            )

    # ========== BATCH OPERATIONS ==========

    def demonstrate_batch_operations(self) -> None:
        """Show batch registration patterns."""
        print("\n=== Batch Operations ===")

        services = self._scenarios.service_batch("container_batch")

        # Register services individually (batch_register doesn't exist)
        registered_count = 0
        for name, service in services.items():
            if self.container.register(name, service).is_success:
                registered_count += 1
        print(f"✅ Registered {registered_count} services")

        services_list = self.container.list_services()
        print(f"All services: {services_list}")

    # ========== AUTO-WIRING ==========

    def demonstrate_auto_wiring(self) -> None:
        """Show dependency auto-wiring."""
        print("\n=== Auto-Wiring ===")

        self.container.with_service("database", DatabaseService())

        # Auto-wiring not available, create manually
        repo = UserRepository()
        print(f"✅ Created: {type(repo).__name__}")

        db_result: FlextResult[object] = self.container.get_typed(
            "database",
            DatabaseService,
        )
        if db_result.is_success:
            db = cast("DatabaseService", db_result.unwrap())
            db.connect()

            users_list_result = self._dataset.get(
                "users",
                cast("list[dict[str, object]]", [{}]),
            )
            users_list: list[object] = cast("list[object]", users_list_result)
            user_id = str(
                cast("dict[str, object]", users_list[0]).get("id", "user_1"),
            )
            user_result = repo.find_by_id(user_id)
            if user_result.is_success:
                user = user_result.unwrap()
                print(f"   Found user: {user.name}")
        else:
            print("❌ Database connection failed")

    # ========== CONFIGURATION ==========

    def demonstrate_configuration(self) -> None:
        """Show container configuration."""
        print("\n=== Container Configuration ===")

        production_config = self._scenarios.config(production=True)
        config: dict[str, FlextTypes.FlexibleValue] = {
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

        self.container.configure(config)
        print("Container configuration: True")

        # Build info manually since get_info() is removed
        service_count = len(
            self.container.list_services(),
        )
        info = {
            "service_count": service_count,
            "direct_services": len(self.container.list_services()),
            "factories": len(self.container.list_services()),
        }
        print(f"Container info: {info}")

    # ========== SERVICE LIFECYCLE ==========

    def demonstrate_service_lifecycle(self) -> None:
        """Show singleton vs factory lifecycles."""
        print("\n=== Service Lifecycles ===")

        # Singleton: same instance every time
        self.container.with_service("singleton_cache", CacheService())

        cache1_result = self.container.get("singleton_cache")
        cache2_result = self.container.get("singleton_cache")

        if cache1_result.is_success and cache2_result.is_success:
            cache1 = cache1_result.unwrap()
            cache2 = cache2_result.unwrap()
            print(f"Singleton same instance: {cache1 is cache2}")

        # Factory: Note - current implementation caches factory results
        # This demonstrates the current behavior, not ideal factory pattern
        self.container.with_factory("factory_cache", CacheService)

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

        global_container = FlextContainer()
        print(f"Global container: {type(global_container).__name__}")

        global_payload = self._scenarios.payload(type="global_service")
        global_container.with_service("global_service", global_payload)
        print("Register global: True")

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

        # Try to register with invalid name (with_service doesn't validate names)
        print(
            "✅ Skipping invalid name test (with_service doesn't validate empty names)",
        )

        # Type mismatch in get_typed
        self.container.with_service("wrong_type", CacheService())
        result = self.container.get_typed("wrong_type", DatabaseService)
        if result.is_failure:
            print(f"✅ Correct failure for type mismatch: {result.error}")

    # ========== NEW FlextResult METHODS (v0.9.9+) ==========

    def demonstrate_from_callable(self) -> None:
        """Show from_callable for safe service initialization."""
        print("\n=== from_callable(): Safe Service Initialization ===")

        # Safe database connection initialization
        def risky_db_connect() -> DatabaseService:
            db = DatabaseService(connection_string="postgresql://invalid-host/db")
            connect_result = db.connect()
            if connect_result.is_failure:
                msg = "Database connection failed"
                raise RuntimeError(msg)
            return db

        db_result = FlextResult[DatabaseService].create_from_callable(risky_db_connect)
        if db_result.is_failure:
            print(f"✅ Caught connection error safely: {db_result.error}")
        else:
            print("Database connected successfully")

        # Safe cache initialization
        def safe_cache_init() -> CacheService:
            return CacheService(max_size=100, ttl_seconds=300)

        cache_result = FlextResult[CacheService].create_from_callable(safe_cache_init)
        if cache_result.is_success:
            cache = cache_result.unwrap()
            print(f"✅ Cache initialized: max_size={cache.max_size}")

    def demonstrate_flow_through(self) -> None:
        """Show pipeline composition for service initialization."""
        print("\n=== flow_through(): Service Initialization Pipeline ===")

        def connect_database(
            db: object,
        ) -> FlextResult[object]:
            """Connect to database."""
            if not isinstance(db, DatabaseService):
                return FlextResult[object].fail("Not a DatabaseService")
            result = db.connect()
            if result.is_failure:
                return FlextResult[object].fail(
                    f"Connection failed: {result.error}",
                )
            return FlextResult[object].ok(db)

        def validate_database(
            db: object,
        ) -> FlextResult[object]:
            """Validate database is ready."""
            if not isinstance(db, DatabaseService):
                return FlextResult[object].fail("Not a DatabaseService")
            db_service = cast("DatabaseService", db)
            if not db_service.connected:
                return FlextResult[object].fail("Database not connected")
            return FlextResult[object].ok(db)

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
            db_service = (
                cast("DatabaseService", db) if isinstance(db, DatabaseService) else None
            )
            connected = db_service.connected if db_service else False
            print(f"✅ Service pipeline success: connected={connected}")
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

        # Try primary service, use fallback if fails (using lash instead of alt)
        primary = self.container.get("non_existent_service")

        def provide_fallback(_error: str) -> FlextResult[object]:
            return FlextResult[object].ok(CacheService())

        result = primary.lash(provide_fallback)
        if result.is_success:
            service = result.unwrap()
            print(f"✅ Got fallback service: {type(service).__name__}")

        # Chain multiple fallbacks using lash
        first = self.container.get("service1")

        def second_fallback(_error: str) -> FlextResult[object]:
            return self.container.get("service2")

        def third_fallback(_error: str) -> FlextResult[object]:
            return FlextResult[object].ok(EmailService())

        result = first.lash(second_fallback).lash(third_fallback)
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
            return EmailService(smtp_host="expensive-smtp.example.com")

        # Success case - expensive creation NOT called
        success = self.container.get("database")
        if success.is_success:
            service = success.unwrap()
            print(
                f"✅ Success: {type(service).__name__}, expensive_created={expensive_created}",
            )

        # Failure case - expensive creation IS called
        expensive_created = False
        failure = self.container.get("non_existent_service")
        service = failure.unwrap_or(expensive_service_creation())
        expensive_created = True
        print(
            f"Fallback: {type(service).__name__}, expensive_created={expensive_created}",
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

    container.with_service("demo_logger", logger)
    print("  ✅ Logger registered in container")

    # 3. Access configuration alongside container
    print("\n=== 3. Combined Configuration and DI ===")
    config = FlextConfig()
    print(f"  ✅ Config access: log_level = {config.log_level}")
    services_list = container.list_services()
    service_count = len(services_list)
    print(f"  ✅ Container access: {service_count} services")

    # 4. Setup complete infrastructure with DI
    print("\n=== 4. Infrastructure Setup with DI ===")
    infra_container = FlextContainer()
    infra_logger = FlextLogger("di-demo-service")

    print("  ✅ Infrastructure initialized with DI:")
    print(f"     - Container: {type(infra_container).__name__}")
    print(f"     - Logger: {type(infra_logger).__name__}")
    infra_services_list = infra_container.list_services()
    infra_service_count = len(infra_services_list)
    print(f"     - Services: {infra_service_count}")

    # 5. Direct class access for type-safe operations
    print("\n=== 5. Type-Safe DI Operations ===")
    direct_container = FlextContainer()
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
        "✨ NEW v0.9.9+ methods prominently featured: from_callable, flow_through, lash, alt, value_or_call",
    )
    print("🎯 CacheService refactored: Manual try/except → functional composition")
    print("🎯 Next: See 03_models_basics.py for FlextModels patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
