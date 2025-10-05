"""Layer 8: Domain models aligned with the FLEXT 1.0.0 modernization charter.

This module provides the complete FlextModels namespace with Entity, Value,
AggregateRoot, and all configuration models for the FLEXT ecosystem. Use
FlextModels for all domain modeling and data validation.

Dependency Layer: 8 (Domain Foundation)
Dependencies: FlextConstants, FlextTypes, FlextExceptions, FlextResult,
              FlextConfig, FlextUtilities, FlextLoggings, FlextMixins
Used by: All FlextCore application and infrastructure modules

Entities, value objects, and aggregates mirror the design captured in
``README.md`` and ``docs/architecture.md`` so downstream packages share a
consistent DDD foundation during the rollout.

Usage:
    ```python
    from flext_core.result import FlextResult
    from flext_core.models import FlextModels


    class User(FlextModels.Entity):
        name: str
        email: str

        @override
        def validate(self: object) -> FlextResult[None]:
            if "@" not in self.email:
                return FlextResult[None].fail("Invalid email")
            return FlextResult[None].ok(None)
    ```

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

# ruff: disable=PLC2701
from __future__ import annotations

import base64
import inspect
import re
import time as time_module
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, time
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import (
    Annotated,
    ClassVar,
    Literal,
    Self,
    cast,
    override,
)
from urllib.parse import ParseResult, urlencode, urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.main import IncEx
from pydantic_settings import BaseSettings, SettingsConfigDict

# Layer 1 - Foundation
from flext_core.constants import FlextConstants

# Layer 2 - Early Foundation
from flext_core.exceptions import FlextExceptions
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes

# Module-level configuration instance for runtime defaults (lazy loaded)
_config: FlextTypes.Any = None


def _get_config() -> FlextTypes.Any:
    """Lazy load configuration to avoid import cycles."""
    global _config
    if _config is None:
        # Lazy import to break circular dependency
        from flext_core.config import FlextConfig

        _config = FlextConfig()
    return _config


class _BaseConfigDict:
    """Shared Pydantic 2.11 ConfigDict patterns for FlextModels classes.

    Private implementation detail - do not use directly.
    Access through FlextModels nested classes that use these configs.
    """

    ARBITRARY: ConfigDict = ConfigDict(
        # Pydantic 2.11 validation features
        validate_assignment=True,
        validate_return=True,
        validate_default=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
        # JSON serialization features
        ser_json_timedelta="iso8601",
        ser_json_bytes="base64",
        # Alias and naming features
        serialize_by_alias=True,
        populate_by_name=True,
        # String processing
        str_strip_whitespace=True,
        str_to_lower=False,
        str_to_upper=False,
        # Performance optimizations
        defer_build=False,
        # Type coercion
        coerce_numbers_to_str=False,
    )

    STRICT: ConfigDict = ConfigDict(
        # Pydantic 2.11 validation features
        validate_assignment=True,
        validate_return=True,
        validate_default=True,
        use_enum_values=True,
        arbitrary_types_allowed=True,
        extra="forbid",
        # JSON serialization features
        ser_json_timedelta="iso8601",
        ser_json_bytes="base64",
        # Alias and naming features
        serialize_by_alias=True,
        populate_by_name=True,
        # String processing
        str_strip_whitespace=True,
        str_to_lower=False,
        str_to_upper=False,
        # Performance optimizations
        defer_build=False,
        # Type coercion
        coerce_numbers_to_str=False,
    )

    FROZEN: ConfigDict = ConfigDict(
        frozen=True,
        extra="forbid",
        use_enum_values=True,
    )


class FlextModels:
    """Domain-Driven Design patterns for FLEXT ecosystem modeling.

    FlextModels provides comprehensive DDD base classes for domain
    modeling throughout the FLEXT ecosystem. Includes Entity, Value
    Object, Aggregate Root patterns with built-in validation and
    event sourcing support. Used across all 32+ dependent projects.

    **Function**: DDD pattern implementations with Pydantic 2
        - Entity base with identity and lifecycle management
        - Value Object base for immutable value types
        - Aggregate Root for consistency boundaries
        - Domain event management and event sourcing
        - CQRS patterns (Command, Query, DomainEvent)
        - Repository and specification patterns
        - Saga pattern for distributed transactions
        - Validation utilities with FlextResult integration
        - Pagination support for queries
        - Metadata and error detail models
        - Configuration models with environment support
        - Request/Response models for APIs

    **Uses**: Pydantic 2 for validation and serialization
        - BaseModel for all domain models with type safety
        - Field validators for business rule enforcement
        - model_config for Pydantic settings and behavior
        - ConfigDict for model configuration
        - computed_field for derived properties
        - FlextResult[T] for operation results
        - FlextConfig for configuration integration
        - FlextLogger for domain event logging
        - FlextConstants for validation limits
        - FlextTypes for type definitions

    **How to use**: Implement domain models with DDD patterns
        ```python
        from flext_core import FlextModels, FlextResult


        # Example 1: Value Object (immutable, compared by value)
        class Email(FlextModels.Value):
            address: str

            def validate(self) -> FlextResult[None]:
                if "@" not in self.address:
                    return FlextResult[None].fail("Invalid email")
                return FlextResult[None].ok(None)


        # Example 2: Entity (has identity and lifecycle)
        class User(FlextModels.Entity):
            name: str
            email: Email
            is_active: bool = False

            def activate(self) -> FlextResult[None]:
                if self.is_active:
                    return FlextResult[None].fail("Already active")
                self.is_active = True
                self.add_domain_event("UserActivated", {"user_id": self.id})
                return FlextResult[None].ok(None)


        # Example 3: Aggregate Root (consistency boundary)
        class Account(FlextModels.AggregateRoot):
            owner: User
            balance: Decimal

            def withdraw(self, amount: Decimal) -> FlextResult[None]:
                if amount > self.balance:
                    return FlextResult[None].fail("Insufficient funds")
                self.balance -= amount
                self.add_domain_event("MoneyWithdrawn", {"amount": str(amount)})
                return FlextResult[None].ok(None)


        # Example 4: CQRS Command pattern
        class CreateUserCommand(FlextModels.Command):
            name: str
            email: str


        # Example 5: CQRS Query pattern with pagination
        class GetUsersQuery(FlextModels.Query):
            pagination: dict = Field(default_factory=dict)


        # Example 6: Domain Event for event sourcing
        class UserCreatedEvent(FlextModels.DomainEvent):
            user_id: str
            timestamp: datetime
        ```

        - [ ] Add domain event versioning for evolution
        - [ ] Implement event store patterns with snapshots
        - [ ] Support aggregate snapshots for performance
        - [ ] Add saga orchestration patterns
        - [ ] Enhance validation DSL for complex rules
        - [ ] Implement optimistic locking for concurrency
        - [ ] Add domain event replay for debugging
        - [ ] Support event sourcing with CQRS projection
        - [ ] Implement specification pattern composition
        - [ ] Add repository pattern with unit of work

    Attributes:
        Entity: Base class for entities with identity.
        Value: Base class for immutable value objects.
        AggregateRoot: Base class for aggregate roots.
        Command: Base class for CQRS commands.
        Query: Base class for CQRS queries.
        DomainEvent: Base class for domain events.
        Pagination: Pagination support for queries.
        Validation: Validation utilities and helpers.
        Config: Configuration models.
        Metadata: Metadata and error detail models.

    Note:
        All models use Pydantic 2.11+ for validation. Entity
        instances have identity (id field). Value objects are
        immutable and compared by value. Aggregate roots manage
        consistency boundaries. Use FlextResult for all operations.

    Warning:
        Value objects must be immutable - never modify after
        creation. Aggregate roots enforce invariants - always
        validate state changes. Domain events are immutable once
        created. Never bypass validation in domain models.

    Example:
        Complete domain modeling with DDD patterns:

        >>> class Money(FlextModels.Value):
        ...     amount: Decimal
        ...     currency: str
        >>> class Account(FlextModels.AggregateRoot):
        ...     balance: Money
        >>> account = Account(balance=Money(amount=100, currency="USD"))
        >>> print(account.balance.amount)
        100

    See Also:
        FlextResult: For railway-oriented error handling.
        FlextContainer: For dependency injection patterns.
        FlextBus: For CQRS command/query dispatching.

    """

    # =========================================================================
    # REUSABLE FIELD TYPES - Common patterns (Pydantic 2.11)
    # =========================================================================
    # These eliminate 1000+ lines of repetitive Field definitions

    # UUID fields
    UuidField = Annotated[str, Field(default_factory=lambda: str(uuid.uuid4()))]

    # Timestamp fields
    TimestampField = Annotated[
        datetime, Field(default_factory=lambda: datetime.now(UTC))
    ]

    # Config-driven fields (pull from global FlextConfig)
    TimeoutField = Annotated[
        int,
        Field(default_factory=lambda: int(_get_config().timeout_seconds)),
    ]
    RetryField = Annotated[
        int,
        Field(default_factory=lambda: _get_config().max_retry_attempts),
    ]
    MaxWorkersField = Annotated[
        int, Field(default_factory=lambda: _get_config().max_workers)
    ]

    # Field-level serialization optimizations using Annotated types
    JsonStr = Annotated[str, Field(json_schema_extra={"format": "json"})]
    Base64Bytes = Annotated[bytes, Field(json_schema_extra={"format": "base64"})]
    IsoDateTime = Annotated[datetime, Field(json_schema_extra={"format": "date-time"})]
    IsoDate = Annotated[date, Field(json_schema_extra={"format": "date"})]
    IsoTime = Annotated[time, Field(json_schema_extra={"format": "time"})]

    # =========================================================================
    # BEHAVIOR MIXINS - Reusable model behaviors (Pydantic 2.11 pattern)
    # =========================================================================

    class BaseModel(BaseModel):
        """Base model class for all FLEXT domain models.

        Extends Pydantic's BaseModel with common FLEXT ecosystem patterns.
        All domain models should inherit from this base class for consistency.

        Provides:
        - Standard configuration for all models
        - Common validation patterns
        - Integration with FLEXT ecosystem
        """

        model_config = ConfigDict(
            validate_assignment=True,
            validate_default=True,
            use_enum_values=True,
            str_strip_whitespace=True,
            from_attributes=True,
        )

    class BaseConfig(BaseSettings):
        """Base configuration class for all FLEXT configuration models.

        Extends BaseModel with configuration-specific patterns for FLEXT
        ecosystem configuration classes. Used for settings, configs, and
        environment-based configuration models.

        Provides:
        - Environment variable integration
        - Configuration validation
        - Settings management patterns
        """

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )

    class IdentifiableMixin(BaseModel):
        """Mixin for models with unique identifiers.

        Provides the `id` field using UuidField pattern with explicit default.
        Used by Entity, Command, DomainEvent, Saga, and other identifiable models.
        """

        id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    class TimestampableMixin(BaseModel):
        """Mixin for models with creation and update timestamps.

        Provides `created_at` and `updated_at` fields with automatic timestamp management.
        Used by Entity, TimestampedModel, and models requiring audit trails.

        Pydantic 2 features:
        - field_serializer for ISO 8601 timestamp serialization
        - Automatic timestamp management
        """

        created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        updated_at: datetime | None = None

        @field_serializer("created_at", "updated_at", when_used="json")
        def serialize_timestamps(self, value: datetime | None) -> str | None:
            """Serialize timestamps to ISO 8601 format for JSON."""
            return value.isoformat() if value else None

        def update_timestamp(self) -> None:
            """Update the updated_at timestamp to current UTC time."""
            self.updated_at = datetime.now(UTC)

    class TimeoutableMixin(BaseModel):
        """Mixin for models with timeout configuration.

        Provides `timeout_seconds` field with default from FlextConstants.
        Can be overridden at runtime using FlextConfig for dynamic configuration.
        Used by Repository, Queue, Bus, Circuit, and other timeout-aware models.

        Pydantic 2 Settings integration:
        - Static defaults from FlextConstants (compile-time)
        - Runtime override via FlextConfig("timeout_seconds") (runtime)
        - Type-safe field validation with Annotated

        Example:
            >>> from flext_core import FlextConfig
            >>> # Static default (compile-time constant)
            >>> mixin = TimeoutableMixin()  # Uses FlextConstants default
            >>> # Runtime configuration override
            >>> config = FlextConfig()
            >>> timeout = config("timeout_seconds")  # Uses FlextConfig callable
            >>> mixin_configured = TimeoutableMixin(timeout_seconds=timeout)

        """

        timeout_seconds: Annotated[
            int,
            Field(default_factory=lambda: int(_get_config().timeout_seconds)),
        ]

        @classmethod
        def from_config(cls) -> Self:
            """Create TimeoutableMixin with timeout from runtime configuration.

            Uses singleton FlextConfig with Pydantic 2 Settings callable pattern.
            No need to pass config - it's always a singleton.

            Returns:
                TimeoutableMixin with timeout from configuration

            Example:
                >>> # Simple - no config parameter needed (singleton)
                >>> mixin = TimeoutableMixin.from_config()
                >>> # Equivalent to:
                >>> config = FlextConfig()
                >>> timeout = config("timeout_seconds")
                >>> mixin = TimeoutableMixin(timeout_seconds=timeout)

            """
            # Use lazy-loaded config to avoid circular imports
            config = _get_config()

            # Demonstrate FlextConfig("parameter") callable usage
            timeout_value = config("timeout_seconds")

            if not isinstance(timeout_value, int):
                # Type guard for runtime safety
                timeout_value = int(str(timeout_value))

            return cls(timeout_seconds=timeout_value)

    class RetryableMixin(BaseModel):
        """Mixin for models with retry configuration.

        Provides `max_retry_attempts` and `retry_policy` fields for retry behavior.
        Used by Repository, HandlerExecutionConfig, and other retry-aware models.

        Pydantic 2 Settings integration:
        - Static defaults from FlextConstants (compile-time)
        - Runtime override via FlextConfig("max_retry_attempts") (runtime)
        - Type-safe field validation with Annotated

        Example:
            >>> from flext_core import FlextConfig
            >>> # Static default (compile-time constant)
            >>> mixin = RetryableMixin()  # Uses FlextConstants default
            >>> # Runtime configuration override
            >>> config = FlextConfig()
            >>> max_retries = config("max_retry_attempts")  # FlextConfig callable
            >>> mixin_configured = RetryableMixin(max_retry_attempts=max_retries)

        """

        max_retry_attempts: Annotated[
            int,
            Field(default_factory=lambda: _get_config().max_retry_attempts),
        ]
        retry_policy: FlextTypes.Reliability.RetryPolicy = Field(default_factory=dict)

        @classmethod
        def from_config(cls) -> Self:
            """Create RetryableMixin with retry settings from runtime configuration.

            Uses singleton FlextConfig with Pydantic 2 Settings callable pattern.
            No need to pass config - it's always a singleton.

            Returns:
                RetryableMixin with retry settings from configuration

            Example:
                >>> # Simple - no config parameter needed (singleton)
                >>> mixin = RetryableMixin.from_config()
                >>> # Equivalent to:
                >>> config = FlextConfig()
                >>> max_retries = config("max_retry_attempts")
                >>> mixin = RetryableMixin(max_retry_attempts=max_retries)

            """
            # Use lazy-loaded config to avoid circular imports
            config = _get_config()

            # Demonstrate FlextConfig("parameter") callable usage
            max_retries = config("max_retry_attempts")

            if not isinstance(max_retries, int):
                # Type guard for runtime safety
                max_retries = int(str(max_retries))

            return cls(max_retry_attempts=max_retries)

    class VersionableMixin(BaseModel):
        """Mixin for models with versioning support.

        Provides `version` field and increment_version method for optimistic locking.
        Used by Entity and models requiring version tracking.
        """

        version: int = Field(
            default=FlextConstants.Performance.DEFAULT_VERSION,
            ge=FlextConstants.Performance.MIN_VERSION,
        )

        def increment_version(self) -> None:
            """Increment the version number for optimistic locking."""
            self.version += 1

    # =========================================================================
    # BASE MODEL CLASSES - Using shared ConfigDict patterns
    # =========================================================================

    # Enhanced validation using FlextUtilities and FlextConstants
    # Base model classes for configuration consolidation with Pydantic 2.11 features
    class ArbitraryTypesModel(BaseModel):
        """Most common pattern: validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True.

        Enhanced with comprehensive Pydantic 2.11 features for better validation and serialization.
        Used by 17+ models in the codebase. This is the recommended base class for all FLEXT models.
        """

        model_config = _BaseConfigDict.ARBITRARY

    # =============================================================================
    # METACLASSES - Foundation metaclasses for advanced model patterns
    # =============================================================================

    class ServiceMeta(type):
        """Combined metaclass for domain services supporting Pydantic, ABC, and Protocol.

        Resolves metaclass conflict when inheriting from Pydantic BaseModel (ModelMetaclass),
        ABC (ABCMeta), and Protocol (ProtocolMeta). Since ProtocolMeta inherits from ABCMeta,
        we only need to combine ModelMetaclass and ProtocolMeta. This metaclass enables:

        - Pydantic validation and serialization
        - ABC abstract methods and protocols

        **Usage in FlextService**:
            ```python
            class FlextService[T](
                FlextModels.ArbitraryTypesModel, ABC, metaclass=FlextModels.ServiceMeta
            ):
                pass
            ```

        **Foundation Pattern**: Used by FlextService and all domain service implementations
        across the FLEXT ecosystem.
        """

    class StrictArbitraryTypesModel(BaseModel):
        """Strict pattern: forbid extra fields, arbitrary types allowed.

        Used by domain service models requiring strict validation.
        Enhanced with comprehensive Pydantic 2.11 features.
        """

        model_config = _BaseConfigDict.STRICT

    class FrozenStrictModel(BaseModel):
        """Immutable pattern: frozen with extra fields forbidden.

        Used by value objects and configuration models.
        """

        model_config = _BaseConfigDict.FROZEN

    # =========================================================================
    # UTILITY TYPES - Centralized type definitions
    # =========================================================================

    class ModelDumpKwargs(ArbitraryTypesModel):
        """Type definition for model dump keyword arguments.

        This model defines the optional parameters that can be passed to
        Pydantic model dump methods, ensuring type safety for serialization operations.
        """

        include: set[str] | None = None
        exclude: set[str] | None = None
        by_alias: bool = False
        exclude_unset: bool = False
        exclude_defaults: bool = False
        exclude_none: bool = False
        round_trip: bool = False
        warnings: bool = True
        mode: str = "python"
        context: FlextTypes.Dict | None = None

    # Base model classes from DDD patterns
    class TimestampedModel(ArbitraryTypesModel, TimestampableMixin):
        """Base class for models with timestamp fields.

        Inherits timestamp functionality from TimestampableMixin.
        Provides created_at and updated_at fields with automatic management.
        """

    class Entity(TimestampedModel, IdentifiableMixin, VersionableMixin):
        """Base class for domain entities with identity.

        Combines TimestampedModel, IdentifiableMixin, and VersionableMixin to provide:
        - id: Unique identifier (from IdentifiableMixin)
        - created_at/updated_at: Timestamps (from TimestampedModel)
        - version: Optimistic locking (from VersionableMixin)
        - domain_events: Event sourcing support
        """

        domain_events: list[object] = Field(default_factory=list)

        @override
        def model_post_init(self, __context: object, /) -> None:
            """Post-initialization hook to set updated_at timestamp."""
            if self.updated_at is None:
                self.updated_at = datetime.now(UTC)

        @override
        def __eq__(self, other: object) -> bool:
            """Identity-based equality for entities."""
            if not isinstance(other, FlextModels.Entity):
                return False
            return self.id == other.id

        @override
        def __hash__(self) -> int:
            """Identity-based hash for entities."""
            return hash(self.id)

        def add_domain_event(self, event_name: str, data: FlextTypes.Dict) -> None:
            """Add a domain event to be dispatched.

            DomainEvent now uses IdentifiableMixin and TimestampableMixin,
            so id and created_at are auto-generated.
            """
            domain_event = FlextModels.DomainEvent(
                event_type=event_name,
                aggregate_id=self.id,
                data=data,
                # Removed occurred_at - auto-generated via TimestampableMixin
            )
            self.domain_events.append(domain_event)

            # Try to find and call event handler method
            event_type = data.get("event_type", "")
            handler_method_name = f"_apply_{str(event_type).lower()}"
            if hasattr(self, handler_method_name):
                try:
                    handler_method = getattr(self, handler_method_name)
                    handler_method(data)
                except Exception as e:
                    # Log exception but don't re-raise to maintain resilience
                    # Lazy import to avoid circular dependency (loggings.py -> models.py)
                    from flext_core.loggings import FlextLogger

                    logger = FlextLogger(__name__)
                    logger.warning(
                        f"Domain event handler {handler_method_name} failed for event {event_name}: {e}"
                    )

            # Increment version after adding domain event (from VersionableMixin)
            self.increment_version()
            self.update_timestamp()

        def clear_domain_events(self) -> FlextTypes.List:
            """Clear and return domain events."""
            events: FlextTypes.List = self.domain_events.copy()
            self.domain_events.clear()
            return events

    class Value(FrozenStrictModel):
        """Base class for value objects - immutable and compared by value."""

        @override
        def __eq__(self, other: object) -> bool:
            """Compare by value."""
            if not isinstance(other, self.__class__):
                return False
            if hasattr(self, "model_dump") and hasattr(other, "model_dump"):
                return self.model_dump() == other.model_dump()
            return False

        @override
        def __hash__(self) -> int:
            """Hash based on values for use in sets/FlextTypes.Dicts."""
            return hash(tuple(self.model_dump().items()))

        @classmethod
        def create(cls, *args: object, **kwargs: object) -> FlextResult[object]:
            """Create value object instance with validation, returns FlextResult."""
            try:
                # Handle single argument case for simple value objects
                if len(args) == 1 and not kwargs:
                    # Get the first field name for single-field value objects
                    field_names = list(cls.model_fields.keys())
                    if len(field_names) == 1:
                        kwargs[field_names[0]] = args[0]
                        args = ()

                instance = cls(*args, **kwargs)
                return FlextResult[object].ok(instance)
            except Exception as e:
                return FlextResult[object].fail(str(e))

    class AggregateRoot(Entity):
        """Base class for aggregate roots - consistency boundaries."""

        _invariants: ClassVar[list[Callable[[], bool]]] = []

        def check_invariants(self) -> None:
            """Check all business invariants."""
            for invariant in self._invariants:
                if not invariant():
                    msg = f"Invariant violated: {invariant.__name__}"
                    raise FlextExceptions.ValidationError(
                        message=msg,
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

        @override
        def model_post_init(self, __context: object, /) -> None:
            """Run after model initialization."""
            super().model_post_init(__context)
            self.check_invariants()

    class Command(StrictArbitraryTypesModel, IdentifiableMixin, TimestampableMixin):
        """Base class for CQRS commands with validation.

        Uses IdentifiableMixin for id and TimestampableMixin for created_at.
        """

        command_type: str = Field(
            default=FlextConstants.Cqrs.DEFAULT_COMMAND_TYPE,
            description="Command type identifier",
        )
        issuer_id: str | None = None

        @field_validator("command_type")
        @classmethod
        def validate_command(cls, v: str) -> str:
            """Auto-set command type from class name if empty."""
            if not v:
                return cls.__name__
            return v

    class DomainEvent(ArbitraryTypesModel, IdentifiableMixin, TimestampableMixin):
        """Base class for domain events.

        Uses IdentifiableMixin for id and TimestampableMixin for created_at.
        """

        event_type: str
        aggregate_id: str
        data: FlextTypes.Domain.EventPayload = Field(default_factory=dict)
        metadata: FlextTypes.Domain.EventMetadata = Field(default_factory=dict)

    class Saga(ArbitraryTypesModel, IdentifiableMixin):
        """Saga pattern for distributed transactions.

        Uses IdentifiableMixin for id.
        """

        steps: Annotated[list[FlextTypes.Dict], Field(default_factory=list)]
        current_step: int = Field(
            default=FlextConstants.Performance.DEFAULT_CURRENT_STEP,
            ge=FlextConstants.Performance.MIN_CURRENT_STEP,
        )
        status: Literal["pending", "running", "completed", "failed", "compensating"] = (
            "pending"
        )
        compensation_data: FlextTypes.Dict = Field(default_factory=dict)

    class Metadata(FrozenStrictModel):
        """Immutable metadata model."""

        created_by: str | None = None
        created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        modified_by: str | None = None
        modified_at: datetime | None = None
        tags: FlextTypes.StringList = Field(default_factory=list)
        attributes: FlextTypes.Dict = Field(default_factory=dict)

    class ErrorDetail(FrozenStrictModel):
        """Immutable error detail model."""

        code: str
        message: str
        field: str | None = None
        details: FlextTypes.Dict = Field(default_factory=dict)

    class ValidationResult(ArbitraryTypesModel):
        """Validation result model."""

        is_valid: bool
        errors: Annotated[list[FlextModels.ErrorDetail], Field(default_factory=list)]
        warnings: FlextTypes.StringList = Field(default_factory=list)

    class Configuration(FrozenStrictModel):
        """Base configuration model - immutable."""

        version: str = FlextConstants.Core.DEFAULT_VERSION
        enabled: bool = True
        settings: FlextTypes.Dict = Field(default_factory=dict)

    class HealthCheck(ArbitraryTypesModel):
        """Health check model for service monitoring."""

        service_name: str
        status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
        checks: FlextTypes.Dict = Field(default_factory=dict)
        last_check: datetime = Field(default_factory=lambda: datetime.now(UTC))
        details: FlextTypes.Dict = Field(default_factory=dict)

    class Metric(ArbitraryTypesModel):
        """Metric model for monitoring."""

        name: str
        value: float
        unit: str | None = None
        timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
        labels: dict[str, str] = Field(default_factory=dict)

    class Audit(ArbitraryTypesModel):
        """Audit trail model."""

        action: str
        user_id: str
        timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
        entity_type: str | None = None
        entity_id: str | None = None
        changes: FlextTypes.Dict = Field(default_factory=dict)
        ip_address: str | None = None

    class Policy(ArbitraryTypesModel):
        """Policy model for business rules."""

        name: str
        description: str | None = None
        rules: Annotated[list[FlextTypes.Dict], Field(default_factory=list)]
        enabled: bool = True
        priority: int = Field(
            default=FlextConstants.Performance.DEFAULT_PRIORITY,
            ge=FlextConstants.Performance.MIN_PRIORITY,
        )

    class Notification(ArbitraryTypesModel):
        """Notification model."""

        type: str
        recipient: str
        subject: str
        body: str
        sent_at: datetime | None = None
        status: Literal["pending", "sent", "failed"] = "pending"

    class Cache(ArbitraryTypesModel):
        """Cache configuration model."""

        key: str
        value: object
        ttl_seconds: int = Field(
            default_factory=lambda: int(
                _get_config().timeout_seconds * 10
            )  # TTL = 10x timeout
        )
        expires_at: datetime | None = None

    class Bus(TimeoutableMixin, RetryableMixin):
        """Enhanced message bus model with config-driven defaults.

        Uses TimeoutableMixin and RetryableMixin for timeout and retry configuration.
        Uses UuidField pattern for bus_id generation.

        Note: Removed BaseModel from inheritance since mixins already inherit from it.
        """

        bus_id: str = Field(default_factory=lambda: f"bus_{uuid.uuid4().hex[:8]}")
        handlers: dict[str, FlextTypes.StringList] = Field(default_factory=dict)
        middlewares: FlextTypes.StringList = Field(default_factory=list)

        model_config = _BaseConfigDict.ARBITRARY

    class Payload[T](ArbitraryTypesModel, IdentifiableMixin, TimestampableMixin):
        """Enhanced payload model with computed field.

        Uses IdentifiableMixin for id and TimestampableMixin for created_at.
        """

        data: T = Field(...)  # Required field, no default
        metadata: FlextTypes.Domain.EventMetadata = Field(default_factory=dict)
        expires_at: datetime | None = None
        correlation_id: str | None = None
        source_service: str | None = None
        message_type: str | None = None

        @computed_field
        def is_expired(self) -> bool:
            """Computed property to check if payload is expired."""
            if self.expires_at is None:
                return False
            return datetime.now(UTC) > self.expires_at

    class Token(ArbitraryTypesModel):
        """Token model for authentication."""

        value: str
        type: Literal["bearer", "api_key", "jwt"] = "bearer"
        expires_at: datetime | None = None
        scopes: FlextTypes.StringList = Field(default_factory=list)

    class Permission(FrozenStrictModel):
        """Immutable permission model."""

        resource: str
        action: str
        conditions: FlextTypes.Dict = Field(default_factory=dict)

    class Role(ArbitraryTypesModel):
        """Role model for authorization."""

        name: str
        description: str | None = None
        permissions: Annotated[
            list[FlextModels.Permission], Field(default_factory=list)
        ]

    class User(Entity):
        """User entity model."""

        username: str
        email: str
        roles: FlextTypes.StringList = Field(default_factory=list)
        is_active: bool = True
        last_login: datetime | None = None

    class Session(ArbitraryTypesModel, IdentifiableMixin, TimestampableMixin):
        """Session model.

        Uses IdentifiableMixin for id and TimestampableMixin for created_at.
        """

        user_id: str
        expires_at: datetime
        data: FlextTypes.Dict = Field(default_factory=dict)

    class Task(ArbitraryTypesModel, IdentifiableMixin, TimestampableMixin):
        """Task model for background processing.

        Uses IdentifiableMixin for id and TimestampableMixin for created_at.
        """

        name: str
        status: Literal["pending", "running", "completed", "failed"] = "pending"
        payload: FlextTypes.Dict = Field(default_factory=dict)
        result: object = None
        error: str | None = None
        started_at: datetime | None = None
        completed_at: datetime | None = None

    class Queue(ArbitraryTypesModel, TimeoutableMixin):
        """Queue model for message processing.

        Uses TimeoutableMixin for timeout_seconds.
        """

        name: str
        messages: Annotated[list[FlextTypes.Dict], Field(default_factory=list)]
        max_size: int = Field(
            default_factory=lambda: FlextConstants.Performance.DEFAULT_CACHE_SIZE
        )

    class Schedule(ArbitraryTypesModel):
        """Schedule model for cron jobs."""

        name: str
        cron_expression: str
        task: str
        enabled: bool = True
        last_run: datetime | None = None
        next_run: datetime | None = None

    class Feature(FrozenStrictModel):
        """Feature flag model."""

        name: str
        enabled: bool = False
        rollout_percentage: float = Field(
            default=FlextConstants.Performance.DEFAULT_ROLLOUT_PERCENTAGE,
            ge=FlextConstants.Performance.MIN_ROLLOUT_PERCENTAGE,
            le=FlextConstants.Performance.MAX_ROLLOUT_PERCENTAGE,
        )
        conditions: FlextTypes.Dict = Field(default_factory=dict)

    class Rate(ArbitraryTypesModel):
        """Rate limiting model."""

        key: str
        limit: int
        window_seconds: int = FlextConstants.Performance.DEFAULT_RATE_LIMIT_WINDOW
        current_count: int = 0
        reset_at: datetime | None = None

    class RateLimiterState(ArbitraryTypesModel):
        """Rate limiter state tracking structure.

        Used for tracking rate limiting state across operations,
        including current count, window boundaries, and reset times.
        """

        key: str
        limit: int
        current_count: int = 0
        window_start: datetime
        window_seconds: int = FlextConstants.Performance.DEFAULT_RATE_LIMIT_WINDOW
        reset_at: datetime | None = None
        is_blocked: bool = False

    class Circuit(ArbitraryTypesModel):
        """Circuit breaker model."""

        name: str
        state: Literal["closed", "open", "half_open"] = "closed"
        failure_count: int = 0
        failure_threshold: int = FlextConstants.Reliability.DEFAULT_FAILURE_THRESHOLD
        timeout_seconds: int = Field(
            default_factory=lambda: int(_get_config().timeout_seconds)
        )
        last_failure: datetime | None = None

    class Retry(ArbitraryTypesModel):
        """Retry model."""

        attempt: int = 0
        max_attempts: int = Field(
            default_factory=lambda: _get_config().max_retry_attempts
        )
        delay_seconds: float = FlextConstants.Performance.DEFAULT_DELAY_SECONDS
        backoff_multiplier: float = (
            FlextConstants.Performance.DEFAULT_BACKOFF_MULTIPLIER
        )

    class Batch(ArbitraryTypesModel, IdentifiableMixin):
        """Batch processing model.

        Uses IdentifiableMixin for id.
        """

        items: Annotated[FlextTypes.List, Field(default_factory=list)]
        size: int = Field(
            default=FlextConstants.Performance.BatchProcessing.SMALL_SIZE, ge=1
        )
        processed_count: int = 0

    class Stream(ArbitraryTypesModel):
        """Stream processing model."""

        stream_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        position: int = 0
        batch_size: int = FlextConstants.Performance.BatchProcessing.STREAM_SIZE
        buffer: Annotated[FlextTypes.List, Field(default_factory=list)]

    class Pipeline(ArbitraryTypesModel):
        """Pipeline model."""

        name: str
        stages: Annotated[list[FlextTypes.Dict], Field(default_factory=list)]
        current_stage: int = 0
        status: Literal["idle", "running", "completed", "failed"] = "idle"

    class Workflow(ArbitraryTypesModel, IdentifiableMixin):
        """Workflow model.

        Uses IdentifiableMixin for id.
        """

        name: str
        steps: Annotated[list[FlextTypes.Dict], Field(default_factory=list)]
        current_step: int = 0
        context: FlextTypes.Dict = Field(default_factory=dict)

    class Archive(ArbitraryTypesModel, IdentifiableMixin, TimestampableMixin):
        """Archive model.

        Uses IdentifiableMixin for id and TimestampableMixin for created_at.
        """

        entity_type: str
        entity_id: str
        archived_by: str
        data: FlextTypes.Dict = Field(default_factory=dict)

    class Import(ArbitraryTypesModel):
        """Import model."""

        import_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        source: str
        format: str
        status: Literal["pending", "processing", "completed", "failed"] = "pending"
        records_total: int = 0
        records_processed: int = 0
        errors: FlextTypes.StringList = Field(default_factory=list)

    class Export(ArbitraryTypesModel):
        """Export model."""

        export_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        format: str
        filters: FlextTypes.Dict = Field(default_factory=dict)
        status: Literal["pending", "processing", "completed", "failed"] = "pending"
        file_path: str | None = None

    class EmailAddress(Value):
        """Enhanced email address value object with Field constraints.

        Pydantic 2 features:
        - JSON schema customization with examples
        - Pattern validation with regex
        - Custom field validator with FlextResult
        """

        address: str = Field(
            ...,
            pattern=FlextConstants.Platform.PATTERN_EMAIL,
            description="Valid email address",
            examples=["user@example.com", "REDACTED_LDAP_BIND_PASSWORD@company.org"],
            json_schema_extra={
                "format": "email",
                "title": "Email Address",
            },
        )

        @field_validator("address")
        @classmethod
        def _validate_email_format(cls, v: str) -> str:
            """Validate email format using centralized FlextUtilities.Validation."""
            result: FlextResult[str] = FlextModels.Validation.validate_email_address(v)
            if result.is_failure:
                raise FlextExceptions.ValidationError(
                    message=result.error or "Invalid email format",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return result.unwrap()

    class Host(Value):
        """Host/hostname value object."""

        hostname: str

        @field_validator("hostname")
        @classmethod
        def validate_host_format(cls, v: str) -> str:
            """Validate hostname format using centralized FlextUtilities.Validation."""
            result: FlextResult[str] = FlextModels.Validation.validate_hostname(v)
            if result.is_failure:
                raise FlextExceptions.ValidationError(
                    message=result.error or "Invalid hostname format",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return result.unwrap()

    class EntityId(Value):
        """Entity identifier value object with validation."""

        value: str

        @field_validator("value")
        @classmethod
        def validate_entity_id_format(cls, v: str) -> str:
            """Validate entity ID format using centralized FlextUtilities.Validation."""
            result = FlextModels.Validation.validate_entity_id(v)
            if result.is_failure:
                raise FlextExceptions.ValidationError(
                    message=result.error or "Invalid entity ID format",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return result.unwrap()

    class Url(Value):
        """Enhanced URL value object."""

        url: str

        @field_validator("url")
        @classmethod
        def _validate_url_format(cls, v: str) -> str:
            """Validate URL format using centralized FlextModels.Validation."""
            result: FlextResult[str] = FlextModels.Validation.validate_url(v)
            if result.is_failure:
                raise FlextExceptions.ValidationError(
                    message=result.error or "Invalid URL format",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return result.unwrap()

    class Project(Entity):
        """Enhanced project entity with advanced validation."""

        name: str
        organization_id: str
        repository_path: str | None = None
        is_test_project: bool = False
        test_framework: str | None = None
        project_type: str = "application"

        @model_validator(mode="after")
        def validate_business_rules(self) -> Self:
            """Complex business rules validation."""
            # Test project consistency
            if self.is_test_project and not self.test_framework:
                self.test_framework = "pytest"  # Default

            # Repository path validation
            if (
                self.repository_path
                and not self.repository_path.startswith("/")
                and not self.repository_path.startswith("http")
            ):
                msg = "Repository path must be absolute or URL"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            return self

    class WorkspaceInfo(AggregateRoot):
        """Enhanced workspace aggregate with advanced validation."""

        workspace_id: str
        name: str
        root_path: str
        projects: Annotated[list[FlextModels.Project], Field(default_factory=list)]
        total_files: int = 0
        total_size_bytes: int = 0

        @model_validator(mode="after")
        def validate_business_rules(self) -> Self:
            """Complex workspace validation."""
            # Workspace consistency
            if self.projects and self.total_files == 0:
                msg = "Workspace with projects must have files"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Size validation
            max_workspace_size = 10 * FlextConstants.Utilities.BYTES_PER_KB**3  # 10GB
            if self.total_size_bytes > max_workspace_size:
                msg = "Workspace too large"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            return self

    class Tag(Value):
        """Tag value object."""

        name: str
        value: str | None = None

    class Label(Value):
        """Label value object."""

        key: str
        value: str

    class Category(Value):
        """Category value object."""

        name: str
        parent: str | None = None

    class Priority(Value):
        """Priority value object."""

        level: int = Field(ge=1, le=10)
        name: str | None = None

    class Status(Value):
        """Status value object."""

        code: str
        description: str | None = None

    class WorkspaceStatus(StrEnum):
        """Workspace status enumeration."""

        INITIALIZING = "initializing"
        READY = "ready"
        ERROR = "error"
        MAINTENANCE = "maintenance"

    # Additional domain models continue with advanced patterns...
    class FactoryRegistrationModel(StrictArbitraryTypesModel):
        """Enhanced factory registration with advanced validation."""

        name: str
        factory: Callable[[], object]
        singleton: bool = False
        dependencies: FlextTypes.StringList = Field(default_factory=list)

        @field_validator("factory")
        @classmethod
        def validate_factory_signature(
            cls, v: Callable[[], object]
        ) -> Callable[[], object]:
            """Validate factory is callable with proper signature."""
            # Check if it's a proper factory (no required args)
            sig = inspect.signature(v)
            required_params = [
                p
                for p in sig.parameters.values()
                if p.default == inspect.Parameter.empty
                and p.kind
                not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
            ]
            if required_params:
                msg = f"Factory cannot have required parameters: {[p.name for p in required_params]}"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            return v

    class BatchRegistrationModel(StrictArbitraryTypesModel):
        """Batch registration model with advanced validation."""

        services: FlextTypes.Dict = Field(default_factory=dict)
        factories: dict[str, Callable[[], object]] = Field(default_factory=dict)
        singletons: FlextTypes.Dict = Field(default_factory=dict)

        @model_validator(mode="after")
        def validate_non_empty(self) -> Self:
            """Ensure at least one registration exists."""
            if not any([self.services, self.factories, self.singletons]):
                msg = "At least one registration required"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return self

    class LogOperation(StrictArbitraryTypesModel):
        """Enhanced log operation model."""

        level: str = Field(default_factory=lambda: "INFO")
        message: str
        context: FlextTypes.Dict = Field(default_factory=dict)
        timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
        source: str | None = None
        operation: str | None = None
        obj: object | None = None

    class ProcessingRequest(ArbitraryTypesModel):
        """Enhanced processing request with advanced validation."""

        operation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        data: FlextTypes.Dict = Field(default_factory=dict)
        context: FlextTypes.Dict = Field(default_factory=dict)
        timeout_seconds: int = Field(
            default_factory=lambda: int(_get_config().timeout_seconds)
        )
        retry_attempts: int = Field(
            default_factory=lambda: _get_config().max_retry_attempts
        )
        enable_validation: bool = True

        model_config = ConfigDict(
            validate_assignment=False,  # Allow invalid values to be set for testing
            use_enum_values=True,
            arbitrary_types_allowed=True,
        )

        @field_validator("context")
        @classmethod
        def validate_context(cls, v: FlextTypes.Dict) -> FlextTypes.Dict:
            """Validate context has required fields."""
            if "correlation_id" not in v:
                v["correlation_id"] = str(uuid.uuid4())
            if "timestamp" not in v:
                v["timestamp"] = datetime.now(UTC).isoformat()
            return v

        def validate_processing_constraints(self) -> FlextResult[None]:
            """Validate constraints that should be checked during processing."""
            max_timeout_seconds = FlextConstants.Utilities.MAX_TIMEOUT_SECONDS
            if self.timeout_seconds > max_timeout_seconds:
                return FlextResult[None].fail(
                    f"Timeout cannot exceed {max_timeout_seconds} seconds"
                )

            return FlextResult[None].ok(None)

    class HandlerRegistration(StrictArbitraryTypesModel):
        """Handler registration with advanced validation."""

        name: str
        handler: object
        event_types: FlextTypes.StringList = Field(default_factory=list)
        priority: int = Field(
            default=FlextConstants.Cqrs.DEFAULT_PRIORITY,
            ge=FlextConstants.Cqrs.MIN_PRIORITY,
            le=FlextConstants.Cqrs.MAX_PRIORITY,
            description="Priority level",
        )

        @field_validator("handler")
        @classmethod
        def validate_handler(cls, v: object) -> Callable[[object], object]:
            """Validate handler is properly callable."""
            if not callable(v):
                error_msg = "Handler must be callable"
                raise FlextExceptions.TypeError(
                    message=error_msg,
                    error_code=FlextConstants.Errors.TYPE_ERROR,
                )
            return cast("Callable[[object], object]", v)

    class BatchProcessingConfig(StrictArbitraryTypesModel):
        """Enhanced batch processing configuration."""

        batch_size: int = Field(default_factory=lambda: _get_config().batch_size)
        max_workers: int = Field(default_factory=lambda: _get_config().max_workers)
        timeout_per_item: int = Field(
            default_factory=lambda: int(_get_config().timeout_seconds)
        )
        continue_on_error: bool = True
        data_items: Annotated[FlextTypes.List, Field(default_factory=list)]

        @field_validator("data_items")
        @classmethod
        def validate_data_items(cls, v: FlextTypes.List) -> FlextTypes.List:
            """Validate data items are not empty when provided."""
            if len(v) > FlextConstants.Performance.BatchProcessing.MAX_ITEMS:
                msg = f"Batch cannot exceed {FlextConstants.Performance.BatchProcessing.MAX_ITEMS} items"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return v

        @field_validator("max_workers")
        @classmethod
        def validate_max_workers(cls, v: int) -> int:
            """Validate max workers is reasonable."""
            max_workers_limit = FlextConstants.Config.MAX_WORKERS_THRESHOLD
            if v > max_workers_limit:
                msg = f"Max workers cannot exceed {max_workers_limit}"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return v

        @model_validator(mode="after")
        def validate_batch(self) -> Self:
            """Validate batch configuration consistency."""
            max_batch_size = (
                FlextConstants.Performance.BatchProcessing.MAX_VALIDATION_SIZE
            )
            if self.batch_size > max_batch_size:
                msg = f"Batch size cannot exceed {max_batch_size}"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Adjust max_workers to not exceed batch_size without triggering validation
            adjusted_workers = min(self.max_workers, self.batch_size)
            # Use direct assignment to __dict__ to bypass Pydantic validation
            self.__dict__["max_workers"] = adjusted_workers

            return self

    class HandlerExecutionConfig(StrictArbitraryTypesModel):
        """Enhanced handler execution configuration."""

        handler_name: str
        input_data: FlextTypes.Dict = Field(default_factory=dict)
        execution_context: FlextTypes.Dict = Field(default_factory=dict)
        timeout_seconds: int = Field(
            default_factory=lambda: int(_get_config().timeout_seconds)
        )
        retry_on_failure: bool = True
        max_retries: int = Field(
            default_factory=lambda: _get_config().max_retry_attempts
        )
        fallback_handlers: FlextTypes.StringList = Field(default_factory=list)

        @field_validator("handler_name")
        @classmethod
        def validate_handler_name(cls, v: str) -> str:
            """Validate handler name format."""
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", v):
                msg = "Handler name must be valid identifier"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return v

    class PipelineConfiguration(ArbitraryTypesModel):
        """Pipeline configuration with advanced validation."""

        name: str = Field(min_length=FlextConstants.Performance.MIN_NAME_LENGTH)
        steps: Annotated[list[FlextTypes.Dict], Field(default_factory=list)]
        parallel_execution: bool = FlextConstants.Cqrs.DEFAULT_PARALLEL_EXECUTION
        stop_on_error: bool = FlextConstants.Cqrs.DEFAULT_STOP_ON_ERROR
        max_parallel: int = Field(
            gt=0, default_factory=lambda: _get_config().max_workers
        )

        @field_validator("name")
        @classmethod
        def validate_name(cls, v: str) -> str:
            """Validate pipeline name."""
            max_name_length = FlextConstants.Validation.MAX_NAME_LENGTH
            if len(v) > max_name_length:
                msg = f"Pipeline name too long (max {max_name_length} characters)"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return v

        @field_validator("steps")
        @classmethod
        def validate_steps(cls, v: list[object]) -> list[object]:
            """Validate pipeline steps."""
            if not v:
                msg = "Pipeline must have at least one step"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return v

    class ProcessingResult(ArbitraryTypesModel):
        """Processing result with computed fields."""

        operation_id: str
        status: Literal["success", "failure", "partial"]
        data: object = None
        errors: FlextTypes.StringList = Field(default_factory=list)
        execution_time_ms: int = 0

        @field_validator("execution_time_ms")
        @classmethod
        def validate_execution_time(cls, v: int) -> int:
            """Validate execution time is reasonable."""
            max_execution_time_ms = FlextConstants.Cqrs.MAX_TIMEOUT  # 5 minutes
            if v > max_execution_time_ms:
                msg = f"Execution time exceeds maximum ({max_execution_time_ms // 60000} minutes)"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return v

        @computed_field
        def execution_time_seconds(self) -> float:
            """Get execution time in seconds."""
            return self.execution_time_ms / 1000.0

    class JsonFormatConfig(ArbitraryTypesModel):
        """Enhanced JSON format configuration."""

        indent: int = Field(default_factory=lambda: 2)
        sort_keys: bool = False
        ensure_ascii: bool = False
        separators: tuple[str, str] = (",", ":")
        default_handler: Callable[[object], object] | None = None

    class TimestampConfig(StrictArbitraryTypesModel):
        """Enhanced timestamp configuration."""

        obj: object
        use_utc: bool = Field(default_factory=lambda: True)
        auto_update: bool = Field(default_factory=lambda: True)
        format: str = "%Y-%m-%dT%H:%M:%S.%fZ"
        timezone: str | None = None
        created_at_field: str = "created_at"
        updated_at_field: str = "updated_at"
        field_names: dict[str, str] = Field(
            default_factory=lambda: {
                "created_at": "created_at",
                "updated_at": "updated_at",
            }
        )

        @field_validator("created_at_field", "updated_at_field")
        @classmethod
        def validate_field_names(cls, v: str) -> str:
            """Validate field names are valid identifiers."""
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
                msg = f"Invalid field name: {v}"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return v

    class SerializationRequest(StrictArbitraryTypesModel):
        """Enhanced serialization request."""

        data: object
        format: Literal["json", "yaml", "toml", "msgpack"] = "json"
        encoding: str = Field(default_factory=lambda: "utf-8")
        compression: Literal["none", "gzip", "bzip2", "lz4"] | None = None
        pretty_print: bool = False
        use_model_dump: bool = True
        indent: int | None = None
        sort_keys: bool = False
        ensure_ascii: bool = False

    class DomainServiceExecutionRequest(ArbitraryTypesModel):
        """Domain service execution request with advanced validation."""

        service_name: str
        method_name: str
        parameters: FlextTypes.Dict = Field(default_factory=dict)
        context: FlextTypes.Dict = Field(default_factory=dict)
        timeout_seconds: int = Field(
            default_factory=lambda: int(_get_config().timeout_seconds)
        )
        execution: bool = False
        enable_validation: bool = True

        @field_validator("context")
        @classmethod
        def validate_context(cls, v: FlextTypes.Dict) -> FlextTypes.Dict:
            """Ensure context has required fields."""
            if "trace_id" not in v:
                v["trace_id"] = str(uuid.uuid4())
            if "span_id" not in v:
                v["span_id"] = str(uuid.uuid4())
            return v

        @field_validator("timeout_seconds")
        @classmethod
        def validate_timeout(cls, v: int) -> int:
            """Validate timeout is reasonable."""
            max_timeout_seconds = FlextConstants.Performance.MAX_TIMEOUT_SECONDS
            if v > max_timeout_seconds:
                msg = f"Timeout cannot exceed {max_timeout_seconds} seconds"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return v

    class DomainServiceValidationRequest(ArbitraryTypesModel):
        """Domain service validation request."""

        entity: FlextTypes.Dict
        rules: FlextTypes.StringList = Field(default_factory=list)
        validate_business_rules: bool = True
        validate_integrity: bool = True
        validate_permissions: bool = False
        context: FlextTypes.Dict = Field(default_factory=dict)

        @model_validator(mode="after")
        def validate_rules(self) -> Self:
            """Validate at least one validation type is enabled."""
            if not any([
                self.validate_business_rules,
                self.validate_integrity,
                self.validate_permissions,
                self.rules,
            ]):
                msg = "At least one validation type must be enabled"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return self

    class DomainServiceBatchRequest(ArbitraryTypesModel):
        """Domain service batch request."""

        service_name: str
        operations: Annotated[list[FlextTypes.Dict], Field(default_factory=list)]
        parallel_execution: bool = False
        stop_on_error: bool = True
        batch_size: int = Field(default_factory=lambda: _get_config().batch_size)
        timeout_per_operation: int = Field(
            default_factory=lambda: int(_get_config().timeout_seconds)
        )

        @field_validator("operations")
        @classmethod
        def validate_operations(cls, v: list[object]) -> list[object]:
            """Validate operations list[object]."""
            if not v:
                msg = "Operations list[object] cannot be empty"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            max_batch_operations = FlextConstants.Performance.MAX_BATCH_OPERATIONS
            if len(v) > max_batch_operations:
                msg = f"Batch cannot exceed {max_batch_operations} operations"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return v

    class DomainServiceMetricsRequest(ArbitraryTypesModel):
        """Domain service metrics request."""

        service_name: str
        metric_types: FlextTypes.StringList = Field(
            default_factory=lambda: ["performance", "errors", "throughput"]
        )
        time_range_seconds: int = FlextConstants.Performance.DEFAULT_TIME_RANGE_SECONDS
        aggregation: Literal["sum", "avg", "min", "max", "count"] = "avg"
        group_by: FlextTypes.StringList = Field(default_factory=list)
        filters: FlextTypes.Dict = Field(default_factory=dict)

        @field_validator("metric_types")
        @classmethod
        def validate_prefix(cls, v: list[object]) -> list[object]:
            """Validate metric types."""
            valid_types = {
                "performance",
                "errors",
                "throughput",
                "latency",
                "availability",
            }
            for metric_type in v:
                if metric_type not in valid_types:
                    msg = f"Invalid metric type: {metric_type}"
                    raise FlextExceptions.ValidationError(
                        message=msg,
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
            return v

    class DomainServiceResourceRequest(ArbitraryTypesModel):
        """Domain service resource request."""

        service_name: str = "default_service"
        resource_type: str = "default_resource"
        resource_id: str | None = None
        resource_limit: int = 1000
        action: Literal["get", "create", "update", "delete", "list[object]"] = "get"
        data: FlextTypes.Dict = Field(default_factory=dict)
        filters: FlextTypes.Dict = Field(default_factory=dict)

        @field_validator("resource_type")
        @classmethod
        def validate_resource_type(cls, v: str) -> str:
            """Validate resource type format."""
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", v):
                msg = "Resource type must be valid identifier"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return v

        @field_validator("resource_limit")
        @classmethod
        def validate_resource_limit(cls, v: int) -> int:
            """Validate resource limit is positive."""
            if v <= FlextConstants.Core.ZERO:
                msg = "Resource limit must be positive"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return v

    class DomainServiceOperation(ArbitraryTypesModel):
        """Domain service operation definition."""

        name: str
        operation_callable: object
        arguments: object = None
        retry_config: FlextModels.RetryConfiguration | None = None
        validate_before_execution: bool = True

    class OperationExecutionRequest(ArbitraryTypesModel):
        """Operation execution request."""

        operation_name: str
        operation_callable: object
        arguments: FlextTypes.Dict = Field(default_factory=dict)
        keyword_arguments: FlextTypes.Dict = Field(default_factory=dict)
        timeout_seconds: int = Field(
            default_factory=lambda: int(_get_config().timeout_seconds)
        )
        retry_config: FlextTypes.Dict = Field(default_factory=dict)

        @field_validator("operation_name")
        @classmethod
        def validate_operation_name(cls, v: str) -> str:
            """Validate operation name format."""
            max_operation_name_length = (
                FlextConstants.Performance.MAX_OPERATION_NAME_LENGTH
            )
            if len(v) > max_operation_name_length:
                msg = "Operation name too long"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return v

        @field_validator("operation_callable", mode="plain")
        @classmethod
        def validate_operation_callable(
            cls, v: object
        ) -> FlextProtocols.Foundation.OperationCallable:
            """Validate operation is callable."""
            if not callable(v):
                error_msg = "Operation must be callable"
                model_name = "OperationExecutionRequest"
                raise ValidationError.from_exception_data(
                    model_name,
                    [
                        {
                            "type": "callable_type",
                            "loc": ("operation_callable",),
                            "input": v,
                            "ctx": {"error": error_msg},
                        }
                    ],
                )
            return cast("FlextProtocols.Foundation.OperationCallable", v)

    class RetryConfiguration(ArbitraryTypesModel):
        """Retry configuration with advanced validation."""

        max_attempts: int = Field(
            default_factory=lambda: _get_config().max_retry_attempts
        )
        initial_delay_seconds: float = Field(
            default=FlextConstants.Performance.DEFAULT_INITIAL_DELAY_SECONDS, gt=0
        )
        max_delay_seconds: float = Field(
            default=FlextConstants.Performance.DEFAULT_MAX_DELAY_SECONDS, gt=0
        )
        exponential_backoff: bool = True
        backoff_multiplier: float = Field(
            default=FlextConstants.Performance.DEFAULT_BACKOFF_MULTIPLIER, ge=1.0
        )
        retry_on_exceptions: Annotated[
            list[type[BaseException]], Field(default_factory=list)
        ]
        retry_on_status_codes: Annotated[FlextTypes.List, Field(default_factory=list)]

        @field_validator("retry_on_status_codes")
        @classmethod
        def validate_backoff_strategy(cls, v: FlextTypes.List) -> FlextTypes.List:
            """Validate status codes are valid HTTP codes."""
            validated_codes: FlextTypes.List = []
            for code in v:
                try:
                    if isinstance(code, (int, str)):
                        code_int = int(str(code))
                        if (
                            not FlextConstants.Platform.MIN_HTTP_STATUS_RANGE
                            <= code_int
                            <= FlextConstants.Platform.MAX_HTTP_STATUS_RANGE
                        ):
                            msg = f"Invalid HTTP status code: {code}"
                            raise FlextExceptions.ValidationError(
                                message=msg,
                                error_code=FlextConstants.Errors.VALIDATION_ERROR,
                            )
                        validated_codes.append(code_int)
                    else:
                        msg = f"Invalid HTTP status code type: {type(code)}"
                        raise FlextExceptions.TypeError(
                            message=msg,
                            error_code=FlextConstants.Errors.TYPE_ERROR,
                        )
                except (ValueError, TypeError) as e:
                    msg = f"Invalid HTTP status code: {code}"
                    raise FlextExceptions.ValidationError(
                        message=msg,
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    ) from e
            return validated_codes

        @model_validator(mode="after")
        def validate_delay_consistency(self) -> Self:
            """Validate delay configuration consistency."""
            if self.max_delay_seconds < self.initial_delay_seconds:
                msg = "max_delay_seconds must be >= initial_delay_seconds"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return self

    class CircuitBreakerConfiguration(ArbitraryTypesModel):
        """Circuit breaker configuration."""

        failure_threshold: int = Field(
            default=FlextConstants.Reliability.DEFAULT_CIRCUIT_BREAKER_THRESHOLD
        )
        recovery_timeout_seconds: int = Field(
            default=FlextConstants.Performance.DEFAULT_RECOVERY_TIMEOUT
        )
        half_open_max_calls: int = Field(
            default_factory=lambda: _get_config().max_retry_attempts
        )
        sliding_window_size: int = Field(
            default_factory=lambda: _get_config().batch_size
        )
        minimum_throughput: int = Field(
            default=FlextConstants.Cqrs.DEFAULT_MINIMUM_THROUGHPUT,
            description="Minimum throughput threshold",
        )
        slow_call_duration_seconds: float = Field(
            default=FlextConstants.Performance.DEFAULT_DELAY_SECONDS, gt=0
        )
        slow_call_rate_threshold: float = Field(
            default=FlextConstants.Validation.MAX_PERCENTAGE / 2.0
        )  # 50%

        @model_validator(mode="after")
        def validate_circuit_breaker_consistency(self) -> Self:
            """Validate circuit breaker configuration."""
            if self.half_open_max_calls > self.sliding_window_size:
                msg = "half_open_max_calls cannot exceed sliding_window_size"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return self

    class ValidationConfiguration(ArbitraryTypesModel):
        """Validation configuration."""

        enable_strict_mode: bool = Field(default_factory=lambda: True)
        max_validation_errors: int = Field(
            default=FlextConstants.Cqrs.DEFAULT_MAX_VALIDATION_ERRORS,
            description="Maximum validation errors",
        )
        validate_on_assignment: bool = True
        validate_on_read: bool = False
        custom_validators: Annotated[FlextTypes.List, Field(default_factory=list)]

        @field_validator("custom_validators")
        @classmethod
        def validate_additional_validators(cls, v: FlextTypes.List) -> FlextTypes.List:
            """Validate custom validators are callable."""
            for validator in v:
                if not callable(validator):
                    msg = "All validators must be callable"
                    raise FlextExceptions.TypeError(
                        message=msg,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )
            return v

    class ServiceExecutionContext(ArbitraryTypesModel):
        """Service execution context."""

        context_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        causation_id: str | None = None
        user_context: FlextTypes.Dict = Field(default_factory=dict)
        security_context: FlextTypes.Dict = Field(default_factory=dict)
        metadata: FlextTypes.Dict = Field(default_factory=dict)
        deadline: datetime | None = None

        @field_validator("correlation_id")
        @classmethod
        def validate_context_name(cls, v: str) -> str:
            """Validate correlation ID format."""
            if not re.match(r"^[a-zA-Z0-9\-_]+$", v):
                msg = "Correlation ID must contain only alphanumeric, dash, and underscore"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return v

        @field_validator("deadline")
        @classmethod
        def validate_correlation_id(cls, v: datetime | None) -> datetime | None:
            """Validate deadline is in the future."""
            if v and v <= datetime.now(UTC):
                msg = "Deadline must be in the future"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return v

    class ConditionalExecutionRequest(ArbitraryTypesModel):
        """Conditional execution request."""

        condition: Callable[[object], bool]
        true_action: Callable[[object], object]
        false_action: Callable[[object], object] | None = None
        context: FlextTypes.Dict = Field(default_factory=dict)

        @field_validator("condition", "true_action", "false_action")
        @classmethod
        def validate_condition(
            cls, v: Callable[[object], object] | None
        ) -> Callable[[object], object] | None:
            """Validate callables."""
            return v

    class StateMachineRequest(ArbitraryTypesModel):
        """State machine request."""

        initial_state: str
        transitions: FlextTypes.Dict = Field(default_factory=dict)
        current_state: str | None = None
        state_data: FlextTypes.Dict = Field(default_factory=dict)

        @field_validator("transitions")
        @classmethod
        def validate_transitions(cls, v: FlextTypes.Dict) -> FlextTypes.Dict:
            """Validate state transitions."""
            if not v:
                msg = "Transitions cannot be empty"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Validate transition structure
            for state, transitions in v.items():
                if not isinstance(transitions, dict):
                    msg = f"Transitions for state {state} must be a FlextTypes.Dict"
                    raise FlextExceptions.TypeError(
                        message=msg,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )
                for event, next_state in cast("FlextTypes.Dict", transitions).items():
                    if not isinstance(next_state, str):
                        msg = f"Next state for {state}.{event} must be a string"
                        raise FlextExceptions.TypeError(
                            message=msg,
                            error_code=FlextConstants.Errors.TYPE_ERROR,
                        )

            return v

    class ResourceManagementRequest(ArbitraryTypesModel):
        """Resource management request."""

        resource_type: str
        resource_id: str | None = None
        action: Literal["acquire", "release", "check", "list[object]"] = "acquire"
        timeout_seconds: int = Field(
            default_factory=lambda: int(_get_config().timeout_seconds)
        )
        metadata: FlextTypes.Dict = Field(default_factory=dict)

        @model_validator(mode="after")
        def validate_resource_manager(self) -> Self:
            """Validate resource management request."""
            if self.action in {"release", "check"} and not self.resource_id:
                msg = f"resource_id required for action: {self.action}"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return self

    class MetricsCollectionRequest(ArbitraryTypesModel):
        """Metrics collection request."""

        metric_name: str
        value: float
        unit: str | None = None
        dimensions: dict[str, str] = Field(default_factory=dict)
        timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

        @field_validator("dimensions")
        @classmethod
        def validate_metrics_collector(cls, v: FlextTypes.Dict) -> FlextTypes.Dict:
            """Validate dimensions."""
            max_dimensions = FlextConstants.Performance.MAX_DIMENSIONS
            if len(v) > max_dimensions:
                msg = f"Too many dimensions (max {max_dimensions})"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return v

    class TransformationRequest(ArbitraryTypesModel):
        """Transformation request."""

        input_data: object
        transformer: Callable[[object], object]
        validation_schema: FlextTypes.Dict | None = None
        error_handler: Callable[[Exception], object] | None = None

        @field_validator("transformer", "error_handler")
        @classmethod
        def validate_transformer(
            cls, v: Callable[[object], object] | None
        ) -> Callable[[object], object] | None:
            """Validate transformer functions."""
            return v

    class StateInitializationRequest(ArbitraryTypesModel):
        """State initialization request."""

        data: object
        state_key: str
        initial_value: object
        ttl_seconds: int | None = None
        persistence_level: Literal["memory", "disk", "distributed"] = "memory"
        field_name: str = "state"
        state: object

        @model_validator(mode="after")
        def validate_state_value(self) -> Self:
            """Validate state initialization."""
            if self.persistence_level == "distributed" and not self.ttl_seconds:
                # Default TTL for distributed state
                self.ttl_seconds = FlextConstants.Performance.DEFAULT_TTL_SECONDS
            return self

    # ============================================================================
    # PHASE 7: SERIALIZATION OPTIMIZATIONS
    # ============================================================================
    # Advanced serialization patterns leveraging Pydantic 2 features
    # for efficient model serialization, custom encoders, and field exclusion

    # No SerializationMixin needed - use Pydantic 2.11's native model_dump() and model_dump_json() directly
    # Examples:
    # model.model_dump(exclude_unset=True, exclude_defaults=True, exclude_none=True)  # compact dict
    # model.model_dump_json(exclude_unset=True, exclude_defaults=True, exclude_none=True)  # compact JSON

    # Custom JSON encoder for complex types
    class FlextJSONEncoder:
        """Custom JSON encoder for FLEXT models with optimized serialization."""

        @staticmethod
        def encode_datetime(dt: datetime) -> str:
            """Encode datetime to ISO format string."""
            return dt.isoformat()

        @staticmethod
        def encode_date(d: date) -> str:
            """Encode date to ISO format string."""
            return d.isoformat()

        @staticmethod
        def encode_time(t: time) -> str:
            """Encode time to ISO format string."""
            return t.isoformat()

        @staticmethod
        def encode_decimal(d: Decimal) -> float | int:
            """Encode Decimal to number."""
            if d % 1 == 0:
                return int(d)
            return float(d)

        @staticmethod
        def encode_uuid(u: uuid.UUID) -> str:
            """Encode UUID to string."""
            return str(u)

        @staticmethod
        def encode_path(p: Path) -> str:
            """Encode Path to string."""
            return str(p)

        @staticmethod
        def encode_bytes(b: bytes) -> str:
            """Encode bytes to base64 string."""
            return base64.b64encode(b).decode("ascii")

    # Model serialization configuration
    class SerializableModel(BaseModel):
        """Base model with optimized serialization support.

        Combines Pydantic BaseModel with SerializationMixin for enhanced
        serialization capabilities. Used by models requiring flexible export options.
        Uses modern Pydantic v2 serialization instead of deprecated json_encoders.
        """

        model_config = ConfigDict(
            # Core settings for serialization
            validate_assignment=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            # Field serialization settings
            ser_json_timedelta="iso8601",
            ser_json_bytes="base64",
            # Schema settings
            json_schema_serialization_defaults_required=True,
            json_schema_mode_override="validation",
        )

    class CompactSerializableModel(BaseModel):
        """Model that defaults to compact serialization.

        Automatically excludes unset, default, and None values during serialization
        unless explicitly overridden. Ideal for API responses and data transfer.
        """

        model_config = ConfigDict(
            validate_assignment=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            populate_by_name=True,
        )

        # Use Pydantic 2.11 built-in model_dump and model_dump_json
        # No need to override - just use the native methods with parameters

    # Serialization context for conditional field inclusion
    class SerializationContext(FrozenStrictModel):
        """Context for controlling serialization behavior."""

        include_sensitive: bool = False
        include_internal: bool = False
        include_computed: bool = True
        include_metadata: bool = True
        target_format: Literal["full", "compact", "minimal"] = "full"
        max_depth: int = 10

    # Model with context-aware serialization
    class ContextAwareModel(BaseModel):
        """Model that serializes based on context settings."""

        @classmethod
        def _get_serialization_context(
            cls, context: FlextTypes.Dict | None
        ) -> FlextTypes.Dict:
            """Extract serialization context from context FlextTypes.Dict."""
            if not context:
                return {
                    "include_sensitive": False,
                    "include_internal": False,
                    "include_computed": True,
                    "include_metadata": True,
                    "target_format": "full",
                    "max_depth": 10,
                }

            return {
                "include_sensitive": context.get("include_sensitive", False),
                "include_internal": context.get("include_internal", False),
                "include_computed": context.get("include_computed", True),
                "include_metadata": context.get("include_metadata", True),
                "target_format": context.get("target_format", "full"),
                "max_depth": context.get("max_depth", 10),
            }

        @override
        def model_dump(
            self,
            *,
            mode: str = "python",
            include: IncEx | None = None,
            exclude: IncEx | None = None,
            context: object | None = None,
            by_alias: bool | None = None,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
            round_trip: bool = False,
            warnings: bool | Literal["none", "warn", "error"] = True,
            fallback: Callable[[object], object]
            | None = None,  # Required for parent compatibility
            serialize_as_any: bool = False,
        ) -> FlextTypes.Dict:
            """Context-aware model dump."""
            if context and isinstance(context, dict):
                ser_context = self._get_serialization_context(
                    cast("FlextTypes.Dict", context)
                )

                # Apply context settings
                target_format = ser_context.get("target_format", "full")
                if target_format == "minimal":
                    exclude_unset = True
                    exclude_defaults = True
                    exclude_none = True
                elif target_format == "compact":
                    exclude_defaults = True
                    exclude_none = True

                # Build exclude set based on context
                exclude_set: set[str] = set()
                if isinstance(exclude, (set, list, tuple)):
                    exclude_set = {
                        str(item)
                        for item in cast(
                            "set[object] | list[object] | tuple[object, ...]", exclude
                        )
                    }
                if not ser_context.get("include_sensitive", False):
                    # Add sensitive fields to exclude
                    exclude_set.update(self._get_sensitive_fields())
                if not ser_context.get("include_internal", False):
                    # Add internal fields to exclude
                    exclude_set.update(self._get_internal_fields())
                if not ser_context.get("include_metadata", True):
                    # Add metadata fields to exclude
                    exclude_set.update(["created_at", "updated_at", "version", "id"])

                # Ensure exclude is proper type for Pydantic (set, dict, or None)
                if exclude_set or isinstance(exclude, (list, tuple)):
                    exclude = exclude_set or None
                elif not isinstance(exclude, (set, dict, type(None))):
                    exclude = None

            return super().model_dump(
                include=include,
                exclude=exclude,
                by_alias=by_alias,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
                round_trip=round_trip,
                warnings=warnings,
                serialize_as_any=serialize_as_any,
                mode=mode,
                context=cast("FlextTypes.Dict | None", context),
                fallback=fallback,  # Pass through for parent compatibility
            )

        @classmethod
        def _get_sensitive_fields(cls) -> set[str]:
            """Get set of sensitive field names to exclude."""
            # Override in subclasses to define sensitive fields
            return {"password", "secret_key", "api_key", "token", "credentials"}

        @classmethod
        def _get_internal_fields(cls) -> set[str]:
            """Get set of internal field names to exclude."""
            # Override in subclasses to define internal fields
            return {"_internal", "_cache", "_state", "domain_events"}

    # Optimized models using serialization features
    class OptimizedEntity(Entity, ContextAwareModel):
        """Entity with optimized serialization support."""

        @classmethod
        @override
        def _get_internal_fields(cls) -> set[str]:
            """Exclude domain events and version from default serialization."""
            return {"domain_events", "version"}

    class OptimizedValue(Value, CompactSerializableModel):
        """Value object with compact serialization by default."""

    class OptimizedCommand(Command, CompactSerializableModel):
        """Command with compact serialization for messaging."""

        @classmethod
        def _get_internal_fields(cls) -> set[str]:
            """Exclude command metadata from default serialization."""
            return {"command_id", "issued_at", "issuer_id"}

    # Batch serialization optimizer
    class BatchSerializer:
        """Optimized batch serialization for collections of models."""

        @staticmethod
        def validate_json_serializable(model: BaseModel) -> None:
            """Validate model is JSON serializable if configuration requires it."""
            # Config defaults from FlextConstants
            if False:  # Always validate JSON serializable
                return

            try:
                # Simple validation - attempt to serialize
                model.model_dump_json()
            except Exception as e:
                msg = f"Model not JSON serializable: {e}"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                ) from e

        @staticmethod
        def serialize_batch(
            models: list[BaseModel],
            output_format: Literal["dict", "json"] = "dict",
            *,
            compact: bool = False,
            parallel: bool = False,
        ) -> list[FlextTypes.Dict] | str:
            """Serialize a batch of models efficiently.

            Args:
                models: List of Pydantic models to serialize
                output_format: Output format ('FlextTypes.Dict' or 'json')
                compact: Use compact serialization
                parallel: Use parallel processing for large batches

            Returns:
                List of FlextTypes.Dicts or JSON array string

            """
            if not models:
                return [] if output_format == "dict" else "[]"

            # Validate JSON serializability if required by configuration
            if output_format == "json":
                for model in models:
                    FlextModels.BatchSerializer.validate_json_serializable(model)

            # Serialization kwargs
            dump_kwargs: FlextTypes.Dict = {}
            if compact:
                dump_kwargs = {
                    "exclude_unset": True,
                    "exclude_defaults": True,
                    "exclude_none": True,
                }

            parallel_threshold = FlextConstants.Performance.PARALLEL_THRESHOLD
            if parallel and len(models) > parallel_threshold:
                # Use thread pool for large batches
                with ThreadPoolExecutor(max_workers=4) as executor:
                    if output_format == "dict":

                        def serialize_to_dict(m: BaseModel) -> FlextTypes.Dict:
                            return m.model_dump(
                                exclude_unset=cast(
                                    "bool", dump_kwargs.get("exclude_unset", False)
                                ),
                                exclude_defaults=cast(
                                    "bool", dump_kwargs.get("exclude_defaults", False)
                                ),
                                exclude_none=cast(
                                    "bool", dump_kwargs.get("exclude_none", False)
                                ),
                            )

                        results = list(executor.map(serialize_to_dict, models))
                    else:

                        def serialize_to_json(m: BaseModel) -> str:
                            return m.model_dump_json(
                                exclude_unset=cast(
                                    "bool", dump_kwargs.get("exclude_unset", False)
                                ),
                                exclude_defaults=cast(
                                    "bool", dump_kwargs.get("exclude_defaults", False)
                                ),
                                exclude_none=cast(
                                    "bool", dump_kwargs.get("exclude_none", False)
                                ),
                            )

                        json_results = list(executor.map(serialize_to_json, models))
                        return f"[{','.join(json_results)}]"
            # Sequential processing for small batches
            elif output_format == "dict":
                results = [
                    m.model_dump(
                        exclude_unset=cast(
                            "bool", dump_kwargs.get("exclude_unset", False)
                        ),
                        exclude_defaults=cast(
                            "bool", dump_kwargs.get("exclude_defaults", False)
                        ),
                        exclude_none=cast(
                            "bool", dump_kwargs.get("exclude_none", False)
                        ),
                    )
                    for m in models
                ]
            else:
                # Build JSON array efficiently
                json_parts = [
                    m.model_dump_json(
                        exclude_unset=cast(
                            "bool", dump_kwargs.get("exclude_unset", False)
                        ),
                        exclude_defaults=cast(
                            "bool", dump_kwargs.get("exclude_defaults", False)
                        ),
                        exclude_none=cast(
                            "bool", dump_kwargs.get("exclude_none", False)
                        ),
                    )
                    for m in models
                ]
                return f"[{','.join(json_parts)}]"

            return results

    # Schema generation optimizer
    class SchemaOptimizer:
        """Optimized schema generation for models."""

        @staticmethod
        def get_optimized_schema(
            model: type[BaseModel],
            *,
            by_alias: bool = True,
            ref_template: str = "#/$defs/{model}",
            mode: Literal["validation", "serialization"] = "validation",
        ) -> FlextTypes.Dict:
            """Get optimized JSON schema for a model.

            Args:
                model: Pydantic model class
                by_alias: Use field aliases in schema
                ref_template: Template for schema references
                mode: Schema mode

            Returns:
                Optimized JSON schema FlextTypes.Dict

            """
            schema = model.model_json_schema(
                by_alias=by_alias,
                ref_template=ref_template,
                mode=mode,
                # Optimization settings
            )

            # Remove unnecessary schema elements
            if "$defs" in schema and not schema["$defs"]:
                del schema["$defs"]

            # Optimize property definitions
            if "properties" in schema:
                for prop_schema in schema["properties"].values():
                    # Remove redundant type arrays with single type
                    if (
                        "type" in prop_schema
                        and isinstance(prop_schema["type"], list)
                        and len(prop_schema["type"]) == 1
                    ):
                        prop_schema["type"] = prop_schema["type"][0]

                    # Remove empty descriptions
                    if "description" in prop_schema and not prop_schema["description"]:
                        del prop_schema["description"]

            return schema

    # Export control decorator
    # No exclude_from_export decorator needed - use Pydantic 2.11's native features:
    # 1. Field(exclude=True) to exclude fields permanently
    # 2. model_dump(exclude={'field1', 'field2'}) to exclude dynamically
    # 3. Use computed_field with exclude parameter for computed fields

    # Model with automatic aliasing
    class CamelCaseModel(BaseModel):
        """Model that automatically uses camelCase for JSON."""

        model_config = ConfigDict(
            validate_assignment=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            alias_generator=lambda field_name: field_name.split("_")[0]
            + "".join(x.title() for x in field_name.split("_")[1:]),
            populate_by_name=True,
        )

    class KebabCaseModel(BaseModel):
        """Model that automatically uses kebab-case for URLs."""

        model_config = ConfigDict(
            validate_assignment=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            alias_generator=lambda field_name: field_name.replace("_", "-"),
            populate_by_name=True,
        )

    # ============================================================================
    # END OF PHASE 7: SERIALIZATION OPTIMIZATIONS
    # ============================================================================

    # ============================================================================
    # PHASE 8: CQRS CONFIGURATION MODELS
    # ============================================================================

    class ValidationRequest(ArbitraryTypesModel):
        """Validation request model for object validation."""

        obj: object = Field(description="Object to validate")
        validation_type: str = Field(
            default="general", description="Type of validation to perform"
        )
        context: FlextTypes.Dict = Field(
            default_factory=dict, description="Validation context"
        )

    class CqrsConfig:
        """CQRS configuration models for bus and handler setup."""

        class Bus(BaseModel):
            """Bus configuration model for CQRS command bus."""

            enable_middleware: bool = Field(
                default=True, description="Enable middleware pipeline"
            )
            enable_metrics: bool = Field(
                default=True, description="Enable metrics collection"
            )
            enable_caching: bool = Field(
                default=True, description="Enable query result caching"
            )
            execution_timeout: int = Field(
                default=FlextConstants.Defaults.TIMEOUT,
                description="Command execution timeout",
            )
            max_cache_size: int = Field(
                default=100,  # Default batch size,
                description="Maximum cache size",
            )
            implementation_path: str = Field(
                default="flext_core.bus:FlextBus", description="Implementation path"
            )

            @field_validator("implementation_path")
            @classmethod
            def validate_implementation_path(cls, v: str) -> str:
                """Validate implementation path format."""
                if ":" not in v:
                    error_msg = "implementation_path must be in 'module:Class' format"
                    raise FlextExceptions.ValidationError(
                        message=error_msg,
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                return v

            @classmethod
            def create_bus_config(
                cls,
                bus_config: FlextTypes.Dict | None = None,
                *,
                enable_middleware: bool = True,
                enable_metrics: bool | None = None,
                enable_caching: bool = True,
                execution_timeout: int = FlextConstants.Defaults.TIMEOUT,
                max_cache_size: int = FlextConstants.Performance.BatchProcessing.DEFAULT_SIZE,
                implementation_path: str = "flext_core.bus:FlextBus",
            ) -> Self:
                """Create bus configuration with defaults and overrides."""
                # Use global config defaults when not explicitly provided
                if enable_metrics is None:
                    enable_metrics = True

                config_data: FlextTypes.Dict = {
                    "enable_middleware": enable_middleware,
                    "enable_metrics": enable_metrics,
                    "enable_caching": enable_caching,
                    "execution_timeout": execution_timeout,
                    "max_cache_size": max_cache_size,
                    "implementation_path": implementation_path,
                }

                if bus_config:
                    config_data.update(bus_config)

                return cls.model_validate(config_data)

        class Handler(BaseModel):
            """Handler configuration model for CQRS handlers."""

            handler_id: str = Field(description="Unique handler identifier")
            handler_name: str = Field(description="Human-readable handler name")
            handler_type: Literal["command", "query", "event", "saga"] = Field(
                default="command",
                description="Handler type",
            )
            handler_mode: Literal["command", "query", "event", "saga"] = Field(
                default="command",
                description="Handler mode",
            )
            command_timeout: int = Field(
                default=FlextConstants.Cqrs.DEFAULT_COMMAND_TIMEOUT,
                description="Command timeout",
            )
            max_command_retries: int = Field(
                default=FlextConstants.Cqrs.DEFAULT_MAX_COMMAND_RETRIES,
                description="Maximum retry attempts",
            )
            metadata: FlextTypes.Dict = Field(
                default_factory=dict, description="Handler metadata"
            )

            @classmethod
            def create_handler_config(
                cls,
                handler_type: Literal["command", "query", "event", "saga"],
                *,
                default_name: str | None = None,
                default_id: str | None = None,
                handler_config: FlextTypes.Dict | None = None,
                command_timeout: int = 0,
                max_command_retries: int = 0,
            ) -> Self:
                """Create handler configuration with defaults and overrides."""
                handler_mode_value = (
                    FlextConstants.Dispatcher.HANDLER_MODE_COMMAND
                    if handler_type == FlextConstants.Cqrs.COMMAND_HANDLER_TYPE
                    else FlextConstants.Dispatcher.HANDLER_MODE_QUERY
                )
                config_data: FlextTypes.Dict = {
                    "handler_id": default_id
                    or f"{handler_type}_handler_{uuid.uuid4().hex[:8]}",
                    "handler_name": default_name or f"{handler_type.title()} Handler",
                    "handler_type": handler_type,
                    "handler_mode": handler_mode_value,
                    "command_timeout": command_timeout,
                    "max_command_retries": max_command_retries,
                    "metadata": {},
                }

                if handler_config:
                    config_data.update(handler_config)

                return cls.model_validate(config_data)

    # ============================================================================
    # REGISTRATION DETAILS MODEL
    # ============================================================================

    class RegistrationDetails(BaseModel):
        """Registration details for handler registration tracking."""

        registration_id: str = Field(description="Unique registration identifier")
        handler_mode: Literal["command", "query"] = Field(
            default="command",
            description="Handler mode",
        )
        timestamp: str = Field(
            default=FlextConstants.Cqrs.DEFAULT_TIMESTAMP,
            description="Registration timestamp",
        )
        status: Literal["active", "inactive"] = Field(
            default="active",
            description="Registration status",
        )

    # ============================================================================
    # VALIDATION UTILITIES (moved from FlextUtilities to avoid circular imports)
    # ============================================================================

    class Validation:
        """Local validation utilities to avoid circular imports."""

        @staticmethod
        def validate_email_address(email: str) -> FlextResult[str]:
            """Enhanced email validation matching FlextModels.EmailAddress pattern."""
            if not email:
                return FlextResult[str].fail("Email cannot be empty")

            # Use pattern from FlextConstants.Platform.PATTERN_EMAIL
            if "@" not in email or "." not in email.rsplit("@", maxsplit=1)[-1]:
                return FlextResult[str].fail(f"Invalid email format: {email}")

            # Length validation using FlextConstants
            if len(email) > FlextConstants.Validation.MAX_EMAIL_LENGTH:
                return FlextResult[str].fail(
                    f"Email too long (max {FlextConstants.Validation.MAX_EMAIL_LENGTH} chars)"
                )

            return FlextResult[str].ok(email.lower())

        @staticmethod
        def validate_hostname(hostname: str) -> FlextResult[str]:
            """Validate hostname format matching FlextModels.Host pattern."""
            # Trim whitespace first
            hostname = hostname.strip()

            # Check if empty after trimming
            if not hostname:
                return FlextResult[str].fail("Hostname cannot be empty")

            # Basic hostname validation
            if len(hostname) > FlextConstants.Validation.MAX_EMAIL_LENGTH:
                return FlextResult[str].fail("Hostname too long")

            if not all(c.isalnum() or c in ".-" for c in hostname):
                return FlextResult[str].fail("Invalid hostname characters")

            return FlextResult[str].ok(hostname.lower())

        @staticmethod
        def validate_entity_id(entity_id: str) -> FlextResult[str]:
            """Validate entity ID format matching FlextModels.EntityId pattern."""
            # Trim whitespace first
            entity_id = entity_id.strip()

            # Check if empty after trimming
            if not entity_id:
                return FlextResult[str].fail("Entity ID cannot be empty")

            # Allow UUIDs, alphanumeric with dashes/underscores
            if not re.match(r"^[a-zA-Z0-9_-]+$", entity_id):
                return FlextResult[str].fail("Invalid entity ID format")

            return FlextResult[str].ok(entity_id)

        @staticmethod
        def validate_url(url: str) -> FlextResult[str]:
            """Validate URL format with comprehensive checks.

            Centralizes URL validation logic that was previously inline in FlextModels.Url.
            """
            try:
                result: ParseResult = urlparse(url)
                if not all([result.scheme, result.netloc]):
                    return FlextResult[str].fail(f"Invalid URL format: {url}")

                # Validate scheme
                if result.scheme not in {"http", "https", "ftp", "ftps", "file"}:
                    return FlextResult[str].fail(
                        f"Unsupported URL scheme: {result.scheme}"
                    )

                # Validate domain
                if result.netloc:
                    domain = result.netloc.split(":")[0]  # Remove port
                    if (
                        not domain
                        or len(domain) > FlextConstants.Validation.MAX_EMAIL_LENGTH
                    ):
                        return FlextResult[str].fail("Invalid domain in URL")

                    # Check for valid characters
                    if not all(c.isalnum() or c in ".-" for c in domain):
                        return FlextResult[str].fail("Invalid characters in domain")

                return FlextResult[str].ok(url)
            except Exception as e:
                return FlextResult[str].fail(f"URL validation failed: {e}")

        @staticmethod
        def validate_command(command: object) -> FlextResult[None]:
            """Validate a command message using centralized validation patterns.

            Args:
                command: The command to validate

            Returns:
                FlextResult[None]: Success if valid, failure with error details

            """
            if command is None:
                return FlextResult[None].fail(
                    "Command cannot be None",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Check if command has required attributes
            if not hasattr(command, "command_type"):
                return FlextResult[None].fail(
                    "Command must have 'command_type' attribute",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Validate command type
            command_type = getattr(command, "command_type", None)
            if not command_type or not isinstance(command_type, str):
                return FlextResult[None].fail(
                    "Command type must be a non-empty string",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            return FlextResult[None].ok(None)

        @staticmethod
        def validate_service_name(name: str) -> FlextResult[str]:
            """Validate service name format for container registration.

            Args:
                name: Service name to validate

            Returns:
                FlextResult[str]: Success with normalized name or failure with error message

            """
            # Type annotation guarantees name is str, so no isinstance check needed
            normalized = name.strip().lower()
            if not normalized:
                return FlextResult[str].fail("Service name cannot be empty")

            # Additional validation for special characters
            if any(char in normalized for char in [".", "/", "\\"]):
                return FlextResult[str].fail("Service name contains invalid characters")

            return FlextResult[str].ok(normalized)

        @staticmethod
        def validate_query(query: object) -> FlextResult[None]:
            """Validate a query message using centralized validation patterns.

            Args:
                query: The query to validate

            Returns:
                FlextResult[None]: Success if valid, failure with error details

            """
            if query is None:
                return FlextResult[None].fail(
                    "Query cannot be None",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Check if query has required attributes
            if not hasattr(query, "query_type"):
                return FlextResult[None].fail(
                    "Query must have 'query_type' attribute",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Validate query type
            query_type = getattr(query, "query_type", None)
            if not query_type or not isinstance(query_type, str):
                return FlextResult[None].fail(
                    "Query type must be a non-empty string",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            return FlextResult[None].ok(None)

        @staticmethod
        def validate_business_rules[T](
            model: T,
            *rules: Callable[[T], FlextResult[None]],
        ) -> FlextResult[T]:
            """Validate business rules with railway patterns.

            Args:
                model: The model to validate
                *rules: Business rule validation functions

            Returns:
                FlextResult[T]: Validated model or accumulated errors

            Example:
                ```python
                result = FlextModels.Validation.validate_business_rules(
                    user_model,
                    lambda u: validate_age(u),
                    lambda u: validate_email_format(u.email),
                    lambda u: validate_permissions(u.roles),
                )
                ```

            """
            return FlextResult.validate_all(model, *rules)

        @staticmethod
        def validate_cross_fields[T](
            model: T,
            field_validators: dict[str, Callable[[T], FlextResult[None]]],
        ) -> FlextResult[T]:
            """Validate cross-field dependencies with railway patterns.

            Args:
                model: The model to validate
                field_validators: Field name to validator mapping

            Returns:
                FlextResult[T]: Validated model or accumulated errors

            Example:
                ```python
                result = FlextModels.Validation.validate_cross_fields(
                    order_model,
                    {
                        "start_date": lambda o: validate_date_range(
                            o.start_date, o.end_date
                        ),
                        "end_date": lambda o: validate_date_range(
                            o.start_date, o.end_date
                        ),
                        "amount": lambda o: validate_amount_range(o.amount, o.currency),
                    },
                )
                ```

            """
            validation_results = [
                validator(model) for validator in field_validators.values()
            ]

            errors = [
                result.error
                for result in validation_results
                if result.is_failure and result.error
            ]

            if errors:
                return FlextResult[T].fail(
                    f"Cross-field validation failed: {'; '.join(errors)}",
                    error_code="CROSS_FIELD_VALIDATION_FAILED",
                    error_data={"field_errors": errors},
                )

            return FlextResult[T].ok(model)

        @staticmethod
        def validate_performance[T: BaseModel](
            model: T,
            max_validation_time_ms: int | None = None,
        ) -> FlextResult[T]:
            """Validate model with performance constraints.

            Args:
                model: The model to validate
                max_validation_time_ms: Maximum validation time in milliseconds

            Returns:
                FlextResult[T]: Validated model or performance error

            Example:
                ```python
                result = FlextModels.Validation.validate_performance(
                    complex_model, max_validation_time_ms=50
                )
                ```

            """
            # Use config value if not provided
            timeout_ms = (
                max_validation_time_ms
                if max_validation_time_ms is not None
                else _get_config().validation_timeout_ms
            )
            start_time = time_module.time()

            try:
                validated_model = model.__class__.model_validate(model.model_dump())
                validation_time = (time_module.time() - start_time) * 1000

                if validation_time > timeout_ms:
                    return FlextResult[T].fail(
                        f"Validation too slow: {validation_time:.2f}ms > {timeout_ms}ms",
                        error_code="PERFORMANCE_VALIDATION_FAILED",
                        error_data={"validation_time_ms": validation_time},
                    )

                return FlextResult[T].ok(validated_model)
            except Exception as e:
                return FlextResult[T].fail(
                    f"Validation failed: {e}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

        @staticmethod
        def validate_batch[T](
            models: list[T],
            *validators: Callable[[T], FlextResult[None]],
            fail_fast: bool = True,
        ) -> FlextResult[list[T]]:
            """Validate a batch of models with railway patterns.

            Args:
                models: List of models to validate
                *validators: Validation functions to apply
                fail_fast: Stop on first failure or accumulate all errors

            Returns:
                FlextResult[list[T]]: All validated models or first failure

            Example:
                ```python
                result = FlextModels.Validation.validate_batch(
                    user_models,
                    lambda u: validate_email(u.email),
                    lambda u: validate_age(u.age),
                    fail_fast=False,
                )
                ```

            """
            if fail_fast:
                return FlextResult.traverse(
                    models,
                    lambda model: FlextResult.validate_all(model, *validators).map(
                        lambda _: model
                    ),
                )
            # Accumulate all errors
            validated_models = []
            all_errors = []

            for model in models:
                validation_result = FlextResult.validate_all(model, *validators)
                if validation_result.is_success:
                    validated_models.append(model)
                else:
                    all_errors.append(validation_result.error or "Validation failed")

            if all_errors:
                return FlextResult[list[T]].fail(
                    f"Batch validation failed: {'; '.join(all_errors)}",
                    error_code="BATCH_VALIDATION_FAILED",
                    error_data={"error_count": len(all_errors), "errors": all_errors},
                )

            return FlextResult[list[T]].ok(validated_models)

        @staticmethod
        def validate_domain_invariants[T](
            model: T,
            invariants: list[Callable[[T], FlextResult[None]]],
        ) -> FlextResult[T]:
            """Validate domain invariants with railway patterns.

            Args:
                model: The model to validate
                invariants: List of domain invariant validation functions

            Returns:
                FlextResult[T]: Validated model or first invariant violation

            Example:
                ```python
                result = FlextModels.Validation.validate_domain_invariants(
                    order_model,
                    [
                        lambda o: validate_order_total(o),
                        lambda o: validate_order_items(o),
                        lambda o: validate_order_customer(o),
                    ],
                )
                ```

            """
            for invariant in invariants:
                result = invariant(model)
                if result.is_failure:
                    return FlextResult[T].fail(
                        f"Domain invariant violation: {result.error}",
                        error_code="DOMAIN_INVARIANT_VIOLATION",
                        error_data={"invariant_error": result.error},
                    )
            return FlextResult[T].ok(model)

        @staticmethod
        def validate_aggregate_consistency[T](
            aggregate: T,
            consistency_rules: dict[str, Callable[[T], FlextResult[None]]],
        ) -> FlextResult[T]:
            """Validate aggregate consistency with railway patterns.

            Args:
                aggregate: The aggregate to validate
                consistency_rules: Dictionary of consistency rule validators

            Returns:
                FlextResult[T]: Validated aggregate or consistency violation

            Example:
                ```python
                result = FlextModels.Validation.validate_aggregate_consistency(
                    order_aggregate,
                    {
                        "total_consistency": lambda a: validate_total_consistency(a),
                        "item_consistency": lambda a: validate_item_consistency(a),
                        "customer_consistency": lambda a: validate_customer_consistency(
                            a
                        ),
                    },
                )
                ```

            """
            violations = []
            for rule_name, validator in consistency_rules.items():
                result = validator(aggregate)
                if result.is_failure:
                    violations.append(f"{rule_name}: {result.error}")

            if violations:
                return FlextResult[T].fail(
                    f"Aggregate consistency violations: {'; '.join(violations)}",
                    error_code="AGGREGATE_CONSISTENCY_VIOLATION",
                    error_data={"violations": violations},
                )

            return FlextResult[T].ok(aggregate)

        @staticmethod
        def validate_event_sourcing[T](
            event: T,
            event_validators: dict[str, Callable[[T], FlextResult[None]]],
        ) -> FlextResult[T]:
            """Validate event sourcing patterns with railway patterns.

            Args:
                event: The domain event to validate
                event_validators: Dictionary of event-specific validators

            Returns:
                FlextResult[T]: Validated event or validation failure

            Example:
                ```python
                result = FlextModels.Validation.validate_event_sourcing(
                    order_created_event,
                    {
                        "event_type": lambda e: validate_event_type(e),
                        "event_data": lambda e: validate_event_data(e),
                        "event_metadata": lambda e: validate_event_metadata(e),
                    },
                )
                ```

            """
            validation_results = [
                validator(event) for validator in event_validators.values()
            ]

            errors = [
                result.error
                for result in validation_results
                if result.is_failure and result.error
            ]

            if errors:
                return FlextResult[T].fail(
                    f"Event validation failed: {'; '.join(errors)}",
                    error_code="EVENT_VALIDATION_FAILED",
                    error_data={"event_errors": errors},
                )

            return FlextResult[T].ok(event)

        @staticmethod
        def validate_cqrs_patterns[T](
            command_or_query: T,
            pattern_type: str,
            validators: list[Callable[[T], FlextResult[None]]],
        ) -> FlextResult[T]:
            """Validate CQRS patterns with railway patterns.

            Args:
                command_or_query: The command or query to validate
                pattern_type: Type of pattern ("command" or "query")
                validators: List of pattern-specific validators

            Returns:
                FlextResult[T]: Validated command/query or validation failure

            Example:
                ```python
                result = FlextModels.Validation.validate_cqrs_patterns(
                    create_order_command,
                    "command",
                    [
                        lambda c: validate_command_structure(c),
                        lambda c: validate_command_data(c),
                        lambda c: validate_command_permissions(c),
                    ],
                )
                ```

            """
            if pattern_type not in {"command", "query"}:
                return FlextResult[T].fail(
                    f"Invalid pattern type: {pattern_type}. Must be 'command' or 'query'",
                    error_code="INVALID_PATTERN_TYPE",
                )

            for validator in validators:
                result = validator(command_or_query)
                if result.is_failure:
                    return FlextResult[T].fail(
                        f"CQRS {pattern_type} validation failed: {result.error}",
                        error_code=f"CQRS_{pattern_type.upper()}_VALIDATION_FAILED",
                        error_data={
                            "pattern_type": pattern_type,
                            "error": result.error,
                        },
                    )

            return FlextResult[T].ok(command_or_query)

    # ============================================================================
    # END OF PHASE 8: CQRS CONFIGURATION MODELS
    # ============================================================================

    # ============================================================================
    # PHASE 9: QUERY AND PAGINATION MODELS
    # ============================================================================

    class Pagination(BaseModel):
        """Pagination model for query results with Pydantic 2 computed fields."""

        page: int = Field(
            default=FlextConstants.Pagination.DEFAULT_PAGE_NUMBER,
            ge=1,
            description="Page number (1-based)",
        )
        size: int = Field(
            default=FlextConstants.Pagination.DEFAULT_PAGE_SIZE,
            ge=1,
            le=1000,
            description="Page size",
        )

        @computed_field
        def offset(self) -> int:
            """Calculate offset from page and size - computed field."""
            return (self.page - 1) * self.size

        @computed_field
        def limit(self) -> int:
            """Get limit (same as size) - computed field."""
            return self.size

        def to_dict(self) -> FlextTypes.Dict:
            """Convert pagination to dictionary."""
            return {
                "page": self.page,
                "size": self.size,
                "offset": self.offset,
                "limit": self.limit,
            }

    class Query(BaseModel):
        """Query model for CQRS query operations."""

        filters: FlextTypes.Dict = Field(
            default_factory=dict, description="Query filters"
        )
        pagination: FlextModels.Pagination | dict[str, int] = Field(
            default_factory=dict,
            description="Pagination settings",
        )
        query_id: str = Field(
            default_factory=lambda: str(uuid.uuid4()), description="Unique query ID"
        )
        query_type: str | None = Field(default=None, description="Type of query")

        @field_validator("pagination", mode="before")
        @classmethod
        def validate_pagination(cls, v: object) -> FlextModels.Pagination:
            """Convert pagination to Pagination instance."""
            if isinstance(v, dict):
                # Extract page and size from dict
                page = v.get("page", 1)
                size = v.get("size", 20)

                # Convert to int if string
                if isinstance(page, str):
                    try:
                        page = int(page)
                    except ValueError:
                        page = 1
                if isinstance(size, str):
                    try:
                        size = int(size)
                    except ValueError:
                        size = 20

                return FlextModels.Pagination(page=page, size=size)
            if isinstance(v, FlextModels.Pagination):
                return v
            # Default pagination
            return FlextModels.Pagination()

        @classmethod
        def validate_query(
            cls, query_payload: FlextTypes.Dict
        ) -> FlextResult[FlextModels.Query]:
            """Validate and create Query from payload."""
            try:
                # Extract the required fields with proper typing
                filters = query_payload.get("filters", {})
                pagination_data = query_payload.get("pagination", {})
                pagination = (
                    pagination_data
                    if isinstance(pagination_data, FlextModels.Pagination)
                    else FlextModels.Pagination(**pagination_data)
                    if isinstance(pagination_data, dict)
                    else FlextModels.Pagination()
                )
                query_id = str(query_payload.get("query_id", str(uuid.uuid4())))
                query_type = query_payload.get("query_type")

                if not isinstance(filters, dict):
                    filters = {}
                # No need to validate pagination dict - Pydantic validator handles conversion

                query = cls(
                    filters=filters,
                    pagination=pagination,  # Pydantic will convert dict to Pagination
                    query_id=query_id,
                    query_type=str(query_type) if query_type is not None else None,
                )
                return FlextResult[FlextModels.Query].ok(query)
            except Exception as e:
                return FlextResult[FlextModels.Query].fail(
                    f"Query validation failed: {e}"
                )

    class OptimizedQuery(Query, SerializableModel):
        """FlextModels.Query with flexible serialization options."""

        def to_query_string(self) -> str:
            """Convert query to URL query string format."""
            params: dict[str, str | int] = {}

            # Add pagination with type guard
            if self.pagination:
                if isinstance(self.pagination, FlextModels.Pagination):
                    params["page"] = self.pagination.page
                    params["size"] = self.pagination.size
                elif isinstance(self.pagination, dict):
                    params["page"] = self.pagination.get("page", 1)
                    params["size"] = self.pagination.get("size", 10)

            # Add filters
            for key, value in self.filters.items():
                if isinstance(value, (list, tuple)):
                    params[f"filter_{key}"] = ",".join(str(v) for v in value)
                else:
                    params[f"filter_{key}"] = str(value)

            return urlencode(params)

    # Factory methods for Pydantic Field default_factory compliance
    @staticmethod
    def _default_pagination() -> Pagination:
        """Factory method for default Pagination instance."""
        return FlextModels.Pagination()

    @staticmethod
    def _default_uuid_str() -> str:
        """Factory method for default UUID string."""
        return str(uuid.uuid4())

    # ============================================================================
    # HTTP PROTOCOL MODELS - Foundation for flext-api and flext-web
    # ============================================================================

    class HttpRequest(Command):
        """Base HTTP request model - foundation for client and server.

        Shared HTTP request primitive for flext-api (HTTP client) and flext-web (web server).
        Provides common request validation, method checking, and URL validation.

        USAGE: Foundation for HTTP client requests and web server request handling.
        EXTENDS: FlextModels.Command (represents a request action/command)

        Usage example:
            from flext_core import FlextModels, FlextResult

            # Create HTTP request
            request = FlextModels.HttpRequest(
                url="https://api.example.com/users",
                method="GET",
                headers={"Authorization": "Bearer token"},
                timeout=30.0
            )

            # Validate method
            if request.method in {"GET", "HEAD", "OPTIONS"}:
                print("Safe HTTP method")

            # Domain libraries extend this:
            # flext-api: Adds client-specific fields (full_url, request_size)
            # flext-web: Adds server-specific fields (request_id, client_ip, user_agent)
        """

        url: str = Field(description="Request URL")
        method: str = Field(default="GET", description="HTTP method")
        headers: dict[str, str] = Field(
            default_factory=dict, description="Request headers"
        )
        body: str | dict | None = Field(default=None, description="Request body")
        timeout: float = Field(
            default_factory=lambda: int(_get_config().timeout_seconds),
            ge=0.0,
            le=300.0,
            description="Request timeout in seconds",
        )

        @computed_field
        def has_body(self) -> bool:
            """Check if request has a body."""
            return self.body is not None

        @computed_field
        def is_secure(self) -> bool:
            """Check if request uses HTTPS."""
            return self.url.startswith("https://")

        @field_validator("method")
        @classmethod
        def validate_method(cls, v: str) -> str:
            """Validate HTTP method using centralized constants."""
            method_upper = v.upper()
            valid_methods = {
                "GET",
                "POST",
                "PUT",
                "DELETE",
                "PATCH",
                "HEAD",
                "OPTIONS",
            }
            if method_upper not in valid_methods:
                error_msg = f"Invalid HTTP method: {v}. Valid methods: {valid_methods}"
                raise FlextExceptions.ValidationError(
                    error_msg,
                    field="method",
                    value=v,
                )
            return method_upper

        @field_validator("url")
        @classmethod
        def validate_url(cls, v: str) -> str:
            """Validate URL format using centralized validation."""
            if not v or not v.strip():
                error_msg = "URL cannot be empty"
                raise FlextExceptions.ValidationError(
                    error_msg,
                    field="url",
                    value=v,
                )

            # Allow relative URLs (starting with /)
            if v.strip().startswith("/"):
                return v.strip()

            # Validate absolute URLs with Pydantic 2 direct validation
            parsed = urlparse(v.strip())
            if not parsed.scheme or not parsed.netloc:
                error_msg = "URL must have scheme and domain"
                raise FlextExceptions.ValidationError(error_msg, field="url", value=v)

            if parsed.scheme not in {"http", "https"}:
                error_msg = "URL must start with http:// or https://"
                raise FlextExceptions.ValidationError(error_msg, field="url", value=v)

            return v.strip()

        @model_validator(mode="after")
        def validate_request_consistency(self) -> Self:
            """Cross-field validation for HTTP request consistency."""
            # Methods without body should not have a body
            methods_without_body = {
                "GET",
                "HEAD",
                "DELETE",
            }
            if self.method in methods_without_body and self.body is not None:
                error_msg = f"HTTP {self.method} requests should not have a body"
                raise FlextExceptions.ValidationError(
                    error_msg,
                    field="body",
                    validation_details=f"Method {self.method} should not have body",
                )

            # Methods with body should have Content-Type header
            if self.method in {"POST", "PUT", "PATCH"} and self.body:
                headers_lower = {k.lower(): v for k, v in self.headers.items()}
                if "content-type" not in headers_lower:
                    # Auto-add Content-Type based on body type
                    if isinstance(self.body, dict):
                        self.headers[FlextConstants.Http.CONTENT_TYPE_HEADER] = (
                            FlextConstants.Http.ContentType.JSON
                        )
                    elif isinstance(self.body, str):
                        self.headers["Content-Type"] = "text/plain"

            return self

    class HttpResponse(Entity):
        """Base HTTP response model - foundation for client and server.

        Shared HTTP response primitive for flext-api (HTTP client) and flext-web (web server).
        Provides common status code validation, success/error checking, and response metadata.

        USAGE: Foundation for HTTP client responses and web server response handling.
        EXTENDS: FlextModels.Entity (represents a response entity with ID and timestamps)

        Usage example:
            from flext_core import FlextModels, FlextConstants

            # Create HTTP response
            response = FlextModels.HttpResponse(
                status_code=200,
                headers={"Content-Type": "application/json"},
                body={"result": "success"},
                elapsed_time=0.123
            )

            # Check response status
            if response.is_success:
                print("Success response")
            elif response.is_client_error:
                print("Client error")

            # Domain libraries extend this:
            # flext-api: Adds client-specific fields (url, method, domain_events)
            # flext-web: Adds server-specific fields (response_id, request_id, content_type)
        """

        status_code: int = Field(
            ge=FlextConstants.Http.HTTP_STATUS_MIN,
            le=FlextConstants.Http.HTTP_STATUS_MAX,
            description="HTTP status code",
        )
        headers: dict[str, str] = Field(
            default_factory=dict, description="Response headers"
        )
        body: str | dict | None = Field(default=None, description="Response body")
        elapsed_time: float | None = Field(
            default=None, ge=0.0, description="Request/response elapsed time in seconds"
        )

        @computed_field
        def is_success(self) -> bool:
            """Check if response indicates success (2xx status codes)."""
            return (
                FlextConstants.Http.HTTP_SUCCESS_MIN
                <= self.status_code
                <= FlextConstants.Http.HTTP_SUCCESS_MAX
            )

        @computed_field
        def is_client_error(self) -> bool:
            """Check if response indicates client error (4xx status codes)."""
            return (
                FlextConstants.Http.HTTP_CLIENT_ERROR_MIN
                <= self.status_code
                <= FlextConstants.Http.HTTP_CLIENT_ERROR_MAX
            )

        @computed_field
        def is_server_error(self) -> bool:
            """Check if response indicates server error (5xx status codes)."""
            return (
                FlextConstants.Http.HTTP_SERVER_ERROR_MIN
                <= self.status_code
                <= FlextConstants.Http.HTTP_SERVER_ERROR_MAX
            )

        @computed_field
        def is_redirect(self) -> bool:
            """Check if response indicates redirect (3xx status codes)."""
            return (
                FlextConstants.Http.HTTP_REDIRECTION_MIN
                <= self.status_code
                <= FlextConstants.Http.HTTP_REDIRECTION_MAX
            )

        @computed_field
        def is_informational(self) -> bool:
            """Check if response is informational (1xx status codes)."""
            return (
                FlextConstants.Http.HTTP_INFORMATIONAL_MIN
                <= self.status_code
                <= FlextConstants.Http.HTTP_INFORMATIONAL_MAX
            )

        @field_validator("status_code")
        @classmethod
        def validate_status_code(cls, v: int) -> int:
            """Validate HTTP status code using centralized constants."""
            if not (
                FlextConstants.Http.HTTP_STATUS_MIN
                <= v
                <= FlextConstants.Http.HTTP_STATUS_MAX
            ):
                error_msg = (
                    f"Invalid HTTP status code: {v}. "
                    f"Must be between {FlextConstants.Http.HTTP_STATUS_MIN} and "
                    f"{FlextConstants.Http.HTTP_STATUS_MAX}"
                )
                raise FlextExceptions.ValidationError(
                    error_msg,
                    field="status_code",
                    value=v,
                )
            return v

        @model_validator(mode="after")
        def validate_response_consistency(self) -> Self:
            """Cross-field validation for HTTP response consistency."""
            # 204 No Content should not have a body
            if (
                self.status_code == FlextConstants.Http.HTTP_NO_CONTENT
                and self.body is not None
            ):
                error_msg = "HTTP 204 No Content responses should not have a body"
                raise FlextExceptions.ValidationError(
                    error_msg,
                    field="body",
                    validation_details="Status 204 No Content should not have body",
                )

            # Validate elapsed time
            if self.elapsed_time is not None and self.elapsed_time < 0:
                error_msg = "Elapsed time cannot be negative"
                raise FlextExceptions.ValidationError(
                    error_msg,
                    field="elapsed_time",
                    value=self.elapsed_time,
                )

            return self

    # ============================================================================
    # END OF FLEXT MODELS - Foundation library complete
    # ============================================================================

    class DomainServiceRetryConfig(RetryConfiguration):
        """Domain service retry configuration."""

        max_attempts: int = 1
        retry_delay: float = 0.0
        backoff_multiplier: float = 1.0
        exponential_backoff: bool = False


__all__ = [
    "FlextModels",
    # ModelDumpKwargs is now FlextModels.ModelDumpKwargs
    # OperationCallable is now FlextProtocols.Foundation.OperationCallable
]
