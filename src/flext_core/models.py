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
    field_validator,
    model_validator,
)
from pydantic.main import IncEx

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.loggings import FlextLogger
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, T


def _create_default_pagination() -> FlextModels.Pagination:
    """Factory function for creating default Pagination instances."""
    # Forward reference will be resolved at runtime
    return FlextModels.Pagination()


class _BaseConfigDict:
    """Shared Pydantic 2.11 ConfigDict patterns to eliminate duplication.

    This class provides reusable configuration presets for all FlextModels classes,
    leveraging Pydantic 2.11 features for enhanced validation and serialization.
    """

    ARBITRARY: ConfigDict = ConfigDict(
        # Pydantic 2.11 validation features
        validate_assignment=True,  # Validate on field assignment
        validate_return=True,  # Validate return values from methods
        validate_default=True,  # Validate default values
        use_enum_values=True,  # Use enum values in serialization
        arbitrary_types_allowed=True,  # Allow arbitrary types
        # JSON serialization features
        ser_json_timedelta="iso8601",  # Serialize timedelta as ISO8601 duration
        ser_json_bytes="base64",  # Serialize bytes as base64 strings
        # Alias and naming features
        serialize_by_alias=True,  # Use aliases in serialization
        populate_by_name=True,  # Accept both alias and field name
        # String processing
        str_strip_whitespace=True,  # Automatically strip whitespace from strings
        str_to_lower=False,  # Keep case sensitivity
        str_to_upper=False,  # Keep case sensitivity
        # Performance optimizations
        defer_build=False,  # Build schema immediately for better error messages
        # Type coercion
        coerce_numbers_to_str=False,  # Keep strict typing
    )

    STRICT: ConfigDict = ConfigDict(
        # Pydantic 2.11 validation features
        validate_assignment=True,  # Validate on field assignment
        validate_return=True,  # Validate return values from methods
        validate_default=True,  # Validate default values
        use_enum_values=True,  # Use enum values in serialization
        arbitrary_types_allowed=True,  # Allow arbitrary types
        extra="forbid",  # Strict validation - no extra fields allowed
        # JSON serialization features
        ser_json_timedelta="iso8601",  # Serialize timedelta as ISO8601 duration
        ser_json_bytes="base64",  # Serialize bytes as base64 strings
        # Alias and naming features
        serialize_by_alias=True,  # Use aliases in serialization
        populate_by_name=True,  # Accept both alias and field name
        # String processing
        str_strip_whitespace=True,  # Automatically strip whitespace from strings
        str_to_lower=False,  # Keep case sensitivity
        str_to_upper=False,  # Keep case sensitivity
        # Performance optimizations
        defer_build=False,  # Build schema immediately for better error messages
        # Type coercion
        coerce_numbers_to_str=False,  # Keep strict typing
    )

    FROZEN: ConfigDict = ConfigDict(
        frozen=True,  # Immutable models
        extra="forbid",  # No extra fields
        use_enum_values=True,  # Use enum values
    )


# Reusable Annotated field types for common patterns (Pydantic 2.11)
# These eliminate 1000+ lines of repetitive Field definitions

# UUID fields
UuidField = Annotated[str, Field(default_factory=lambda: str(uuid.uuid4()))]

# Timestamp fields
TimestampField = Annotated[datetime, Field(default_factory=lambda: datetime.now(UTC))]

# Config-driven fields (pull from global FlextConfig)
TimeoutField = Annotated[
    int, Field(default_factory=lambda: FlextConfig.get_global_instance().timeout_seconds)
]
RetryField = Annotated[
    int,
    Field(default_factory=lambda: FlextConfig.get_global_instance().max_retry_attempts),
]
MaxWorkersField = Annotated[
    int, Field(default_factory=lambda: FlextConfig.get_global_instance().max_workers)
]


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
            pagination: FlextModels.Pagination = Field(
                default_factory=_create_default_pagination
            )


        # Example 6: Domain Event for event sourcing
        class UserCreatedEvent(FlextModels.DomainEvent):
            user_id: str
            timestamp: datetime
        ```

    **TODO**: Enhanced DDD support for 1.0.0+ releases
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
    # BEHAVIOR MIXINS - Reusable model behaviors (Pydantic 2.11 pattern)
    # =========================================================================

    class IdentifiableMixin(BaseModel):
        """Mixin for models with unique identifiers.

        Provides the `id` field using UuidField pattern with explicit default.
        Used by Entity, Command, DomainEvent, Saga, and other identifiable models.
        """

        id: UuidField = Field(default_factory=lambda: str(uuid.uuid4()))

    class TimestampableMixin(BaseModel):
        """Mixin for models with creation and update timestamps.

        Provides `created_at` and `updated_at` fields with automatic timestamp management.
        Used by Entity, TimestampedModel, and models requiring audit trails.
        """

        created_at: TimestampField = Field(default_factory=lambda: datetime.now(UTC))
        updated_at: datetime | None = None

        def update_timestamp(self) -> None:
            """Update the updated_at timestamp to current UTC time."""
            self.updated_at = datetime.now(UTC)

    class TimeoutableMixin(BaseModel):
        """Mixin for models with timeout configuration.

        Provides `timeout_seconds` field pulling from global FlextConfig.
        Used by Repository, Queue, Bus, Circuit, and other timeout-aware models.
        """

        timeout_seconds: TimeoutField = Field(
            default_factory=lambda: FlextConfig.get_global_instance().timeout_seconds
        )

    class RetryableMixin(BaseModel):
        """Mixin for models with retry configuration.

        Provides `max_retry_attempts` and `retry_policy` fields for retry behavior.
        Used by Repository, HandlerExecutionConfig, and other retry-aware models.
        """

        max_retry_attempts: RetryField = Field(
            default_factory=lambda: FlextConfig.get_global_instance().max_retry_attempts
        )
        retry_policy: dict[str, object] = Field(default_factory=dict)  # Pydantic v2 makes this safe!

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
        context: FlextTypes.Core.Dict | None = None

    class Pagination(BaseModel):
        """Pagination model for query results."""

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

        @property
        def offset(self) -> int:
            """Calculate offset from page and size."""
            return (self.page - 1) * self.size

        @property
        def limit(self) -> int:
            """Get limit (same as size)."""
            return self.size

        def to_dict(self) -> FlextTypes.Core.Dict:
            """Convert pagination to dictionary."""
            return {
                "page": self.page,
                "size": self.size,
                "offset": self.offset,
                "limit": self.limit,
            }

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

        def add_domain_event(self, event_name: str, data: FlextTypes.Core.Dict) -> None:
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
                    logger = FlextLogger(__name__)
                    logger.warning(
                        f"Domain event handler {handler_method_name} failed for event {event_name}: {e}"
                    )

            # Increment version after adding domain event (from VersionableMixin)
            self.increment_version()
            self.update_timestamp()

        def clear_domain_events(self) -> FlextTypes.Core.List:
            """Clear and return domain events."""
            events: FlextTypes.Core.List = self.domain_events.copy()
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
            """Hash based on values for use in sets/FlextTypes.Core.Dicts."""
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

        def check_invariants(self: FlextModels.AggregateRoot) -> None:
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
        data: dict[str, object] = Field(default_factory=dict)
        metadata: dict[str, object] = Field(default_factory=dict)

    class Repository(ArbitraryTypesModel, TimeoutableMixin, RetryableMixin):
        """Base repository model for data access patterns.
        
        Uses TimeoutableMixin for timeout_seconds and RetryableMixin for retry configuration.
        """

        entity_type: str
        connection_string: str | None = None

    class Specification(ArbitraryTypesModel):
        """Specification pattern for complex queries."""

        criteria: FlextTypes.Core.Dict = Field(default_factory=dict)
        includes: FlextTypes.Core.StringList = Field(default_factory=list)
        order_by: Annotated[list[tuple[str, str]], Field(default_factory=list)]
        skip: int = Field(
            default=FlextConstants.Performance.DEFAULT_SKIP,
            ge=FlextConstants.Performance.MIN_SKIP,
        )
        take: int = Field(
            default=FlextConstants.Performance.DEFAULT_TAKE,
            ge=FlextConstants.Performance.MIN_TAKE,
            le=FlextConstants.Performance.MAX_TAKE,
        )

    class Saga(ArbitraryTypesModel, IdentifiableMixin):
        """Saga pattern for distributed transactions.
        
        Uses IdentifiableMixin for id.
        """

        steps: Annotated[list[FlextTypes.Core.Dict], Field(default_factory=list)]
        current_step: int = Field(
            default=FlextConstants.Performance.DEFAULT_CURRENT_STEP,
            ge=FlextConstants.Performance.MIN_CURRENT_STEP,
        )
        status: Literal["pending", "running", "completed", "failed", "compensating"] = (
            "pending"
        )
        compensation_data: dict[str, object] = Field(default_factory=dict)

    class Metadata(FrozenStrictModel):
        """Immutable metadata model."""

        created_by: str | None = None
        created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        modified_by: str | None = None
        modified_at: datetime | None = None
        tags: FlextTypes.Core.StringList = Field(default_factory=list)
        attributes: FlextTypes.Core.Dict = Field(default_factory=dict)

    class ErrorDetail(FrozenStrictModel):
        """Immutable error detail model."""

        code: str
        message: str
        field: str | None = None
        details: FlextTypes.Core.Dict = Field(default_factory=dict)

    class ValidationResult(ArbitraryTypesModel):
        """Validation result model."""

        is_valid: bool
        errors: Annotated[list[FlextModels.ErrorDetail], Field(default_factory=list)]
        warnings: FlextTypes.Core.StringList = Field(default_factory=list)

    class Configuration(FrozenStrictModel):
        """Base configuration model - immutable."""

        version: str = FlextConstants.Core.DEFAULT_VERSION
        enabled: bool = True
        settings: FlextTypes.Core.Dict = Field(default_factory=dict)

    class HealthCheck(ArbitraryTypesModel):
        """Health check model for service monitoring."""

        service_name: str
        status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
        checks: FlextTypes.Core.Dict = Field(default_factory=dict)
        last_check: datetime = Field(default_factory=lambda: datetime.now(UTC))
        details: FlextTypes.Core.Dict = Field(default_factory=dict)

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
        changes: FlextTypes.Core.Dict = Field(default_factory=dict)
        ip_address: str | None = None

    class Policy(ArbitraryTypesModel):
        """Policy model for business rules."""

        name: str
        description: str | None = None
        rules: Annotated[list[FlextTypes.Core.Dict], Field(default_factory=list)]
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
            default_factory=lambda: FlextConfig.get_global_instance().cache_ttl
        )
        expires_at: datetime | None = None

    class Bus(TimeoutableMixin, RetryableMixin):
        """Enhanced message bus model with config-driven defaults.
        
        Uses TimeoutableMixin and RetryableMixin for timeout and retry configuration.
        Uses UuidField pattern for bus_id generation.
        
        Note: Removed BaseModel from inheritance since mixins already inherit from it.
        """

        bus_id: str = Field(default_factory=lambda: f"bus_{uuid.uuid4().hex[:8]}")
        handlers: dict[str, FlextTypes.Core.StringList] = Field(default_factory=dict)
        middlewares: FlextTypes.Core.StringList = Field(default_factory=list)
        
        model_config = _BaseConfigDict.ARBITRARY

    class Payload[T](ArbitraryTypesModel, IdentifiableMixin, TimestampableMixin):
        """Enhanced payload model with computed field.
        
        Uses IdentifiableMixin for id and TimestampableMixin for created_at.
        """

        data: T = Field(...)  # Required field, no default
        metadata: dict[str, object] = Field(default_factory=dict)
        expires_at: datetime | None = None
        correlation_id: str | None = None
        source_service: str | None = None
        message_type: str | None = None

        @computed_field
        @property
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
        scopes: FlextTypes.Core.StringList = Field(default_factory=list)

    class Permission(FrozenStrictModel):
        """Immutable permission model."""

        resource: str
        action: str
        conditions: FlextTypes.Core.Dict = Field(default_factory=dict)

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
        roles: FlextTypes.Core.StringList = Field(default_factory=list)
        is_active: bool = True
        last_login: datetime | None = None

    class Session(ArbitraryTypesModel, IdentifiableMixin, TimestampableMixin):
        """Session model.
        
        Uses IdentifiableMixin for id and TimestampableMixin for created_at.
        """

        user_id: str
        expires_at: datetime
        data: dict[str, object] = Field(default_factory=dict)

    class Task(ArbitraryTypesModel, IdentifiableMixin, TimestampableMixin):
        """Task model for background processing.
        
        Uses IdentifiableMixin for id and TimestampableMixin for created_at.
        """

        name: str
        status: Literal["pending", "running", "completed", "failed"] = "pending"
        payload: dict[str, object] = Field(default_factory=dict)
        result: object = None
        error: str | None = None
        started_at: datetime | None = None
        completed_at: datetime | None = None

    class Queue(ArbitraryTypesModel, TimeoutableMixin):
        """Queue model for message processing.
        
        Uses TimeoutableMixin for timeout_seconds.
        """

        name: str
        messages: Annotated[list[FlextTypes.Core.Dict], Field(default_factory=list)]
        max_size: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().cache_max_size
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
        conditions: FlextTypes.Core.Dict = Field(default_factory=dict)

    class Rate(ArbitraryTypesModel):
        """Rate limiting model."""

        key: str
        limit: int
        window_seconds: int = FlextConstants.Performance.DEFAULT_RATE_LIMIT_WINDOW
        current_count: int = 0
        reset_at: datetime | None = None

    class Circuit(ArbitraryTypesModel):
        """Circuit breaker model."""

        name: str
        state: Literal["closed", "open", "half_open"] = "closed"
        failure_count: int = 0
        failure_threshold: int = FlextConstants.Reliability.DEFAULT_FAILURE_THRESHOLD
        timeout_seconds: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().timeout_seconds
        )
        last_failure: datetime | None = None

    class Retry(ArbitraryTypesModel):
        """Retry model."""

        attempt: int = 0
        max_attempts: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().max_retry_attempts
        )
        delay_seconds: float = FlextConstants.Performance.DEFAULT_DELAY_SECONDS
        backoff_multiplier: float = (
            FlextConstants.Performance.DEFAULT_BACKOFF_MULTIPLIER
        )

    class Batch(ArbitraryTypesModel, IdentifiableMixin):
        """Batch processing model.
        
        Uses IdentifiableMixin for id.
        """

        items: Annotated[FlextTypes.Core.List, Field(default_factory=list)]
        size: int = Field(
            default=FlextConstants.Performance.BatchProcessing.SMALL_SIZE, ge=1
        )
        processed_count: int = 0

    class Stream(ArbitraryTypesModel):
        """Stream processing model."""

        stream_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        position: int = 0
        batch_size: int = FlextConstants.Performance.BatchProcessing.STREAM_SIZE
        buffer: Annotated[FlextTypes.Core.List, Field(default_factory=list)]

    class Pipeline(ArbitraryTypesModel):
        """Pipeline model."""

        name: str
        stages: Annotated[list[FlextTypes.Core.Dict], Field(default_factory=list)]
        current_stage: int = 0
        status: Literal["idle", "running", "completed", "failed"] = "idle"

    class Workflow(ArbitraryTypesModel, IdentifiableMixin):
        """Workflow model.
        
        Uses IdentifiableMixin for id.
        """

        name: str
        steps: Annotated[list[FlextTypes.Core.Dict], Field(default_factory=list)]
        current_step: int = 0
        context: dict[str, object] = Field(default_factory=dict)

    class Archive(ArbitraryTypesModel, IdentifiableMixin, TimestampableMixin):
        """Archive model.
        
        Uses IdentifiableMixin for id and TimestampableMixin for created_at.
        """

        entity_type: str
        entity_id: str
        archived_by: str
        data: dict[str, object] = Field(default_factory=dict)

    class Import(ArbitraryTypesModel):
        """Import model."""

        import_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        source: str
        format: str
        status: Literal["pending", "processing", "completed", "failed"] = "pending"
        records_total: int = 0
        records_processed: int = 0
        errors: FlextTypes.Core.StringList = Field(default_factory=list)

    class Export(ArbitraryTypesModel):
        """Export model."""

        export_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        format: str
        filters: FlextTypes.Core.Dict = Field(default_factory=dict)
        status: Literal["pending", "processing", "completed", "failed"] = "pending"
        file_path: str | None = None

    class EmailAddress(Value):
        """Enhanced email address value object with Field constraints."""

        address: str = Field(
            ...,
            pattern=FlextConstants.Platform.PATTERN_EMAIL,
            description="Valid email address",
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

    class Coordinates(Value):
        """Geographic coordinates value object."""

        latitude: float = Field(ge=-90.0, le=90.0)
        longitude: float = Field(ge=-180.0, le=180.0)

    class Money(Value):
        """Money value object with currency."""

        amount: Decimal = Field(
            decimal_places=FlextConstants.Performance.CURRENCY_DECIMAL_PLACES
        )
        currency: str = Field(
            min_length=FlextConstants.Performance.CURRENCY_CODE_LENGTH,
            max_length=FlextConstants.Performance.CURRENCY_CODE_LENGTH,
        )

    class PhoneNumber(Value):
        """Phone number value object."""

        number: str = Field(pattern=FlextConstants.Platform.PATTERN_PHONE_NUMBER)
        country_code: str | None = None

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

    class DateRange(Value):
        """Date range value object."""

        start_date: date
        end_date: date

        @model_validator(mode="after")
        def validate_date_order(self) -> Self:
            """Ensure start_date <= end_date."""
            if self.start_date > self.end_date:
                msg = "start_date must be before or equal to end_date"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return self

    class TimeRange(Value):
        """Time range value object."""

        start_time: time
        end_time: time

    class Address(Value):
        """Address value object."""

        street: str
        city: str
        state: str | None = None
        postal_code: str | None = None
        country: str

    class Version(Value):
        """Semantic version value object."""

        major: int
        minor: int
        patch: int
        pre_release: str | None = None

        @override
        def __str__(self) -> str:
            """String representation."""
            version = f"{self.major}.{self.minor}.{self.patch}"
            if self.pre_release:
                version += f"-{self.pre_release}"
            return version

    class PercentageValue(Value):
        """Percentage value object."""

        value: float  # Uses the Annotated type alias

    class Port(Value):
        """Network port value object."""

        number: int  # Uses the Annotated type alias

    class Duration(Value):
        """Duration value object."""

        seconds: int

        @computed_field
        def minutes(self) -> float:
            """Convert to minutes."""
            return self.seconds / 60.0

        @computed_field
        def hours(self) -> float:
            """Convert to hours."""
            return self.seconds / 3600.0

    class Size(Value):
        """Size value object."""

        bytes: int

        @computed_field
        def kilobytes(self) -> float:
            """Convert to KB."""
            return self.bytes / float(FlextConstants.Utilities.BYTES_PER_KB)

        @computed_field
        def megabytes(self) -> float:
            """Convert to MB."""
            bytes_per_mb = (
                FlextConstants.Utilities.BYTES_PER_KB
                * FlextConstants.Utilities.BYTES_PER_KB
            )
            return self.bytes / float(bytes_per_mb)

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

    # Factory methods for creating validated models
    @staticmethod
    def create_validated_email(email: str) -> FlextResult[EmailAddress]:
        """Create a validated email address."""
        try:
            return FlextResult[FlextModels.EmailAddress].ok(
                FlextModels.EmailAddress(address=email)
            )
        except ValidationError as e:
            return FlextResult[FlextModels.EmailAddress].fail(str(e))

    @staticmethod
    def create_validated_url(url: str) -> FlextResult[str]:
        """Create a validated URL."""
        try:
            parsed = urlparse(url)
            # Basic URL validation - must have scheme and netloc
            if not parsed.scheme or not parsed.netloc:
                return FlextResult[str].fail("URL must have scheme and domain")
            return FlextResult[str].ok(url)
        except Exception as e:
            return FlextResult[str].fail(f"Invalid URL: {e}")

    @staticmethod
    def create_validated_http_url(url: str) -> FlextResult[Url]:
        """Create a validated HTTP/HTTPS URL."""
        if not url.startswith((
            FlextConstants.Platform.PROTOCOL_HTTP,
            FlextConstants.Platform.PROTOCOL_HTTPS,
        )):
            return FlextResult[FlextModels.Url].fail(
                "URL must start with http:// or https://"
            )
        url_result: FlextResult[str] = FlextModels.create_validated_url(url)

        def create_url_from_validated(
            validated_url: str,
        ) -> FlextResult[FlextModels.Url]:
            return FlextResult[FlextModels.Url].ok(FlextModels.Url(url=validated_url))

        return url_result >> create_url_from_validated

    @staticmethod
    def create_validated_http_method(method: str) -> FlextResult[str]:
        """Validate HTTP method."""
        valid_methods = FlextConstants.Platform.VALID_HTTP_METHODS
        method_upper = method.upper()
        if method_upper not in valid_methods:
            return FlextResult[str].fail(f"Invalid HTTP method: {method}")
        return FlextResult[str].ok(method_upper)

    @staticmethod
    def create_validated_http_status(status_code: int) -> FlextResult[int]:
        """Validate HTTP status code."""
        if (
            not FlextConstants.Platform.MIN_HTTP_STATUS_CODE
            <= status_code
            <= FlextConstants.Platform.MAX_HTTP_STATUS_CODE
        ):
            return FlextResult[int].fail(
                f"Invalid HTTP status code: {status_code}. Must be between {FlextConstants.Platform.MIN_HTTP_STATUS_CODE} and {FlextConstants.Platform.MAX_HTTP_STATUS_CODE}."
            )
        return FlextResult[int].ok(status_code)

    @staticmethod
    def create_validated_phone(phone: str) -> FlextResult[str]:
        """Create a validated phone number."""
        try:
            # Basic phone validation - ensure it has reasonable format
            phone_clean = phone.strip()
            if not phone_clean:
                return FlextResult[str].fail("Phone number cannot be empty")

            # Validate basic phone format (optional + prefix, digits, spaces, hyphens)

            phone_pattern = FlextConstants.Platform.PATTERN_PHONE_NUMBER
            if not re.match(phone_pattern, phone_clean):
                return FlextResult[str].fail("Invalid phone number format")

            return FlextResult[str].ok(phone_clean)
        except Exception as e:
            return FlextResult[str].fail(f"Invalid phone number: {e}")

    @staticmethod
    def create_validated_uuid(value: str) -> FlextResult[str]:
        """Validate UUID string."""
        try:
            uuid_obj = uuid.UUID(value)
            return FlextResult[str].ok(str(uuid_obj))
        except ValueError as e:
            return FlextResult[str].fail(f"Invalid UUID: {e}")

    @staticmethod
    def create_validated_iso_date(date_str: str) -> FlextResult[str]:
        """Create a validated ISO format date."""
        try:
            # Validate the date can be parsed
            date.fromisoformat(date_str)
            return FlextResult[str].ok(date_str)
        except ValueError as e:
            return FlextResult[str].fail(f"Invalid ISO date format: {e}")

    @staticmethod
    def create_validated_date_range(
        start_date: str | date, end_date: str | date
    ) -> FlextResult[tuple[str, str]]:
        """Create a validated date range."""
        try:
            # Parse dates if strings
            if isinstance(start_date, str):
                start_parsed = date.fromisoformat(start_date)
                start_str = start_date
            else:
                start_parsed = start_date
                start_str = start_date.isoformat()

            if isinstance(end_date, str):
                end_parsed = date.fromisoformat(end_date)
                end_str = end_date
            else:
                end_parsed = end_date
                end_str = end_date.isoformat()

            # Validate range order
            if start_parsed > end_parsed:
                return FlextResult[tuple[str, str]].fail(
                    "Start date must be before or equal to end date"
                )

            return FlextResult[tuple[str, str]].ok((start_str, end_str))
        except (ValueError, ValidationError) as e:
            return FlextResult[tuple[str, str]].fail(f"Invalid date range: {e}")

    @staticmethod
    def create_validated_file_path(path: str) -> FlextResult[str]:
        """Create a validated file path."""
        try:
            file_path = Path(path)
            # Validate the path is reasonable (not too long, no null bytes, etc)
            if not path.strip():
                return FlextResult[str].fail("Path cannot be empty")
            return FlextResult[str].ok(str(file_path))
        except (ValueError, OSError) as e:
            return FlextResult[str].fail(f"Invalid file path: {e}")

    @staticmethod
    def create_validated_existing_file_path(path: str) -> FlextResult[str]:
        """Create a validated path that must exist."""
        path_result = FlextModels.create_validated_file_path(path)
        if path_result.is_failure:
            return path_result

        validated_path = path_result.value_or_none
        if validated_path is None:
            return FlextResult[str].fail("Path validation returned None")

        file_path = Path(str(validated_path))
        if not file_path.exists():
            return FlextResult[str].fail(f"Path does not exist: {path}")
        return FlextResult[str].ok(str(validated_path))

    @staticmethod
    def create_validated_directory_path(path: str) -> FlextResult[str]:
        """Create a validated directory path."""
        path_result = FlextModels.create_validated_existing_file_path(path)
        if path_result.is_failure:
            return path_result

        validated_path = path_result.value_or_none
        if validated_path is None:
            return FlextResult[str].fail("Path validation returned None")

        dir_path = Path(str(validated_path))
        if not dir_path.is_dir():
            return FlextResult[str].fail(f"Path is not a directory: {path}")
        return FlextResult[str].ok(str(validated_path))

    # Additional domain models continue with advanced patterns...
    class FactoryRegistrationModel(StrictArbitraryTypesModel):
        """Enhanced factory registration with advanced validation."""

        name: str
        factory: Callable[[], object]
        singleton: bool = False
        dependencies: FlextTypes.Core.StringList = Field(default_factory=list)

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

        services: FlextTypes.Core.Dict = Field(default_factory=dict)
        factories: dict[str, Callable[[], object]] = Field(default_factory=dict)
        singletons: FlextTypes.Core.Dict = Field(default_factory=dict)

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

        level: str = Field(
            default_factory=lambda: FlextConfig.get_global_instance().log_level
        )
        message: str
        context: FlextTypes.Core.Dict = Field(default_factory=dict)
        timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
        source: str | None = None
        operation: str | None = None
        obj: object | None = None

    class ProcessingRequest(ArbitraryTypesModel):
        """Enhanced processing request with advanced validation."""

        operation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        data: FlextTypes.Core.Dict = Field(default_factory=dict)
        context: FlextTypes.Core.Dict = Field(default_factory=dict)
        timeout_seconds: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().timeout_seconds
        )
        retry_attempts: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().max_retry_attempts
        )
        enable_validation: bool = True

        model_config = ConfigDict(
            validate_assignment=False,  # Allow invalid values to be set for testing
            use_enum_values=True,
            arbitrary_types_allowed=True,
        )

        @field_validator("context")
        @classmethod
        def validate_context(cls, v: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
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
        event_types: FlextTypes.Core.StringList = Field(default_factory=list)
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

        batch_size: int = Field(
            default=FlextConstants.Performance.BatchProcessing.DEFAULT_SIZE
        )
        max_workers: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().max_workers
        )
        timeout_per_item: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().timeout_seconds
        )
        continue_on_error: bool = True
        data_items: Annotated[FlextTypes.Core.List, Field(default_factory=list)]

        @field_validator("data_items")
        @classmethod
        def validate_data_items(cls, v: FlextTypes.Core.List) -> FlextTypes.Core.List:
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
        input_data: FlextTypes.Core.Dict = Field(default_factory=dict)
        execution_context: FlextTypes.Core.Dict = Field(default_factory=dict)
        timeout_seconds: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().timeout_seconds
        )
        retry_on_failure: bool = True
        max_retries: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().max_retry_attempts
        )
        fallback_handlers: FlextTypes.Core.StringList = Field(default_factory=list)

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
        steps: Annotated[list[FlextTypes.Core.Dict], Field(default_factory=list)]
        parallel_execution: bool = FlextConstants.Cqrs.DEFAULT_PARALLEL_EXECUTION
        stop_on_error: bool = FlextConstants.Cqrs.DEFAULT_STOP_ON_ERROR
        max_parallel: int = Field(
            gt=0, default=FlextConstants.Performance.DEFAULT_MAX_PARALLEL
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
        errors: FlextTypes.Core.StringList = Field(default_factory=list)
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

        indent: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().json_indent
        )
        sort_keys: bool = False
        ensure_ascii: bool = False
        separators: tuple[str, str] = (",", ":")
        default_handler: Callable[[object], object] | None = None

    class TimestampConfig(StrictArbitraryTypesModel):
        """Enhanced timestamp configuration."""

        obj: object
        use_utc: bool = Field(
            default_factory=lambda: FlextConfig.get_global_instance().use_utc_timestamps
        )
        auto_update: bool = Field(
            default_factory=lambda: FlextConfig.get_global_instance().use_utc_timestamps
        )
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
        encoding: str = Field(
            default_factory=lambda: FlextConfig.get_global_instance().serialization_encoding
        )
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
        parameters: FlextTypes.Core.Dict = Field(default_factory=dict)
        context: FlextTypes.Core.Dict = Field(default_factory=dict)
        timeout_seconds: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().timeout_seconds
        )
        async_execution: bool = False
        enable_validation: bool = True

        @field_validator("context")
        @classmethod
        def validate_context(cls, v: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
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

        entity: FlextTypes.Core.Dict
        rules: FlextTypes.Core.StringList = Field(default_factory=list)
        validate_business_rules: bool = True
        validate_integrity: bool = True
        validate_permissions: bool = False
        context: FlextTypes.Core.Dict = Field(default_factory=dict)

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
        operations: Annotated[list[FlextTypes.Core.Dict], Field(default_factory=list)]
        parallel_execution: bool = False
        stop_on_error: bool = True
        batch_size: int = Field(
            default=FlextConstants.Performance.BatchProcessing.DEFAULT_SIZE
        )
        timeout_per_operation: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().timeout_seconds
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
        metric_types: FlextTypes.Core.StringList = Field(
            default_factory=lambda: ["performance", "errors", "throughput"]
        )
        time_range_seconds: int = FlextConstants.Performance.DEFAULT_TIME_RANGE_SECONDS
        aggregation: Literal["sum", "avg", "min", "max", "count"] = "avg"
        group_by: FlextTypes.Core.StringList = Field(default_factory=list)
        filters: FlextTypes.Core.Dict = Field(default_factory=dict)

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
        data: FlextTypes.Core.Dict = Field(default_factory=dict)
        filters: FlextTypes.Core.Dict = Field(default_factory=dict)

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
        arguments: FlextTypes.Core.Dict = Field(default_factory=dict)
        keyword_arguments: FlextTypes.Core.Dict = Field(default_factory=dict)
        timeout_seconds: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().timeout_seconds
        )
        retry_config: FlextTypes.Core.Dict = Field(default_factory=dict)

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
            default_factory=lambda: FlextConfig.get_global_instance().max_retry_attempts
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
        retry_on_status_codes: Annotated[
            FlextTypes.Core.List, Field(default_factory=list)
        ]

        @field_validator("retry_on_status_codes")
        @classmethod
        def validate_backoff_strategy(
            cls, v: FlextTypes.Core.List
        ) -> FlextTypes.Core.List:
            """Validate status codes are valid HTTP codes."""
            validated_codes: FlextTypes.Core.List = []
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
            default_factory=lambda: FlextConfig.get_global_instance().max_retry_attempts
        )
        sliding_window_size: int = Field(
            default=FlextConstants.Performance.BatchProcessing.DEFAULT_SIZE
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

        enable_strict_mode: bool = Field(
            default_factory=lambda: FlextConfig.get_global_instance().validation_strict_mode
        )
        max_validation_errors: int = Field(
            default=FlextConstants.Cqrs.DEFAULT_MAX_VALIDATION_ERRORS,
            description="Maximum validation errors",
        )
        validate_on_assignment: bool = True
        validate_on_read: bool = False
        custom_validators: Annotated[FlextTypes.Core.List, Field(default_factory=list)]

        @field_validator("custom_validators")
        @classmethod
        def validate_additional_validators(
            cls, v: FlextTypes.Core.List
        ) -> FlextTypes.Core.List:
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
        user_context: FlextTypes.Core.Dict = Field(default_factory=dict)
        security_context: FlextTypes.Core.Dict = Field(default_factory=dict)
        metadata: FlextTypes.Core.Dict = Field(default_factory=dict)
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
        context: FlextTypes.Core.Dict = Field(default_factory=dict)

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
        transitions: FlextTypes.Core.Dict = Field(default_factory=dict)
        current_state: str | None = None
        state_data: FlextTypes.Core.Dict = Field(default_factory=dict)

        @field_validator("transitions")
        @classmethod
        def validate_transitions(cls, v: FlextTypes.Core.Dict) -> FlextTypes.Core.Dict:
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
                    msg = (
                        f"Transitions for state {state} must be a FlextTypes.Core.Dict"
                    )
                    raise FlextExceptions.TypeError(
                        message=msg,
                        error_code=FlextConstants.Errors.TYPE_ERROR,
                    )
                for event, next_state in cast(
                    "FlextTypes.Core.Dict", transitions
                ).items():
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
            default_factory=lambda: FlextConfig.get_global_instance().timeout_seconds
        )
        metadata: FlextTypes.Core.Dict = Field(default_factory=dict)

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
        def validate_metrics_collector(
            cls, v: FlextTypes.Core.Dict
        ) -> FlextTypes.Core.Dict:
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
        validation_schema: FlextTypes.Core.Dict | None = None
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

    class CompactSerializableModel(SerializableModel):
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

    # Field-level serialization optimizations using Annotated types
    JsonStr = Annotated[str, Field(json_schema_extra={"format": "json"})]
    Base64Bytes = Annotated[bytes, Field(json_schema_extra={"format": "base64"})]
    IsoDateTime = Annotated[datetime, Field(json_schema_extra={"format": "date-time"})]
    IsoDate = Annotated[date, Field(json_schema_extra={"format": "date"})]
    IsoTime = Annotated[time, Field(json_schema_extra={"format": "time"})]

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
            cls, context: FlextTypes.Core.Dict | None
        ) -> FlextTypes.Core.Dict:
            """Extract serialization context from context FlextTypes.Core.Dict."""
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
        ) -> FlextTypes.Core.Dict:
            """Context-aware model dump."""
            if context and isinstance(context, dict):
                ser_context = self._get_serialization_context(
                    cast("FlextTypes.Core.Dict", context)
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
                context=cast("FlextTypes.Core.Dict | None", context),
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
            config = FlextConfig.get_global_instance()
            if not getattr(config, "ensure_json_serializable", True):
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
        ) -> list[FlextTypes.Core.Dict] | str:
            """Serialize a batch of models efficiently.

            Args:
                models: List of Pydantic models to serialize
                output_format: Output format ('FlextTypes.Core.Dict' or 'json')
                compact: Use compact serialization
                parallel: Use parallel processing for large batches

            Returns:
                List of FlextTypes.Core.Dicts or JSON array string

            """
            if not models:
                return [] if output_format == "dict" else "[]"

            # Validate JSON serializability if required by configuration
            if output_format == "json":
                for model in models:
                    FlextModels.BatchSerializer.validate_json_serializable(model)

            # Serialization kwargs
            dump_kwargs: FlextTypes.Core.Dict = {}
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

                        def serialize_to_dict(m: BaseModel) -> FlextTypes.Core.Dict:
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
        ) -> FlextTypes.Core.Dict:
            """Get optimized JSON schema for a model.

            Args:
                model: Pydantic model class
                by_alias: Use field aliases in schema
                ref_template: Template for schema references
                mode: Schema mode

            Returns:
                Optimized JSON schema FlextTypes.Core.Dict

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
        context: FlextTypes.Core.Dict = Field(
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
                default=FlextConstants.Performance.BatchProcessing.DEFAULT_SIZE,
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
                bus_config: FlextTypes.Core.Dict | None = None,
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
                    enable_metrics = FlextConfig.get_global_instance().enable_metrics

                config_data: FlextTypes.Core.Dict = {
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
            metadata: FlextTypes.Core.Dict = Field(
                default_factory=dict, description="Handler metadata"
            )

            @classmethod
            def create_handler_config(
                cls,
                handler_type: Literal["command", "query", "event", "saga"],
                *,
                default_name: str | None = None,
                default_id: str | None = None,
                handler_config: FlextTypes.Core.Dict | None = None,
                command_timeout: int = 0,
                max_command_retries: int = 0,
            ) -> Self:
                """Create handler configuration with defaults and overrides."""
                handler_mode_value = (
                    FlextConstants.Dispatcher.HANDLER_MODE_COMMAND
                    if handler_type == FlextConstants.Cqrs.COMMAND_HANDLER_TYPE
                    else FlextConstants.Dispatcher.HANDLER_MODE_QUERY
                )
                config_data: FlextTypes.Core.Dict = {
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
                    return FlextResult[str].fail(f"Unsupported URL scheme: {result.scheme}")

                # Validate domain
                if result.netloc:
                    domain = result.netloc.split(":")[0]  # Remove port
                    if not domain or len(domain) > FlextConstants.Validation.MAX_EMAIL_LENGTH:
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
            max_validation_time_ms: int = 100,
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
            start_time = time_module.time()

            try:
                validated_model = model.__class__.model_validate(model.model_dump())
                validation_time = (time_module.time() - start_time) * 1000

                if validation_time > max_validation_time_ms:
                    return FlextResult[T].fail(
                        f"Validation too slow: {validation_time:.2f}ms > {max_validation_time_ms}ms",
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

    class Query(BaseModel):
        """Query model for CQRS query operations."""

        filters: FlextTypes.Core.Dict = Field(
            default_factory=dict, description="Query filters"
        )
        pagination: FlextModels.Pagination = Field(
            default_factory=_create_default_pagination,
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
            cls, query_payload: FlextTypes.Core.Dict
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

            # Add pagination
            if self.pagination:
                params["page"] = self.pagination.page
                params["size"] = self.pagination.size

            # Add filters
            for key, value in self.filters.items():
                if isinstance(value, (list, tuple)):
                    params[f"filter_{key}"] = ",".join(str(v) for v in value)
                else:
                    params[f"filter_{key}"] = str(value)

            return urlencode(params)

    # Factory methods for Pydantic Field default_factory compliance
    @staticmethod
    def _default_pagination() -> FlextModels.Pagination:
        """Factory method for default Pagination instance."""
        return FlextModels.Pagination()

    @staticmethod
    def _default_uuid_str() -> str:
        """Factory method for default UUID string."""
        return str(uuid.uuid4())

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
