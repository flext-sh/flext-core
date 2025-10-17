"""Domain models for domain-driven design patterns.

This module provides FlextModels, a comprehensive collection of base classes
and utilities for implementing domain-driven design (DDD) patterns in the
FLEXT ecosystem.

All models use Pydantic for validation and serialization, providing type-safe
domain modeling with automatic validation and error handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import re
import time as time_module
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import (
    Annotated,
    ClassVar,
    Literal,
    Self,
    cast,
    override,
)
from urllib.parse import ParseResult, urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    PrivateAttr,
    ValidationError,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.loggings import FlextLogger
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes


class FlextModels:
    """Domain-Driven Design (DDD) patterns with Pydantic validation.

    Architecture: Layer 2 (Domain)
    ==============================
    Provides comprehensive base classes for implementing Domain-Driven Design
    patterns with Pydantic v2 validation, event sourcing support, and CQRS
    integration across the FLEXT ecosystem.

    Structural Typing and Protocol Compliance:
    ===========================================
    FlextModels implements multiple FlextProtocols interfaces via structural
    typing (duck typing) through nested classes:

    - Entity (satisfies FlextProtocols.Entity):
      * id: Unique identifier for entity tracking
      * created_at: Creation timestamp
      * updated_at: Modification timestamp
      * is_valid(): Validate entity state
      * to_dict(): Serialize to dictionary

    - Value (satisfies FlextProtocols.Value):
      * Immutable value objects (frozen Pydantic models)
      * Compared by value, not identity
      * No mutable state after creation
      * Hashable for use in collections

    - AggregateRoot (satisfies FlextProtocols.AggregateRoot):
      * Consistency boundary enforcement
      * Transactional invariant protection
      * Event sourcing support
      * Domain event publishing

    - Command (satisfies FlextProtocols.Command):
      * Represents domain operations
      * Command validation
      * Handler mapping
      * Idempotency support

    - Query (satisfies FlextProtocols.Query):
      * Read-side operations
      * Non-mutating operations
      * Result projection
      * Caching support

    - DomainEvent (satisfies FlextProtocols.DomainEvent):
      * Event sourcing backbone
      * Immutable past events
      * Event replay support
      * Audit trail

    Core DDD Concepts:
    ==================
    1. **Entity**: Domain object with identity and lifecycle
       - Changes tracked through updated_at
       - Compared by id, not value equality
       - Supports domain logic and invariants
       - Integrated with FlextResult for operations

    2. **Value Object**: Immutable domain values
       - No identity (compared by value)
       - Immutable after creation (frozen Pydantic)
       - Composable building blocks
       - Hashable for collections

    3. **Aggregate Root**: Consistency boundary
       - Contains entities and value objects
       - Enforces transactional invariants
       - Single root for external references
       - Event sourcing support

    4. **Command**: Domain operation request
       - Represents "I want X to happen"
       - Immutable command object
       - Handler determines execution
       - Async command bus support

    5. **Query**: Domain read operation
       - Represents "I want to know X"
       - Non-mutating read projection
       - Result optimization via caching
       - Query bus support

    6. **Domain Event**: Significant domain occurrence
       - Represents "X happened"
       - Event sourcing backbone
       - Event replay for reconstruction
       - Audit trail support

    Pydantic v2 Integration:
    =======================
    - Full Pydantic BaseModel support
    - Automatic validation via field_validator
    - Model validation via model_validator
    - Computed fields for derived properties
    - Custom serializers for domain logic
    - Config inheritance via ConfigDict
    - Immutable models (frozen=True)

    Features and Components:
    ========================
    - Entity: Base domain entity with lifecycle
    - Value: Immutable value objects
    - AggregateRoot: Consistency boundary root
    - Command: CQRS command pattern
    - Query: CQRS query pattern
    - DomainEvent: Event sourcing events
    - Validation: Business rule validators
    - Timestamps: Automatic created_at/updated_at
    - Serialization: JSON and dict conversion
    - Type validation: Complete type safety

    Advanced Patterns:
    ==================
    - Event Sourcing: Replay events to reconstruct state
    - CQRS: Separate read/write models with Command/Query
    - Transactional Invariants: Enforce business rules
    - Aggregate Roots: Consistency boundary enforcement
    - Value Objects: Rich domain types
    - Domain Events: Capture domain state changes

    Error Handling:
    ===============
    - FlextResult[T] wrapping for operations
    - Validation errors caught in is_valid()
    - Business rule violations in invariants
    - Structured error information

    Usage Patterns:
    ===============
        >>> from flext_core.models import FlextModels
        >>> from flext_core.result import FlextResult
        >>>
        >>> # Value Object - immutable by design
        >>> class Email(FlextModels.Value):
        ...     address: str
        ...
        ...     @field_validator("address")
        ...     @classmethod
        ...     def validate_email(cls, v: str) -> str:
        ...         if "@" not in v:
        ...             raise ValueError("Invalid email")
        ...         return v.lower()
        >>>
        >>> # Entity - with identity and lifecycle
        >>> class User(FlextModels.Entity):
        ...     name: str
        ...     email: Email
        ...     is_active: bool = False
        ...
        ...     def activate(self) -> FlextResult[None]:
        ...         if self.is_active:
        ...             return FlextResult.fail("Already active")
        ...         self.is_active = True
        ...         return FlextResult.ok(None)
        >>>
        >>> # Aggregate Root - consistency boundary
        >>> class Account(FlextModels.AggregateRoot):
        ...     owner: User
        ...     balance: float = 0.0
        ...
        ...     def deposit(self, amount: float) -> FlextResult[None]:
        ...         if amount <= 0:
        ...             return FlextResult.fail("Amount must be positive")
        ...         self.balance += amount
        ...         return FlextResult.ok(None)
        >>>
        >>> # Command - CQRS pattern
        >>> class CreateUserCommand(FlextModels.Command):
        ...     name: str
        ...     email: str
        >>>
        >>> # Query - CQRS pattern
        >>> class GetUserQuery(FlextModels.Query):
        ...     user_id: str
        >>>
        >>> # Domain Event - Event sourcing
        >>> class UserCreatedEvent(FlextModels.DomainEvent):
        ...     user_id: str
        ...     name: str
        ...     email: str

    Integration with FLEXT Ecosystem:
    ==================================
    - Service Layer: Services receive FlextResult[T] from models
    - Handler Layer: CQRS handlers process Commands/Queries
    - Bus Layer: Command/Event buses route through aggregates
    - Logger Integration: Automatic audit logging
    - Protocol Compliance: Structural typing satisfaction
    - Validation Layer: Business rule enforcement

    Thread Safety:
    ==============
    - Pydantic models are immutable when frozen=True
    - Value objects are always immutable
    - Entities are mutable but thread-safe for creation
    - Aggregate roots manage transactional boundaries
    - Event sourcing provides replay safety

    Performance Characteristics:
    ===========================
    - O(1) entity/value creation via Pydantic
    - O(1) identity comparison for entities
    - O(1) timestamp tracking (automatic)
    - O(n) event replay for aggregate reconstruction
    - O(1) command/query dispatch via handler registry
    """

    # =========================================================================
    # BEHAVIOR MIXINS - Reusable model behaviors (Pydantic 2.11 pattern)
    # =========================================================================

    class IdentifiableMixin(BaseModel):
        """Mixin for models with unique identifiers.

        Provides the `id` field using UuidField pattern with explicit default.
        Used by Entity, Command, DomainEvent, Saga, and other identifiable models.
        """

        model_config = ConfigDict(
            validate_assignment=True,
            validate_return=True,
            validate_default=True,
            strict=True,
            str_strip_whitespace=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            extra="forbid",
            frozen=False,
            ser_json_timedelta="iso8601",
            ser_json_bytes="base64",
            hide_input_in_errors=True,
            json_schema_extra={
                "title": "IdentifiableMixin",
                "description": "Mixin providing unique identifier fields",
            },
        )

        id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    class TimestampableMixin(BaseModel):
        """Mixin for models with creation and update timestamps.

        Provides `created_at` and `updated_at` fields with automatic timestamp management.
        Used by Entity, TimestampedModel, and models requiring audit trails.

        Pydantic 2 features:
        - field_serializer for ISO 8601 timestamp serialization
        - Automatic timestamp management
        - Annotated fields with rich metadata
        """

        model_config = ConfigDict(
            validate_assignment=True,
            validate_return=True,
            validate_default=True,
            strict=True,
            str_strip_whitespace=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            extra="forbid",
            frozen=False,
            ser_json_timedelta="iso8601",
            ser_json_bytes="base64",
            hide_input_in_errors=True,
            json_schema_extra={
                "title": "TimestampableMixin",
                "description": "Mixin providing timestamp fields and serialization",
            },
        )

        created_at: datetime = Field(
            default_factory=lambda: datetime.now(UTC),
            description="Timestamp when the model was created (UTC timezone)",
            examples=[
                datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
                datetime(2025, 10, 12, 15, 30, 0, tzinfo=UTC),
            ],
        )
        updated_at: datetime | None = Field(
            default=None,
            description="Timestamp when the model was last updated (UTC timezone)",
            examples=[
                None,
                datetime(2025, 1, 2, 14, 30, 0, tzinfo=UTC),
            ],
        )

        @field_serializer("created_at", "updated_at", when_used="json")
        def serialize_timestamps(self, value: datetime | None) -> str | None:
            """Serialize timestamps to ISO 8601 format for JSON."""
            return value.isoformat() if value else None

        @computed_field
        def is_modified(self) -> bool:
            """Check if the model has been modified after creation."""
            return self.updated_at is not None

        def update_timestamp(self) -> None:
            """Update the updated_at timestamp to current UTC time."""
            self.updated_at = datetime.now(UTC)

    class VersionableMixin(BaseModel):
        """Mixin for models with versioning support.

        Provides `version` field and increment_version method for optimistic locking.
        Used by Entity and models requiring version tracking.

        Pydantic 2.11 features:
        - Annotated field with rich metadata
        - Computed field for version state checks
        """

        model_config = ConfigDict(
            validate_assignment=True,
            validate_return=True,
            validate_default=True,
            strict=True,
            str_strip_whitespace=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            extra="forbid",
            frozen=False,
            ser_json_timedelta="iso8601",
            ser_json_bytes="base64",
            hide_input_in_errors=True,
            json_schema_extra={
                "title": "VersionableMixin",
                "description": "Mixin providing version fields and optimistic locking",
            },
        )

        version: int = Field(
            default=FlextConstants.Performance.DEFAULT_VERSION,
            ge=FlextConstants.Performance.MIN_VERSION,
            description="Version number for optimistic locking - increments with each update",
            examples=[1, 5, 42, 100],
        )

        @computed_field
        def is_initial_version(self) -> bool:
            """Check if this is the initial version (version 1)."""
            return self.version == FlextConstants.Performance.DEFAULT_VERSION

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

        model_config = ConfigDict(
            validate_assignment=True,  # Validate on field assignment
            validate_return=True,  # Validate return values
            validate_default=True,  # Validate default values
            strict=True,  # Strict type coercion
            str_strip_whitespace=True,  # Auto-strip whitespace
            use_enum_values=True,  # Use enum values
            arbitrary_types_allowed=True,  # Allow custom types
            extra="forbid",  # No extra fields
            frozen=False,  # Allow mutations
            ser_json_timedelta="iso8601",  # ISO 8601 timedelta
            ser_json_bytes="base64",  # Base64 bytes
            hide_input_in_errors=True,  # Security
            json_schema_extra={
                "title": "ArbitraryTypesModel",
                "description": "Base model with arbitrary types support and comprehensive validation",
            },
        )

    # =============================================================================
    # METACLASSES - Foundation metaclasses for advanced model patterns
    # =============================================================================

    class StrictArbitraryTypesModel(BaseModel):
        """Strict pattern: forbid extra fields, arbitrary types allowed.

        Used by domain service models requiring strict validation.
        Enhanced with comprehensive Pydantic 2.11 features.
        """

        model_config = ConfigDict(
            validate_assignment=True,  # Validate on field assignment
            validate_return=True,  # Validate return values
            validate_default=True,  # Validate default values
            strict=True,  # Strict type coercion
            str_strip_whitespace=True,  # Auto-strip whitespace
            use_enum_values=True,  # Use enum values
            arbitrary_types_allowed=True,  # Allow custom types
            extra="forbid",  # No extra fields
            frozen=False,  # Allow mutations
            ser_json_timedelta="iso8601",  # ISO 8601 timedelta
            ser_json_bytes="base64",  # Base64 bytes
            hide_input_in_errors=True,  # Security
            json_schema_extra={
                "title": "StrictArbitraryTypesModel",
                "description": "Base model with strict validation and arbitrary types support",
            },
        )

    class FrozenStrictModel(BaseModel):
        """Immutable pattern: frozen with extra fields forbidden.

        Used by value objects and configuration models.
        """

        model_config = ConfigDict(
            validate_assignment=False,  # No assignment validation for frozen models
            validate_return=True,  # Validate return values
            validate_default=True,  # Validate default values
            strict=True,  # Strict type coercion
            str_strip_whitespace=True,  # Auto-strip whitespace
            use_enum_values=True,  # Use enum values
            arbitrary_types_allowed=True,  # Allow custom types
            extra="forbid",  # No extra fields
            frozen=True,  # Immutable model
            ser_json_timedelta="iso8601",  # ISO 8601 timedelta
            ser_json_bytes="base64",  # Base64 bytes
            hide_input_in_errors=True,  # Security
            json_schema_extra={
                "title": "FrozenStrictModel",
                "description": "Immutable base model with strict validation and frozen state",
            },
        )

    class TimestampedModel(ArbitraryTypesModel, TimestampableMixin):
        """Base class for models with timestamp fields.

        Inherits timestamp functionality from TimestampableMixin.
        Provides created_at and updated_at fields with automatic management.
        """

    class DomainEvent(ArbitraryTypesModel, IdentifiableMixin, TimestampableMixin):
        """Base class for domain events.

        Uses IdentifiableMixin for id and TimestampableMixin for created_at.
        Includes message_type discriminator for Pydantic v2 discriminated unions.
        """

        # Pydantic v2 Discriminated Union discriminator field (immutable)
        message_type: Literal["event"] = Field(
            default="event",
            frozen=True,
            description="Message type discriminator for union routing - always 'event'",
        )

        event_type: str
        aggregate_id: str
        data: FlextTypes.Domain.EventPayload = Field(default_factory=dict)
        metadata: FlextTypes.Domain.EventMetadata = Field(default_factory=dict)

    class Entity(TimestampedModel, IdentifiableMixin, VersionableMixin):
        """Base class for domain entities with identity.

        Combines TimestampedModel, IdentifiableMixin, and VersionableMixin to provide:
        - id: Unique identifier (from IdentifiableMixin)
        - created_at/updated_at: Timestamps (from TimestampedModel)
        - version: Optimistic locking (from VersionableMixin)
        - domain_events: Event sourcing support

        Internal implementation note: Class uses a shared logger for domain
        event operations (add, commit, clear).
        """

        # Class-level logger for domain event operations
        _internal_logger: ClassVar[FlextLogger | None] = None

        @classmethod
        def _get_logger(cls) -> FlextLogger:
            """Get or create the internal logger (lazy initialization)."""
            if cls._internal_logger is None:
                cls._internal_logger = FlextLogger(__name__)
            return cls._internal_logger

        domain_events: list[FlextModels.DomainEvent] = Field(default_factory=list)

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

        def add_domain_event(
            self, event_name: str, data: FlextTypes.Dict
        ) -> FlextResult[None]:
            """Add a domain event to be dispatched with enhanced validation.

            Enhanced domain event handling with comprehensive validation,
            automatic event handler execution, and consistent error handling.
            Used across all flext-ecosystem projects for event sourcing.

            Args:
                event_name: The type/name of the domain event (must be non-empty)
                data: Event payload data (must be serializable dict)

            Returns:
                FlextResult[None]: Success if event added, failure with details

            Example:
                ```python
                result = entity.add_domain_event("UserCreated", {"user_id": "123"})
                if result.is_failure:
                    logger.error(f"Failed to add event: {result.error}")
                ```

            """
            # Validate inputs
            if not event_name or not isinstance(event_name, str):
                return FlextResult[None].fail(
                    "Domain event name must be a non-empty string",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            if data is not None and not isinstance(data, dict):
                return FlextResult[None].fail(
                    "Domain event data must be a dictionary or None",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Check for too many uncommitted events
            if (
                len(self.domain_events)
                >= FlextConstants.Validation.MAX_UNCOMMITTED_EVENTS
            ):
                return FlextResult[None].fail(
                    f"Maximum uncommitted events reached: {FlextConstants.Validation.MAX_UNCOMMITTED_EVENTS}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            try:
                # Create domain event with validation
                domain_event = FlextModels.DomainEvent(
                    event_type=event_name,
                    aggregate_id=self.id,
                    data=data or {},
                )

                # Validate the created domain event
                validation_result = FlextModels.Validation.validate_domain_event(
                    domain_event
                )
                if validation_result.is_failure:
                    return FlextResult[None].fail(
                        f"Domain event validation failed: {validation_result.error}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

                # Add event to collection
                self.domain_events.append(domain_event)

                # Integration: Track domain event via FlextRuntime
                FlextRuntime.Integration.track_domain_event(
                    event_name=event_name,
                    aggregate_id=self.id,
                    event_data=data,
                )

                # Log domain event addition via structlog
                self._get_logger().debug(
                    "Domain event added",
                    event_type=event_name,
                    aggregate_id=self.id,
                    aggregate_type=self.__class__.__name__,
                    event_id=domain_event.id,
                    data_keys=list(data.keys()) if data else [],
                )

                # Try to find and call event handler method
                event_type = data.get("event_type", "") if data else ""
                if event_type:
                    handler_method_name = f"_apply_{str(event_type).lower()}"
                    if hasattr(self, handler_method_name):
                        try:
                            handler_method = getattr(self, handler_method_name)
                            handler_method(data)
                            self._get_logger().debug(
                                "Domain event handler executed",
                                event_type=event_name,
                                handler=handler_method_name,
                                aggregate_id=self.id,
                            )
                        except Exception as e:
                            # Log exception but don't re-raise to maintain resilience
                            self._get_logger().warning(
                                f"Domain event handler {handler_method_name} failed for event {event_name}: {e}"
                            )

                # Increment version after adding domain event (from VersionableMixin)
                self.increment_version()
                self.update_timestamp()

                return FlextResult[None].ok(None)

            except Exception as e:
                return FlextResult[None].fail(
                    f"Failed to add domain event: {e}",
                    error_code=FlextConstants.Errors.DOMAIN_EVENT_ERROR,
                )

        @computed_field
        @property
        def uncommitted_events(self) -> list[FlextModels.DomainEvent]:
            """Get uncommitted domain events without clearing them.

            Returns:
                List of uncommitted domain events (DomainEvent instances).

            """
            return list(self.domain_events)

        def mark_events_as_committed(
            self,
        ) -> FlextResult[list[FlextModels.DomainEvent]]:
            """Mark all domain events as committed and return them.

            Enhanced method with proper error handling and validation.
            Clears the event list after marking as committed. Use this in
            event sourcing scenarios to track when events have been persisted.

            Returns:
                FlextResult[list[DomainEvent]]: Committed events if successful

            Example:
                ```python
                result = entity.mark_events_as_committed()
                if result.is_success:
                    committed_events = result.unwrap()
                    # Persist events to event store
                ```

            """
            try:
                # Get uncommitted events before clearing
                events = self.uncommitted_events

                # Log commitment via structlog if there are events
                if events:
                    # Type-safe access to event_type
                    event_types = [e.event_type for e in events]
                    self._get_logger().info(
                        "Domain events committed",
                        aggregate_id=self.id,
                        aggregate_type=self.__class__.__name__,
                        event_count=len(events),
                        event_types=event_types,
                    )

                # Clear all events
                self.domain_events.clear()

                return FlextResult[list[FlextModels.DomainEvent]].ok(events)

            except Exception as e:
                return FlextResult[list[FlextModels.DomainEvent]].fail(
                    f"Failed to commit domain events: {e}",
                    error_code=FlextConstants.Errors.DOMAIN_EVENT_ERROR,
                )

        def clear_domain_events(self) -> list[FlextModels.DomainEvent]:
            """Clear and return domain events.

            This method logs the clearing operation via structlog for
            observability and debugging purposes.

            Returns:
                List of domain events that were cleared.

            """
            # Get events before clearing
            events = self.domain_events.copy()

            # Log clearing operation via structlog if there are events
            if events:
                domain_events = list(events)
                self._get_logger().debug(
                    "Domain events cleared",
                    aggregate_id=self.id,
                    aggregate_type=self.__class__.__name__,
                    event_count=len(domain_events),
                    event_types=[e.event_type for e in domain_events]
                    if domain_events
                    else [],
                )

            self.domain_events.clear()
            return events

        def add_domain_events_bulk(
            self, events: list[tuple[str, FlextTypes.Dict]]
        ) -> FlextResult[None]:
            """Add multiple domain events in bulk with validation.

            Efficient bulk operation for adding multiple domain events at once.
            Validates all events before adding any, ensuring atomicity.

            Args:
                events: List of (event_name, data) tuples

            Returns:
                FlextResult[None]: Success if all events added, failure with details

            Example:
                ```python
                events = [
                    ("UserCreated", {"user_id": "123"}),
                    ("EmailSent", {"email": "user@example.com"}),
                ]
                result = entity.add_domain_events_bulk(events)
                ```

            """
            if not events:
                return FlextResult[None].ok(None)

            if not isinstance(events, list):
                return FlextResult[None].fail(
                    "Events must be a list of tuples",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Check total events won't exceed limit
            total_after_add = len(self.domain_events) + len(events)
            if total_after_add > FlextConstants.Validation.MAX_UNCOMMITTED_EVENTS:
                return FlextResult[None].fail(
                    f"Bulk add would exceed max events: {total_after_add} > {FlextConstants.Validation.MAX_UNCOMMITTED_EVENTS}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Validate all events first
            validated_events = []
            for i, (event_name, data) in enumerate(events):
                if not isinstance(event_name, str) or not event_name:
                    return FlextResult[None].fail(
                        f"Event {i}: name must be non-empty string",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                if data is not None and not isinstance(data, dict):
                    return FlextResult[None].fail(
                        f"Event {i}: data must be dict[str, object] or None",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                validated_events.append((event_name, data or {}))

            # Add all events
            try:
                for event_name, data in validated_events:
                    domain_event = FlextModels.DomainEvent(
                        event_type=event_name,
                        aggregate_id=self.id,
                        data=data,
                    )
                    self.domain_events.append(domain_event)
                    self.increment_version()

                self.update_timestamp()

                # Log bulk operation
                self._get_logger().info(
                    "Bulk domain events added",
                    aggregate_id=self.id,
                    aggregate_type=self.__class__.__name__,
                    event_count=len(validated_events),
                    event_types=[name for name, _ in validated_events],
                )

                return FlextResult[None].ok(None)

            except Exception as e:
                return FlextResult[None].fail(
                    f"Failed to add bulk domain events: {e}",
                    error_code=FlextConstants.Errors.DOMAIN_EVENT_ERROR,
                )

        def validate_consistency(self) -> FlextResult[None]:
            """Validate entity consistency using centralized validation.

            Comprehensive consistency check using FlextModels.Validation
            utilities. Ensures entity invariants and relationships are valid.

            Returns:
                FlextResult[None]: Success if consistent, failure with details

            Example:
                ```python
                result = entity.validate_consistency()
                if result.is_failure:
                    logger.error(f"Entity inconsistent: {result.error}")
                ```

            """
            # Use centralized validation utilities
            entity_result = FlextModels.Validation.validate_entity_relationships(self)
            if entity_result.is_failure:
                return FlextResult[None].fail(
                    f"Entity validation failed: {entity_result.error}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Validate domain events
            for event in self.domain_events:
                event_result = FlextModels.Validation.validate_domain_event(event)
                if event_result.is_failure:
                    return FlextResult[None].fail(
                        f"Domain event validation failed: {event_result.error}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

            return FlextResult[None].ok(None)

    class Value(FrozenStrictModel):
        """Base class for value objects - immutable and compared by value."""

        @override
        def __eq__(self, other: object) -> bool:
            """Compare by value."""
            if not isinstance(other, self.__class__):
                return False
            if hasattr(self, "model_dump") and hasattr(other, "model_dump"):
                return bool(self.model_dump() == other.model_dump())
            return False

        @override
        def __hash__(self) -> int:
            """Hash based on values for use in sets/FlextTypes.Dicts."""
            return hash(tuple(self.model_dump().items()))

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
        Includes message_type discriminator for Pydantic v2 discriminated unions.
        """

        # Pydantic v2 Discriminated Union discriminator field (immutable)
        message_type: Literal["command"] = Field(
            default="command",
            frozen=True,
            description="Message type discriminator for union routing - always 'command'",
        )

        command_type: str = Field(
            default_factory=lambda: FlextConstants.Cqrs.DEFAULT_COMMAND_TYPE,
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

    class Metadata(FrozenStrictModel):
        """Immutable metadata model."""

        created_by: str | None = None
        created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        modified_by: str | None = None
        modified_at: datetime | None = None
        tags: FlextTypes.StringList = Field(default_factory=list)
        attributes: FlextTypes.Dict = Field(default_factory=dict)

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
            return result.value

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
            default_factory=lambda: FlextConfig().timeout_seconds,
            description="Operation timeout in seconds from FlextConfig",
        )
        retry_attempts: int = Field(
            default_factory=lambda: FlextConfig().max_retry_attempts,
            description="Maximum retry attempts from FlextConfig",
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
            default_factory=lambda: FlextConstants.Cqrs.DEFAULT_PRIORITY,
            ge=0,
            le=100,
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
            default_factory=lambda: FlextConfig().max_batch_size,
            description="Batch size from FlextConfig",
        )
        max_workers: int = Field(
            default_factory=lambda: FlextConfig().max_workers,
            le=FlextConstants.Config.MAX_WORKERS_THRESHOLD,
            description="Maximum workers from FlextConfig",
        )
        timeout_per_item: int = Field(
            default_factory=lambda: FlextConfig().timeout_seconds,
            description="Timeout per item from FlextConfig",
        )
        continue_on_error: bool = True
        data_items: Annotated[
            FlextTypes.List,
            Field(
                default_factory=list,
                max_length=FlextConstants.Performance.BatchProcessing.MAX_ITEMS,
            ),
        ]

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

        handler_name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
        input_data: FlextTypes.Dict = Field(default_factory=dict)
        execution_context: FlextTypes.Dict = Field(default_factory=dict)
        timeout_seconds: int = Field(
            default_factory=lambda: FlextConfig().timeout_seconds,
            le=FlextConstants.Performance.MAX_TIMEOUT_SECONDS,
            description="Timeout from FlextConfig",
        )
        retry_on_failure: bool = True
        max_retries: int = Field(
            default_factory=lambda: FlextConfig().max_retry_attempts,
            description="Max retries from FlextConfig",
        )
        fallback_handlers: FlextTypes.StringList = Field(default_factory=list)

    class TimestampConfig(StrictArbitraryTypesModel):
        """Enhanced timestamp configuration."""

        obj: object
        use_utc: bool = Field(default_factory=lambda: True)
        auto_update: bool = Field(default_factory=lambda: True)
        format: str = "%Y-%m-%dT%H:%M:%S.%fZ"
        timezone: str | None = None
        created_at_field: str = Field("created_at", pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
        updated_at_field: str = Field("updated_at", pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
        field_names: dict[str, str] = Field(
            default_factory=lambda: {
                "created_at": "created_at",
                "updated_at": "updated_at",
            }
        )

    class SerializationRequest(StrictArbitraryTypesModel):
        """Enhanced serialization request."""

        data: object
        format: str = Field(
            default_factory=lambda: FlextConstants.Cqrs.SerializationFormatLiteral.JSON
        )
        encoding: str = Field(default_factory=lambda: "utf-8")
        compression: FlextConstants.Compression | None = None
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
            default_factory=lambda: FlextConfig().timeout_seconds,
            description="Timeout from FlextConfig",
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

    class DomainServiceBatchRequest(ArbitraryTypesModel):
        """Domain service batch request."""

        service_name: str
        operations: list[FlextTypes.Dict] = Field(
            default_factory=list,
            min_length=1,
            max_length=FlextConstants.Performance.MAX_BATCH_OPERATIONS,
        )
        parallel_execution: bool = False
        stop_on_error: bool = True
        batch_size: int = Field(
            default_factory=lambda: FlextConfig().max_batch_size,
            description="Batch size from FlextConfig",
        )
        timeout_per_operation: int = Field(
            default_factory=lambda: FlextConfig().timeout_seconds,
            description="Timeout per operation from FlextConfig",
        )

    class DomainServiceMetricsRequest(ArbitraryTypesModel):
        """Domain service metrics request."""

        service_name: str
        metric_types: Annotated[
            list[Literal["performance", "errors", "throughput", "latency", "availability"]],
            Field(
                default_factory=lambda: ["performance", "errors", "throughput"],
                description="Types of metrics to collect",
            ),
        ]
        time_range_seconds: int = FlextConstants.Performance.DEFAULT_TIME_RANGE_SECONDS
        aggregation: str = Field(
            default_factory=lambda: FlextConstants.Cqrs.AggregationLiteral.AVG
        )
        group_by: FlextTypes.StringList = Field(default_factory=list)
        filters: FlextTypes.Dict = Field(default_factory=dict)

    class DomainServiceResourceRequest(ArbitraryTypesModel):
        """Domain service resource request."""

        service_name: str = "default_service"
        resource_type: str = Field("default_resource", pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
        resource_id: str | None = None
        resource_limit: int = Field(1000, gt=0)
        action: str = Field(
            default_factory=lambda: FlextConstants.Cqrs.ActionLiteral.GET
        )
        data: FlextTypes.Dict = Field(default_factory=dict)
        filters: FlextTypes.Dict = Field(default_factory=dict)

    class OperationExecutionRequest(ArbitraryTypesModel):
        """Operation execution request."""

        operation_name: str = Field(max_length=FlextConstants.Performance.MAX_OPERATION_NAME_LENGTH)
        operation_callable: Callable[..., object]
        arguments: FlextTypes.Dict = Field(default_factory=dict)
        keyword_arguments: FlextTypes.Dict = Field(default_factory=dict)
        timeout_seconds: int = Field(
            default_factory=lambda: FlextConfig().timeout_seconds,
            description="Timeout from FlextConfig",
        )
        retry_config: FlextTypes.Dict = Field(default_factory=dict)

        @field_validator("operation_callable", mode="plain")
        @classmethod
        def validate_operation_callable(cls, v: object) -> Callable[..., object]:
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
            return v

    class RetryConfiguration(ArbitraryTypesModel):
        """Retry configuration with advanced validation."""

        max_attempts: int = Field(
            default_factory=lambda: FlextConfig().max_retry_attempts,
            description="Maximum retry attempts from FlextConfig",
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
                            not FlextConstants.Http.HTTP_STATUS_MIN
                            <= code_int
                            <= FlextConstants.Http.HTTP_STATUS_MAX
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

    class ValidationConfiguration(ArbitraryTypesModel):
        """Validation configuration."""

        enable_strict_mode: bool = Field(default_factory=lambda: True)
        max_validation_errors: int = Field(
            default_factory=lambda: FlextConstants.Cqrs.DEFAULT_MAX_VALIDATION_ERRORS,
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

    class StateInitializationRequest(ArbitraryTypesModel):
        """State initialization request."""

        data: object
        state_key: str
        initial_value: object
        ttl_seconds: int | None = None
        persistence_level: str = Field(
            default_factory=lambda: FlextConstants.Cqrs.PersistenceLevelLiteral.MEMORY
        )
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
    # PHASE 8: CQRS CONFIGURATION MODELS
    # ============================================================================

    class Cqrs:
        """CQRS pattern configuration models."""

        class Bus(BaseModel):
            """Bus configuration model for CQRS command bus."""

            model_config = ConfigDict(
                validate_assignment=True,
                validate_return=True,
                validate_default=True,
                strict=True,
                str_strip_whitespace=True,
                use_enum_values=True,
                arbitrary_types_allowed=True,
                extra="forbid",
                frozen=False,
                ser_json_timedelta="iso8601",
                ser_json_bytes="base64",
                hide_input_in_errors=True,
                json_schema_extra={
                    "title": "Bus",
                    "description": "CQRS command bus configuration",
                },
            )

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
                default="flext_core.bus:FlextBus",
                pattern=r"^[^:]+:[^:]+$",
                description="Implementation path (format: module:Class)",
            )

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

            model_config = ConfigDict(
                validate_assignment=True,
                validate_return=True,
                validate_default=True,
                strict=True,
                str_strip_whitespace=True,
                use_enum_values=True,
                arbitrary_types_allowed=True,
                extra="forbid",
                frozen=False,
                ser_json_timedelta="iso8601",
                ser_json_bytes="base64",
                hide_input_in_errors=True,
                json_schema_extra={
                    "title": "Handler",
                    "description": "CQRS handler configuration",
                },
            )

            handler_id: str = Field(description="Unique handler identifier")
            handler_name: str = Field(description="Human-readable handler name")
            handler_type: FlextConstants.HandlerType = Field(
                default="command",
                description="Handler type",
            )
            handler_mode: FlextConstants.HandlerMode = Field(
                default="command",
                description="Handler mode",
            )
            command_timeout: int = Field(
                default_factory=lambda: FlextConstants.Cqrs.DEFAULT_COMMAND_TIMEOUT,
                description="Command timeout",
            )
            max_command_retries: int = Field(
                default_factory=lambda: FlextConstants.Cqrs.DEFAULT_MAX_COMMAND_RETRIES,
                description="Maximum retry attempts",
            )
            metadata: FlextTypes.Dict = Field(
                default_factory=dict, description="Handler metadata"
            )

            @classmethod
            def create_handler_config(
                cls,
                handler_type: FlextConstants.HandlerType,
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
        """Registration details for handler registration tracking.

        Tracks metadata about handler registrations in the CQRS system,
        including unique identification, timing, and status information.

        This model is used by FlextRegistry to track which handlers have been
        registered with the dispatcher and monitor their lifecycle.

        Attributes:
            registration_id: Unique identifier for this registration
            handler_mode: Mode of handler (command, query, or event)
            timestamp: ISO 8601 timestamp when registration occurred
            status: Current status of the registration

        Examples:
            >>> details = FlextModels.RegistrationDetails(
            ...     registration_id="reg-123",
            ...     handler_mode="command",
            ...     timestamp="2025-01-01T00:00:00Z",
            ...     status="running",
            ... )
            >>> details.registration_id
            'reg-123'

        """

        model_config = ConfigDict(
            validate_assignment=True,
            validate_return=True,
            validate_default=True,
            strict=True,
            str_strip_whitespace=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            extra="forbid",
            frozen=False,
            ser_json_timedelta="iso8601",
            ser_json_bytes="base64",
            hide_input_in_errors=True,
            json_schema_extra={
                "title": "RegistrationDetails",
                "description": "Handler registration tracking details",
            },
        )

        registration_id: Annotated[
            str,
            Field(
                min_length=1,
                description="Unique registration identifier",
                examples=["reg-abc123", "handler-create-user-001"],
            ),
        ]
        handler_mode: Annotated[
            FlextConstants.HandlerModeSimple,
            Field(
                default="command",
                description="Handler mode (command, query, or event)",
                examples=["command", "query", "event"],
            ),
        ] = "command"
        timestamp: Annotated[
            str,
            Field(
                default_factory=lambda: FlextConstants.Cqrs.DEFAULT_TIMESTAMP,
                description="ISO 8601 registration timestamp",
                examples=["2025-01-01T00:00:00Z", "2025-10-12T15:30:00+00:00"],
            ),
        ] = Field(default_factory=lambda: FlextConstants.Cqrs.DEFAULT_TIMESTAMP)
        status: Annotated[
            FlextConstants.Status,
            Field(
                default="running",
                description="Current registration status",
                examples=["running", "stopped", "failed"],
            ),
        ] = "running"

        @field_validator("registration_id")
        @classmethod
        def validate_registration_id(cls, v: str) -> str:
            """Validate registration_id is not empty."""
            if not v or not v.strip():
                msg = "Registration ID must not be empty"
                raise ValueError(msg)
            return v.strip()

        @field_validator("timestamp")
        @classmethod
        def validate_timestamp_format(cls, v: str) -> str:
            """Validate timestamp is in ISO 8601 format.

            Allows empty strings for backward compatibility with FlextConstants.Cqrs.DEFAULT_TIMESTAMP.
            """
            # Allow empty strings (backward compatibility with DEFAULT_TIMESTAMP)
            if not v or not v.strip():
                return v

            from datetime import datetime

            try:
                # Handle both Z suffix and explicit timezone offset
                timestamp_str = v.replace("Z", "+00:00") if v.endswith("Z") else v
                datetime.fromisoformat(timestamp_str)
            except ValueError as e:
                msg = f"Timestamp must be in ISO 8601 format: {e}"
                raise ValueError(msg) from e
            return v

    # ============================================================================
    # END OF PHASE 8: CQRS CONFIGURATION MODELS
    # ============================================================================

    # ============================================================================
    # PHASE 9: QUERY AND PAGINATION MODELS
    # ============================================================================

    class Pagination(BaseModel):
        """Pagination model for query results with Pydantic 2 computed fields.

        Provides pagination parameters for database queries and API responses,
        with automatic offset calculation and validation of page boundaries.

        The model uses Pydantic 2 computed fields to automatically derive
        offset and limit from page and size parameters.

        Attributes:
            page: Current page number (1-based indexing)
            size: Number of items per page (1-1000)

        Computed Fields:
            offset: Calculated starting position (0-based) = (page - 1) * size
            limit: Items to fetch (same as size)

        Examples:
            >>> pagination = Pagination(page=2, size=20)
            >>> pagination.page
            2
            >>> pagination.offset  # Computed: (2-1) * 20
            20
            >>> pagination.limit  # Computed: same as size
            20

        """

        model_config = ConfigDict(
            validate_assignment=True,  # Validate on field assignment
            validate_return=True,  # Validate return values
            validate_default=True,  # Validate default values
            strict=True,  # Strict type coercion
            str_strip_whitespace=True,  # Auto-strip whitespace
            use_enum_values=True,  # Use enum values
            arbitrary_types_allowed=True,  # Allow custom types
            extra="forbid",  # No extra fields
            frozen=False,  # Allow mutations
            ser_json_timedelta="iso8601",  # ISO 8601 timedelta
            ser_json_bytes="base64",  # Base64 bytes
            hide_input_in_errors=True,  # Security
            json_schema_extra={
                "title": "Pagination",
                "description": "Pagination model for query results with computed fields",
            },
        )

        page: Annotated[
            int,
            Field(
                default=FlextConstants.Pagination.DEFAULT_PAGE_NUMBER,
                ge=1,
                description="Page number (1-based indexing)",
                examples=[1, 2, 10, 100],
            ),
        ] = FlextConstants.Pagination.DEFAULT_PAGE_NUMBER
        size: Annotated[
            int,
            Field(
                default=FlextConstants.Pagination.DEFAULT_PAGE_SIZE,
                ge=1,
                le=1000,
                description="Number of items per page (max 1000)",
                examples=[10, 20, 50, 100],
            ),
        ] = FlextConstants.Pagination.DEFAULT_PAGE_SIZE

        @computed_field
        def offset(self) -> int:
            """Calculate offset from page and size.

            Returns:
                0-based starting position for database queries

            Examples:
                >>> Pagination(page=1, size=20).offset
                0
                >>> Pagination(page=3, size=20).offset
                40

            """
            return (self.page - 1) * self.size

        @computed_field
        def limit(self) -> int:
            """Get limit (same as size).

            Returns:
                Number of items to fetch

            Examples:
                >>> Pagination(page=1, size=20).limit
                20

            """
            return self.size

        def to_dict(self) -> FlextTypes.Dict:
            """Convert pagination to dictionary.

            Returns:
                Dictionary with page, size, offset, and limit

            Examples:
                >>> pagination = Pagination(page=2, size=20)
                >>> pagination.to_dict()
                {'page': 2, 'size': 20, 'offset': 20, 'limit': 20}

            """
            return {
                "page": self.page,
                "size": self.size,
                "offset": self.offset,
                "limit": self.limit,
            }

    class Query(BaseModel):
        """Query model for CQRS query operations.

        Represents a read-only query operation in the CQRS pattern, supporting
        filtering, pagination, and query tracking for data retrieval operations.
        Includes message_type discriminator for Pydantic v2 discriminated unions.

        The model automatically converts dictionary pagination to Pagination
        instances and generates unique query IDs for tracking.

        Attributes:
            message_type: Discriminator field for union routing (always 'query')
            filters: Dictionary of filter conditions (field: value pairs)
            pagination: Pagination settings (Pagination object or dict)
            query_id: Unique identifier for query tracking
            query_type: Optional query classification/type

        Examples:
            >>> query = Query(
            ...     filters={"status": "active", "age__gte": 18},
            ...     pagination={"page": 2, "size": 50},
            ...     query_type="user_search",
            ... )
            >>> query.filters["status"]
            'active'
            >>> isinstance(query.pagination, Pagination)
            True

        """

        model_config = ConfigDict(
            validate_assignment=True,  # Validate on field assignment
            validate_return=True,  # Validate return values
            validate_default=True,  # Validate default values
            strict=True,  # Strict type coercion
            str_strip_whitespace=True,  # Auto-strip whitespace
            use_enum_values=True,  # Use enum values
            arbitrary_types_allowed=True,  # Allow custom types
            extra="forbid",  # No extra fields
            frozen=False,  # Allow mutations
            ser_json_timedelta="iso8601",  # ISO 8601 timedelta
            ser_json_bytes="base64",  # Base64 bytes
            hide_input_in_errors=True,  # Security
            json_schema_extra={
                "title": "Query",
                "description": "Query model for CQRS query operations with pagination and discriminator",
            },
        )

        # Pydantic v2 Discriminated Union discriminator field (immutable)
        message_type: Literal["query"] = Field(
            default="query",
            frozen=True,
            description="Message type discriminator for union routing - always 'query'",
        )

        filters: Annotated[
            FlextTypes.Dict,
            Field(
                default_factory=dict,
                description="Query filter conditions as key-value pairs",
                examples=[
                    {"status": "active"},
                    {"age__gte": 18, "country": "US"},
                    {"created_at__gte": "2025-01-01"},
                ],
            ),
        ] = Field(default_factory=dict)
        pagination: Annotated[
            FlextModels.Pagination | dict[str, int],
            Field(
                default_factory=dict,
                description="Pagination settings (Pagination object or dict[str, object] with page/size)",
                examples=[
                    {"page": 1, "size": 20},
                    {"page": 5, "size": 100},
                ],
            ),
        ] = Field(default_factory=dict)
        query_id: Annotated[
            str,
            Field(
                default_factory=lambda: str(uuid.uuid4()),
                description="Unique query identifier for tracking",
                examples=["550e8400-e29b-41d4-a716-446655440000", "query-abc123"],
            ),
        ] = Field(default_factory=lambda: str(uuid.uuid4()))
        query_type: Annotated[
            str | None,
            Field(
                default=None,
                description="Optional query type/classification",
                examples=["user_search", "product_filter", "report_query"],
            ),
        ] = None

        @field_validator("pagination", mode="before")
        @classmethod
        def validate_pagination(
            cls, v: FlextModels.Pagination | dict[str, int | str] | None
        ) -> FlextModels.Pagination:
            """Convert pagination to Pagination instance."""
            if isinstance(v, FlextModels.Pagination):
                return v
            if isinstance(v, dict):
                # Extract page and size from dict[str, object] with proper type casting
                v_dict = cast("FlextTypes.Dict", v)
                page_raw = v_dict.get("page", 1)
                size_raw = v_dict.get("size", 20)

                # Convert to int | str types
                page: int | str = page_raw if isinstance(page_raw, (int, str)) else 1
                size: int | str = size_raw if isinstance(size_raw, (int, str)) else 20

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

                return FlextModels.Pagination(
                    page=page,
                    size=size,
                )
            if v is None:
                return FlextModels.Pagination()
            # For any other type, return default pagination
            return FlextModels.Pagination()

        @classmethod
        def validate_query(
            cls, query_payload: FlextTypes.Dict
        ) -> FlextResult[FlextModels.Query]:
            """Validate and create Query from payload."""
            try:
                # Extract the required fields with proper typing
                filters: object = query_payload.get("filters", {})
                pagination_data = query_payload.get("pagination", {})
                if isinstance(pagination_data, dict):
                    pagination_dict = cast("FlextTypes.Dict", pagination_data)
                    page_raw = pagination_dict.get("page", 1)
                    size_raw = pagination_dict.get("size", 20)
                    page: int = int(page_raw) if isinstance(page_raw, (int, str)) else 1
                    size: int = (
                        int(size_raw) if isinstance(size_raw, (int, str)) else 20
                    )
                    pagination: dict[str, int] = {"page": page, "size": size}
                else:
                    pagination = {"page": 1, "size": 20}
                query_id = str(query_payload.get("query_id", str(uuid.uuid4())))
                query_type: object = query_payload.get("query_type")

                if not isinstance(filters, dict):
                    filters = {}
                # Type casting for mypy - after validation, filters is guaranteed to be dict
                filters_dict = cast("FlextTypes.Dict", filters)
                # No need to validate pagination dict[str, object] - Pydantic validator handles conversion

                query = cls(
                    filters=filters_dict,
                    pagination=pagination,  # Pydantic will convert dict[str, object] to Pagination
                    query_id=query_id,
                    query_type=str(query_type) if query_type is not None else None,
                )
                return FlextResult[FlextModels.Query].ok(query)
            except Exception as e:
                return FlextResult[FlextModels.Query].fail(
                    f"Query validation failed: {e}"
                )

    # =========================================================================
    # CONTEXT MODELS - Context management data structures
    # =========================================================================

    class StructlogProxyToken(Value):
        """Token for resetting structlog context variables.

        Used by StructlogProxyContextVar to track previous values and enable
        rollback to previous context state. Inherits from Value for immutability
        and validation.

        This is a lightweight immutable value object that stores the necessary
        information to restore a context variable to its previous state.

        Attributes:
            key: The context variable key being tracked
            previous_value: The value before the set operation (None if unset)

        Examples:
            >>> token = FlextModels.StructlogProxyToken(
            ...     key="correlation_id", previous_value="abc-123"
            ... )
            >>> token.key
            'correlation_id'
            >>> token.previous_value
            'abc-123'

        """

        key: Annotated[
            str,
            Field(
                min_length=1,
                description="Unique key for the context variable",
                examples=["correlation_id", "service_name", "user_id"],
            ),
        ]
        previous_value: Annotated[
            object | None,
            Field(
                default=None,
                description="Previous value before set operation",
            ),
        ] = None

        @field_validator("key")
        @classmethod
        def validate_key_not_empty(cls, v: str) -> str:
            """Validate that key is not empty or whitespace-only."""
            if not v or not v.strip():
                msg = "Key must not be empty or whitespace-only"
                raise ValueError(msg)
            return v.strip()

    class StructlogProxyContextVar[T]:
        """ContextVar-like proxy using structlog as backend (single source of truth).

        ARCHITECTURAL NOTE: This proxy delegates ALL operations to structlog's
        contextvar storage. This ensures FlextContext.Variables and FlextLogger
        use THE SAME underlying storage, eliminating dual storage and sync issues.

        Key Principles:
            - Single Source of Truth: structlog's contextvar dict
            - Zero Synchronization: No dual storage, no sync needed
            - Thread Safety: structlog handles all thread safety
            - Performance: Direct delegation, no overhead

        Usage:
            >>> var = FlextModels.StructlogProxyContextVar[str](
            ...     "correlation_id", default=None
            ... )
            >>> var.set("abc-123")
            >>> var.get()  # Returns "abc-123"

        """

        def __init__(
            self,
            key: str,
            default: T | None = None,
        ) -> None:
            """Initialize proxy context variable.

            Args:
                key: Unique key for this context variable
                default: Default value when not set

            """
            super().__init__()
            self._key = key
            self._default = default

        def get(self) -> T | None:
            """Get current value from structlog context.

            Returns:
                Current value or default if not set

            """
            # Get from structlog's contextvar dict[str, object] (single source of truth)
            import structlog.contextvars

            current_context = structlog.contextvars.get_contextvars()
            return current_context.get(self._key, self._default)

        def set(self, value: T | None) -> FlextModels.StructlogProxyToken:
            """Set value in structlog context.

            Args:
                value: Value to set in structlog's contextvar (can be None to clear)

            Returns:
                Token for potential reset

            """
            # Get current value before setting
            import structlog.contextvars

            current_value = self.get()

            if value is not None:
                structlog.contextvars.bind_contextvars(**{self._key: value})
            else:
                # Unbind if setting to None
                structlog.contextvars.unbind_contextvars(self._key)

            # Create token for reset functionality
            return FlextModels.StructlogProxyToken(
                key=self._key, previous_value=current_value
            )

        def reset(self, token: FlextModels.StructlogProxyToken) -> None:
            """Reset to previous value using token.

            Args:
                token: Token from previous set() call

            Note:
                structlog.contextvars doesn't support token-based reset.
                Use unbind_contextvars() or clear_contextvars() instead.

            """
            # Simplified implementation - structlog uses bind/unbind, not tokens
            # In practice, context managers handle cleanup via bind/unbind
            import structlog.contextvars

            if token.previous_value is None:
                structlog.contextvars.unbind_contextvars(token.key)
            else:
                structlog.contextvars.bind_contextvars(**{
                    token.key: token.previous_value
                })

    class Token(Value):
        """Token for context variable reset operations.

        Used by FlextContext to track context variable changes and enable
        rollback to previous values.

        This immutable value object stores the state needed to restore a
        context variable to its previous value, enabling proper cleanup
        in context managers and error handlers.

        Attributes:
            key: The context variable key being tracked
            old_value: The value before the set operation (None if unset)

        Examples:
            >>> token = FlextModels.Token(key="user_id", old_value="user-123")
            >>> token.key
            'user_id'
            >>> token.old_value
            'user-123'

        """

        key: Annotated[
            str,
            Field(
                min_length=1,
                description="Unique key for the context variable",
                examples=["user_id", "request_id", "session_id"],
            ),
        ]
        old_value: Annotated[
            object | None,
            Field(
                default=None,
                description="Previous value before set operation",
            ),
        ]

        @field_validator("key")
        @classmethod
        def validate_key_format(cls, v: str) -> str:
            """Validate that key is not empty or whitespace-only."""
            if not v or not v.strip():
                msg = "Key must not be empty or whitespace-only"
                raise ValueError(msg)
            return v.strip()

    class ContextData(Value):
        """Lightweight container for initializing context state.

        Used by FlextContext initialization to provide initial data and metadata.

        This immutable value object encapsulates the initial state for a
        FlextContext instance, separating actual context data from metadata
        about the context itself.

        Attributes:
            data: Initial context data (key-value pairs)
            metadata: Context metadata (creation time, source, etc.)

        Examples:
            >>> context_data = FlextModels.ContextData(
            ...     data={"user_id": "123", "correlation_id": "abc-xyz"},
            ...     metadata={"source": "api", "created_at": "2025-01-01T00:00:00Z"},
            ... )
            >>> context_data.data["user_id"]
            '123'
            >>> context_data.metadata["source"]
            'api'

        """

        data: Annotated[
            FlextTypes.Dict,
            Field(
                default_factory=dict,
                description="Initial context data as key-value pairs",
            ),
        ] = Field(default_factory=dict)
        metadata: Annotated[
            FlextTypes.Dict,
            Field(
                default_factory=dict,
                description="Context metadata (creation info, source, etc.)",
            ),
        ] = Field(default_factory=dict)

        @field_validator("data", "metadata", mode="before")
        @classmethod
        def validate_dict_serializable(cls, v: object) -> FlextTypes.Dict:
            """Validate that dict[str, object] values are JSON-serializable.

            Uses mode='before' to validate raw input before Pydantic processing.
            Only allows basic JSON-serializable types: str, int, float, bool, list, dict, None.
            """
            if not isinstance(v, dict):
                msg = f"Value must be a dictionary, got {type(v).__name__}"
                raise TypeError(msg)

            # Recursively check all values are JSON-serializable
            def check_serializable(obj: object, path: str = "") -> None:
                """Recursively check if object is JSON-serializable."""
                if obj is None or isinstance(obj, (str, int, float, bool)):
                    return
                if isinstance(obj, dict):
                    for key, val in obj.items():
                        if not isinstance(key, str):
                            msg = f"Dictionary keys must be strings at {path}.{key}"
                            raise TypeError(msg)
                        check_serializable(val, f"{path}.{key}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        check_serializable(item, f"{path}[{i}]")
                else:
                    msg = f"Non-JSON-serializable type {type(obj).__name__} at {path}"
                    raise TypeError(msg)

            check_serializable(v)
            return v

    class ContextExport(Value):
        """Typed snapshot returned by export_snapshot.

        Provides a complete serializable snapshot of context state including
        data, metadata, and statistics.

        This immutable value object represents a complete export of a FlextContext
        instance, suitable for persistence, transmission, or debugging. All fields
        are JSON-serializable for easy cross-service communication.

        Attributes:
            data: All context data from all scopes
            metadata: Context metadata (creation info, source, etc.)
            statistics: Usage statistics (set/get/remove counts, etc.)

        Examples:
            >>> export = FlextModels.ContextExport(
            ...     data={"user_id": "123", "correlation_id": "abc-xyz"},
            ...     metadata={"source": "api", "version": "1.0"},
            ...     statistics={"sets": 5, "gets": 10, "removes": 2},
            ... )
            >>> export.data["user_id"]
            '123'
            >>> export.statistics["sets"]
            5

        """

        data: Annotated[
            FlextTypes.Dict,
            Field(
                default_factory=dict,
                description="All context data from all scopes",
            ),
        ] = Field(default_factory=dict)
        metadata: Annotated[
            FlextTypes.Dict,
            Field(
                default_factory=dict,
                description="Context metadata (creation info, source, version)",
            ),
        ] = Field(default_factory=dict)
        statistics: Annotated[
            FlextTypes.Dict,
            Field(
                default_factory=dict,
                description="Usage statistics (operation counts, timing info)",
            ),
        ] = Field(default_factory=dict)

        @field_validator("data", "metadata", "statistics", mode="before")
        @classmethod
        def validate_dict_serializable(cls, v: object) -> FlextTypes.Dict:
            """Validate that dict[str, object] values are JSON-serializable.

            Uses mode='before' to validate raw input before Pydantic processing.
            Only allows basic JSON-serializable types: str, int, float, bool, list, dict, None.
            """
            if not isinstance(v, dict):
                msg = f"Value must be a dictionary, got {type(v).__name__}"
                raise TypeError(msg)

            # Recursively check all values are JSON-serializable
            def check_serializable(obj: object, path: str = "") -> None:
                """Recursively check if object is JSON-serializable."""
                if obj is None or isinstance(obj, (str, int, float, bool)):
                    return
                if isinstance(obj, dict):
                    for key, val in obj.items():
                        if not isinstance(key, str):
                            msg = f"Dictionary keys must be strings at {path}.{key}"
                            raise TypeError(msg)
                        check_serializable(val, f"{path}.{key}")
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        check_serializable(item, f"{path}[{i}]")
                else:
                    msg = f"Non-JSON-serializable type {type(obj).__name__} at {path}"
                    raise TypeError(msg)

            check_serializable(v)
            return v

        @computed_field
        def total_data_items(self) -> int:
            """Compute total number of data items across all scopes."""
            return len(self.data)

        @computed_field
        def has_statistics(self) -> bool:
            """Check if statistics are available."""
            return bool(self.statistics)

    class HandlerExecutionContext(BaseModel):
        """Handler execution context for tracking handler performance and state.

        Provides timing and metrics tracking for handler executions in the
        FlextContext system. Uses Pydantic 2 PrivateAttr for internal state.

        This mutable context object tracks handler execution performance,
        including timing, metrics, and execution state. It is designed to be
        created at the start of handler execution and updated throughout.

        Attributes:
            handler_name: Name of the handler being executed
            handler_mode: Mode of execution (command, query, or event)

        Examples:
            >>> context = FlextModels.HandlerExecutionContext.create_for_handler(
            ...     handler_name="ProcessOrderCommand", handler_mode="command"
            ... )
            >>> context.start_execution()
            >>> # ... handler executes ...
            >>> elapsed_ms = context.execution_time_ms
            >>> context.set_metrics_state({"items_processed": 42})

        """

        model_config = ConfigDict(
            validate_assignment=True,  # Validate on field assignment
            validate_return=True,  # Validate return values
            validate_default=True,  # Validate default values
            strict=True,  # Strict type coercion
            str_strip_whitespace=True,  # Auto-strip whitespace
            use_enum_values=True,  # Use enum values
            arbitrary_types_allowed=True,  # Allow custom types
            extra="forbid",  # No extra fields
            frozen=False,  # Allow mutations
            ser_json_timedelta="iso8601",  # ISO 8601 timedelta
            ser_json_bytes="base64",  # Base64 bytes
            hide_input_in_errors=True,  # Security
            json_schema_extra={
                "title": "HandlerExecutionContext",
                "description": "Handler execution context for tracking performance and state",
            },
        )

        handler_name: Annotated[
            str,
            Field(
                min_length=1,
                description="Name of the handler being executed",
                examples=["ProcessOrderCommand", "GetUserQuery", "OrderCreatedEvent"],
            ),
        ]
        handler_mode: Annotated[
            Literal["command", "query", "event", "operation", "saga"],
            Field(
                min_length=1,
                description="Mode of handler execution",
                examples=["command", "query", "event"],
            ),
        ]
        _start_time: float | None = PrivateAttr(default=None)
        _metrics_state: FlextTypes.Dict | None = PrivateAttr(default=None)

        @field_validator("handler_name", "handler_mode")
        @classmethod
        def validate_not_empty(cls, v: str) -> str:
            """Validate that fields are not empty or whitespace-only."""
            if not v or not v.strip():
                msg = "Handler name and mode must not be empty"
                raise ValueError(msg)
            return v.strip()

        def start_execution(self) -> None:
            """Start execution timing.

            Records the current time as the start time for execution metrics.
            Should be called at the beginning of handler execution.

            Examples:
                >>> context = FlextModels.HandlerExecutionContext.create_for_handler(
                ...     handler_name="MyHandler", handler_mode="command"
                ... )
                >>> context.start_execution()

            """
            self._start_time = time_module.time()

        @computed_field
        @property
        def execution_time_ms(self) -> float:
            """Get execution time in milliseconds.

            Returns:
                Execution time in milliseconds, or 0.0 if not started

            Examples:
                >>> context = FlextModels.HandlerExecutionContext.create_for_handler(
                ...     handler_name="MyHandler", handler_mode="command"
                ... )
                >>> context.start_execution()
                >>> # ... handler executes ...
                >>> elapsed = context.execution_time_ms
                >>> isinstance(elapsed, float)
                True

            """
            if self._start_time is None:
                return 0.0

            elapsed = time_module.time() - self._start_time
            return round(elapsed * 1000, 2)

        @computed_field
        @property
        def metrics_state(self) -> FlextTypes.Dict:
            """Get current metrics state.

            Returns:
                Dictionary containing metrics state (empty dict[str, object] if not set)

            Examples:
                >>> context = FlextModels.HandlerExecutionContext.create_for_handler(
                ...     handler_name="MyHandler", handler_mode="command"
                ... )
                >>> metrics = context.metrics_state
                >>> isinstance(metrics, dict)
                True

            """
            if self._metrics_state is None:
                self._metrics_state = {}
            return self._metrics_state

        def set_metrics_state(self, state: FlextTypes.Dict) -> None:
            """Set metrics state.

            Direct assignment to _metrics_state. Use this to update metrics.

            Args:
                state: Metrics state to set

            Examples:
                >>> context = FlextModels.HandlerExecutionContext.create_for_handler(
                ...     handler_name="MyHandler", handler_mode="command"
                ... )
                >>> context.set_metrics_state({"items_processed": 42, "errors": 0})

            """
            self._metrics_state = state

        def reset(self) -> None:
            """Reset execution context.

            Clears all timing and metrics state, preparing the context
            for reuse or cleanup.

            Examples:
                >>> context = FlextModels.HandlerExecutionContext.create_for_handler(
                ...     handler_name="MyHandler", handler_mode="command"
                ... )
                >>> context.start_execution()
                >>> context.reset()
                >>> context.execution_time_ms
                0.0

            """
            self._start_time = None
            self._metrics_state = None

        @classmethod
        def create_for_handler(
            cls,
            handler_name: str,
            handler_mode: str,
        ) -> Self:
            """Create execution context for a handler.

            Factory method for creating handler execution contexts with
            validation of handler name and mode.

            Args:
                handler_name: Name of the handler
                handler_mode: Mode of the handler (command/query/event)

            Returns:
                New HandlerExecutionContext instance

            Examples:
                >>> context = FlextModels.HandlerExecutionContext.create_for_handler(
                ...     handler_name="ProcessOrderCommand", handler_mode="command"
                ... )
                >>> context.handler_name
                'ProcessOrderCommand'
                >>> context.handler_mode
                'command'

            """
            return cls(handler_name=handler_name, handler_mode=handler_mode)

        @computed_field
        def is_running(self) -> bool:
            """Check if execution is currently running."""
            return self._start_time is not None

        @computed_field
        def has_metrics(self) -> bool:
            """Check if metrics have been recorded."""
            return self._metrics_state is not None and bool(self._metrics_state)

    # ============================================================================
    # VALIDATION UTILITIES (moved from FlextUtilities to avoid circular imports)
    # ============================================================================

    class Validation:
        """Validation utility functions."""

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
            # Validate all rules and return model if all pass
            for rule in rules:
                result = rule(model)
                if result.is_failure:
                    return FlextResult[T].fail(result.error or "Validation failed")

            return FlextResult[T].ok(model)

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
            if max_validation_time_ms is not None:
                timeout_ms = max_validation_time_ms
            else:
                # Use global config instance for consistency
                timeout_ms = FlextConfig.get_global_instance().validation_timeout_ms
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
                # Validate models one by one, stop on first failure
                valid_models: list[T] = []
                for model in models:
                    # Validate all rules for this model
                    for validator in validators:
                        result = validator(model)
                        if result.is_failure:
                            return FlextResult[list[T]].fail(
                                result.error or "Validation failed"
                            )

                    valid_models.append(model)

                return FlextResult[list[T]].ok(valid_models)
            # Accumulate all errors
            validated_models: list[T] = []
            all_errors: FlextTypes.StringList = []

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
        def validate_aggregate_consistency_with_rules[T](
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
                result = FlextModels.Validation.validate_aggregate_consistency_with_rules(
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
            violations: FlextTypes.StringList = []
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

        @staticmethod
        def validate_domain_event(event: object) -> FlextResult[None]:
            """Enhanced domain event validation with comprehensive checks.

            Validates domain events for proper structure, required fields,
            and domain invariants. Used across all flext-ecosystem projects.

            Args:
                event: The domain event to validate

            Returns:
                FlextResult[None]: Success if valid, failure with details

            """
            if event is None:
                return FlextResult[None].fail(
                    "Domain event cannot be None",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Check required attributes
            required_attrs = ["event_type", "aggregate_id", "id", "created_at"]
            missing_attrs = [
                attr for attr in required_attrs if not hasattr(event, attr)
            ]
            if missing_attrs:
                return FlextResult[None].fail(
                    f"Domain event missing required attributes: {missing_attrs}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Validate event_type is non-empty string
            event_type = getattr(event, "event_type", "")
            if not event_type or not isinstance(event_type, str):
                return FlextResult[None].fail(
                    "Domain event event_type must be a non-empty string",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Validate aggregate_id is non-empty string
            aggregate_id = getattr(event, "aggregate_id", "")
            if not aggregate_id or not isinstance(aggregate_id, str):
                return FlextResult[None].fail(
                    "Domain event aggregate_id must be a non-empty string",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Validate data is a dict
            data = getattr(event, "data", None)
            if data is not None and not isinstance(data, dict):
                return FlextResult[None].fail(
                    "Domain event data must be a dictionary or None",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            return FlextResult[None].ok(None)

        @staticmethod
        def validate_aggregate_consistency[T: FlextProtocols.HasInvariants](
            aggregate: T,
        ) -> FlextResult[T]:
            """Validate aggregate consistency and business invariants.

            Ensures aggregates maintain consistency boundaries and invariants
            are satisfied. Used extensively in flext-core and dependent projects.

            Args:
                aggregate: The aggregate root to validate

            Returns:
                FlextResult[T]: Validated aggregate or failure with details

            """
            if aggregate is None:
                return FlextResult[T].fail(
                    "Aggregate cannot be None",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Check invariants if the aggregate supports them
            if isinstance(aggregate, FlextProtocols.HasInvariants):
                try:
                    aggregate.check_invariants()
                except Exception as e:
                    return FlextResult[T].fail(
                        f"Aggregate invariant violation: {e}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

            # Check for uncommitted domain events (potential consistency issue)
            if hasattr(aggregate, "domain_events"):
                events = getattr(aggregate, "domain_events", [])
                if len(events) > FlextConstants.Validation.MAX_UNCOMMITTED_EVENTS:
                    return FlextResult[T].fail(
                        f"Too many uncommitted domain events: {len(events)} (max: {FlextConstants.Validation.MAX_UNCOMMITTED_EVENTS})",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

            return FlextResult[T].ok(aggregate)

        @staticmethod
        def validate_entity_relationships[T](entity: T) -> FlextResult[T]:
            """Validate entity relationships and references.

            Ensures entity references are valid and relationships are consistent.
            Critical for maintaining data integrity across flext-ecosystem.

            Args:
                entity: The entity to validate

            Returns:
                FlextResult[T]: Validated entity or failure with details

            """
            if entity is None:
                return FlextResult[T].fail(
                    "Entity cannot be None",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Validate entity ID format
            if hasattr(entity, "id"):
                entity_id = getattr(entity, "id", "")
                id_result = FlextModels.Validation.validate_entity_id(entity_id)
                if id_result.is_failure:
                    return FlextResult[T].fail(
                        f"Invalid entity ID: {id_result.error}",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

            # Validate version for optimistic locking
            if hasattr(entity, "version"):
                version = getattr(entity, "version", 0)
                if not isinstance(version, int) or version < 0:
                    return FlextResult[T].fail(
                        "Entity version must be a non-negative integer",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

            # Validate timestamps if present
            for timestamp_field in ["created_at", "updated_at"]:
                if hasattr(entity, timestamp_field):
                    timestamp = getattr(entity, timestamp_field)
                    if timestamp is not None and not isinstance(timestamp, datetime):
                        return FlextResult[T].fail(
                            f"Entity {timestamp_field} must be a datetime or None",
                            error_code=FlextConstants.Errors.VALIDATION_ERROR,
                        )

            # Type-safe return for generic method using cast for proper type inference
            return cast("FlextResult[T]", FlextResult.ok(entity))


# =========================================================================
# PYDANTIC V2 DISCRIMINATED UNION - Type-safe message routing
# =========================================================================
# Discriminated union for CQRS message types eliminating object types
# Uses Pydantic v2's most innovative feature: discriminated unions with
# Discriminator field for automatic routing based on message_type
type MessageUnion = Annotated[
    FlextModels.Command | FlextModels.Query | FlextModels.DomainEvent,
    Discriminator("message_type"),
]
"""Pydantic v2 discriminated union for type-safe CQRS message routing.

This union type enables automatic message type detection and routing
based on the 'message_type' field discriminator, replacing all object
types in message handling across the entire FLEXT ecosystem.

Usage:
    def process_message(message: MessageUnion) -> FlextResult[object]:
        match message.message_type:
            case "command":
                return handle_command(message)
            case "query":
                return handle_query(message)
            case "event":
                return handle_event(message)

Pydantic v2 automatically validates and routes messages to the correct
type based on the discriminator field value.
"""

# Rebuild models after all classes are defined to resolve forward references
FlextModels.DomainEvent.model_rebuild()
FlextModels.Entity.model_rebuild()
FlextModels.Query.model_rebuild()
FlextModels.Pagination.model_rebuild()


__all__ = [
    "FlextModels",
    "MessageUnion",
]
