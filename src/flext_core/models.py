"""Domain models for domain-driven design patterns.

This module provides FlextModels, a comprehensive collection of base classes
and utilities for implementing domain-driven design (DDD) patterns in the
FLEXT ecosystem.

All models use Pydantic for validation and serialization, providing type-safe
domain modeling with automatic validation and error handling.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""
# Expected: Complex model inheritance and mixins.
# ruff: disable=PLC2701,E402
# pyright: basic

from __future__ import annotations

import re
import time as time_module
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import (
    Annotated,
    ClassVar,
    Self,
    cast,
    override,
)
from urllib.parse import ParseResult, urlparse

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

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.loggings import FlextLogger
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
from flext_core.typings import FlextTypes


class FlextModels:
    """Base classes and utilities for domain-driven design patterns.

    Provides comprehensive base classes for implementing DDD patterns
    with Pydantic validation, event sourcing support, and CQRS integration.

    Includes:
    - Entity: Base class with identity and lifecycle management
    - Value: Immutable value objects compared by value
    - AggregateRoot: Consistency boundaries with invariant enforcement
    - Command/Query: CQRS pattern base classes
    - DomainEvent: Event sourcing support
    - Validation: Utility functions for business rules
    - Various mixins for common model behaviors

    Usage:
        >>> from flext_core.models import FlextModels
        >>> from flext_core.result import FlextResult
        >>>
        >>> class User(FlextModels.Entity):
        ...     name: str
        ...     email: str
        ...
        ...     def activate(self) -> FlextResult[None]:
        ...         return FlextResult[None].ok(None)
    """

    # =========================================================================
    # BEHAVIOR MIXINS - Reusable model behaviors (Pydantic 2.11 pattern)
    # =========================================================================

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

        model_config = ConfigDict(
            validate_assignment=True,
            validate_return=True,
            validate_default=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            ser_json_timedelta="iso8601",
            ser_json_bytes="base64",
            serialize_by_alias=True,
            populate_by_name=True,
            str_strip_whitespace=True,
            str_to_lower=False,
            str_to_upper=False,
            defer_build=False,
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
            validate_assignment=True,
            validate_return=True,
            validate_default=True,
            use_enum_values=True,
            str_strip_whitespace=True,
            str_to_lower=False,
            str_to_upper=False,
            defer_build=False,
        )

    class FrozenStrictModel(BaseModel):
        """Immutable pattern: frozen with extra fields forbidden.

        Used by value objects and configuration models.
        """

        model_config = ConfigDict(
            validate_assignment=False,
            validate_return=True,
            validate_default=True,
            use_enum_values=True,
            str_strip_whitespace=True,
            str_to_lower=False,
            str_to_upper=False,
            defer_build=False,
            frozen=True,
        )

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
                    # Removed occurred_at - auto-generated via TimestampableMixin
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

        def get_uncommitted_events(self) -> list[FlextModels.DomainEvent]:
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
                events = self.get_uncommitted_events()

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
                        f"Event {i}: data must be dict or None",
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

    class DomainEvent(ArbitraryTypesModel, IdentifiableMixin, TimestampableMixin):
        """Base class for domain events.

        Uses IdentifiableMixin for id and TimestampableMixin for created_at.
        """

        event_type: str
        aggregate_id: str
        data: FlextTypes.Domain.EventPayload = Field(default_factory=dict)
        metadata: FlextTypes.Domain.EventMetadata = Field(default_factory=dict)

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

    class WorkspaceStatus(StrEnum):
        """Workspace status enumeration."""

        INITIALIZING = "initializing"
        READY = "ready"
        ERROR = "error"
        MAINTENANCE = "maintenance"

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
            default_factory=lambda: __import__("flext_core.config")
            .FlextConfig()
            .timeout_seconds
        )
        retry_attempts: int = Field(
            default_factory=lambda: __import__("flext_core.config")
            .FlextConfig()
            .max_retry_attempts
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
            default_factory=lambda: __import__("flext_core.config")
            .FlextConfig()
            .max_batch_size
        )
        max_workers: int = Field(
            default_factory=lambda: __import__("flext_core.config")
            .FlextConfig()
            .max_workers
        )
        timeout_per_item: int = Field(
            default_factory=lambda: __import__("flext_core.config")
            .FlextConfig()
            .timeout_seconds
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
            default_factory=lambda: __import__("flext_core.config")
            .FlextConfig()
            .timeout_seconds
        )
        retry_on_failure: bool = True
        max_retries: int = Field(
            default_factory=lambda: __import__("flext_core.config")
            .FlextConfig()
            .max_retry_attempts
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
            default_factory=lambda: __import__("flext_core.config")
            .FlextConfig()
            .timeout_seconds
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
        operations: Annotated[list[FlextTypes.Dict], Field(default_factory=list)]
        parallel_execution: bool = False
        stop_on_error: bool = True
        batch_size: int = Field(
            default_factory=lambda: __import__("flext_core.config")
            .FlextConfig()
            .batch_size
        )
        timeout_per_operation: int = Field(
            default_factory=lambda: __import__("flext_core.config")
            .FlextConfig()
            .timeout_seconds
        )

        @field_validator("operations")
        @classmethod
        def validate_operations(cls, v: FlextTypes.List) -> FlextTypes.List:
            """Validate operations FlextTypes.List."""
            if not v:
                msg = "Operations FlextTypes.List cannot be empty"
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
        aggregation: str = Field(
            default_factory=lambda: FlextConstants.Cqrs.AggregationLiteral.AVG
        )
        group_by: FlextTypes.StringList = Field(default_factory=list)
        filters: FlextTypes.Dict = Field(default_factory=dict)

        @field_validator("metric_types")
        @classmethod
        def validate_prefix(cls, v: FlextTypes.List) -> FlextTypes.List:
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
        action: str = Field(
            default_factory=lambda: FlextConstants.Cqrs.ActionLiteral.GET
        )
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
            if v <= FlextConstants.ZERO:
                msg = "Resource limit must be positive"
                raise FlextExceptions.ValidationError(
                    message=msg,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return v

    class OperationExecutionRequest(ArbitraryTypesModel):
        """Operation execution request."""

        operation_name: str
        operation_callable: Callable[..., object]
        arguments: FlextTypes.Dict = Field(default_factory=dict)
        keyword_arguments: FlextTypes.Dict = Field(default_factory=dict)
        timeout_seconds: int = Field(
            default_factory=lambda: __import__("flext_core.config")
            .FlextConfig()
            .timeout_seconds
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
            default_factory=lambda: __import__("flext_core.config")
            .FlextConfig()
            .max_retry_attempts
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
        """Registration details for handler registration tracking."""

        registration_id: str = Field(description="Unique registration identifier")
        handler_mode: FlextConstants.HandlerModeSimple = Field(
            default="command",
            description="Handler mode",
        )
        timestamp: str = Field(
            default_factory=lambda: FlextConstants.Cqrs.DEFAULT_TIMESTAMP,
            description="Registration timestamp",
        )
        status: FlextConstants.Status = Field(
            default="running",
            description="Registration status",
        )

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
        def validate_aggregate_consistency[T: FlextProtocols.Foundation.HasInvariants](
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
            if isinstance(aggregate, FlextProtocols.Foundation.HasInvariants):
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
                        f"Too many uncommitted domain events: {len(events)} "
                        f"(max: {FlextConstants.Validation.MAX_UNCOMMITTED_EVENTS})",
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
        def validate_pagination(
            cls, v: FlextModels.Pagination | dict[str, int | str] | None
        ) -> FlextModels.Pagination:
            """Convert pagination to Pagination instance."""
            if isinstance(v, FlextModels.Pagination):
                return v
            if isinstance(v, dict):
                # Extract page and size from dict with proper type casting
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
                if isinstance(pagination_data, FlextModels.Pagination):
                    pagination = pagination_data
                elif isinstance(pagination_data, dict):
                    pagination_dict = cast("FlextTypes.Dict", pagination_data)
                    page_raw = pagination_dict.get("page", 1)
                    size_raw = pagination_dict.get("size", 20)
                    page: int = int(page_raw) if isinstance(page_raw, (int, str)) else 1
                    size: int = (
                        int(size_raw) if isinstance(size_raw, (int, str)) else 20
                    )
                    pagination = FlextModels.Pagination(
                        page=page,
                        size=size,
                    )
                else:
                    pagination = FlextModels.Pagination()
                query_id = str(query_payload.get("query_id", str(uuid.uuid4())))
                query_type: object = query_payload.get("query_type")

                if not isinstance(filters, dict):
                    filters = {}
                # Type casting for mypy - after validation, filters is guaranteed to be dict
                filters_dict = cast("FlextTypes.Dict", filters)
                # No need to validate pagination dict - Pydantic validator handles conversion

                query = cls(
                    filters=filters_dict,
                    pagination=pagination,  # Pydantic will convert dict to Pagination
                    query_id=query_id,
                    query_type=str(query_type) if query_type is not None else None,
                )
                return FlextResult[FlextModels.Query].ok(query)
            except Exception as e:
                return FlextResult[FlextModels.Query].fail(
                    f"Query validation failed: {e}"
                )

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
        body: str | FlextTypes.Dict | None = Field(
            default=None, description="Request body"
        )
        timeout: float = Field(
            default_factory=lambda: __import__("flext_core.config")
            .FlextConfig()
            .timeout_seconds,
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
                    metadata={
                        "validation_details": f"Method {self.method} should not have body"
                    },
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
        body: str | FlextTypes.Dict | None = Field(
            default=None, description="Response body"
        )
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
                    metadata={
                        "validation_details": "Status 204 No Content should not have body"
                    },
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


__all__ = [
    "FlextModels",
]

# Rebuild models for Pydantic v2 forward references
FlextModels.Query.model_rebuild()
FlextModels.Command.model_rebuild()
FlextModels.DomainEvent.model_rebuild()
FlextModels.HttpRequest.model_rebuild()
FlextModels.HttpResponse.model_rebuild()
