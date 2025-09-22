"""Domain models aligned with the FLEXT 1.0.0 modernization charter.

Entities, value objects, and aggregates mirror the design captured in
``README.md`` and ``docs/architecture.md`` so downstream packages share a
consistent DDD foundation during the rollout.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import json
import re
import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import (
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    Self,
    TypedDict,
    cast,
)
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, T

# Type alias for Pydantic's IncEx (include/exclude) parameter
IncEx = Any  # Simplified to avoid complex recursive type issues

# Using pure Pydantic 2.11 - no custom protocols needed


class ModelDumpKwargs(TypedDict, total=False):
    """Type definition for model_dump kwargs."""

    exclude_unset: bool
    exclude_defaults: bool
    exclude_none: bool
    exclude: IncEx | None
    include: IncEx | None
    by_alias: bool
    round_trip: bool
    warnings: bool | Literal["none", "warn", "error"]
    serialize_as_any: bool
    indent: int | None
    mode: Literal["json", "python"]
    context: dict[str, object] | None


class FlextModels:
    """Enhanced unified model container with advanced Pydantic 2 optimizations.

    Provides comprehensive domain-driven models for FLEXT infrastructure leveraging:
    - Direct FlextConfig integration for dynamic defaults
    - Advanced Field constraints with Pydantic validation
    - Computed fields for derived properties
    - Consolidated model configuration hierarchy
    - FlextUtilities integration for enhanced validation

    This class serves as the central export point for all FLEXT models,
    implementing Railway-oriented error handling through FlextResult[T].
    """

    # Enhanced validation using FlextUtilities and FlextConstants
    # Base model classes for configuration consolidation
    class ArbitraryTypesModel(BaseModel):
        """Most common pattern: validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True.

        Used by 17+ models in the codebase.
        """

        model_config = ConfigDict(
            validate_assignment=True, use_enum_values=True, arbitrary_types_allowed=True
        )

    class StrictArbitraryTypesModel(BaseModel):
        """Strict pattern: forbid extra fields, arbitrary types allowed.

        Used by domain service models requiring strict validation.
        """

        model_config = ConfigDict(
            validate_assignment=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            extra="forbid",
        )

    class FrozenStrictModel(BaseModel):
        """Immutable pattern: frozen with extra fields forbidden.

        Used by value objects and configuration models.
        """

        model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)

    # Base model classes from DDD patterns
    class TimestampedModel(ArbitraryTypesModel):
        """Base class for models with timestamp fields."""

        created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        updated_at: datetime | None = None

        def update_timestamp(self) -> None:
            """Update the updated_at timestamp."""
            self.updated_at = datetime.now(UTC)

    class Entity(TimestampedModel):
        """Base class for domain entities with identity."""

        id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        version: int = Field(
            default=FlextConstants.Performance.DEFAULT_VERSION,
            ge=FlextConstants.Performance.MIN_VERSION,
        )
        domain_events: FlextTypes.Core.List = Field(default_factory=list)

        def model_post_init(self, __context: object, /) -> None:
            """Post-initialization hook to set updated_at timestamp."""
            if self.updated_at is None:
                self.updated_at = datetime.now(UTC)

        def __eq__(self, other: object) -> bool:
            """Identity-based equality for entities."""
            if not isinstance(other, FlextModels.Entity):
                return False
            return self.id == other.id

        def __hash__(self) -> int:
            """Identity-based hash for entities."""
            return hash(self.id)

        def add_domain_event(self, event_name: str, data: FlextTypes.Core.Dict) -> None:
            """Add a domain event to be dispatched."""
            domain_event = FlextModels.DomainEvent(
                event_type=event_name,
                aggregate_id=self.id,
                data=data,
                occurred_at=datetime.now(UTC),
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
                    # Exception handling as expected by test line 357-358
                    # This is intentional - some domain events may not have corresponding handlers
                    # Log the exception for debugging but continue execution as per test requirements
                    msg = f"Error applying domain event: {e}"
                    raise FlextExceptions.ProcessingError(msg) from e

            # Increment version after adding domain event
            self.increment_version()

        def clear_domain_events(self) -> FlextTypes.Core.List:
            """Clear and return domain events."""
            events = self.domain_events.copy()
            self.domain_events.clear()
            return events

        def increment_version(self) -> None:
            """Increment the entity version for optimistic locking."""
            self.version += 1
            self.updated_at = datetime.now(UTC)

    class Value(FrozenStrictModel):
        """Base class for value objects - immutable and compared by value."""

        def __eq__(self, other: object) -> bool:
            """Compare by value."""
            if not isinstance(other, self.__class__):
                return False
            if hasattr(self, "model_dump") and hasattr(other, "model_dump"):
                return self.model_dump() == other.model_dump()
            return False

        def __hash__(self) -> int:
            """Hash based on values for use in sets/dict[str, object]s."""
            return hash(tuple(self.model_dump().items()))

        @classmethod
        def create(cls, *args: object, **kwargs: object) -> FlextResult[Any]:
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
                return FlextResult[Any].ok(instance)
            except Exception as e:
                return FlextResult[Any].fail(str(e))

    class AggregateRoot(Entity):
        """Base class for aggregate roots - consistency boundaries."""

        _invariants: ClassVar[list[Callable[[], bool]]] = []

        def check_invariants(self) -> None:
            """Check all business invariants."""
            for invariant in self._invariants:
                if not invariant():
                    msg = f"Invariant violated: {invariant.__name__}"
                    raise ValueError(msg)

        def model_post_init(self, __context: object, /) -> None:
            """Run after model initialization."""
            super().model_post_init(__context)
            self.check_invariants()

    class Command(StrictArbitraryTypesModel):
        """Base class for CQRS commands with validation."""

        command_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        command_type: str = Field(default="")
        issued_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        issuer_id: str | None = None

        @field_validator("command_type")
        @classmethod
        def validate_command(cls, v: str) -> str:
            """Auto-set command type from class name if empty."""
            if not v:
                return cls.__name__
            return v

    class Query(ArbitraryTypesModel):
        """Base class for CQRS queries with pagination."""

        query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        pagination: dict[str, int] = Field(
            default_factory=lambda: {
                "page": FlextConstants.Performance.DEFAULT_PAGE_NUMBER,
                "size": FlextConstants.Performance.DEFAULT_PAGE_SIZE,
            }
        )
        filters: FlextTypes.Core.Dict = Field(default_factory=dict)
        sort_by: str | None = None
        sort_order: Literal["asc", "desc"] = "asc"

        @field_validator("pagination")
        @classmethod
        def validate_pagination(cls, v: dict[str, object]) -> dict[str, object]:
            """Validate pagination parameters using FlextConstants."""
            page = v.get("page", FlextConstants.Performance.DEFAULT_PAGE_NUMBER)
            size = v.get("size", FlextConstants.Performance.DEFAULT_PAGE_SIZE)

            # Ensure types are integers
            if not isinstance(page, int):
                page = (
                    int(page)
                    if isinstance(page, (str, float))
                    else FlextConstants.Performance.DEFAULT_PAGE_NUMBER
                )
            if not isinstance(size, int):
                size = (
                    int(size)
                    if isinstance(size, (str, float))
                    else FlextConstants.Performance.DEFAULT_PAGE_SIZE
                )

            if page < 1:
                msg = "pagination.page must be a positive integer"
                raise ValueError(msg)

            if size < 1 or size > FlextConstants.Cqrs.MAX_PAGE_SIZE:
                msg = f"pagination.size must be between 1 and {FlextConstants.Cqrs.MAX_PAGE_SIZE}"
                raise ValueError(msg)

            return {"page": page, "size": size}

        @model_validator(mode="after")
        def validate_query_consistency(self) -> Self:
            """Validate overall query consistency."""
            # Additional validation logic can be added here
            return self

        @classmethod
        def validate_query(
            cls, data: FlextTypes.Core.Dict
        ) -> FlextResult[FlextModels.Query]:
            """Validate and create a query instance."""
            try:
                query = cls.model_validate(data)
                return FlextResult[FlextModels.Query].ok(query)
            except ValidationError as e:
                error_details: list[str] = []
                for error in e.errors():
                    field = ".".join(str(x) for x in error["loc"])
                    message = error["msg"]
                    error_details.append(f"{field}: {message}")
                return FlextResult[FlextModels.Query].fail(
                    f"FlextModels.Query validation failed: {', '.join(error_details)}"
                )

    class DomainEvent(ArbitraryTypesModel):
        """Base class for domain events."""

        event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        event_type: str
        aggregate_id: str
        occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        data: FlextTypes.Core.Dict = Field(default_factory=dict)
        metadata: FlextTypes.Core.Dict = Field(default_factory=dict)

    class Repository(ArbitraryTypesModel):
        """Base repository model for data access patterns."""

        entity_type: str
        connection_string: str | None = None
        timeout_seconds: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().timeout_seconds
        )
        retry_policy: FlextTypes.Core.Dict = Field(default_factory=dict)

    class Specification(ArbitraryTypesModel):
        """Specification pattern for complex queries."""

        criteria: FlextTypes.Core.Dict = Field(default_factory=dict)
        includes: list[str] = Field(default_factory=list)
        order_by: list[tuple[str, str]] = Field(default_factory=list)
        skip: int = Field(
            default=FlextConstants.Performance.DEFAULT_SKIP,
            ge=FlextConstants.Performance.MIN_SKIP,
        )
        take: int = Field(
            default=FlextConstants.Performance.DEFAULT_TAKE,
            ge=FlextConstants.Performance.MIN_TAKE,
            le=FlextConstants.Performance.MAX_TAKE,
        )

    class Saga(ArbitraryTypesModel):
        """Saga pattern for distributed transactions."""

        saga_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        steps: list[dict[str, object]] = Field(default_factory=list)
        current_step: int = Field(
            default=FlextConstants.Performance.DEFAULT_CURRENT_STEP,
            ge=FlextConstants.Performance.MIN_CURRENT_STEP,
        )
        status: Literal["pending", "running", "completed", "failed", "compensating"] = (
            "pending"
        )
        compensation_data: FlextTypes.Core.Dict = Field(default_factory=dict)

    class Metadata(FrozenStrictModel):
        """Immutable metadata model."""

        created_by: str | None = None
        created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        modified_by: str | None = None
        modified_at: datetime | None = None
        tags: list[str] = Field(default_factory=list)
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
        errors: list[FlextModels.ErrorDetail] = Field(default_factory=list)
        warnings: list[str] = Field(default_factory=list)

    class Configuration(FrozenStrictModel):
        """Base configuration model - immutable."""

        version: str = FlextConstants.Core.DEFAULT_VERSION
        enabled: bool = True
        settings: FlextTypes.Core.Dict = Field(default_factory=dict)

    class HealthCheck(ArbitraryTypesModel):
        """Health check model for service monitoring."""

        service_name: str
        status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
        checks: dict[str, bool] = Field(default_factory=dict)
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
        rules: list[dict[str, object]] = Field(default_factory=list)
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

    class Bus(BaseModel):
        """Enhanced message bus model with config-driven defaults."""

        bus_id: str = Field(default_factory=lambda: f"bus_{uuid.uuid4().hex[:8]}")
        handlers: dict[str, list[str]] = Field(default_factory=dict)
        middlewares: list[str] = Field(default_factory=list)
        timeout_seconds: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().timeout_seconds
        )
        retry_policy: FlextTypes.Core.Dict = Field(default_factory=dict)

    class Payload(ArbitraryTypesModel, Generic[T]):
        """Enhanced payload model with computed field."""

        data: T = Field(...)  # Required field, no default
        metadata: FlextTypes.Core.Dict = Field(default_factory=dict)
        created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        expires_at: datetime | None = None
        correlation_id: str | None = None
        source_service: str | None = None
        message_type: str | None = None
        message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

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
        scopes: list[str] = Field(default_factory=list)

    class Permission(FrozenStrictModel):
        """Immutable permission model."""

        resource: str
        action: str
        conditions: FlextTypes.Core.Dict = Field(default_factory=dict)

    class Role(ArbitraryTypesModel):
        """Role model for authorization."""

        name: str
        description: str | None = None
        permissions: list[FlextModels.Permission] = Field(default_factory=list)

    class User(Entity):
        """User entity model."""

        username: str
        email: str
        roles: list[str] = Field(default_factory=list)
        is_active: bool = True
        last_login: datetime | None = None

    class Session(ArbitraryTypesModel):
        """Session model."""

        session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        user_id: str
        started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        expires_at: datetime
        data: FlextTypes.Core.Dict = Field(default_factory=dict)

    class Task(ArbitraryTypesModel):
        """Task model for background processing."""

        task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        name: str
        status: Literal["pending", "running", "completed", "failed"] = "pending"
        payload: FlextTypes.Core.Dict = Field(default_factory=dict)
        result: object = None
        error: str | None = None
        created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        started_at: datetime | None = None
        completed_at: datetime | None = None

    class Queue(ArbitraryTypesModel):
        """Queue model for message processing."""

        name: str
        messages: list[dict[str, object]] = Field(default_factory=list)
        max_size: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().cache_max_size
        )
        processing_timeout: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().timeout_seconds
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

    class Batch(ArbitraryTypesModel):
        """Batch processing model."""

        batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        items: list[object] = Field(default_factory=list)
        size: int = Field(
            default=FlextConstants.Performance.DEFAULT_BATCH_SIZE_SMALL, ge=1
        )
        processed_count: int = 0

    class Stream(ArbitraryTypesModel):
        """Stream processing model."""

        stream_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        position: int = 0
        batch_size: int = FlextConstants.Performance.DEFAULT_STREAM_BATCH_SIZE
        buffer: list[object] = Field(default_factory=list)

    class Pipeline(ArbitraryTypesModel):
        """Pipeline model."""

        name: str
        stages: list[dict[str, object]] = Field(default_factory=list)
        current_stage: int = 0
        status: Literal["idle", "running", "completed", "failed"] = "idle"

    class Workflow(ArbitraryTypesModel):
        """Workflow model."""

        workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        name: str
        steps: list[dict[str, object]] = Field(default_factory=list)
        current_step: int = 0
        context: FlextTypes.Core.Dict = Field(default_factory=dict)

    class Archive(ArbitraryTypesModel):
        """Archive model."""

        archive_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        entity_type: str
        entity_id: str
        archived_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
        archived_by: str
        data: FlextTypes.Core.Dict = Field(default_factory=dict)

    class Import(ArbitraryTypesModel):
        """Import model."""

        import_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        source: str
        format: str
        status: Literal["pending", "processing", "completed", "failed"] = "pending"
        records_total: int = 0
        records_processed: int = 0
        errors: list[str] = Field(default_factory=list)

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
            """Validate email format using simple regex validation."""
            if "@" not in v or "." not in v.rsplit("@", maxsplit=1)[-1]:
                msg = f"Invalid email format: {v}"
                raise ValueError(msg)
            return v.lower()

    class Host(Value):
        """Host/hostname value object."""

        hostname: str

        @field_validator("hostname")
        @classmethod
        def validate_host_format(cls, v: str) -> str:
            """Validate hostname format."""
            # Trim whitespace first
            v = v.strip()

            # Check if empty after trimming
            if not v:
                msg = "Hostname cannot be empty"
                raise ValueError(msg)

            # Basic hostname validation
            if len(v) > FlextConstants.Validation.MAX_EMAIL_LENGTH:
                msg = "Hostname too long"
                raise ValueError(msg)
            if not all(c.isalnum() or c in ".-" for c in v):
                msg = "Invalid hostname characters"
                raise ValueError(msg)
            return v.lower()

    class EntityId(Value):
        """Entity identifier value object with validation."""

        value: str

        @field_validator("value")
        @classmethod
        def validate_entity_id_format(cls, v: str) -> str:
            """Validate entity ID format."""
            # Trim whitespace first
            v = v.strip()

            # Check if empty after trimming
            if not v:
                msg = "Entity ID cannot be empty"
                raise ValueError(msg)

            # Allow UUIDs, alphanumeric with dashes/underscores
            if not re.match(r"^[a-zA-Z0-9_-]+$", v):
                msg = "Invalid entity ID format"
                raise ValueError(msg)
            return v

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
            """Validate URL format."""
            try:
                result = urlparse(v)
                if not all([result.scheme, result.netloc]):
                    msg = f"Invalid URL format: {v}"
                    raise ValueError(msg)

                # Validate scheme
                if result.scheme not in {"http", "https", "ftp", "ftps", "file"}:
                    msg = f"Unsupported URL scheme: {result.scheme}"
                    raise ValueError(msg)

                # Validate domain
                if result.netloc:
                    # Basic domain validation
                    domain = result.netloc.split(":")[0]  # Remove port
                    if (
                        not domain
                        or len(domain) > FlextConstants.Validation.MAX_EMAIL_LENGTH
                    ):
                        msg = "Invalid domain in URL"
                        raise ValueError(msg)

                    # Check for valid characters
                    if not all(c.isalnum() or c in ".-" for c in domain):
                        msg = "Invalid characters in domain"
                        raise ValueError(msg)

                return v
            except Exception as e:
                msg = f"URL validation failed: {e}"
                raise ValueError(msg) from e

    class DateRange(Value):
        """Date range value object."""

        start_date: date
        end_date: date

        @model_validator(mode="after")
        def validate_date_order(self) -> Self:
            """Ensure start_date <= end_date."""
            if self.start_date > self.end_date:
                msg = "start_date must be before or equal to end_date"
                raise ValueError(msg)
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
            bytes_per_mb = FlextConstants.Utilities.BYTES_PER_KB * FlextConstants.Utilities.BYTES_PER_KB
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
                raise ValueError(msg)

            return self

    class WorkspaceInfo(AggregateRoot):
        """Enhanced workspace aggregate with advanced validation."""

        workspace_id: str
        name: str
        root_path: str
        projects: list[FlextModels.Project] = Field(default_factory=list)
        total_files: int = 0
        total_size_bytes: int = 0

        @model_validator(mode="after")
        def validate_business_rules(self) -> Self:
            """Complex workspace validation."""
            # Workspace consistency
            if self.projects and self.total_files == 0:
                msg = "Workspace with projects must have files"
                raise ValueError(msg)

            # Size validation
            max_workspace_size = 10 * FlextConstants.Utilities.BYTES_PER_KB ** 3  # 10GB
            if self.total_size_bytes > max_workspace_size:
                msg = "Workspace too large"
                raise ValueError(msg)

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
            from urllib.parse import urlparse

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
        url_result = FlextModels.create_validated_url(url)
        return url_result >> (
            lambda validated_url: FlextResult[FlextModels.Url].ok(
                FlextModels.Url(url=validated_url)
            )
        )

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
            import re

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

        file_path = Path(path_result.unwrap())
        if not file_path.exists():
            return FlextResult[str].fail(f"Path does not exist: {path}")
        return FlextResult[str].ok(path_result.unwrap())

    @staticmethod
    def create_validated_directory_path(path: str) -> FlextResult[str]:
        """Create a validated directory path."""
        path_result = FlextModels.create_validated_existing_file_path(path)
        if path_result.is_failure:
            return path_result

        dir_path = Path(path_result.unwrap())
        if not dir_path.is_dir():
            return FlextResult[str].fail(f"Path is not a directory: {path}")
        return FlextResult[str].ok(path_result.unwrap())

    # Additional domain models continue with advanced patterns...
    class FactoryRegistrationModel(StrictArbitraryTypesModel):
        """Enhanced factory registration with advanced validation."""

        name: str
        factory: Callable[..., object]
        singleton: bool = False
        dependencies: list[str] = Field(default_factory=list)

        @field_validator("factory")
        @classmethod
        def validate_factory_signature(
            cls, v: Callable[..., object]
        ) -> Callable[..., object]:
            """Validate factory is callable with proper signature."""
            if not callable(v):
                msg = "Factory must be callable"
                raise TypeError(msg)

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
                raise ValueError(msg)

            return v

    class BatchRegistrationModel(StrictArbitraryTypesModel):
        """Batch registration model with advanced validation."""

        services: FlextTypes.Core.Dict = Field(default_factory=dict)
        factories: dict[str, Callable[..., object]] = Field(default_factory=dict)
        singletons: FlextTypes.Core.Dict = Field(default_factory=dict)

        @model_validator(mode="after")
        def validate_non_empty(self) -> Self:
            """Ensure at least one registration exists."""
            if not any([self.services, self.factories, self.singletons]):
                msg = "At least one registration required"
                raise ValueError(msg)
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

        @field_validator("context")
        @classmethod
        def validate_context(cls, v: dict[str, object]) -> dict[str, object]:
            """Validate context has required fields."""
            if "correlation_id" not in v:
                v["correlation_id"] = str(uuid.uuid4())
            if "timestamp" not in v:
                v["timestamp"] = datetime.now(UTC).isoformat()
            return v

        @field_validator("timeout_seconds")
        @classmethod
        def validate_timeout(cls, v: int) -> int:
            """Validate timeout is within acceptable range."""
            max_timeout_seconds = 3600  # 1 hour max
            if v > max_timeout_seconds:
                msg = f"Timeout cannot exceed {max_timeout_seconds} seconds"
                raise ValueError(msg)
            return v

    class HandlerRegistration(StrictArbitraryTypesModel):
        """Handler registration with advanced validation."""

        name: str
        handler: Callable[..., object]
        event_types: list[str] = Field(default_factory=list)
        priority: int = Field(default=0, ge=0, le=100)

        @field_validator("handler")
        @classmethod
        def validate_handler(cls, v: Callable[..., object]) -> Callable[..., object]:
            """Validate handler is properly callable."""
            if not callable(v):
                msg = "Handler must be callable"
                raise TypeError(msg)
            return v

    class BatchProcessingConfig(StrictArbitraryTypesModel):
        """Enhanced batch processing configuration."""

        batch_size: int = Field(default=100)
        max_workers: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().max_workers
        )
        timeout_per_item: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().timeout_seconds
        )
        continue_on_error: bool = True
        data_items: list[object] = Field(default_factory=list)

        @field_validator("data_items")
        @classmethod
        def validate_data_items(cls, v: list[object]) -> list[object]:
            """Validate data items are not empty when provided."""
            if len(v) > FlextConstants.Performance.MAX_BATCH_ITEMS:
                msg = f"Batch cannot exceed {FlextConstants.Performance.MAX_BATCH_ITEMS} items"
                raise ValueError(msg)
            return v

        @field_validator("max_workers")
        @classmethod
        def validate_max_workers(cls, v: int) -> int:
            """Validate max workers is reasonable."""
            max_workers_limit = 50
            if v > max_workers_limit:
                msg = f"Max workers cannot exceed {max_workers_limit}"
                raise ValueError(msg)
            return v

        @model_validator(mode="after")
        def validate_batch(self) -> Self:
            """Validate batch configuration consistency."""
            max_batch_size = FlextConstants.Performance.MAX_BATCH_SIZE_VALIDATION
            if self.batch_size > max_batch_size:
                msg = f"Batch size cannot exceed {max_batch_size}"
                raise ValueError(msg)

            self.max_workers = min(self.max_workers, self.batch_size)  # Auto-adjust

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
        fallback_handlers: list[str] = Field(default_factory=list)

        @field_validator("handler_name")
        @classmethod
        def validate_handler_name(cls, v: str) -> str:
            """Validate handler name format."""
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", v):
                msg = "Handler name must be valid identifier"
                raise ValueError(msg)
            return v

    class PipelineConfiguration(ArbitraryTypesModel):
        """Pipeline configuration with advanced validation."""

        name: str = Field(min_length=FlextConstants.Performance.MIN_NAME_LENGTH)
        steps: list[dict[str, object]] = Field(default_factory=list)
        parallel_execution: bool = False
        stop_on_error: bool = True
        max_parallel: int = Field(gt=0, default=4)

        @field_validator("name")
        @classmethod
        def validate_name(cls, v: str) -> str:
            """Validate pipeline name."""
            max_name_length = 100
            if len(v) > max_name_length:
                msg = f"Pipeline name too long (max {max_name_length} characters)"
                raise ValueError(msg)
            return v

        @field_validator("steps")
        @classmethod
        def validate_steps(cls, v: list[object]) -> list[object]:
            """Validate pipeline steps."""
            if not v:
                msg = "Pipeline must have at least one step"
                raise ValueError(msg)
            return v

    class ProcessingResult(ArbitraryTypesModel):
        """Processing result with computed fields."""

        operation_id: str
        status: Literal["success", "failure", "partial"]
        data: object = None
        errors: list[str] = Field(default_factory=list)
        execution_time_ms: int = 0

        @field_validator("execution_time_ms")
        @classmethod
        def validate_execution_time(cls, v: int) -> int:
            """Validate execution time is reasonable."""
            max_execution_time_ms = 300000  # 5 minutes
            if v > max_execution_time_ms:
                msg = f"Execution time exceeds maximum ({max_execution_time_ms // 60000} minutes)"
                raise ValueError(msg)
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
                raise ValueError(msg)
            return v

    class LoggerInitializationModel(ArbitraryTypesModel):
        """Logger initialization with advanced validation."""

        name: str
        log_level: str = Field(
            default_factory=lambda: FlextConfig.get_global_instance().log_level
        )
        structured_output: bool = True
        include_source: bool = True
        json_output: bool | None = None

        @field_validator("log_level")
        @classmethod
        def validate_log_level(cls, v: str) -> str:
            """Validate log level is valid."""
            valid_levels = FlextConstants.Logging.VALID_LEVELS
            v_upper = v.upper()
            if v_upper not in valid_levels:
                msg = f"Invalid log level: {v}. Must be one of {valid_levels}"
                raise ValueError(msg)
            return v_upper

    class LoggerConfigurationModel(ArbitraryTypesModel):
        """Logger configuration model for global configuration."""

        log_level: str = Field(
            default_factory=lambda: FlextConfig.get_global_instance().log_level
        )
        json_output: bool | None = None
        include_source: bool = Field(
            default_factory=lambda: FlextConstants.Logging.INCLUDE_SOURCE
        )
        structured_output: bool = Field(
            default_factory=lambda: FlextConstants.Logging.STRUCTURED_OUTPUT
        )
        log_verbosity: str = Field(
            default_factory=lambda: FlextConstants.Logging.VERBOSITY
        )

        @field_validator("log_level")
        @classmethod
        def validate_log_level(cls, v: str) -> str:
            """Validate log level is valid."""
            valid_levels = FlextConstants.Logging.VALID_LEVELS
            v_upper = v.upper()
            if v_upper not in valid_levels:
                msg = f"Invalid log level: {v}. Must be one of {valid_levels}"
                raise ValueError(msg)
            return v_upper

    class LogContextModel(ArbitraryTypesModel):
        """Log context model with validation."""

        correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        request_id: str | None = None
        user_id: str | None = None
        session_id: str | None = None
        extra: FlextTypes.Core.Dict = Field(default_factory=dict)

        @field_validator("extra")
        @classmethod
        def validate_context_data(cls, v: dict[str, object]) -> dict[str, object]:
            """Validate context data is serializable."""
            # Ensure all values are JSON serializable
            try:
                json.dumps(v)
            except (TypeError, ValueError) as e:
                msg = f"Context data must be JSON serializable: {e}"
                raise ValueError(msg) from e
            return v

    class LoggerContextBindingModel(ArbitraryTypesModel):
        """Logger context binding model."""

        logger_name: str
        context_data: FlextTypes.Core.Dict = Field(default_factory=dict)
        bind_type: Literal["temporary", "permanent"] = "temporary"
        clear_existing: bool = False
        force_new_instance: bool = False
        copy_request_context: bool = False
        copy_permanent_context: bool = False

        @field_validator("context_data")
        @classmethod
        def validate_context_data(cls, v: dict[str, object]) -> dict[str, object]:
            """Validate context data."""
            max_context_keys = 100
            if len(v) > max_context_keys:
                msg = f"Context data too large (max {max_context_keys} keys)"
                raise ValueError(msg)
            return v

    class LoggerRequestContextModel(ArbitraryTypesModel):
        """Logger request context model."""

        request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        method: str | None = None
        path: str | None = None
        headers: dict[str, str] = Field(default_factory=dict)
        query_params: dict[str, str] = Field(default_factory=dict)
        correlation_id: str | None = None
        user_id: str | None = None
        endpoint: str | None = None
        custom_data: dict[str, str] = Field(default_factory=dict)

        @model_validator(mode="after")
        def validate_request_context(self) -> Self:
            """Validate request context consistency."""
            if (
                self.method
                and self.method not in FlextConstants.Platform.VALID_HTTP_METHODS
            ):
                msg = f"Invalid HTTP method: {self.method}"
                raise ValueError(msg)
            return self

    class LoggerPermanentContextModel(ArbitraryTypesModel):
        """Logger permanent context model."""

        app_name: str
        app_version: str
        environment: str
        host: str | None = None
        metadata: FlextTypes.Core.Dict = Field(default_factory=dict)
        permanent_context: FlextTypes.Core.Dict = Field(default_factory=dict)
        replace_existing: bool = False
        merge_strategy: Literal["replace", "update", "merge_deep"] = "update"

        @model_validator(mode="after")
        def validate_permanent_context(self) -> Self:
            """Validate permanent context."""
            valid_envs = set(FlextConstants.Config.ENVIRONMENTS)
            if self.environment.lower() not in valid_envs:
                msg = f"Invalid environment: {self.environment}"
                raise ValueError(msg)
            return self

    class OperationTrackingModel(ArbitraryTypesModel):
        """Operation tracking model."""

        operation_name: str
        start_time: datetime = Field(default_factory=lambda: datetime.now(UTC))
        end_time: datetime | None = None
        success: bool | None = None
        error_message: str | None = None
        metadata: FlextTypes.Core.Dict = Field(default_factory=dict)

        @field_validator("metadata")
        @classmethod
        def validate_metadata(cls, v: dict[str, object]) -> dict[str, object]:
            """Validate metadata is not too large."""
            max_metadata_size = FlextConstants.Performance.MAX_METADATA_SIZE
            if len(str(v)) > max_metadata_size:
                msg = "Metadata too large"
                raise ValueError(msg)
            return v

    class PerformanceTrackingModel(ArbitraryTypesModel):
        """Performance tracking model."""

        operation: str
        duration_ms: int
        cpu_usage: float | None = None
        memory_usage: float | None = None
        io_operations: int = 0
        cache_hits: int = 0
        cache_misses: int = 0
        is_critical: bool = False

        @model_validator(mode="after")
        def validate_critical_threshold(self) -> Self:
            """Mark as critical if thresholds exceeded."""
            critical_duration_ms = FlextConstants.Performance.CRITICAL_DURATION_MS
            critical_usage_percent = FlextConstants.Performance.CRITICAL_USAGE_PERCENT
            if self.duration_ms > critical_duration_ms:
                self.is_critical = True
            if self.cpu_usage and self.cpu_usage > critical_usage_percent:
                self.is_critical = True
            if self.memory_usage and self.memory_usage > critical_usage_percent:
                self.is_critical = True
            return self

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

        @field_validator("context")
        @classmethod
        def validate_context(cls, v: dict[str, object]) -> dict[str, object]:
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
                raise ValueError(msg)
            return v

    class DomainServiceValidationRequest(ArbitraryTypesModel):
        """Domain service validation request."""

        entity: FlextTypes.Core.Dict
        rules: list[str] = Field(default_factory=list)
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
                raise ValueError(msg)
            return self

    class DomainServiceBatchRequest(ArbitraryTypesModel):
        """Domain service batch request."""

        service_name: str
        operations: list[dict[str, object]] = Field(default_factory=list)
        parallel_execution: bool = False
        stop_on_error: bool = True
        batch_size: int = Field(default=100)
        timeout_per_operation: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().timeout_seconds
        )

        @field_validator("operations")
        @classmethod
        def validate_operations(cls, v: list[object]) -> list[object]:
            """Validate operations list[object]."""
            if not v:
                msg = "Operations list[object] cannot be empty"
                raise ValueError(msg)
            max_batch_operations = FlextConstants.Performance.MAX_BATCH_OPERATIONS
            if len(v) > max_batch_operations:
                msg = f"Batch cannot exceed {max_batch_operations} operations"
                raise ValueError(msg)
            return v

    class DomainServiceMetricsRequest(ArbitraryTypesModel):
        """Domain service metrics request."""

        service_name: str
        metric_types: list[str] = Field(
            default_factory=lambda: ["performance", "errors", "throughput"]
        )
        time_range_seconds: int = FlextConstants.Performance.DEFAULT_TIME_RANGE_SECONDS
        aggregation: Literal["sum", "avg", "min", "max", "count"] = "avg"
        group_by: list[str] = Field(default_factory=list)
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
                    raise ValueError(msg)
            return v

    class DomainServiceResourceRequest(ArbitraryTypesModel):
        """Domain service resource request."""

        service_name: str
        resource_type: str
        resource_id: str | None = None
        action: Literal["get", "create", "update", "delete", "list[object]"] = "get"
        data: FlextTypes.Core.Dict = Field(default_factory=dict)
        filters: FlextTypes.Core.Dict = Field(default_factory=dict)

        @field_validator("resource_type")
        @classmethod
        def validate_resource_type(cls, v: str) -> str:
            """Validate resource type format."""
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", v):
                msg = "Resource type must be valid identifier"
                raise ValueError(msg)
            return v

    class OperationExecutionRequest(ArbitraryTypesModel):
        """Operation execution request."""

        operation_name: str
        operation_callable: Callable[..., object]
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
                raise ValueError(msg)
            return v

        @field_validator("operation_callable")
        @classmethod
        def validate_operation_callable(
            cls, v: Callable[..., object]
        ) -> Callable[..., object]:
            """Validate operation is callable."""
            if not callable(v):
                msg = "Operation must be callable"
                raise TypeError(msg)
            return v

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
        retry_on_exceptions: list[type[Exception]] = Field(default_factory=list)
        retry_on_status_codes: list[int] = Field(default_factory=list)

        @field_validator("retry_on_status_codes")
        @classmethod
        def validate_backoff_strategy(cls, v: list[object]) -> list[int]:
            """Validate status codes are valid HTTP codes."""
            validated_codes: list[int] = []
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
                            raise ValueError(msg)
                        validated_codes.append(code_int)
                    else:
                        msg = f"Invalid HTTP status code type: {type(code)}"
                        raise TypeError(msg)
                except (ValueError, TypeError) as e:
                    msg = f"Invalid HTTP status code: {code}"
                    raise ValueError(msg) from e
            return validated_codes

        @model_validator(mode="after")
        def validate_delay_consistency(self) -> Self:
            """Validate delay configuration consistency."""
            if self.max_delay_seconds < self.initial_delay_seconds:
                msg = "max_delay_seconds must be >= initial_delay_seconds"
                raise ValueError(msg)
            return self

    class CircuitBreakerConfiguration(ArbitraryTypesModel):
        """Circuit breaker configuration."""

        failure_threshold: int = Field(default=5)
        recovery_timeout_seconds: int = Field(
            default=FlextConstants.Performance.DEFAULT_RECOVERY_TIMEOUT
        )
        half_open_max_calls: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().max_retry_attempts
        )
        sliding_window_size: int = Field(default=100)
        minimum_throughput: int = Field(default=10)
        slow_call_duration_seconds: float = Field(default=FlextConstants.Performance.DEFAULT_DELAY_SECONDS, gt=0)
        slow_call_rate_threshold: float = Field(default=FlextConstants.Validation.MAX_PERCENTAGE / 2.0)  # 50%

        @model_validator(mode="after")
        def validate_circuit_breaker_consistency(self) -> Self:
            """Validate circuit breaker configuration."""
            if self.half_open_max_calls > self.sliding_window_size:
                msg = "half_open_max_calls cannot exceed sliding_window_size"
                raise ValueError(msg)
            return self

    class ValidationConfiguration(ArbitraryTypesModel):
        """Validation configuration."""

        enable_strict_mode: bool = False
        max_validation_errors: int = Field(default=10)
        validate_on_assignment: bool = True
        validate_on_read: bool = False
        custom_validators: list[Callable[..., object]] = Field(default_factory=list)

        @field_validator("custom_validators")
        @classmethod
        def validate_additional_validators(cls, v: list[object]) -> list[object]:
            """Validate custom validators are callable."""
            for validator in v:
                if not callable(validator):
                    msg = "All validators must be callable"
                    raise TypeError(msg)
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
                raise ValueError(msg)
            return v

        @field_validator("deadline")
        @classmethod
        def validate_correlation_id(cls, v: datetime | None) -> datetime | None:
            """Validate deadline is in the future."""
            if v and v <= datetime.now(UTC):
                msg = "Deadline must be in the future"
                raise ValueError(msg)
            return v

    class ConditionalExecutionRequest(ArbitraryTypesModel):
        """Conditional execution request."""

        condition: Callable[[object], bool]
        true_action: Callable[..., object]
        false_action: Callable[..., object] | None = None
        context: FlextTypes.Core.Dict = Field(default_factory=dict)

        @field_validator("condition", "true_action", "false_action")
        @classmethod
        def validate_condition(
            cls, v: Callable[..., object] | None
        ) -> Callable[..., object] | None:
            """Validate callables."""
            if v is not None and not callable(v):
                msg = "Must be callable"
                raise ValueError(msg)
            return v

    class StateMachineRequest(ArbitraryTypesModel):
        """State machine request."""

        initial_state: str
        transitions: dict[str, dict[str, str]] = Field(default_factory=dict)
        current_state: str | None = None
        state_data: FlextTypes.Core.Dict = Field(default_factory=dict)

        @field_validator("transitions")
        @classmethod
        def validate_transitions(cls, v: dict[str, object]) -> dict[str, object]:
            """Validate state transitions."""
            if not v:
                msg = "Transitions cannot be empty"
                raise ValueError(msg)

            # Validate transition structure
            for state, transitions in v.items():
                if not isinstance(transitions, dict):
                    msg = f"Transitions for state {state} must be a dict[str, object]"
                    raise TypeError(msg)
                for event, next_state in cast(
                    "dict[object, object]", transitions
                ).items():
                    if not isinstance(next_state, str):
                        msg = f"Next state for {state}.{event} must be a string"
                        raise TypeError(msg)

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
                raise ValueError(msg)
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
        def validate_metrics_collector(cls, v: dict[str, object]) -> dict[str, object]:
            """Validate dimensions."""
            max_dimensions = FlextConstants.Performance.MAX_DIMENSIONS
            if len(v) > max_dimensions:
                msg = f"Too many dimensions (max {max_dimensions})"
                raise ValueError(msg)
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
            cls, v: Callable[..., object] | None
        ) -> Callable[..., object] | None:
            """Validate transformer functions."""
            if v is not None and not callable(v):
                msg = "Must be callable"
                raise ValueError(msg)
            return v

    class FallbackConfiguration(ArbitraryTypesModel):
        """Fallback configuration."""

        primary_service: Callable[..., object]
        fallback_services: list[Callable[..., object]] = Field(default_factory=list)
        max_fallback_attempts: int = Field(
            default_factory=lambda: FlextConfig.get_global_instance().max_retry_attempts
        )
        fallback_delay_seconds: float = Field(default=FlextConstants.Performance.DEFAULT_FALLBACK_DELAY, ge=0)

        @field_validator("fallback_services")
        @classmethod
        def validate_fallback_services(cls, v: list[object]) -> list[object]:
            """Validate fallback services."""
            for service in v:
                if not callable(service):
                    msg = "All fallback services must be callable"
                    raise TypeError(msg)
            return v

        @model_validator(mode="after")
        def validate_fallback_limits(self) -> Self:
            """Validate fallback configuration."""
            self.max_fallback_attempts = min(
                self.max_fallback_attempts, len(self.fallback_services)
            )
            return self

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
            import base64

            return base64.b64encode(b).decode("ascii")

    # Model serialization configuration
    class SerializableModel(BaseModel):
        """Base model with optimized serialization support.

        Combines Pydantic BaseModel with SerializationMixin for enhanced
        serialization capabilities. Used by models requiring flexible export options.
        """

        model_config = ConfigDict(
            # Core settings for serialization
            validate_assignment=True,
            use_enum_values=True,
            arbitrary_types_allowed=True,
            # JSON serialization settings
            json_encoders={
                datetime: lambda dt: dt.isoformat(),
                date: lambda d: d.isoformat(),
                time: lambda t: t.isoformat(),
                Decimal: str,
                uuid.UUID: str,
                Path: str,
                bytes: lambda b: b.decode(FlextConstants.Mixins.DEFAULT_ENCODING)
                if isinstance(b, bytes)
                else str(b),
            },
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
            cls, context: dict[str, object] | None
        ) -> dict[str, object]:
            """Extract serialization context from context dict[str, object]."""
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
            fallback: Callable[[object], object] | None = None,
            serialize_as_any: bool = False,
        ) -> FlextTypes.Core.Dict:
            """Context-aware model dump."""
            if context and isinstance(context, dict):
                ser_context = self._get_serialization_context(
                    cast("dict[str, object]", context)
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

                if exclude_set:
                    exclude = exclude_set

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
                context=cast("dict[str, object] | None", context),
                fallback=fallback,
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

    class OptimizedQuery(Query, SerializableModel):
        """FlextModels.Query with flexible serialization options."""

        def to_query_string(self) -> str:
            """Convert query to URL query string format."""
            from urllib.parse import urlencode

            params: dict[str, str | int] = {}

            # Add pagination
            if self.pagination:
                params["page"] = self.pagination.get("page", 1)
                params["size"] = self.pagination.get("size", 10)

            # Add filters
            for key, value in self.filters.items():
                if isinstance(value, (list, tuple)):
                    params[f"filter_{key}"] = ",".join(
                        str(v) for v in cast("list[object]", value)
                    )
                else:
                    params[f"filter_{key}"] = str(value)

            # Add sort
            if self.sort_by:
                params["sort"] = f"{self.sort_by}:{self.sort_order}"

            return urlencode(params)

    # Batch serialization optimizer
    class BatchSerializer:
        """Optimized batch serialization for collections of models."""

        @staticmethod
        def serialize_batch(
            models: list[BaseModel],
            output_format: Literal["dict", "json"] = "dict",
            *,
            compact: bool = False,
            parallel: bool = False,
        ) -> list[dict[str, object]] | str:
            """Serialize a batch of models efficiently.

            Args:
                models: List of Pydantic models to serialize
                output_format: Output format ('dict[str, object]' or 'json')
                compact: Use compact serialization
                parallel: Use parallel processing for large batches

            Returns:
                List of dict[str, object]s or JSON array string

            """
            if not models:
                return [] if output_format == "dict" else "[]"

            # Serialization kwargs
            dump_kwargs: dict[str, Any] = {}
            if compact:
                dump_kwargs = {
                    "exclude_unset": True,
                    "exclude_defaults": True,
                    "exclude_none": True,
                }

            parallel_threshold = FlextConstants.Performance.PARALLEL_THRESHOLD
            if parallel and len(models) > parallel_threshold:
                # Use thread pool for large batches
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=4) as executor:
                    if output_format == "dict":

                        def serialize_to_dict(m: BaseModel) -> dict[str, object]:
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
                Optimized JSON schema dict[str, object]

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
        context: dict[str, object] = Field(
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
                default=FlextConstants.Performance.DEFAULT_BATCH_SIZE,
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
                    raise ValueError(error_msg)
                return v

            @classmethod
            def create_bus_config(
                cls,
                bus_config: dict[str, object] | None = None,
                *,
                enable_middleware: bool = True,
                enable_metrics: bool = True,
                enable_caching: bool = True,
                execution_timeout: int = FlextConstants.Defaults.TIMEOUT,
                max_cache_size: int = FlextConstants.Performance.DEFAULT_BATCH_SIZE,
                implementation_path: str = "flext_core.bus:FlextBus",
            ) -> Self:
                """Create bus configuration with defaults and overrides."""
                config_data: dict[str, object] = {
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
            handler_type: Literal["command", "query"] = Field(
                default="command",
                description="Handler type",
            )
            handler_mode: Literal["command", "query"] = Field(
                default="command",
                description="Handler mode",
            )
            command_timeout: int = Field(default=0, description="Command timeout")
            max_command_retries: int = Field(
                default=0, description="Maximum retry attempts"
            )
            metadata: dict[str, object] = Field(
                default_factory=dict, description="Handler metadata"
            )

            @classmethod
            def create_handler_config(
                cls,
                handler_type: Literal["command", "query"],
                *,
                default_name: str | None = None,
                default_id: str | None = None,
                handler_config: dict[str, object] | None = None,
                command_timeout: int = 0,
                max_command_retries: int = 0,
            ) -> Self:
                """Create handler configuration with defaults and overrides."""
                config_data: dict[str, object] = {
                    "handler_id": default_id
                    or f"{handler_type}_handler_{uuid.uuid4().hex[:8]}",
                    "handler_name": default_name or f"{handler_type.title()} Handler",
                    "handler_type": handler_type,
                    "handler_mode": handler_type,
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
            default="command", description="Handler mode"
        )
        timestamp: str = Field(default="", description="Registration timestamp")
        status: Literal["active", "inactive"] = Field(
            default="active",
            description="Registration status",
        )

    # ============================================================================
    # END OF PHASE 8: CQRS CONFIGURATION MODELS
    # ============================================================================
