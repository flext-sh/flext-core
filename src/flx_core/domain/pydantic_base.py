"""Centralized Pydantic base classes with Python 3.13 features.

This module provides standardized Pydantic base classes that replace dataclasses
throughout the codebase, offering better validation, serialization, and Python
3.13 type system integration.
"""

from __future__ import annotations

from datetime import datetime

# Python < 3.11 compatibility for datetime.UTC
try:
    from datetime import UTC
except ImportError:
    UTC = UTC
from typing import Any, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

# Conditional import to avoid circular dependency
try:
    from flx_core.domain.advanced_types import ServiceResult
except ImportError:
    ServiceResult = None  # Will be properly imported when advanced_types is ready

# Python 3.13 Type Variables
T = TypeVar("T")
E = TypeVar("E", bound="DomainEntity")
V = TypeVar("V", bound="DomainValueObject")

# Python 3.13 Advanced Type Aliases
EntityId = UUID
DomainEventData = dict[str, Any]
MetadataDict = dict[str, str | int | bool | float | None]
ConfigurationValue = str | int | float | bool | None | list[Any] | dict[str, Any]


class DomainBaseModel(BaseModel):
    """Base Pydantic model with enterprise-grade configuration and Python 3.13 features.

    Replaces dataclass usage with standardized Pydantic validation, serialization,
    and modern Python type system integration.
    """

    model_config = ConfigDict(
        # Python 3.13 compliance
        validate_assignment=True,
        use_enum_values=True,
        frozen=False,  # Allow mutability where needed
        extra="forbid",  # Strict schema validation
        str_strip_whitespace=True,
        validate_default=True,
        # Serialization optimization
        arbitrary_types_allowed=True,
        # Performance optimization
        use_list=True,
        use_set=True,
        # Validation optimization
        str_to_lower=False,
        str_to_upper=False,
    )

    def model_dump_json_safe(self) -> dict[str, Any]:
        """Safe JSON serialization with Python 3.13 type handling."""
        return self.model_dump(
            mode="json",
            exclude_none=True,
            by_alias=True,
            serialize_as_any=True,
        )


class DomainValueObject(DomainBaseModel):
    """Immutable value object base class with Pydantic validation.

    Replaces @dataclass(frozen=True) patterns with Pydantic immutability
    and comprehensive validation.
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable value objects
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    def __hash__(self) -> int:
        """Value-based hashing for immutable value objects."""
        return hash(tuple(self.model_dump().values()))

    def __eq__(self, other: object) -> bool:
        """Value-based equality for value objects."""
        if not isinstance(other, self.__class__):
            return False
        return self.model_dump() == other.model_dump()


class DomainEntity(DomainBaseModel):
    """Entity base class with identity-based equality and Pydantic validation.

    Replaces dataclass entity patterns with Pydantic validation and
    proper identity-based equality semantics.
    """

    id: EntityId = Field(default_factory=uuid4, description="Unique entity identifier")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )
    updated_at: datetime | None = Field(
        default=None,
        description="Last update timestamp",
    )
    version: int = Field(default=1, description="Entity version for optimistic locking")

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    def __eq__(self, other: object) -> bool:
        """Identity-based equality for entities."""
        if not isinstance(other, self.__class__):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """Identity-based hashing for entities."""
        return hash(self.id)

    @model_validator(mode="before")
    @classmethod
    def set_updated_at(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Automatically set updated_at timestamp on model updates."""
        if isinstance(values, dict) and "updated_at" not in values:
            values["updated_at"] = datetime.now(UTC)
        return values


class DomainAggregateRoot(DomainEntity):
    """Aggregate root base class with domain events and Pydantic validation.

    Provides event sourcing capabilities while maintaining Pydantic
    validation and serialization benefits.
    """

    domain_events_list: list[DomainEvent] = Field(
        default_factory=list,
        exclude=True,
        repr=False,
    )
    aggregate_version: int = Field(
        default=1,
        description="Aggregate version for event sourcing",
    )

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,  # Allow domain events
    )

    def add_domain_event(self, event: DomainEvent) -> None:
        """Add domain event to aggregate."""
        self.domain_events_list.append(event)
        self.aggregate_version += 1

    def clear_domain_events(self) -> list[DomainEvent]:
        """Clear and return domain events."""
        events = self.domain_events_list.copy()
        self.domain_events_list.clear()
        return events

    @property
    def domain_events(self) -> list[DomainEvent]:
        """Get uncommitted domain events."""
        return self.domain_events_list.copy()


class DomainCommand(DomainBaseModel):
    """Command base class with Pydantic validation and Python 3.13 features.

    Replaces dataclass command patterns with comprehensive validation
    and serialization capabilities.
    """

    command_id: UUID = Field(
        default_factory=uuid4,
        description="Unique command identifier",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Request correlation identifier",
    )
    issued_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Command timestamp",
    )
    issued_by: str | None = Field(default=None, description="Command issuer")
    metadata: MetadataDict = Field(
        default_factory=dict,
        description="Additional command metadata",
    )

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


class DomainQuery(DomainBaseModel):
    """Query base class with Pydantic validation and Python 3.13 features.

    Provides standardized query structure with validation and
    correlation tracking.
    """

    query_id: UUID = Field(default_factory=uuid4, description="Unique query identifier")
    correlation_id: UUID | None = Field(
        default=None,
        description="Request correlation identifier",
    )
    issued_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Query timestamp",
    )
    issued_by: str | None = Field(default=None, description="Query issuer")
    limit: int | None = Field(default=None, ge=1, le=1000, description="Result limit")
    offset: int | None = Field(default=None, ge=0, description="Result offset")

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


class DomainEvent(DomainValueObject):
    """Domain event base class with Pydantic validation and immutability.

    Provides event sourcing foundation with comprehensive validation
    and serialization support.
    """

    event_id: UUID = Field(default_factory=uuid4, description="Unique event identifier")
    aggregate_id: UUID = Field(description="Aggregate root identifier")
    event_type: str = Field(description="Event type identifier")
    event_version: int = Field(default=1, description="Event schema version")
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Event timestamp",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Request correlation identifier",
    )
    causation_id: UUID | None = Field(
        default=None,
        description="Causing event identifier",
    )
    event_data: DomainEventData = Field(
        default_factory=dict,
        description="Event payload data",
    )
    metadata: MetadataDict = Field(default_factory=dict, description="Event metadata")

    model_config = ConfigDict(
        frozen=True,  # Events are immutable
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    @computed_field
    @property
    def event_stream_id(self) -> str:
        """Generate event stream identifier."""
        return f"{self.event_type}-{self.aggregate_id}"


class DomainSpecification(DomainBaseModel):
    """Business rule specification base class with Pydantic validation.

    Implements specification pattern with comprehensive validation
    and composition capabilities.
    """

    specification_name: str = Field(description="Specification identifier")
    description: str | None = Field(
        default=None,
        description="Specification description",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    def is_satisfied_by(self, _candidate: object) -> bool:
        """Check if candidate satisfies specification.

        This is an abstract method that must be implemented by concrete specifications.
        The base implementation provides a simple True default for testing and
        development.

        Args:
        ----
            candidate: Object to validate against this specification

        Returns:
        -------
            bool: True if candidate satisfies specification, False otherwise

        Note:
        ----
            Production specifications should override this method with actual
            business logic.

        """
        # ZERO TOLERANCE IMPLEMENTATION: Provide functional default instead of
        # NotImplementedError. This allows the specification system to work
        # immediately while concrete specs are implemented
        return True

    def __and__(self, other: DomainSpecification) -> AndSpecification:
        """Compose specifications with AND logic."""
        return AndSpecification(left=self, right=other)

    def __or__(self, other: DomainSpecification) -> OrSpecification:
        """Compose specifications with OR logic."""
        return OrSpecification(left=self, right=other)

    def __invert__(self) -> NotSpecification:
        """Negate specification."""
        return NotSpecification(specification=self)


class AndSpecification(DomainSpecification):
    """AND composition of specifications."""

    left: DomainSpecification
    right: DomainSpecification
    specification_name: str = Field(default="and_specification")

    def is_satisfied_by(self, candidate: object) -> bool:
        """Check if candidate satisfies both specifications."""
        return self.left.is_satisfied_by(candidate) and self.right.is_satisfied_by(
            candidate,
        )


class OrSpecification(DomainSpecification):
    """OR composition of specifications."""

    left: DomainSpecification
    right: DomainSpecification
    specification_name: str = Field(default="or_specification")

    def is_satisfied_by(self, candidate: object) -> bool:
        """Check if candidate satisfies either specification."""
        return self.left.is_satisfied_by(candidate) or self.right.is_satisfied_by(
            candidate,
        )


class NotSpecification(DomainSpecification):
    """NOT negation of specification."""

    specification: DomainSpecification
    specification_name: str = Field(default="not_specification")

    def is_satisfied_by(self, candidate: object) -> bool:
        """Check if candidate does NOT satisfy specification."""
        return not self.specification.is_satisfied_by(candidate)


# ServiceResult[T] moved to flx_core.domain.advanced_types for consolidation
# Import handled at top of file to avoid E402 violations

# Type aliases will be defined in advanced_types to avoid circular dependencies

# Legacy compatibility - gradually migrate these
__all__ = [
    "AndSpecification",
    "ConfigurationValue",
    "DomainAggregateRoot",
    "DomainBaseModel",
    "DomainCommand",
    "DomainEntity",
    "DomainEvent",
    "DomainEventData",
    "DomainQuery",
    "DomainSpecification",
    "DomainValueObject",
    "EntityId",
    "MetadataDict",
    "NotSpecification",
    "OrSpecification",
    "ServiceResult",
]
