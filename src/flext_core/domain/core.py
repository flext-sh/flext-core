"""Core domain abstractions - SINGLE SOURCE OF TRUTH.

Modern Python 3.13 + Pydantic v2 + SOLID principles.
Zero tolerance for duplication.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, ClassVar, TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


# Domain Exceptions
class DomainError(Exception):
    """Base domain exception."""


class ValidationError(DomainError):
    """Domain validation error."""


class RepositoryError(DomainError):
    """Repository operation error."""


class NotFoundError(RepositoryError):
    """Entity not found error."""


# Modern type variables
T = TypeVar("T")
ID = TypeVar("ID")


class DomainModel(BaseModel):
    """Base for ALL domain models - SINGLE CONFIGURATION."""

    model_config = ConfigDict(
        # Performance
        validate_assignment=True,
        use_enum_values=True,
        arbitrary_types_allowed=False,
        # Safety
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        # Modern
        populate_by_name=True,
    )


class ValueObject(DomainModel):
    """Immutable value objects."""

    model_config = ConfigDict(
        **DomainModel.model_config,
        frozen=True,  # Immutable
    )


class Entity[ID](DomainModel):
    """Entities with identity."""

    id: ID = Field(default_factory=lambda: uuid4())  # type: ignore[assignment]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime | None = None

    def __eq__(self, other: object) -> bool:
        """Check equality based on ID."""
        return isinstance(other, self.__class__) and self.id == other.id

    def __hash__(self) -> int:
        """Generate hash based on class and ID."""
        return hash((self.__class__, self.id))


class DomainEvent(ValueObject):
    """Base domain event."""

    event_id: UUID = Field(default_factory=uuid4)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    event_type: str = Field(init=False)

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: ANN401
        """Auto-set event type from class name on subclass creation."""
        super().__init_subclass__(**kwargs)
        # Auto-set event type from class name
        cls.model_fields["event_type"].default = cls.__name__


class AggregateRoot(Entity[ID]):
    """Aggregate root with domain events."""

    _events: ClassVar[list[DomainEvent]] = []

    def add_event(self, event: DomainEvent) -> None:
        """Add domain event to aggregate."""
        self._events.append(event)

    @classmethod
    def get_events(cls) -> list[DomainEvent]:
        """Get and clear all domain events."""
        events = cls._events.copy()
        cls._events.clear()
        return events


class Repository[T, ID](ABC):
    """Repository interface."""

    @abstractmethod
    async def save(self, entity: T) -> T:
        """Save entity to repository."""
        ...

    @abstractmethod
    async def get(self, entity_id: ID) -> T | None:
        """Get entity by ID from repository."""
        ...

    @abstractmethod
    async def delete(self, entity_id: ID) -> bool:
        """Delete entity by ID from repository."""
        ...


# Result type for operations
class ServiceResult[T]:
    """Type-safe result pattern."""

    def __init__(self, *, success: bool, data: T | None = None, error: str | None = None) -> None:
        """Initialize service result."""
        self.success = success
        self.data = data
        self.error = error

    @classmethod
    def ok(cls, data: T) -> ServiceResult[T]:
        """Create successful result."""
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error: str) -> ServiceResult[T]:
        """Create failed result."""
        return cls(success=False, error=error)

    def unwrap(self) -> T:
        """Unwrap successful result or raise error."""
        if not self.success or self.data is None:
            msg = f"Cannot unwrap failed result: {self.error}"
            raise RuntimeError(msg)
        return self.data
