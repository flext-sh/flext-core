"""Core Abstractions - Fundamental Interface Definitions.

Provides the most basic abstractions that all other components build upon.
These are pure interfaces with no implementation details.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

# Type variables for generic abstractions
TEntity = TypeVar("TEntity")
TId = TypeVar("TId")
TValueObject = TypeVar("TValueObject")


class AbstractValueObject(abc.ABC):
    """Base abstraction for all value objects.

    Value objects are immutable objects that represent a descriptive
    aspect of the domain with no conceptual identity.

    🎯 PRINCIPLES:
    - Immutable
    - Equality based on content, not identity
    - No side effects
    """

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        """Value objects are equal if their content is equal."""

    @abc.abstractmethod
    def __hash__(self) -> int:
        """Value objects must be hashable since they're immutable."""


class AbstractEntity[TId](abc.ABC):
    """Base abstraction for all entities.

    Entities are objects that have a distinct identity that runs
    through time and different states.

    🎯 PRINCIPLES:
    - Has identity (ID)
    - Mutable state
    - Equality based on identity, not state
    - Lifecycle management
    """

    @property
    @abc.abstractmethod
    def id(self) -> TId:
        """Unique identifier for this entity."""

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        """Entities are equal if they have the same ID and type."""

    @abc.abstractmethod
    def __hash__(self) -> int:
        """Entities are hashable based on their ID."""


class AbstractRepository[TEntity, TId](abc.ABC):
    """Base abstraction for all repositories.

    Repositories encapsulate the logic needed to access data sources.
    They centralize common data access functionality.

    🎯 PRINCIPLES:
    - Encapsulates data access logic
    - Domain-focused interface
    - Technology-agnostic
    - Unit of Work compatible
    """

    @abc.abstractmethod
    async def find_by_id(self, entity_id: TId) -> TEntity | None:
        """Find entity by its unique identifier."""

    @abc.abstractmethod
    async def find_all(self) -> Sequence[TEntity]:
        """Find all entities in the repository."""

    @abc.abstractmethod
    async def save(self, entity: TEntity) -> None:
        """Persist entity to the repository."""

    @abc.abstractmethod
    async def delete(self, entity: TEntity) -> None:
        """Remove entity from the repository."""


class AbstractService(abc.ABC):
    """Base abstraction for all services.

    Services represent domain concepts that don't naturally fit
    within entities or value objects.

    🎯 PRINCIPLES:
    - Stateless
    - Express domain concepts
    - Coordinate between entities
    - Business logic encapsulation
    """

    @abc.abstractmethod
    def validate_invariants(self) -> bool:
        """Validate service domain invariants."""


class AbstractAggregateRoot(AbstractEntity[TId], abc.ABC):
    """Base abstraction for aggregate roots.

    Aggregate roots are entities that serve as entry points to
    aggregates and ensure consistency boundaries.

    🎯 PRINCIPLES:
    - Entry point to aggregate
    - Maintains consistency
    - Publishes domain events
    - Transaction boundary
    """

    @abc.abstractmethod
    def get_domain_events(self) -> Sequence[Any]:
        """Get all domain events raised by this aggregate."""

    @abc.abstractmethod
    def clear_domain_events(self) -> None:
        """Clear all domain events after they've been processed."""


class AbstractDomainEvent(abc.ABC):
    """Base abstraction for domain events.

    Domain events represent something important that happened
    in the domain that other parts of the system care about.

    🎯 PRINCIPLES:
    - Immutable
    - Past tense naming
    - Contains minimal data
    - Cross-boundary communication
    """

    @property
    @abc.abstractmethod
    def occurred_at(self) -> Any:  # Should be timestamp type
        """When the event occurred."""

    @property
    @abc.abstractmethod
    def event_type(self) -> str:
        """Type identifier for the event."""
