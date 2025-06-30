"""Repository and Unit of Work contracts for FLEXT Core."""

from abc import ABC, abstractmethod
from typing import Any, Protocol, TypeVar, runtime_checkable

# Type variables for generic repository pattern
EntityId = TypeVar("EntityId")
Entity = TypeVar("Entity")
T = TypeVar("T")


@runtime_checkable
class EntityInterface(Protocol):
    """Protocol for entities with an ID."""

    id: Any


class RepositoryInterface[Entity, EntityId](ABC):
    """Abstract repository interface."""

    @abstractmethod
    async def get_by_id(self, entity_id: EntityId) -> Entity | None:
        """Get entity by ID."""

    @abstractmethod
    async def get_all(self) -> list[Entity]:
        """Get all entities."""

    @abstractmethod
    async def save(self, entity: Entity) -> Entity:
        """Save entity."""

    @abstractmethod
    async def delete(self, entity_id: EntityId) -> bool:
        """Delete entity by ID."""

    @abstractmethod
    async def exists(self, entity_id: EntityId) -> bool:
        """Check if entity exists."""


class UnitOfWorkInterface(ABC):
    """Abstract unit of work interface."""

    @abstractmethod
    async def __aenter__(self) -> UnitOfWorkInterface:
        """Enter async context manager."""

    @abstractmethod
    async def __aexit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object | None) -> None:
        """Exit async context manager."""

    @abstractmethod
    async def commit(self) -> None:
        """Commit transaction."""

    @abstractmethod
    async def rollback(self) -> None:
        """Rollback transaction."""

    @abstractmethod
    def get_repository(self, entity_class: type[T], model_class: type[Any]) -> RepositoryInterface[T, Any]:
        """Get repository for entity type."""
