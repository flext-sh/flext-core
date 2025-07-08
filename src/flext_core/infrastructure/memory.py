"""In-memory implementations - COMPLETE and EFFICIENT.

All repository implementations in ONE place.
Zero duplication, maximum performance.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from flext_core.domain import Repository
from flext_core.domain.pipeline import Pipeline


# Define a protocol for entities with ID
class HasId(Protocol):
    """Protocol for entities with ID."""

    id: Any


T = TypeVar("T", bound=HasId)
ID = TypeVar("ID")


class InMemoryRepository(Repository[T, Any]):
    """Generic in-memory repository - SOLID principles."""

    def __init__(self) -> None:
        """Initialize empty in-memory storage."""
        self._storage: dict[Any, T] = {}

    async def save(self, entity: T) -> T:
        """Save entity."""
        # Get ID from entity (assuming it has an 'id' attribute)
        entity_id = entity.id
        self._storage[entity_id] = entity
        return entity

    async def get(self, entity_id: Any) -> T | None:  # noqa: ANN401
        """Get entity by ID."""
        return self._storage.get(entity_id)

    async def delete(self, entity_id: Any) -> bool:  # noqa: ANN401
        """Delete entity."""
        if entity_id in self._storage:
            del self._storage[entity_id]
            return True
        return False

    async def list_all(self) -> list[T]:
        """List all entities."""
        return list(self._storage.values())

    async def count(self) -> int:
        """Count entities."""
        return len(self._storage)

    def clear(self) -> None:
        """Clear all entities."""
        self._storage.clear()


# Type-specific alias for better type safety
PipelineRepository = InMemoryRepository[Pipeline]
