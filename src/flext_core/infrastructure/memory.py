"""In-memory implementations.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Any
from typing import Protocol
from typing import TypeVar
from typing import runtime_checkable

from flext_core.domain.pipeline import Pipeline


@runtime_checkable
class HasId(Protocol):
    """Protocol for entities with ID."""

    id: Any


T = TypeVar("T", bound=HasId)
ID = TypeVar("ID")


class InMemoryRepository[T: HasId]:
    """Generic in-memory repository - SOLID principles."""

    def __init__(self) -> None:
        """Initialize empty repository storage."""
        self._storage: dict[Any, T] = {}

    async def save(self, entity: T) -> T:
        """Save entity to repository.

        Args:
            entity: Entity to save

        Returns:
            Saved entity

        """
        # Get ID from entity (assuming it has an 'id' attribute)
        entity_id = entity.id
        self._storage[entity_id] = entity
        return entity

    async def get(self, entity_id: object) -> T | None:
        """Get entity by ID.

        Args:
            entity_id: ID of entity to retrieve

        Returns:
            Entity if found, None otherwise

        """
        return self._storage.get(entity_id)

    async def delete(self, entity_id: object) -> bool:
        """Delete entity by ID.

        Args:
            entity_id: ID of entity to delete

        Returns:
            True if deleted, False if not found

        """
        if entity_id in self._storage:
            del self._storage[entity_id]
            return True
        return False

    async def list_all(self) -> list[T]:
        """List all entities.

        Returns:
            List of all entities

        """
        return list(self._storage.values())

    async def count(self) -> int:
        """Count total entities.

        Returns:
            Number of entities

        """
        return len(self._storage)

    def clear(self) -> None:
        """Clear all entities from repository."""
        self._storage.clear()


# Type-specific alias for better type safety
PipelineRepository = InMemoryRepository[Pipeline]


__all__ = [
    "HasId",
    "InMemoryRepository",
    "PipelineRepository",
]
