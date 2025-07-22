"""In-memory implementations with Dependency Inversion Principle compliance.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

DIP-compliant infrastructure implementations that depend on domain abstractions,
not on concrete infrastructure details.
"""

from __future__ import annotations

from typing import Any
from typing import Protocol
from typing import TypeVar
from typing import runtime_checkable

# DIP compliance - depend on domain abstractions, not infrastructure concretions
from flext_core.domain.core import Repository


@runtime_checkable
class HasId(Protocol):
    """Protocol for entities with ID."""

    id: Any


T = TypeVar("T", bound=HasId)
ID = TypeVar("ID")


class InMemoryRepository[T: HasId, ID](Repository[T, ID]):
    """Generic in-memory repository - DIP compliant implementation."""

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

    async def find_by_id(self, entity_id: ID) -> T | None:
        """Find an entity by ID.

        Args:
            entity_id: ID of entity to retrieve

        Returns:
            Entity if found, None otherwise

        """
        return self._storage.get(entity_id)

    async def delete(self, entity_id: ID) -> bool:
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

    async def find_all(self) -> list[T]:
        """Find all entities.

        Returns:
            List of all entities

        Raises:
            RepositoryError: If find operation fails

        """
        return list(self._storage.values())

    async def count(self) -> int:
        """Count total entities.

        Returns:
            Number of entities

        Raises:
            RepositoryError: If count operation fails

        """
        return len(self._storage)

    def clear(self) -> None:
        """Clear all entities from repository."""
        self._storage.clear()


# No type-specific aliases - use dependency injection instead


__all__ = [
    "HasId",
    "InMemoryRepository",
    # No concrete aliases exported
]
