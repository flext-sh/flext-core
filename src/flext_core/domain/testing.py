"""Testing utilities for FLEXT components.

This module consolidates testing utilities to eliminate duplication
across different projects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import TypeVar

from flext_core.foundation.abstractions import AbstractRepository

if TYPE_CHECKING:
    from collections.abc import Sequence

T = TypeVar("T")
ID = TypeVar("ID")


class MockRepository(AbstractRepository[T, ID]):
    """Generic mock repository for testing.

    This class provides a simple mock implementation of the Repository
    interface for testing purposes.
    """

    def __init__(self) -> None:
        """Initialize the mock repository."""
        self._storage: dict[ID, T] = {}

    async def save(self, entity: T) -> None:
        """Save an entity to the mock repository.

        Args:
            entity: The entity to save.

        """
        # Store entity using its ID
        entity_id = getattr(entity, "id", None)
        if entity_id is not None:
            self._storage[entity_id] = entity

    async def find_by_id(self, entity_id: ID) -> T | None:
        """Get an entity by ID from the mock repository.

        Args:
            entity_id: The ID of the entity to retrieve.

        Returns:
            The entity if found, None otherwise.

        """
        return self._storage.get(entity_id)

    async def delete(self, entity: T) -> None:
        """Delete an entity from the mock repository.

        Args:
            entity: The entity to delete.

        """
        entity_id = getattr(entity, "id", None)
        if entity_id is not None and entity_id in self._storage:
            del self._storage[entity_id]

    async def find_all(self) -> Sequence[T]:
        """Find all entities in the mock repository.

        Returns:
            List of all entities.

        """
        return list(self._storage.values())

    async def count(self) -> int:
        """Count entities in the mock repository.

        Returns:
            Number of entities in the repository.

        """
        return len(self._storage)
