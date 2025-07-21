"""Testing utilities for FLEXT components.

This module consolidates testing utilities to eliminate duplication
across different projects.
"""

from __future__ import annotations

from typing import TypeVar

from flext_core.domain import Repository

T = TypeVar("T")
ID = TypeVar("ID")


class MockRepository(Repository[T, ID]):
    """Generic mock repository for testing.

    This class provides a simple mock implementation of the Repository
    interface for testing purposes.
    """

    def __init__(self) -> None:
        """Initialize the mock repository."""
        self._storage: dict[ID, T] = {}

    async def save(self, entity: T) -> T:
        """Save an entity to the mock repository.

        Args:
            entity: The entity to save.

        Returns:
            The saved entity.

        """
        # For now, just return the entity as-is
        # In a real implementation, this would assign an ID and store it
        return entity

    async def get(self, entity_id: ID) -> T | None:
        """Get an entity by ID from the mock repository.

        Args:
            entity_id: The ID of the entity to retrieve.

        Returns:
            The entity if found, None otherwise.

        """
        return self._storage.get(entity_id)

    async def delete(self, entity_id: ID) -> bool:
        """Delete an entity from the mock repository.

        Args:
            entity_id: The ID of the entity to delete.

        Returns:
            True if the entity was deleted, False otherwise.

        """
        if entity_id in self._storage:
            del self._storage[entity_id]
            return True
        return False

    async def find_all(self) -> list[T]:
        """Find all entities in the mock repository.

        Returns:
            List of all entities.

        """
        return list(self._storage.values())
