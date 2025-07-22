"""Base repository interfaces and implementations for FLEXT Core.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides base repository interfaces and implementations
for FLEXT Core.  It includes interfaces for CRUD operations and
provides an in-memory implementation for testing.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Generic
from typing import TypeVar

# DIP compliance - use domain repository interface
from flext_core.domain.core import Repository

# Type variables for generic repository
EntityType = TypeVar("EntityType")
IdType = TypeVar("IdType")


class InMemoryRepository[EntityType, IdType](Repository[EntityType, IdType]):
    """In-memory repository implementation for testing."""

    def __init__(self) -> None:
        """Initialize the in-memory repository."""
        self._entities: dict[IdType, EntityType] = {}

    async def find_by_id(self, entity_id: IdType) -> EntityType | None:
        """Get entity by ID.

        Returns:
            The entity if found, None otherwise.

        """
        return self._entities.get(entity_id)

    async def save(self, entity: EntityType) -> EntityType:
        """Save entity.

        Returns:
            The saved entity.

        """
        entity_id = getattr(entity, "id", None)
        if entity_id is not None:
            self._entities[entity_id] = entity
        return entity

    async def delete(self, entity_id: IdType) -> bool:
        """Delete entity by ID.

        Returns:
            True if entity was deleted, False otherwise.

        """
        if entity_id in self._entities:
            del self._entities[entity_id]
            return True
        return False

    async def find_all(self) -> list[EntityType]:
        """List all entities.

        Returns:
            List of all entities.

        """
        return list(self._entities.values())

    async def count(self) -> int:
        """Count total entities.

        Returns:
            Number of entities.

        """
        return len(self._entities)
