"""Base repository class for FLEXT components.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides a unified base class for all repository
implementations to eliminate duplication.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar

if TYPE_CHECKING:
    from flext_core.domain.core import Repository

T = TypeVar("T")


class BaseComponentRepository[T]:
    """Base repository class for FLEXT components.

    Provides common repository functionality for taps, targets,
    and other components.
    """

    def __init__(self, repository: Repository[T, Any]) -> None:
        """Initialize repository helper.

        Args:
            repository: Concrete repository implementation to use

        """
        self._repository = repository
        self._cache: dict[str, Any] = {}

    def get_by_name(self, name: str) -> T | None:
        """Get entity by name.

        Args:
            name: Entity name

        Returns:
            Entity or None if not found

        """
        entities = asyncio.run(self._repository.find_all())
        for entity in entities:
            if hasattr(entity, "name") and entity.name == name:
                return entity
        return None

    def get_by_type(self, entity_type: str) -> list[T]:
        """Get entities by type.

        Args:
            entity_type: Entity type

        Returns:
            List of entities of the specified type

        """
        all_entities = asyncio.run(self._repository.find_all())
        return [
            entity
            for entity in all_entities
            if hasattr(entity, "type") and entity.type == entity_type
        ]

    def get_active(self) -> list[T]:
        """Get active entities.

        Returns:
            List of active entities

        """
        all_entities = asyncio.run(self._repository.find_all())
        return [
            entity
            for entity in all_entities
            if hasattr(entity, "is_active") and entity.is_active()
        ]

    def get_inactive(self) -> list[T]:
        """Get inactive entities.

        Returns:
            List of inactive entities

        """
        all_entities = asyncio.run(self._repository.find_all())
        return [
            entity
            for entity in all_entities
            if hasattr(entity, "is_active") and not entity.is_active()
        ]

    def count_by_type(self, entity_type: str) -> int:
        """Count entities by type.

        Args:
            entity_type: Entity type

        Returns:
            Number of entities of the specified type

        """
        return len(self.get_by_type(entity_type))

    def count_active(self) -> int:
        """Count active entities.

        Returns:
            Number of active entities

        """
        return len(self.get_active())

    def count_inactive(self) -> int:
        """Count inactive entities.

        Returns:
            Number of inactive entities

        """
        return len(self.get_inactive())

    def get_cache(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None

        """
        return self._cache.get(key)

    def set_cache(self, key: str, value: Any) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache

        """
        self._cache[key] = value

    def clear_cache(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def get_cache_keys(self) -> list[str]:
        """Get all cache keys.

        Returns:
            List of cache keys

        """
        return list(self._cache.keys())
