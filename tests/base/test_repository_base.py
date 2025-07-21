"""Tests for BaseComponentRepository class.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from flext_core.base.repository_base import BaseComponentRepository


class MockEntity:
    """Mock entity for testing repository functionality."""

    def __init__(self, name: str, entity_type: str, active: bool = True) -> None:
        """Initialize mock entity."""
        self.name = name
        self.type = entity_type
        self._active = active

    def is_active(self) -> bool:
        """Check if entity is active."""
        return self._active


class MockRepository(BaseComponentRepository[MockEntity]):
    """Mock repository implementation for testing."""

    def __init__(self, entities: list[MockEntity] | None = None) -> None:
        """Initialize mock repository."""
        super().__init__()
        self._test_entities = entities or []

    async def get_by_id(self, entity_id: Any) -> MockEntity | None:
        """Mock implementation of get_by_id."""
        return None  # Not used in the tests we're focusing on

    async def save(self, entity: MockEntity) -> MockEntity:
        """Mock implementation of save."""
        return entity

    async def delete(self, entity_id: Any) -> bool:
        """Mock implementation of delete."""
        return True

    async def list_all(self) -> list[MockEntity]:
        """Mock implementation of list_all."""
        return self._test_entities


class TestBaseComponentRepository:
    """Test BaseComponentRepository functionality."""

    def test_initialization(self) -> None:
        """Test repository initialization."""
        repo = MockRepository()

        assert isinstance(repo._cache, dict)
        assert len(repo._cache) == 0

    def test_get_by_name_found(self) -> None:
        """Test getting entity by name when found."""
        entities = [
            MockEntity("entity1", "type1"),
            MockEntity("entity2", "type2"),
            MockEntity("entity3", "type1"),
        ]
        repo = MockRepository(entities)

        result = repo.get_by_name("entity2")

        assert result is not None
        assert result.name == "entity2"
        assert result.type == "type2"

    def test_get_by_name_not_found(self) -> None:
        """Test getting entity by name when not found."""
        entities = [
            MockEntity("entity1", "type1"),
            MockEntity("entity2", "type2"),
        ]
        repo = MockRepository(entities)

        result = repo.get_by_name("nonexistent")

        assert result is None

    def test_get_by_name_empty_repository(self) -> None:
        """Test getting entity by name from empty repository."""
        repo = MockRepository([])

        result = repo.get_by_name("entity1")

        assert result is None

    def test_get_by_name_entity_without_name_attribute(self) -> None:
        """Test getting entity by name when entity has no name attribute."""

        # Create an entity without name attribute
        class EntityWithoutName:
            def __init__(self, entity_type: str) -> None:
                self.type = entity_type

        entities = [EntityWithoutName("type1")]
        repo = MockRepository([])

        # Mock list_all to return entities without name attribute
        with patch.object(repo, "list_all", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = entities
            result = repo.get_by_name("any_name")

        assert result is None

    def test_get_by_type_found(self) -> None:
        """Test getting entities by type when found."""
        entities = [
            MockEntity("entity1", "type1"),
            MockEntity("entity2", "type2"),
            MockEntity("entity3", "type1"),
            MockEntity("entity4", "type3"),
        ]
        repo = MockRepository(entities)

        result = repo.get_by_type("type1")

        assert len(result) == 2
        assert all(entity.type == "type1" for entity in result)
        assert {entity.name for entity in result} == {"entity1", "entity3"}

    def test_get_by_type_not_found(self) -> None:
        """Test getting entities by type when none found."""
        entities = [
            MockEntity("entity1", "type1"),
            MockEntity("entity2", "type2"),
        ]
        repo = MockRepository(entities)

        result = repo.get_by_type("nonexistent_type")

        assert result == []

    def test_get_by_type_empty_repository(self) -> None:
        """Test getting entities by type from empty repository."""
        repo = MockRepository([])

        result = repo.get_by_type("type1")

        assert result == []

    def test_get_by_type_entity_without_type_attribute(self) -> None:
        """Test getting entities by type when entity has no type attribute."""

        # Create an entity without type attribute
        class EntityWithoutType:
            def __init__(self, name: str) -> None:
                self.name = name

        entities = [EntityWithoutType("entity1")]
        repo = MockRepository([])

        # Mock list_all to return entities without type attribute
        with patch.object(repo, "list_all", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = entities
            result = repo.get_by_type("any_type")

        assert result == []

    def test_get_active(self) -> None:
        """Test getting active entities."""
        entities = [
            MockEntity("entity1", "type1", active=True),
            MockEntity("entity2", "type2", active=False),
            MockEntity("entity3", "type1", active=True),
            MockEntity("entity4", "type3", active=False),
        ]
        repo = MockRepository(entities)

        result = repo.get_active()

        assert len(result) == 2
        assert all(entity.is_active() for entity in result)
        assert {entity.name for entity in result} == {"entity1", "entity3"}

    def test_get_active_empty_repository(self) -> None:
        """Test getting active entities from empty repository."""
        repo = MockRepository([])

        result = repo.get_active()

        assert result == []

    def test_get_active_entity_without_is_active_method(self) -> None:
        """Test getting active entities when entity has no is_active method."""

        # Create an entity without is_active method
        class EntityWithoutIsActive:
            def __init__(self, name: str) -> None:
                self.name = name

        entities = [EntityWithoutIsActive("entity1")]
        repo = MockRepository([])

        # Mock list_all to return entities without is_active method
        with patch.object(repo, "list_all", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = entities
            result = repo.get_active()

        assert result == []

    def test_get_inactive(self) -> None:
        """Test getting inactive entities."""
        entities = [
            MockEntity("entity1", "type1", active=True),
            MockEntity("entity2", "type2", active=False),
            MockEntity("entity3", "type1", active=True),
            MockEntity("entity4", "type3", active=False),
        ]
        repo = MockRepository(entities)

        result = repo.get_inactive()

        assert len(result) == 2
        assert all(not entity.is_active() for entity in result)
        assert {entity.name for entity in result} == {"entity2", "entity4"}

    def test_get_inactive_empty_repository(self) -> None:
        """Test getting inactive entities from empty repository."""
        repo = MockRepository([])

        result = repo.get_inactive()

        assert result == []

    def test_get_inactive_entity_without_is_active_method(self) -> None:
        """Test getting inactive entities when entity has no is_active method."""

        # Create an entity without is_active method
        class EntityWithoutIsActive:
            def __init__(self, name: str) -> None:
                self.name = name

        entities = [EntityWithoutIsActive("entity1")]
        repo = MockRepository([])

        # Mock list_all to return entities without is_active method
        with patch.object(repo, "list_all", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = entities
            result = repo.get_inactive()

        assert result == []

    def test_count_by_type(self) -> None:
        """Test counting entities by type."""
        entities = [
            MockEntity("entity1", "type1"),
            MockEntity("entity2", "type2"),
            MockEntity("entity3", "type1"),
            MockEntity("entity4", "type3"),
        ]
        repo = MockRepository(entities)

        result_type1 = repo.count_by_type("type1")
        result_type2 = repo.count_by_type("type2")
        result_nonexistent = repo.count_by_type("nonexistent")

        assert result_type1 == 2
        assert result_type2 == 1
        assert result_nonexistent == 0

    def test_count_active(self) -> None:
        """Test counting active entities."""
        entities = [
            MockEntity("entity1", "type1", active=True),
            MockEntity("entity2", "type2", active=False),
            MockEntity("entity3", "type1", active=True),
            MockEntity("entity4", "type3", active=False),
        ]
        repo = MockRepository(entities)

        result = repo.count_active()

        assert result == 2

    def test_count_active_empty_repository(self) -> None:
        """Test counting active entities in empty repository."""
        repo = MockRepository([])

        result = repo.count_active()

        assert result == 0

    def test_count_inactive(self) -> None:
        """Test counting inactive entities."""
        entities = [
            MockEntity("entity1", "type1", active=True),
            MockEntity("entity2", "type2", active=False),
            MockEntity("entity3", "type1", active=True),
            MockEntity("entity4", "type3", active=False),
        ]
        repo = MockRepository(entities)

        result = repo.count_inactive()

        assert result == 2

    def test_count_inactive_empty_repository(self) -> None:
        """Test counting inactive entities in empty repository."""
        repo = MockRepository([])

        result = repo.count_inactive()

        assert result == 0


class TestBaseComponentRepositoryCache:
    """Test BaseComponentRepository caching functionality."""

    def test_cache_operations_basic(self) -> None:
        """Test basic cache operations."""
        repo = MockRepository()

        # Test setting and getting cache values
        repo.set_cache("key1", "value1")
        repo.set_cache("key2", 42)

        assert repo.get_cache("key1") == "value1"
        assert repo.get_cache("key2") == 42

    def test_cache_get_nonexistent_key(self) -> None:
        """Test getting nonexistent cache key."""
        repo = MockRepository()

        result = repo.get_cache("nonexistent")

        assert result is None

    def test_cache_overwrite_value(self) -> None:
        """Test overwriting cache value."""
        repo = MockRepository()

        repo.set_cache("key1", "original")
        repo.set_cache("key1", "updated")

        assert repo.get_cache("key1") == "updated"

    def test_cache_complex_values(self) -> None:
        """Test caching complex values."""
        repo = MockRepository()

        complex_value = {"nested": {"data": [1, 2, 3]}}
        repo.set_cache("complex", complex_value)

        result = repo.get_cache("complex")

        assert result == complex_value
        assert result is complex_value  # Should be the same object

    def test_clear_cache(self) -> None:
        """Test clearing cache."""
        repo = MockRepository()

        repo.set_cache("key1", "value1")
        repo.set_cache("key2", "value2")
        assert len(repo._cache) == 2

        repo.clear_cache()

        assert len(repo._cache) == 0
        assert repo.get_cache("key1") is None
        assert repo.get_cache("key2") is None

    def test_get_cache_keys_empty(self) -> None:
        """Test getting cache keys when cache is empty."""
        repo = MockRepository()

        keys = repo.get_cache_keys()

        assert keys == []

    def test_get_cache_keys_with_values(self) -> None:
        """Test getting cache keys with values."""
        repo = MockRepository()

        repo.set_cache("key1", "value1")
        repo.set_cache("key2", "value2")
        repo.set_cache("key3", "value3")

        keys = repo.get_cache_keys()

        assert set(keys) == {"key1", "key2", "key3"}
        assert len(keys) == 3

    def test_get_cache_keys_after_clear(self) -> None:
        """Test getting cache keys after clearing cache."""
        repo = MockRepository()

        repo.set_cache("key1", "value1")
        repo.set_cache("key2", "value2")
        repo.clear_cache()

        keys = repo.get_cache_keys()

        assert keys == []

    def test_cache_none_value(self) -> None:
        """Test caching None value."""
        repo = MockRepository()

        repo.set_cache("none_key", None)

        # This should return None, but it was explicitly set
        result = repo.get_cache("none_key")
        assert result is None

        # The key should exist in cache keys
        keys = repo.get_cache_keys()
        assert "none_key" in keys

    def test_cache_isolation_between_instances(self) -> None:
        """Test that cache is isolated between repository instances."""
        repo1 = MockRepository()
        repo2 = MockRepository()

        repo1.set_cache("key1", "value1")
        repo2.set_cache("key2", "value2")

        assert repo1.get_cache("key1") == "value1"
        assert repo1.get_cache("key2") is None
        assert repo2.get_cache("key1") is None
        assert repo2.get_cache("key2") == "value2"


class TestBaseComponentRepositoryIntegration:
    """Test BaseComponentRepository integration scenarios."""

    def test_filtering_and_counting_consistency(self) -> None:
        """Test that filtering and counting methods are consistent."""
        entities = [
            MockEntity("entity1", "typeA", active=True),
            MockEntity("entity2", "typeA", active=False),
            MockEntity("entity3", "typeB", active=True),
            MockEntity("entity4", "typeB", active=False),
            MockEntity("entity5", "typeA", active=True),
        ]
        repo = MockRepository(entities)

        # Test type filtering consistency
        type_a_entities = repo.get_by_type("typeA")
        type_a_count = repo.count_by_type("typeA")
        assert len(type_a_entities) == type_a_count == 3

        # Test active filtering consistency
        active_entities = repo.get_active()
        active_count = repo.count_active()
        assert len(active_entities) == active_count == 3

        # Test inactive filtering consistency
        inactive_entities = repo.get_inactive()
        inactive_count = repo.count_inactive()
        assert len(inactive_entities) == inactive_count == 2

    def test_all_entities_accounted_for(self) -> None:
        """Test that active + inactive = total entities."""
        entities = [
            MockEntity("entity1", "type1", active=True),
            MockEntity("entity2", "type2", active=False),
            MockEntity("entity3", "type1", active=True),
            MockEntity("entity4", "type3", active=False),
            MockEntity("entity5", "type2", active=True),
        ]
        repo = MockRepository(entities)

        active_count = repo.count_active()
        inactive_count = repo.count_inactive()
        total_entities = len(entities)

        assert active_count + inactive_count == total_entities

    def test_cache_and_filtering_independence(self) -> None:
        """Test that cache operations don't interfere with filtering."""
        entities = [
            MockEntity("entity1", "type1", active=True),
            MockEntity("entity2", "type2", active=False),
        ]
        repo = MockRepository(entities)

        # Perform some cache operations
        repo.set_cache("test_key", "test_value")

        # Filtering should still work correctly
        assert len(repo.get_by_type("type1")) == 1
        assert repo.count_active() == 1
        assert repo.count_inactive() == 1

        # Cache should still work
        assert repo.get_cache("test_key") == "test_value"
