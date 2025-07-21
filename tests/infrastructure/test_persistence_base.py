"""Tests for flext_core.infrastructure.persistence.base module.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides tests for the flext_core.infrastructure.persistence.base
module. It includes tests for the Repository abstract base class and the
InMemoryRepository implementation.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from flext_core.domain.pydantic_base import DomainEntity
from flext_core.infrastructure.persistence.base import InMemoryRepository, Repository


# Test entities for testing repository functionality
class DemoEntity(DomainEntity):
    """Demo entity for repository tests."""

    name: str
    value: int = 0

    # DomainEntity already has id: UUID field


class TestRepository:
    """Test Repository abstract base class."""

    def test_repository_is_abstract(self) -> None:
        """Test that Repository is abstract and cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            Repository()  # type: ignore[abstract]

    def test_repository_interface_methods(self) -> None:
        """Test that Repository defines the required interface methods."""
        # Check that all abstract methods are defined
        abstract_methods = Repository.__abstractmethods__

        expected_methods = {
            "get_by_id",
            "save",
            "delete",
            "list_all",
        }

        assert abstract_methods == expected_methods

    def test_repository_type_parameters(self) -> None:
        """Test Repository type parameter specification."""
        # Repository should be generic with EntityType and IdType parameters
        assert hasattr(Repository, "__orig_bases__")

        # Test that we can create typed subclasses
        class TestRepo(Repository[DemoEntity, UUID]):
            async def get_by_id(self, entity_id: UUID) -> DemoEntity | None:
                return None

            async def save(self, entity: DemoEntity) -> DemoEntity:
                return entity

            async def delete(self, entity_id: UUID) -> bool:
                return False

            async def list_all(self) -> list[DemoEntity]:
                return []

        # Should be able to instantiate the concrete implementation
        repo = TestRepo()
        assert repo is not None


class TestInMemoryRepository:
    """Test InMemoryRepository implementation."""

    def test_in_memory_repository_creation(self) -> None:
        """Test InMemoryRepository can be created."""
        repo = InMemoryRepository[DemoEntity, UUID]()

        assert repo is not None
        assert hasattr(repo, "_entities")
        assert repo._entities == {}

    @pytest.mark.asyncio
    async def test_save_entity(self) -> None:
        """Test saving entity to in-memory repository."""
        repo = InMemoryRepository[DemoEntity, UUID]()
        test_id = uuid4()
        entity = DemoEntity(id=test_id, name="Test Entity", value=100)

        saved_entity = await repo.save(entity)

        assert saved_entity == entity
        assert test_id in repo._entities
        assert repo._entities[test_id] == entity

    @pytest.mark.asyncio
    async def test_get_by_id_existing(self) -> None:
        """Test getting existing entity by ID."""
        repo = InMemoryRepository[DemoEntity, UUID]()
        test_id = uuid4()
        entity = DemoEntity(id=test_id, name="Another Entity", value=200)

        await repo.save(entity)
        retrieved_entity = await repo.get_by_id(test_id)

        assert retrieved_entity is not None
        assert retrieved_entity == entity
        assert retrieved_entity.id == test_id
        assert retrieved_entity.name == "Another Entity"
        assert retrieved_entity.value == 200

    @pytest.mark.asyncio
    async def test_get_by_id_non_existing(self) -> None:
        """Test getting non-existing entity by ID."""
        repo = InMemoryRepository[DemoEntity, UUID]()
        non_existing_id = uuid4()

        retrieved_entity = await repo.get_by_id(non_existing_id)

        assert retrieved_entity is None

    @pytest.mark.asyncio
    async def test_delete_existing_entity(self) -> None:
        """Test deleting existing entity."""
        repo = InMemoryRepository[DemoEntity, UUID]()
        test_id = uuid4()
        entity = DemoEntity(id=test_id, name="To Delete", value=300)

        await repo.save(entity)
        assert test_id in repo._entities

        result = await repo.delete(test_id)

        assert result is True
        assert test_id not in repo._entities

    @pytest.mark.asyncio
    async def test_delete_non_existing_entity(self) -> None:
        """Test deleting non-existing entity."""
        repo = InMemoryRepository[DemoEntity, UUID]()
        non_existing_id = uuid4()

        result = await repo.delete(non_existing_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_list_all_empty(self) -> None:
        """Test listing all entities from empty repository."""
        repo = InMemoryRepository[DemoEntity, UUID]()

        entities = await repo.list_all()

        assert entities == []

    @pytest.mark.asyncio
    async def test_list_all_with_entities(self) -> None:
        """Test listing all entities with multiple entities."""
        repo = InMemoryRepository[DemoEntity, UUID]()

        id1, id2, id3 = uuid4(), uuid4(), uuid4()
        entity1 = DemoEntity(id=id1, name="Entity 1", value=400)
        entity2 = DemoEntity(id=id2, name="Entity 2", value=500)
        entity3 = DemoEntity(id=id3, name="Entity 3", value=600)

        await repo.save(entity1)
        await repo.save(entity2)
        await repo.save(entity3)

        entities = await repo.list_all()

        assert len(entities) == 3
        assert entity1 in entities
        assert entity2 in entities
        assert entity3 in entities

    @pytest.mark.asyncio
    async def test_save_update_existing_entity(self) -> None:
        """Test updating existing entity."""
        repo = InMemoryRepository[DemoEntity, UUID]()
        test_id = uuid4()

        # Save initial entity
        entity = DemoEntity(id=test_id, name="Original", value=700)
        await repo.save(entity)

        # Update entity
        updated_entity = DemoEntity(id=test_id, name="Updated", value=777)
        saved_entity = await repo.save(updated_entity)

        assert saved_entity == updated_entity
        assert repo._entities[test_id] == updated_entity
        assert repo._entities[test_id].name == "Updated"
        assert repo._entities[test_id].value == 777

    @pytest.mark.asyncio
    async def test_repository_with_different_id_types(self) -> None:
        """Test repository with different ID types."""

        # Test with integer IDs
        class IntEntity(DomainEntity):
            simple_id: int
            name: str

        int_repo = InMemoryRepository[IntEntity, int]()
        int_entity = IntEntity(simple_id=123, name="Integer ID")

        # Note: The save method uses getattr(entity, "id") which gets
        # the UUID id, not simple_id
        # So this test demonstrates the limitation of the current implementation
        await int_repo.save(int_entity)

        # Should be stored with UUID id, not simple_id
        entities = await int_repo.list_all()
        assert len(entities) == 1
        assert entities[0].name == "Integer ID"

    @pytest.mark.asyncio
    async def test_concurrent_operations(self) -> None:
        """Test concurrent repository operations."""
        repo = InMemoryRepository[DemoEntity, UUID]()

        # Create multiple entities
        entities = []
        for i in range(10):
            entity_id = uuid4()
            entity = DemoEntity(id=entity_id, name=f"Entity {i}", value=i * 100)
            entities.append((entity_id, entity))

        # Save all entities
        for _entity_id, entity in entities:
            await repo.save(entity)

        # Verify all entities are stored
        stored_entities = await repo.list_all()
        assert len(stored_entities) == 10

        # Delete some entities (even-indexed)
        for i in range(0, 10, 2):
            entity_id, _ = entities[i]
            result = await repo.delete(entity_id)
            assert result is True

        # Verify correct entities remain
        remaining_entities = await repo.list_all()
        assert len(remaining_entities) == 5

        # Verify correct entities were deleted
        for i in range(10):
            entity_id, _ = entities[i]
            retrieved_entity: DemoEntity | None = await repo.get_by_id(entity_id)
            if i % 2 == 0:
                assert retrieved_entity is None  # Even indices should be deleted
            else:
                assert retrieved_entity is not None  # Odd indices should remain
                assert isinstance(retrieved_entity, DemoEntity)
                assert retrieved_entity.name == f"Entity {i}"

    @pytest.mark.asyncio
    async def test_repository_inheritance_compliance(self) -> None:
        """Test that InMemoryRepository properly implements Repository interface."""
        repo = InMemoryRepository[DemoEntity, UUID]()

        # Verify it's an instance of Repository
        assert isinstance(repo, Repository)

        # Verify all abstract methods are implemented
        assert hasattr(repo, "get_by_id")
        assert hasattr(repo, "save")
        assert hasattr(repo, "delete")
        assert hasattr(repo, "list_all")

        # Verify methods are callable
        assert callable(repo.get_by_id)
        assert callable(repo.save)
        assert callable(repo.delete)
        assert callable(repo.list_all)

    @pytest.mark.asyncio
    async def test_complex_entity_operations(self) -> None:
        """Test complex entity operations and relationships."""
        repo = InMemoryRepository[DemoEntity, UUID]()

        # Create entities with relationships (simulated)
        parent_id, child1_id, child2_id = uuid4(), uuid4(), uuid4()
        parent = DemoEntity(id=parent_id, name="Parent Entity", value=1000)
        child1 = DemoEntity(id=child1_id, name="Child 1", value=100)
        child2 = DemoEntity(id=child2_id, name="Child 2", value=200)

        # Save all entities
        await repo.save(parent)
        await repo.save(child1)
        await repo.save(child2)

        # Verify all saved
        all_entities = await repo.list_all()
        assert len(all_entities) == 3

        # Simulate finding related entities
        parent_entity = await repo.get_by_id(parent_id)
        assert parent_entity is not None
        assert parent_entity.name == "Parent Entity"

        # Simulate bulk operations
        child_ids = [child1_id, child2_id]
        for child_id in child_ids:
            child = await repo.get_by_id(child_id)
            assert child is not None

            # Update child value
            updated_child = DemoEntity(
                id=child.id,
                name=child.name,
                value=child.value * 2,
            )
            await repo.save(updated_child)

        # Verify updates
        updated_child1 = await repo.get_by_id(child1_id)
        updated_child2 = await repo.get_by_id(child2_id)

        assert updated_child1 is not None
        assert updated_child1.value == 200  # 100 * 2
        assert updated_child2 is not None
        assert updated_child2.value == 400  # 200 * 2

    @pytest.mark.asyncio
    async def test_repository_edge_cases(self) -> None:
        """Test repository edge cases and error conditions."""
        repo = InMemoryRepository[DemoEntity, UUID]()

        # Test saving entity and retrieving
        test_id = uuid4()
        entity = DemoEntity(id=test_id, name="Edge Case Entity", value=999)

        saved = await repo.save(entity)
        assert saved == entity

        # Test retrieving saved entity
        retrieved = await repo.get_by_id(test_id)
        assert retrieved == entity

    @pytest.mark.asyncio
    async def test_entity_without_accessible_id(self) -> None:
        """Test saving entity where getattr(entity, 'id') doesn't work as expected."""
        repo = InMemoryRepository[DemoEntity, UUID]()

        # Create entity normally
        entity = DemoEntity(name="Normal Entity", value=123)
        # entity.id will be auto-generated UUID by DomainEntity

        saved = await repo.save(entity)
        assert saved == entity

        # Should be stored with its UUID id
        entities = await repo.list_all()
        assert len(entities) == 1
        assert entities[0].name == "Normal Entity"

        # Can retrieve by its id
        retrieved = await repo.get_by_id(entity.id)
        assert retrieved == entity

    @pytest.mark.asyncio
    async def test_repository_persistence_across_operations(self) -> None:
        """Test that repository maintains state across operations."""
        repo = InMemoryRepository[DemoEntity, UUID]()

        # Initial state
        assert len(await repo.list_all()) == 0

        # Add entity
        id1 = uuid4()
        entity1 = DemoEntity(id=id1, name="Persistent 1", value=1111)
        await repo.save(entity1)
        assert len(await repo.list_all()) == 1

        # Add another entity
        id2 = uuid4()
        entity2 = DemoEntity(id=id2, name="Persistent 2", value=2222)
        await repo.save(entity2)
        assert len(await repo.list_all()) == 2

        # Delete one entity
        await repo.delete(id1)
        remaining = await repo.list_all()
        assert len(remaining) == 1
        assert remaining[0].id == id2

        # Update remaining entity
        updated_entity = DemoEntity(id=id2, name="Updated Persistent", value=3333)
        await repo.save(updated_entity)

        final_entities = await repo.list_all()
        assert len(final_entities) == 1
        assert final_entities[0].name == "Updated Persistent"
        assert final_entities[0].value == 3333

    @pytest.mark.asyncio
    async def test_repository_type_safety(self) -> None:
        """Test repository type safety with different entity types."""
        # Create repository
        test_repo = InMemoryRepository[DemoEntity, UUID]()

        # Create entity
        test_entity = DemoEntity(name="Test Entity", value=1000)

        # Save to repository
        await test_repo.save(test_entity)

        # Verify entity is stored
        test_entities = await test_repo.list_all()

        assert len(test_entities) == 1
        assert isinstance(test_entities[0], DemoEntity)

        # Test that repositories maintain type safety
        retrieved = await test_repo.get_by_id(test_entity.id)
        assert retrieved is not None
        assert isinstance(retrieved, DemoEntity)
