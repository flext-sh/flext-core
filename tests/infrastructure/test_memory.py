"""Tests for flext_core.infrastructure.memory module."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from flext_core.infrastructure.memory import HasId, InMemoryRepository


class MockEntity:
    """Mock entity for repository testing."""

    def __init__(self, entity_id: UUID, name: str, value: int = 0) -> None:
        self.id = entity_id
        self.name = name
        self.value = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MockEntity):
            return False
        return (
            self.id == other.id
            and self.name == other.name
            and self.value == other.value
        )

    def __hash__(self) -> int:
        return hash((self.id, self.name, self.value))


class TestInMemoryRepository:
    """Test InMemoryRepository functionality."""

    @pytest.fixture
    def repository(self) -> InMemoryRepository[MockEntity, UUID]:
        """Create InMemoryRepository for testing."""
        return InMemoryRepository[MockEntity, UUID]()

    @pytest.mark.asyncio
    async def test_save_entity_success(
        self, repository: InMemoryRepository[MockEntity, UUID]
    ) -> None:
        """Test successful entity save."""
        entity_id = uuid4()
        entity = MockEntity(entity_id, "test_entity", 42)

        result = await repository.save(entity)

        assert result == entity

        # Verify entity is stored
        get_result = await repository.get(entity_id)
        assert get_result == entity

    @pytest.mark.asyncio
    async def test_save_duplicate_entity_overwrites(
        self, repository: InMemoryRepository[MockEntity, UUID]
    ) -> None:
        """Test saving duplicate entity overwrites existing."""
        entity_id = uuid4()
        entity1 = MockEntity(entity_id, "entity1")
        entity2 = MockEntity(entity_id, "entity2")

        # Save first entity
        await repository.save(entity1)

        # Save duplicate (should overwrite)
        result = await repository.save(entity2)
        assert result == entity2

        # Verify overwrite
        get_result = await repository.get(entity_id)
        assert get_result is not None
        assert get_result.name == "entity2"

    @pytest.mark.asyncio
    async def test_get_entity_success(
        self, repository: InMemoryRepository[MockEntity, UUID]
    ) -> None:
        """Test successful entity retrieval."""
        entity_id = uuid4()
        entity = MockEntity(entity_id, "test_entity")

        await repository.save(entity)
        result = await repository.get(entity_id)

        assert result == entity

    @pytest.mark.asyncio
    async def test_get_entity_not_found(
        self, repository: InMemoryRepository[MockEntity, UUID]
    ) -> None:
        """Test entity retrieval when ID not found."""
        non_existent_id = uuid4()

        result = await repository.get(non_existent_id)

        assert result is None

    @pytest.mark.asyncio
    async def test_delete_entity_success(
        self, repository: InMemoryRepository[MockEntity, UUID]
    ) -> None:
        """Test successful entity deletion."""
        entity_id = uuid4()
        entity = MockEntity(entity_id, "to_delete")

        # Save entity
        await repository.save(entity)

        # Delete entity
        result = await repository.delete(entity_id)

        assert result is True

        # Verify entity is deleted
        get_result = await repository.get(entity_id)
        assert get_result is None

    @pytest.mark.asyncio
    async def test_delete_non_existent_entity_returns_false(
        self, repository: InMemoryRepository[MockEntity, UUID]
    ) -> None:
        """Test deleting non-existent entity returns False."""
        non_existent_id = uuid4()

        result = await repository.delete(non_existent_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_list_all_entities(
        self, repository: InMemoryRepository[MockEntity, UUID]
    ) -> None:
        """Test listing all entities."""
        entities = [
            MockEntity(uuid4(), "entity1", 1),
            MockEntity(uuid4(), "entity2", 2),
            MockEntity(uuid4(), "entity3", 3),
        ]

        # Add entities
        for entity in entities:
            await repository.save(entity)

        # List all entities
        result = await repository.list_all()

        assert len(result) == 3

        # Verify all entities are present
        for entity in entities:
            assert any(e.id == entity.id for e in result)

    @pytest.mark.asyncio
    async def test_list_all_empty_repository(
        self, repository: InMemoryRepository[MockEntity, UUID]
    ) -> None:
        """Test listing entities from empty repository."""
        result = await repository.list_all()

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_count_entities(
        self, repository: InMemoryRepository[MockEntity, UUID]
    ) -> None:
        """Test counting entities in repository."""
        # Count empty repository
        count_empty = await repository.count()
        assert count_empty == 0

        # Add entities
        for i in range(5):
            entity = MockEntity(uuid4(), f"entity_{i}")
            await repository.save(entity)

        # Count populated repository
        count_populated = await repository.count()
        assert count_populated == 5

    @pytest.mark.asyncio
    async def test_clear_repository(
        self, repository: InMemoryRepository[MockEntity, UUID]
    ) -> None:
        """Test clearing repository."""
        # Add entities
        for i in range(3):
            entity = MockEntity(uuid4(), f"entity_{i}")
            await repository.save(entity)

        # Verify entities exist
        count_before = await repository.count()
        assert count_before == 3

        # Clear repository
        repository.clear()

        # Verify repository is empty
        count_after = await repository.count()
        assert count_after == 0


class TestProtocolImplementation:
    """Test HasId protocol implementation."""

    def test_has_id_protocol_compliance(self) -> None:
        """Test that MockEntity implements HasId protocol."""
        entity_id = uuid4()
        entity = MockEntity(entity_id, "test")

        # Should have id attribute
        assert hasattr(entity, "id")
        assert entity.id == entity_id

        # Should be recognized as HasId
        assert isinstance(entity, HasId)


class TestMemoryStorageIntegration:
    """Test integration between repository and storage."""

    @pytest.mark.asyncio
    async def test_repository_storage_integration(self) -> None:
        """Test repository and storage working together."""
        repository = InMemoryRepository[MockEntity, UUID]()

        # Add multiple entities
        entities = []
        for i in range(5):
            entity = MockEntity(uuid4(), f"entity_{i}", i)
            entities.append(entity)
            await repository.save(entity)

        # Verify all entities can be retrieved
        for entity in entities:
            get_result = await repository.get(entity.id)
            assert get_result == entity

        # Verify list operation
        list_result = await repository.list_all()
        assert len(list_result) == 5

    @pytest.mark.asyncio
    async def test_repository_persistence_across_operations(self) -> None:
        """Test that repository maintains data across multiple operations."""
        repository = InMemoryRepository[MockEntity, UUID]()

        entity_id = uuid4()
        original_entity = MockEntity(entity_id, "persistent", 100)

        # Save entity
        await repository.save(original_entity)

        # Perform other operations
        other_id = uuid4()
        other_entity = MockEntity(other_id, "other", 200)
        await repository.save(other_entity)
        await repository.delete(other_id)

        # Verify original entity still exists
        result = await repository.get(entity_id)
        assert result == original_entity
