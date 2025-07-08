"""InMemory repository tests - 100% coverage.

Tests for the in-memory repository implementation.
"""

from __future__ import annotations

from flext_core import (
    Entity,
    InMemoryRepository,
    Pipeline,
    PipelineName,
)


class TestEntity(Entity[str]):
    """Test entity for repository tests."""

    id: str
    name: str
    value: int = 0


class TestInMemoryRepository:
    """Test InMemoryRepository implementation."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.repo: InMemoryRepository[TestEntity] = InMemoryRepository()

    async def test_save_new_entity(self) -> None:
        """Test saving new entity."""
        entity = TestEntity(id="test-1", name="Test Entity")

        saved = await self.repo.save(entity)

        assert saved.id == "test-1"
        assert saved.name == "Test Entity"
        assert saved == entity

    async def test_save_updates_existing(self) -> None:
        """Test saving updates existing entity."""
        # Save initial
        entity = TestEntity(id="test-1", name="Original", value=1)
        await self.repo.save(entity)

        # Update and save again (bypass immutability for test)
        entity.__dict__["name"] = "Updated"
        entity.__dict__["value"] = 2
        await self.repo.save(entity)

        # Verify update
        retrieved = await self.repo.get("test-1")
        assert retrieved is not None
        assert retrieved.name == "Updated"
        assert retrieved.value == 2

    async def test_get_existing_entity(self) -> None:
        """Test getting existing entity."""
        entity = TestEntity(id="test-1", name="Test")
        await self.repo.save(entity)

        retrieved = await self.repo.get("test-1")

        assert retrieved is not None
        assert retrieved.id == "test-1"
        assert retrieved.name == "Test"

    async def test_get_non_existent_entity(self) -> None:
        """Test getting non-existent entity returns None."""
        retrieved = await self.repo.get("non-existent")
        assert retrieved is None

    async def test_delete_existing_entity(self) -> None:
        """Test deleting existing entity."""
        entity = TestEntity(id="test-1", name="Test")
        await self.repo.save(entity)

        # Delete
        result = await self.repo.delete("test-1")
        assert result is True

        # Verify deleted
        retrieved = await self.repo.get("test-1")
        assert retrieved is None

    async def test_delete_non_existent_entity(self) -> None:
        """Test deleting non-existent entity."""
        result = await self.repo.delete("non-existent")
        assert result is False

    async def test_list_all_with_no_entities(self) -> None:
        """Test list_all when repository is empty."""
        results = await self.repo.list_all()
        assert results == []

    async def test_list_all_entities(self) -> None:
        """Test listing all entities."""
        # Save multiple entities
        entity1 = TestEntity(id="test-1", name="Entity 1")
        entity2 = TestEntity(id="test-2", name="Entity 2")
        entity3 = TestEntity(id="test-3", name="Entity 3")

        await self.repo.save(entity1)
        await self.repo.save(entity2)
        await self.repo.save(entity3)

        # List all
        results = await self.repo.list_all()

        assert len(results) == 3
        ids = {e.id for e in results}
        assert ids == {"test-1", "test-2", "test-3"}

    async def test_count_entities(self) -> None:
        """Test counting entities."""
        # Start with empty
        assert await self.repo.count() == 0

        # Save entities
        entity1 = TestEntity(id="test-1", name="Apple", value=10)
        entity2 = TestEntity(id="test-2", name="Banana", value=20)
        entity3 = TestEntity(id="test-3", name="Apple", value=30)

        await self.repo.save(entity1)
        assert await self.repo.count() == 1

        await self.repo.save(entity2)
        assert await self.repo.count() == 2

        await self.repo.save(entity3)
        assert await self.repo.count() == 3

        # Delete one
        await self.repo.delete("test-2")
        assert await self.repo.count() == 2

    async def test_repository_isolation(self) -> None:
        """Test that each repository instance is isolated."""
        repo1: InMemoryRepository[TestEntity] = InMemoryRepository()
        repo2: InMemoryRepository[TestEntity] = InMemoryRepository()

        # Save to repo1
        entity = TestEntity(id="test-1", name="Test")
        await repo1.save(entity)

        # Verify repo1 has entity, repo2 is empty
        assert await repo1.get("test-1") is not None
        assert await repo2.get("test-1") is None
        assert await repo1.count() == 1
        assert await repo2.count() == 0

        # Clear repo1, verify repo2 still empty
        repo1.clear()
        assert await repo1.count() == 0
        assert await repo2.count() == 0


class TestInMemoryRepositoryWithPipeline:
    """Test InMemoryRepository with Pipeline entities."""

    def setup_method(self) -> None:
        """Setup test fixtures."""
        self.repo: InMemoryRepository[Pipeline] = InMemoryRepository()

    async def test_pipeline_repository_operations(self) -> None:
        """Test repository operations with Pipeline aggregate."""
        # Create pipeline
        pipeline = Pipeline(
            name=PipelineName(value="test-pipeline"),
            description="Test pipeline",
        )

        # Save
        saved = await self.repo.save(pipeline)
        assert saved.id == pipeline.id

        # Get
        retrieved = await self.repo.get(pipeline.id)
        assert retrieved is not None
        assert retrieved.name.value == "test-pipeline"

        # List all pipelines
        all_pipelines = await self.repo.list_all()
        assert len(all_pipelines) == 1
        assert all_pipelines[0].is_active is True

        # Deactivate and save
        pipeline.deactivate()
        await self.repo.save(pipeline)

        # List again and check status
        all_pipelines = await self.repo.list_all()
        assert len(all_pipelines) == 1
        assert all_pipelines[0].is_active is False

        # Delete
        deleted = await self.repo.delete(pipeline.id)
        assert deleted is True
        assert await self.repo.get(pipeline.id) is None

    async def test_pipeline_list_and_clear(self) -> None:
        """Test listing pipelines and clearing repository."""
        # Create pipelines
        pipeline1 = Pipeline(name=PipelineName(value="data-pipeline"))
        pipeline2 = Pipeline(name=PipelineName(value="etl-pipeline"))
        pipeline3 = Pipeline(name=PipelineName(value="data-processor"))

        await self.repo.save(pipeline1)
        await self.repo.save(pipeline2)
        await self.repo.save(pipeline3)

        # List all pipelines
        results = await self.repo.list_all()
        assert len(results) == 3

        names = {p.name.value for p in results}
        assert names == {"data-pipeline", "etl-pipeline", "data-processor"}

        # Test count
        assert await self.repo.count() == 3

        # Clear repository
        self.repo.clear()

        # Verify empty
        assert await self.repo.count() == 0
        assert await self.repo.list_all() == []
