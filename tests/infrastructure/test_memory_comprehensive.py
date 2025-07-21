"""Comprehensive tests for flext_core.infrastructure.memory module.

This file provides additional test coverage to reach 95%+ coverage,
complementing the existing test_memory.py without duplication.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from flext_core.domain.pipeline import Pipeline, PipelineName
from flext_core.infrastructure.memory import (
    HasId,
    InMemoryRepository,
)


class TestHasIdProtocol:
    """Test HasId protocol comprehensive coverage."""

    def test_has_id_protocol_runtime_checkable(self) -> None:
        """Test HasId protocol runtime checking."""

        # Test with valid entity
        class ValidEntity:
            def __init__(self, entity_id: str) -> None:
                self.id = entity_id

        valid_entity = ValidEntity("test_id")
        assert isinstance(valid_entity, HasId)

        # Test with invalid entity (no id attribute)
        class InvalidEntity:
            def __init__(self, name: str) -> None:
                self.name = name

        invalid_entity = InvalidEntity("test_name")
        assert not isinstance(invalid_entity, HasId)

    def test_has_id_protocol_with_different_id_types(self) -> None:
        """Test HasId protocol with different ID types."""

        # Test with UUID
        class UuidEntity:
            def __init__(self, entity_id: UUID) -> None:
                self.id = entity_id

        uuid_entity = UuidEntity(uuid4())
        assert isinstance(uuid_entity, HasId)

        # Test with int ID
        class IntEntity:
            def __init__(self, entity_id: int) -> None:
                self.id = entity_id

        int_entity = IntEntity(123)
        assert isinstance(int_entity, HasId)

        # Test with string ID
        class StrEntity:
            def __init__(self, entity_id: str) -> None:
                self.id = entity_id

        str_entity = StrEntity("string_id")
        assert isinstance(str_entity, HasId)

    def test_has_id_protocol_any_annotation(self) -> None:
        """Test HasId protocol with Any type annotation."""
        from flext_core.infrastructure.memory import HasId

        # Verify the protocol has the expected annotation
        assert hasattr(HasId, "__annotations__")
        assert "id" in HasId.__annotations__
        # The annotation is Any, but the exact import path may vary
        assert str(HasId.__annotations__["id"]).endswith("Any")


class TestInMemoryRepositoryEdgeCases:
    """Test InMemoryRepository edge cases and error conditions."""

    class ComplexEntity:
        """Entity with complex ID structure."""

        def __init__(self, entity_id: tuple[str, int], data: dict[str, Any]) -> None:
            self.id = entity_id
            self.data = data

        def __eq__(self, other: object) -> bool:
            return (
                isinstance(other, self.__class__)
                and self.id == other.id
                and self.data == other.data
            )

        def __hash__(self) -> int:
            return hash((self.id, tuple(sorted(self.data.items()))))

    @pytest.fixture
    def repository(
        self,
    ) -> InMemoryRepository[TestInMemoryRepositoryEdgeCases.ComplexEntity, UUID]:
        """Create repository for complex entities."""
        return InMemoryRepository[TestInMemoryRepositoryEdgeCases.ComplexEntity, UUID]()

    @pytest.mark.asyncio
    async def test_save_with_complex_id(
        self,
        repository: InMemoryRepository[TestInMemoryRepositoryEdgeCases.ComplexEntity, UUID],
    ) -> None:
        """Test saving entity with complex ID type."""
        complex_id = ("prefix", 12345)
        entity = self.ComplexEntity(complex_id, {"key": "value"})

        saved = await repository.save(entity)
        assert saved == entity

        retrieved = await repository.get(complex_id)
        assert retrieved == entity

    @pytest.mark.asyncio
    async def test_save_with_none_id(
        self,
        repository: InMemoryRepository[TestInMemoryRepositoryEdgeCases.ComplexEntity, UUID],
    ) -> None:
        """Test saving entity with None ID."""

        class NoneIdEntity:
            def __init__(self) -> None:
                self.id = None

        none_entity = NoneIdEntity()
        none_repo = InMemoryRepository[NoneIdEntity, UUID]()

        saved = await none_repo.save(none_entity)
        assert saved == none_entity

        # Should be able to retrieve with None key
        retrieved = await none_repo.get(None)
        assert retrieved == none_entity

    @pytest.mark.asyncio
    async def test_get_with_different_key_types(
        self,
        repository: InMemoryRepository[TestInMemoryRepositoryEdgeCases.ComplexEntity, UUID],
    ) -> None:
        """Test get method with different key types."""
        # Save entities with different ID types
        entities = [
            self.ComplexEntity(("str", 1), {"type": "string_int"}),
            self.ComplexEntity(("123", 2), {"type": "string_int2"}),
            self.ComplexEntity(("a", 3), {"type": "string_int3"}),
        ]

        for entity in entities:
            await repository.save(entity)

        # Retrieve each entity
        for entity in entities:
            retrieved = await repository.get(entity.id)
            assert retrieved == entity

        # Test with non-existent complex keys
        non_existent_keys = [
            ("not", "found"),
            (999, "missing"),
            ("single",),  # Wrong tuple size
        ]

        for key in non_existent_keys:
            result = await repository.get(key)
            assert result is None

    @pytest.mark.asyncio
    async def test_delete_with_complex_ids(
        self,
        repository: InMemoryRepository[TestInMemoryRepositoryEdgeCases.ComplexEntity, UUID],
    ) -> None:
        """Test delete method with complex ID types."""
        complex_id = ("delete", 1)
        entity = self.ComplexEntity(complex_id, {"status": "to_delete"})

        # Save entity
        await repository.save(entity)

        # Verify it exists
        assert await repository.get(complex_id) is not None

        # Delete it
        deleted = await repository.delete(complex_id)
        assert deleted is True

        # Verify it's gone
        assert await repository.get(complex_id) is None

        # Try to delete again (should return False)
        deleted_again = await repository.delete(complex_id)
        assert deleted_again is False

    @pytest.mark.asyncio
    async def test_storage_internal_dict_behavior(
        self,
        repository: InMemoryRepository[TestInMemoryRepositoryEdgeCases.ComplexEntity, UUID],
    ) -> None:
        """Test internal storage dictionary behavior."""
        # Test that the internal storage behaves correctly with various key types
        entities = []
        for i in range(5):
            entity = self.ComplexEntity((f"key_{i}", i), {"index": i})
            entities.append(entity)
            await repository.save(entity)

        # Verify internal storage state
        assert len(repository._storage) == 5

        # Test list_all returns all values
        all_entities = await repository.list_all()
        assert len(all_entities) == 5

        # Verify count matches storage size
        count = await repository.count()
        assert count == len(repository._storage)

        # Clear and verify
        repository.clear()
        assert len(repository._storage) == 0
        assert await repository.count() == 0


class TestDIPCompliantRepository:
    """Test DIP-compliant repository pattern without concrete type aliases."""

    @pytest.mark.asyncio
    async def test_generic_repository_with_pipeline(self) -> None:
        """Test InMemoryRepository as generic implementation with Pipeline."""
        # DIP compliance - use generic repository, not concrete aliases
        pipeline_repo = InMemoryRepository[Pipeline, str]()

        # Create pipeline entity
        pipeline = Pipeline(
            pipeline_name=PipelineName(value="Test Pipeline"),
            pipeline_description="Test description",
        )

        # Test basic operations with Pipeline entities
        saved = await pipeline_repo.save(pipeline)
        assert saved == pipeline
        assert saved.pipeline_name.value == "Test Pipeline"

        # Retrieve pipeline (use entity.id, not pipeline_id!)
        retrieved = await pipeline_repo.get_by_id(pipeline.id)
        assert retrieved is not None
        assert retrieved.pipeline_name.value == "Test Pipeline"

        # Test count and list with pipelines
        count = await pipeline_repo.count()
        assert count == 1

        all_pipelines = await pipeline_repo.find_all()
        assert len(all_pipelines) == 1
        assert all_pipelines[0].pipeline_name.value == "Test Pipeline"

    @pytest.mark.asyncio
    async def test_repository_multiple_pipelines_dip(self) -> None:
        """Test InMemoryRepository with multiple pipelines (DIP compliant)."""
        pipeline_repo = InMemoryRepository[Pipeline, str]()

        # Create multiple pipelines
        pipelines = []
        for i in range(3):
            pipeline = Pipeline(
                pipeline_name=PipelineName(value=f"Pipeline {i}"),
                pipeline_description=f"Description {i}",
            )
            pipelines.append(pipeline)
            await pipeline_repo.save(pipeline)

        # Verify all pipelines are stored
        count = await pipeline_repo.count()
        assert count == 3

        # Verify each pipeline can be retrieved (use entity.id!)
        for pipeline in pipelines:
            retrieved = await pipeline_repo.get(pipeline.id)
            assert retrieved is not None
            assert retrieved.pipeline_name.value == pipeline.pipeline_name.value

        # Test delete operations (use entity.id!)
        deleted = await pipeline_repo.delete(pipelines[0].id)
        assert deleted is True

        remaining_count = await pipeline_repo.count()
        assert remaining_count == 2


class TestTypeVariableAndGenericCoverage:
    """Test type variable and generic coverage."""

    def test_type_variables_definition(self) -> None:
        """Test type variables are properly defined."""
        from flext_core.infrastructure.memory import ID, T

        # T should be bound to HasId
        assert T.__bound__ == HasId

        # ID should be unbound TypeVar
        assert ID.__bound__ is None

    @pytest.mark.asyncio
    async def test_generic_repository_type_safety(self) -> None:
        """Test generic repository type safety."""

        # Test with different entity types
        class EntityA:
            def __init__(self, entity_id: str, type_name: str = "A") -> None:
                self.id = entity_id
                self.type_name = type_name

        class EntityB:
            def __init__(self, entity_id: str, type_name: str = "B") -> None:
                self.id = entity_id
                self.type_name = type_name

        # Create typed repositories
        repo_a = InMemoryRepository[EntityA, str]()
        repo_b = InMemoryRepository[EntityB, int]()

        # Add entities to each repo
        entity_a = EntityA("a1")
        entity_b = EntityB("b1")

        await repo_a.save(entity_a)
        await repo_b.save(entity_b)

        # Verify type separation
        retrieved_a = await repo_a.get("a1")
        retrieved_b = await repo_b.get("b1")

        assert retrieved_a is not None
        assert retrieved_a.type_name == "A"

        assert retrieved_b is not None
        assert retrieved_b.type_name == "B"

        # Cross-repo access should return None
        assert await repo_a.get("b1") is None
        assert await repo_b.get("a1") is None


class TestModuleExports:
    """Test module exports and __all__ coverage."""

    def test_all_exports(self) -> None:
        """Test that __all__ exports are correct (DIP compliant)."""
        from flext_core.infrastructure.memory import __all__

        expected_exports = [
            "HasId",
            "InMemoryRepository",
            # No concrete aliases exported - DIP compliance
        ]

        assert set(__all__) == set(expected_exports)

    def test_import_all_exports(self) -> None:
        """Test that all exported items can be imported (DIP compliant)."""
        from flext_core.infrastructure.memory import (
            HasId,
            InMemoryRepository,
        )

        # Verify types are available
        assert HasId is not None
        assert InMemoryRepository is not None

        # DIP compliance - no concrete type aliases, use generic repository


class TestAsyncBehaviorEdgeCases:
    """Test async behavior edge cases."""

    @pytest.mark.asyncio
    async def test_concurrent_operations(self) -> None:
        """Test concurrent repository operations."""
        import asyncio

        class ConcurrentEntity:
            def __init__(self, entity_id: str, value: int) -> None:
                self.id = entity_id
                self.value = value

        repo = InMemoryRepository[ConcurrentEntity, UUID]()

        # Create tasks for concurrent operations
        async def save_entity(i: int) -> ConcurrentEntity:
            entity = ConcurrentEntity(f"entity_{i}", i)
            return await repo.save(entity)

        # Run concurrent saves
        tasks = [save_entity(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # Verify all entities were saved
        assert len(results) == 10
        final_count = await repo.count()
        assert final_count == 10

        # Verify all entities can be retrieved
        for i in range(10):
            entity = await repo.get(f"entity_{i}")
            assert entity is not None
            assert entity.value == i

    @pytest.mark.asyncio
    async def test_repository_state_consistency(self) -> None:
        """Test repository maintains state consistency."""

        class StateEntity:
            def __init__(self, entity_id: str, state: str) -> None:
                self.id = entity_id
                self.state = state

        repo = InMemoryRepository[StateEntity, UUID]()

        # Perform series of operations
        entity1 = StateEntity("1", "initial")
        await repo.save(entity1)

        # Update same entity
        entity1_updated = StateEntity("1", "updated")
        await repo.save(entity1_updated)

        # Add different entity
        entity2 = StateEntity("2", "second")
        await repo.save(entity2)

        # Delete first entity
        await repo.delete("1")

        # Verify final state
        assert await repo.get("1") is None
        retrieved_2 = await repo.get("2")
        assert retrieved_2 is not None
        assert retrieved_2.state == "second"

        final_count = await repo.count()
        assert final_count == 1
