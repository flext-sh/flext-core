"""Comprehensive tests for FlextEntity domain entity base class.

This module provides complete test coverage for the FlextEntity class,
focusing on missing test cases to achieve 80%+ coverage.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime

import pytest
from pydantic import ValidationError

from flext_core.domain.entity import FlextEntity


class ConcreteFlextEntity(FlextEntity):
    """Concrete entity implementation for comprehensive testing."""

    name: str
    status: str = "active"

    def validate_domain_rules(self) -> None:
        """Validate test entity domain rules."""
        if not self.name.strip():
            msg = "Entity name cannot be empty"
            raise ValueError(msg)


class TestFlextEntityFieldValidators:
    """Test FlextEntity field validators."""

    def test_validate_entity_id_empty_string(self) -> None:
        """Test entity ID validation with empty string."""
        with pytest.raises(ValidationError) as exc_info:
            ConcreteFlextEntity(id="", name="Test")

        assert "Entity ID cannot be empty or whitespace-only" in str(
            exc_info.value,
        )

    def test_validate_entity_id_whitespace_only(self) -> None:
        """Test entity ID validation with whitespace-only string."""
        with pytest.raises(ValidationError) as exc_info:
            ConcreteFlextEntity(id="   ", name="Test")

        assert "Entity ID cannot be empty or whitespace-only" in str(
            exc_info.value,
        )

    def test_validate_entity_id_valid_with_whitespace(self) -> None:
        """Test entity ID validation strips whitespace from valid ID."""
        entity = ConcreteFlextEntity(id="  valid-id  ", name="Test")
        assert entity.id == "valid-id"

    def test_validate_entity_version_zero(self) -> None:
        """Test entity version validation with zero."""
        with pytest.raises(ValidationError) as exc_info:
            ConcreteFlextEntity(name="Test", version=0)

        # Pydantic ge=1 validation catches this before custom validator
        assert "Input should be greater than or equal to 1" in str(
            exc_info.value,
        )

    def test_validate_entity_version_negative(self) -> None:
        """Test entity version validation with negative number."""
        with pytest.raises(ValidationError) as exc_info:
            ConcreteFlextEntity(name="Test", version=-1)

        # Pydantic ge=1 validation catches this before custom validator
        assert "Input should be greater than or equal to 1" in str(
            exc_info.value,
        )

    def test_validate_entity_version_valid(self) -> None:
        """Test entity version validation with valid positive number."""
        entity = ConcreteFlextEntity(name="Test", version=5)
        assert entity.version == 5


class TestFlextEntityEquality:
    """Test FlextEntity equality behavior."""

    def test_equality_with_non_entity(self) -> None:
        """Test entity equality with non-FlextEntity object."""
        entity = ConcreteFlextEntity(name="Test")

        assert entity != "not an entity"
        assert entity != 123
        assert entity != {"id": entity.id}
        assert entity is not None

    def test_equality_with_different_entity_types(self) -> None:
        """Test entity equality with different entity types but same ID."""

        class AnotherEntity(FlextEntity):
            title: str

            def validate_domain_rules(self) -> None:
                pass

        entity_id = "test-entity-123"
        entity1 = ConcreteFlextEntity(id=entity_id, name="Test")
        entity2 = AnotherEntity(id=entity_id, title="Test")

        # Should be equal because they have the same ID
        assert entity1 == entity2


class TestFlextEntityVersioning:
    """Test FlextEntity version management."""

    def test_with_version_valid_increment(self) -> None:
        """Test creating entity with incremented version."""
        entity = ConcreteFlextEntity(name="Test", version=1)
        updated_entity = entity.with_version(2)

        assert updated_entity.version == 2
        assert updated_entity.id == entity.id
        assert updated_entity.name == entity.name
        # Same ID = same entity (DDD principle)
        assert updated_entity == entity

    def test_with_version_large_increment(self) -> None:
        """Test creating entity with large version increment."""
        entity = ConcreteFlextEntity(name="Test", version=1)
        updated_entity = entity.with_version(100)

        assert updated_entity.version == 100
        assert updated_entity.id == entity.id

    def test_with_version_same_version(self) -> None:
        """Test with_version with same version number."""
        entity = ConcreteFlextEntity(name="Test", version=5)

        with pytest.raises(
            ValueError,
            match="New version must be greater than current version",
        ):
            entity.with_version(5)

    def test_with_version_lower_version(self) -> None:
        """Test with_version with lower version number."""
        entity = ConcreteFlextEntity(name="Test", version=5)

        with pytest.raises(
            ValueError,
            match="New version must be greater than current version",
        ):
            entity.with_version(3)

    def test_with_version_preserves_all_fields(self) -> None:
        """Test with_version preserves all entity fields."""
        entity = ConcreteFlextEntity(
            id="custom-id",
            name="Test Entity",
            status="inactive",
            version=1,
        )

        updated_entity = entity.with_version(2)

        assert updated_entity.id == entity.id
        assert updated_entity.name == entity.name
        assert updated_entity.status == entity.status
        assert updated_entity.created_at == entity.created_at
        assert updated_entity.version == 2

    def test_with_version_maintains_type(self) -> None:
        """Test with_version returns same entity type."""
        entity = ConcreteFlextEntity(name="Test", version=1)
        updated_entity = entity.with_version(2)

        assert type(updated_entity) is type(entity)
        assert isinstance(updated_entity, ConcreteFlextEntity)


class TestFlextEntityEdgeCases:
    """Test FlextEntity edge cases and error conditions."""

    def test_entity_id_type_preservation(self) -> None:
        """Test that entity ID is always string type."""
        entity = ConcreteFlextEntity(id="123", name="Test")
        assert isinstance(entity.id, str)
        assert entity.id == "123"

    def test_entity_creation_with_all_optional_fields(self) -> None:
        """Test entity creation with all optional fields specified."""
        custom_time = datetime.now(UTC)
        entity = ConcreteFlextEntity(
            id="custom-id",
            name="Test",
            status="custom-status",
            created_at=custom_time,
            version=10,
        )

        assert entity.id == "custom-id"
        assert entity.name == "Test"
        assert entity.status == "custom-status"
        assert entity.created_at == custom_time
        assert entity.version == 10

    def test_entity_hash_stability(self) -> None:
        """Test that entity hash is stable across different instances."""
        entity_id = "stable-id"
        entity1 = ConcreteFlextEntity(id=entity_id, name="Name1")
        entity2 = ConcreteFlextEntity(id=entity_id, name="Name2")

        hash1 = hash(entity1)
        hash2 = hash(entity2)

        assert hash1 == hash2

        # Hash should be consistent across multiple calls
        assert hash(entity1) == hash1

    def test_entity_string_representations(self) -> None:
        """Test entity string representations."""
        entity = ConcreteFlextEntity(
            id="test-123",
            name="Test Entity",
            status="active",
        )

        str_repr = str(entity)
        repr_str = repr(entity)

        # __str__ should contain class name and ID
        assert "ConcreteFlextEntity" in str_repr
        assert "test-123" in str_repr

        # __repr__ should contain all fields
        assert "ConcreteFlextEntity" in repr_str
        assert "test-123" in repr_str
        assert "Test Entity" in repr_str
        assert "active" in repr_str

    def test_entity_model_dump_excludes_private_fields(self) -> None:
        """Test that model_dump works correctly."""
        entity = ConcreteFlextEntity(name="Test", status="active")
        data = entity.model_dump()

        assert isinstance(data, dict)
        assert "id" in data
        assert "name" in data
        assert "status" in data
        assert "created_at" in data
        assert "version" in data

        # Verify data types
        assert isinstance(data["id"], str)
        assert isinstance(data["name"], str)
        assert isinstance(data["status"], str)
        assert isinstance(data["version"], int)


class TestFlextEntityValidation:
    """Test FlextEntity validation integration."""

    def test_domain_rules_validation_called(self) -> None:
        """Test that domain rules validation is properly integrated."""
        # This test verifies the domain rules method exists and can be called
        entity = ConcreteFlextEntity(name="Test")

        # Should not raise any exception for valid entity
        entity.validate_domain_rules()

        # Verify that invalid entities would raise exceptions
        entity_invalid = ConcreteFlextEntity(name="")
        with pytest.raises(ValueError, match="Entity name cannot be empty"):
            entity_invalid.validate_domain_rules()

    def test_pydantic_field_validation_integration(self) -> None:
        """Test integration with Pydantic field validation."""
        # Test that Pydantic validation works as expected
        with pytest.raises(ValidationError):
            ConcreteFlextEntity()  # Missing required 'name' field

        # Test successful creation with valid data
        entity = ConcreteFlextEntity(name="Valid Name")
        assert entity.name == "Valid Name"
