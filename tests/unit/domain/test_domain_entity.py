"""Comprehensive tests for FlextEntity domain entity base class.

This module provides complete test coverage for the FlextEntity class,
focusing on missing test cases to achieve 80%+ coverage.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from tests.shared_test_domain import (
    ConcreteFlextEntity,
    TestDomainFactory,
    TestUserStatus,
)

from flext_core import FlextEntity
from flext_core.exceptions import FlextValidationError
from flext_core.result import FlextResult

# Constants
EXPECTED_BULK_SIZE = 2


class TestFlextEntityFieldValidators:
    """Test FlextEntity field validators."""

    def test_validate_entity_id_empty_string(self) -> None:
        """Test entity ID validation with empty string."""
        with pytest.raises(FlextValidationError) as exc_info:
            ConcreteFlextEntity(id="", name="Test", email="test@example.com")

        if "Entity ID cannot be empty" not in str(exc_info.value):
            raise AssertionError(
                f"Expected {'Entity ID cannot be empty'} in {exc_info.value!s}",
            )

    def test_validate_entity_id_whitespace_only(self) -> None:
        """Test entity ID validation with whitespace-only string."""
        with pytest.raises(FlextValidationError) as exc_info:
            ConcreteFlextEntity(id="   ", name="Test", email="test@example.com")

        if "Entity ID cannot be empty" not in str(exc_info.value):
            raise AssertionError(
                f"Expected {'Entity ID cannot be empty'} in {exc_info.value!s}",
            )

    def test_validate_entity_id_valid_with_whitespace(self) -> None:
        """Test entity ID validation strips whitespace from valid ID."""
        entity = ConcreteFlextEntity(
            id="  valid-id  ",
            name="Test",
            email="test@example.com",
        )
        if entity.id != "valid-id":
            raise AssertionError(f"Expected {'valid-id'}, got {entity.id}")

    def test_validate_entity_version_zero(self) -> None:
        """Test entity version validation with zero."""
        with pytest.raises(ValidationError) as exc_info:
            ConcreteFlextEntity(
                id="test-id",
                name="Test",
                email="test@example.com",
                version=0,
            )

        # Pydantic ge=1 validation catches this before custom validator
        if "Input should be greater than or equal to 1" not in str(exc_info.value):
            raise AssertionError(
                f"Expected {'Input should be greater than or equal to 1'} in {exc_info.value!s}",
            )

    def test_validate_entity_version_negative(self) -> None:
        """Test entity version validation with negative number."""
        with pytest.raises(ValidationError) as exc_info:
            ConcreteFlextEntity(
                id="test-id",
                name="Test",
                email="test@example.com",
                version=-1,
            )

        # Pydantic ge=1 validation catches this before custom validator
        if "Input should be greater than or equal to 1" not in str(exc_info.value):
            raise AssertionError(
                f"Expected {'Input should be greater than or equal to 1'} in {exc_info.value!s}",
            )

    def test_validate_entity_version_valid(self) -> None:
        """Test entity version validation with valid positive number."""
        entity = ConcreteFlextEntity(
            id="test-id",
            name="Test",
            email="test@example.com",
            version=5,
        )
        if entity.version != 5:
            raise AssertionError(f"Expected {5}, got {entity.version}")


class TestFlextEntityEquality:
    """Test FlextEntity equality behavior."""

    def test_equality_with_non_entity(self) -> None:
        """Test entity equality with non-FlextEntity object."""
        entity = ConcreteFlextEntity(
            id="test-id",
            name="Test",
            email="test@example.com",
        )

        assert entity != "not an entity"
        assert entity != 123
        assert entity != {"id": entity.id}
        assert entity is not None

    def test_equality_with_different_entity_types(self) -> None:
        """Test entity equality with different entity types but same ID."""

        class AnotherEntity(FlextEntity):
            title: str

            def validate_domain_rules(self) -> FlextResult[None]:
                return FlextResult.ok(None)

        entity_id = "test-entity-123"
        entity1 = ConcreteFlextEntity(
            id=entity_id,
            name="Test",
            email="test@example.com",
        )
        entity2 = AnotherEntity(id=entity_id, title="Test")

        # Should be equal because they have the same ID
        if entity1 != entity2:
            raise AssertionError(f"Expected {entity2}, got {entity1}")


class TestFlextEntityVersioning:
    """Test FlextEntity version management."""

    def test_with_version_valid_increment(self) -> None:
        """Test creating entity with incremented version."""
        entity_result = TestDomainFactory.create_concrete_entity(
            name="Test",
            id="test-id",
            version=1,
        )
        assert entity_result.success
        entity = entity_result.data
        assert entity is not None
        updated_entity = entity.with_version(2)

        if updated_entity.version != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {updated_entity.version}")
        assert updated_entity.id == entity.id
        if updated_entity.name != entity.name:
            raise AssertionError(f"Expected {entity.name}, got {updated_entity.name}")
        # Same ID = same entity (DDD principle)
        if updated_entity != entity:
            raise AssertionError(f"Expected {entity}, got {updated_entity}")

    def test_with_version_large_increment(self) -> None:
        """Test creating entity with large version increment."""
        entity_result = TestDomainFactory.create_concrete_entity(
            name="Test",
            id="test-id",
            version=1,
        )
        assert entity_result.success
        entity = entity_result.data
        assert entity is not None
        updated_entity = entity.with_version(100)

        if updated_entity.version != 100:
            raise AssertionError(f"Expected {100}, got {updated_entity.version}")
        assert updated_entity.id == entity.id

    def test_with_version_same_version(self) -> None:
        """Test with_version with same version number."""
        entity = ConcreteFlextEntity(
            id="test-id",
            name="Test",
            email="test@example.com",
            version=5,
        )

        with pytest.raises(
            FlextValidationError,
            match="New version must be greater than current version",
        ):
            entity.with_version(5)

    def test_with_version_lower_version(self) -> None:
        """Test with_version with lower version number."""
        entity = ConcreteFlextEntity(
            id="test-id",
            name="Test",
            email="test@example.com",
            version=5,
        )

        with pytest.raises(
            FlextValidationError,
            match="New version must be greater than current version",
        ):
            entity.with_version(3)

    def test_with_version_preserves_all_fields(self) -> None:
        """Test with_version preserves all entity fields."""
        entity = ConcreteFlextEntity(
            id="custom-id",
            name="Test Entity",
            email="test@example.com",
            status=TestUserStatus.INACTIVE,
            version=1,
        )

        updated_entity = entity.with_version(2)

        if updated_entity.id != entity.id:
            raise AssertionError(f"Expected {entity.id}, got {updated_entity.id}")
        assert updated_entity.name == entity.name
        if updated_entity.status != entity.status:
            raise AssertionError(
                f"Expected {entity.status}, got {updated_entity.status}",
            )
        assert updated_entity.created_at == entity.created_at
        if updated_entity.version != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {updated_entity.version}")

    def test_with_version_maintains_type(self) -> None:
        """Test with_version returns same entity type."""
        entity = ConcreteFlextEntity(
            id="test-id",
            name="Test",
            email="test@example.com",
            version=1,
        )
        updated_entity = entity.with_version(2)

        assert type(updated_entity) is type(entity)
        assert isinstance(updated_entity, ConcreteFlextEntity)


class TestFlextEntityEdgeCases:
    """Test FlextEntity edge cases and error conditions."""

    def test_entity_id_type_preservation(self) -> None:
        """Test that entity ID is always string type."""
        entity = ConcreteFlextEntity(id="123", name="Test", email="test@example.com")
        assert isinstance(entity.id, str)
        if entity.id != "123":
            raise AssertionError(f"Expected {'123'}, got {entity.id}")

    def test_entity_creation_with_all_optional_fields(self) -> None:
        """Test entity creation with all optional fields specified."""
        custom_time = datetime.now(UTC)
        entity = ConcreteFlextEntity(
            id="custom-id",
            name="Test",
            email="test@example.com",
            status=TestUserStatus.ACTIVE,
            created_at=custom_time,
            version=10,
        )

        if entity.id != "custom-id":
            raise AssertionError(f"Expected {'custom-id'}, got {entity.id}")
        assert entity.name == "Test"
        if entity.status != "active":  # Updated expectation since we changed the value
            raise AssertionError(f"Expected {'active'}, got {entity.status}")
        assert entity.created_at == custom_time
        if entity.version != 10:
            raise AssertionError(f"Expected {10}, got {entity.version}")

    def test_entity_hash_stability(self) -> None:
        """Test that entity hash is stable across different instances."""
        entity_id = "stable-id"
        entity1 = ConcreteFlextEntity(
            id=entity_id,
            name="Name1",
            email="test1@example.com",
        )
        entity2 = ConcreteFlextEntity(
            id=entity_id,
            name="Name2",
            email="test2@example.com",
        )

        hash1 = hash(entity1)
        hash2 = hash(entity2)

        if hash1 != hash2:
            raise AssertionError(f"Expected {hash2}, got {hash1}")

        # Hash should be consistent across multiple calls
        if hash(entity1) != hash1:
            raise AssertionError(f"Expected {hash1}, got {hash(entity1)}")

    def test_entity_string_representations(self) -> None:
        """Test entity string representations."""
        entity = ConcreteFlextEntity(
            id="test-123",
            name="Test Entity",
            email="test@example.com",
            status="active",
        )

        str_repr = str(entity)
        repr_str = repr(entity)

        # __str__ should contain class name and ID
        # ConcreteFlextEntity is an alias for TestUser, so check for TestUser
        if "TestUser" not in str_repr:
            raise AssertionError(f"Expected {'TestUser'} in {str_repr}")
        assert "test-123" in str_repr

        # __repr__ should contain all fields
        if "TestUser" not in repr_str:
            raise AssertionError(f"Expected {'TestUser'} in {repr_str}")
        assert "test-123" in repr_str
        if "Test Entity" not in repr_str:
            raise AssertionError(f"Expected {'Test Entity'} in {repr_str}")
        assert "active" in repr_str

    def test_entity_model_dump_excludes_private_fields(self) -> None:
        """Test that model_dump works correctly."""
        entity = ConcreteFlextEntity(
            id="test-id",
            name="Test",
            email="test@example.com",
            status="active",
        )
        data = entity.model_dump()

        assert isinstance(data, dict)
        if "id" not in data:
            raise AssertionError(f"Expected {'id'} in {data}")
        assert "name" in data
        if "status" not in data:
            raise AssertionError(f"Expected {'status'} in {data}")
        assert "created_at" in data
        if "version" not in data:
            raise AssertionError(f"Expected {'version'} in {data}")

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
        entity = ConcreteFlextEntity(
            id="test-id",
            name="Test",
            email="test@example.com",
        )

        # Should not raise any exception for valid entity
        result = entity.validate_domain_rules()
        assert result.success

        # Verify that invalid entities would return failure
        entity_invalid = ConcreteFlextEntity(
            id="test-id-2",
            name="",
            email="test@example.com",
        )
        result_invalid = entity_invalid.validate_domain_rules()
        assert result_invalid.is_failure
        if (
            result_invalid.error is None
            or "Entity name cannot be empty" not in result_invalid.error
        ):
            raise AssertionError(
                f"Expected 'Entity name cannot be empty' in {result_invalid.error}",
            )

    def test_pydantic_field_validation_integration(self) -> None:
        """Test integration with Pydantic field validation."""
        # Test that Pydantic validation works as expected
        with pytest.raises(ValidationError):
            ConcreteFlextEntity(id="test-id", name="", email="test@example.com")

        # Test successful creation with valid data
        entity = ConcreteFlextEntity(
            id="test-id",
            name="Valid Name",
            email="test@example.com",
        )
        if entity.name != "Valid Name":
            raise AssertionError(f"Expected {'Valid Name'}, got {entity.name}")
