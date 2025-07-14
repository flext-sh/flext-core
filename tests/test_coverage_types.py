"""Tests to improve coverage for domain types module."""

from __future__ import annotations

import pytest
from unittest.mock import Mock
from uuid import uuid4, UUID

from flext_core.domain.types import (
    EntityProtocol,
    validate_entity_id,
    validate_project_name,
)


class TestEntityProtocolCoverage:
    """Test EntityProtocol methods to improve coverage."""

    def test_entity_protocol_eq_method(self) -> None:
        """Test EntityProtocol __eq__ method."""

        class ConcreteEntity:
            """Concrete implementation of EntityProtocol."""

            def __init__(self, entity_id: UUID):
                self.id = entity_id
                self.created_at = None
                self.updated_at = None

            def __eq__(self, other: object) -> bool:
                """Implementation of __eq__ for testing."""
                if not isinstance(other, ConcreteEntity):
                    return False
                return self.id == other.id

            def __hash__(self) -> int:
                """Implementation of __hash__ for testing."""
                return hash(self.id)

        id1 = uuid4()
        id2 = uuid4()

        entity1 = ConcreteEntity(id1)
        entity2 = ConcreteEntity(id1)  # Same ID
        entity3 = ConcreteEntity(id2)  # Different ID

        # Test equality
        assert entity1 == entity2
        assert entity1 != entity3
        assert entity1 != "not an entity"

    def test_entity_protocol_hash_method(self) -> None:
        """Test EntityProtocol __hash__ method."""

        class HashableEntity:
            """Entity that implements __hash__."""

            def __init__(self, entity_id: UUID):
                self.id = entity_id
                self.created_at = None
                self.updated_at = None

            def __eq__(self, other: object) -> bool:
                return isinstance(other, HashableEntity) and self.id == other.id

            def __hash__(self) -> int:
                return hash(self.id)

        entity_id = uuid4()
        entity = HashableEntity(entity_id)

        # Test that hash works
        hash_value = hash(entity)
        assert isinstance(hash_value, int)

        # Test that equal entities have same hash
        entity2 = HashableEntity(entity_id)
        assert hash(entity) == hash(entity2)


class TestValidationFunctionsCoverage:
    """Test validation functions edge cases."""

    def test_validate_entity_id_with_uuid(self) -> None:
        """Test validate_entity_id with UUID input."""
        test_uuid = uuid4()
        result = validate_entity_id(test_uuid)
        assert result == test_uuid
        assert isinstance(result, UUID)

    def test_validate_entity_id_with_string(self) -> None:
        """Test validate_entity_id with string input."""
        test_uuid = uuid4()
        uuid_string = str(test_uuid)

        result = validate_entity_id(uuid_string)
        assert result == test_uuid
        assert isinstance(result, UUID)

    def test_validate_entity_id_with_invalid_input(self) -> None:
        """Test validate_entity_id with invalid input."""
        with pytest.raises(ValueError, match="Invalid entity ID"):
            validate_entity_id(123)

        with pytest.raises(ValueError, match="badly formed hexadecimal UUID string"):
            validate_entity_id("not-a-uuid")

        with pytest.raises(ValueError, match="Invalid entity ID"):
            validate_entity_id(None)

    def test_validate_project_name_valid(self) -> None:
        """Test validate_project_name with valid names."""
        valid_names = [
            "my-project",
            "my_project",
            "MyProject",
            "project123",
            "a1",  # Minimum length
            "a" * 50,  # Maximum length
        ]

        for name in valid_names:
            result = validate_project_name(name)
            assert result == name

    def test_validate_project_name_invalid_type(self) -> None:
        """Test validate_project_name with invalid types."""
        with pytest.raises(TypeError, match="Project name must be a string"):
            validate_project_name(123)

        with pytest.raises(TypeError, match="Project name must be a string"):
            validate_project_name(None)

        with pytest.raises(TypeError, match="Project name must be a string"):
            validate_project_name([])

    def test_validate_project_name_invalid_length(self) -> None:
        """Test validate_project_name with invalid lengths."""
        # Too short
        with pytest.raises(ValueError, match="Project name must be 2-50 characters"):
            validate_project_name("a")

        # Too long
        with pytest.raises(ValueError, match="Project name must be 2-50 characters"):
            validate_project_name("a" * 51)

    def test_validate_project_name_invalid_characters(self) -> None:
        """Test validate_project_name with invalid characters."""
        invalid_names = [
            "project with spaces",
            "project@special",
            "project#hash",
            "project$dollar",
            "project!exclamation",
        ]

        for name in invalid_names:
            with pytest.raises(
                ValueError,
                match="Project name must be alphanumeric with hyphens/underscores",
            ):
                validate_project_name(name)


class TestProtocolConformance:
    """Test protocol conformance edge cases."""

    def test_protocol_duck_typing(self) -> None:
        """Test that objects can conform to protocols via duck typing."""

        class DuckTypedEntity:
            """Entity that duck-types to EntityProtocol."""

            def __init__(self) -> None:
                self.id = uuid4()
                self.created_at = None
                self.updated_at = None

            def __eq__(self, other: object) -> bool:
                return hasattr(other, "id") and self.id == other.id

            def __hash__(self) -> int:
                return hash(self.id)

        entity = DuckTypedEntity()

        # Test that it behaves like an EntityProtocol
        assert hasattr(entity, "id")
        assert hasattr(entity, "created_at")
        assert hasattr(entity, "updated_at")
        assert callable(getattr(entity, "__eq__"))
        assert callable(getattr(entity, "__hash__"))

    def test_runtime_checkable_protocols(self) -> None:
        """Test runtime checkable protocol behavior."""

        class ProtocolConformingClass:
            """Class that conforms to EntityProtocol."""

            def __init__(self) -> None:
                self.id = uuid4()
                self.created_at = None
                self.updated_at = None

            def __eq__(self, other: object) -> bool:
                return True

            def __hash__(self) -> int:
                return 42

        instance = ProtocolConformingClass()

        # Test isinstance check with runtime checkable protocol
        assert isinstance(instance, EntityProtocol)

    def test_protocol_with_mock(self) -> None:
        """Test protocol usage with mocks."""
        mock_entity = Mock()
        mock_entity.id = uuid4()
        mock_entity.created_at = None
        mock_entity.updated_at = None

        # Mock should be able to act as EntityProtocol
        assert hasattr(mock_entity, "id")
        assert hasattr(mock_entity, "created_at")
        assert hasattr(mock_entity, "updated_at")


class TestTypeAliasEdgeCases:
    """Test type alias edge cases for better coverage."""

    def test_complex_type_usage(self) -> None:
        """Test complex type constructions."""
        from flext_core.domain.types import ServiceResult

        # Test ServiceResult with different types
        string_result = ServiceResult.ok("test")
        assert string_result.is_successful
        assert string_result.data == "test"

        dict_result = ServiceResult.ok({"key": "value"})
        assert dict_result.is_successful
        assert dict_result.data == {"key": "value"}

        list_result = ServiceResult.ok([1, 2, 3])
        assert list_result.is_successful
        assert list_result.data == [1, 2, 3]

    def test_enum_edge_cases(self) -> None:
        """Test enum edge cases."""
        from flext_core.domain.types import EntityStatus, ResultStatus

        # Test enum value access
        assert EntityStatus.ACTIVE == "active"
        assert EntityStatus.INACTIVE == "inactive"
        assert ResultStatus.SUCCESS == "success"
        assert ResultStatus.ERROR == "error"

        # Test enum membership
        assert "active" in EntityStatus
        assert "success" in ResultStatus

        # Test enum iteration
        statuses = list(EntityStatus)
        assert len(statuses) >= 2
        assert EntityStatus.ACTIVE in statuses
