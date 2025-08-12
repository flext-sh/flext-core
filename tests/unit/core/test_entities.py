"""Comprehensive tests for entities.py module.

This test suite provides complete coverage of the entity system,
testing all aspects including entity creation, validation, versioning,
domain events, and factory patterns to achieve near 100% coverage.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, cast
from unittest.mock import MagicMock, patch

import pytest

from flext_core.entities import FlextEntity, FlextEntityFactory
from flext_core.exceptions import FlextValidationError
from flext_core.payload import FlextEvent
from flext_core.result import FlextResult


class EntityFactory(Protocol):
    """Protocol for entity factory functions."""

    def __call__(self, **kwargs: object) -> FlextResult[object]:
        """Create entity instance."""
        ...


pytestmark = [pytest.mark.unit, pytest.mark.core]

# Constants
EXPECTED_BULK_SIZE = 2
EXPECTED_DATA_COUNT = 3


# Create a concrete test entity for testing
class SampleUser(FlextEntity):
    """Test user entity for comprehensive testing."""

    name: str
    email: str
    age: int = 0
    is_active: bool = True

    def get_display_name(self) -> str:
        """Get user display name."""
        return f"{self.name} ({self.email})"

    def is_adult(self) -> bool:
        """Check if user is adult."""
        return self.age >= 18

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for user entity."""
        if self.age < 0:
            return FlextResult.fail("Age cannot be negative")
        if not self.email or "@" not in self.email:
            return FlextResult.fail("Invalid email format")
        return FlextResult.ok(None)


class SampleBadUser(FlextEntity):
    """Test user entity with validation issues."""

    name: str
    email: str
    age: int = -1  # Invalid default age

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for bad user entity."""
        # This entity intentionally has validation issues
        return FlextResult.fail("Always fails")


# Test models work without explicit model_rebuild
# as Pydantic handles forward references automatically


@pytest.mark.unit
class TestFlextEntity:
    """Test FlextEntity functionality."""

    def test_entity_creation_valid(self) -> None:
        """Test creating a valid entity."""
        user = SampleUser(
            id="user_1",
            name="John Doe",
            email="john@example.com",
            age=25,
        )

        if user.id != "user_1":
            id_msg: str = f"Expected {'user_1'}, got {user.id}"
            raise AssertionError(id_msg)
        assert user.name == "John Doe"
        if user.email != "john@example.com":
            email_msg: str = f"Expected {'john@example.com'}, got {user.email}"
            raise AssertionError(email_msg)
        assert user.age == 25
        if user.version != 1:
            version_msg: str = f"Expected {1}, got {user.version}"
            raise AssertionError(version_msg)
        if user.is_active is not True:
            active_msg: str = f"Expected True, got {user.is_active is True}"
            raise AssertionError(active_msg)
        assert isinstance(user.created_at, datetime)
        if len(user.domain_events) != 0:
            events_msg: str = f"Expected {0}, got {len(user.domain_events)}"
            raise AssertionError(events_msg)

    def test_entity_id_validation_empty(self) -> None:
        """Test entity ID validation with empty ID."""
        with pytest.raises(FlextValidationError) as exc_info:
            SampleUser(
                id="",
                name="John Doe",
                email="john@example.com",
            )
        if "Entity ID cannot be empty" not in str(exc_info.value):
            msg: str = f"Expected {'Entity ID cannot be empty'} in {exc_info.value!s}"
            raise AssertionError(msg)

    def test_entity_id_validation_whitespace(self) -> None:
        """Test entity ID validation with whitespace ID."""
        with pytest.raises(FlextValidationError) as exc_info:
            SampleUser(
                id="   ",
                name="John Doe",
                email="john@example.com",
            )
        if "Entity ID cannot be empty" not in str(exc_info.value):
            msg: str = f"Expected {'Entity ID cannot be empty'} in {exc_info.value!s}"
            raise AssertionError(msg)

    def test_entity_id_validation_strips_whitespace(self) -> None:
        """Test entity ID validation strips whitespace."""
        user = SampleUser(
            id="  user_1  ",
            name="John Doe",
            email="john@example.com",
        )
        if user.id != "user_1":
            msg: str = f"Expected {'user_1'}, got {user.id}"
            raise AssertionError(msg)

    def test_entity_equality_same_id(self) -> None:
        """Test entity equality based on ID."""
        user1 = SampleUser(id="user_1", name="John", email="john@example.com")
        user2 = SampleUser(id="user_1", name="Jane", email="jane@example.com")

        if user1 != user2:  # Same ID, different attributes:
            msg: str = f"Expected {user2}, got {user1}"
            raise AssertionError(msg)

    def test_entity_equality_different_id(self) -> None:
        """Test entity inequality with different IDs."""
        user1 = SampleUser(id="user_1", name="John", email="john@example.com")
        user2 = SampleUser(id="user_2", name="John", email="john@example.com")

        assert user1 != user2

    def test_entity_equality_with_non_entity(self) -> None:
        """Test entity equality with non-entity object."""
        user = SampleUser(id="user_1", name="John", email="john@example.com")

        assert user != "not an entity"
        assert user != 42
        assert user != {"id": "user_1"}

    def test_entity_hash_same_id(self) -> None:
        """Test entity hash based on ID."""
        user1 = SampleUser(id="user_1", name="John", email="john@example.com")
        user2 = SampleUser(id="user_1", name="Jane", email="jane@example.com")

        if hash(user1) != hash(user2):
            msg: str = f"Expected {hash(user2)}, got {hash(user1)}"
            raise AssertionError(msg)

    def test_entity_hash_different_id(self) -> None:
        """Test entity hash with different IDs."""
        user1 = SampleUser(id="user_1", name="John", email="john@example.com")
        user2 = SampleUser(id="user_2", name="John", email="john@example.com")

        assert hash(user1) != hash(user2)

    def test_entity_str_representation(self) -> None:
        """Test entity string representation."""
        user = SampleUser(id="user_1", name="John", email="john@example.com")

        if str(user) != "SampleUser(id=user_1)":
            msg: str = f"Expected {'SampleUser(id=user_1)'}, got {user!s}"
            raise AssertionError(msg)

    def test_entity_repr_representation(self) -> None:
        """Test entity detailed representation."""
        user = SampleUser(id="user_1", name="John", email="john@example.com", age=25)

        repr_str = repr(user)
        if "SampleUser(" not in repr_str:
            repr_msg: str = f"Expected {'SampleUser('} in {repr_str}"
            raise AssertionError(repr_msg)
        assert "id=user_1" in repr_str
        if "name=John" not in repr_str:
            name_repr_msg: str = f"Expected {'name=John'} in {repr_str}"
            raise AssertionError(name_repr_msg)
        assert "email=john@example.com" in repr_str
        if "age=25" not in repr_str:
            raise AssertionError(f"Expected {'age=25'} in {repr_str}")
        assert "domain_events" not in repr_str  # Should be excluded

    def test_increment_version_success(self) -> None:
        """Test successful version increment."""
        user = SampleUser(id="user_1", name="John", email="john@example.com")

        result = user.increment_version()

        assert result.success
        new_user = cast("SampleUser", result.data)
        assert new_user is not None
        if new_user.version != user.version + 1:
            raise AssertionError(f"Expected {user.version + 1}, got {new_user.version}")
        assert new_user.id == user.id
        if new_user.name != user.name:
            raise AssertionError(f"Expected {user.name}, got {new_user.name}")

    def test_increment_version_domain_validation_failure(self) -> None:
        """Test version increment with domain validation failure."""
        user = SampleBadUser(id="user_1", name="John", email="john@example.com")

        result = user.increment_version()

        assert result.is_failure
        if "Always fails" not in (result.error or ""):
            raise AssertionError(f"Expected 'Always fails' in {result.error}")

    def test_increment_version_construction_error(self) -> None:
        """Test version increment with construction error."""
        user = SampleUser(id="user_1", name="John", email="john@example.com")

        # Mock the class constructor to cause TypeError during increment
        with patch.object(SampleUser, "__init__", side_effect=TypeError("Mock error")):
            result = user.increment_version()

        assert result.is_failure
        if "Failed to increment version" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Failed to increment version' in {result.error}"
            )

    def test_copy_with_success(self) -> None:
        """Test successful copy with changes."""
        user = SampleUser(id="user_1", name="John", email="john@example.com", age=25)

        result = user.copy_with(name="Jane", age=30)

        assert result.success
        new_user = cast("SampleUser", result.data)
        assert new_user is not None
        if new_user.name != "Jane":
            raise AssertionError(f"Expected {'Jane'}, got {new_user.name}")
        assert new_user.age == 30
        if new_user.version != user.version + 1:  # Auto-incremented:
            version_copy_msg: str = (
                f"Expected {user.version + 1}, got {new_user.version}"
            )
            raise AssertionError(version_copy_msg)
        assert new_user.id == user.id  # Unchanged
        if new_user.email != user.email:  # Unchanged:
            email_copy_msg: str = f"Expected {user.email}, got {new_user.email}"
            raise AssertionError(email_copy_msg)

    def test_copy_with_explicit_version(self) -> None:
        """Test copy with explicit version provided."""
        user = SampleUser(id="user_1", name="John", email="john@example.com", version=5)

        result = user.copy_with(name="Jane", version=10)

        assert result.success
        new_user = cast("SampleUser", result.data)
        assert new_user is not None
        if new_user.version != 10:  # Explicit version used:
            msg: str = f"Expected {10}, got {new_user.version}"
            raise AssertionError(msg)
        assert new_user.name == "Jane"

    def test_copy_with_no_changes(self) -> None:
        """Test copy with no changes - version should not increment."""
        user = SampleUser(id="user_1", name="John", email="john@example.com", version=5)

        result = user.copy_with()

        assert result.success
        new_user = cast("SampleUser", result.data)
        assert new_user is not None
        if new_user.version != 5:  # Version not incremented when no changes:
            msg: str = f"Expected {5}, got {new_user.version}"
            raise AssertionError(msg)

    def test_copy_with_domain_validation_failure(self) -> None:
        """Test copy with domain validation failure."""
        user = SampleUser(id="user_1", name="John", email="john@example.com")

        result = user.copy_with(email="invalid_email")  # No @ symbol

        assert result.is_failure
        if "Invalid email format" not in (result.error or ""):
            msg: str = f"Expected 'Invalid email format' in {result.error}"
            raise AssertionError(msg)

    def test_copy_with_construction_error(self) -> None:
        """Test copy with construction error."""
        user = SampleUser(id="user_1", name="John", email="john@example.com")

        # Mock the class constructor to cause ValueError during copy
        with patch.object(SampleUser, "__init__", side_effect=ValueError("Mock error")):
            result = user.copy_with(name="Jane")

        assert result.is_failure
        if "Failed to copy entity" not in (result.error or ""):
            msg: str = f"Expected 'Failed to copy entity' in {result.error}"
            raise AssertionError(msg)

    def test_add_domain_event_success(self) -> None:
        """Test successful domain event addition."""
        user = SampleUser(id="user_1", name="John", email="john@example.com")

        result = user.add_domain_event("user_created", {"action": "create"})

        assert result.success
        if len(user.domain_events) != 1:
            raise AssertionError(f"Expected {1}, got {len(user.domain_events)}")
        event = user.domain_events[0]
        if event.event_type != "user_created":
            event_type_msg: str = f"Expected {'user_created'}, got {event.event_type}"
            raise AssertionError(event_type_msg)
        assert event.data == {"action": "create"}
        if event.aggregate_id != "user_1":
            aggregate_id_msg: str = f"Expected {'user_1'}, got {event.aggregate_id}"
            raise AssertionError(aggregate_id_msg)
        assert event.version == 1

    def test_add_domain_event_creation_failure(self) -> None:
        """Test domain event addition with event creation failure."""
        user = SampleUser(id="user_1", name="John", email="john@example.com")

        # Mock FlextEvent.create_event to fail
        with patch.object(
            FlextEvent,
            "create_event",
            return_value=FlextResult.fail("Event creation failed"),
        ):
            result = user.add_domain_event("user_created", {"action": "create"})

        assert result.is_failure
        if "Failed to create event" not in (result.error or ""):
            msg: str = f"Expected 'Failed to create event' in {result.error}"
            raise AssertionError(msg)
        if len(user.domain_events) != 0:
            raise AssertionError(f"Expected {0}, got {len(user.domain_events)}")

    def test_clear_events(self) -> None:
        """Test clearing domain events."""
        user = SampleUser(id="user_1", name="John", email="john@example.com")

        # Add some events
        user.add_domain_event("event1", {"data": "test1"})
        user.add_domain_event("event2", {"data": "test2"})

        if len(user.domain_events) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(user.domain_events)}")

        events = user.clear_events()

        if len(events) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(events)}")
        assert len(user.domain_events) == 0
        if events[0].event_type != "event1":
            msg: str = f"Expected {'event1'}, got {events[0].event_type}"
            raise AssertionError(msg)
        assert events[1].event_type == "event2"

    def test_validate_field_success_with_field_definition(self) -> None:
        """Test field validation success with field definition."""
        user = SampleUser(id="user_1", name="John", email="john@example.com")

        # Mock FlextFields.get_field_by_name to return success
        mock_field = MagicMock()
        mock_field.validate_value.return_value = FlextResult.ok(None)

        with patch(
            "flext_core.fields.FlextFields.get_field_by_name",
            return_value=FlextResult.ok(mock_field),
        ):
            result = user.validate_field("name", "John")

        assert result.success

    def test_validate_field_failure_with_field_definition(self) -> None:
        """Test field validation failure with field definition."""
        user = SampleUser(id="user_1", name="John", email="john@example.com")

        # Mock FlextFields.get_field_by_name to return success but validation fails
        mock_field = MagicMock()
        mock_field.validate_value.return_value = FlextResult.fail("Validation failed")

        with patch(
            "flext_core.fields.FlextFields.get_field_by_name",
            return_value=FlextResult.ok(mock_field),
        ):
            result = user.validate_field("name", "")

        assert result.is_failure
        if "Validation failed" not in (result.error or ""):
            raise AssertionError(f"Expected 'Validation failed' in {result.error}")

    def test_validate_field_no_field_definition(self) -> None:
        """Test field validation with no field definition found."""
        user = SampleUser(id="user_1", name="John", email="john@example.com")

        # Mock FlextFields.get_field_by_name to return failure (no field found)
        with patch(
            "flext_core.fields.FlextFields.get_field_by_name",
            return_value=FlextResult.fail("Field not found"),
        ):
            result = user.validate_field("unknown_field", "value")

        assert result.success  # Should succeed when no field definition found

    def test_validate_field_exception_handling(self) -> None:
        """Test field validation with exception handling."""
        user = SampleUser(id="user_1", name="John", email="john@example.com")

        # Mock FlextFields.get_field_by_name to raise exception
        with patch(
            "flext_core.fields.FlextFields.get_field_by_name",
            side_effect=ImportError("Import failed"),
        ):
            result = user.validate_field("name", "John")

        assert result.is_failure
        if "Field validation error" not in (result.error or ""):
            raise AssertionError(f"Expected 'Field validation error' in {result.error}")

    def test_validate_all_fields_success(self) -> None:
        """Test validation of all fields with success."""
        user = SampleUser(id="user_1", name="John", email="john@example.com")

        # Mock field validation to succeed for all fields
        mock_field = MagicMock()
        mock_field.validate_value.return_value = FlextResult.ok(None)

        with patch(
            "flext_core.fields.FlextFields.get_field_by_name",
            return_value=FlextResult.ok(mock_field),
        ):
            result = user.validate_all_fields()

        assert result.success

    def test_validate_all_fields_with_failures(self) -> None:
        """Test validation of all fields with some failures."""
        user = SampleUser(id="user_1", name="John", email="john@example.com")

        # Mock field validation to fail for some fields
        def mock_get_field(field_name: str) -> FlextResult[MagicMock]:
            mock_field = MagicMock()
            if field_name == "name":
                mock_field.validate_value.return_value = FlextResult.fail(
                    "Name validation failed",
                )
            else:
                mock_field.validate_value.return_value = FlextResult.ok(None)
            return FlextResult.ok(mock_field)

        with patch(
            "flext_core.fields.FlextFields.get_field_by_name",
            side_effect=mock_get_field,
        ):
            result = user.validate_all_fields()

        assert result.is_failure
        if "Field validation errors:" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Field validation errors:' in {result.error}"
            )
        assert result.error is not None
        assert "name: Name validation failed" in result.error

    def test_validate_all_fields_no_field_definitions(self) -> None:
        """Test validation of all fields with no field definitions."""
        user = SampleUser(id="user_1", name="John", email="john@example.com")

        # Mock FlextFields.get_field_by_name to always return failure (no fields found)
        with patch(
            "flext_core.fields.FlextFields.get_field_by_name",
            return_value=FlextResult.fail("Field not found"),
        ):
            result = user.validate_all_fields()

        assert result.success  # Should succeed when no field definitions found

    def test_with_version_success(self) -> None:
        """Test with_version method with valid version."""
        user = SampleUser(id="user_1", name="John", email="john@example.com", version=1)

        new_user = user.with_version(2)

        if new_user.version != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {new_user.version}")
        assert new_user.id == user.id
        if new_user.name != user.name:
            raise AssertionError(f"Expected {user.name}, got {new_user.name}")
        assert new_user is not user  # Should be a new instance

    def test_with_version_invalid_version(self) -> None:
        """Test with_version method with invalid version."""
        user = SampleUser(id="user_1", name="John", email="john@example.com", version=5)

        # Version must be greater than current version
        with pytest.raises(FlextValidationError) as exc_info:
            user.with_version(3)  # Lower than current version

        if "New version must be greater than current version" not in str(
            exc_info.value
        ):
            raise AssertionError(
                f"Expected {'New version must be greater than current version'} in {exc_info.value!s}"
            )

    def test_with_version_equal_version(self) -> None:
        """Test with_version method with equal version."""
        user = SampleUser(id="user_1", name="John", email="john@example.com", version=5)

        # Version must be greater, not equal
        with pytest.raises(FlextValidationError) as exc_info:
            user.with_version(5)  # Equal to current version

        if "New version must be greater than current version" not in str(
            exc_info.value
        ):
            raise AssertionError(
                f"Expected {'New version must be greater than current version'} in {exc_info.value!s}"
            )

    def test_with_version_domain_validation_failure(self) -> None:
        """Test with_version with domain validation failure."""
        user = SampleBadUser(
            id="user_1",
            name="John",
            email="john@example.com",
            version=1,
        )

        with pytest.raises(FlextValidationError, match="Always fails"):
            user.with_version(2)

    def test_with_version_construction_error(self) -> None:
        """Test with_version method with construction error."""
        user = SampleUser(id="user_1", name="John", email="john@example.com", version=1)

        # Mock the class constructor to cause TypeError during construction
        with (
            patch.object(SampleUser, "__init__", side_effect=TypeError("Mock error")),
            pytest.raises(FlextValidationError) as exc_info,
        ):
            user.with_version(2)

        if "Failed to set version" not in str(exc_info.value):
            raise AssertionError(
                f"Expected {'Failed to set version'} in {exc_info.value!s}"
            )


@pytest.mark.unit
class TestFlextEntityFactory:
    """Test FlextEntityFactory functionality."""

    def test_create_entity_factory_basic(self) -> None:
        """Test creating a basic entity factory."""
        factory = FlextEntityFactory.create_entity_factory(SampleUser)

        assert callable(factory)

    def test_factory_create_entity_success(self) -> None:
        """Test successful entity creation through factory."""
        factory = FlextEntityFactory.create_entity_factory(SampleUser)

        result = cast("EntityFactory", factory)(
            name="John", email="john@example.com", age=25
        )

        assert result.success
        user = cast("SampleUser", result.data)
        assert isinstance(user, SampleUser)
        if user.name != "John":
            raise AssertionError(f"Expected {'John'}, got {user.name}")
        assert user.email == "john@example.com"
        if user.age != 25:
            raise AssertionError(f"Expected {25}, got {user.age}")
        assert user.version == 1
        assert user.id  # Should have generated ID

    def test_factory_with_defaults(self) -> None:
        """Test factory with default values."""
        defaults = {"age": 18, "is_active": True}
        factory = FlextEntityFactory.create_entity_factory(
            SampleUser, cast("dict[str, object]", defaults)
        )

        result = cast("EntityFactory", factory)(name="John", email="john@example.com")

        assert result.success
        user = cast("SampleUser", result.data)
        if user.age != 18:  # From defaults:
            msg: str = f"Expected {18}, got {user.age}"
            raise AssertionError(msg)
        assert user.is_active is True  # From defaults

    def test_factory_override_defaults(self) -> None:
        """Test factory overriding default values."""
        defaults = {"age": 18, "is_active": False}
        factory = FlextEntityFactory.create_entity_factory(
            SampleUser, cast("dict[str, object]", defaults)
        )

        result = cast("EntityFactory", factory)(
            name="John", email="john@example.com", age=25, is_active=True
        )

        assert result.success
        user = cast("SampleUser", result.data)
        if user.age != 25:  # Overridden:
            msg: str = f"Expected {25}, got {user.age}"
            raise AssertionError(msg)
        assert user.is_active is True  # Overridden

    def test_factory_generate_id_when_not_provided(self) -> None:
        """Test factory generates ID when not provided."""
        factory = FlextEntityFactory.create_entity_factory(SampleUser)

        result = cast("EntityFactory", factory)(name="John", email="john@example.com")

        assert result.success
        user = cast("SampleUser", result.data)
        assert user.id
        assert len(user.id) > 0

    def test_factory_use_provided_id(self) -> None:
        """Test factory uses provided ID."""
        factory = FlextEntityFactory.create_entity_factory(SampleUser)

        result = cast("EntityFactory", factory)(
            id="custom_id", name="John", email="john@example.com"
        )

        assert result.success
        user = cast("SampleUser", result.data)
        if user.id != "custom_id":
            raise AssertionError(f"Expected {'custom_id'}, got {user.id}")

    def test_factory_handle_empty_id(self) -> None:
        """Test factory handles empty ID by generating new one."""
        factory = FlextEntityFactory.create_entity_factory(SampleUser)

        result = cast("EntityFactory", factory)(
            id="", name="John", email="john@example.com"
        )

        assert result.success
        user = cast("SampleUser", result.data)
        assert user.id
        assert user.id != ""

    def test_factory_set_default_version(self) -> None:
        """Test factory sets default version."""
        factory = FlextEntityFactory.create_entity_factory(SampleUser)

        result = cast("EntityFactory", factory)(name="John", email="john@example.com")

        assert result.success
        user = cast("SampleUser", result.data)
        if user.version != 1:
            raise AssertionError(f"Expected {1}, got {user.version}")

    def test_factory_use_provided_version(self) -> None:
        """Test factory uses provided version."""
        factory = FlextEntityFactory.create_entity_factory(SampleUser)

        result = cast("EntityFactory", factory)(
            name="John", email="john@example.com", version=5
        )

        assert result.success
        user = cast("SampleUser", result.data)
        if user.version != 5:
            raise AssertionError(f"Expected {5}, got {user.version}")

    def test_factory_domain_validation_failure(self) -> None:
        """Test factory with domain validation failure."""
        factory = FlextEntityFactory.create_entity_factory(SampleBadUser)

        result = cast("EntityFactory", factory)(name="John", email="john@example.com")

        assert result.is_failure
        if "Always fails" not in (result.error or ""):
            raise AssertionError(f"Expected 'Always fails' in {result.error}")

    def test_factory_model_validation_error(self) -> None:
        """Test factory with model validation error."""
        factory = FlextEntityFactory.create_entity_factory(SampleUser)

        # Missing required field 'name'
        result = cast("EntityFactory", factory)(email="john@example.com")

        assert result.is_failure
        if "Entity creation failed" not in (result.error or ""):
            raise AssertionError(f"Expected 'Entity creation failed' in {result.error}")

    def test_factory_type_error_handling(self) -> None:
        """Test factory handles TypeError."""
        factory = FlextEntityFactory.create_entity_factory(SampleUser)

        # Mock model_validate to raise TypeError
        with patch.object(
            SampleUser,
            "model_validate",
            side_effect=TypeError("Type error"),
        ):
            result = cast("EntityFactory", factory)(
                name="John", email="john@example.com"
            )

        assert result.is_failure
        if "Entity creation failed" not in (result.error or ""):
            raise AssertionError(f"Expected 'Entity creation failed' in {result.error}")

    def test_factory_import_error_handling(self) -> None:
        """Test factory handles ImportError."""
        factory = FlextEntityFactory.create_entity_factory(SampleUser)

        # Mock FlextGenerators.generate_entity_id to raise ImportError
        with patch(
            "flext_core.entities.FlextGenerators.generate_entity_id",
            side_effect=ImportError("Import error"),
        ):
            result = cast("EntityFactory", factory)(
                name="John", email="john@example.com"
            )

        assert result.is_failure
        if "Import error" not in (result.error or ""):
            raise AssertionError(f"Expected 'Import error' in {result.error}")


@pytest.mark.unit
class TestEntityIntegration:
    """Test entity integration scenarios."""

    def test_entity_lifecycle_with_events(self) -> None:
        """Test complete entity lifecycle with domain events."""
        # Create entity
        user = SampleUser(id="user_1", name="John", email="john@example.com")

        # Add domain events
        user.add_domain_event("user_created", {"timestamp": "2023-01-01"})
        user.add_domain_event("profile_updated", {"field": "name"})

        if len(user.domain_events) != EXPECTED_BULK_SIZE:
            msg: str = f"Expected {2}, got {len(user.domain_events)}"
            raise AssertionError(msg)

        # Modify entity
        result = user.copy_with(name="John Doe")
        assert result.success

        updated_user = cast("SampleUser", result.data)
        assert updated_user is not None
        if updated_user.name != "John Doe":
            raise AssertionError(f"Expected {'John Doe'}, got {updated_user.name}")
        assert updated_user.version == EXPECTED_BULK_SIZE

        # Original events are preserved (entity is immutable)
        if len(user.domain_events) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(user.domain_events)}")
        assert len(updated_user.domain_events) == 0  # New entity has no events

        # Clear events
        events = user.clear_events()
        if len(events) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(events)}")
        assert len(user.domain_events) == 0

    def test_factory_with_complex_scenario(self) -> None:
        """Test factory with complex entity creation scenario."""
        defaults = {
            "age": 21,
            "is_active": False,
        }

        factory = FlextEntityFactory.create_entity_factory(
            SampleUser, cast("dict[str, object]", defaults)
        )

        # Create multiple entities
        results = [
            cast("EntityFactory", factory)(name="Alice", email="alice@example.com"),
            cast("EntityFactory", factory)(name="Bob", email="bob@example.com", age=30),
            cast("EntityFactory", factory)(
                id="admin",
                name="Admin",
                email="admin@example.com",
                is_active=True,
            ),
        ]

        if not all(result.success for result in results):
            msg = f"Expected {all(result.success for result in results)} in {results}"
            raise AssertionError(msg)

        users = [cast("SampleUser", result.data) for result in results]

        # Check Alice (uses defaults)
        if users[0].name != "Alice":
            raise AssertionError(f"Expected {'Alice'}, got {users[0].name}")
        assert users[0].age == 21  # From defaults
        assert users[0].is_active is False  # From defaults

        # Check Bob (overrides age)
        if users[1].name != "Bob":
            raise AssertionError(f"Expected {'Bob'}, got {users[1].name}")
        assert users[1].age == 30  # Overridden
        assert users[1].is_active is False  # From defaults

        # Check Admin (custom ID and overrides)
        if users[2].id != "admin":
            raise AssertionError(f"Expected {'admin'}, got {users[2].id}")
        assert users[2].name == "Admin"
        assert users[2].is_active is True  # Overridden

        # All should have different IDs (except admin)
        assert users[0].id != users[1].id
        if users[2].id != "admin":
            raise AssertionError(f"Expected {'admin'}, got {users[2].id}")


class TestEntityCoverageImprovements:
    """Test cases specifically for improving coverage of entities.py module."""

    def test_validate_all_fields_with_internal_fields(self) -> None:
        """Test validate_all_fields skips internal fields (line 469 coverage)."""

        class EntityWithInternalFields(FlextEntity):
            name: str

            def validate_domain_rules(self) -> FlextResult[None]:
                """Validate domain rules."""
                return FlextResult.ok(None)

            def model_dump(self, **kwargs: object) -> dict[str, object]:
                """Override model_dump to include internal fields for testing."""
                normal_data = super().model_dump()
                # Add internal fields that should be skipped by validate_all_fields
                normal_data["_internal_field"] = "internal_value"
                normal_data["_another_internal"] = "another_internal_value"
                return normal_data

        # Create entity
        entity = EntityWithInternalFields(id="test-123", name="Test")

        # validate_all_fields should skip _internal_field, _another_internal, and domain_events (line 469)
        result = entity.validate_all_fields()

        # Should succeed because internal fields are skipped during validation
        assert result.success
