"""Comprehensive tests for types_system module.

Tests for the modern FLEXT type system including validation functions,
Pydantic models, and type annotations.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from pydantic import ValidationError

from flext_core.payload import FlextPayload
from flext_core.types_system import FlextConfigKey
from flext_core.types_system import FlextEntityId
from flext_core.types_system import FlextEventType
from flext_core.types_system import FlextIdentifier
from flext_core.types_system import FlextServiceName
from flext_core.types_system import FlextTypedDict
from flext_core.types_system import flext_validate_config_key
from flext_core.types_system import flext_validate_event_type
from flext_core.types_system import flext_validate_identifier
from flext_core.types_system import flext_validate_non_empty_string
from flext_core.types_system import flext_validate_service_name


class TestValidationFunctions:
    """Test validation functions for string types."""

    def test_flext_validate_non_empty_string_valid(self) -> None:
        """Test non-empty string validation with valid inputs."""
        assert flext_validate_non_empty_string("hello") == "hello"
        assert flext_validate_non_empty_string("  world  ") == "world"
        assert flext_validate_non_empty_string("123") == "123"
        assert flext_validate_non_empty_string("test-value") == "test-value"

    def test_flext_validate_non_empty_string_invalid(self) -> None:
        """Test non-empty string validation with invalid inputs."""
        with pytest.raises(ValueError, match="Cannot be empty"):
            flext_validate_non_empty_string("")

        with pytest.raises(ValueError, match="Cannot be empty"):
            flext_validate_non_empty_string("   ")

        with pytest.raises(ValueError, match="Cannot be empty"):
            flext_validate_non_empty_string("\t\n")

    def test_flext_validate_identifier_valid(self) -> None:
        """Test identifier validation with valid inputs."""
        assert flext_validate_identifier("test") == "test"
        assert flext_validate_identifier("user-123") == "user-123"
        assert flext_validate_identifier("item_456") == "item_456"
        assert flext_validate_identifier("Product1") == "Product1"
        assert flext_validate_identifier("a") == "a"

    def test_flext_validate_identifier_invalid(self) -> None:
        """Test identifier validation with invalid inputs."""
        with pytest.raises(ValueError, match="only letters, numbers"):
            flext_validate_identifier("test@domain")

        with pytest.raises(ValueError, match="only letters, numbers"):
            flext_validate_identifier("test.name")

        with pytest.raises(ValueError, match="only letters, numbers"):
            flext_validate_identifier("test space")

        with pytest.raises(ValueError, match="255 characters"):
            flext_validate_identifier("x" * 256)

    def test_flext_validate_service_name_valid(self) -> None:
        """Test service name validation with valid inputs."""
        assert flext_validate_service_name("database") == "database"
        assert flext_validate_service_name("user-service") == "user-service"
        assert flext_validate_service_name("email_sender") == "email_sender"
        assert flext_validate_service_name("cache-v2") == "cache-v2"

    def test_flext_validate_service_name_invalid(self) -> None:
        """Test service name validation with invalid inputs."""
        with pytest.raises(ValueError, match="Cannot start or end"):
            flext_validate_service_name("-service")

        with pytest.raises(ValueError, match="Cannot start or end"):
            flext_validate_service_name("service-")

        with pytest.raises(ValueError, match="consecutive dashes"):
            flext_validate_service_name("service--name")

        with pytest.raises(ValueError, match="consecutive.*underscores"):
            flext_validate_service_name("service__name")

    def test_flext_validate_config_key_valid(self) -> None:
        """Test config key validation with valid inputs."""
        assert flext_validate_config_key("database.host") == "database.host"
        assert flext_validate_config_key("api.timeout") == "api.timeout"
        assert flext_validate_config_key("log-level") == "log-level"
        assert (
            flext_validate_config_key("app.db.connection")
            == "app.db.connection"
        )

    def test_flext_validate_config_key_invalid(self) -> None:
        """Test config key validation with invalid inputs."""
        with pytest.raises(ValueError, match="Cannot start or end"):
            flext_validate_config_key(".config")

        with pytest.raises(ValueError, match="Cannot start or end"):
            flext_validate_config_key("config.")

        with pytest.raises(ValueError, match="consecutive dots"):
            flext_validate_config_key("config..key")

        with pytest.raises(ValueError, match="only letters, numbers"):
            flext_validate_config_key("config@key")

    def test_flext_validate_event_type_valid(self) -> None:
        """Test event type validation with valid inputs."""
        assert flext_validate_event_type("user.created") == "user.created"
        assert (
            flext_validate_event_type("order.completed") == "order.completed"
        )
        assert (
            flext_validate_event_type("app.domain.event") == "app.domain.event"
        )
        assert (
            flext_validate_event_type("system-v2.started")
            == "system-v2.started"
        )

    def test_flext_validate_event_type_invalid(self) -> None:
        """Test event type validation with invalid inputs."""
        with pytest.raises(ValueError, match="at least one dot"):
            flext_validate_event_type("event")

        with pytest.raises(ValueError, match="empty parts"):
            flext_validate_event_type("user..created")

        with pytest.raises(ValueError, match="empty parts"):
            flext_validate_event_type("user.created.")

        with pytest.raises(ValueError, match="only letters, numbers"):
            flext_validate_event_type("user@domain.created")


class TestAnnotatedTypes:
    """Test Pydantic annotated types."""

    def test_flext_entity_id_valid(self) -> None:
        """Test FlextEntityId with valid values."""

        class TestModel(BaseModel):
            entity_id: FlextEntityId

        # Valid entity IDs
        model1 = TestModel(entity_id="user-123")
        assert model1.entity_id == "user-123"

        model2 = TestModel(entity_id="order_456")
        assert model2.entity_id == "order_456"

    def test_flext_entity_id_invalid(self) -> None:
        """Test FlextEntityId with invalid values."""

        class TestModel(BaseModel):
            entity_id: FlextEntityId

        with pytest.raises(ValidationError):
            TestModel(entity_id="")

        with pytest.raises(ValidationError):
            TestModel(entity_id="user@domain")

    def test_flext_service_name_valid(self) -> None:
        """Test FlextServiceName with valid values."""

        class TestModel(BaseModel):
            service: FlextServiceName

        model = TestModel(service="database")
        assert model.service == "database"

        model2 = TestModel(service="user-service")
        assert model2.service == "user-service"

    def test_flext_service_name_invalid(self) -> None:
        """Test FlextServiceName with invalid values."""

        class TestModel(BaseModel):
            service: FlextServiceName

        with pytest.raises(ValidationError):
            TestModel(service="-service")

        with pytest.raises(ValidationError):
            TestModel(service="service--name")

    def test_flext_config_key_valid(self) -> None:
        """Test FlextConfigKey with valid values."""

        class TestModel(BaseModel):
            config_key: FlextConfigKey

        model = TestModel(config_key="database.host")
        assert model.config_key == "database.host"

    def test_flext_event_type_valid(self) -> None:
        """Test FlextEventType with valid values."""

        class TestModel(BaseModel):
            event: FlextEventType

        model = TestModel(event="user.created")
        assert model.event == "user.created"


class TestFlextIdentifier:
    """Test FlextIdentifier class."""

    def test_identifier_auto_generation(self) -> None:
        """Test automatic UUID generation."""
        id1 = FlextIdentifier()
        id2 = FlextIdentifier()

        assert len(str(id1)) == 36  # UUID length
        assert len(str(id2)) == 36
        assert str(id1) != str(id2)  # Different UUIDs

    def test_identifier_custom_value(self) -> None:
        """Test custom identifier values."""
        identifier = FlextIdentifier(value="user-123")
        assert str(identifier) == "user-123"
        assert identifier.value == "user-123"

    def test_identifier_validation(self) -> None:
        """Test identifier validation."""
        with pytest.raises(ValidationError):
            FlextIdentifier(value="")

        with pytest.raises(ValidationError):
            FlextIdentifier(value="   ")

    def test_identifier_equality(self) -> None:
        """Test identifier equality comparison."""
        id1 = FlextIdentifier(value="test-123")
        id2 = FlextIdentifier(value="test-123")
        id3 = FlextIdentifier(value="test-456")

        assert id1 == id2
        assert id1 != id3
        assert id1 == "test-123"
        assert id1 != "test-456"

    def test_identifier_hash(self) -> None:
        """Test identifier hashing."""
        id1 = FlextIdentifier(value="test-123")
        id2 = FlextIdentifier(value="test-123")

        assert hash(id1) == hash(id2)
        assert hash(id1) == hash("test-123")

    def test_identifier_immutability(self) -> None:
        """Test that identifier is immutable."""
        identifier = FlextIdentifier(value="test-123")

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            identifier.value = "new-value"  # type: ignore[misc]

    def test_identifier_string_representation(self) -> None:
        """Test string representation methods."""
        identifier = FlextIdentifier(value="test-id")

        assert str(identifier) == "test-id"
        assert repr(identifier).startswith("FlextIdentifier")
        assert "test-id" in repr(identifier)


class TestFlextPayload:
    """Test FlextPayload class from types_system module."""

    def test_payload_empty_creation(self) -> None:
        """Test creating empty payload."""
        payload = FlextPayload()
        assert isinstance(payload, FlextPayload)

    def test_payload_with_data(self) -> None:
        """Test payload with initial data."""
        payload = FlextPayload(user_id="123", action="login")
        assert payload.get("user_id") == "123"
        assert payload.get("action") == "login"

    def test_payload_get_method(self) -> None:
        """Test get method with defaults."""
        payload = FlextPayload(key="value")
        assert payload.get("key") == "value"
        assert payload.get("missing") is None
        assert payload.get("missing", "default") == "default"

    def test_payload_has_method(self) -> None:
        """Test has method for field existence."""
        payload = FlextPayload(field="value")
        assert payload.has("field") is True
        assert payload.has("missing") is False

    def test_payload_to_dict_method(self) -> None:
        """Test conversion to dictionary."""
        payload = FlextPayload(key1="value1", key2="value2")
        data = payload.model_dump()
        assert isinstance(data, dict)
        assert data["key1"] == "value1"
        assert data["key2"] == "value2"

    def test_payload_immutability(self) -> None:
        """Test that payload is immutable."""
        payload = FlextPayload(key="value")

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            payload.key = "new_value"  # type: ignore[attr-defined]

    def test_payload_mixed_types(self) -> None:
        """Test payload with various data types."""
        payload = FlextPayload(
            string_val="text",
            int_val=42,
            bool_val=True,
            list_val=[1, 2, 3],
            dict_val={"nested": "data"},
        )

        assert payload.get("string_val") == "text"
        assert payload.get("int_val") == 42
        assert payload.get("bool_val") is True
        assert payload.get("list_val") == [1, 2, 3]
        assert payload.get("dict_val") == {"nested": "data"}


class TestFlextTypedDict:
    """Test FlextTypedDict class."""

    def test_typed_dict_creation(self) -> None:
        """Test creating typed dictionary."""
        data = FlextTypedDict(name="Alice", age=30, active=True)
        assert data.get("name") == "Alice"
        assert data.get("age") == 30
        assert data.get("active") is True

    def test_typed_dict_get_method(self) -> None:
        """Test get method with defaults."""
        data = FlextTypedDict(key="value")
        assert data.get("key") == "value"
        assert data.get("missing") is None
        assert data.get("missing", "default") == "default"

    def test_typed_dict_has_method(self) -> None:
        """Test has method for key existence."""
        data = FlextTypedDict(field="value")
        assert data.has("field") is True
        assert data.has("missing") is False

    def test_typed_dict_keys_method(self) -> None:
        """Test keys method."""
        data = FlextTypedDict(key1="value1", key2="value2")
        keys = data.keys()
        assert isinstance(keys, list)
        assert "key1" in keys
        assert "key2" in keys

    def test_typed_dict_items_method(self) -> None:
        """Test items method."""
        data = FlextTypedDict(key1="value1", key2="value2")
        items = data.items()
        assert isinstance(items, list)
        assert len(items) >= 2

        # Check that our data is in items
        item_dict = dict(items)
        assert item_dict["key1"] == "value1"
        assert item_dict["key2"] == "value2"

    def test_typed_dict_immutability(self) -> None:
        """Test that typed dict is immutable."""
        data = FlextTypedDict(key="value")

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            data.key = "new_value"  # type: ignore[attr-defined]

    def test_typed_dict_empty(self) -> None:
        """Test empty typed dictionary."""
        data = FlextTypedDict()
        assert isinstance(data.keys(), list)
        assert isinstance(data.items(), list)


class TestTypeSystemIntegration:
    """Integration tests for type system components."""

    def test_types_with_pydantic_models(self) -> None:
        """Test using FLEXT types in Pydantic models."""

        class UserModel(BaseModel):
            user_id: FlextEntityId
            service: FlextServiceName
            config_key: FlextConfigKey
            event_type: FlextEventType

        user = UserModel(
            user_id="user-123",
            service="user-service",
            config_key="app.user.timeout",
            event_type="user.created",
        )

        assert user.user_id == "user-123"
        assert user.service == "user-service"
        assert user.config_key == "app.user.timeout"
        assert user.event_type == "user.created"

    def test_identifier_in_payload(self) -> None:
        """Test using FlextIdentifier in FlextPayload."""
        identifier = FlextIdentifier(value="test-id")
        payload = FlextPayload(id=str(identifier), data="test-data")

        assert payload.get("id") == "test-id"
        assert payload.get("data") == "test-data"

    def test_nested_payload_structures(self) -> None:
        """Test nested payload structures."""
        inner_payload = FlextPayload(nested="value")
        outer_payload = FlextPayload(
            id="outer-123",
            inner=inner_payload.model_dump(),
        )

        assert outer_payload.get("id") == "outer-123"
        inner_data = outer_payload.get("inner")
        assert isinstance(inner_data, dict)
        assert inner_data["nested"] == "value"

    def test_type_validation_errors(self) -> None:
        """Test that validation errors are properly raised."""

        class TestModel(BaseModel):
            entity_id: FlextEntityId
            service_name: FlextServiceName

        # Test various invalid inputs
        with pytest.raises(ValidationError):
            TestModel(entity_id="", service_name="valid-service")

        with pytest.raises(ValidationError):
            TestModel(entity_id="valid-id", service_name="-invalid")

        with pytest.raises(ValidationError):
            TestModel(entity_id="invalid@id", service_name="valid-service")
