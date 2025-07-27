"""Extended tests for FlextIntegerField, FlextBooleanField, and FlextFieldRegistry."""

from __future__ import annotations

import pytest

from flext_core.constants import FlextFieldType
from flext_core.fields import (
    FlextBooleanField,
    FlextFieldMetadata,
    FlextFieldRegistry,
    FlextIntegerField,
    FlextStringField,
)


class TestFlextIntegerField:
    """Test FlextIntegerField concrete implementation."""

    def test_integer_field_creation_minimal(self) -> None:
        """Test integer field creation with minimal parameters."""
        field = FlextIntegerField(
            field_id="test_id",
            field_name="test_field",
        )

        assert field.field_id == "test_id"
        assert field.field_name == "test_field"
        assert field.field_type == FlextFieldType.INTEGER
        assert field.required is True  # Default is True
        assert field.default_value is None

    def test_integer_field_creation_with_constraints(self) -> None:
        """Test integer field creation with value constraints."""
        field = FlextIntegerField(
            field_id="test_id",
            field_name="test_field",
            min_value=0,
            max_value=100,
        )

        assert field.metadata.min_value == 0
        assert field.metadata.max_value == 100

    def test_integer_field_creation_with_custom_metadata(self) -> None:
        """Test integer field creation with custom metadata."""
        custom_metadata = FlextFieldMetadata(
            description="Custom integer field",
            min_value=10,
        )

        field = FlextIntegerField(
            field_id="test_id",
            field_name="test_field",
            metadata=custom_metadata,
        )

        assert field.metadata is custom_metadata
        assert field.metadata.description == "Custom integer field"
        assert field.metadata.min_value == 10

    def test_integer_field_validate_value_success(self) -> None:
        """Test integer field validation success."""
        field = FlextIntegerField(
            field_id="test_id",
            field_name="test_field",
        )

        result = field.validate_value(42)
        assert result.is_success

    def test_integer_field_validate_value_failure(self) -> None:
        """Test integer field validation failure."""
        field = FlextIntegerField(
            field_id="test_id",
            field_name="test_field",
        )

        result = field.validate_value("not_an_integer")
        assert result.is_failure
        assert result.error is not None

    def test_integer_field_validate_value_success_case(self) -> None:
        """Test integer field validation with valid value."""
        field = FlextIntegerField(
            field_id="test_id",
            field_name="test_field",
        )

        result = field.validate_value(42)
        assert result.is_success

    def test_integer_field_serialize_value(self) -> None:
        """Test integer field serialization."""
        field = FlextIntegerField(
            field_id="test_id",
            field_name="test_field",
        )

        result = field.serialize_value(42)
        assert result == 42

    def test_integer_field_deserialize_value_integer(self) -> None:
        """Test integer field deserialization from integer."""
        field = FlextIntegerField(
            field_id="test_id",
            field_name="test_field",
        )

        result = field.deserialize_value(42)
        assert result == 42

    def test_integer_field_deserialize_value_string(self) -> None:
        """Test integer field deserialization from string."""
        field = FlextIntegerField(
            field_id="test_id",
            field_name="test_field",
        )

        result = field.deserialize_value("42")
        assert result == 42

    def test_integer_field_deserialize_value_invalid_string(self) -> None:
        """Test integer field deserialization from invalid string."""
        field = FlextIntegerField(
            field_id="test_id",
            field_name="test_field",
        )

        with pytest.raises(ValueError, match="Cannot deserialize"):
            field.deserialize_value("not_a_number")

    def test_integer_field_deserialize_value_invalid_type(self) -> None:
        """Test integer field deserialization from invalid type."""
        field = FlextIntegerField(
            field_id="test_id",
            field_name="test_field",
        )

        with pytest.raises(ValueError, match="Cannot deserialize"):
            field.deserialize_value([])

    def test_integer_field_with_all_parameters(self) -> None:
        """Test integer field with all parameters."""
        metadata = FlextFieldMetadata(description="Test integer field")

        field = FlextIntegerField(
            field_id="test_id",
            field_name="test_field",
            required=True,
            default_value=10,
            min_value=0,
            max_value=100,
            metadata=metadata,
        )

        assert field.field_id == "test_id"
        assert field.field_name == "test_field"
        assert field.field_type == FlextFieldType.INTEGER
        assert field.required is True
        assert field.default_value == 10
        assert field.metadata is metadata


class TestFlextBooleanField:
    """Test FlextBooleanField concrete implementation."""

    def test_boolean_field_creation_minimal(self) -> None:
        """Test boolean field creation with minimal parameters."""
        field = FlextBooleanField(
            field_id="test_id",
            field_name="test_field",
        )

        assert field.field_id == "test_id"
        assert field.field_name == "test_field"
        assert field.field_type == FlextFieldType.BOOLEAN
        assert field.required is True  # Default is True
        assert field.default_value is None

    def test_boolean_field_creation_with_custom_metadata(self) -> None:
        """Test boolean field creation with custom metadata."""
        custom_metadata = FlextFieldMetadata(
            description="Custom boolean field",
        )

        field = FlextBooleanField(
            field_id="test_id",
            field_name="test_field",
            metadata=custom_metadata,
        )

        assert field.metadata is custom_metadata
        assert field.metadata.description == "Custom boolean field"

    def test_boolean_field_validate_value_success(self) -> None:
        """Test boolean field validation success."""
        field = FlextBooleanField(
            field_id="test_id",
            field_name="test_field",
        )

        result = field.validate_value(value=True)
        assert result.is_success

        result = field.validate_value(value=False)
        assert result.is_success

    def test_boolean_field_validate_value_failure(self) -> None:
        """Test boolean field validation failure."""
        field = FlextBooleanField(
            field_id="test_id",
            field_name="test_field",
        )

        result = field.validate_value("not_a_boolean")
        assert result.is_failure
        assert result.error is not None

    def test_boolean_field_validate_value_success_case(self) -> None:
        """Test boolean field validation with valid boolean."""
        field = FlextBooleanField(
            field_id="test_id",
            field_name="test_field",
        )

        test_value = True
        result = field.validate_value(test_value)
        assert result.is_success

    def test_boolean_field_serialize_value(self) -> None:
        """Test boolean field serialization."""
        field = FlextBooleanField(
            field_id="test_id",
            field_name="test_field",
        )

        result = field.serialize_value(value=True)
        assert result is True

        result = field.serialize_value(value=False)
        assert result is False

    def test_boolean_field_deserialize_value_boolean(self) -> None:
        """Test boolean field deserialization from boolean."""
        field = FlextBooleanField(
            field_id="test_id",
            field_name="test_field",
        )

        result = field.deserialize_value(value=True)
        assert result is True

        result = field.deserialize_value(value=False)
        assert result is False

    def test_boolean_field_deserialize_value_string_true(self) -> None:
        """Test boolean field deserialization from true strings."""
        field = FlextBooleanField(
            field_id="test_id",
            field_name="test_field",
        )

        for true_value in ["true", "True", "TRUE", "1", "yes", "YES", "on"]:
            result = field.deserialize_value(true_value)
            assert result is True, f"Failed for value: {true_value}"

    def test_boolean_field_deserialize_value_string_false(self) -> None:
        """Test boolean field deserialization from false strings."""
        field = FlextBooleanField(
            field_id="test_id",
            field_name="test_field",
        )

        for false_value in ["false", "False", "FALSE", "0", "no", "off"]:
            result = field.deserialize_value(false_value)
            assert result is False, f"Failed for value: {false_value}"

    def test_boolean_field_deserialize_value_integer(self) -> None:
        """Test boolean field deserialization from integer."""
        field = FlextBooleanField(
            field_id="test_id",
            field_name="test_field",
        )

        result = field.deserialize_value(value=1)
        assert result is True

        result = field.deserialize_value(value=0)
        assert result is False

        result = field.deserialize_value(value=42)
        assert result is True

    def test_boolean_field_deserialize_value_invalid_type(self) -> None:
        """Test boolean field deserialization from invalid type."""
        field = FlextBooleanField(
            field_id="test_id",
            field_name="test_field",
        )

        with pytest.raises(ValueError, match="Cannot deserialize"):
            field.deserialize_value([])

    def test_boolean_field_with_all_parameters(self) -> None:
        """Test boolean field with all parameters."""
        metadata = FlextFieldMetadata(description="Test boolean field")

        field = FlextBooleanField(
            field_id="test_id",
            field_name="test_field",
            required=True,
            default_value=False,
            metadata=metadata,
        )

        assert field.field_id == "test_id"
        assert field.field_name == "test_field"
        assert field.field_type == FlextFieldType.BOOLEAN
        assert field.required is True
        assert field.default_value is False
        assert field.metadata is metadata


class TestFlextFieldRegistry:
    """Test FlextFieldRegistry implementation."""

    def test_registry_creation(self) -> None:
        """Test registry creation."""
        registry = FlextFieldRegistry()
        assert isinstance(registry, FlextFieldRegistry)
        assert registry.get_all_fields() == {}

    def test_register_field(self) -> None:
        """Test registering a field."""
        registry = FlextFieldRegistry()
        field = FlextStringField(
            field_id="test_id",
            field_name="test_field",
        )

        registry.register_field(field)
        registered_field = registry.get_field("test_id")
        assert registered_field is field

    def test_get_field_existing(self) -> None:
        """Test getting an existing field."""
        registry = FlextFieldRegistry()
        field = FlextStringField(
            field_id="test_id",
            field_name="test_field",
        )
        registry.register_field(field)

        result = registry.get_field("test_id")
        assert result is field

    def test_get_field_non_existing(self) -> None:
        """Test getting a non-existing field."""
        registry = FlextFieldRegistry()

        result = registry.get_field("non_existing")
        assert result is None

    def test_get_all_fields(self) -> None:
        """Test getting all registered fields."""
        registry = FlextFieldRegistry()
        field1 = FlextStringField(
            field_id="field1",
            field_name="Field 1",
        )
        field2 = FlextIntegerField(
            field_id="field2",
            field_name="Field 2",
        )

        registry.register_field(field1)
        registry.register_field(field2)

        all_fields = registry.get_all_fields()
        assert len(all_fields) == 2
        assert all_fields["field1"] is field1
        assert all_fields["field2"] is field2

    def test_get_fields_by_type(self) -> None:
        """Test getting fields by type."""
        registry = FlextFieldRegistry()
        string_field1 = FlextStringField(
            field_id="string1",
            field_name="String 1",
        )
        string_field2 = FlextStringField(
            field_id="string2",
            field_name="String 2",
        )
        integer_field = FlextIntegerField(
            field_id="integer1",
            field_name="Integer 1",
        )

        registry.register_field(string_field1)
        registry.register_field(string_field2)
        registry.register_field(integer_field)

        string_fields = registry.get_fields_by_type(FlextFieldType.STRING)
        assert len(string_fields) == 2
        assert string_field1 in string_fields
        assert string_field2 in string_fields

        integer_fields = registry.get_fields_by_type(FlextFieldType.INTEGER)
        assert len(integer_fields) == 1
        assert integer_field in integer_fields

        boolean_fields = registry.get_fields_by_type(FlextFieldType.BOOLEAN)
        assert len(boolean_fields) == 0

    def test_validate_all_fields_success(self) -> None:
        """Test validating all fields with success."""
        registry = FlextFieldRegistry()
        string_field = FlextStringField(
            field_id="name",
            field_name="Name",
        )
        integer_field = FlextIntegerField(
            field_id="age",
            field_name="Age",
        )

        registry.register_field(string_field)
        registry.register_field(integer_field)

        data = {
            "name": "John Doe",
            "age": 30,
        }

        result = registry.validate_all_fields(data)
        assert result.is_success

    def test_validate_all_fields_missing_required(self) -> None:
        """Test validating all fields with missing required field."""
        registry = FlextFieldRegistry()
        required_field = FlextStringField(
            field_id="name",
            field_name="Name",
            required=True,
        )

        registry.register_field(required_field)

        data: dict[str, str] = {}  # Missing required field

        result = registry.validate_all_fields(data)
        assert result.is_failure
        assert result.error is not None

    def test_validate_all_fields_invalid_value(self) -> None:
        """Test validating all fields with invalid value."""
        registry = FlextFieldRegistry()
        integer_field = FlextIntegerField(
            field_id="age",
            field_name="Age",
        )

        registry.register_field(integer_field)

        data = {
            "age": "not_an_integer",
        }

        result = registry.validate_all_fields(data)
        assert result.is_failure
        assert result.error is not None

    def test_validate_all_fields_mixed_results(self) -> None:
        """Test validating all fields with mixed success/failure."""
        registry = FlextFieldRegistry()
        required_field = FlextStringField(
            field_id="required_name",
            field_name="Required Name",
            required=True,
        )
        optional_field = FlextStringField(
            field_id="optional_desc",
            field_name="Optional Description",
        )
        integer_field = FlextIntegerField(
            field_id="age",
            field_name="Age",
        )

        registry.register_field(required_field)
        registry.register_field(optional_field)
        registry.register_field(integer_field)

        # Missing required field, but other fields are valid
        data = {
            "optional_desc": "Some description",
            "age": 25,
        }

        result = registry.validate_all_fields(data)
        assert result.is_failure  # Because required field is missing
        assert result.error is not None

    def test_validate_all_fields_empty_registry(self) -> None:
        """Test validating with empty registry."""
        registry = FlextFieldRegistry()

        data = {
            "some_field": "some_value",
        }

        result = registry.validate_all_fields(data)
        assert result.is_success  # No fields to validate = success
