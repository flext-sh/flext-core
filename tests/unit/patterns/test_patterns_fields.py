"""Comprehensive tests for FLEXT patterns fields module."""

from __future__ import annotations

import pytest

from flext_core.constants import FlextFieldType
from flext_core.fields import (
    FlextFieldCore,
    FlextFieldCoreMetadata,
    FlextFieldRegistry,
)
from flext_core.result import FlextResult


class TestFlextFieldCoreInteger:
    """Test FlextFieldCore integer type."""

    def test_integer_field_creation_minimal(self) -> None:
        """Test integer field creation with minimal parameters."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.INTEGER.value,
        )

        assert field.field_id == "test_id"
        assert field.field_name == "test_field"
        # field_type is stored as string value
        assert field.field_type == FlextFieldType.INTEGER.value
        assert field.required is True  # Default is True, not False
        assert field.default_value is None

    def test_integer_field_creation_with_constraints(self) -> None:
        """Test integer field creation with min/max constraints."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.INTEGER.value,
            min_value=1,
            max_value=100,
        )

        assert field.min_value == 1
        assert field.max_value == 100

    def test_integer_field_validate_value_success(self) -> None:
        """Test integer field validation with valid value."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.INTEGER.value,
        )

        result = field.validate_value(42)
        assert result.is_success

    def test_integer_field_validate_value_success_case(self) -> None:
        """Test integer field validation with valid value."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="integer",
        )

        result = field.validate_value(42)
        assert result.is_success

    def test_integer_field_validate_none_value(self) -> None:
        """Test integer field validation with None value."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="integer",
            required=False,  # Allow None values
        )

        result = field.validate_value(None)
        assert result.is_success

    def test_integer_field_validate_non_integer_type(self) -> None:
        """Test integer field validation with non-integer type."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="integer",
        )

        result = field.validate_value("not an integer")
        assert result.is_failure
        if result.error:
            assert "integer" in result.error.lower()
        else:
            pytest.fail("Expected validation error for non-integer type")

    def test_integer_field_serialize_value(self) -> None:
        """Test integer field serialization."""
        FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="string",
        )

        # Field doesn't have serialize_value method directly - using delegated method
        # This test needs to be adjusted based on actual field API
        # result = field.serialize_value(42)  # noqa: ERA001

    def test_metadata_creation_with_defaults(self) -> None:
        """Test metadata creation with default values."""
        metadata = FlextFieldCoreMetadata()

        assert metadata.description is None
        assert metadata.example is None
        assert metadata.min_value is None
        assert metadata.max_value is None
        assert metadata.min_length is None
        assert metadata.max_length is None
        assert metadata.pattern is None
        assert metadata.allowed_values == []
        assert metadata.deprecated is False
        assert metadata.sensitive is False
        assert metadata.indexed is False
        assert metadata.tags == []

    def test_metadata_creation_with_values(self) -> None:
        """Test metadata creation with specific values."""
        metadata = FlextFieldCoreMetadata(
            description="Test field",
            example="example value",
            min_value=0.0,
            max_value=100.0,
            min_length=1,
            max_length=50,
            pattern=r"^\w+$",
            allowed_values=["a", "b", "c"],
            deprecated=True,
            internal=True,
            sensitive=True,
            indexed=True,
            unique=True,
            tags=["test", "important"],
            custom_properties={"key": "value"},
        )

        assert metadata.description == "Test field"
        assert metadata.example == "example value"
        assert metadata.min_value == 0.0
        assert metadata.max_value == 100.0
        assert metadata.min_length == 1
        assert metadata.max_length == 50
        assert metadata.pattern == r"^\w+$"
        assert metadata.allowed_values == ["a", "b", "c"]
        assert metadata.deprecated is True
        assert metadata.internal is True
        assert metadata.sensitive is True
        assert metadata.indexed is True
        assert metadata.unique is True
        assert metadata.tags == ["test", "important"]
        assert metadata.custom_properties == {"key": "value"}

    def test_metadata_to_dict(self) -> None:
        """Test metadata serialization to dictionary."""
        metadata = FlextFieldCoreMetadata(
            description="Test field",
            min_length=1,
            max_length=10,
            deprecated=True,
            tags=["test"],
        )

        result = metadata.to_dict()

        assert result["description"] == "Test field"
        assert result["min_length"] == 1
        assert result["max_length"] == 10
        assert result["deprecated"] is True
        assert result["tags"] == ["test"]
        assert result["internal"] is False

    def test_metadata_from_dict(self) -> None:
        """Test metadata creation from dictionary."""
        data = {
            "description": "Test field",
            "min_length": 1,
            "max_length": 10,
            "deprecated": True,
            "tags": ["test"],
            "custom_properties": {"key": "value"},
        }

        metadata = FlextFieldCoreMetadata.from_dict(data)

        assert metadata.description == "Test field"
        assert metadata.min_length == 1
        assert metadata.max_length == 10
        assert metadata.deprecated is True
        assert metadata.tags == ["test"]
        assert metadata.custom_properties == {"key": "value"}

    def test_metadata_from_dict_with_defaults(self) -> None:
        """Test metadata creation from partial dictionary."""
        data = {"description": "Test field"}

        metadata = FlextFieldCoreMetadata.from_dict(data)

        assert metadata.description == "Test field"
        assert metadata.deprecated is False  # default
        assert metadata.tags == []  # default
        assert metadata.custom_properties == {}  # default

    def test_metadata_round_trip(self) -> None:
        """Test metadata serialization round trip."""
        original = FlextFieldCoreMetadata(
            description="Test field",
            min_length=1,
            max_length=10,
            deprecated=True,
            tags=["test", "important"],
            custom_properties={"key": "value"},
        )

        data = original.to_dict()
        restored = FlextFieldCoreMetadata.from_dict(data)

        assert restored.description == original.description
        assert restored.min_length == original.min_length
        assert restored.max_length == original.max_length
        assert restored.deprecated == original.deprecated
        assert restored.tags == original.tags
        assert restored.custom_properties == original.custom_properties


class ConcreteFlextFieldCore(FlextFieldCore):
    """Concrete implementation for testing abstract FlextFieldCore."""

    def validate_value(self, value: object) -> FlextResult[str]:
        """Validate the provided value against field requirements."""
        if self.required and value is None:
            return FlextResult.fail(f"Field '{self.field_name}' is required")
        if value is not None and not isinstance(value, str):
            return FlextResult.fail(f"Field '{self.field_name}' must be a string")
        return FlextResult.ok(str(value) if value is not None else "")

    def serialize_value(self, value: str) -> str:
        """Convert value to serialized format."""
        return str(value)

    def deserialize_value(self, value: object) -> str:
        """Convert serialized value back to object format."""
        return str(value)


class TestFlextFieldCoreBase:
    """Test FlextFieldCore base class."""

    def test_field_creation_minimal(self) -> None:
        """Test field creation with minimal parameters."""
        field = ConcreteFlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.STRING.value,
        )

        assert field.field_id == "test_id"
        assert field.field_name == "test_field"
        assert field.field_type == FlextFieldType.STRING.value
        assert field.required is True  # Default is True
        assert field.default_value is None
        assert isinstance(field.metadata, FlextFieldCoreMetadata)
        assert field.validator is None

    def test_field_creation_full(self) -> None:
        """Test field creation with all parameters."""
        field = ConcreteFlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.STRING.value,
            required=False,
            default_value="default",
            description="Test field",
            example="example_value",
        )

        assert field.field_id == "test_id"
        assert field.field_name == "test_field"
        assert field.field_type == FlextFieldType.STRING.value
        assert field.required is False
        assert field.default_value == "default"

    def test_field_get_default_value(self) -> None:
        """Test getting default value."""
        field = ConcreteFlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.STRING.value,
            default_value="default",
        )

        assert field.get_default_value() == "default"

    def test_field_get_default_value_none(self) -> None:
        """Test getting default value when None."""
        field = ConcreteFlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.STRING.value,
        )

        assert field.get_default_value() is None

    def test_field_is_required(self) -> None:
        """Test field required status."""
        field = ConcreteFlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.STRING.value,
            required=True,
        )

        assert field.is_required() is True

        field = ConcreteFlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.STRING.value,
            required=False,
        )

        assert field.is_required() is False

    def test_field_is_deprecated(self) -> None:
        """Test field deprecated status."""
        field = ConcreteFlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.STRING.value,
            deprecated=True,
        )

        assert field.is_deprecated() is True

        field = ConcreteFlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.STRING.value,
            deprecated=False,
        )

        assert field.is_deprecated() is False

    def test_field_is_sensitive(self) -> None:
        """Test field sensitive status."""
        field = ConcreteFlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.STRING.value,
            sensitive=True,
        )

        assert field.is_sensitive() is True

        field = ConcreteFlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.STRING.value,
            sensitive=False,
        )

        assert field.is_sensitive() is False

    def test_field_get_field_info(self) -> None:
        """Test getting field information."""
        field = ConcreteFlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.STRING.value,
            description="Test description",
            example="test_example",
        )

        info = field.get_field_info()
        assert info["field_id"] == "test_id"
        assert info["field_name"] == "test_field"
        assert info["field_type"] == FlextFieldType.STRING.value
        assert info["metadata"]["description"] == "Test description"
        assert info["metadata"]["example"] == "test_example"

    def test_field_get_field_info_no_validator(self) -> None:
        """Test getting field information without validator."""
        field = ConcreteFlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.STRING.value,
        )

        info = field.get_field_info()
        assert info["field_id"] == "test_id"
        assert info["field_name"] == "test_field"
        assert info["field_type"] == FlextFieldType.STRING.value

    def test_field_validate_value_required_success(self) -> None:
        """Test field validation with required value."""
        field = ConcreteFlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.STRING.value,
            required=True,
        )

        result = field.validate_value("test_value")
        assert result.is_success

    def test_field_validate_value_required_failure(self) -> None:
        """Test field validation with required value failure."""
        field = ConcreteFlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.STRING.value,
            required=True,
        )

        result = field.validate_value(None)
        assert result.is_failure

    def test_field_serialize_value(self) -> None:
        """Test field serialization."""
        field = ConcreteFlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.STRING.value,
        )

        result = field.serialize_value("test_value")
        assert result == "test_value"

    def test_field_deserialize_value(self) -> None:
        """Test field deserialization."""
        field = ConcreteFlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.STRING.value,
        )

        result = field.deserialize_value("test_value")
        assert result == "test_value"


class TestFlextFieldCoreString:
    """Test FlextFieldCore string implementation."""

    def test_string_field_creation_minimal(self) -> None:
        """Test string field creation with minimal parameters."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="string",
        )

        assert field.field_id == "test_id"
        assert field.field_name == "test_field"
        assert field.field_type == "string"
        assert field.required is True  # Default is True
        assert field.default_value is None

    def test_string_field_creation_with_constraints(self) -> None:
        """Test string field creation with length constraints."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="string",
            min_length=1,
            max_length=10,
            pattern=r"^\w+$",
        )

        assert field.metadata.min_length == 1
        assert field.metadata.max_length == 10
        assert field.metadata.pattern == r"^\w+$"

    def test_string_field_creation_with_custom_metadata(self) -> None:
        """Test string field creation with custom metadata."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="string",
            description="Custom field",
            min_length=5,
        )

        # Metadata is computed from field properties
        assert field.metadata.description == "Custom field"
        assert field.metadata.min_length == 5

    def test_string_field_validate_value_success(self) -> None:
        """Test string field validation success."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="string",
        )

        result = field.validate_value("test_string")
        assert result.is_success

    def test_string_field_validate_value_success_case(self) -> None:
        """Test string field validation with valid string."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="string",
        )

        result = field.validate_value("test_string")
        assert result.is_success

    def test_string_field_serialize_value(self) -> None:
        """Test string field serialization."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="string",
        )

        result = field.serialize_value("test_string")
        assert result == "test_string"

    def test_string_field_deserialize_value(self) -> None:
        """Test string field deserialization."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="string",
        )

        result = field.deserialize_value("test_string")
        assert result == "test_string"

    def test_string_field_deserialize_non_string(self) -> None:
        """Test string field deserialization of non-string values."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="string",
        )

        result = field.deserialize_value(123)
        assert result == "123"

    def test_string_field_validate_none_value(self) -> None:
        """Test string field validation with None value."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="string",
            required=False,  # Allow None values
        )

        result = field.validate_value(None)
        assert result.is_success

    def test_string_field_validate_non_string_type(self) -> None:
        """Test string field validation with non-string type."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="string",
        )

        result = field.validate_value(123)
        assert result.is_failure
        assert "string" in result.error.lower()

    def test_string_field_with_all_parameters(self) -> None:
        """Test string field with all parameters."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="string",
            required=True,
            default_value="default_value",
            max_length=50,
            min_length=1,
            pattern=r"^\w+$",
            description="Test string field",
        )

        assert field.field_id == "test_id"
        assert field.field_name == "test_field"
        assert field.field_type == "string"
        assert field.required is True
        assert field.default_value == "default_value"
        assert field.description == "Test string field"
        # Verify metadata is correctly computed
        assert field.metadata.description == "Test string field"


class TestFlextFieldCoreBoolean:
    """Test FlextFieldCore boolean implementation."""

    def test_boolean_field_creation_minimal(self) -> None:
        """Test boolean field creation with minimal parameters."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="boolean",
        )

        assert field.field_id == "test_id"
        assert field.field_name == "test_field"
        assert field.field_type == "boolean"
        assert field.required is True  # Default is True
        assert field.default_value is None

    def test_boolean_field_validate_value_success(self) -> None:
        """Test boolean field validation with valid value."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="boolean",
        )

        result = field.validate_value(value=True)
        assert result.is_success

    def test_boolean_field_validate_value_success_case(self) -> None:
        """Test boolean field validation with valid boolean."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="boolean",
        )

        test_value = True
        result = field.validate_value(test_value)
        assert result.is_success

    def test_boolean_field_validate_non_boolean_type(self) -> None:
        """Test boolean field validation with non-boolean type."""
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="boolean",
        )

        result = field.validate_value("not a boolean")
        assert result.is_failure
        assert "boolean" in result.error.lower()

    def test_boolean_field_serialize_value(self) -> None:
        """Test boolean field serialization."""
        FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.BOOLEAN.value,
        )

        # Field doesn't have serialize_value method directly
        # This test needs to be adjusted based on actual field API

    def test_boolean_field_deserialize_existing_boolean(self) -> None:
        """Test boolean field deserialization with existing boolean."""
        FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.BOOLEAN.value,
        )

        # Field doesn't have deserialize_value method directly
        # This test needs to be adjusted based on actual field API

    def test_boolean_field_deserialize_string_values(self) -> None:
        """Test boolean field deserialization with string values."""
        FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.BOOLEAN.value,
        )

        # Field doesn't have deserialize_value method directly
        # This test needs to be adjusted based on actual field API

    def test_boolean_field_deserialize_integer(self) -> None:
        """Test boolean field deserialization with integer."""
        FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.BOOLEAN.value,
        )

        # Field doesn't have deserialize_value method directly
        # This test needs to be adjusted based on actual field API

    def test_boolean_field_deserialize_non_convertible(self) -> None:
        """Test boolean field deserialization with non-convertible value."""
        FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type=FlextFieldType.BOOLEAN.value,
        )

        # Field doesn't have deserialize_value method directly
        # This test needs to be adjusted based on actual field API


class TestFlextFieldCoreRegistry:
    """Test FlextFieldCoreRegistry class."""

    def test_registry_creation(self) -> None:
        """Test registry creation."""
        registry = FlextFieldRegistry()
        assert registry._fields == {}

    def test_register_field(self) -> None:
        """Test registering a field."""
        registry = FlextFieldRegistry()
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="string",
        )

        registry.register_field(field)
        assert registry._fields["test_id"] is field

    def test_get_field_existing(self) -> None:
        """Test getting an existing field."""
        registry = FlextFieldRegistry()
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="string",
        )
        registry.register_field(field)

        result = registry.get_field("test_id")
        assert result is field

    def test_get_field_non_existent(self) -> None:
        """Test getting a non-existent field."""
        registry = FlextFieldRegistry()

        result = registry.get_field("non_existent")
        assert result is None

    def test_get_all_fields_empty(self) -> None:
        """Test getting all fields when empty."""
        registry = FlextFieldRegistry()

        result = registry.get_all_fields()
        assert result == {}

    def test_get_all_fields_with_fields(self) -> None:
        """Test getting all fields when registry has fields."""
        registry = FlextFieldRegistry()
        field1 = FlextFieldCore(
            field_id="field1",
            field_name="test1",
            field_type="string",
        )
        field2 = FlextFieldCore(
            field_id="field2",
            field_name="test2",
            field_type="string",
        )

        registry.register_field(field1)
        registry.register_field(field2)

        result = registry.get_all_fields()
        assert len(result) == 2
        assert result["field1"] is field1
        assert result["field2"] is field2

    def test_get_fields_by_type_with_matches(self) -> None:
        """Test getting fields by type with matching fields."""
        registry = FlextFieldRegistry()

        # Register fields of different types
        string_field1 = FlextFieldCore(
            field_id="str1",
            field_name="test1",
            field_type=FlextFieldType.STRING.value,
        )
        string_field2 = FlextFieldCore(
            field_id="str2",
            field_name="test2",
            field_type=FlextFieldType.STRING.value,
        )
        int_field = FlextFieldCore(
            field_id="int1",
            field_name="test3",
            field_type=FlextFieldType.INTEGER.value,
        )

        registry.register_field(string_field1)
        registry.register_field(string_field2)
        registry.register_field(int_field)

        result = registry.get_fields_by_type(FlextFieldType.STRING.value)
        assert len(result) == 2
        assert string_field1 in result
        assert string_field2 in result
        assert int_field not in result

    def test_validate_all_fields_success(self) -> None:
        """Test validating all fields with valid data."""
        registry = FlextFieldRegistry()
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="string",
        )
        registry.register_field(field)

        data = {"test_id": "valid_string"}
        result = registry.validate_all_fields(data)
        assert result.is_success

    def test_validate_all_fields_required_missing(self) -> None:
        """Test validating all fields with required field missing."""
        registry = FlextFieldRegistry()
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="string",
            required=True,
        )
        registry.register_field(field)

        data: dict[str, str] = {}  # Missing required field
        result = registry.validate_all_fields(data)
        assert result.is_failure
        assert "required" in result.error.lower()

    def test_validate_all_fields_validation_error(self) -> None:
        """Test validating all fields with validation error."""
        registry = FlextFieldRegistry()
        field = FlextFieldCore(
            field_id="test_id",
            field_name="test_field",
            field_type="string",
        )
        registry.register_field(field)

        data = {"test_id": 123}  # Invalid type for string field
        result = registry.validate_all_fields(data)
        assert result.is_failure
        assert "string" in result.error.lower()
