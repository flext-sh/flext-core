"""Extended tests for FlextFieldCore factory methods and FlextFieldRegistry."""

from __future__ import annotations

from flext_core.constants import FlextFieldType
from flext_core.fields import (
    FlextFieldRegistry,
    FlextFields,
)


class TestFlextFieldsFactory:
    """Test FlextFields factory methods."""

    def test_create_integer_field_minimal(self) -> None:
        """Test integer field creation with minimal parameters."""
        field = FlextFields.create_integer_field(
            field_id="test_id",
            field_name="test_field",
        )

        assert field.field_id == "test_id"
        assert field.field_name == "test_field"
        assert field.field_type == FlextFieldType.INTEGER.value
        assert field.required is True  # Default is True
        assert field.default_value is None

    def test_create_integer_field_with_constraints(self) -> None:
        """Test integer field creation with value constraints."""
        field = FlextFields.create_integer_field(
            field_id="test_id",
            field_name="test_field",
            min_value=0,
            max_value=100,
        )

        assert field.min_value == 0
        assert field.max_value == 100

    def test_create_integer_field_with_description(self) -> None:
        """Test integer field creation with description."""
        field = FlextFields.create_integer_field(
            field_id="test_id",
            field_name="test_field",
            description="Custom integer field",
            min_value=10,
        )

        assert field.description == "Custom integer field"
        assert field.min_value == 10

    def test_integer_field_validate_value_success(self) -> None:
        """Test integer field validation success."""
        field = FlextFields.create_integer_field(
            field_id="test_id",
            field_name="test_field",
        )

        result = field.validate_value(42)
        assert result.is_success

    def test_integer_field_validate_value_failure(self) -> None:
        """Test integer field validation failure."""
        field = FlextFields.create_integer_field(
            field_id="test_id",
            field_name="test_field",
        )

        result = field.validate_value("not_an_integer")
        assert result.is_failure
        assert result.error is not None

    def test_integer_field_get_field_schema(self) -> None:
        """Test integer field schema retrieval."""
        field = FlextFields.create_integer_field(
            field_id="test_id",
            field_name="test_field",
            min_value=0,
            max_value=100,
        )

        schema = field.get_field_schema()
        assert schema["field_id"] == "test_id"
        assert schema["field_name"] == "test_field"
        assert schema["field_type"] == FlextFieldType.INTEGER.value
        assert schema["min_value"] == 0
        assert schema["max_value"] == 100

    def test_create_boolean_field_minimal(self) -> None:
        """Test boolean field creation with minimal parameters."""
        field = FlextFields.create_boolean_field(
            field_id="test_id",
            field_name="test_field",
        )

        assert field.field_id == "test_id"
        assert field.field_name == "test_field"
        assert field.field_type == FlextFieldType.BOOLEAN.value
        assert field.required is True  # Default is True
        assert field.default_value is None

    def test_create_boolean_field_with_description(self) -> None:
        """Test boolean field creation with description."""
        field = FlextFields.create_boolean_field(
            field_id="test_id",
            field_name="test_field",
            description="Custom boolean field",
        )

        assert field.description == "Custom boolean field"

    def test_boolean_field_validate_value_success(self) -> None:
        """Test boolean field validation success."""
        field = FlextFields.create_boolean_field(
            field_id="test_id",
            field_name="test_field",
        )

        result = field.validate_value(value=True)
        assert result.is_success

        result = field.validate_value(value=False)
        assert result.is_success

    def test_boolean_field_validate_value_failure(self) -> None:
        """Test boolean field validation failure."""
        field = FlextFields.create_boolean_field(
            field_id="test_id",
            field_name="test_field",
        )

        result = field.validate_value("not_a_boolean")
        assert result.is_failure
        assert result.error is not None

    def test_create_string_field_minimal(self) -> None:
        """Test string field creation with minimal parameters."""
        field = FlextFields.create_string_field(
            field_id="test_id",
            field_name="test_field",
        )

        assert field.field_id == "test_id"
        assert field.field_name == "test_field"
        assert field.field_type == FlextFieldType.STRING.value
        assert field.required is True  # Default is True
        assert field.default_value is None

    def test_create_string_field_with_pattern(self) -> None:
        """Test string field creation with pattern."""
        field = FlextFields.create_string_field(
            field_id="test_id",
            field_name="test_field",
            pattern=r"^[A-Z]+$",
            min_length=1,
            max_length=10,
        )

        assert field.pattern == r"^[A-Z]+$"
        assert field.min_length == 1
        assert field.max_length == 10


class TestFlextFieldRegistry:
    """Test FlextFieldRegistry implementation."""

    def test_registry_creation(self) -> None:
        """Test registry creation."""
        registry = FlextFieldRegistry()
        assert isinstance(registry, FlextFieldRegistry)
        assert registry.get_field_count() == 0

    def test_register_field(self) -> None:
        """Test registering a field."""
        registry = FlextFieldRegistry()
        field = FlextFields.create_string_field(
            field_id="test_id",
            field_name="test_field",
        )

        result = registry.register_field(field)
        assert result.is_success

        retrieved_result = registry.get_field_by_id("test_id")
        assert retrieved_result.is_success
        assert retrieved_result.data is field

    def test_get_field_existing(self) -> None:
        """Test getting an existing field."""
        registry = FlextFieldRegistry()
        field = FlextFields.create_string_field(
            field_id="test_id",
            field_name="test_field",
        )
        registry.register_field(field)

        result = registry.get_field_by_id("test_id")
        assert result.is_success
        assert result.data is field

    def test_get_field_non_existing(self) -> None:
        """Test getting a non-existing field."""
        registry = FlextFieldRegistry()

        result = registry.get_field_by_id("non_existing")
        assert result.is_failure

    def test_list_field_names(self) -> None:
        """Test listing all field names."""
        registry = FlextFieldRegistry()
        field1 = FlextFields.create_string_field(
            field_id="field1",
            field_name="Field 1",
        )
        field2 = FlextFields.create_integer_field(
            field_id="field2",
            field_name="Field 2",
        )

        registry.register_field(field1)
        registry.register_field(field2)

        field_names = registry.list_field_names()
        assert len(field_names) == 2
        assert "Field 1" in field_names
        assert "Field 2" in field_names

    def test_list_field_ids(self) -> None:
        """Test listing all field IDs."""
        registry = FlextFieldRegistry()
        field1 = FlextFields.create_string_field(
            field_id="field1",
            field_name="Field 1",
        )
        field2 = FlextFields.create_integer_field(
            field_id="field2",
            field_name="Field 2",
        )

        registry.register_field(field1)
        registry.register_field(field2)

        field_ids = registry.list_field_ids()
        assert len(field_ids) == 2
        assert "field1" in field_ids
        assert "field2" in field_ids

    def test_clear_registry(self) -> None:
        """Test clearing the registry."""
        registry = FlextFieldRegistry()
        field = FlextFields.create_string_field(
            field_id="test_id",
            field_name="test_field",
        )

        registry.register_field(field)
        assert registry.get_field_count() == 1

        registry.clear_registry()
        assert registry.get_field_count() == 0

    def test_remove_field(self) -> None:
        """Test removing a field from registry."""
        registry = FlextFieldRegistry()
        field = FlextFields.create_string_field(
            field_id="test_id",
            field_name="test_field",
        )

        registry.register_field(field)
        assert registry.get_field_count() == 1

        result = registry.remove_field("test_id")
        assert result is True
        assert registry.get_field_count() == 0

        # Try to remove non-existing field
        result = registry.remove_field("non_existing")
        assert result is False
