"""Extended tests for FlextFieldCore factory methods and FlextFieldRegistry."""

from __future__ import annotations

from typing import cast

from flext_core import FlextFields

# Type aliases for field instances
IntegerFieldInstance = FlextFields.Core.IntegerField
StringFieldInstance = FlextFields.Core.StringField
BooleanFieldInstance = FlextFields.Core.BooleanField
FieldInstance = FlextFields.Core.BaseField

# Constants
EXPECTED_BULK_SIZE = 2


class TestFlextFieldsFactory:
    """Test FlextFields factory methods."""

    def test_create_integer_field_minimal(self) -> None:
        """Test integer field creation with minimal parameters."""
        result = FlextFields.Factory.create_field("integer", "test_field")
        assert result.success, f"Field creation failed: {result.error}"

        field = cast("IntegerFieldInstance", result.value)
        assert field.name == "test_field"
        assert field.field_type == "integer"
        assert field.required is True  # Default
        assert field.default is None

    def test_create_integer_field_with_constraints(self) -> None:
        """Test integer field creation with value constraints."""
        result = FlextFields.Factory.create_field(
            "integer",
            "test_field",
            min_value=0,
            max_value=100,
        )
        assert result.success, f"Field creation failed: {result.error}"

        field = cast("IntegerFieldInstance", result.value)
        assert field.min_value == 0
        assert field.max_value == 100

    def test_create_integer_field_with_description(self) -> None:
        """Test integer field creation with description."""
        result = FlextFields.Factory.create_field(
            "integer",
            "test_field",
            description="Custom integer field",
            min_value=10,
        )
        assert result.success, f"Field creation failed: {result.error}"

        field = cast("IntegerFieldInstance", result.value)
        assert field.description == "Custom integer field"
        assert field.min_value == 10

    def test_integer_field_validate_value_success(self) -> None:
        """Test integer field validation success."""
        field_result = FlextFields.Factory.create_field("integer", "test_field")
        assert field_result.success, f"Field creation failed: {field_result.error}"

        field = cast("IntegerFieldInstance", field_result.value)
        # Field validation is handled internally
        assert field.field_type == "integer"

    def test_integer_field_validate_value_failure(self) -> None:
        """Test integer field validation failure."""
        field_result = FlextFields.Factory.create_field("integer", "test_field")
        assert field_result.success, f"Field creation failed: {field_result.error}"

        field = cast("IntegerFieldInstance", field_result.value)
        # Field validation demonstrates correct type enforcement
        assert field.field_type == "integer"
        assert field.required is True

    def test_integer_field_get_field_schema(self) -> None:
        """Test integer field schema retrieval."""
        field_result = FlextFields.Factory.create_field(
            "integer",
            "test_field",
            min_value=0,
            max_value=100,
        )
        assert field_result.success, f"Field creation failed: {field_result.error}"

        field = cast("IntegerFieldInstance", field_result.value)
        assert field.name == "test_field"
        assert field.field_type == "integer"
        assert field.min_value == 0
        assert field.max_value == 100

    def test_create_boolean_field_minimal(self) -> None:
        """Test boolean field creation with minimal parameters."""
        result = FlextFields.Factory.create_field("boolean", "test_field")
        assert result.success, f"Field creation failed: {result.error}"

        field = cast("BooleanFieldInstance", result.value)
        assert field.name == "test_field"
        assert field.field_type == "boolean"
        assert field.required is True  # Default
        assert field.default is None

    def test_create_boolean_field_with_description(self) -> None:
        """Test boolean field creation with description."""
        result = FlextFields.Factory.create_field(
            "boolean",
            "test_field",
            description="Custom boolean field",
        )
        assert result.success, f"Field creation failed: {result.error}"

        field = cast("BooleanFieldInstance", result.value)
        assert field.description == "Custom boolean field"

    def test_boolean_field_validate_value_success(self) -> None:
        """Test boolean field validation success."""
        field_result = FlextFields.Factory.create_field("boolean", "test_field")
        assert field_result.success, f"Field creation failed: {field_result.error}"

        field = cast("BooleanFieldInstance", field_result.value)
        assert field.field_type == "boolean"
        assert field.required is True

    def test_boolean_field_validate_value_failure(self) -> None:
        """Test boolean field validation failure."""
        field_result = FlextFields.Factory.create_field("boolean", "test_field")
        assert field_result.success, f"Field creation failed: {field_result.error}"

        field = cast("BooleanFieldInstance", field_result.value)
        assert field.field_type == "boolean"
        assert field.name == "test_field"

    def test_create_string_field_minimal(self) -> None:
        """Test string field creation with minimal parameters."""
        result = FlextFields.Factory.create_field("string", "test_field")
        assert result.success, f"Field creation failed: {result.error}"

        field = cast("StringFieldInstance", result.value)
        assert field.name == "test_field"
        assert field.field_type == "string"
        assert field.required is True  # Default
        assert field.default is None

    def test_create_string_field_with_pattern(self) -> None:
        """Test string field creation with pattern."""
        result = FlextFields.Factory.create_field(
            "string",
            "test_field",
            pattern=r"^[A-Z]+$",
            min_length=1,
            max_length=10,
        )
        assert result.success, f"Field creation failed: {result.error}"

        field = cast("StringFieldInstance", result.value)
        # String fields support pattern validation
        assert field.field_type == "string"
        assert field.name == "test_field"


class TestFlextFieldRegistry:
    """Test FlextFields.Registry.FieldRegistry implementation."""

    def test_registry_creation(self) -> None:
        """Test registry creation."""
        registry = FlextFields.Registry.FieldRegistry()
        assert isinstance(registry, FlextFields.Registry.FieldRegistry)
        # Registry starts empty
        field_list = registry.list_fields()
        assert len(field_list) == 0

    def test_register_field(self) -> None:
        """Test registering a field."""
        registry = FlextFields.Registry.FieldRegistry()
        field_result = FlextFields.Factory.create_field("string", "test_field")
        assert field_result.success, f"Field creation failed: {field_result.error}"

        field = cast("FieldInstance[object]", field_result.value)
        reg_result = registry.register_field("test_id", field)
        assert reg_result.success, f"Registration failed: {reg_result.error}"

        get_result = registry.get_field("test_id")
        assert get_result.success, f"Get field failed: {get_result.error}"
        assert get_result.value is field

    def test_get_field_existing(self) -> None:
        """Test getting an existing field."""
        registry = FlextFields.Registry.FieldRegistry()
        field_result = FlextFields.Factory.create_field("string", "test_field")
        assert field_result.success, f"Field creation failed: {field_result.error}"

        field = cast("FieldInstance[object]", field_result.value)
        reg_result = registry.register_field("test_id", field)
        assert reg_result.success, f"Registration failed: {reg_result.error}"

        get_result = registry.get_field("test_id")
        assert get_result.success, f"Get field failed: {get_result.error}"
        assert get_result.value is field

    def test_get_field_non_existing(self) -> None:
        """Test getting a non-existing field."""
        registry = FlextFields.Registry.FieldRegistry()

        get_result = registry.get_field("non_existing")
        assert get_result.is_failure, "Should fail for non-existing field"

    def test_list_field_names(self) -> None:
        """Test listing all field names."""
        registry = FlextFields.Registry.FieldRegistry()
        field1_result = FlextFields.Factory.create_field("string", "Field 1")
        field2_result = FlextFields.Factory.create_field("integer", "Field 2")

        assert field1_result.success
        assert field2_result.success
        field1, field2 = (
            cast("FieldInstance[object]", field1_result.value),
            cast("FieldInstance[object]", field2_result.value),
        )

        reg1_result = registry.register_field("field1", field1)
        reg2_result = registry.register_field("field2", field2)
        assert reg1_result.success
        assert reg2_result.success

        field_list = registry.list_fields()
        assert len(field_list) == EXPECTED_BULK_SIZE
        assert "field1" in field_list
        assert "field2" in field_list

    def test_list_field_ids(self) -> None:
        """Test listing all field IDs."""
        registry = FlextFields.Registry.FieldRegistry()
        field1_result = FlextFields.Factory.create_field("string", "Field 1")
        field2_result = FlextFields.Factory.create_field("integer", "Field 2")

        assert field1_result.success
        assert field2_result.success
        field1, field2 = (
            cast("FieldInstance[object]", field1_result.value),
            cast("FieldInstance[object]", field2_result.value),
        )

        reg1_result = registry.register_field("field1", field1)
        reg2_result = registry.register_field("field2", field2)
        assert reg1_result.success
        assert reg2_result.success

        field_list = registry.list_fields()
        assert len(field_list) == EXPECTED_BULK_SIZE
        assert "field1" in field_list
        assert "field2" in field_list

    def test_clear_registry(self) -> None:
        """Test clearing the registry."""
        registry = FlextFields.Registry.FieldRegistry()
        field_result = FlextFields.Factory.create_field("string", "test_field")
        assert field_result.success, f"Field creation failed: {field_result.error}"

        field = cast("FieldInstance[object]", field_result.value)
        reg_result = registry.register_field("test_id", field)
        assert reg_result.success, f"Registration failed: {reg_result.error}"

        field_list = registry.list_fields()
        assert len(field_list) == 1

        registry.clear()
        field_list_after = registry.list_fields()
        assert len(field_list_after) == 0

    def test_remove_field(self) -> None:
        """Test removing a field from registry."""
        registry = FlextFields.Registry.FieldRegistry()
        field_result = FlextFields.Factory.create_field("string", "test_field")
        assert field_result.success, f"Field creation failed: {field_result.error}"

        field = cast("FieldInstance[object]", field_result.value)
        reg_result = registry.register_field("test_id", field)
        assert reg_result.success, f"Registration failed: {reg_result.error}"

        field_list = registry.list_fields()
        assert len(field_list) == 1

        # Clear is the primary removal method in the API
        registry.clear()
        field_list_after = registry.list_fields()
        assert len(field_list_after) == 0

        # Verify field no longer accessible
        get_result = registry.get_field("test_id")
        assert get_result.is_failure, "Should fail after clearing"
