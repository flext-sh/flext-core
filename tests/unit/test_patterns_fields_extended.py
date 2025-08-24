"""Extended tests for FlextFieldCore factory methods and FlextFieldRegistry."""


from __future__ import annotations

from flext_core import (
    FlextFieldRegistry,
    FlextFields,
    FlextFieldType,
)

# Constants
EXPECTED_BULK_SIZE = 2


class TestFlextFieldsFactory:
    """Test FlextFields factory methods."""

    def test_create_integer_field_minimal(self) -> None:
        """Test integer field creation with minimal parameters."""
        field = FlextFields.create_integer_field(
            field_id="test_id",
            field_name="test_field",
        )

        if field.field_id != "test_id":
            raise AssertionError(f"Expected test_id, got {field.field_id}")
        assert field.field_name == "test_field"
        if field.field_type != FlextFieldType.INTEGER.value:
            raise AssertionError(
                f"Expected {FlextFieldType.INTEGER.value}, got {field.field_type}",
            )
        if field.required is not True:  # Default
            raise AssertionError(
                f"Expected True, got {field.required is True}",
            )  # Default
        assert field.default_value is None

    def test_create_integer_field_with_constraints(self) -> None:
        """Test integer field creation with value constraints."""
        field = FlextFields.create_integer_field(
            field_id="test_id",
            field_name="test_field",
            min_value=0,
            max_value=100,
        )

        if field.min_value != 0:
            raise AssertionError(f"Expected 0, got {field.min_value}")
        assert field.max_value == 100

    def test_create_integer_field_with_description(self) -> None:
        """Test integer field creation with description."""
        field = FlextFields.create_integer_field(
            field_id="test_id",
            field_name="test_field",
            description="Custom integer field",
            min_value=10,
        )

        if field.description != "Custom integer field":
            raise AssertionError(
                f"Expected Custom integer field, got {field.description}",
            )
        assert field.min_value == 10

    def test_integer_field_validate_value_success(self) -> None:
        """Test integer field validation success."""
        field = FlextFields.create_integer_field(
            field_id="test_id",
            field_name="test_field",
        )

        result = field.validate_value(42)
        assert result.success

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
        if schema["field_id"] != "test_id":
            raise AssertionError(f"Expected test_id, got {schema['field_id']}")
        assert schema["field_name"] == "test_field"
        if schema["field_type"] != FlextFieldType.INTEGER.value:
            raise AssertionError(
                f"Expected {FlextFieldType.INTEGER.value}, got {schema['field_type']}",
            )
        assert schema["min_value"] == 0
        if schema["max_value"] != 100:
            raise AssertionError(f"Expected 100, got {schema['max_value']}")

    def test_create_boolean_field_minimal(self) -> None:
        """Test boolean field creation with minimal parameters."""
        field = FlextFields.create_boolean_field(
            field_id="test_id",
            field_name="test_field",
        )

        if field.field_id != "test_id":
            raise AssertionError(f"Expected test_id, got {field.field_id}")
        assert field.field_name == "test_field"
        if field.field_type != FlextFieldType.BOOLEAN.value:
            raise AssertionError(
                f"Expected {FlextFieldType.BOOLEAN.value}, got {field.field_type}",
            )
        if field.required is not True:  # Default
            raise AssertionError(
                f"Expected True, got {field.required is True}",
            )  # Default
        assert field.default_value is None

    def test_create_boolean_field_with_description(self) -> None:
        """Test boolean field creation with description."""
        field = FlextFields.create_boolean_field(
            field_id="test_id",
            field_name="test_field",
            description="Custom boolean field",
        )

        if field.description != "Custom boolean field":
            raise AssertionError(
                f"Expected Custom boolean field, got {field.description}",
            )

    def test_boolean_field_validate_value_success(self) -> None:
        """Test boolean field validation success."""
        field = FlextFields.create_boolean_field(
            field_id="test_id",
            field_name="test_field",
        )

        result = field.validate_value(True)
        assert result.success

        result = field.validate_value(False)
        assert result.success

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

        if field.field_id != "test_id":
            raise AssertionError(f"Expected test_id, got {field.field_id}")
        assert field.field_name == "test_field"
        if field.field_type != FlextFieldType.STRING.value:
            raise AssertionError(
                f"Expected {FlextFieldType.STRING.value}, got {field.field_type}",
            )
        if field.required is not True:  # Default
            raise AssertionError(
                f"Expected True, got {field.required is True}",
            )  # Default
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

        if field.pattern != r"^[A-Z]+$":
            raise AssertionError(f"Expected {r'^[A-Z]+$'}, got {field.pattern}")
        assert field.min_length == 1
        if field.max_length != 10:
            raise AssertionError(f"Expected 10, got {field.max_length}")


class TestFlextFieldRegistry:
    """Test FlextFieldRegistry implementation."""

    def test_registry_creation(self) -> None:
        """Test registry creation."""
        registry = FlextFieldRegistry()
        assert isinstance(registry, FlextFieldRegistry)
        if registry.get_field_count() != 0:
            raise AssertionError(f"Expected 0, got {registry.get_field_count()}")

    def test_register_field(self) -> None:
        """Test registering a field."""
        registry = FlextFieldRegistry()
        field = FlextFields.create_string_field(
            field_id="test_id",
            field_name="test_field",
        )

        result = registry.register_field(field)
        assert result.success

        retrieved_result = registry.get_field_by_id("test_id")
        assert retrieved_result.success
        assert retrieved_result.value is field

    def test_get_field_existing(self) -> None:
        """Test getting an existing field."""
        registry = FlextFieldRegistry()
        field = FlextFields.create_string_field(
            field_id="test_id",
            field_name="test_field",
        )
        registry.register_field(field)

        result = registry.get_field_by_id("test_id")
        assert result.success
        assert result.value is field

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
        if len(field_names) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected 2, got {len(field_names)}")
        if "Field 1" not in field_names:
            raise AssertionError(f"Expected Field 1 in {field_names}")
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
        if len(field_ids) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected 2, got {len(field_ids)}")
        if "field1" not in field_ids:
            raise AssertionError(f"Expected field1 in {field_ids}")
        assert "field2" in field_ids

    def test_clear_registry(self) -> None:
        """Test clearing the registry."""
        registry = FlextFieldRegistry()
        field = FlextFields.create_string_field(
            field_id="test_id",
            field_name="test_field",
        )

        registry.register_field(field)
        if registry.get_field_count() != 1:
            raise AssertionError(f"Expected 1, got {registry.get_field_count()}")

        registry.clear_registry()
        if registry.get_field_count() != 0:
            raise AssertionError(f"Expected 0, got {registry.get_field_count()}")

    def test_remove_field(self) -> None:
        """Test removing a field from registry."""
        registry = FlextFieldRegistry()
        field = FlextFields.create_string_field(
            field_id="test_id",
            field_name="test_field",
        )

        registry.register_field(field)
        if registry.get_field_count() != 1:
            raise AssertionError(f"Expected 1, got {registry.get_field_count()}")

        result = registry.remove_field("test_id")
        if not result:
            raise AssertionError(f"Expected True, got {result}")
        if registry.get_field_count() != 0:
            raise AssertionError(f"Expected 0, got {registry.get_field_count()}")

        # Try to remove non-existing field
        result = registry.remove_field("non_existing")
        if result:
            raise AssertionError(f"Expected False, got {result}")
