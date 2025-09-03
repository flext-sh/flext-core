"""Extended test coverage for fields.py module."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from hypothesis import given, strategies as st

from flext_core import FlextFields
from flext_core.constants import FlextConstants

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextFieldsCore:
    """Test FlextFields.Core functionality."""

    def test_base_field_creation_with_validation(self) -> None:
        """Test BaseField creation with validation."""
        with pytest.raises(ValueError):
            FlextFields.Core.StringField("")  # Empty name

        with pytest.raises(ValueError):
            FlextFields.Core.StringField("   ")  # Whitespace only

    def test_string_field_comprehensive(self) -> None:
        """Test StringField comprehensively."""
        field = FlextFields.Core.StringField(
            "test_field",
            min_length=3,
            max_length=10,
            pattern=r"^[a-z]+$",
            default="test",
        )

        assert field.name == "test_field"
        assert field.min_length == 3
        assert field.max_length == 10
        assert field.pattern is not None

        # Valid string
        result = field.validate("hello")
        assert result.success
        assert result.value == "hello"

        # Too short
        result = field.validate("ab")
        assert result.failure
        assert result.error is not None
        assert "too short" in result.error.lower()

        # Too long
        result = field.validate("verylongstring")
        assert result.failure
        assert result.error is not None
        assert "too long" in result.error.lower()

        # Pattern mismatch
        result = field.validate("Hello123")
        assert result.failure
        assert result.error is not None
        assert "pattern" in result.error.lower()

    def test_integer_field_comprehensive(self) -> None:
        """Test IntegerField comprehensively."""
        field = FlextFields.Core.IntegerField(
            "int_field", min_value=0, max_value=100, default=50
        )

        # Valid integer
        result = field.validate(25)
        assert result.success
        assert result.value == 25

        # Boolean should fail
        result = field.validate(True)
        assert result.failure

        # Too small
        result = field.validate(-5)
        assert result.failure
        assert result.error is not None
        assert "too small" in result.error.lower()

        # Too large
        result = field.validate(150)
        assert result.failure
        assert result.error is not None
        assert "too large" in result.error.lower()

    def test_float_field_comprehensive(self) -> None:
        """Test FloatField comprehensively."""
        field = FlextFields.Core.FloatField(
            "float_field", min_value=0.0, max_value=100.0, precision=2, default=50.5
        )

        # Valid float
        result = field.validate(25.7)
        assert result.success

        # Boolean should fail
        result = field.validate(True)
        assert result.failure

        # Integer should be accepted and converted
        result = field.validate(25)
        assert result.success
        assert result.value == 25.0

        # Test precision
        result = field.validate(25.123456)
        assert result.success
        assert result.value == 25.12

    def test_boolean_field_comprehensive(self) -> None:
        """Test BooleanField comprehensively."""
        field = FlextFields.Core.BooleanField("bool_field", default=False)

        # Direct boolean
        result = field.validate(True)
        assert result.success
        assert result.value is True

        # String representations
        true_strings = ["true", "yes", "1", "on", "enabled"]
        for s in true_strings:
            result = field.validate(s)
            assert result.success
            assert result.value is True

        false_strings = ["false", "no", "0", "off", "disabled"]
        for s in false_strings:
            result = field.validate(s)
            assert result.success
            assert result.value is False

        # Numeric values
        result = field.validate(1)
        assert result.success
        assert result.value is True

        result = field.validate(0)
        assert result.success
        assert result.value is False

    def test_email_field_comprehensive(self) -> None:
        """Test EmailField comprehensively."""
        field = FlextFields.Core.EmailField("email_field")

        # Valid email
        result = field.validate("test@example.com")
        assert result.success

        # Invalid email - no @
        result = field.validate("testexample.com")
        assert result.failure

    def test_uuid_field_comprehensive(self) -> None:
        """Test UuidField comprehensively."""
        field = FlextFields.Core.UuidField("uuid_field")

        # Valid UUID
        valid_uuid = str(uuid.uuid4())
        result = field.validate(valid_uuid)
        assert result.success
        assert result.value == valid_uuid

        # Invalid UUID
        result = field.validate("not-a-uuid")
        assert result.failure
        assert "Invalid UUID format" in result.error

        # Not required field with None
        field = FlextFields.Core.UuidField("uuid_field", required=False)
        result = field.validate(None)
        assert result.success
        assert result.value  # Should generate a UUID

    def test_datetime_field_comprehensive(self) -> None:
        """Test DateTimeField comprehensively."""
        now = datetime.now(UTC)
        field = FlextFields.Core.DateTimeField(
            "datetime_field",
            date_format="%Y-%m-%d %H:%M:%S",
            min_date=datetime(2020, 1, 1, tzinfo=UTC),
            max_date=datetime(2030, 12, 31, tzinfo=UTC),
        )

        # Valid datetime object
        result = field.validate(now)
        assert result.success

        # Valid ISO string
        result = field.validate("2025-06-15T10:30:00Z")
        assert result.success

        # Invalid format
        result = field.validate("invalid-date")
        assert result.failure

        # Date too early
        early_date = datetime(2019, 1, 1, tzinfo=UTC)
        result = field.validate(early_date)
        assert result.failure
        assert "too early" in result.error.lower()

    def test_field_metadata(self) -> None:
        """Test field metadata extraction."""
        field = FlextFields.Core.StringField(
            "test_field",
            required=True,
            default="default_value",
            description="Test description",
            min_length=5,
        )

        metadata = field.get_metadata()
        assert metadata["name"] == "test_field"
        assert metadata["type"] == "string"
        assert metadata["required"] is True
        assert metadata["default"] == "default_value"
        assert metadata["description"] == "Test description"

    def test_field_type_property(self) -> None:
        """Test field_type property."""
        string_field = FlextFields.Core.StringField("test")
        assert string_field.field_type == "string"

        integer_field = FlextFields.Core.IntegerField("test")
        assert integer_field.field_type == "integer"

        boolean_field = FlextFields.Core.BooleanField("test")
        assert boolean_field.field_type == "boolean"


class TestFlextFieldsValidation:
    """Test FlextFields.Validation functionality."""

    def test_validate_field(self) -> None:
        """Test validate_field static method."""
        field = FlextFields.Core.StringField("test", min_length=3)

        result = FlextFields.Validation.validate_field(field, "hello")
        assert result.success
        assert result.value == "hello"

        result = FlextFields.Validation.validate_field(field, "hi")
        assert result.failure

    def test_validate_multiple_fields(self) -> None:
        """Test validate_multiple_fields static method."""
        fields = [
            FlextFields.Core.StringField("name", min_length=2),
            FlextFields.Core.IntegerField("age", min_value=0),
            FlextFields.Core.EmailField("email"),
        ]

        values = {"name": "John", "age": 25, "email": "john@example.com"}

        result = FlextFields.Validation.validate_multiple_fields(fields, values)
        assert result.success
        assert result.value["name"] == "John"
        assert result.value["age"] == 25

        # Test with invalid values
        invalid_values = {
            "name": "J",  # Too short
            "age": -5,  # Too small
            "email": "invalid-email",
        }

        result = FlextFields.Validation.validate_multiple_fields(fields, invalid_values)
        assert result.failure
        assert "name:" in result.error
        assert "age:" in result.error


class TestFlextFieldsRegistry:
    """Test FlextFields.Registry functionality."""

    def test_field_registry_basic(self) -> None:
        """Test basic FieldRegistry operations."""
        registry = FlextFields.Registry.FieldRegistry()
        field = FlextFields.Core.StringField("test_field")

        # Register field
        result = registry.register_field("test", field)
        assert result.success

        # Get field
        result = registry.get_field("test")
        assert result.success
        assert result.value.name == "test_field"

        # Get non-existent field
        result = registry.get_field("missing")
        assert result.failure

    def test_field_registry_validation(self) -> None:
        """Test FieldRegistry validation."""
        registry = FlextFields.Registry.FieldRegistry()
        field = FlextFields.Core.StringField("test_field")

        # Empty name
        result = registry.register_field("", field)
        assert result.failure

        result = registry.register_field("   ", field)
        assert result.failure

        # Get with empty name
        result = registry.get_field("")
        assert result.failure

    def test_field_type_registration(self) -> None:
        """Test field type registration."""
        registry = FlextFields.Registry.FieldRegistry()

        # Register field type
        result = registry.register_field_type("custom", FlextFields.Core.StringField)
        assert result.success

        # Get field type
        result = registry.get_field_type("custom")
        assert result.success
        assert result.value == FlextFields.Core.StringField

        # Get non-existent type
        result = registry.get_field_type("missing")
        assert result.failure

    def test_registry_listing(self) -> None:
        """Test registry listing methods."""
        registry = FlextFields.Registry.FieldRegistry()
        field = FlextFields.Core.StringField("test")

        registry.register_field("field1", field)
        registry.register_field("field2", field)
        registry.register_field_type("type1", FlextFields.Core.StringField)

        fields = registry.list_fields()
        assert "field1" in fields
        assert "field2" in fields

        types = registry.list_field_types()
        assert "type1" in types

    def test_registry_metadata(self) -> None:
        """Test registry metadata functionality."""
        registry = FlextFields.Registry.FieldRegistry()
        field = FlextFields.Core.StringField("test", description="Test field")

        registry.register_field("test", field)

        result = registry.get_field_metadata("test")
        assert result.success
        assert result.value["name"] == "test"
        assert result.value["description"] == "Test field"

    def test_registry_clear(self) -> None:
        """Test registry clear functionality."""
        registry = FlextFields.Registry.FieldRegistry()
        field = FlextFields.Core.StringField("test")

        registry.register_field("test", field)
        registry.register_field_type("custom", FlextFields.Core.StringField)

        assert len(registry.list_fields()) > 0
        assert len(registry.list_field_types()) > 0

        registry.clear()

        assert len(registry.list_fields()) == 0
        assert len(registry.list_field_types()) == 0


class TestFlextFieldsSchema:
    """Test FlextFields.Schema functionality."""

    def test_field_processor_basic(self) -> None:
        """Test basic FieldProcessor operations."""
        processor = FlextFields.Schema.FieldProcessor()
        schema = {
            "fields": [
                {
                    "name": "username",
                    "type": "string",
                    "required": True,
                    "constraints": {"min_length": 3},
                }
            ],
            "metadata": {"version": "1.0"},
        }

        result = processor.process_field_schema(schema)
        assert result.success
        assert len(result.value["fields"]) == 1

    def test_field_processor_validation(self) -> None:
        """Test FieldProcessor validation."""
        processor = FlextFields.Schema.FieldProcessor()

        # Missing required keys
        invalid_schema = {
            "fields": [
                {"name": "test"}  # Missing "type"
            ]
        }

        result = processor.process_field_schema(invalid_schema)
        assert result.success  # Should still succeed but with empty fields

    def test_process_multiple_schemas(self) -> None:
        """Test processing multiple schemas."""
        processor = FlextFields.Schema.FieldProcessor()
        schemas = [
            {"fields": [{"name": "field1", "type": "string"}], "metadata": {}},
            {"fields": [{"name": "field2", "type": "integer"}], "metadata": {}},
        ]

        result = processor.process_multiple_fields_schema(schemas)
        assert result.success
        assert len(result.value) == 2


class TestFlextFieldsFactory:
    """Test FlextFields.Factory functionality."""

    def test_create_field_basic(self) -> None:
        """Test basic field creation."""
        result = FlextFields.Factory.create_field("string", "test_field")
        assert result.success
        assert isinstance(result.value, FlextFields.Core.StringField)

        result = FlextFields.Factory.create_field("integer", "int_field")
        assert result.success
        assert isinstance(result.value, FlextFields.Core.IntegerField)

        result = FlextFields.Factory.create_field("boolean", "bool_field")
        assert result.success
        assert isinstance(result.value, FlextFields.Core.BooleanField)

    def test_create_field_with_config(self) -> None:
        """Test field creation with configuration."""
        result = FlextFields.Factory.create_field(
            "string",
            "configured_field",
            min_length=5,
            max_length=20,
            required=False,
            description="Configured field",
        )
        assert result.success
        field = result.value
        assert field.name == "configured_field"
        assert field.required is False

    def test_create_unknown_field_type(self) -> None:
        """Test creating unknown field type."""
        result = FlextFields.Factory.create_field("unknown", "test")
        assert result.failure
        assert "Unknown field type" in result.error

    def test_create_fields_from_schema(self) -> None:
        """Test creating fields from schema."""
        schema = {
            "fields": [
                {
                    "name": "username",
                    "type": "string",
                    "min_length": 3,
                    "required": True,
                },
                {
                    "name": "age",
                    "type": "integer",
                    "min_value": 0,
                    "required": False,
                    "default": 0,
                },
            ]
        }

        result = FlextFields.Factory.create_fields_from_schema(schema)
        assert result.success
        assert len(result.value) == 2
        assert result.value[0].name == "username"
        assert result.value[1].name == "age"

    def test_create_fields_from_invalid_schema(self) -> None:
        """Test creating fields from invalid schema."""
        # Missing fields key
        result = FlextFields.Factory.create_fields_from_schema({})
        assert result.failure

        # Invalid fields structure
        schema = {"fields": "not_a_list"}
        result = FlextFields.Factory.create_fields_from_schema(schema)
        assert result.failure

    def test_field_builder(self) -> None:
        """Test FieldBuilder pattern."""
        builder = FlextFields.Factory.FieldBuilder("string", "test_field")

        result = (
            builder.with_required(required=False)
            .with_default("default_value")
            .with_description("Test field")
            .with_length(min_length=2, max_length=50)
            .with_pattern(r"^[a-zA-Z]+$")
            .build()
        )

        assert result.success
        field = result.value
        assert field.name == "test_field"
        assert field.required is False

    def test_field_builder_numeric(self) -> None:
        """Test FieldBuilder with numeric field."""
        builder = FlextFields.Factory.FieldBuilder("integer", "number_field")

        result = builder.with_range(min_value=1, max_value=100).with_default(50).build()

        assert result.success


class TestFlextFieldsMetadata:
    """Test FlextFields.Metadata functionality."""

    def test_analyze_field(self) -> None:
        """Test field analysis."""
        field = FlextFields.Core.StringField(
            "test_field",
            min_length=3,
            max_length=20,
            pattern=r"^[a-z]+$",
            description="Test field",
        )

        result = FlextFields.Metadata.analyze_field(field)
        assert result.success

        analysis = result.value
        assert analysis["field_class"] == "StringField"
        assert "string_constraints" in analysis
        assert analysis["string_constraints"]["min_length"] == 3
        assert analysis["string_constraints"]["max_length"] == 20

    def test_analyze_numeric_field(self) -> None:
        """Test analysis of numeric field."""
        field = FlextFields.Core.IntegerField("int_field", min_value=0, max_value=100)

        result = FlextFields.Metadata.analyze_field(field)
        assert result.success

        analysis = result.value
        assert "numeric_constraints" in analysis
        assert analysis["numeric_constraints"]["min_value"] == 0
        assert analysis["numeric_constraints"]["max_value"] == 100

    def test_get_field_summary(self) -> None:
        """Test field summary generation."""
        fields = [
            FlextFields.Core.StringField("name", required=True),
            FlextFields.Core.IntegerField("age", required=True, default=0),
            FlextFields.Core.BooleanField("active", required=False, default=True),
            FlextFields.Core.EmailField("email", required=True),
        ]

        result = FlextFields.Metadata.get_field_summary(fields)
        assert result.success

        summary = result.value
        assert summary["total_fields"] == 4
        assert summary["required_fields"] == 3
        assert summary["optional_fields"] == 1
        assert summary["fields_with_defaults"] == 2
        assert "string" in summary["field_types"]
        assert "integer" in summary["field_types"]


class TestFlextFieldsConfiguration:
    """Test FlextFields configuration functionality."""

    def test_configure_fields_system(self) -> None:
        """Test fields system configuration."""
        config = {
            "environment": "development",
            "log_level": "DEBUG",
            "validation_level": "strict",
            "enable_field_validation": True,
        }

        result = FlextFields.configure_fields_system(config)
        assert result.success
        validated_config = result.value
        assert validated_config["environment"] == "development"
        assert validated_config["log_level"] == "DEBUG"
        assert validated_config["enable_field_validation"] is True

    def test_configure_invalid_environment(self) -> None:
        """Test configuration with invalid environment."""
        config = {"environment": "invalid_env"}

        result = FlextFields.configure_fields_system(config)
        assert result.failure
        assert "Invalid environment" in result.error

    def test_get_fields_system_config(self) -> None:
        """Test getting current fields system configuration."""
        result = FlextFields.get_fields_system_config()
        assert result.success

        config = result.value
        assert "environment" in config
        assert "available_field_types" in config
        assert "supported_constraints" in config
        assert len(config["available_field_types"]) >= 7

    def test_create_environment_fields_config(self) -> None:
        """Test creating environment-specific configuration."""
        result = FlextFields.create_environment_fields_config("production")
        assert result.success
        config = result.value
        assert config["environment"] == "production"
        assert config["log_level"] == "WARNING"

        result = FlextFields.create_environment_fields_config("development")
        assert result.success
        config = result.value
        assert config["environment"] == "development"
        assert config["log_level"] == "DEBUG"

        # Invalid environment
        result = FlextFields.create_environment_fields_config("invalid")
        assert result.failure

    def test_optimize_fields_performance(self) -> None:
        """Test fields performance optimization."""
        config = {"performance_level": "high"}

        result = FlextFields.optimize_fields_performance(config)
        assert result.success

        optimized = result.value
        assert optimized["performance_level"] == "high"
        assert optimized["enable_field_caching"] is True
        assert optimized["cache_validation_results"] is True

        # Low performance
        config = {"performance_level": "low"}
        result = FlextFields.optimize_fields_performance(config)
        assert result.success
        optimized = result.value
        assert optimized["enable_field_caching"] is False


class TestFlextFieldsLegacyCompatibility:
    """Test legacy compatibility functions."""

    def test_create_string_field_legacy(self) -> None:
        """Test legacy string field creation."""
        result = FlextFields.create_string_field(
            name="test_field", required=True, min_length=3, description="Test field"
        )
        assert result.success
        field = result.value
        assert field.name == "test_field"

    def test_create_integer_field_legacy(self) -> None:
        """Test legacy integer field creation."""
        result = FlextFields.create_integer_field(
            name="int_field", min_value=0, default=10
        )
        assert result.success

    def test_create_boolean_field_legacy(self) -> None:
        """Test legacy boolean field creation."""
        result = FlextFields.create_boolean_field(name="bool_field", default=True)
        assert result.success


class TestFlextFieldsEdgeCases:
    """Test edge cases and error conditions."""

    def test_field_with_none_values(self) -> None:
        """Test field behavior with None values."""
        field = FlextFields.Core.StringField("test", required=False, default="default")

        result = field.validate(None)
        assert result.success
        assert result.value == "default"

    def test_field_validation_errors(self) -> None:
        """Test various validation error conditions."""
        string_field = FlextFields.Core.StringField("test")

        # Non-string input
        result = string_field.validate(123)
        assert result.failure
        assert FlextConstants.Messages.TYPE_MISMATCH in result.error

        # Test integer field with boolean
        int_field = FlextFields.Core.IntegerField("test")
        result = int_field.validate(True)
        assert result.failure

    @given(st.text(min_size=1, max_size=50))
    def test_string_field_with_random_input(self, text: str) -> None:
        """Property-based test for string field."""
        field = FlextFields.Core.StringField("test", min_length=1, max_length=100)

        result = field.validate(text)
        if len(text) <= 100:
            assert result.success
            assert result.value == text
        else:
            assert result.failure

    @given(st.integers())
    def test_integer_field_with_random_input(self, value: int) -> None:
        """Property-based test for integer field."""
        field = FlextFields.Core.IntegerField("test")

        result = field.validate(value)
        assert result.success
        assert result.value == value

    def test_datetime_field_edge_cases(self) -> None:
        """Test datetime field edge cases."""
        field = FlextFields.Core.DateTimeField("test", required=False)

        # None value
        result = field.validate(None)
        assert result.success
        assert isinstance(result.value, datetime)

        # Invalid type
        result = field.validate(123)
        assert result.failure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
