"""Comprehensive tests for FlextFields field definition and validation system."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from flext_core.constants import FlextFieldType
from flext_core.exceptions import FlextTypeError, FlextValidationError
from flext_core.fields import (
    FlextFieldCore,
    FlextFieldCoreMetadata,
    FlextFieldMetadata,
    FlextFieldRegistry,
    FlextFields,
    flext_create_boolean_field,
    flext_create_integer_field,
    flext_create_string_field,
)

# Constants
EXPECTED_BULK_SIZE = 2
EXPECTED_DATA_COUNT = 3


class TestFlextFieldCore:
    """Test FlextFieldCore functionality."""

    def test_field_core_creation(self) -> None:
        """Test basic field creation."""
        field = FlextFieldCore(
            field_id="test_field",
            field_name="test_name",
            field_type=FlextFieldType.STRING.value,
        )

        if field.field_id != "test_field":
            raise AssertionError(f"Expected {'test_field'}, got {field.field_id}")
        assert field.field_name == "test_name"
        if field.field_type != FlextFieldType.STRING.value:
            raise AssertionError(
                f"Expected {FlextFieldType.STRING.value}, got {field.field_type}"
            )
        if not (field.required):
            raise AssertionError(f"Expected True, got {field.required}")
        assert field.default_value is None

    def test_field_with_all_parameters(self) -> None:
        """Test field creation with comprehensive parameters."""
        field = FlextFieldCore(
            field_id="comprehensive_field",
            field_name="comprehensive_name",
            field_type=FlextFieldType.STRING.value,
            required=False,
            default_value="default",
            min_length=5,
            max_length=50,
            pattern=r"^[A-Z][a-z]+$",
            allowed_values=["Test", "Demo", "Sample"],
            description="Test field description",
            example="Test",
            deprecated=True,
            sensitive=True,
            indexed=True,
            tags=["validation", "test"],
        )

        if field.field_id != "comprehensive_field":
            raise AssertionError(
                f"Expected {'comprehensive_field'}, got {field.field_id}"
            )
        if field.required:
            raise AssertionError(f"Expected False, got {field.required}")
        assert field.default_value == "default"
        if field.min_length != 5:
            raise AssertionError(f"Expected {5}, got {field.min_length}")
        assert field.max_length == 50
        if field.pattern != r"^[A-Z][a-z]+$":
            raise AssertionError(f"Expected {r'^[A-Z][a-z]+$'}, got {field.pattern}")
        assert field.allowed_values == ["Test", "Demo", "Sample"]
        if field.description != "Test field description":
            raise AssertionError(
                f"Expected {'Test field description'}, got {field.description}"
            )
        assert field.example == "Test"
        if not (field.deprecated):
            raise AssertionError(f"Expected True, got {field.deprecated}")
        assert field.sensitive is True
        if not (field.indexed):
            raise AssertionError(f"Expected True, got {field.indexed}")
        if field.tags != ["validation", "test"]:
            raise AssertionError(f"Expected {['validation', 'test']}, got {field.tags}")

    def test_field_immutability(self) -> None:
        """Test that fields are immutable."""
        field = FlextFieldCore(
            field_id="immutable_field",
            field_name="immutable_name",
            field_type=FlextFieldType.STRING.value,
        )

        # Should not be able to modify frozen model
        with pytest.raises((AttributeError, ValidationError)):
            field.field_name = "new_name"

    def test_field_pattern_validation(self) -> None:
        """Test regex pattern validation."""
        # Valid pattern
        field = FlextFieldCore(
            field_id="pattern_field",
            field_name="pattern_name",
            field_type=FlextFieldType.STRING.value,
            pattern=r"^\d{3}-\d{2}-\d{4}$",
        )
        if field.pattern != r"^\d{3}-\d{2}-\d{4}$":
            raise AssertionError(
                f"Expected {r'^\d{3}-\d{2}-\d{4}$'}, got {field.pattern}"
            )

        # Invalid pattern should raise error
        with pytest.raises(FlextValidationError) as exc_info:
            FlextFieldCore(
                field_id="invalid_pattern",
                field_name="invalid_pattern",
                field_type=FlextFieldType.STRING.value,
                pattern=r"[invalid regex(",
            )

        error = exc_info.value
        if "Invalid regex pattern" not in str(error):
            raise AssertionError(f"Expected {'Invalid regex pattern'} in {error!s}")

    def test_field_length_validation(self) -> None:
        """Test min/max length validation."""
        # Valid lengths
        field = FlextFieldCore(
            field_id="length_field",
            field_name="length_name",
            field_type=FlextFieldType.STRING.value,
            min_length=5,
            max_length=10,
        )
        if field.min_length != 5:
            raise AssertionError(f"Expected {5}, got {field.min_length}")
        assert field.max_length == 10

        with pytest.raises(FlextValidationError) as exc_info:
            FlextFieldCore(
                field_id="invalid_length",
                field_name="invalid_length",
                field_type=FlextFieldType.STRING.value,
                min_length=10,
                max_length=5,
            )

        error = exc_info.value
        if "max_length must be greater than min_length" not in str(error):
            raise AssertionError(
                f"Expected {'max_length must be greater than min_length'} in {error!s}"
            )

    def test_string_value_validation(self) -> None:
        """Test string value validation."""
        field = FlextFieldCore(
            field_id="string_field",
            field_name="string_name",
            field_type=FlextFieldType.STRING.value,
            min_length=3,
            max_length=10,
            pattern=r"^[A-Za-z]+$",
            allowed_values=["hello", "world", "test"],
        )

        # Valid string
        is_valid, error = field.validate_field_value("hello")
        if not (is_valid):
            raise AssertionError(f"Expected True, got {is_valid}")
        assert error is None

        # Too short
        is_valid, error = field.validate_field_value("hi")
        if is_valid:
            raise AssertionError(f"Expected False, got {is_valid}")
        assert "too short" in error

        # Too long
        is_valid, error = field.validate_field_value("verylongstring")
        if is_valid:
            raise AssertionError(f"Expected False, got {is_valid}")
        assert "too long" in error

        # Pattern mismatch
        is_valid, error = field.validate_field_value("hello123")
        if is_valid:
            raise AssertionError(f"Expected False, got {is_valid}")
        assert "does not match pattern" in error

        # Not in allowed values
        is_valid, error = field.validate_field_value("invalid")
        if is_valid:
            raise AssertionError(f"Expected False, got {is_valid}")
        assert "not in allowed list" in error

        # Wrong type
        is_valid, error = field.validate_field_value(123)
        if is_valid:
            raise AssertionError(f"Expected False, got {is_valid}")
        assert "Expected string" in error

    def test_integer_value_validation(self) -> None:
        """Test integer value validation."""
        field = FlextFieldCore(
            field_id="int_field",
            field_name="int_name",
            field_type=FlextFieldType.INTEGER.value,
            min_value=10,
            max_value=100,
        )

        # Valid integer
        is_valid, error = field.validate_field_value(50)
        if not (is_valid):
            raise AssertionError(f"Expected True, got {is_valid}")
        assert error is None

        # Too small
        is_valid, error = field.validate_field_value(5)
        if is_valid:
            raise AssertionError(f"Expected False, got {is_valid}")
        assert "too small" in error

        # Too large
        is_valid, error = field.validate_field_value(150)
        if is_valid:
            raise AssertionError(f"Expected False, got {is_valid}")
        assert "too large" in error

        # Wrong type
        is_valid, error = field.validate_field_value("50")
        if is_valid:
            raise AssertionError(f"Expected False, got {is_valid}")
        assert "Expected integer" in error

    def test_boolean_value_validation(self) -> None:
        """Test boolean value validation."""
        field = FlextFieldCore(
            field_id="bool_field",
            field_name="bool_name",
            field_type=FlextFieldType.BOOLEAN.value,
        )

        # Valid boolean
        is_valid, error = field.validate_field_value(value=True)
        if not (is_valid):
            raise AssertionError(f"Expected True, got {is_valid}")
        assert error is None

        is_valid, error = field.validate_field_value(value=False)
        if not (is_valid):
            raise AssertionError(f"Expected True, got {is_valid}")
        assert error is None

        # Wrong type
        is_valid, error = field.validate_field_value("true")
        if is_valid:
            raise AssertionError(f"Expected False, got {is_valid}")
        assert "Expected boolean" in error

    def test_required_field_validation(self) -> None:
        """Test required field validation."""
        # Required field without default
        required_field = FlextFieldCore(
            field_id="required_field",
            field_name="required_name",
            field_type=FlextFieldType.STRING.value,
            required=True,
        )

        is_valid, error = required_field.validate_field_value(None)
        if is_valid:
            raise AssertionError(f"Expected False, got {is_valid}")
        assert "is required" in error

        # Optional field
        optional_field = FlextFieldCore(
            field_id="optional_field",
            field_name="optional_name",
            field_type=FlextFieldType.STRING.value,
            required=False,
        )

        is_valid, error = optional_field.validate_field_value(None)
        if not (is_valid):
            raise AssertionError(f"Expected True, got {is_valid}")
        assert error is None

        # Required field with default
        default_field = FlextFieldCore(
            field_id="default_field",
            field_name="default_name",
            field_type=FlextFieldType.STRING.value,
            required=True,
            default_value="default",
        )

        is_valid, error = default_field.validate_field_value(None)
        if not (is_valid):
            raise AssertionError(f"Expected True, got {is_valid}")
        assert error is None

    def test_validate_value_result_pattern(self) -> None:
        """Test validate_value method with FlextResult."""
        field = FlextFieldCore(
            field_id="result_field",
            field_name="result_name",
            field_type=FlextFieldType.STRING.value,
            min_length=3,
        )

        # Valid value
        result = field.validate_value("hello")
        assert result.is_success
        if result.data != "hello":
            raise AssertionError(f"Expected {'hello'}, got {result.data}")

        # Invalid value
        result = field.validate_value("hi")
        assert result.is_failure
        if "too short" not in (result.error or ""):
            raise AssertionError(f"Expected 'too short' in {result.error}")

        # None value with default
        field_with_default = FlextFieldCore(
            field_id="default_field",
            field_name="default_name",
            field_type=FlextFieldType.STRING.value,
            default_value="default_value",
        )

        result = field_with_default.validate_value(None)
        assert result.is_success
        if result.data != "default_value":
            raise AssertionError(f"Expected {'default_value'}, got {result.data}")

    def test_field_metadata_methods(self) -> None:  # noqa: C901
        """Test field metadata access methods."""
        field = FlextFieldCore(
            field_id="meta_field",
            field_name="meta_name",
            field_type=FlextFieldType.STRING.value,
            description="Test description",
            example="example",
            deprecated=True,
            sensitive=True,
            indexed=True,
            tags=["meta", "test"],
        )

        # Test has_tag
        if not (field.has_tag("meta")):
            raise AssertionError(f"Expected True, got {field.has_tag('meta')}")
        if field.has_tag("nonexistent"):
            raise AssertionError(f"Expected False, got {field.has_tag('nonexistent')}")

        # Test get_field_schema
        schema = field.get_field_schema()
        assert isinstance(schema, dict)
        if schema["field_id"] != "meta_field":
            raise AssertionError(f"Expected {'meta_field'}, got {schema['field_id']}")
        assert schema["description"] == "Test description"

        # Test get_field_metadata
        metadata = field.get_field_metadata()
        assert isinstance(metadata, dict)
        if metadata["description"] != "Test description":
            raise AssertionError(
                f"Expected {'Test description'}, got {metadata['description']}"
            )
        if not (metadata["deprecated"]):
            raise AssertionError(f"Expected True, got {metadata['deprecated']}")
        assert metadata["sensitive"] is True
        if not (metadata["indexed"]):
            raise AssertionError(f"Expected True, got {metadata['indexed']}")
        if metadata["tags"] != ["meta", "test"]:
            raise AssertionError(f"Expected {['meta', 'test']}, got {metadata['tags']}")

        # Test backward compatibility methods
        assert field.get_default_value() is None
        if not (field.is_required()):
            raise AssertionError(f"Expected True, got {field.is_required()}")
        assert field.is_deprecated() is True
        if not (field.is_sensitive()):
            raise AssertionError(f"Expected True, got {field.is_sensitive()}")

        # Test get_field_info
        info = field.get_field_info()
        assert isinstance(info, dict)
        if info["field_id"] != "meta_field":
            raise AssertionError(f"Expected {'meta_field'}, got {info['field_id']}")
        assert info["field_name"] == "meta_name"
        if not (info["required"]):
            raise AssertionError(f"Expected True, got {info['required']}")
        if "metadata" not in info:
            raise AssertionError(f"Expected {'metadata'} in {info}")

    def test_field_serialization(self) -> None:  # noqa: C901
        """Test field value serialization."""
        string_field = FlextFieldCore(
            field_id="string_field",
            field_name="string_name",
            field_type=FlextFieldType.STRING.value,
        )

        # String serialization
        if string_field.serialize_value("hello") != "hello":
            raise AssertionError(
                f"Expected {'hello'}, got {string_field.serialize_value('hello')}"
            )
        assert string_field.serialize_value(123) == "123"
        assert string_field.serialize_value(None) is None

        # Integer serialization
        int_field = FlextFieldCore(
            field_id="int_field",
            field_name="int_name",
            field_type=FlextFieldType.INTEGER.value,
        )

        if int_field.serialize_value(42) != 42:
            raise AssertionError(f"Expected {42}, got {int_field.serialize_value(42)}")
        assert int_field.serialize_value(math.pi) == EXPECTED_DATA_COUNT
        # Non-numeric strings unchanged
        if int_field.serialize_value("123") != "123":
            raise AssertionError(
                f"Expected {'123'}, got {int_field.serialize_value('123')}"
            )

        # Boolean serialization
        bool_field = FlextFieldCore(
            field_id="bool_field",
            field_name="bool_name",
            field_type=FlextFieldType.BOOLEAN.value,
        )

        if not (bool_field.serialize_value(value=True)):
            raise AssertionError(
                f"Expected True, got {bool_field.serialize_value(value=True)}"
            )
        if bool_field.serialize_value(value=False):
            raise AssertionError(
                f"Expected False, got {bool_field.serialize_value(value=False)}"
            )
        if not (bool_field.serialize_value("true")):
            raise AssertionError(
                f"Expected True, got {bool_field.serialize_value('true')}"
            )
        if bool_field.serialize_value("false"):
            raise AssertionError(
                f"Expected False, got {bool_field.serialize_value('false')}"
            )
        if not (bool_field.serialize_value("1")):
            raise AssertionError(
                f"Expected True, got {bool_field.serialize_value('1')}"
            )
        if bool_field.serialize_value("0"):
            raise AssertionError(
                f"Expected False, got {bool_field.serialize_value('0')}"
            )
        if not (bool_field.serialize_value("yes")):
            raise AssertionError(
                f"Expected True, got {bool_field.serialize_value('yes')}"
            )
        if bool_field.serialize_value("no"):
            raise AssertionError(
                f"Expected False, got {bool_field.serialize_value('no')}"
            )

    def test_field_deserialization(self) -> None:  # noqa: C901
        """Test field value deserialization."""
        # String deserialization
        string_field = FlextFieldCore(
            field_id="string_field",
            field_name="string_name",
            field_type=FlextFieldType.STRING.value,
            default_value="default",
        )

        if string_field.deserialize_value("hello") != "hello":
            raise AssertionError(
                f"Expected {'hello'}, got {string_field.deserialize_value('hello')}"
            )
        assert string_field.deserialize_value(123) == "123"
        if string_field.deserialize_value(None) != "default":
            raise AssertionError(
                f"Expected {'default'}, got {string_field.deserialize_value(None)}"
            )

        # Integer deserialization
        int_field = FlextFieldCore(
            field_id="int_field",
            field_name="int_name",
            field_type=FlextFieldType.INTEGER.value,
        )

        if int_field.deserialize_value(42) != 42:
            raise AssertionError(
                f"Expected {42}, got {int_field.deserialize_value(42)}"
            )
        assert int_field.deserialize_value("123") == 123
        if int_field.deserialize_value("456") != 456:
            raise AssertionError(
                f"Expected {456}, got {int_field.deserialize_value('456')}"
            )
        assert int_field.deserialize_value(math.pi) == EXPECTED_DATA_COUNT
        if int_field.deserialize_value("not_a_number") != "not_a_number":
            raise AssertionError(
                f"Expected {'not_a_number'}, got {int_field.deserialize_value('not_a_number')}"
            )

        # Float deserialization
        float_field = FlextFieldCore(
            field_id="float_field",
            field_name="float_name",
            field_type="float",
        )

        # Use approximate comparison for floating point values
        pi_value = float_field.deserialize_value(math.pi)
        if not (abs(pi_value - math.pi) < 1e-10):
            raise AssertionError(f"Expected approximately {math.pi}, got {pi_value}")
        # Test float deserialization from string - expect exact value
        if float_field.deserialize_value("3.14") != math.pi:
            raise AssertionError(
                f"Expected 3.14, got {float_field.deserialize_value('3.14')}"
            )
        if float_field.deserialize_value(42) != 42.0:
            raise AssertionError(
                f"Expected {42.0}, got {float_field.deserialize_value(42)}"
            )
        assert float_field.deserialize_value("not_a_number") == "not_a_number"

        # Boolean deserialization
        bool_field = FlextFieldCore(
            field_id="bool_field",
            field_name="bool_name",
            field_type=FlextFieldType.BOOLEAN.value,
        )

        if not (bool_field.deserialize_value(value=True)):
            raise AssertionError(
                f"Expected True, got {bool_field.deserialize_value(value=True)}"
            )
        if bool_field.deserialize_value(value=False):
            raise AssertionError(
                f"Expected False, got {bool_field.deserialize_value(value=False)}"
            )
        if not (bool_field.deserialize_value("true")):
            raise AssertionError(
                f"Expected True, got {bool_field.deserialize_value('true')}"
            )
        if bool_field.deserialize_value("false"):
            raise AssertionError(
                f"Expected False, got {bool_field.deserialize_value('false')}"
            )
        if not (bool_field.deserialize_value("1")):
            raise AssertionError(
                f"Expected True, got {bool_field.deserialize_value('1')}"
            )
        if bool_field.deserialize_value("0"):
            raise AssertionError(
                f"Expected False, got {bool_field.deserialize_value('0')}"
            )
        if not (bool_field.deserialize_value(1)):
            raise AssertionError(
                f"Expected True, got {bool_field.deserialize_value(1)}"
            )
        if bool_field.deserialize_value(0):
            raise AssertionError(
                f"Expected False, got {bool_field.deserialize_value(0)}"
            )

        # Test error conditions for boolean deserialization
        with pytest.raises(FlextTypeError):
            bool_field.deserialize_value([1, 2, 3])

        with pytest.raises(FlextTypeError):
            bool_field.deserialize_value({"key": "value"})

    def test_field_metadata_property(self) -> None:
        """Test field metadata property."""
        field = FlextFieldCore(
            field_id="metadata_field",
            field_name="metadata_name",
            field_type=FlextFieldType.STRING.value,
            description="Test description",
        )

        metadata = field.metadata
        assert isinstance(metadata, FlextFieldMetadata)
        if metadata.field_id != "metadata_field":
            raise AssertionError(
                f"Expected {'metadata_field'}, got {metadata.field_id}"
            )
        assert metadata.field_name == "metadata_name"
        if metadata.description != "Test description":
            raise AssertionError(
                f"Expected {'Test description'}, got {metadata.description}"
            )


class TestFlextFieldMetadata:
    """Test FlextFieldMetadata functionality."""

    def test_metadata_creation(self) -> None:
        """Test metadata creation."""
        metadata = FlextFieldMetadata(
            field_id="test_id",
            field_name="test_name",
            field_type="string",
            description="Test description",
        )

        if metadata.field_id != "test_id":
            raise AssertionError(f"Expected {'test_id'}, got {metadata.field_id}")
        assert metadata.field_name == "test_name"
        if metadata.field_type != "string":
            raise AssertionError(f"Expected {'string'}, got {metadata.field_type}")
        assert metadata.description == "Test description"

    def test_metadata_defaults(self) -> None:
        """Test metadata with default values."""
        metadata = FlextFieldMetadata()

        if metadata.field_id != "unknown":
            raise AssertionError(f"Expected {'unknown'}, got {metadata.field_id}")
        assert metadata.field_name == "unknown"
        if metadata.field_type != "string":
            raise AssertionError(f"Expected {'string'}, got {metadata.field_type}")
        if not (metadata.required):
            raise AssertionError(f"Expected True, got {metadata.required}")
        assert metadata.default_value is None
        if metadata.deprecated:
            raise AssertionError(f"Expected False, got {metadata.deprecated}")
        assert metadata.sensitive is False
        if metadata.indexed:
            raise AssertionError(f"Expected False, got {metadata.indexed}")
        assert internal.invalid is False
        if metadata.unique:
            raise AssertionError(f"Expected False, got {metadata.unique}")
        assert metadata.tags == []
        if metadata.allowed_values != []:
            raise AssertionError(f"Expected {[]}, got {metadata.allowed_values}")
        assert metadata.custom_properties == {}

    def test_metadata_from_field(self) -> None:  # noqa: C901
        """Test creating metadata from field."""
        field = FlextFieldCore(
            field_id="source_field",
            field_name="source_name",
            field_type=FlextFieldType.STRING.value,
            required=False,
            default_value="default",
            min_length=5,
            max_length=50,
            pattern=r"^[A-Z]",
            allowed_values=["Test", "Demo"],
            description="Source description",
            example="Test",
            tags=["source", "test"],
            deprecated=True,
            sensitive=True,
            indexed=True,
        )

        metadata = FlextFieldMetadata.from_field(field)

        if metadata.field_id != "source_field":
            raise AssertionError(f"Expected {'source_field'}, got {metadata.field_id}")
        assert metadata.field_name == "source_name"
        if metadata.field_type != FlextFieldType.STRING.value:
            raise AssertionError(
                f"Expected {FlextFieldType.STRING.value}, got {metadata.field_type}"
            )
        if metadata.required:
            raise AssertionError(f"Expected False, got {metadata.required}")
        assert metadata.default_value == "default"
        if metadata.min_length != 5:
            raise AssertionError(f"Expected {5}, got {metadata.min_length}")
        assert metadata.max_length == 50
        if metadata.pattern != r"^[A-Z]":
            raise AssertionError(f"Expected {r'^[A-Z]'}, got {metadata.pattern}")
        assert metadata.allowed_values == ["Test", "Demo"]
        if metadata.description != "Source description":
            raise AssertionError(
                f"Expected {'Source description'}, got {metadata.description}"
            )
        assert metadata.example == "Test"
        if metadata.tags != ["source", "test"]:
            raise AssertionError(f"Expected {['source', 'test']}, got {metadata.tags}")
        if not (metadata.deprecated):
            raise AssertionError(f"Expected True, got {metadata.deprecated}")
        assert metadata.sensitive is True
        if not (metadata.indexed):
            raise AssertionError(f"Expected True, got {metadata.indexed}")
        # Fields not in FlextFieldCore should have defaults
        if internal.invalid:
            raise AssertionError(f"Expected False, got {internal.invalid}")
        assert metadata.unique is False
        if metadata.custom_properties != {}:
            raise AssertionError(f"Expected {{}}, got {metadata.custom_properties}")

    def test_metadata_to_dict(self) -> None:
        """Test metadata dictionary conversion."""
        metadata = FlextFieldMetadata(
            field_id="dict_field",
            field_name="dict_name",
            description="Dict description",
        )

        data = metadata.to_dict()
        assert isinstance(data, dict)
        if data["field_id"] != "dict_field":
            raise AssertionError(f"Expected {'dict_field'}, got {data['field_id']}")
        assert data["field_name"] == "dict_name"
        if data["description"] != "Dict description":
            raise AssertionError(
                f"Expected {'Dict description'}, got {data['description']}"
            )

    def test_metadata_from_dict(self) -> None:
        """Test creating metadata from dictionary."""
        data = {
            "field_id": "from_dict",
            "field_name": "from_dict_name",
            "field_type": "integer",
            "required": False,
            "description": "From dict description",
        }

        metadata = FlextFieldMetadata.from_dict(data)
        if metadata.field_id != "from_dict":
            raise AssertionError(f"Expected {'from_dict'}, got {metadata.field_id}")
        assert metadata.field_name == "from_dict_name"
        if metadata.field_type != "integer":
            raise AssertionError(f"Expected {'integer'}, got {metadata.field_type}")
        if metadata.required:
            raise AssertionError(f"Expected False, got {metadata.required}")
        assert metadata.description == "From dict description"

    def test_metadata_from_dict_with_defaults(self) -> None:
        """Test creating metadata from dictionary with missing fields."""
        data = {"description": "Partial description"}

        metadata = FlextFieldMetadata.from_dict(data)
        if metadata.field_id != "unknown":  # Default:
            raise AssertionError(f"Expected {'unknown'}, got {metadata.field_id}")
        assert metadata.field_name == "unknown"  # Default
        if metadata.field_type != "string":  # Default:
            raise AssertionError(f"Expected {'string'}, got {metadata.field_type}")
        assert metadata.required is True  # Default
        if metadata.description != "Partial description":
            raise AssertionError(
                f"Expected {'Partial description'}, got {metadata.description}"
            )

    def test_metadata_immutability(self) -> None:
        """Test that metadata is immutable."""
        metadata = FlextFieldMetadata(field_id="immutable")

        with pytest.raises((AttributeError, ValidationError)):
            metadata.field_id = "new_id"


class TestFlextFieldRegistry:
    """Test FlextFieldRegistry functionality."""

    def test_registry_creation(self) -> None:
        """Test registry creation."""
        registry = FlextFieldRegistry()

        assert isinstance(registry.fields_dict, dict)
        assert isinstance(registry.field_names_dict, dict)
        if len(registry.fields_dict) != 0:
            raise AssertionError(f"Expected {0}, got {len(registry.fields_dict)}")
        assert len(registry.field_names_dict) == 0

    def test_field_registration(self) -> None:
        """Test field registration."""
        registry = FlextFieldRegistry()
        field = FlextFieldCore(
            field_id="test_field",
            field_name="test_name",
            field_type=FlextFieldType.STRING.value,
        )

        result = registry.register_field(field)
        assert result.is_success

        if field.field_id not in registry.fields_dict:
            raise AssertionError(f"Expected {field.field_id} in {registry.fields_dict}")
        assert field.field_name in registry.field_names_dict
        if registry.field_names_dict[field.field_name] != field.field_id:
            raise AssertionError(
                f"Expected {field.field_id}, got {registry.field_names_dict[field.field_name]}"
            )

    def test_field_registration_id_conflict(self) -> None:
        """Test field registration with ID conflict."""
        registry = FlextFieldRegistry()
        field1 = FlextFieldCore(
            field_id="conflict_id",
            field_name="name1",
            field_type=FlextFieldType.STRING.value,
        )
        field2 = FlextFieldCore(
            field_id="conflict_id",
            field_name="name2",
            field_type=FlextFieldType.STRING.value,
        )

        # Register first field
        result1 = registry.register_field(field1)
        assert result1.is_success

        # Try to register second field with same ID
        result2 = registry.register_field(field2)
        assert result2.is_failure
        if "already registered" not in result2.error:
            raise AssertionError(f"Expected {'already registered'} in {result2.error}")

    def test_field_registration_name_conflict(self) -> None:
        """Test field registration with name conflict."""
        registry = FlextFieldRegistry()
        field1 = FlextFieldCore(
            field_id="id1",
            field_name="conflict_name",
            field_type=FlextFieldType.STRING.value,
        )
        field2 = FlextFieldCore(
            field_id="id2",
            field_name="conflict_name",
            field_type=FlextFieldType.STRING.value,
        )

        # Register first field
        result1 = registry.register_field(field1)
        assert result1.is_success

        # Try to register second field with same name
        result2 = registry.register_field(field2)
        assert result2.is_failure
        if "already registered" not in result2.error:
            raise AssertionError(f"Expected {'already registered'} in {result2.error}")

    def test_get_field_by_id(self) -> None:
        """Test getting field by ID."""
        registry = FlextFieldRegistry()
        field = FlextFieldCore(
            field_id="lookup_id",
            field_name="lookup_name",
            field_type=FlextFieldType.STRING.value,
        )

        registry.register_field(field)

        # Successful lookup
        result = registry.get_field_by_id("lookup_id")
        assert result.is_success
        if result.data != field:
            raise AssertionError(f"Expected {field}, got {result.data}")

        # Failed lookup
        result = registry.get_field_by_id("nonexistent")
        assert result.is_failure
        if "not found" not in (result.error or ""):
            raise AssertionError(f"Expected 'not found' in {result.error}")

    def test_get_field_by_name(self) -> None:
        """Test getting field by name."""
        registry = FlextFieldRegistry()
        field = FlextFieldCore(
            field_id="name_lookup_id",
            field_name="name_lookup_name",
            field_type=FlextFieldType.STRING.value,
        )

        registry.register_field(field)

        # Successful lookup
        result = registry.get_field_by_name("name_lookup_name")
        assert result.is_success
        if result.data != field:
            raise AssertionError(f"Expected {field}, got {result.data}")

        # Failed lookup
        result = registry.get_field_by_name("nonexistent")
        assert result.is_failure
        if "not found" not in (result.error or ""):
            raise AssertionError(f"Expected 'not found' in {result.error}")

    def test_registry_backward_compatibility(self) -> None:
        """Test backward compatibility methods."""
        registry = FlextFieldRegistry()
        field = FlextFieldCore(
            field_id="compat_id",
            field_name="compat_name",
            field_type=FlextFieldType.STRING.value,
        )

        registry.register_field(field)

        # Test get_field (backward compatibility)
        found_field = registry.get_field("compat_id")
        if found_field != field:
            raise AssertionError(f"Expected {field}, got {found_field}")

        found_field = registry.get_field("nonexistent")
        assert found_field is None

        # Test get_all_fields (backward compatibility)
        all_fields = registry.get_all_fields()
        assert isinstance(all_fields, dict)
        if "compat_id" not in all_fields:
            raise AssertionError(f"Expected {'compat_id'} in {all_fields}")
        if all_fields["compat_id"] != field:
            raise AssertionError(f"Expected {field}, got {all_fields['compat_id']}")

        # Test _fields property (backward compatibility)
        if registry._fields != registry.fields_dict:
            raise AssertionError(
                f"Expected {registry.fields_dict}, got {registry._fields}"
            )

    def test_registry_management_methods(self) -> None:  # noqa: C901
        """Test registry management methods."""
        registry = FlextFieldRegistry()
        field1 = FlextFieldCore(
            field_id="mgmt1",
            field_name="mgmt_name1",
            field_type=FlextFieldType.STRING.value,
        )
        field2 = FlextFieldCore(
            field_id="mgmt2",
            field_name="mgmt_name2",
            field_type=FlextFieldType.INTEGER.value,
        )

        registry.register_field(field1)
        registry.register_field(field2)

        # Test list methods
        field_names = registry.list_field_names()
        if len(field_names) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(field_names)}")
        if "mgmt_name1" not in field_names:
            raise AssertionError(f"Expected {'mgmt_name1'} in {field_names}")
        assert "mgmt_name2" in field_names

        field_ids = registry.list_field_ids()
        if len(field_ids) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(field_ids)}")
        if "mgmt1" not in field_ids:
            raise AssertionError(f"Expected {'mgmt1'} in {field_ids}")
        assert "mgmt2" in field_ids

        # Test count
        if registry.get_field_count() != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {registry.get_field_count()}")

        # Test removal
        removed = registry.remove_field("mgmt1")
        if not (removed):
            raise AssertionError(f"Expected True, got {removed}")
        if registry.get_field_count() != 1:
            raise AssertionError(f"Expected {1}, got {registry.get_field_count()}")
        if "mgmt1" in registry.fields_dict:
            raise AssertionError(f"Expected {'mgmt1'} not in {registry.fields_dict}")
        assert "mgmt_name1" not in registry.field_names_dict

        # Try to remove non-existent field
        removed = registry.remove_field("nonexistent")
        if removed:
            raise AssertionError(f"Expected False, got {removed}")

        # Test clear
        registry.clear_registry()
        if registry.get_field_count() != 0:
            raise AssertionError(f"Expected {0}, got {registry.get_field_count()}")
        assert len(registry.fields_dict) == 0
        if len(registry.field_names_dict) != 0:
            raise AssertionError(f"Expected {0}, got {len(registry.field_names_dict)}")

    def test_validate_all_fields(self) -> None:
        """Test validating all fields in registry."""
        registry = FlextFieldRegistry()

        # Required field
        required_field = FlextFieldCore(
            field_id="required_field",
            field_name="required_name",
            field_type=FlextFieldType.STRING.value,
            required=True,
            min_length=3,
        )

        # Optional field
        optional_field = FlextFieldCore(
            field_id="optional_field",
            field_name="optional_name",
            field_type=FlextFieldType.INTEGER.value,
            required=False,
            min_value=10,
        )

        registry.register_field(required_field)
        registry.register_field(optional_field)

        # Valid data
        valid_data = {
            "required_field": "hello",
            "optional_field": 20,
        }
        result = registry.validate_all_fields(valid_data)
        assert result.is_success

        # Missing required field
        invalid_data = {"optional_field": 20}
        result = registry.validate_all_fields(invalid_data)
        assert result.is_failure
        if "is missing" not in (result.error or ""):
            raise AssertionError(f"Expected 'is missing' in {result.error}")

        # Invalid field value
        invalid_value_data = {
            "required_field": "hi",  # Too short
            "optional_field": 20,
        }
        result = registry.validate_all_fields(invalid_value_data)
        assert result.is_failure
        if "too short" not in (result.error or ""):
            raise AssertionError(f"Expected 'too short' in {result.error}")

    def test_get_fields_by_type(self) -> None:
        """Test getting fields by type."""
        registry = FlextFieldRegistry()

        string_field1 = FlextFieldCore(
            field_id="str1",
            field_name="string1",
            field_type=FlextFieldType.STRING.value,
        )
        string_field2 = FlextFieldCore(
            field_id="str2",
            field_name="string2",
            field_type=FlextFieldType.STRING.value,
        )
        int_field = FlextFieldCore(
            field_id="int1",
            field_name="integer1",
            field_type=FlextFieldType.INTEGER.value,
        )

        registry.register_field(string_field1)
        registry.register_field(string_field2)
        registry.register_field(int_field)

        # Get string fields using enum
        string_fields = registry.get_fields_by_type(FlextFieldType.STRING)
        if len(string_fields) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(string_fields)}")
        if string_field1 not in string_fields:
            raise AssertionError(f"Expected {string_field1} in {string_fields}")
        assert string_field2 in string_fields

        # Get integer fields using string
        int_fields = registry.get_fields_by_type("integer")
        if len(int_fields) != 1:
            raise AssertionError(f"Expected {1}, got {len(int_fields)}")
        if int_field not in int_fields:
            raise AssertionError(f"Expected {int_field} in {int_fields}")

        # Get non-existent type
        boolean_fields = registry.get_fields_by_type(FlextFieldType.BOOLEAN)
        if len(boolean_fields) != 0:
            raise AssertionError(f"Expected {0}, got {len(boolean_fields)}")


class TestFlextFields:
    """Test FlextFields factory methods."""

    def setUp(self) -> None:
        """Clear registry before each test."""
        FlextFields.clear_registry()

    def test_create_string_field(self) -> None:
        """Test creating string field."""
        field = FlextFields.create_string_field(
            field_id="string_factory",
            field_name="string_factory_name",
            required=False,
            default_value="default",
            min_length=5,
            max_length=50,
            pattern=r"^[A-Z]",
            allowed_values=["Test", "Demo"],
            description="Factory string field",
            example="Test",
            deprecated=True,
            sensitive=True,
            indexed=True,
            tags=["factory", "test"],
        )

        if field.field_id != "string_factory":
            raise AssertionError(f"Expected {'string_factory'}, got {field.field_id}")
        assert field.field_name == "string_factory_name"
        if field.field_type != FlextFieldType.STRING.value:
            raise AssertionError(
                f"Expected {FlextFieldType.STRING.value}, got {field.field_type}"
            )
        if field.required:
            raise AssertionError(f"Expected False, got {field.required}")
        assert field.default_value == "default"
        if field.min_length != 5:
            raise AssertionError(f"Expected {5}, got {field.min_length}")
        assert field.max_length == 50
        if field.pattern != r"^[A-Z]":
            raise AssertionError(f"Expected {r'^[A-Z]'}, got {field.pattern}")
        assert field.allowed_values == ["Test", "Demo"]
        if field.description != "Factory string field":
            raise AssertionError(
                f"Expected {'Factory string field'}, got {field.description}"
            )
        assert field.example == "Test"
        if not (field.deprecated):
            raise AssertionError(f"Expected True, got {field.deprecated}")
        assert field.sensitive is True
        if not (field.indexed):
            raise AssertionError(f"Expected True, got {field.indexed}")
        if field.tags != ["factory", "test"]:
            raise AssertionError(f"Expected {['factory', 'test']}, got {field.tags}")

    def test_create_integer_field(self) -> None:
        """Test creating integer field."""
        field = FlextFields.create_integer_field(
            field_id="int_factory",
            field_name="int_factory_name",
            required=False,
            default_value=42,
            min_value=10,
            max_value=100,
            description="Factory integer field",
            example=50,
            deprecated=False,
            sensitive=False,
            indexed=True,
            tags=["factory", "int"],
        )

        if field.field_id != "int_factory":
            raise AssertionError(f"Expected {'int_factory'}, got {field.field_id}")
        assert field.field_name == "int_factory_name"
        if field.field_type != FlextFieldType.INTEGER.value:
            raise AssertionError(
                f"Expected {FlextFieldType.INTEGER.value}, got {field.field_type}"
            )
        if field.required:
            raise AssertionError(f"Expected False, got {field.required}")
        assert field.default_value == 42
        if field.min_value != 10.0:
            raise AssertionError(f"Expected {10.0}, got {field.min_value}")
        assert field.max_value == 100.0
        if field.description != "Factory integer field":
            raise AssertionError(
                f"Expected {'Factory integer field'}, got {field.description}"
            )
        assert field.example == 50
        if field.deprecated:
            raise AssertionError(f"Expected False, got {field.deprecated}")
        assert field.sensitive is False
        if not (field.indexed):
            raise AssertionError(f"Expected True, got {field.indexed}")
        if field.tags != ["factory", "int"]:
            raise AssertionError(f"Expected {['factory', 'int']}, got {field.tags}")

    def test_create_boolean_field(self) -> None:
        """Test creating boolean field."""
        field = FlextFields.create_boolean_field(
            field_id="bool_factory",
            field_name="bool_factory_name",
            required=False,
            default_value=True,
            description="Factory boolean field",
            example=False,
            deprecated=False,
            sensitive=False,
            indexed=False,
            tags=["factory", "bool"],
        )

        if field.field_id != "bool_factory":
            raise AssertionError(f"Expected {'bool_factory'}, got {field.field_id}")
        assert field.field_name == "bool_factory_name"
        if field.field_type != FlextFieldType.BOOLEAN.value:
            raise AssertionError(
                f"Expected {FlextFieldType.BOOLEAN.value}, got {field.field_type}"
            )
        if field.required:
            raise AssertionError(f"Expected False, got {field.required}")
        if not (field.default_value):
            raise AssertionError(f"Expected True, got {field.default_value}")
        if field.description != "Factory boolean field":
            raise AssertionError(
                f"Expected {'Factory boolean field'}, got {field.description}"
            )
        if field.example:
            raise AssertionError(f"Expected False, got {field.example}")
        assert field.deprecated is False
        if field.sensitive:
            raise AssertionError(f"Expected False, got {field.sensitive}")
        assert field.indexed is False
        if field.tags != ["factory", "bool"]:
            raise AssertionError(f"Expected {['factory', 'bool']}, got {field.tags}")

    def test_factory_registry_integration(self) -> None:
        """Test factory integration with registry."""
        FlextFields.clear_registry()

        # Create and register field
        field = FlextFields.create_string_field(
            field_id="registry_test",
            field_name="registry_test_name",
        )

        result = FlextFields.register_field(field)
        assert result.is_success

        # Test lookup methods
        lookup_result = FlextFields.get_field_by_id("registry_test")
        assert lookup_result.is_success
        if lookup_result.data != field:
            raise AssertionError(f"Expected {field}, got {lookup_result.data}")

        name_lookup_result = FlextFields.get_field_by_name("registry_test_name")
        assert name_lookup_result.is_success
        if name_lookup_result.data != field:
            raise AssertionError(f"Expected {field}, got {name_lookup_result.data}")

        # Test registry management
        field_names = FlextFields.list_field_names()
        if "registry_test_name" not in field_names:
            raise AssertionError(f"Expected {'registry_test_name'} in {field_names}")

        if FlextFields.get_field_count() != 1:
            raise AssertionError(f"Expected {1}, got {FlextFields.get_field_count()}")

        FlextFields.clear_registry()
        if FlextFields.get_field_count() != 0:
            raise AssertionError(f"Expected {0}, got {FlextFields.get_field_count()}")

    def test_factory_with_minimal_config(self) -> None:
        """Test factory methods with minimal configuration."""
        # String field with minimal config
        string_field = FlextFields.create_string_field(
            field_id="minimal_string",
            field_name="minimal_string_name",
        )
        if string_field.field_type != FlextFieldType.STRING.value:
            raise AssertionError(
                f"Expected {FlextFieldType.STRING.value}, got {string_field.field_type}"
            )
        assert string_field.required is True  # Default
        assert string_field.description is None

        # Integer field with minimal config
        int_field = FlextFields.create_integer_field(
            field_id="minimal_int",
            field_name="minimal_int_name",
        )
        if int_field.field_type != FlextFieldType.INTEGER.value:
            raise AssertionError(
                f"Expected {FlextFieldType.INTEGER.value}, got {int_field.field_type}"
            )
        assert int_field.required is True  # Default

        # Boolean field with minimal config
        bool_field = FlextFields.create_boolean_field(
            field_id="minimal_bool",
            field_name="minimal_bool_name",
        )
        if bool_field.field_type != FlextFieldType.BOOLEAN.value:
            raise AssertionError(
                f"Expected {FlextFieldType.BOOLEAN.value}, got {bool_field.field_type}"
            )
        assert bool_field.required is True  # Default


class TestConvenienceFunctions:
    """Test convenience functions with automatic registration."""

    def setUp(self) -> None:
        """Clear registry before each test."""
        FlextFields.clear_registry()

    def test_flext_create_string_field(self) -> None:
        """Test convenience string field creation."""
        field = flext_create_string_field(
            field_id="conv_string",
            field_name="conv_string_name",
            description="Convenience string field",
            min_length=3,
        )

        if field.field_id != "conv_string":
            raise AssertionError(f"Expected {'conv_string'}, got {field.field_id}")
        assert field.field_name == "conv_string_name"
        if field.description != "Convenience string field":
            raise AssertionError(
                f"Expected {'Convenience string field'}, got {field.description}"
            )
        assert field.min_length == EXPECTED_DATA_COUNT

        # Field should be automatically registered
        lookup_result = FlextFields.get_field_by_id("conv_string")
        assert lookup_result.is_success
        if lookup_result.data != field:
            raise AssertionError(f"Expected {field}, got {lookup_result.data}")

    def test_flext_create_integer_field(self) -> None:
        """Test convenience integer field creation."""
        field = flext_create_integer_field(
            field_id="conv_int",
            field_name="conv_int_name",
            description="Convenience integer field",
            min_value=10,
            max_value=100,
        )

        if field.field_id != "conv_int":
            raise AssertionError(f"Expected {'conv_int'}, got {field.field_id}")
        assert field.field_name == "conv_int_name"
        if field.description != "Convenience integer field":
            raise AssertionError(
                f"Expected {'Convenience integer field'}, got {field.description}"
            )
        assert field.min_value == 10.0
        if field.max_value != 100.0:
            raise AssertionError(f"Expected {100.0}, got {field.max_value}")

        # Field should be automatically registered
        lookup_result = FlextFields.get_field_by_id("conv_int")
        assert lookup_result.is_success

    def test_flext_create_boolean_field(self) -> None:
        """Test convenience boolean field creation."""
        field = flext_create_boolean_field(
            field_id="conv_bool",
            field_name="conv_bool_name",
            description="Convenience boolean field",
            default_value=False,
        )

        if field.field_id != "conv_bool":
            raise AssertionError(f"Expected {'conv_bool'}, got {field.field_id}")
        assert field.field_name == "conv_bool_name"
        if field.description != "Convenience boolean field":
            raise AssertionError(
                f"Expected {'Convenience boolean field'}, got {field.description}"
            )
        if field.default_value:
            raise AssertionError(f"Expected False, got {field.default_value}")

        # Field should be automatically registered
        lookup_result = FlextFields.get_field_by_id("conv_bool")
        assert lookup_result.is_success

    def test_convenience_function_registration_conflict(self) -> None:
        """Test convenience function behavior with registration conflicts."""
        # Create first field
        field1 = flext_create_string_field(
            field_id="conflict_conv",
            field_name="conflict_conv_name",
        )
        assert field1 is not None

        # Try to create second field with same ID - should raise exception
        with pytest.raises(FlextValidationError) as exc_info:
            flext_create_string_field(
                field_id="conflict_conv",
                field_name="different_name",
            )

        error = exc_info.value
        if "already registered" not in str(error).lower():
            raise AssertionError(
                f"Expected {'already registered'} in {str(error).lower()}"
            )


class TestFieldEdgeCases:
    """Test edge cases and error conditions."""

    def test_field_with_enum_vs_string_types(self) -> None:
        """Test field behavior with enum vs string types."""
        # Field with enum type
        enum_field = FlextFieldCore(
            field_id="enum_field",
            field_name="enum_name",
            field_type=FlextFieldType.STRING,  # Enum
        )

        # Field with string type
        string_field = FlextFieldCore(
            field_id="string_field",
            field_name="string_name",
            field_type="string",  # String
        )

        # Both should work for serialization/deserialization
        if enum_field.serialize_value("test") != "test":
            raise AssertionError(
                f"Expected {'test'}, got {enum_field.serialize_value('test')}"
            )
        assert string_field.serialize_value("test") == "test"

    def test_field_validation_edge_cases(self) -> None:
        """Test field validation edge cases."""
        field = FlextFieldCore(
            field_id="edge_field",
            field_name="edge_name",
            field_type="unknown_type",  # Unknown type
        )

        # Unknown type should pass validation (fallback behavior)
        is_valid, error = field.validate_field_value("any_value")
        if not (is_valid):
            raise AssertionError(f"Expected True, got {is_valid}")
        assert error is None

    def test_field_serialization_edge_cases(self) -> None:
        """Test serialization edge cases."""
        field = FlextFieldCore(
            field_id="serial_field",
            field_name="serial_name",
            field_type="unknown_type",
        )

        # Unknown type should return value unchanged
        if field.serialize_value("unchanged") != "unchanged":
            raise AssertionError(
                f"Expected {'unchanged'}, got {field.serialize_value('unchanged')}"
            )
        assert field.deserialize_value("unchanged") == "unchanged"

    def test_field_boolean_serialization_edge_cases(self) -> None:
        """Test boolean serialization edge cases."""
        bool_field = FlextFieldCore(
            field_id="bool_edge",
            field_name="bool_edge_name",
            field_type=FlextFieldType.BOOLEAN.value,
        )

        # Test various truthy/falsy string values
        if not (bool_field.serialize_value("TRUE")):
            raise AssertionError(
                f"Expected True, got {bool_field.serialize_value('TRUE')}"
            )
        if bool_field.serialize_value("FALSE"):
            raise AssertionError(
                f"Expected False, got {bool_field.serialize_value('FALSE')}"
            )
        if not (bool_field.serialize_value("ON")):
            raise AssertionError(
                f"Expected True, got {bool_field.serialize_value('ON')}"
            )
        if bool_field.serialize_value("OFF"):
            raise AssertionError(
                f"Expected False, got {bool_field.serialize_value('OFF')}"
            )
        if not (bool_field.serialize_value("YES")):
            raise AssertionError(
                f"Expected True, got {bool_field.serialize_value('YES')}"
            )
        if bool_field.serialize_value("NO"):
            raise AssertionError(
                f"Expected False, got {bool_field.serialize_value('NO')}"
            )
        assert bool_field.serialize_value("random") is False

    def test_registry_edge_cases(self) -> None:
        """Test registry edge cases."""
        registry = FlextFieldRegistry()
        field = FlextFieldCore(
            field_id="edge_registry",
            field_name="edge_registry_name",
            field_type=FlextFieldType.STRING.value,
        )

        registry.register_field(field)

        # Test get_field_by_name with missing field_id in fields_dict
        # (corrupted state)
        # Manually corrupt the registry
        del registry.fields_dict[field.field_id]

        result = registry.get_field_by_name("edge_registry_name")
        assert result.is_failure
        if "not found" not in (result.error or ""):
            raise AssertionError(f"Expected 'not found' in {result.error}")

    def test_field_performance_with_large_allowed_values(self) -> None:
        """Test field performance with large allowed values list."""
        large_allowed_values = [f"value_{i}" for i in range(1000)]

        field = FlextFieldCore(
            field_id="perf_field",
            field_name="perf_name",
            field_type=FlextFieldType.STRING.value,
            allowed_values=large_allowed_values,
        )

        # Should handle large allowed values list efficiently
        is_valid, error = field.validate_field_value("value_500")
        if not (is_valid):
            raise AssertionError(f"Expected True, got {is_valid}")
        assert error is None

        is_valid, error = field.validate_field_value("not_in_list")
        if is_valid:
            raise AssertionError(f"Expected False, got {is_valid}")
        assert "not in allowed list" in error

    def test_field_memory_efficiency(self) -> None:
        """Test field memory efficiency."""
        # Create many fields to test memory usage
        fields = []
        for i in range(100):
            field = FlextFieldCore(
                field_id=f"memory_field_{i}",
                field_name=f"memory_name_{i}",
                field_type=FlextFieldType.STRING.value,
            )
            fields.append(field)

        if len(fields) != 100:
            raise AssertionError(f"Expected {100}, got {len(fields)}")

        # All fields should be properly created
        for i, field in enumerate(fields):
            if field.field_id != f"memory_field_{i}":
                raise AssertionError(
                    f"Expected {f'memory_field_{i}'}, got {field.field_id}"
                )
            assert field.field_name == f"memory_name_{i}"


class TestBackwardCompatibility:
    """Test backward compatibility features."""

    def test_flext_field_core_metadata_alias(self) -> None:
        """Test FlextFieldCoreMetadata alias."""
        # FlextFieldCoreMetadata should be an alias for FlextFieldMetadata
        assert FlextFieldCoreMetadata is FlextFieldMetadata

        # Should work for creating instances
        metadata = FlextFieldCoreMetadata(
            field_id="alias_test",
            field_name="alias_test_name",
        )
        assert isinstance(metadata, FlextFieldMetadata)
        if metadata.field_id != "alias_test":
            raise AssertionError(f"Expected {'alias_test'}, got {metadata.field_id}")

    def test_field_backward_compatibility_methods(self) -> None:
        """Test backward compatibility methods on field."""
        field = FlextFieldCore(
            field_id="compat_field",
            field_name="compat_name",
            field_type=FlextFieldType.STRING.value,
            default_value="default",
            required=False,
            deprecated=True,
            sensitive=True,
        )

        # Test backward compatibility methods
        if field.get_default_value() != "default":
            raise AssertionError(
                f"Expected {'default'}, got {field.get_default_value()}"
            )
        if field.is_required():
            raise AssertionError(f"Expected False, got {field.is_required()}")
        if not (field.is_deprecated()):
            raise AssertionError(f"Expected True, got {field.is_deprecated()}")
        assert field.is_sensitive() is True

        # Test get_field_info comprehensive output
        info = field.get_field_info()
        assert isinstance(info, dict)
        if info["field_id"] != "compat_field":
            raise AssertionError(f"Expected {'compat_field'}, got {info['field_id']}")
        assert info["field_name"] == "compat_name"
        if info["field_type"] != FlextFieldType.STRING.value:
            raise AssertionError(
                f"Expected {FlextFieldType.STRING.value}, got {info['field_type']}"
            )
        if info["required"]:
            raise AssertionError(f"Expected False, got {info['required']}")
        assert info["default"] == "default"
        if "metadata" not in info:
            raise AssertionError(f"Expected {'metadata'} in {info}")
        assert info["has_validator"] is False  # No validator set


class TestIntegrationScenarios:
    """Test integration scenarios between components."""

    def test_complete_field_lifecycle(self) -> None:
        """Test complete field lifecycle from creation to validation."""
        # Clear registry
        FlextFields.clear_registry()

        # Create field with comprehensive configuration
        field = FlextFields.create_string_field(
            field_id="lifecycle_field",
            field_name="lifecycle_name",
            required=True,
            min_length=5,
            max_length=20,
            pattern=r"^[A-Z][a-z]+$",
            allowed_values=["Hello", "World", "Test"],
            description="Lifecycle test field",
            example="Hello",
            tags=["lifecycle", "test"],
        )

        # Register field
        result = FlextFields.register_field(field)
        assert result.is_success

        # Lookup field by ID
        lookup_result = FlextFields.get_field_by_id("lifecycle_field")
        assert lookup_result.is_success
        retrieved_field = lookup_result.data

        # Validate values through retrieved field
        valid_result = retrieved_field.validate_value("Hello")
        assert valid_result.is_success
        if valid_result.data != "Hello":
            raise AssertionError(f"Expected {'Hello'}, got {valid_result.data}")

        # Doesn't match pattern
        invalid_result = retrieved_field.validate_value("hello")
        assert invalid_result.is_failure

        # Test field metadata
        metadata = retrieved_field.metadata
        if metadata.field_id != "lifecycle_field":
            raise AssertionError(
                f"Expected {'lifecycle_field'}, got {metadata.field_id}"
            )
        assert metadata.description == "Lifecycle test field"

    def test_registry_validation_integration(self) -> None:
        """Test registry validation with multiple fields."""
        registry = FlextFieldRegistry()

        # Create multiple fields with different requirements
        string_field = FlextFieldCore(
            field_id="multi_string",
            field_name="multi_string_name",
            field_type=FlextFieldType.STRING.value,
            required=True,
            min_length=3,
        )

        int_field = FlextFieldCore(
            field_id="multi_int",
            field_name="multi_int_name",
            field_type=FlextFieldType.INTEGER.value,
            required=False,
            min_value=10,
            max_value=100,
        )

        bool_field = FlextFieldCore(
            field_id="multi_bool",
            field_name="multi_bool_name",
            field_type=FlextFieldType.BOOLEAN.value,
            required=True,
        )

        # Register all fields
        registry.register_field(string_field)
        registry.register_field(int_field)
        registry.register_field(bool_field)

        # Test comprehensive validation
        valid_data = {
            "multi_string": "hello",
            "multi_int": 50,
            "multi_bool": True,
        }
        result = registry.validate_all_fields(valid_data)
        assert result.is_success

        # Test with missing required field
        incomplete_data = {
            "multi_string": "hello",
            "multi_int": 50,
            # missing multi_bool (required)
        }
        result = registry.validate_all_fields(incomplete_data)
        assert result.is_failure

        # Test with invalid field values
        invalid_data = {
            "multi_string": "hi",  # Too short
            "multi_int": 150,  # Too large
            "multi_bool": True,
        }
        result = registry.validate_all_fields(invalid_data)
        assert result.is_failure

    def test_field_type_filtering(self) -> None:
        """Test filtering fields by type across registry."""
        registry = FlextFieldRegistry()

        # Create fields of different types
        string_fields = [
            FlextFieldCore(
                field_id=f"str_{i}",
                field_name=f"str_name_{i}",
                field_type=FlextFieldType.STRING.value,
            )
            for i in range(3)
        ]
        int_fields = [
            FlextFieldCore(
                field_id=f"int_{i}",
                field_name=f"int_name_{i}",
                field_type=FlextFieldType.INTEGER.value,
            )
            for i in range(2)
        ]
        bool_fields = [
            FlextFieldCore(
                field_id=f"bool_{i}",
                field_name=f"bool_name_{i}",
                field_type=FlextFieldType.BOOLEAN.value,
            )
            for i in range(1)
        ]

        # Register all fields
        for field in string_fields + int_fields + bool_fields:
            registry.register_field(field)

        # Test type filtering
        retrieved_string_fields = registry.get_fields_by_type(FlextFieldType.STRING)
        if len(retrieved_string_fields) != EXPECTED_DATA_COUNT:
            raise AssertionError(f"Expected {3}, got {len(retrieved_string_fields)}")

        retrieved_int_fields = registry.get_fields_by_type("integer")
        if len(retrieved_int_fields) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(retrieved_int_fields)}")

        retrieved_bool_fields = registry.get_fields_by_type(FlextFieldType.BOOLEAN)
        if len(retrieved_bool_fields) != 1:
            raise AssertionError(f"Expected {1}, got {len(retrieved_bool_fields)}")

        # Verify field identity
        for original, _retrieved in zip(
            string_fields,
            retrieved_string_fields,
            strict=False,
        ):
            if original not in retrieved_string_fields:
                raise AssertionError(
                    f"Expected {original} in {retrieved_string_fields}"
                )
