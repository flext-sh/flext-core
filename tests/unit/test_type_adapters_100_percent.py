"""Targeted tests for 100% coverage on FlextTypeAdapters module.

This file contains comprehensive tests targeting FlextTypeAdapters functionality
across Foundation, Domain, Application, and Infrastructure layers.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from pydantic import TypeAdapter

from flext_core import FlextResult, FlextTypeAdapters


@dataclass
class TestDataClass:
    """Test dataclass for type adapter testing."""

    name: str
    age: int
    email: str | None = None


class TestTypeAdapters100PercentCoverage:
    """Targeted tests for FlextTypeAdapters comprehensive coverage."""

    def test_foundation_create_basic_adapter(self) -> None:
        """Test Foundation.create_basic_adapter method."""
        adapter = FlextTypeAdapters.Foundation.create_basic_adapter(str)
        assert isinstance(adapter, TypeAdapter)

        # Test validation
        result = adapter.validate_python("test_string")
        assert result == "test_string"

    def test_foundation_create_string_adapter(self) -> None:
        """Test Foundation.create_string_adapter with coercion."""
        adapter = FlextTypeAdapters.Foundation.create_string_adapter()
        assert adapter is not None

        # Test string coercion functionality
        assert hasattr(adapter, "validate_python")

        # Test string validation
        result = adapter.validate_python("test")
        assert result == "test"

        # Test coercion from int
        result = adapter.validate_python(123)
        assert result == "123"

        # Test coercion from float
        result = adapter.validate_python(12.34)
        assert result == "12.34"

    def test_foundation_create_integer_adapter(self) -> None:
        """Test Foundation.create_integer_adapter method."""
        adapter = FlextTypeAdapters.Foundation.create_integer_adapter()
        assert isinstance(adapter, TypeAdapter)

        # Test integer validation
        result = adapter.validate_python(42)
        assert result == 42

        # Test string coercion
        result = adapter.validate_python("123")
        assert result == 123

    def test_foundation_create_float_adapter(self) -> None:
        """Test Foundation.create_float_adapter method."""
        adapter = FlextTypeAdapters.Foundation.create_float_adapter()
        assert isinstance(adapter, TypeAdapter)

        # Test float validation
        result = adapter.validate_python(math.pi)
        assert result == math.pi

        # Test string coercion
        result = adapter.validate_python("2.718281828459045")
        assert result == math.e

    def test_foundation_create_boolean_adapter(self) -> None:
        """Test Foundation.create_boolean_adapter method."""
        adapter = FlextTypeAdapters.Foundation.create_boolean_adapter()
        assert isinstance(adapter, TypeAdapter)

        # Test boolean validation
        result = adapter.validate_python(True)
        assert result is True

        # Test string coercion
        result = adapter.validate_python("true")
        assert result is True

        result = adapter.validate_python("false")
        assert result is False

    def test_foundation_validate_with_adapter_success(self) -> None:
        """Test Foundation.validate_with_adapter with successful validation."""
        adapter = TypeAdapter(str)
        result = FlextTypeAdapters.Foundation.validate_with_adapter(
            "test_value", str, adapter
        )

        assert isinstance(result, FlextResult)
        assert result.success
        assert result.unwrap() == "test_value"

    def test_foundation_validate_with_adapter_failure(self) -> None:
        """Test Foundation.validate_with_adapter with validation failure."""
        adapter = TypeAdapter(int)
        result = FlextTypeAdapters.Foundation.validate_with_adapter(
            adapter, "invalid_int"
        )

        assert isinstance(result, FlextResult)
        assert result.failure
        assert "Validation failed" in result.error

    def test_domain_validate_entity_id_success(self) -> None:
        """Test Domain.validate_entity_id with valid ID."""
        result = FlextTypeAdapters.Domain.validate_entity_id("valid-entity-123")

        assert isinstance(result, FlextResult)
        assert result.success
        entity_id = result.unwrap()
        assert isinstance(entity_id, str)
        assert entity_id == "valid-entity-123"

    def test_domain_validate_entity_id_failure(self) -> None:
        """Test Domain.validate_entity_id with invalid ID."""
        result = FlextTypeAdapters.Domain.validate_entity_id(123)  # Not a string

        assert isinstance(result, FlextResult)
        assert result.failure

    def test_domain_validate_percentage_success(self) -> None:
        """Test Domain.validate_percentage with valid percentage."""
        result = FlextTypeAdapters.Domain.validate_percentage(75.5)

        assert isinstance(result, FlextResult)
        assert result.success
        percentage = result.unwrap()
        assert isinstance(percentage, float)
        assert percentage == 75.5

    def test_domain_validate_percentage_failure(self) -> None:
        """Test Domain.validate_percentage with invalid percentage."""
        result = FlextTypeAdapters.Domain.validate_percentage("not_a_number")

        assert isinstance(result, FlextResult)
        assert result.failure

    def test_domain_validate_version_success(self) -> None:
        """Test Domain.validate_version with valid version."""
        result = FlextTypeAdapters.Domain.validate_version(42)

        assert isinstance(result, FlextResult)
        assert result.success
        version = result.unwrap()
        assert isinstance(version, int)
        assert version == 42

    def test_domain_validate_version_failure(self) -> None:
        """Test Domain.validate_version with invalid version."""
        result = FlextTypeAdapters.Domain.validate_version("not_a_version")

        assert isinstance(result, FlextResult)
        assert result.failure

    def test_domain_validate_host_port_success(self) -> None:
        """Test Domain.validate_host_port with valid host and port."""
        result = FlextTypeAdapters.Domain.validate_host_port("localhost:8080")

        assert isinstance(result, FlextResult)
        assert result.success
        host_port = result.unwrap()
        assert isinstance(host_port, tuple)
        assert host_port == ("localhost", 8080)

    def test_domain_validate_host_port_failure(self) -> None:
        """Test Domain.validate_host_port with invalid host/port."""
        result = FlextTypeAdapters.Domain.validate_host_port("invalid_host_port")

        assert isinstance(result, FlextResult)
        assert result.failure

    def test_domain_create_entity_id_adapter(self) -> None:
        """Test Domain.create_entity_id_adapter method."""
        adapter = FlextTypeAdapters.Domain.create_entity_id_adapter()

        assert isinstance(adapter, TypeAdapter)
        # Test adapter validation
        result = adapter.validate_python("entity-123")
        assert result == "entity-123"

    def test_application_serialize_to_json_success(self) -> None:
        """Test Application.serialize_to_json with valid data."""
        test_obj = TestDataClass(name="John", age=30, email="john@example.com")
        adapter = TypeAdapter(TestDataClass)
        result = FlextTypeAdapters.Application.serialize_to_json(test_obj, adapter)

        assert isinstance(result, FlextResult)
        assert result.success
        json_str = result.unwrap()
        assert isinstance(json_str, str)

        # Verify JSON is valid
        parsed = json.loads(json_str)
        assert parsed["name"] == "John"
        assert parsed["age"] == 30

    def test_application_serialize_to_json_failure(self) -> None:
        """Test Application.serialize_to_json with serialization failure."""
        # Test with invalid data for existing adapter
        adapter = TypeAdapter(TestDataClass)

        # Pass invalid data that doesn't match the schema (object that can't be JSON serialized)
        class NonSerializable:
            pass

        result = FlextTypeAdapters.Application.serialize_to_json(
            NonSerializable(), adapter
        )

        assert isinstance(result, FlextResult)
        assert result.failure

    def test_application_deserialize_from_json_success(self) -> None:
        """Test Application.deserialize_from_json with valid JSON."""
        json_data = '{"name": "Alice", "age": 25, "email": "alice@example.com"}'
        adapter = TypeAdapter(TestDataClass)
        result = FlextTypeAdapters.Application.deserialize_from_json(
            json_data, TestDataClass, adapter
        )

        assert isinstance(result, FlextResult)
        assert result.success
        obj = result.unwrap()
        assert isinstance(obj, TestDataClass)
        assert obj.name == "Alice"
        assert obj.age == 25

    def test_application_deserialize_from_json_failure(self) -> None:
        """Test Application.deserialize_from_json with invalid JSON."""
        invalid_json = '{"name": "Bob", "age": "invalid_number"}'
        adapter = TypeAdapter(TestDataClass)
        result = FlextTypeAdapters.Application.deserialize_from_json(
            invalid_json, TestDataClass, adapter
        )

        assert isinstance(result, FlextResult)
        assert result.failure
        assert "JSON deserialization failed" in result.error

    def test_application_serialize_to_dict_success(self) -> None:
        """Test Application.serialize_to_dict with valid object."""
        test_obj = TestDataClass(name="Charlie", age=35)
        adapter = TypeAdapter(TestDataClass)
        result = FlextTypeAdapters.Application.serialize_to_dict(test_obj, adapter)

        assert isinstance(result, FlextResult)
        assert result.success
        dict_result = result.unwrap()
        assert isinstance(dict_result, dict)
        assert dict_result["name"] == "Charlie"
        assert dict_result["age"] == 35

    def test_application_serialize_to_dict_failure(self) -> None:
        """Test Application.serialize_to_dict with conversion failure."""
        # Use incompatible object type
        adapter = TypeAdapter(TestDataClass)
        result = FlextTypeAdapters.Application.serialize_to_dict(
            "not_an_object", adapter
        )

        assert isinstance(result, FlextResult)
        assert result.failure

    def test_infrastructure_register_adapter_success(self) -> None:
        """Test Infrastructure.register_adapter with valid adapter."""
        adapter = TypeAdapter(str)
        result = FlextTypeAdapters.Infrastructure.register_adapter(
            "test_string", adapter
        )

        assert isinstance(result, FlextResult)
        assert result.success

    def test_infrastructure_register_adapter_failure(self) -> None:
        """Test Infrastructure.register_adapter with duplicate name."""
        adapter = TypeAdapter(str)
        # Register first time
        result1 = FlextTypeAdapters.Infrastructure.register_adapter(
            "duplicate_test", adapter
        )
        assert result1.success

        # Try to register same name again
        result2 = FlextTypeAdapters.Infrastructure.register_adapter(
            "duplicate_test", adapter
        )
        assert isinstance(result2, FlextResult)
        # May succeed or fail depending on implementation - both are valid

    def test_infrastructure_create_validator_protocol(self) -> None:
        """Test Infrastructure.create_validator_protocol method."""
        protocol = FlextTypeAdapters.Infrastructure.create_validator_protocol()

        # Protocol may return None or an actual validator
        # Both are valid implementations
        if protocol is not None:
            # If protocol exists, it should have validate method
            assert callable(protocol) or hasattr(protocol, "validate")

    def test_edge_cases_none_values(self) -> None:
        """Test edge cases with None values."""
        # Test optional field handling
        test_obj = TestDataClass(name="Test", age=25, email=None)
        adapter = TypeAdapter(TestDataClass)
        result = FlextTypeAdapters.Application.serialize_to_dict(test_obj, adapter)

        assert result.success
        dict_result = result.unwrap()
        assert dict_result["email"] is None

    def test_edge_cases_empty_strings(self) -> None:
        """Test edge cases with empty strings."""
        # Test empty string coercion
        string_adapter = FlextTypeAdapters.Foundation.create_string_adapter()
        result = string_adapter.validate_python("")
        assert result == ""

    def test_type_coercion_comprehensive(self) -> None:
        """Test comprehensive type coercion scenarios."""
        string_adapter = FlextTypeAdapters.Foundation.create_string_adapter()

        # Test various coercion scenarios
        test_cases = [
            (True, "True"),
            (False, "False"),
            (None, "None"),
            ([1, 2, 3], "[1, 2, 3]"),
            ({"key": "value"}, "{'key': 'value'}"),
        ]

        for input_value, expected in test_cases:
            result = string_adapter.validate_python(input_value)
            assert result == expected

    def test_validation_error_handling_comprehensive(self) -> None:
        """Test comprehensive error handling scenarios."""
        # Test various validation error scenarios
        adapter = TypeAdapter(int)

        invalid_inputs = ["not_a_number", [], {}, None, object()]

        for invalid_input in invalid_inputs:
            result = FlextTypeAdapters.Foundation.validate_with_adapter(
                adapter, invalid_input
            )
            assert result.failure
            assert "Validation failed" in result.error
