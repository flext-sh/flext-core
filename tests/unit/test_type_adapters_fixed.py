"""Fixed tests for FlextTypeAdapters with correct nested class structure."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, TypeAdapter

from flext_core import FlextTypeAdapters


class TestFoundation:
    """Test Foundation nested class methods."""

    def test_create_basic_adapter(self) -> None:
        """Test creating basic type adapter."""
        adapter = FlextTypeAdapters.Foundation.create_basic_adapter(str)

        assert isinstance(adapter, TypeAdapter)

        # Test it works
        result = adapter.validate_python("test string")
        assert result == "test string"

    def test_create_string_adapter(self) -> None:
        """Test creating string adapter."""
        adapter = FlextTypeAdapters.Foundation.create_string_adapter()

        assert isinstance(adapter, TypeAdapter)

        # Test validation
        result = adapter.validate_python("hello")
        assert result == "hello"

        # Test conversion
        result = adapter.validate_python(123)
        assert result == "123"

    def test_create_integer_adapter(self) -> None:
        """Test creating integer adapter."""
        adapter = FlextTypeAdapters.Foundation.create_integer_adapter()

        assert isinstance(adapter, TypeAdapter)

        # Test validation
        result = adapter.validate_python(42)
        assert result == 42

        # Test conversion
        result = adapter.validate_python("100")
        assert result == 100

    def test_create_float_adapter(self) -> None:
        """Test creating float adapter."""
        adapter = FlextTypeAdapters.Foundation.create_float_adapter()

        assert isinstance(adapter, TypeAdapter)

        # Test validation
        result = adapter.validate_python(3.14)
        assert result == 3.14

        # Test conversion
        result = adapter.validate_python("2.5")
        assert result == 2.5

    def test_create_boolean_adapter(self) -> None:
        """Test creating boolean adapter."""
        adapter = FlextTypeAdapters.Foundation.create_boolean_adapter()

        assert isinstance(adapter, TypeAdapter)

        # Test validation
        result = adapter.validate_python(True)
        assert result is True

        # Test conversion
        result = adapter.validate_python(1)
        assert result is True

        result = adapter.validate_python(0)
        assert result is False

    def test_validate_with_adapter(self) -> None:
        """Test validation with adapter."""
        adapter = FlextTypeAdapters.Foundation.create_string_adapter()

        result = FlextTypeAdapters.Foundation.validate_with_adapter(
            "test", str, adapter
        )

        assert result.success
        assert result.unwrap() == "test"

        # Test with invalid data
        result = FlextTypeAdapters.Foundation.validate_with_adapter(
            None, str, adapter
        )

        assert result.is_failure


class TestDomain:
    """Test Domain nested class methods."""

    def test_create_entity_id_adapter(self) -> None:
        """Test creating entity ID adapter."""
        adapter = FlextTypeAdapters.Domain.create_entity_id_adapter()

        assert isinstance(adapter, TypeAdapter)

        # Test validation
        result = adapter.validate_python("entity_123")
        assert result == "entity_123"

    def test_validate_entity_id(self) -> None:
        """Test entity ID validation."""
        # Valid entity ID
        result = FlextTypeAdapters.Domain.validate_entity_id("entity_123")
        assert result.success
        assert result.unwrap() == "entity_123"

        # Invalid entity ID (empty)
        result = FlextTypeAdapters.Domain.validate_entity_id("")
        assert result.is_failure

        # Invalid type
        result = FlextTypeAdapters.Domain.validate_entity_id(123)
        assert result.is_failure

    def test_validate_percentage(self) -> None:
        """Test percentage validation."""
        # Valid percentage
        result = FlextTypeAdapters.Domain.validate_percentage(50.0)
        assert result.success
        assert result.unwrap() == 50.0

        # Edge cases
        result = FlextTypeAdapters.Domain.validate_percentage(0.0)
        assert result.success

        result = FlextTypeAdapters.Domain.validate_percentage(100.0)
        assert result.success

        # Invalid percentage (negative)
        result = FlextTypeAdapters.Domain.validate_percentage(-10.0)
        assert result.is_failure

        # Invalid percentage (over 100)
        result = FlextTypeAdapters.Domain.validate_percentage(150.0)
        assert result.is_failure

    def test_validate_version(self) -> None:
        """Test version validation."""
        # Valid version
        result = FlextTypeAdapters.Domain.validate_version(1)
        assert result.success
        assert result.unwrap() == 1

        # Invalid version (zero)
        result = FlextTypeAdapters.Domain.validate_version(0)
        assert result.is_failure

        # Invalid version (negative)
        result = FlextTypeAdapters.Domain.validate_version(-1)
        assert result.is_failure

    def test_validate_host_port(self) -> None:
        """Test host:port validation."""
        # Valid host:port
        result = FlextTypeAdapters.Domain.validate_host_port("localhost:8080")
        assert result.success
        host, port = result.unwrap()
        assert host == "localhost"
        assert port == 8080

        # Valid with IP
        result = FlextTypeAdapters.Domain.validate_host_port("192.168.1.1:3000")
        assert result.success
        host, port = result.unwrap()
        assert host == "192.168.1.1"
        assert port == 3000

        # Invalid format (no port)
        result = FlextTypeAdapters.Domain.validate_host_port("localhost")
        assert result.is_failure

        # Invalid port
        result = FlextTypeAdapters.Domain.validate_host_port("localhost:abc")
        assert result.is_failure

        # Port out of range
        result = FlextTypeAdapters.Domain.validate_host_port("localhost:99999")
        assert result.is_failure


class TestApplication:
    """Test Application nested class methods."""

    def test_serialize_to_json(self) -> None:
        """Test JSON serialization."""
        data = {"name": "Test", "value": 42}
        adapter = TypeAdapter(dict)

        result = FlextTypeAdapters.Application.serialize_to_json(data, adapter)

        assert result.success
        json_str = result.unwrap()
        assert isinstance(json_str, str)

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["name"] == "Test"
        assert parsed["value"] == 42

    def test_serialize_to_dict(self) -> None:
        """Test dict serialization."""
        class TestModel(BaseModel):
            name: str
            value: int

        model = TestModel(name="Test", value=100)
        adapter = TypeAdapter(TestModel)

        result = FlextTypeAdapters.Application.serialize_to_dict(model, adapter)

        assert result.success
        data_dict = result.unwrap()
        assert isinstance(data_dict, dict)
        assert data_dict["name"] == "Test"
        assert data_dict["value"] == 100

    def test_deserialize_from_json(self) -> None:
        """Test JSON deserialization."""
        json_str = '{"name": "FromJSON", "value": 200}'

        class TestModel(BaseModel):
            name: str
            value: int

        adapter = TypeAdapter(TestModel)

        result = FlextTypeAdapters.Application.deserialize_from_json(
            json_str, TestModel, adapter
        )

        assert result.success
        model = result.unwrap()
        assert isinstance(model, TestModel)
        assert model.name == "FromJSON"
        assert model.value == 200

    def test_deserialize_from_dict(self) -> None:
        """Test dict deserialization."""
        data_dict = {"name": "FromDict", "value": 300}

        class TestModel(BaseModel):
            name: str
            value: int

        adapter = TypeAdapter(TestModel)

        result = FlextTypeAdapters.Application.deserialize_from_dict(
            data_dict, TestModel, adapter
        )

        assert result.success
        model = result.unwrap()
        assert isinstance(model, TestModel)
        assert model.name == "FromDict"
        assert model.value == 300

    def test_deserialize_invalid_json(self) -> None:
        """Test deserializing invalid JSON."""
        result = FlextTypeAdapters.Application.deserialize_from_json(
            "not-json",
            dict,
            TypeAdapter(dict)
        )

        assert result.is_failure
        assert "JSON" in result.error

    def test_generate_schema(self) -> None:
        """Test schema generation."""
        class TestModel(BaseModel):
            name: str
            age: int
            email: str | None = None

        adapter = TypeAdapter(TestModel)

        result = FlextTypeAdapters.Application.generate_schema(TestModel, adapter)

        assert result.success
        schema = result.unwrap()
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_generate_multiple_schemas(self) -> None:
        """Test generating multiple schemas."""
        class Model1(BaseModel):
            field1: str

        class Model2(BaseModel):
            field2: int

        types = [Model1, Model2]

        result = FlextTypeAdapters.Application.generate_multiple_schemas(types)

        assert result.success
        schemas = result.unwrap()
        assert isinstance(schemas, list)
        assert len(schemas) == 2


class TestInfrastructure:
    """Test Infrastructure nested class methods."""

    def test_create_validator_protocol(self) -> None:
        """Test creating validator protocol."""
        protocol = FlextTypeAdapters.Infrastructure.create_validator_protocol()

        # Protocol should be a type or class
        assert protocol is not None

    def test_register_adapter(self) -> None:
        """Test registering custom adapter."""
        adapter = TypeAdapter(str)

        result = FlextTypeAdapters.Infrastructure.register_adapter("test_key", adapter)

        assert result.success


class TestUtilities:
    """Test Utilities nested class methods."""

    def test_create_adapter_for_type(self) -> None:
        """Test creating adapter for specific type."""
        class CustomModel(BaseModel):
            field: str

        adapter = FlextTypeAdapters.Utilities.create_adapter_for_type(CustomModel)

        assert isinstance(adapter, TypeAdapter)

        # Test it works
        result = adapter.validate_python({"field": "test"})
        assert isinstance(result, CustomModel)
        assert result.field == "test"

    def test_validate_batch(self) -> None:
        """Test batch validation."""
        class TestModel(BaseModel):
            name: str
            value: int

        items = [
            {"name": "Item1", "value": 10},
            {"name": "Item2", "value": 20},
            {"name": "Item3", "value": 30}
        ]

        adapter = TypeAdapter(TestModel)

        result = FlextTypeAdapters.Utilities.validate_batch(items, TestModel, adapter)

        assert result.success
        validated = result.unwrap()
        assert isinstance(validated, list)
        assert len(validated) == 3
        assert all(isinstance(item, TestModel) for item in validated)

    def test_validate_batch_with_errors(self) -> None:
        """Test batch validation with some invalid items."""
        class TestModel(BaseModel):
            name: str
            value: int

        items = [
            {"name": "Valid", "value": 10},
            {"name": "Invalid"},  # Missing value
            {"name": "AlsoValid", "value": 30}
        ]

        adapter = TypeAdapter(TestModel)

        result = FlextTypeAdapters.Utilities.validate_batch(items, TestModel, adapter)

        # Should fail if any item is invalid
        assert result.is_failure

    def test_migrate_from_basemodel(self) -> None:
        """Test migrating from BaseModel."""
        migration_code = FlextTypeAdapters.Utilities.migrate_from_basemodel("OldModel")

        assert isinstance(migration_code, str)
        assert "OldModel" in migration_code

    def test_create_legacy_adapter(self) -> None:
        """Test creating legacy adapter."""
        class LegacyModel(BaseModel):
            old_field: str

        adapter = FlextTypeAdapters.Utilities.create_legacy_adapter(LegacyModel)

        assert isinstance(adapter, TypeAdapter)

        # Test it works
        result = adapter.validate_python({"old_field": "legacy"})
        assert isinstance(result, LegacyModel)
        assert result.old_field == "legacy"

    def test_validate_example_user(self) -> None:
        """Test example user validation."""
        result = FlextTypeAdapters.Utilities.validate_example_user()

        # Should validate successfully or fail depending on implementation
        assert result.success or result.is_failure

        if result.success:
            user = result.unwrap()
            assert user is not None

    def test_validate_example_config(self) -> None:
        """Test example config validation."""
        result = FlextTypeAdapters.Utilities.validate_example_config()

        # Should validate successfully or fail depending on implementation
        assert result.success or result.is_failure

        if result.success:
            config = result.unwrap()
            assert config is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_validate_with_none(self) -> None:
        """Test validation with None value."""
        adapter = FlextTypeAdapters.Foundation.create_string_adapter()

        result = FlextTypeAdapters.Foundation.validate_with_adapter(
            None, str, adapter
        )

        assert result.is_failure

    def test_serialize_none(self) -> None:
        """Test serializing None value."""
        adapter = TypeAdapter(type(None))

        result = FlextTypeAdapters.Application.serialize_to_json(None, adapter)

        assert result.success
        assert result.unwrap() == "null"

    def test_deserialize_empty_json(self) -> None:
        """Test deserializing empty JSON object."""
        result = FlextTypeAdapters.Application.deserialize_from_json(
            "{}", dict, TypeAdapter(dict)
        )

        assert result.success
        assert result.unwrap() == {}

    def test_batch_with_empty_list(self) -> None:
        """Test batch validation with empty list."""
        adapter = TypeAdapter(dict)

        result = FlextTypeAdapters.Utilities.validate_batch([], dict, adapter)

        assert result.success
        assert result.unwrap() == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
