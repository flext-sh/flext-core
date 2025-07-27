"""Comprehensive tests for FlextPayload."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from flext_core.payload import FlextPayload


class TestFlextPayloadCreation:
    """Test FlextPayload creation and basic functionality."""

    def test_empty_payload_creation(self) -> None:
        """Test creating an empty payload."""
        payload = FlextPayload()

        assert isinstance(payload, FlextPayload)
        assert payload.keys() == []
        assert payload.items() == []

    def test_payload_with_kwargs(self) -> None:
        """Test creating payload with keyword arguments."""
        payload = FlextPayload(
            user_id="123",
            action="login",
            timestamp="2025-01-01T00:00:00Z",
        )

        assert payload.user_id == "123"
        assert payload.action == "login"
        assert payload.timestamp == "2025-01-01T00:00:00Z"

    def test_payload_with_mixed_types(self) -> None:
        """Test payload with different data types."""
        payload = FlextPayload(
            string_field="text",
            int_field=42,
            bool_field=True,
            float_field=math.pi,
            list_field=[1, 2, 3],
            dict_field={"nested": "value"},
        )

        assert payload.string_field == "text"
        assert payload.int_field == 42
        assert payload.bool_field is True
        assert payload.float_field == math.pi
        assert payload.list_field == [1, 2, 3]
        assert payload.dict_field == {"nested": "value"}

    def test_payload_immutability(self) -> None:
        """Test that payload is immutable after creation."""
        payload = FlextPayload(user_id="123")

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            payload.user_id = "456"

    def test_payload_string_stripping(self) -> None:
        """Test string whitespace preservation in extra fields."""
        payload = FlextPayload(
            normal_field="  trimmed  ",
            empty_field="   ",
        )

        # Extra fields preserve original string values
        assert payload.normal_field == "  trimmed  "
        assert payload.empty_field == "   "


class TestFlextPayloadAttributeAccess:
    """Test FlextPayload attribute access methods."""

    def test_getattr_existing_field(self) -> None:
        """Test __getattr__ for existing fields."""
        payload = FlextPayload(user_id="123", name="John")

        assert payload.user_id == "123"
        assert payload.name == "John"

    def test_getattr_nonexistent_field(self) -> None:
        """Test __getattr__ for non-existent fields."""
        payload = FlextPayload(user_id="123")

        with pytest.raises(AttributeError, match="object has no attribute"):
            _ = payload.nonexistent_field

    def test_has_method_existing_field(self) -> None:
        """Test has() method for existing fields."""
        payload = FlextPayload(user_id="123", active=True)

        assert payload.has("user_id") is True
        assert payload.has("active") is True

    def test_has_method_nonexistent_field(self) -> None:
        """Test has() method for non-existent fields."""
        payload = FlextPayload(user_id="123")

        assert payload.has("nonexistent") is False
        assert payload.has("missing_field") is False

    def test_has_method_empty_payload(self) -> None:
        """Test has() method on empty payload."""
        payload = FlextPayload()

        assert payload.has("any_field") is False

    def test_get_method_existing_field(self) -> None:
        """Test get() method for existing fields."""
        payload = FlextPayload(user_id="123", count=42)

        assert payload.get("user_id") == "123"
        assert payload.get("count") == 42

    def test_get_method_nonexistent_field_no_default(self) -> None:
        """Test get() method for non-existent field without default."""
        payload = FlextPayload(user_id="123")

        assert payload.get("nonexistent") is None

    def test_get_method_nonexistent_field_with_default(self) -> None:
        """Test get() method for non-existent field with default."""
        payload = FlextPayload(user_id="123")

        assert payload.get("nonexistent", "default") == "default"
        assert payload.get("missing", 42) == 42
        assert payload.get("absent", []) == []

    def test_get_method_empty_payload(self) -> None:
        """Test get() method on empty payload."""
        payload = FlextPayload()

        assert payload.get("any_field") is None
        assert payload.get("any_field", "default") == "default"


class TestFlextPayloadIteration:
    """Test FlextPayload iteration and collection methods."""

    def test_keys_method_with_data(self) -> None:
        """Test keys() method with data."""
        payload = FlextPayload(
            user_id="123",
            action="login",
            timestamp="2025-01-01",
        )

        keys = payload.keys()
        assert isinstance(keys, list)
        assert set(keys) == {"user_id", "action", "timestamp"}

    def test_keys_method_empty_payload(self) -> None:
        """Test keys() method on empty payload."""
        payload = FlextPayload()

        keys = payload.keys()
        assert keys == []

    def test_items_method_with_data(self) -> None:
        """Test items() method with data."""
        payload = FlextPayload(user_id="123", active=True)

        items = payload.items()
        assert isinstance(items, list)
        assert len(items) == 2
        assert ("user_id", "123") in items
        assert ("active", True) in items

    def test_items_method_empty_payload(self) -> None:
        """Test items() method on empty payload."""
        payload = FlextPayload()

        items = payload.items()
        assert items == []

    def test_keys_and_items_consistency(self) -> None:
        """Test that keys() and items() are consistent."""
        payload = FlextPayload(
            field1="value1",
            field2="value2",
            field3=42,
        )

        keys = payload.keys()
        items = payload.items()

        # Same number of elements
        assert len(keys) == len(items)

        # All keys from keys() should be in items()
        item_keys = [item[0] for item in items]
        assert set(keys) == set(item_keys)


class TestFlextPayloadEdgeCases:
    """Test FlextPayload edge cases and special scenarios."""

    def test_payload_with_none_values(self) -> None:
        """Test payload with None values."""
        payload = FlextPayload(
            valid_field="value",
            none_field=None,
        )

        assert payload.valid_field == "value"
        assert payload.none_field is None
        assert payload.has("none_field") is True
        assert payload.get("none_field") is None

    def test_payload_with_empty_string_key(self) -> None:
        """Test payload behavior with empty string as field name."""
        # This should work since pydantic allows empty string fields
        payload = FlextPayload(**{"": "empty_key_value"})

        assert payload.has("") is True
        assert payload.get("") == "empty_key_value"
        assert "" in payload

    def test_payload_with_special_characters_in_keys(self) -> None:
        """Test payload with special characters in field names."""
        payload = FlextPayload(
            **{
                "field-with-dash": "value1",
                "field.with.dot": "value2",
                "field_with_underscore": "value3",
            },
        )

        assert payload.get("field-with-dash") == "value1"
        assert payload.get("field.with.dot") == "value2"
        assert payload.get("field_with_underscore") == "value3"

    def test_payload_with_large_number_of_fields(self) -> None:
        """Test payload with many fields."""
        fields = {f"field_{i}": f"value_{i}" for i in range(100)}
        payload = FlextPayload(**fields)

        assert len(payload.keys()) == 100
        assert len(payload.items()) == 100

        # Check a few random fields
        assert payload.get("field_0") == "value_0"
        assert payload.get("field_50") == "value_50"
        assert payload.get("field_99") == "value_99"


class TestFlextPayloadIntegration:
    """Test FlextPayload integration with other systems."""

    def test_payload_json_serialization(self) -> None:
        """Test that payload can be serialized to JSON."""
        payload = FlextPayload(
            user_id="123",
            action="login",
            metadata={"ip": "192.168.1.1", "agent": "browser"},
        )

        # Test model_dump
        data = payload.model_dump()
        assert isinstance(data, dict)
        assert data["user_id"] == "123"
        assert data["action"] == "login"
        assert data["metadata"]["ip"] == "192.168.1.1"

    def test_payload_creation_from_dict(self) -> None:
        """Test creating payload from dictionary."""
        data = {
            "user_id": "123",
            "permissions": ["read", "write"],
            "settings": {"theme": "dark", "lang": "en"},
        }

        payload = FlextPayload(**data)

        assert payload.user_id == "123"
        assert payload.permissions == ["read", "write"]
        assert payload.settings == {"theme": "dark", "lang": "en"}

    def test_payload_repr_and_str(self) -> None:
        """Test string representation of payload."""
        payload = FlextPayload(user_id="123", action="login")

        # These should not raise exceptions
        str_repr = str(payload)
        repr_str = repr(payload)

        assert isinstance(str_repr, str)
        assert isinstance(repr_str, str)
        assert "FlextPayload" in repr_str

    def test_payload_equality(self) -> None:
        """Test payload equality comparison."""
        payload1 = FlextPayload(user_id="123", action="login")
        payload2 = FlextPayload(user_id="123", action="login")
        payload3 = FlextPayload(user_id="456", action="login")

        assert payload1 == payload2
        assert payload1 != payload3
        assert payload2 != payload3

    def test_payload_hash(self) -> None:
        """Test that payload can be hashed (since it's frozen)."""
        payload = FlextPayload(user_id="123", action="login")

        # Should not raise an exception
        hash_value = hash(payload)
        assert isinstance(hash_value, int)

        # Same payload should have same hash
        payload2 = FlextPayload(user_id="123", action="login")
        assert hash(payload) == hash(payload2)

    def test_empty_payload_edge_cases(self) -> None:
        """Test empty payload edge cases for full coverage."""
        # Create payload with no extra fields to test __pydantic_extra__ = None
        payload = FlextPayload()

        # Test get method when __pydantic_extra__ is None (line 116)
        assert payload.get("missing_field") is None
        assert payload.get("missing_field", "default") == "default"

        # Test keys method when __pydantic_extra__ is None (line 133)
        assert payload.keys() == []

        # Test items method when __pydantic_extra__ is None (line 150)
        assert payload.items() == []

        # Test has method on empty payload
        assert payload.has("any_field") is False
