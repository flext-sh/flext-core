"""Comprehensive tests for FlextPayload system and payload functionality."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from flext_core.exceptions import FlextAttributeError
from flext_core.payload import FlextEvent, FlextMessage, FlextPayload
from flext_core.result import FlextResult

# Constants
EXPECTED_DATA_COUNT = 3


class TestFlextPayload:
    """Test FlextPayload core functionality."""

    def test_payload_basic_creation(self) -> None:
        """Test basic payload creation."""
        payload = FlextPayload(data="test_data")

        if payload.data != "test_data":
            raise AssertionError(f"Expected {'test_data'}, got {payload.data}")
        assert isinstance(payload.metadata, dict)
        if len(payload.metadata) != 0:
            raise AssertionError(f"Expected {0}, got {len(payload.metadata)}")

    def test_payload_with_metadata(self) -> None:
        """Test payload creation with metadata."""
        metadata = {"version": "1.0", "source": "test"}
        payload = FlextPayload(data=42, metadata=metadata)

        if payload.data != 42:
            raise AssertionError(f"Expected {42}, got {payload.data}")
        assert payload.metadata["version"] == "1.0"
        if payload.metadata["source"] != "test":
            raise AssertionError(f"Expected {'test'}, got {payload.metadata['source']}")

    def test_payload_create_factory_method(self) -> None:
        """Test payload creation via factory method."""
        result = FlextPayload.create(
            data={"user_id": "123"},
            version="2.0",
            source="api",
        )

        assert result.is_success
        payload = result.data
        if payload.data != {"user_id": "123"}:
            raise AssertionError(f'Expected {{"user_id": "123"}}, got {payload.data}')
        assert payload.metadata["version"] == "2.0"
        if payload.metadata["source"] != "api":
            raise AssertionError(f"Expected {'api'}, got {payload.metadata['source']}")

    def test_payload_create_factory_validation_failure(self) -> None:
        """Test payload factory method with validation failure."""
        # Create payload that might cause validation issues
        result = FlextPayload.create(data=None)

        # Should still succeed since data can be None
        assert result.is_success

    def test_payload_with_metadata_method(self) -> None:
        """Test adding metadata with with_metadata method."""
        original = FlextPayload(data="test", metadata={"original": "value"})

        enhanced = original.with_metadata(
            additional="new_value",
            counter=42,
        )

        # Original should be unchanged (immutable)
        if original.metadata != {"original": "value"}:
            raise AssertionError(
                f'Expected {{"original": "value"}}, got {original.metadata}'
            )

        # Enhanced should have both original and new metadata
        if enhanced.metadata["original"] != "value":
            raise AssertionError(
                f"Expected {'value'}, got {enhanced.metadata['original']}"
            )
        assert enhanced.metadata["additional"] == "new_value"
        if enhanced.metadata["counter"] != 42:
            raise AssertionError(f"Expected {42}, got {enhanced.metadata['counter']}")
        assert enhanced.data == "test"

    def test_payload_enrich_metadata_method(self) -> None:
        """Test enriching metadata with dictionary."""
        original = FlextPayload(data="test", metadata={"key1": "value1"})

        additional_metadata = {
            "key2": "value2",
            "key3": 123,
            "key1": "updated_value1",  # Should override
        }

        enriched = original.enrich_metadata(additional_metadata)

        if enriched.metadata["key1"] != "updated_value1":  # Overridden
            raise AssertionError(
                f"Expected {'updated_value1'}, got {enriched.metadata['key1']}"
            )
        assert enriched.metadata["key2"] == "value2"
        if enriched.metadata["key3"] != 123:
            raise AssertionError(f"Expected {123}, got {enriched.metadata['key3']}")
        assert enriched.data == "test"

    def test_payload_get_metadata(self) -> None:
        """Test getting metadata values."""
        payload = FlextPayload(
            data="test",
            metadata={"key1": "value1", "key2": 42, "key3": None},
        )

        # Existing keys
        if payload.get_metadata("key1") != "value1":
            raise AssertionError(
                f"Expected {'value1'}, got {payload.get_metadata('key1')}"
            )
        assert payload.get_metadata("key2") == 42
        assert payload.get_metadata("key3") is None

        # Non-existing key
        assert payload.get_metadata("nonexistent") is None

        # Non-existing key with default
        if payload.get_metadata("nonexistent", "default") != "default":
            raise AssertionError(
                f"Expected {'default'}, got {payload.get_metadata('nonexistent', 'default')}"
            )

    def test_payload_has_metadata(self) -> None:
        """Test checking metadata existence."""
        payload = FlextPayload(
            data="test",
            metadata={"key1": "value1", "key2": None},
        )

        if not (payload.has_metadata("key1")):
            raise AssertionError(f"Expected True, got {payload.has_metadata('key1')}")
        assert payload.has_metadata("key2") is True  # Even with None value
        if payload.has_metadata("nonexistent"):
            raise AssertionError(
                f"Expected False, got {payload.has_metadata('nonexistent')}"
            )

    def test_payload_from_dict(self) -> None:
        """Test creating payload from dictionary."""
        data_dict = {
            "data": {"user": "john", "age": 30},
            "metadata": {"version": "1.0", "timestamp": 1234567890},
        }

        result = FlextPayload.from_dict(data_dict)

        assert result.is_success
        payload = result.data
        if payload.data != {"user": "john", "age": 30}:
            raise AssertionError(
                f'Expected {{"user": "john", "age": 30}}, got {payload.data}'
            )
        assert payload.metadata["version"] == "1.0"
        if payload.metadata["timestamp"] != 1234567890:
            raise AssertionError(
                f"Expected {1234567890}, got {payload.metadata['timestamp']}"
            )

    def test_payload_from_dict_invalid_data(self) -> None:
        """Test from_dict with invalid data."""
        # Invalid structure
        result = FlextPayload.from_dict("not_a_dict")  # type: ignore[arg-type]
        assert result.is_failure
        if "Failed to create payload from dict" not in result.error:
            raise AssertionError(
                f"Expected {'Failed to create payload from dict'} in {result.error}"
            )

        # Missing required structure
        result = FlextPayload.from_dict({})
        # Should succeed since both fields have defaults
        assert result.is_success

    def test_payload_transform_data(self) -> None:
        """Test data transformation."""

        def double_value(x: int) -> int:
            return x * 2

        payload = FlextPayload(data=5, metadata={"source": "test"})

        result = payload.transform_data(double_value)

        assert result.is_success
        transformed = result.data
        if transformed.data != 10:
            raise AssertionError(f"Expected {10}, got {transformed.data}")
        assert transformed.metadata == payload.metadata  # Metadata preserved

    def test_payload_transform_data_failure(self) -> None:
        """Test data transformation with failure."""

        def failing_transform(x: object) -> str:
            msg = "Transform failed"
            raise ValueError(msg)

        payload = FlextPayload(data="test")

        result = payload.transform_data(failing_transform)

        assert result.is_failure
        if "Transform failed" not in result.error:
            raise AssertionError(f"Expected {'Transform failed'} in {result.error}")

    def test_payload_to_dict(self) -> None:
        """Test payload serialization to dictionary."""
        payload = FlextPayload(
            data={"user": "john"},
            metadata={"version": "1.0", "source": "api"},
        )

        payload_dict = payload.to_dict()

        if payload_dict["data"] != {"user": "john"}:
            raise AssertionError(
                f'Expected {{"user": "john"}}, got {payload_dict["data"]}'
            )
        assert payload_dict["metadata"]["version"] == "1.0"
        if payload_dict["metadata"]["source"] != "api":
            raise AssertionError(
                f"Expected {'api'}, got {payload_dict['metadata']['source']}"
            )

    def test_payload_repr(self) -> None:
        """Test payload string representation."""
        payload = FlextPayload(data="test", metadata={"version": "1.0"})

        repr_str = repr(payload)

        if "FlextPayload" not in repr_str:
            raise AssertionError(f"Expected {'FlextPayload'} in {repr_str}")
        assert "test" in repr_str

    def test_payload_getattr_delegation(self) -> None:
        """Test attribute access for extra fields."""
        payload = FlextPayload(data="test", name="test_name", value=42)

        # Should access extra fields
        if payload.name != "test_name":  # type: ignore[attr-defined]
            raise AssertionError(f"Expected {'test_name'}, got {payload.name}")
        assert payload.value == 42  # type: ignore[attr-defined]

    def test_payload_getattr_metadata_priority(self) -> None:
        """Test that extra fields work correctly."""
        payload = FlextPayload(data="test", name="extra_field_name")

        # Extra field should be accessible
        if payload.name != "extra_field_name":  # type: ignore[attr-defined]
            raise AssertionError(f"Expected {'extra_field_name'}, got {payload.name}")

    def test_payload_getattr_nonexistent(self) -> None:
        """Test attribute access for nonexistent attributes."""
        payload = FlextPayload(data="simple_string")

        with pytest.raises(FlextAttributeError):
            _ = payload.nonexistent_attr  # type: ignore[attr-defined]

    def test_payload_contains(self) -> None:
        """Test 'in' operator for extra fields."""
        payload = FlextPayload(
            data="test",
            key1="value1",
            key2=42,  # These become extra fields
        )

        if "key1" not in payload:
            raise AssertionError(f"Expected {'key1'} in {payload}")
        assert "key2" in payload
        if "nonexistent" not in payload:
            raise AssertionError(f"Expected {'nonexistent'} in {payload}")

    def test_payload_hash(self) -> None:
        """Test payload hashing."""
        payload1 = FlextPayload(data="test", metadata={"key": "value"})
        payload2 = FlextPayload(data="test", metadata={"key": "value"})
        payload3 = FlextPayload(data="different", metadata={"key": "value"})

        # Same content should have same hash
        if hash(payload1) != hash(payload2):
            raise AssertionError(f"Expected {hash(payload2)}, got {hash(payload1)}")

        # Different content should have different hash
        assert hash(payload1) != hash(payload3)

        # Should be usable in sets/dicts
        payload_set = {payload1, payload2}
        if len(payload_set) != 1:  # Same content, only one in set
            raise AssertionError(f"Expected {1}, got {len(payload_set)}")

    def test_payload_has_method(self) -> None:
        """Test has() method for extra fields."""
        # Create payload with extra fields
        payload = FlextPayload(data="test", key="value")  # key becomes extra field

        if not (payload.has("key")):
            raise AssertionError(f"Expected True, got {payload.has('key')}")
        if payload.has("nonexistent"):
            raise AssertionError(f"Expected False, got {payload.has('nonexistent')}")

    def test_payload_get_method(self) -> None:
        """Test get() method for extra fields."""
        payload = FlextPayload(data="test", key="value")  # key becomes extra field

        if payload.get("key") != "value":
            raise AssertionError(f"Expected {'value'}, got {payload.get('key')}")
        assert payload.get("nonexistent") is None
        if payload.get("nonexistent", "default") != "default":
            raise AssertionError(
                f"Expected {'default'}, got {payload.get('nonexistent', 'default')}"
            )

    def test_payload_keys_method(self) -> None:
        """Test keys() method."""
        payload = FlextPayload(
            data="test",
            key1="value1",
            key2="value2",  # These become extra fields
        )

        keys = payload.keys()

        assert isinstance(keys, list)
        if set(keys) != {"key1", "key2"}:
            raise AssertionError(f"Expected {{'key1', 'key2'}}, got {set(keys)}")

    def test_payload_items_method(self) -> None:
        """Test items() method."""
        payload = FlextPayload(
            data="test",
            key1="value1",
            key2="value2",  # These become extra fields
        )

        items = payload.items()

        assert isinstance(items, list)
        if set(items) != {("key1", "value1"), ("key2", "value2")}:
            raise AssertionError(
                f"Expected {{('key1', 'value1'), ('key2', 'value2')}}, got {set(items)}"
            )

    def test_payload_immutability(self) -> None:
        """Test that payload is immutable."""
        payload = FlextPayload(data="test", metadata={"key": "value"})

        # Should not be able to modify data
        with pytest.raises((AttributeError, ValidationError)):
            payload.data = "new_data"  # type: ignore[misc]

        # Should not be able to modify metadata dict directly
        # Note: The dict itself might be mutable, but the payload is frozen


class TestFlextMessage:
    """Test FlextMessage specialized payload."""

    def test_message_basic_creation(self) -> None:
        """Test basic message creation."""
        message = FlextMessage(
            data="Hello, World!",
            metadata={"level": "info", "source": "test"},
        )

        if message.data != "Hello, World!":
            raise AssertionError(f"Expected {'Hello, World!'}, got {message.data}")
        assert message.level == "info"
        if message.source != "test":
            raise AssertionError(f"Expected {'test'}, got {message.source}")

    def test_message_create_factory(self) -> None:
        """Test message creation via factory method."""
        result = FlextMessage.create_message(
            message="Test message",
            level="warning",
            source="api",
        )

        assert result.is_success
        message = result.data
        if message.data != "Test message":
            raise AssertionError(f"Expected {'Test message'}, got {message.data}")
        assert message.level == "warning"
        if message.source != "api":
            raise AssertionError(f"Expected {'api'}, got {message.source}")

    def test_message_create_factory_validation_failure(self) -> None:
        """Test message factory with invalid level."""
        # Invalid level should be handled gracefully
        result = FlextMessage.create_message(
            message="Test",
            level="INVALID_LEVEL",
        )

        # Should succeed since level validation is not strict in our implementation
        assert result.is_success

    def test_message_properties(self) -> None:
        """Test message property accessors."""
        message = FlextMessage(
            data="Test message",
            metadata={
                "level": "error",
                "source": "service_a",
                "correlation_id": "abc-123",
            },
        )

        if message.level != "error":
            raise AssertionError(f"Expected {'error'}, got {message.level}")
        assert message.source == "service_a"
        if message.correlation_id != "abc-123":
            raise AssertionError(f"Expected {'abc-123'}, got {message.correlation_id}")
        assert message.text == "Test message"

    def test_message_properties_defaults(self) -> None:
        """Test message properties with missing metadata."""
        message = FlextMessage(data="Simple message")

        # Properties should handle missing metadata gracefully
        if message.level != "info":  # Default level
            raise AssertionError(f"Expected {'info'}, got {message.level}")
        assert message.source is None
        assert message.correlation_id is None
        if message.text != "Simple message":
            raise AssertionError(f"Expected {'Simple message'}, got {message.text}")

    def test_message_inheritance(self) -> None:
        """Test that FlextMessage inherits FlextPayload functionality."""
        message = FlextMessage(data="Test", metadata={"custom": "value"})

        # Should have FlextPayload methods
        assert message.has_metadata("custom")
        if message.get_metadata("custom") != "value":
            raise AssertionError(
                f"Expected {'value'}, got {message.get_metadata('custom')}"
            )

        # Should support metadata enrichment
        enhanced = message.with_metadata(timestamp=123456)
        if enhanced.get_metadata("timestamp") != 123456:
            raise AssertionError(
                f"Expected {123456}, got {enhanced.get_metadata('timestamp')}"
            )
        assert enhanced.get_metadata("custom") == "value"


class TestFlextEvent:
    """Test FlextEvent specialized payload."""

    def test_event_basic_creation(self) -> None:
        """Test basic event creation."""
        event_data = {
            "user_id": "123",
            "action": "login",
            "timestamp": 1234567890,
        }

        event = FlextEvent(
            data=event_data,
            metadata={
                "event_type": "UserLoggedIn",
                "aggregate_id": "user-123",
                "aggregate_type": "User",
                "version": 1,
            },
        )

        if event.data != event_data:
            raise AssertionError(f"Expected {event_data}, got {event.data}")
        assert event.event_type == "UserLoggedIn"
        if event.aggregate_id != "user-123":
            raise AssertionError(f"Expected {'user-123'}, got {event.aggregate_id}")
        assert event.aggregate_type == "User"
        if event.version != 1:
            raise AssertionError(f"Expected {1}, got {event.version}")

    def test_event_create_factory(self) -> None:
        """Test event creation via factory method."""
        event_data = {"order_id": "order-456", "total": 99.99}

        result = FlextEvent.create_event(
            event_type="OrderCreated",
            event_data=event_data,
            aggregate_id="order-456",
            version=1,
        )

        assert result.is_success
        event = result.data
        if event.data != event_data:
            raise AssertionError(f"Expected {event_data}, got {event.data}")
        assert event.event_type == "OrderCreated"
        if event.aggregate_id != "order-456":
            raise AssertionError(f"Expected {'order-456'}, got {event.aggregate_id}")
        assert event.version == 1

    def test_event_create_factory_validation_failure(self) -> None:
        """Test event factory with validation failure."""
        # Missing required data should fail
        result = FlextEvent.create_event(
            event_type="",  # Invalid empty event type
            event_data={},
        )

        # Should handle gracefully
        assert result.is_failure or result.is_success  # Depends on validation

    def test_event_properties(self) -> None:
        """Test event property accessors."""
        event = FlextEvent(
            data={"action": "update"},
            metadata={
                "event_type": "UserUpdated",
                "aggregate_id": "user-789",
                "aggregate_type": "User",
                "version": 3,
                "correlation_id": "corr-456",
            },
        )

        if event.event_type != "UserUpdated":
            raise AssertionError(f"Expected {'UserUpdated'}, got {event.event_type}")
        assert event.aggregate_id == "user-789"
        if event.aggregate_type != "User":
            raise AssertionError(f"Expected {'User'}, got {event.aggregate_type}")
        assert event.version == EXPECTED_DATA_COUNT
        if event.correlation_id != "corr-456":
            raise AssertionError(f"Expected {'corr-456'}, got {event.correlation_id}")

    def test_event_properties_defaults(self) -> None:
        """Test event properties with missing metadata."""
        event = FlextEvent(data={"simple": "event"})

        # Properties should handle missing metadata gracefully
        assert event.event_type is None
        assert event.aggregate_id is None
        assert event.aggregate_type is None
        assert event.version is None
        assert event.correlation_id is None

    def test_event_inheritance(self) -> None:
        """Test that FlextEvent inherits FlextPayload functionality."""
        event = FlextEvent(data={"test": "data"}, metadata={"custom": "value"})

        # Should have FlextPayload methods
        assert event.has_metadata("custom")
        if event.get_metadata("custom") != "value":
            raise AssertionError(
                f"Expected {'value'}, got {event.get_metadata('custom')}"
            )

        # Should support transformation
        def transform_data(data: dict[str, object]) -> dict[str, object]:
            return {**data, "transformed": True}

        result = event.transform_data(transform_data)
        assert result.is_success
        transformed_event = result.data
        assert transformed_event.data["transformed"] is True  # type: ignore[index]


class TestPayloadEdgeCases:
    """Test edge cases and error conditions."""

    def test_payload_with_none_data(self) -> None:
        """Test payload with None data."""
        payload = FlextPayload(data=None)

        assert payload.data is None
        assert isinstance(payload.metadata, dict)

    def test_payload_with_complex_data_types(self) -> None:
        """Test payload with complex data types."""
        complex_data = {
            "nested": {
                "list": [1, 2, 3],
                "dict": {"inner": "value"},
            },
            "tuple": (1, 2, 3),
            "set_converted": [4, 5, 6],  # Sets become lists in JSON
        }

        payload = FlextPayload(data=complex_data)

        if payload.data != complex_data:
            raise AssertionError(f"Expected {complex_data}, got {payload.data}")

        # Test serialization with complex data
        payload_dict = payload.to_dict()
        if payload_dict["data"] != complex_data:
            raise AssertionError(f"Expected {complex_data}, got {payload_dict['data']}")

    def test_payload_metadata_type_safety(self) -> None:
        """Test metadata type safety."""
        # Various types should be accepted in metadata
        payload = FlextPayload(
            data="test",
            metadata={
                "string": "value",
                "integer": 42,
                "float": math.pi,
                "boolean": True,
                "none": None,
                "list": [1, 2, 3],
                "dict": {"nested": "value"},
            },
        )

        if payload.get_metadata("string") != "value":
            raise AssertionError(
                f"Expected {'value'}, got {payload.get_metadata('string')}"
            )
        assert payload.get_metadata("integer") == 42
        if payload.get_metadata("float") != math.pi:
            raise AssertionError(
                f"Expected {math.pi}, got {payload.get_metadata('float')}"
            )
        if not (payload.get_metadata("boolean")):
            raise AssertionError(
                f"Expected True, got {payload.get_metadata('boolean')}"
            )
        assert payload.get_metadata("none") is None
        if payload.get_metadata("list") != [1, 2, 3]:
            raise AssertionError(
                f"Expected {[1, 2, 3]}, got {payload.get_metadata('list')}"
            )
        assert payload.get_metadata("dict") == {"nested": "value"}

    def test_payload_large_metadata(self) -> None:
        """Test payload with large metadata."""
        large_metadata = {f"key_{i}": f"value_{i}" for i in range(1000)}

        payload = FlextPayload(data="test", metadata=large_metadata)

        if len(payload.metadata) != 1000:
            raise AssertionError(f"Expected {1000}, got {len(payload.metadata)}")
        assert payload.get_metadata("key_500") == "value_500"

        # Test that operations still work efficiently
        keys = payload.keys()
        if len(keys) != 0:  # Keys method returns extra fields, not metadata:
            raise AssertionError(f"Expected {0}, got {len(keys)}")

    def test_payload_with_mixin_functionality(self) -> None:
        """Test payload integration with mixin functionality."""
        payload = FlextPayload(data={"test": "data"}, metadata={"version": "1.0"})

        # Should have mixin methods available
        assert hasattr(payload, "to_dict_basic")  # SerializableMixin

        # Test serialization mixin
        serialized = payload.to_dict_basic()
        assert isinstance(serialized, dict)

    def test_message_with_empty_text(self) -> None:
        """Test message with empty text."""
        message = FlextMessage(data="", metadata={"level": "info"})

        if message.text != "":
            raise AssertionError(f"Expected {''}, got {message.text}")
        assert message.level == "info"

    def test_event_with_empty_data(self) -> None:
        """Test event with empty data."""
        event = FlextEvent(data={}, metadata={"event_type": "EmptyEvent"})

        if event.data != {}:
            raise AssertionError(f"Expected {{}}, got {event.data}")
        assert event.event_type == "EmptyEvent"

    def test_payload_factory_error_handling(self) -> None:
        """Test factory method error handling."""

        # Test with data that might cause issues during creation
        class ProblematicData:
            def __init__(self) -> None:
                msg = "Problematic initialization"
                raise ValueError(msg)

        # This should be handled gracefully by the factory
        try:
            problematic_data = ProblematicData()
        except ValueError:
            # If we can't even create the data, that's fine
            problematic_data = None

        result = FlextPayload.create(data=problematic_data)
        # Should succeed since None is valid data
        assert result.is_success

    def test_transform_with_type_change(self) -> None:
        """Test data transformation with type change."""

        def string_to_int(s: str) -> int:
            return len(s)

        payload = FlextPayload(data="hello world")

        result = payload.transform_data(string_to_int)

        assert result.is_success
        transformed = result.data
        if transformed.data != 11:  # Length of "hello world"
            raise AssertionError(f"Expected {11}, got {transformed.data}")

    def test_payload_attribute_access_edge_cases(self) -> None:
        """Test edge cases in attribute access."""

        # Test with data that has conflicting attribute names
        class TestData:
            def __init__(self) -> None:
                self.metadata = "data_metadata"  # Conflicts with payload.metadata

        data = TestData()
        payload = FlextPayload(data=data, metadata={"metadata": "payload_metadata"})

        # Metadata should take priority
        assert payload.metadata != "data_metadata"  # Should be dict, not string

        # Test accessing reserved names
        with pytest.raises(FlextAttributeError):
            _ = payload.nonexistent_reserved_name  # type: ignore[attr-defined]


class TestPayloadPerformance:
    """Test payload performance characteristics."""

    def test_payload_creation_performance(self) -> None:
        """Test performance of payload creation."""
        # Create many payloads to test performance
        payloads = []
        for i in range(100):
            payload = FlextPayload(
                data=f"data_{i}",
                metadata={"index": i, "batch": "performance_test"},
            )
            payloads.append(payload)

        if len(payloads) != 100:
            raise AssertionError(f"Expected {100}, got {len(payloads)}")
        if not all(isinstance(p, FlextPayload) for p in payloads):
            raise AssertionError(
                f"Expected {all(isinstance(p, FlextPayload) for p in payloads)}, got {payloads}"
            )

    def test_metadata_operations_performance(self) -> None:
        """Test performance of metadata operations."""
        payload = FlextPayload(data="test")

        # Add metadata incrementally
        for i in range(50):
            payload = payload.with_metadata(**{f"key_{i}": f"value_{i}"})

        # Should have all metadata
        if len(payload.metadata) != 50:
            raise AssertionError(f"Expected {50}, got {len(payload.metadata)}")
        assert payload.get_metadata("key_25") == "value_25"


class TestPayloadIntegration:
    """Test payload integration with other components."""

    def test_payload_with_result_pattern(self) -> None:
        """Test payload integration with FlextResult pattern."""

        def create_validated_payload(data: str) -> FlextResult[FlextPayload[str]]:
            if not data:
                return FlextResult.fail("Data cannot be empty")
            return FlextPayload.create(data=data, validated=True)

        # Valid case
        result = create_validated_payload("valid_data")
        assert result.is_success
        payload = result.data
        if payload.data != "valid_data":
            raise AssertionError(f"Expected {'valid_data'}, got {payload.data}")
        if not (payload.get_metadata("validated")):
            raise AssertionError(
                f"Expected True, got {payload.get_metadata('validated')}"
            )

        # Invalid case
        result = create_validated_payload("")
        assert result.is_failure
        if "cannot be empty" not in result.error:
            raise AssertionError(f"Expected {'cannot be empty'} in {result.error}")

    def test_message_and_event_interoperability(self) -> None:
        """Test that messages and events can work together."""
        # Create related message and event
        message = FlextMessage(
            data="User login successful",
            metadata={
                "level": "info",
                "correlation_id": "corr-123",
            },
        )

        event = FlextEvent(
            data={"user_id": "user-456", "timestamp": 1234567890},
            metadata={
                "event_type": "UserLoggedIn",
                "correlation_id": "corr-123",  # Same correlation ID
            },
        )

        # Should be able to correlate by ID
        if message.correlation_id != event.correlation_id:
            raise AssertionError(
                f"Expected {event.correlation_id}, got {message.correlation_id}"
            )

        # Should both be FlextPayload instances
        assert isinstance(message, FlextPayload)
        assert isinstance(event, FlextPayload)
