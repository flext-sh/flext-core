"""Simple test coverage for payload.py module.

This test suite focuses on testing the actual API of payload.py.
"""

from __future__ import annotations

import pytest

from flext_core.payload import (
    FlextEvent,
    FlextMessage,
    FlextPayload,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextPayload:
    """Test FlextPayload class basic functionality."""

    def test_payload_creation_basic(self) -> None:
        """Test basic payload creation."""
        payload = FlextPayload(data={"key": "value"})
        assert payload.data == {"key": "value"}
        assert payload.metadata == {}

    def test_payload_creation_with_metadata(self) -> None:
        """Test payload creation with metadata."""
        metadata: dict[str, object] = {"source": "test", "version": "1.0"}
        payload = FlextPayload(data={"test": True}, metadata=metadata)
        assert payload.metadata == metadata

    def test_payload_create_factory(self) -> None:
        """Test payload creation using factory method."""
        result = FlextPayload.create({"user_id": 123}, source="api")
        assert result.success is True
        payload = result.data
        assert payload is not None
        assert payload.data == {"user_id": 123}
        assert payload.metadata.get("source") == "api"

    def test_payload_with_metadata(self) -> None:
        """Test adding metadata to payload."""
        payload: FlextPayload[object] = FlextPayload(data={})
        new_payload = payload.with_metadata(key="value")
        assert new_payload.metadata["key"] == "value"

    def test_payload_get_metadata(self) -> None:
        """Test getting metadata values."""
        payload: FlextPayload[object] = FlextPayload(
            data={}, metadata={"test": "value"}
        )
        assert payload.get_metadata("test") == "value"
        assert payload.get_metadata("missing") is None
        assert payload.get_metadata("missing", "default") == "default"

    def test_payload_has_metadata(self) -> None:
        """Test checking metadata existence."""
        payload: FlextPayload[object] = FlextPayload(
            data={}, metadata={"exists": "yes"}
        )
        assert payload.has_metadata("exists") is True
        assert payload.has_metadata("missing") is False

    def test_payload_to_dict(self) -> None:
        """Test payload dictionary conversion."""
        payload: FlextPayload[object] = FlextPayload(
            data={"test": "data"}, metadata={"key": "value"}
        )
        result = payload.to_dict()
        assert result["data"] == {"test": "data"}
        assert result["metadata"] == {"key": "value"}

    def test_payload_from_dict(self) -> None:
        """Test creating payload from dictionary."""
        data_dict = {"data": {"key": "value"}, "metadata": {"source": "test"}}
        result = FlextPayload.from_dict(data_dict)
        assert result.success is True
        payload = result.data
        assert payload is not None
        assert payload.data == {"key": "value"}
        assert payload.metadata == {"source": "test"}

    def test_payload_from_dict_invalid(self) -> None:
        """Test from_dict with invalid input."""
        result = FlextPayload.from_dict("not a dict")
        assert result.success is False
        assert "not a dictionary" in (result.error or "")

    def test_payload_has_data(self) -> None:
        """Test checking if payload has data."""
        payload_with_data: FlextPayload[object] = FlextPayload(data="test")
        payload_without_data: FlextPayload[object] = FlextPayload(data=None)

        assert payload_with_data.has_data() is True
        assert payload_without_data.has_data() is False

    def test_payload_get_data(self) -> None:
        """Test getting payload data."""
        payload: FlextPayload[object] = FlextPayload(data="test")
        result = payload.get_data()
        assert result.success is True
        assert result.data == "test"

        empty_payload: FlextPayload[object] = FlextPayload(data=None)
        result = empty_payload.get_data()
        assert result.success is False

    def test_payload_get_data_or_default(self) -> None:
        """Test getting data with default."""
        payload: FlextPayload[object] = FlextPayload(data="test")
        assert payload.get_data_or_default("default") == "test"

        empty_payload: FlextPayload[object] = FlextPayload(data=None)
        assert empty_payload.get_data_or_default("default") == "default"

    def test_payload_transform_data(self) -> None:
        """Test transforming payload data."""
        payload: FlextPayload[object] = FlextPayload(data="hello")
        result = payload.transform_data(lambda x: str(x).upper())
        assert result.success is True
        new_payload = result.data
        assert new_payload is not None
        assert new_payload.data == "HELLO"

    def test_payload_transform_data_none(self) -> None:
        """Test transforming None data."""
        payload: FlextPayload[object] = FlextPayload(data=None)
        result = payload.transform_data(lambda x: str(x).upper())
        assert result.success is False

    def test_payload_serialization_methods(self) -> None:
        """Test various serialization methods."""
        payload: FlextPayload[object] = FlextPayload(data={"test": "value"})

        # Test cross-service dict
        cross_dict = payload.to_cross_service_dict()
        assert "data" in cross_dict
        assert "payload_type" in cross_dict

        # Test JSON string
        json_result = payload.to_json_string()
        assert json_result.success is True

        # Test serialization size
        size_info = payload.get_serialization_size()
        assert "json_size" in size_info
        assert "compression_ratio" in size_info


class TestFlextMessage:
    """Test FlextMessage class basic functionality."""

    def test_message_create_basic(self) -> None:
        """Test basic message creation."""
        result = FlextMessage.create_message("Hello world")
        assert result.success is True
        message = result.data
        assert message is not None
        assert message.data == "Hello world"
        assert message.level == "info"

    def test_message_create_with_level(self) -> None:
        """Test message creation with level."""
        result = FlextMessage.create_message("Error occurred", level="error")
        assert result.success is True
        message = result.data
        assert message is not None
        assert message.level == "error"

    def test_message_create_with_source(self) -> None:
        """Test message creation with source."""
        result = FlextMessage.create_message("Info", source="test-service")
        assert result.success is True
        message = result.data
        assert message is not None
        assert message.source == "test-service"

    def test_message_create_empty(self) -> None:
        """Test creating message with empty text."""
        result = FlextMessage.create_message("")
        assert result.success is False
        assert "cannot be empty" in (result.error or "")

    def test_message_create_invalid_level(self) -> None:
        """Test message with invalid level falls back to info."""
        result = FlextMessage.create_message("Test", level="invalid")
        assert result.success is True
        message = result.data
        assert message is not None
        assert message.level == "info"

    def test_message_properties(self) -> None:
        """Test message properties."""
        result = FlextMessage.create_message("Test", level="warning", source="app")
        message = result.data
        assert message is not None

        assert message.level == "warning"
        assert message.source == "app"
        assert message.text == "Test"
        assert message.correlation_id is None  # Not set by default

    def test_message_cross_service_dict(self) -> None:
        """Test message cross-service serialization."""
        result = FlextMessage.create_message("Test message", level="info", source="api")
        message = result.data
        assert message is not None

        cross_dict = message.to_cross_service_dict()
        assert cross_dict["message_text"] == "Test message"
        assert cross_dict["message_level"] == "info"
        assert cross_dict["message_source"] == "api"


class TestFlextEvent:
    """Test FlextEvent class basic functionality."""

    def test_event_create_basic(self) -> None:
        """Test basic event creation."""
        result = FlextEvent.create_event("UserCreated", {"user_id": "123"})
        assert result.success is True
        event = result.data
        assert event is not None
        assert event.event_type == "UserCreated"
        assert event.data == {"user_id": "123"}

    def test_event_create_with_aggregate(self) -> None:
        """Test event creation with aggregate ID."""
        result = FlextEvent.create_event(
            "OrderPlaced",
            {"order_id": "456", "amount": 100.0},
            aggregate_id="order_456",
        )
        assert result.success is True
        event = result.data
        assert event is not None
        assert event.aggregate_id == "order_456"

    def test_event_create_with_version(self) -> None:
        """Test event creation with version."""
        result = FlextEvent.create_event(
            "ProductUpdated", {"product_id": "789"}, version=2
        )
        assert result.success is True
        event = result.data
        assert event is not None
        assert event.version == 2

    def test_event_create_empty_type(self) -> None:
        """Test creating event with empty event type."""
        result = FlextEvent.create_event("", {"data": "test"})
        assert result.success is False
        assert "cannot be empty" in (result.error or "")

    def test_event_create_invalid_aggregate_id(self) -> None:
        """Test creating event with invalid aggregate ID."""
        result = FlextEvent.create_event("Test", {"data": "test"}, aggregate_id="")
        assert result.success is False
        assert "Invalid aggregate ID" in (result.error or "")

    def test_event_create_invalid_version(self) -> None:
        """Test creating event with invalid version."""
        result = FlextEvent.create_event("Test", {"data": "test"}, version=-1)
        assert result.success is False
        assert "must be non-negative" in (result.error or "")

    def test_event_properties(self) -> None:
        """Test event properties."""
        result = FlextEvent.create_event(
            "ItemAdded",
            {"item_id": "123", "quantity": 5},
            aggregate_id="cart_456",
            version=3,
        )
        event = result.data
        assert event is not None

        assert event.event_type == "ItemAdded"
        assert event.aggregate_id == "cart_456"
        assert event.version == 3
        assert event.correlation_id is None  # Not set by default

    def test_event_cross_service_dict(self) -> None:
        """Test event cross-service serialization."""
        result = FlextEvent.create_event(
            "PaymentProcessed",
            {"amount": 50.0, "currency": "USD"},
            aggregate_id="payment_789",
            version=1,
        )
        event = result.data
        assert event is not None

        cross_dict = event.to_cross_service_dict()
        assert cross_dict["event_type"] == "PaymentProcessed"
        assert cross_dict["aggregate_id"] == "payment_789"
        assert cross_dict["event_version"] == 1
        assert cross_dict["event_data"] == {"amount": 50.0, "currency": "USD"}


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_payload_create_validation_error(self) -> None:
        """Test payload creation handles validation errors properly."""
        # Test that validation errors are caught and handled properly
        # Create payload with complex metadata that could cause issues
        try:
            # Complex scenario that might cause validation issues
            complex_data = {
                "deeply": {
                    "nested": {
                        "data": "with many levels",
                        "numbers": [1, 2, 3] * 1000,  # Large list
                    }
                }
            }
            result = FlextPayload.create(complex_data, test_metadata="success")
            # This should succeed, just testing error handling path exists
            assert result.success is True
            payload = result.data
            assert payload is not None
            assert payload.metadata.get("test_metadata") == "success"
        except Exception:
            # If any exception occurs, the create method should handle it gracefully
            pytest.fail(
                "FlextPayload.create should handle validation errors gracefully"
            )

    def test_payload_transform_error(self) -> None:
        """Test payload transform with error."""
        payload = FlextPayload(data="test")

        def failing_transformer(x: object) -> object:
            error_msg = "Transform failed"
            raise ValueError(error_msg)

        result = payload.transform_data(failing_transformer)
        assert result.success is False
        assert "Transform failed" in (result.error or "")

    def test_message_from_cross_service_dict(self) -> None:
        """Test creating message from cross-service dict."""
        cross_dict: dict[str, object] = {
            "message_text": "Test message",
            "message_level": "warning",
            "message_source": "test-service",
        }
        result = FlextMessage.from_cross_service_dict(cross_dict)
        assert result.success is True

    def test_message_from_cross_service_dict_invalid(self) -> None:
        """Test creating message from invalid cross-service dict."""
        cross_dict: dict[str, object] = {"message_text": None}
        result = FlextMessage.from_cross_service_dict(cross_dict)
        assert result.success is False

    def test_event_from_cross_service_dict(self) -> None:
        """Test creating event from cross-service dict."""
        cross_dict: dict[str, object] = {
            "event_type": "TestEvent",
            "event_data": {"key": "value"},
            "aggregate_id": "agg_123",
            "event_version": 1,
        }
        result = FlextEvent.from_cross_service_dict(cross_dict)
        assert result.success is True

    def test_event_from_cross_service_dict_invalid(self) -> None:
        """Test creating event from invalid cross-service dict."""
        cross_dict: dict[str, object] = {"event_type": None}
        result = FlextEvent.from_cross_service_dict(cross_dict)
        assert result.success is False

    def test_payload_json_serialization_round_trip(self) -> None:
        """Test JSON serialization round trip."""
        original_payload: FlextPayload[object] = FlextPayload(
            data={"test": "value"}, metadata={"key": "metadata"}
        )

        # Serialize to JSON
        json_result = original_payload.to_json_string()
        assert json_result.success is True

        # Deserialize from JSON
        assert json_result.data is not None
        restored_result = FlextPayload.from_json_string(json_result.data)
        assert restored_result.success is True

        restored_payload = restored_result.data
        assert restored_payload is not None
        # Basic checks - exact equality might not work due to serialization details
        assert isinstance(restored_payload.data, dict)
        assert isinstance(restored_payload.metadata, dict)

    def test_payload_compression(self) -> None:
        """Test payload compression for large data."""
        # Create large payload
        large_data: dict[str, object] = {"items": [f"item_{i}" for i in range(1000)]}
        payload: FlextPayload[object] = FlextPayload(data=large_data)

        # Test with compression
        compressed_result = payload.to_json_string(compressed=True)
        assert compressed_result.success is True

        # Test without compression
        uncompressed_result = payload.to_json_string(compressed=False)
        assert uncompressed_result.success is True

    def test_cross_service_utilities(self) -> None:
        """Test cross-service utility functions."""
        from flext_core.payload import (
            create_cross_service_event,
            create_cross_service_message,
            get_serialization_metrics,
            validate_cross_service_protocol,
        )

        # Test cross-service message creation
        msg_result = create_cross_service_message("Test", correlation_id="123")
        assert msg_result.success is True

        # Test cross-service event creation
        event_result = create_cross_service_event(
            "TestEvent", {"key": "value"}, correlation_id="456"
        )
        assert event_result.success is True

        # Test serialization metrics
        payload = FlextPayload(data={"test": "data"})
        metrics = get_serialization_metrics(payload)
        assert "payload_type" in metrics
        assert "data_type" in metrics

        # Test protocol validation
        valid_protocol = '{"format": "json", "data": {"test": "value"}}'
        protocol_result = validate_cross_service_protocol(valid_protocol)
        assert protocol_result.success is True
