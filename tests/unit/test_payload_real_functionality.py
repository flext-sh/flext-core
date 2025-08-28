"""Comprehensive real functional tests for flext_core.payload module.

Tests the actual business functionality of all payload classes without mocks,
focusing on real patterns, validation, serialization, and cross-service messaging.

Classes Tested:
- FlextPayload: Generic type-safe payload container
- FlextMessage: Specialized string message payload
- FlextEvent: Domain event payload with aggregate tracking
"""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime

import pytest
from pydantic_core import ValidationError

from flext_core.constants import FlextConstants
from flext_core.payload import (
    FlextPayload,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


# =============================================================================
# FLEXT PAYLOAD TESTS - Generic Type-Safe Container
# =============================================================================


class TestFlextPayloadRealFunctionality:
    """Test FlextPayload real functionality with type-safe containers."""

    def test_payload_creation_with_primitive_data(self) -> None:
        """Test FlextPayload creation with primitive data types."""
        # String payload
        result = FlextPayload.create("test string", source="unit_test")
        assert result.is_success
        payload = result.value
        assert payload.data == "test string"
        assert payload.value == "test string"  # Modern .value access
        assert payload.metadata["source"] == "unit_test"

        # Integer payload
        result = FlextPayload.create(42, version=1)
        assert result.is_success
        payload = result.value
        assert payload.data == 42
        assert payload.value == 42
        assert payload.metadata["version"] == 1

        # Boolean payload
        result = FlextPayload.create(True, flag_type="boolean")
        assert result.is_success
        payload = result.value
        assert payload.data is True
        assert payload.value is True
        assert payload.metadata["flag_type"] == "boolean"

    def test_payload_creation_with_complex_data(self) -> None:
        """Test FlextPayload with complex nested data structures."""
        complex_data = {
            "user_id": "user_123",
            "preferences": {
                "theme": "dark",
                "notifications": ["email", "push"],
                "settings": {"language": "en", "timezone": "UTC"},
            },
            "metadata": {
                "created_at": "2024-01-01T00:00:00Z",
                "tags": ["premium", "beta"],
            },
        }

        result = FlextPayload.create(complex_data, operation="user_preferences_update")
        assert result.is_success

        payload = result.value
        assert payload.value == complex_data
        assert payload.value["user_id"] == "user_123"
        assert payload.value["preferences"]["theme"] == "dark"
        assert payload.value["preferences"]["settings"]["language"] == "en"
        assert payload.metadata["operation"] == "user_preferences_update"

    def test_payload_metadata_enrichment(self) -> None:
        """Test FlextPayload metadata enrichment functionality."""
        # Create initial payload
        result = FlextPayload.create({"initial": "data"}, version=1, source="test")
        assert result.is_success

        original = result.value
        assert len(original.metadata) == 2
        assert original.metadata["version"] == 1
        assert original.metadata["source"] == "test"

        # Enrich with additional metadata using with_metadata
        enriched = original.with_metadata(
            timestamp=datetime.now(UTC).isoformat(),
            user_id="user_123",
            operation="data_processing",
        )

        assert enriched.value == original.value  # Same data
        assert len(enriched.metadata) == 5  # Original + 3 new
        assert enriched.metadata["version"] == 1  # Preserved
        assert enriched.metadata["source"] == "test"  # Preserved
        assert "timestamp" in enriched.metadata
        assert enriched.metadata["user_id"] == "user_123"
        assert enriched.metadata["operation"] == "data_processing"

    def test_payload_metadata_enrichment_from_dict(self) -> None:
        """Test FlextPayload metadata enrichment from dictionary."""
        result = FlextPayload.create({"test": "data"})
        assert result.is_success

        original = result.value

        additional_metadata = {
            "request_id": "req_123",
            "correlation_id": "corr_456",
            "trace_id": "trace_789",
        }

        enriched = original.enrich_metadata(additional_metadata)

        assert enriched.value == original.value
        assert enriched.metadata["request_id"] == "req_123"
        assert enriched.metadata["correlation_id"] == "corr_456"
        assert enriched.metadata["trace_id"] == "trace_789"

    def test_payload_create_from_dict(self) -> None:
        """Test FlextPayload creation from dictionary."""
        data_dict = {
            "data": {"user": "john", "action": "login"},
            "metadata": {"timestamp": "2024-01-01T00:00:00Z"},
        }

        result = FlextPayload.create_from_dict(data_dict)
        assert result.is_success

        payload = result.value
        assert payload.value == {"user": "john", "action": "login"}
        assert payload.metadata["timestamp"] == "2024-01-01T00:00:00Z"

    def test_payload_immutability(self) -> None:
        """Test FlextPayload immutability (frozen=True)."""
        result = FlextPayload.create({"test": "data"}, version=1)
        assert result.is_success

        payload = result.value

        # Should not be able to modify data or metadata directly due to frozen=True
        with pytest.raises((AttributeError, TypeError, ValidationError)):
            payload.data = {"modified": "data"}

        with pytest.raises((AttributeError, TypeError, ValidationError)):
            payload.metadata = {"modified": "metadata"}

    def test_payload_serialization_patterns(self) -> None:
        """Test FlextPayload serialization patterns."""
        test_data = {
            "id": "test_123",
            "name": "Test Object",
            "attributes": ["attr1", "attr2"],
            "config": {"enabled": True, "timeout": 30},
        }

        result = FlextPayload.create(test_data, operation="test", timestamp=time.time())
        assert result.is_success

        payload = result.value

        # Test model_dump serialization
        serialized = payload.model_dump(by_alias=True)
        assert "payloadData" in serialized  # Uses serialization_alias
        assert "payloadMetadata" in serialized  # Uses serialization_alias
        assert serialized["payloadData"] == test_data
        assert serialized["payloadMetadata"]["operation"] == "test"


# =============================================================================
# FLEXT MESSAGE TESTS - Specialized String Message Payload
# =============================================================================


class TestFlextMessageRealFunctionality:
    """Test FlextMessage real functionality with message handling."""

    def test_message_creation_with_levels(self) -> None:
        """Test FlextMessage creation with different message levels."""
        # Info level (default)
        result = FlextPayload.create_message("System initialized successfully")
        assert result.is_success
        message = result.value
        assert message.value == "System initialized successfully"
        assert message.metadata["level"] == "info"

        # Warning level
        result = FlextPayload.create_message(
            "Database connection slow", level="warning", source="database_monitor"
        )
        assert result.is_success
        message = result.value
        assert message.value == "Database connection slow"
        assert message.metadata["level"] == "warning"
        assert message.metadata["source"] == "database_monitor"

        # Error level
        result = FlextPayload.create_message(
            "Failed to authenticate user", level="error", source="auth_service"
        )
        assert result.is_success
        message = result.value
        assert message.value == "Failed to authenticate user"
        assert message.metadata["level"] == "error"
        assert message.metadata["source"] == "auth_service"

        # Critical level
        result = FlextPayload.create_message(
            "System shutdown initiated", level="critical", source="system_manager"
        )
        assert result.is_success
        message = result.value
        assert message.metadata["level"] == "critical"

    def test_message_level_validation_and_fallback(self) -> None:
        """Test FlextMessage level validation with fallback to 'info'."""
        # Invalid level should fallback to 'info'
        result = FlextPayload.create_message(
            "Test message with invalid level", level="invalid_level"
        )
        assert result.is_success
        message = result.value
        assert message.metadata["level"] == "info"  # Should fallback

        # Test all valid levels
        valid_levels = ["info", "warning", "error", "debug", "critical"]
        for level in valid_levels:
            result = FlextPayload.create_message(f"Test {level} message", level=level)
            assert result.is_success
            message = result.value
            assert message.metadata["level"] == level

    def test_message_validation_failures(self) -> None:
        """Test FlextMessage validation failures."""
        # Empty message should fail
        result = FlextPayload.create_message("")
        assert result.is_failure
        assert "cannot be empty" in result.error

        # None message should fail
        result = FlextPayload.create_message(None)
        assert result.is_failure

        # Whitespace-only message should fail
        result = FlextPayload.create_message("   ")
        assert result.is_failure

    def test_message_source_tracking(self) -> None:
        """Test FlextMessage source tracking functionality."""
        # Without source
        result = FlextPayload.create_message("Test message without source")
        assert result.is_success
        message = result.value
        assert "source" not in message.metadata

        # With source
        result = FlextPayload.create_message(
            "Test message with source", source="unit_test_module"
        )
        assert result.is_success
        message = result.value
        assert message.metadata["source"] == "unit_test_module"

    def test_message_inheritance_from_payload(self) -> None:
        """Test FlextMessage inherits FlextPayload functionality."""
        result = FlextPayload.create_message(
            "Test message", level="info", source="test"
        )
        assert result.is_success

        message = result.value

        # Should inherit metadata enrichment
        enriched = message.with_metadata(
            correlation_id="corr_123", timestamp=time.time()
        )

        assert enriched.value == "Test message"
        assert enriched.metadata["level"] == "info"
        assert enriched.metadata["source"] == "test"
        assert "correlation_id" in enriched.metadata
        assert "timestamp" in enriched.metadata

    def test_message_complex_workflow(self) -> None:
        """Test FlextMessage in complex logging workflow."""
        # Create system startup messages
        messages = []

        # Startup sequence
        result = FlextPayload.create_message(
            "Starting application initialization", level="info", source="app_manager"
        )
        assert result.is_success
        messages.append(result.value)

        # Configuration loading
        result = FlextPayload.create_message(
            "Loading configuration from config.yaml",
            level="debug",
            source="config_loader",
        )
        assert result.is_success
        messages.append(result.value)

        # Warning during startup
        result = FlextPayload.create_message(
            "Configuration file missing optional section 'advanced'",
            level="warning",
            source="config_loader",
        )
        assert result.is_success
        messages.append(result.value)

        # Successful startup
        result = FlextPayload.create_message(
            "Application started successfully on port 8080",
            level="info",
            source="web_server",
        )
        assert result.is_success
        messages.append(result.value)

        # Verify the message sequence
        assert len(messages) == 4
        assert all(isinstance(msg, FlextPayload) for msg in messages)
        assert messages[0].metadata["level"] == "info"
        assert messages[1].metadata["level"] == "debug"
        assert messages[2].metadata["level"] == "warning"
        assert messages[3].metadata["level"] == "info"

        # All should be different sources
        sources = {msg.metadata.get("source") for msg in messages}
        assert len(sources) == 3  # app_manager, config_loader, web_server


# =============================================================================
# FLEXT EVENT TESTS - Domain Event Payload with Aggregate Tracking
# =============================================================================


class TestFlextEventRealFunctionality:
    """Test FlextEvent real functionality with domain events."""

    def test_event_creation_basic(self) -> None:
        """Test FlextEvent creation with basic event data."""
        event_data = {
            "user_id": "user_123",
            "email": "user@example.com",
            "timestamp": datetime.now(UTC).isoformat(),
        }

        result = FlextPayload.create_event(
            "UserRegistered", event_data, aggregate_id="user_123", version=1
        )

        assert result.is_success
        event = result.value
        assert event.value == event_data
        assert event.metadata["event_type"] == "UserRegistered"
        assert event.metadata["aggregate_id"] == "user_123"
        assert event.metadata["version"] == 1

    def test_event_creation_without_optional_fields(self) -> None:
        """Test FlextEvent creation without optional aggregate_id and version."""
        event_data = {"action": "system_restart", "reason": "maintenance"}

        result = FlextPayload.create_event("SystemRestarted", event_data)
        assert result.is_success

        event = result.value
        assert event.value == event_data
        assert event.metadata["event_type"] == "SystemRestarted"
        assert "aggregate_id" not in event.metadata
        assert "version" not in event.metadata

    def test_event_validation_failures(self) -> None:
        """Test FlextEvent validation failures."""
        event_data = {"test": "data"}

        # Empty event_type should fail
        result = FlextPayload.create_event("", event_data)
        assert result.is_failure
        assert "cannot be empty" in result.error

        # None event_type should fail
        result = FlextPayload.create_event(None, event_data)
        assert result.is_failure

        # Empty aggregate_id should fail
        result = FlextPayload.create_event("TestEvent", event_data, aggregate_id="")
        assert result.is_failure
        assert "Invalid aggregate ID" in result.error

        # Negative version should fail
        result = FlextPayload.create_event("TestEvent", event_data, version=-1)
        assert result.is_failure
        assert "must be non-negative" in result.error

    def test_event_domain_driven_design_patterns(self) -> None:
        """Test FlextEvent with real DDD event sourcing patterns."""
        # User aggregate events sequence
        events = []

        # 1. User Registration
        registration_data = {
            "user_id": "user_456",
            "email": "john.doe@example.com",
            "name": "John Doe",
            "registration_source": "web_app",
        }

        result = FlextPayload.create_event(
            "UserRegistered", registration_data, aggregate_id="user_456", version=1
        )
        assert result.is_success
        events.append(result.value)

        # 2. Email Verification
        verification_data = {
            "user_id": "user_456",
            "verification_token": "token_abc123",
            "verified_at": datetime.now(UTC).isoformat(),
        }

        result = FlextPayload.create_event(
            "EmailVerified", verification_data, aggregate_id="user_456", version=2
        )
        assert result.is_success
        events.append(result.value)

        # 3. Profile Update
        profile_data = {
            "user_id": "user_456",
            "updates": {
                "first_name": "John",
                "last_name": "Doe",
                "bio": "Software developer",
            },
            "updated_by": "user_456",
        }

        result = FlextPayload.create_event(
            "ProfileUpdated", profile_data, aggregate_id="user_456", version=3
        )
        assert result.is_success
        events.append(result.value)

        # 4. Account Deactivation
        deactivation_data = {
            "user_id": "user_456",
            "reason": "user_request",
            "deactivated_at": datetime.now(UTC).isoformat(),
        }

        result = FlextPayload.create_event(
            "AccountDeactivated", deactivation_data, aggregate_id="user_456", version=4
        )
        assert result.is_success
        events.append(result.value)

        # Verify event sequence
        assert len(events) == 4

        event_types = [event.metadata["event_type"] for event in events]
        expected_types = [
            "UserRegistered",
            "EmailVerified",
            "ProfileUpdated",
            "AccountDeactivated",
        ]
        assert event_types == expected_types

        # All should have same aggregate_id
        aggregate_ids = {event.metadata["aggregate_id"] for event in events}
        assert len(aggregate_ids) == 1
        assert next(iter(aggregate_ids)) == "user_456"

        # Versions should be sequential
        versions = [event.metadata["version"] for event in events]
        assert versions == [1, 2, 3, 4]

    def test_event_complex_event_data(self) -> None:
        """Test FlextEvent with complex nested event data."""
        complex_event_data = {
            "order_id": "order_789",
            "customer": {
                "customer_id": "customer_123",
                "name": "Alice Smith",
                "email": "alice@example.com",
            },
            "items": [
                {
                    "product_id": "prod_001",
                    "name": "Laptop",
                    "quantity": 1,
                    "unit_price": 999.99,
                    "total_price": 999.99,
                },
                {
                    "product_id": "prod_002",
                    "name": "Mouse",
                    "quantity": 2,
                    "unit_price": 29.99,
                    "total_price": 59.98,
                },
            ],
            "totals": {
                "subtotal": 1059.97,
                "tax": 84.80,
                "shipping": 15.00,
                "total": 1159.77,
            },
            "payment": {
                "method": "credit_card",
                "status": "authorized",
                "transaction_id": "txn_abc123",
            },
            "fulfillment": {
                "warehouse_id": "warehouse_west",
                "estimated_shipping": "2024-01-15",
                "tracking_enabled": True,
            },
        }

        result = FlextPayload.create_event(
            "OrderPlaced", complex_event_data, aggregate_id="order_789", version=1
        )

        assert result.is_success
        event = result.value
        assert event.value == complex_event_data
        assert event.value["order_id"] == "order_789"
        assert event.value["customer"]["customer_id"] == "customer_123"
        assert len(event.value["items"]) == 2
        assert event.value["totals"]["total"] == 1159.77
        assert event.metadata["event_type"] == "OrderPlaced"

    def test_event_inheritance_from_payload(self) -> None:
        """Test FlextEvent inherits FlextPayload functionality."""
        event_data = {"test": "event"}

        result = FlextPayload.create_event(
            "TestEvent", event_data, aggregate_id="test_123", version=1
        )
        assert result.is_success

        event = result.value

        # Should inherit metadata enrichment
        enriched = event.with_metadata(
            correlation_id="corr_789", causation_id="cause_456"
        )

        assert enriched.value == event_data
        assert enriched.metadata["event_type"] == "TestEvent"
        assert enriched.metadata["aggregate_id"] == "test_123"
        assert enriched.metadata["version"] == 1
        assert enriched.metadata["correlation_id"] == "corr_789"
        assert enriched.metadata["causation_id"] == "cause_456"


# =============================================================================
# SERIALIZATION AND CROSS-SERVICE TESTS - Constants and Utilities
# =============================================================================


class TestPayloadSerializationRealFunctionality:
    """Test payload serialization constants and cross-service functionality."""

    def test_serialization_constants_exist(self) -> None:
        """Test that all serialization constants are properly defined."""
        # Version constant should be defined
        assert FlextConstants.Observability.FLEXT_SERIALIZATION_VERSION is not None

        # Format constants should be defined
        assert FlextConstants.Observability.SERIALIZATION_FORMAT_JSON is not None
        assert FlextConstants.Observability.SERIALIZATION_FORMAT_JSON_COMPRESSED is not None

        # Size and compression constants
        assert FlextConstants.Performance.PAYLOAD_COMPRESSION_LEVEL == 6

    @pytest.mark.skip(reason="Go type mappings not yet implemented")
    def test_go_type_mappings_comprehensive(self) -> None:
        """Test Go type mappings for cross-service communication."""
        # TODO: Implement Go type mappings when bridge functionality is added

    @pytest.mark.skip(reason="Python to Go type mappings not yet implemented")
    def test_python_to_go_type_mappings(self) -> None:
        """Test Python to Go type mappings for serialization."""
        # TODO: Implement Python to Go type mappings when bridge functionality is added

    def test_payload_serialization_with_go_bridge_types(self) -> None:
        """Test payload serialization considering Go bridge type compatibility."""
        # Create payload with types that need Go bridge conversion
        go_compatible_data = {
            "string_field": "test string",  # string
            "integer_field": 42,  # int64
            "float_field": math.pi,  # float64
            "boolean_field": True,  # bool
            "map_field": {"nested": "object"},  # map[string]interface{}
            "array_field": [1, 2, 3, "mixed"],  # []interface{}
        }

        result = FlextPayload.create(go_compatible_data, target_service="go_service")
        assert result.is_success

        payload = result.value
        serialized = payload.model_dump()

        # Verify all types can be serialized
        assert isinstance(serialized["data"]["string_field"], str)
        assert isinstance(serialized["data"]["integer_field"], int)
        assert isinstance(serialized["data"]["float_field"], float)
        assert isinstance(serialized["data"]["boolean_field"], bool)
        assert isinstance(serialized["data"]["map_field"], dict)
        assert isinstance(serialized["data"]["array_field"], list)

    def test_large_payload_size_handling(self) -> None:
        """Test payload handling for large data approaching compression threshold."""
        # Create data approaching MAX_UNCOMPRESSED_SIZE
        large_data = {
            "bulk_data": "x" * 60000,  # 60KB of data
            "metadata": {
                "size_info": "approaching_compression_threshold",
                "test_purpose": "large_payload_handling",
            },
        }

        result = FlextPayload.create(large_data, size_test=True)
        assert result.is_success

        payload = result.value

        # Should handle large payloads without issues
        assert len(payload.value["bulk_data"]) == 60000
        assert payload.metadata["size_test"] is True

        # Should be able to serialize large payloads
        serialized = payload.model_dump()
        assert len(serialized["data"]["bulk_data"]) == 60000


# =============================================================================
# INTEGRATION TESTS - Multiple Components Working Together
# =============================================================================


class TestPayloadIntegrationRealFunctionality:
    """Test integration scenarios with multiple payload components."""

    def test_complete_message_event_workflow(self) -> None:
        """Test complete workflow using messages and events together."""
        # 1. Create system startup message
        startup_result = FlextPayload.create_message(
            "System startup initiated", level="info", source="system_manager"
        )
        assert startup_result.is_success
        startup_message = startup_result.value

        # 2. Create system started event
        startup_event_data = {
            "system_id": "system_001",
            "startup_time": datetime.now(UTC).isoformat(),
            "configuration": {"environment": "production", "version": "1.0.0"},
        }

        event_result = FlextPayload.create_event(
            "SystemStarted", startup_event_data, aggregate_id="system_001", version=1
        )
        assert event_result.is_success
        startup_event = event_result.value

        # 3. Create completion message
        completion_result = FlextPayload.create_message(
            "System startup completed successfully",
            level="info",
            source="system_manager",
        )
        assert completion_result.is_success
        completion_message = completion_result.value

        # Verify workflow integrity
        assert startup_message.metadata["level"] == "info"
        assert startup_event.metadata["event_type"] == "SystemStarted"
        assert completion_message.metadata["level"] == "info"

        # All should be different instances but related workflow
        workflow_items = [startup_message, startup_event, completion_message]
        assert len(workflow_items) == 3
        assert all(hasattr(item, "metadata") for item in workflow_items)

    def test_event_sourcing_aggregate_reconstruction(self) -> None:
        """Test event sourcing pattern with aggregate reconstruction from events."""
        aggregate_id = "order_555"
        events = []

        # Event 1: Order Created
        create_data = {
            "order_id": aggregate_id,
            "customer_id": "customer_789",
            "items": [{"product": "laptop", "quantity": 1, "price": 1200.00}],
            "total": 1200.00,
        }
        result = FlextPayload.create_event(
            "OrderCreated", create_data, aggregate_id=aggregate_id, version=1
        )
        assert result.is_success
        events.append(result.value)

        # Event 2: Payment Added
        payment_data = {
            "order_id": aggregate_id,
            "payment_method": "credit_card",
            "amount": 1200.00,
            "transaction_id": "txn_001",
        }
        result = FlextPayload.create_event(
            "PaymentAdded", payment_data, aggregate_id=aggregate_id, version=2
        )
        assert result.is_success
        events.append(result.value)

        # Event 3: Order Shipped
        shipping_data = {
            "order_id": aggregate_id,
            "tracking_number": "track_123",
            "carrier": "FedEx",
            "shipped_at": datetime.now(UTC).isoformat(),
        }
        result = FlextPayload.create_event(
            "OrderShipped", shipping_data, aggregate_id=aggregate_id, version=3
        )
        assert result.is_success
        events.append(result.value)

        # Event 4: Order Delivered
        delivery_data = {
            "order_id": aggregate_id,
            "delivered_at": datetime.now(UTC).isoformat(),
            "signature": "customer_signature",
        }
        result = FlextPayload.create_event(
            "OrderDelivered", delivery_data, aggregate_id=aggregate_id, version=4
        )
        assert result.is_success
        events.append(result.value)

        # Reconstruct aggregate state from events
        aggregate_state = {"order_id": aggregate_id, "status": "created"}

        for event in events:
            event_type = event.metadata["event_type"]
            event_data = event.value

            if event_type == "OrderCreated":
                aggregate_state.update(
                    {
                        "customer_id": event_data["customer_id"],
                        "items": event_data["items"],
                        "total": event_data["total"],
                        "status": "created",
                    }
                )
            elif event_type == "PaymentAdded":
                aggregate_state.update(
                    {
                        "payment_method": event_data["payment_method"],
                        "transaction_id": event_data["transaction_id"],
                        "status": "paid",
                    }
                )
            elif event_type == "OrderShipped":
                aggregate_state.update(
                    {
                        "tracking_number": event_data["tracking_number"],
                        "carrier": event_data["carrier"],
                        "status": "shipped",
                    }
                )
            elif event_type == "OrderDelivered":
                aggregate_state.update(
                    {
                        "delivered_at": event_data["delivered_at"],
                        "status": "delivered",
                    }
                )

        # Verify reconstructed state
        assert aggregate_state["order_id"] == aggregate_id
        assert aggregate_state["customer_id"] == "customer_789"
        assert aggregate_state["total"] == 1200.00
        assert aggregate_state["payment_method"] == "credit_card"
        assert aggregate_state["tracking_number"] == "track_123"
        assert aggregate_state["status"] == "delivered"

    def test_cross_service_payload_patterns(self) -> None:
        """Test cross-service communication payload patterns."""
        # Service A creates payload for Service B
        service_a_data = {
            "request_id": "req_001",
            "operation": "user_validation",
            "user_id": "user_999",
            "validation_rules": ["email", "age", "country"],
        }

        result = FlextPayload.create(
            service_a_data,
            source_service="user_service",
            target_service="validation_service",
            correlation_id="corr_123",
        )
        assert result.is_success

        cross_service_payload = result.value

        # Service B processes and responds
        response_data = {
            "request_id": "req_001",
            "validation_result": {"email": "valid", "age": "valid", "country": "valid"},
            "overall_status": "passed",
        }

        response_result = FlextPayload.create(
            response_data,
            source_service="validation_service",
            target_service="user_service",
            correlation_id="corr_123",
            response_to="req_001",
        )
        assert response_result.is_success

        response_payload = response_result.value

        # Verify cross-service correlation
        assert (
            cross_service_payload.metadata["correlation_id"]
            == response_payload.metadata["correlation_id"]
        )
        assert (
            cross_service_payload.metadata["source_service"]
            != response_payload.metadata["source_service"]
        )
        assert response_payload.metadata["response_to"] == "req_001"
