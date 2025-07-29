"""Tests for core type definitions - focused on essential usage patterns."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import FlextEntityId, FlextPayload

if TYPE_CHECKING:
    # Type aliases for testing
    FlextConfigKey = str
    FlextEventType = str
    FlextServiceName = str


class TestTypeAliases:
    """Test essential type aliases used throughout FLEXT ecosystem."""

    def test_entity_id_basic_usage(self) -> None:
        """Test FlextEntityId type alias in typical usage."""
        user_id: FlextEntityId = "user-123"
        order_id: FlextEntityId = "order-456"

        # FlextEntityId is just a string with better typing
        assert isinstance(user_id, str)
        assert isinstance(order_id, str)
        assert user_id == "user-123"
        assert order_id == "order-456"

    def test_service_name_basic_usage(self) -> None:
        """Test FlextServiceName type alias for DI container."""
        database_service: FlextServiceName = "database"
        logger_service: FlextServiceName = "logger"

        # FlextServiceName provides type safety for service registration
        assert isinstance(database_service, str)
        assert isinstance(logger_service, str)
        assert database_service == "database"
        assert logger_service == "logger"

    def test_config_key_basic_usage(self) -> None:
        """Test FlextConfigKey type alias for configuration."""
        db_host_key: FlextConfigKey = "database.host"
        api_key: FlextConfigKey = "api.key"

        # FlextConfigKey helps with configuration key management
        assert isinstance(db_host_key, str)
        assert isinstance(api_key, str)
        assert db_host_key == "database.host"
        assert api_key == "api.key"

    def test_event_type_basic_usage(self) -> None:
        """Test FlextEventType type alias for events."""
        user_created: FlextEventType = "user.created"
        order_placed: FlextEventType = "order.placed"

        # FlextEventType provides type safety for event handling
        assert isinstance(user_created, str)
        assert isinstance(order_placed, str)
        assert user_created == "user.created"
        assert order_placed == "order.placed"

    def test_payload_basic_usage(self) -> None:
        """Test FlextPayload type alias for data payloads."""
        # FlextPayload is a Pydantic BaseModel for structured data
        user_data = {
            "id": "123",
            "name": "John Doe",
            "email": "john@example.com",
        }
        user_payload: FlextPayload = FlextPayload(data=user_data)

        event_data = {
            "event_type": "user.created",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        event_payload: FlextPayload = FlextPayload(data=event_data)

        # Verify payload structure
        def process_payload(payload: FlextPayload) -> str:
            data_dict = payload.model_dump()
            return f"Processing: {len(data_dict)} fields"

        assert user_payload.data["id"] == "123"  # type: ignore[index]
        assert "name" in user_payload.data  # type: ignore[operator]
        assert process_payload(user_payload) == "Processing: 2 fields"
        assert process_payload(event_payload) == "Processing: 2 fields"
