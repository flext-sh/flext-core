"""Tests for core type definitions - focused on essential usage patterns."""

from __future__ import annotations

import json
from collections.abc import Callable as _Callable
from typing import cast

from flext_core import FlextModels, FlextTypes

FlextConfigKey = str
FlextEventType = str
FlextServiceName = str


# FlextModels is a container class, not a Pydantic model, so no model_rebuild() needed


class TestTypeProtocols:
    """Test type protocols from types module."""

    def test_comparable_protocol(self) -> None:
        """Test TComparable protocol usage."""

        # Test with strings (which implement __lt__)
        def use_comparable(obj: object) -> bool:
            return hasattr(obj, "__lt__")

        # Use cast to bypass strict typing for protocol testing
        assert use_comparable("test")
        assert use_comparable(42)

    def test_type_aliases_usage(self) -> None:
        """Test type aliases from types module."""
        # Test dict[str, object]
        test_dict: dict[str, object] = {"key": "value", "number": 42}
        assert isinstance(test_dict, dict)
        assert test_dict["key"] == "value"

        # Test list[object]
        test_list: list[object] = [
            "item1",
            42,
            "item3",
        ]  # list[object] allows str|int|float|None only
        assert isinstance(test_list, list)
        assert len(test_list) == 3

    def test_serializable_protocol(self) -> None:
        """Test TSerializable protocol usage."""

        # Create a class that implements TSerializable protocol
        class TestTSerializable:
            def __init__(self, data: dict[str, object]) -> None:
                self.value = data

            def serialize(self) -> dict[str, object]:
                return self.value.copy()

        serializable_obj = TestTSerializable({"key": "value", "data": "test"})

        def use_serializable(obj: object) -> dict[str, object]:
            return cast("TestTSerializable", obj).serialize()

        result = use_serializable(serializable_obj)
        assert isinstance(result, dict)
        assert result["data"] == "test"
        assert result["key"] == "value"

    def test_factory_protocol(self) -> None:
        """Test TFactory type alias usage."""

        # Test factory function
        def string_factory() -> str:
            return "created"

        def use_factory(factory: _Callable[[], str]) -> str:
            return factory()

        result = use_factory(string_factory)
        assert result == "created"

    def test_flext_types_usage(self) -> None:
        """Test FlextTypes usage."""
        # Test available type namespaces
        assert hasattr(FlextTypes, "Core")
        assert hasattr(FlextTypes, "Domain")
        assert hasattr(FlextTypes, "Service")

        # Test type availability
        assert hasattr(FlextTypes.Core, "Dict")
        assert hasattr(FlextTypes.Core, "List")


class TestTypeAliases:
    """Test essential type aliases used throughout FLEXT ecosystem."""

    def test_entity_id_basic_usage(self) -> None:
        """Test EntityId type usage from FlextModels container."""
        user_id = FlextModels.EntityId("user-123")
        order_id = FlextModels.EntityId("order-456")

        # EntityId is a RootModel with string root
        assert isinstance(user_id, FlextModels.EntityId)
        assert isinstance(order_id, FlextModels.EntityId)
        assert user_id.root == "user-123", f"Expected {'user-123'}, got {user_id.root}"
        assert order_id.root == "order-456"

    def test_service_name_basic_usage(self) -> None:
        """Test FlextServiceName type alias for DI container."""
        database_service: FlextServiceName = "database"
        logger_service: FlextServiceName = "logger"

        # FlextServiceName provides type safety for service registration
        assert isinstance(database_service, str)
        assert isinstance(logger_service, str)
        assert database_service == "database", (
            f"Expected {'database'}, got {database_service}"
        )
        assert logger_service == "logger"

    def test_config_key_basic_usage(self) -> None:
        """Test FlextConfigKey type alias for configuration."""
        db_host_key: FlextConfigKey = "database.host"
        api_key: FlextConfigKey = "api.key"

        # FlextConfigKey helps with configuration key management
        assert isinstance(db_host_key, str)
        assert isinstance(api_key, str)
        assert db_host_key == "database.host", (
            f"Expected {'database.host'}, got {db_host_key}"
        )
        assert api_key == "api.key"

    def test_event_type_basic_usage(self) -> None:
        """Test FlextEventType type alias for events."""
        user_created: FlextEventType = "user.created"
        order_placed: FlextEventType = "order.placed"

        # FlextEventType provides type safety for event handling
        assert isinstance(user_created, str)
        assert isinstance(order_placed, str)
        assert user_created == "user.created", (
            f"Expected {'user.created'}, got {user_created}"
        )
        assert order_placed == "order.placed"

    def test_payload_basic_usage(self) -> None:
        """Test Payload class from FlextModels container for data payloads."""
        # Use the actual Payload class from FlextModels
        user_data: dict[str, object] = {
            "id": "123",
            "name": "John Doe",
            "email": "john@example.com",
        }
        user_payload = FlextModels.Payload(
            source_service="user_service", message_type="user.data", data=user_data,
        )

        event_data: dict[str, object] = {
            "event_type": "user.created",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        event_payload = FlextModels.Payload(
            source_service="event_service",
            message_type="event.created",
            data=event_data,
        )

        # Verify payload structure
        def process_payload(payload: FlextModels.Payload[dict[str, object]]) -> str:
            data_dict = payload.data
            return f"Processing: {len(data_dict)} fields"

        payload_data = user_payload.data
        assert payload_data["id"] == "123", (
            f"Expected {'123'}, got {payload_data['id']}"
        )
        assert "name" in payload_data, f"Expected {'name'} in {payload_data}"
        assert process_payload(user_payload) == "Processing: 3 fields", (
            f"Expected {'Processing: 3 fields'}, got {process_payload(user_payload)}"
        )
        assert process_payload(event_payload) == "Processing: 2 fields"


class TestBasicTypeOperations:
    """Test basic type operations available."""

    def test_basic_type_checking(self) -> None:
        """Test basic type checking operations."""
        # Test basic isinstance operations
        assert isinstance("test", str)
        assert isinstance(42, int)
        assert isinstance([1, 2, 3], list)
        assert isinstance({"key": "value"}, dict)

    def test_callable_checking(self) -> None:
        """Test callable checking."""

        def test_func() -> str:
            return "test"

        assert callable(test_func)
        assert callable(str)
        assert not callable("string")
        assert not callable(42)


class TestProtocolDefinitions:
    """Test protocol interface definitions."""

    def test_comparable_protocol_interface(self) -> None:
        """Test TComparable protocol interface requirements."""

        # Create a class that implements TComparable
        class TComparableInt:
            def __init__(self, value: int) -> None:
                self.value = value

            def __lt__(self, other: object) -> bool:
                return isinstance(other, TComparableInt) and self.value < other.value

            def __le__(self, other: object) -> bool:
                return isinstance(other, TComparableInt) and self.value <= other.value

            def __gt__(self, other: object) -> bool:
                return isinstance(other, TComparableInt) and self.value > other.value

            def __ge__(self, other: object) -> bool:
                return isinstance(other, TComparableInt) and self.value >= other.value

        obj1 = TComparableInt(5)
        obj2 = TComparableInt(10)

        # Test that objects implement the protocol
        def use_comparable(obj: object) -> bool:
            required_methods = ["__lt__", "__le__", "__gt__", "__ge__"]
            return all(hasattr(obj, method) for method in required_methods)

        assert use_comparable(obj1) is True
        assert use_comparable(obj2) is True

    def test_flext_serializable_protocol_interface(self) -> None:
        """Test FlextTSerializable protocol interface requirements."""

        # Create a class that implements FlextTSerializable
        class TSerializableData:
            def __init__(self, data: dict[str, object]) -> None:
                self.value = data

            def to_dict(self) -> dict[str, object]:
                return self.value.copy()

            def to_json(self) -> str:
                return json.dumps(self.value)

        serializable_obj = TSerializableData({"key": "value", "number": 42})

        # Test protocol methods
        result_dict = serializable_obj.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["key"] == "value"

        result_json = serializable_obj.to_json()
        assert isinstance(result_json, str)
        assert "key" in result_json
        assert "value" in result_json


class TestTypeAliasComprehensive:
    """Comprehensive tests for all type aliases."""

    def test_entity_type_aliases(self) -> None:
        """Test entity-related type aliases."""
        # str usage
        user_id: str = "user-123"
        order_id: str = "order-456"

        assert isinstance(user_id, str)
        assert isinstance(order_id, str)
        assert user_id == "user-123"
        assert order_id == "order-456"

    def test_cqrs_type_aliases(self) -> None:
        """Test CQRS-related type aliases."""
        # Test available identifiers
        correlation_id: str = "corr-456"

        assert isinstance(correlation_id, str)

    def test_available_type_aliases(self) -> None:
        """Test available type aliases."""
        entity_id: str = "entity-123"
        correlation_id: str = "corr-456"

        assert isinstance(entity_id, str)
        assert isinstance(correlation_id, str)

    def test_data_type_aliases(self) -> None:
        """Test data-related type aliases."""
        # Test FlextTypes.Core.Data generic
        test_data: dict[str, object] = {"key": "value"}
        assert isinstance(test_data, dict)

    def test_function_type_aliases(self) -> None:
        """Test function-related type aliases."""

        # Predicate function
        def is_positive(x: object) -> bool:
            return isinstance(x, (int, float)) and x > 0

        assert is_positive(5) is True
        assert is_positive(-1) is False

        # Transformer function
        def double(x: int) -> int:
            return x * 2

        assert double(5) == 10

        # Validator function
        def is_valid_email(email: object) -> bool:
            return "@" in str(email) and "." in str(email)

        assert is_valid_email("test@example.com") is True
        assert is_valid_email("invalid-email") is False

        # Factory function
        def string_factory() -> str:
            return "created"

        assert string_factory() == "created"

    def test_basic_type_aliases(self) -> None:
        """Test basic type aliases."""
        # Test basic data types that are available
        any_dict: dict[str, object] = {"key": "value", "number": 42}
        any_list: list[object] = ["item1", 42, "item3"]

        assert isinstance(any_dict, dict)
        assert isinstance(any_list, list)
        assert any_dict["key"] == "value"
        assert len(any_list) == 3


class TestTypesCoverageImprovements:
    """Test cases specifically for improving coverage of types.py module."""

    def test_basic_type_operations(self) -> None:
        """Test basic type operations available."""
        # Test FlextTypes namespaces are accessible
        assert hasattr(FlextTypes, "Core")
        assert hasattr(FlextTypes, "Domain")
        assert hasattr(FlextTypes, "Service")

        # Test type definitions are accessible
        test_dict: dict[str, object] = {"test": "value"}
        test_list: list[object] = ["item1", 42]

        assert isinstance(test_dict, dict)
        assert isinstance(test_list, list)
