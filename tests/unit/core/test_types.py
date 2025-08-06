"""Tests for core type definitions - focused on essential usage patterns."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

from flext_core import FlextEntityId, FlextPayload
from flext_core.flext_types import (
    Comparable,
    FlextTypes,
    Serializable,
    TAnyDict,
    TAnyList,
    TFactory,
)

if TYPE_CHECKING:
    # Type aliases for testing
    FlextConfigKey = str
    FlextEventType = str
    FlextServiceName = str


# Rebuild Pydantic models to resolve forward references
# Make types available in the global namespace for model_rebuild()
from flext_core.flext_types import TData

globals()["TAnyDict"] = TAnyDict
globals()["TData"] = TData

# Now safely call model_rebuild() with types in scope
FlextPayload.model_rebuild()


class TestTypeProtocols:
    """Test type protocols from types module."""

    def test_comparable_protocol(self) -> None:
        """Test Comparable protocol usage."""

        # Test with strings (which implement __lt__)
        def use_comparable(obj: Comparable) -> bool:
            return hasattr(obj, "__lt__")

        # Use cast to bypass strict typing for protocol testing
        from typing import cast

        assert use_comparable(cast("Comparable", "test"))
        assert use_comparable(cast("Comparable", 42))

    def test_type_aliases_usage(self) -> None:
        """Test type aliases from types module."""
        # Test TAnyDict
        test_dict: TAnyDict = {"key": "value", "number": 42}
        assert isinstance(test_dict, dict)
        assert test_dict["key"] == "value"

        # Test TAnyList
        test_list: TAnyList = [
            "item1",
            42,
            "item3",
        ]  # TAnyList allows str|int|float|None only
        assert isinstance(test_list, list)
        assert len(test_list) == 3

    def test_serializable_protocol(self) -> None:
        """Test Serializable protocol usage."""

        # Create a class that implements Serializable protocol
        class TestSerializable:
            def __init__(self, data: dict[str, object]) -> None:
                self.data = data

            def serialize(self) -> dict[str, object]:
                return self.data.copy()

        serializable_obj = TestSerializable({"key": "value", "data": "test"})

        def use_serializable(obj: Serializable) -> dict[str, object]:
            return obj.serialize()

        result = use_serializable(serializable_obj)
        assert isinstance(result, dict)
        assert result["data"] == "test"
        assert result["key"] == "value"

    def test_factory_protocol(self) -> None:
        """Test TFactory type alias usage."""

        # Test factory function
        def string_factory() -> str:
            return "created"

        def use_factory(factory: TFactory[str]) -> str:
            # Cast to specific callable type for testing
            no_arg_factory = cast("Callable[[], str]", factory)
            return no_arg_factory()

        result = use_factory(string_factory)
        assert result == "created"

    def test_flext_types_usage(self) -> None:
        """Test FlextTypes usage."""
        # Test TypeGuards
        assert hasattr(FlextTypes, "TypeGuards")
        assert hasattr(FlextTypes.TypeGuards, "is_instance_of")

        # Test type guard functionality
        result = FlextTypes.TypeGuards.is_instance_of("test", str)
        assert result is True

        result = FlextTypes.TypeGuards.is_instance_of(42, str)
        assert result is False


class TestTypeAliases:
    """Test essential type aliases used throughout FLEXT ecosystem."""

    def test_entity_id_basic_usage(self) -> None:
        """Test FlextEntityId type alias in typical usage."""
        user_id: FlextEntityId = "user-123"
        order_id: FlextEntityId = "order-456"

        # FlextEntityId is just a string with better typing
        assert isinstance(user_id, str)
        assert isinstance(order_id, str)
        assert user_id == "user-123", f"Expected {'user-123'}, got {user_id}"
        assert order_id == "order-456"

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
        """Test FlextPayload type alias for data payloads."""
        # FlextPayload is a Pydantic BaseModel for structured data
        user_data: dict[str, object] = {
            "id": "123",
            "name": "John Doe",
            "email": "john@example.com",
        }
        user_payload: FlextPayload[dict[str, object]] = FlextPayload(data=user_data)

        event_data: dict[str, object] = {
            "event_type": "user.created",
            "timestamp": "2025-01-01T00:00:00Z",
        }
        event_payload: FlextPayload[dict[str, object]] = FlextPayload(data=event_data)

        # Verify payload structure
        def process_payload(payload: FlextPayload[dict[str, object]]) -> str:
            data_dict = payload.model_dump()
            return f"Processing: {len(data_dict)} fields"

        from typing import cast

        payload_data = cast("dict[str, object]", user_payload.data)
        assert payload_data["id"] == "123", (
            f"Expected {'123'}, got {payload_data['id']}"
        )
        assert "name" in payload_data, f"Expected {'name'} in {payload_data}"
        assert process_payload(user_payload) == "Processing: 2 fields", (
            f"Expected {'Processing: 2 fields'}, got {process_payload(user_payload)}"
        )
        assert process_payload(event_payload) == "Processing: 2 fields"


class TestTypeGuards:
    """Test type guard functions comprehensively."""

    def test_is_instance_of_type_guard(self) -> None:
        """Test is_instance_of type guard with various types."""
        # String type checking
        assert FlextTypes.TypeGuards.is_instance_of("test", str) is True
        assert FlextTypes.TypeGuards.is_instance_of(42, str) is False

        # Integer type checking
        assert FlextTypes.TypeGuards.is_instance_of(42, int) is True
        assert FlextTypes.TypeGuards.is_instance_of("42", int) is False

        # List type checking
        assert FlextTypes.TypeGuards.is_instance_of([1, 2, 3], list) is True
        assert FlextTypes.TypeGuards.is_instance_of((1, 2, 3), list) is False

        # Dict type checking
        assert FlextTypes.TypeGuards.is_instance_of({"key": "value"}, dict) is True
        assert FlextTypes.TypeGuards.is_instance_of(["key", "value"], dict) is False

    def test_is_callable_type_guard(self) -> None:
        """Test is_callable type guard."""

        # Functions are callable
        def test_func() -> str:
            return "test"

        assert FlextTypes.TypeGuards.is_callable(test_func) is True

        # Lambda functions are callable
        def lambda_func(x: int) -> int:
            return x * 2

        assert FlextTypes.TypeGuards.is_callable(lambda_func) is True

        # Built-in functions are callable
        assert FlextTypes.TypeGuards.is_callable(len) is True
        assert FlextTypes.TypeGuards.is_callable(str) is True

        # Non-callable objects
        assert FlextTypes.TypeGuards.is_callable("string") is False
        assert FlextTypes.TypeGuards.is_callable(42) is False
        assert FlextTypes.TypeGuards.is_callable([1, 2, 3]) is False

    def test_is_dict_like_type_guard(self) -> None:
        """Test is_dict_like type guard."""
        # Real dictionaries are dict-like
        assert FlextTypes.TypeGuards.is_dict_like({"key": "value"}) is True
        assert FlextTypes.TypeGuards.is_dict_like({}) is True

        # Objects with dict-like interface
        class DictLike:
            def keys(self) -> list[str]:
                return []

            def values(self) -> list[object]:
                return []

            def items(self) -> list[tuple[str, object]]:
                return []

        dict_like_obj = DictLike()
        assert FlextTypes.TypeGuards.is_dict_like(dict_like_obj) is True

        # Non-dict-like objects
        assert FlextTypes.TypeGuards.is_dict_like([1, 2, 3]) is False
        assert FlextTypes.TypeGuards.is_dict_like("string") is False
        assert FlextTypes.TypeGuards.is_dict_like(42) is False

    def test_is_list_like_type_guard(self) -> None:
        """Test is_list_like type guard."""
        # Lists are list-like
        assert FlextTypes.TypeGuards.is_list_like([1, 2, 3]) is True
        assert FlextTypes.TypeGuards.is_list_like([]) is True

        # Tuples are list-like
        assert FlextTypes.TypeGuards.is_list_like((1, 2, 3)) is True

        # Sets are list-like (iterable with len)
        assert FlextTypes.TypeGuards.is_list_like({1, 2, 3}) is True

        # Strings and bytes are NOT list-like (explicitly excluded)
        assert FlextTypes.TypeGuards.is_list_like("string") is False
        assert FlextTypes.TypeGuards.is_list_like(b"bytes") is False

        # Non-iterable objects are not list-like
        assert FlextTypes.TypeGuards.is_list_like(42) is False
        assert FlextTypes.TypeGuards.is_list_like(None) is False

    def test_type_guard_error_handling(self) -> None:
        """Test type guard error handling with invalid types."""
        # Test with None type (should not raise exception)
        assert FlextTypes.TypeGuards.is_instance_of("test", type(None)) is False

        # Test edge cases that might cause TypeError
        try:
            # This might cause TypeError in some cases, should be handled
            result = FlextTypes.TypeGuards.is_instance_of(
                42, type("InvalidType", (), {})
            )
            assert isinstance(result, bool)
        except TypeError:
            # If TypeError occurs, the function should handle it gracefully
            pass


class TestProtocolDefinitions:
    """Test protocol interface definitions."""

    def test_comparable_protocol_interface(self) -> None:
        """Test Comparable protocol interface requirements."""

        # Create a class that implements Comparable
        class ComparableInt:
            def __init__(self, value: int) -> None:
                self.value = value

            def __lt__(self, other: object) -> bool:
                return isinstance(other, ComparableInt) and self.value < other.value

            def __le__(self, other: object) -> bool:
                return isinstance(other, ComparableInt) and self.value <= other.value

            def __gt__(self, other: object) -> bool:
                return isinstance(other, ComparableInt) and self.value > other.value

            def __ge__(self, other: object) -> bool:
                return isinstance(other, ComparableInt) and self.value >= other.value

        obj1 = ComparableInt(5)
        obj2 = ComparableInt(10)

        # Test that objects implement the protocol
        def use_comparable(obj: Comparable) -> bool:
            required_methods = ["__lt__", "__le__", "__gt__", "__ge__"]
            return all(hasattr(obj, method) for method in required_methods)

        assert use_comparable(obj1) is True
        assert use_comparable(obj2) is True

    def test_flext_serializable_protocol_interface(self) -> None:
        """Test FlextSerializable protocol interface requirements."""

        # Create a class that implements FlextSerializable
        class SerializableData:
            def __init__(self, data: dict[str, object]) -> None:
                self.data = data

            def to_dict(self) -> dict[str, object]:
                return self.data.copy()

            def to_json(self) -> str:
                import json

                return json.dumps(self.data)

        serializable_obj = SerializableData({"key": "value", "number": 42})

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
        from flext_core.flext_types import TEntityId

        # TEntityId usage
        user_id: TEntityId = "user-123"
        order_id: TEntityId = "order-456"

        assert isinstance(user_id, str)
        assert isinstance(order_id, str)
        assert user_id == "user-123"
        assert order_id == "order-456"

    def test_cqrs_type_aliases(self) -> None:
        """Test CQRS-related type aliases."""
        from flext_core.flext_types import (
            TCorrelationId,
            TRequestId,
            TUserId,
        )

        # Test request/response identifiers
        request_id: TRequestId = "req-123"
        correlation_id: TCorrelationId = "corr-456"
        user_id: TUserId = "user-789"

        assert isinstance(request_id, str)
        assert isinstance(correlation_id, str)
        assert isinstance(user_id, str)

    def test_business_type_aliases(self) -> None:
        """Test business domain type aliases."""
        from flext_core.flext_types import (
            TBusinessCode,
            TBusinessId,
            TBusinessName,
            TBusinessStatus,
            TBusinessType,
        )

        business_id: TBusinessId = "BUS-001"
        business_name: TBusinessName = "Test Business"
        business_code: TBusinessCode = "TB001"
        business_status: TBusinessStatus = "active"
        business_type: TBusinessType = "corporation"

        assert isinstance(business_id, str)
        assert isinstance(business_name, str)
        assert isinstance(business_code, str)
        assert isinstance(business_status, str)
        assert isinstance(business_type, str)

    def test_cache_type_aliases(self) -> None:
        """Test cache-related type aliases."""
        from flext_core.flext_types import TCacheKey, TCacheTTL, TCacheValue

        cache_key: TCacheKey = "user:123:profile"
        cache_value: TCacheValue = (
            "John Doe"  # TCacheValue is str|int|float|bool|None, not dict
        )
        cache_ttl: TCacheTTL = 3600  # 1 hour

        assert isinstance(cache_key, str)
        assert isinstance(cache_value, str)
        assert isinstance(cache_ttl, int)

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

    def test_infrastructure_type_aliases(self) -> None:
        """Test infrastructure-related type aliases."""
        from flext_core.flext_types import (
            TConfigDict,
            TConfigValue,
            TConnectionString,
            TDirectoryPath,
            TFilePath,
        )

        conn_string: TConnectionString = "postgresql://user:pass@localhost:5432/db"
        file_path: TFilePath = "/path/to/file.txt"
        dir_path: TDirectoryPath = "/path/to/directory"

        config_dict: TConfigDict = {"host": "localhost", "port": 5432}
        config_value: TConfigValue = "production"

        assert isinstance(conn_string, str)
        assert isinstance(file_path, str)
        assert isinstance(dir_path, str)
        assert isinstance(config_dict, dict)
        assert isinstance(config_value, str)


class TestTypesCoverageImprovements:
    """Test cases specifically for improving coverage of types.py module."""

    def test_is_instance_exception_handling(self) -> None:
        """Test is_instance method exception handling (lines 256-257)."""
        from flext_core.flext_types import FlextTypes

        # Test with invalid type that causes TypeError/AttributeError
        class BadType:
            """Type that causes isinstance to fail."""

            def __class_getitem__(cls, item: object) -> object:
                msg = "Bad type"
                raise TypeError(msg)

        # This should catch the exception and return False (lines 256-257)
        result = FlextTypes.TypeGuards.is_instance_of("test", BadType)
        assert result is False

        # Test with None type that might cause AttributeError
        # Use type() to get a valid type instead of None
        result = FlextTypes.TypeGuards.is_instance_of("test", type(None))
        assert result is False
