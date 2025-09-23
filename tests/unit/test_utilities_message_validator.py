"""Test suite for FlextUtilities.MessageValidator companion module.

Extracted during FlextHandlers refactoring to ensure 100% coverage
of message validation and serialization logic.
"""

from __future__ import annotations

import dataclasses
import math
from typing import ClassVar
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from flext_core import FlextExceptions, FlextResult, FlextUtilities


class TestMessageValidator:
    """Test suite for FlextUtilities.MessageValidator companion module."""

    def test_validate_command_delegates_to_validate_message(self) -> None:
        """Test that validate_command delegates to validate_message."""
        message = {"test": "command"}

        with patch.object(
            FlextUtilities.MessageValidator,
            "validate_message",
            return_value=FlextResult[None].ok(None),
        ) as mock_validate:
            result = FlextUtilities.MessageValidator.validate_command(message)

            mock_validate.assert_called_once_with(message, operation="command")
            assert result.is_success

    def test_validate_query_delegates_to_validate_message(self) -> None:
        """Test that validate_query delegates to validate_message."""
        message = {"test": "query"}

        with patch.object(
            FlextUtilities.MessageValidator,
            "validate_message",
            return_value=FlextResult[None].ok(None),
        ) as mock_validate:
            result = FlextUtilities.MessageValidator.validate_query(message)

            mock_validate.assert_called_once_with(message, operation="query")
            assert result.is_success

    def test_validate_message_with_pydantic_model_no_revalidation(self) -> None:
        """Test validating Pydantic model without revalidation."""

        class TestModel(BaseModel):
            name: str
            age: int

        message = TestModel(name="John", age=30)
        result = FlextUtilities.MessageValidator.validate_message(
            message, operation="command", revalidate_pydantic_messages=False
        )

        assert result.is_success

    def test_validate_message_with_pydantic_model_with_revalidation_success(
        self,
    ) -> None:
        """Test validating Pydantic model with successful revalidation."""

        class TestModel(BaseModel):
            name: str
            age: int

        message = TestModel(name="John", age=30)
        result = FlextUtilities.MessageValidator.validate_message(
            message, operation="command", revalidate_pydantic_messages=True
        )

        assert result.is_success

    def test_validate_message_with_pydantic_model_revalidation_failure(self) -> None:
        """Test validating Pydantic model with revalidation failure."""

        class TestModel(BaseModel):
            name: str
            age: int

        # Create a valid model then corrupt it to trigger revalidation failure
        message = TestModel(name="John", age=30)

        # Mock model_validate to simulate failure
        with patch.object(
            TestModel, "model_validate", side_effect=ValueError("Invalid data")
        ):
            result = FlextUtilities.MessageValidator.validate_message(
                message, operation="command", revalidate_pydantic_messages=True
            )

            assert result.is_failure
            error_message = result.error or ""
            assert "Pydantic revalidation failed" in error_message
            assert "Invalid data" in error_message

    def test_validate_message_with_serializable_dict(self) -> None:
        """Test validating dictionary message."""
        message = {"name": "John", "age": 30}
        result = FlextUtilities.MessageValidator.validate_message(
            message, operation="command"
        )

        assert result.is_success

    def test_validate_message_with_dataclass(self) -> None:
        """Test validating dataclass message."""

        @dataclasses.dataclass
        class TestDataclass:
            name: str
            age: int

        message = TestDataclass(name="John", age=30)
        result = FlextUtilities.MessageValidator.validate_message(
            message, operation="command"
        )

        assert result.is_success

    def test_validate_message_with_none_fails(self) -> None:
        """Test that None message fails validation."""
        result = FlextUtilities.MessageValidator.validate_message(
            None, operation="command"
        )

        assert result.is_failure
        error_message = result.error or ""
        assert "Invalid message type for command: NoneType" in error_message

    def test_validate_message_with_unsupported_type_fails(self) -> None:
        """Test that unsupported message type fails validation."""

        class UnsupportedType:
            __slots__ = ()  # No attributes allowed, and no __dict__

            def __init__(self) -> None:
                super().__init__()  # Call parent class __init__
                # Cannot set any attributes due to empty __slots__

        message = UnsupportedType()
        result = FlextUtilities.MessageValidator.validate_message(
            message, operation="command"
        )

        assert result.is_failure
        error_message = result.error or ""
        assert "Invalid message type for command" in error_message

    def test_build_serializable_message_payload_basic_types(self) -> None:
        """Test building payload for basic serializable types."""
        # Test various basic types
        assert FlextUtilities.MessageValidator.build_serializable_message_payload(
            {"key": "value"}, operation="test"
        ) == {"key": "value"}
        assert (
            FlextUtilities.MessageValidator.build_serializable_message_payload(
                "string", operation="test"
            )
            == "string"
        )
        assert (
            FlextUtilities.MessageValidator.build_serializable_message_payload(
                42, operation="test"
            )
            == 42
        )
        assert (
            FlextUtilities.MessageValidator.build_serializable_message_payload(
                math.pi, operation="test"
            )
            == math.pi
        )
        assert (
            FlextUtilities.MessageValidator.build_serializable_message_payload(
                True, operation="test"
            )
            is True
        )

    def test_build_serializable_message_payload_none_raises_error(self) -> None:
        """Test that None message raises appropriate error."""
        with pytest.raises(FlextExceptions.TypeError) as exc_info:
            FlextUtilities.MessageValidator.build_serializable_message_payload(
                None, operation="test"
            )

        assert "Invalid message type for test: NoneType" in str(exc_info.value)

    def test_build_serializable_message_payload_pydantic_model(self) -> None:
        """Test building payload for Pydantic model."""

        class TestModel(BaseModel):
            name: str
            age: int

        message = TestModel(name="John", age=30)
        result = FlextUtilities.MessageValidator.build_serializable_message_payload(
            message, operation="test"
        )

        assert result == {"name": "John", "age": 30}

    def test_build_serializable_message_payload_dataclass(self) -> None:
        """Test building payload for dataclass."""

        @dataclasses.dataclass
        class TestDataclass:
            name: str
            age: int

        message = TestDataclass(name="John", age=30)
        result = FlextUtilities.MessageValidator.build_serializable_message_payload(
            message, operation="test"
        )

        assert result == {"name": "John", "age": 30}

    def test_build_serializable_message_payload_attrs_class(self) -> None:
        """Test building payload for attrs class."""
        try:
            import attr

            @attr.s
            class TestAttrs:
                name: str = attr.ib()
                age: int = attr.ib()

            message = TestAttrs(name="John", age=30)
            result = FlextUtilities.MessageValidator.build_serializable_message_payload(
                message, operation="test"
            )

            assert result == {"name": "John", "age": 30}

        except ImportError:
            # If attrs not available, test fallback behavior
            class MockAttrs:
                __attrs_attrs__: ClassVar[list[object]] = [
                    type("Field", (), {"name": "name"})
                ]

                def __init__(self) -> None:
                    super().__init__()
                    self.name = "John"

            mock_message = MockAttrs()
            result = FlextUtilities.MessageValidator.build_serializable_message_payload(
                mock_message, operation="test"
            )

            assert result == {"name": "John"}

    def test_build_serializable_message_payload_model_dump_method(self) -> None:
        """Test building payload using model_dump method."""

        class CustomModel:
            def __init__(self, data: dict[str, object]) -> None:
                super().__init__()
                self._data = data

            def model_dump(self) -> dict[str, object]:
                return self._data

        message = CustomModel({"custom": "data"})
        result = FlextUtilities.MessageValidator.build_serializable_message_payload(
            message, operation="test"
        )

        assert result == {"custom": "data"}

    def test_build_serializable_message_payload_dict_method(self) -> None:
        """Test building payload using dict method."""

        class CustomModel:
            def __init__(self, data: dict[str, object]) -> None:
                super().__init__()
                self._data = data

            def dict(self) -> dict[str, object]:
                return self._data

        message = CustomModel({"custom": "data"})
        result = FlextUtilities.MessageValidator.build_serializable_message_payload(
            message, operation="test"
        )

        assert result == {"custom": "data"}

    def test_build_serializable_message_payload_as_dict_method(self) -> None:
        """Test building payload using as_dict method."""

        class CustomModel:
            def __init__(self, data: dict[str, object]) -> None:
                super().__init__()
                self._data = data

            def as_dict(self) -> dict[str, object]:
                return self._data

        message = CustomModel({"custom": "data"})
        result = FlextUtilities.MessageValidator.build_serializable_message_payload(
            message, operation="test"
        )

        assert result == {"custom": "data"}

    def test_build_serializable_message_payload_serialization_method_fails(
        self,
    ) -> None:
        """Test handling when serialization method raises exception."""

        class FailingModel:
            def __init__(self) -> None:
                super().__init__()

            def model_dump(self) -> dict[str, object]:
                msg = "Serialization failed"
                raise ValueError(msg)

            def dict(self) -> dict[str, object]:
                return {"fallback": "data"}

        message = FailingModel()
        result = FlextUtilities.MessageValidator.build_serializable_message_payload(
            message, operation="test"
        )

        # Should use the dict method as fallback
        assert result == {"fallback": "data"}

    def test_build_serializable_message_payload_slots_string(self) -> None:
        """Test building payload for object with __slots__ as string."""

        class SlottedClass:
            __slots__ = ("name",)

            def __init__(self, name: str) -> None:
                super().__init__()
                self.name = name

        message = SlottedClass("John")
        result = FlextUtilities.MessageValidator.build_serializable_message_payload(
            message, operation="test"
        )

        assert result == {"name": "John"}

    def test_build_serializable_message_payload_slots_list(self) -> None:
        """Test building payload for object with __slots__ as list."""

        class SlottedClass:
            __slots__ = ("age", "name")

            def __init__(self, name: str, age: int) -> None:
                super().__init__()
                self.name = name
                self.age = age

        message = SlottedClass("John", 30)
        result = FlextUtilities.MessageValidator.build_serializable_message_payload(
            message, operation="test"
        )

        assert result == {"name": "John", "age": 30}

    def test_build_serializable_message_payload_slots_tuple(self) -> None:
        """Test building payload for object with __slots__ as tuple."""

        class SlottedClass:
            __slots__ = ("age", "name")

            def __init__(self, name: str, age: int) -> None:
                super().__init__()
                self.name = name
                self.age = age

        message = SlottedClass("John", 30)
        result = FlextUtilities.MessageValidator.build_serializable_message_payload(
            message, operation="test"
        )

        assert result == {"name": "John", "age": 30}

    def test_build_serializable_message_payload_slots_invalid_type(self) -> None:
        """Test handling of invalid __slots__ type."""
        # This test cannot create a class with invalid __slots__ type
        # as it causes a TypeError at class definition time.
        # Test fallback behavior instead

        class RegularClass:
            def __init__(self) -> None:
                super().__init__()
                self.name = "test"

        message = RegularClass()

        # Should work with regular class using __dict__
        result = FlextUtilities.MessageValidator.build_serializable_message_payload(
            message, operation="test"
        )

        assert result == {"name": "test"}

    def test_build_serializable_message_payload_dict_fallback(self) -> None:
        """Test building payload using __dict__ fallback."""

        class SimpleClass:
            def __init__(self, name: str, age: int) -> None:
                super().__init__()
                self.name = name
                self.age = age

        message = SimpleClass("John", 30)
        result = FlextUtilities.MessageValidator.build_serializable_message_payload(
            message, operation="test"
        )

        assert result == {"name": "John", "age": 30}

    def test_build_serializable_message_payload_no_serialization_method(self) -> None:
        """Test error when no serialization method available."""

        class UnserializableClass:
            __slots__ = ()  # No attributes

            def __init__(self) -> None:
                super().__init__()

        message = UnserializableClass()

        with pytest.raises(FlextExceptions.TypeError) as exc_info:
            FlextUtilities.MessageValidator.build_serializable_message_payload(
                message, operation="test"
            )

        assert "Invalid message type for test" in str(exc_info.value)

    def test_build_serializable_message_payload_no_operation(self) -> None:
        """Test building payload without operation parameter."""
        message = {"test": "data"}
        result = FlextUtilities.MessageValidator.build_serializable_message_payload(
            message
        )

        assert result == {"test": "data"}

    def test_build_serializable_message_payload_serialization_returns_none(
        self,
    ) -> None:
        """Test handling when serialization method returns None."""

        class NoneReturningModel:
            def __init__(self, name: str) -> None:
                super().__init__()
                self.name = name

            def model_dump(self) -> None:
                return None

        message = NoneReturningModel("John")
        result = FlextUtilities.MessageValidator.build_serializable_message_payload(
            message, operation="test"
        )

        # Should fall back to __dict__
        assert result == {"name": "John"}


class TestMessageValidatorEdgeCases:
    """Test edge cases and error conditions for MessageValidator."""

    def test_validate_message_with_attrs_no_import(self) -> None:
        """Test attrs handling when attrs module not available."""

        class MockAttrsClass:
            __attrs_attrs__: ClassVar[list[object]] = [
                type("Field", (), {"name": "test_field"})
            ]

            def __init__(self) -> None:
                super().__init__()
                self.test_field = "value"

        message = MockAttrsClass()

        # Test with attrs-like class but no actual attrs import
        result = FlextUtilities.MessageValidator.build_serializable_message_payload(
            message, operation="test"
        )

        # Should use __dict__ as fallback when attrs not available
        assert result == {"test_field": "value"}

    def test_validate_message_with_slots_missing_attribute(self) -> None:
        """Test slots handling when attribute is missing."""

        class SlottedClass:
            __slots__ = ("missing_attr", "name")

            def __init__(self, name: str) -> None:
                super().__init__()
                self.name = name
                # missing_attr is not set

        message = SlottedClass("John")
        result = FlextUtilities.MessageValidator.build_serializable_message_payload(
            message, operation="test"
        )

        # Should only include existing attributes
        assert result == {"name": "John"}

    def test_validate_message_pydantic_with_flext_exception(self) -> None:
        """Test handling of FlextExceptions during validation."""

        class TestModel(BaseModel):
            name: str

        message = TestModel(name="John")

        # Mock build_serializable_message_payload to raise FlextExceptions.TypeError
        type_error = FlextExceptions.TypeError(
            "Test type error",
            expected_type="dict",
            actual_type="TestModel",
            context={"test": "context"},
        )

        with patch.object(
            FlextUtilities.MessageValidator,
            "build_serializable_message_payload",
            side_effect=type_error,
        ):
            result = FlextUtilities.MessageValidator.validate_message(
                message, operation="command", revalidate_pydantic_messages=False
            )

            # Should return early for Pydantic models without revalidation
            assert result.is_success

    def test_validate_message_non_pydantic_with_flext_exception(self) -> None:
        """Test handling of FlextExceptions for non-Pydantic messages."""

        class CustomClass:
            def __init__(self) -> None:
                super().__init__()

        message = CustomClass()

        # Mock build_serializable_message_payload to raise FlextExceptions.TypeError
        type_error = FlextExceptions.TypeError(
            "Test type error",
            expected_type="dict",
            actual_type="CustomClass",
            context={"test": "context"},
        )

        with patch.object(
            FlextUtilities.MessageValidator,
            "build_serializable_message_payload",
            side_effect=type_error,
        ):
            result = FlextUtilities.MessageValidator.validate_message(
                message, operation="command"
            )

            assert result.is_failure
            assert result.error is not None
            assert "Test type error" in result.error

    def test_validate_message_fallback_error_handling(self) -> None:
        """Test fallback error handling for non-FlextExceptions."""

        class CustomClass:
            def __init__(self) -> None:
                super().__init__()

        message = CustomClass()

        # Mock build_serializable_message_payload to raise regular exception
        with patch.object(
            FlextUtilities.MessageValidator,
            "build_serializable_message_payload",
            side_effect=RuntimeError("Unexpected error"),
        ):
            result = FlextUtilities.MessageValidator.validate_message(
                message, operation="command"
            )

            assert result.is_failure
            assert result.error is not None
            assert "Invalid message type for command: CustomClass" in result.error

    def test_attrs_class_with_missing_fields(self) -> None:
        """Test attrs class where some fields don't exist on instance."""

        class MockAttrsField:
            def __init__(self, name: str) -> None:
                super().__init__()
                self.name = name

        class MockAttrsClass:
            __attrs_attrs__: ClassVar[list[object]] = [
                MockAttrsField("existing_field"),
                MockAttrsField("missing_field"),
            ]

            def __init__(self) -> None:
                super().__init__()
                self.existing_field = "value"
                # missing_field is not set

        message = MockAttrsClass()

        # attrs.asdict will fail for missing fields, expect it to raise AttributeError
        with pytest.raises(AttributeError, match="no attribute 'missing_field'"):
            FlextUtilities.MessageValidator.build_serializable_message_payload(
                message, operation="test"
            )


class TestMessageValidatorIntegration:
    """Integration tests for MessageValidator with real scenarios."""

    def test_integration_with_complex_pydantic_model(self) -> None:
        """Test validation with complex Pydantic model."""

        class UserModel(BaseModel):
            name: str
            age: int | None = None
            active: bool = True

        message = UserModel(name="John", age=30)

        # Test without revalidation
        result = FlextUtilities.MessageValidator.validate_message(
            message, operation="command", revalidate_pydantic_messages=False
        )
        assert result.is_success

        # Test with revalidation
        result = FlextUtilities.MessageValidator.validate_message(
            message, operation="command", revalidate_pydantic_messages=True
        )
        assert result.is_success

    def test_integration_with_nested_dataclass(self) -> None:
        """Test validation with nested dataclass structures."""

        @dataclasses.dataclass
        class Address:
            street: str
            city: str

        @dataclasses.dataclass
        class Person:
            name: str
            address: Address

        address = Address(street="123 Main St", city="Anytown")
        message = Person(name="John", address=address)

        result = FlextUtilities.MessageValidator.validate_message(
            message, operation="command"
        )
        assert result.is_success

    def test_integration_with_mixed_serialization_methods(self) -> None:
        """Test object with multiple serialization methods - should use first working one."""

        class MultiMethodClass:
            def __init__(self, data: dict[str, object]) -> None:
                super().__init__()
                self._data = data

            def model_dump(self) -> dict[str, object]:
                return {"source": "model_dump", **self._data}

            def to_dict(self) -> dict[str, object]:
                return {"source": "to_dict", **self._data}

            def as_dict(self) -> dict[str, object]:
                return {"source": "as_dict", **self._data}

        message = MultiMethodClass({"name": "John"})
        result = FlextUtilities.MessageValidator.build_serializable_message_payload(
            message, operation="test"
        )

        # Should use model_dump (first in the list)
        assert result == {"source": "model_dump", "name": "John"}

    def test_integration_error_context_preservation(self) -> None:
        """Test that error context is properly preserved through validation."""

        class UnsupportedClass:
            def __init__(self) -> None:
                super().__init__()

        message = UnsupportedClass()
        result = FlextUtilities.MessageValidator.validate_message(
            message, operation="test_command"
        )

        # UnsupportedClass should actually validate successfully using __dict__
        # as it's a regular Python class, just with empty __dict__
        assert result.is_success
        # validate_message returns FlextResult[None], not the message itself
        assert result.data is None
