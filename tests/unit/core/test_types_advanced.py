"""Tests for FLEXT Core advanced types - reducing boilerplate."""

from __future__ import annotations

from datetime import UTC, datetime

from flext_core.result import FlextResult
from flext_core.types import (
    FlextEither as Either,
    FlextPipe as Pipe,
    flext_ensure_dict as ensure_dict,
    flext_ensure_list as ensure_list,
    flext_ensure_result as ensure_result,
    flext_is_identifiable as is_identifiable,
    flext_is_result_type as is_result_type,
    flext_is_serializable as is_serializable,
    flext_is_timestamped as is_timestamped,
    flext_is_validatable as is_validatable,
)


class TestEitherType:
    """Test Either type functionality."""

    def test_either_right_creation(self) -> None:
        """Test creating right (success) values."""
        either = Either.right("success")
        assert either.is_right
        assert not either.is_left

    def test_either_left_creation(self) -> None:
        """Test creating left (error) values."""
        either = Either.left("error")
        assert either.is_left
        assert not either.is_right

    def test_either_map_on_right(self) -> None:
        """Test mapping function on right value."""
        either = Either.right("hello")
        result = either.map(str.upper)
        assert result.is_right
        assert result._value == "HELLO"

    def test_either_map_on_left(self) -> None:
        """Test mapping function on left value (should not apply)."""
        either = Either.left("error")
        result = either.map(str.upper)
        assert result.is_left
        assert result._value == "error"

    def test_either_flat_map_on_right(self) -> None:
        """Test flat mapping on right value."""
        either = Either.right("hello")
        result = either.flat_map(lambda x: Either.right(x.upper()))
        assert result.is_right
        assert result._value == "HELLO"

    def test_either_flat_map_on_left(self) -> None:
        """Test flat mapping on left value (should not apply)."""
        either = Either.left("error")
        result = either.flat_map(lambda x: Either.right(x.upper()))
        assert result.is_left
        assert result._value == "error"


class TestPipeType:
    """Test Pipe type functionality."""

    def test_pipe_usage(self) -> None:
        """Test using pipe type alias."""

        def double(x: int) -> FlextResult[int]:
            return FlextResult.ok(x * 2)

        # Pipe is just a type alias, so we use it directly
        pipe: Pipe = double
        result = pipe(5)
        assert result.is_success
        assert result.data == 10

    def test_pipe_composition(self) -> None:
        """Test composing pipe functions."""

        def add_one(x: int) -> FlextResult[int]:
            return FlextResult.ok(x + 1)

        def double(x: int) -> FlextResult[int]:
            return FlextResult.ok(x * 2)

        # Compose pipes manually using flat_map
        def combined_pipe(x: int) -> FlextResult[int]:
            return add_one(x).flat_map(double)

        result = combined_pipe(5)
        assert result.is_success
        assert result.data == 12  # (5 + 1) * 2


class TestTypeUtilities:
    """Test type checking and conversion utilities."""

    def test_is_result_type_with_result(self) -> None:
        """Test is_result_type with FlextResult."""
        result = FlextResult.ok("test")
        assert is_result_type(result)

    def test_is_result_type_with_non_result(self) -> None:
        """Test is_result_type with regular object."""
        assert not is_result_type("test")
        assert not is_result_type(42)

    def test_is_identifiable_with_id(self) -> None:
        """Test is_identifiable with object having id property."""

        class MockObject:
            @property
            def id(self) -> str:
                return "123"

        obj = MockObject()
        assert is_identifiable(obj)

    def test_is_identifiable_without_id(self) -> None:
        """Test is_identifiable with object lacking id attribute."""

        class MockObject:
            pass

        obj = MockObject()
        assert not is_identifiable(obj)

    def test_is_serializable_with_methods(self) -> None:
        """Test is_serializable with object having required methods."""

        class MockObject:
            def to_dict(self) -> dict[str, str]:
                return {"key": "value"}

            @classmethod
            def from_dict(cls, data: dict[str, str]) -> MockObject:
                return cls()

        obj = MockObject()
        assert is_serializable(obj)

    def test_is_serializable_without_methods(self) -> None:
        """Test is_serializable with object lacking required methods."""

        class MockObject:
            pass

        obj = MockObject()
        assert not is_serializable(obj)

    def test_is_timestamped_with_timestamps(self) -> None:
        """Test is_timestamped with object having timestamp attributes."""

        class MockObject:
            def __init__(self) -> None:
                self.created_at = datetime.now(UTC)
                self.updated_at = datetime.now(UTC)

        obj = MockObject()
        assert is_timestamped(obj)

    def test_is_timestamped_without_timestamps(self) -> None:
        """Test is_timestamped with object lacking timestamp attributes."""

        class MockObject:
            pass

        obj = MockObject()
        assert not is_timestamped(obj)

    def test_is_validatable_with_validate(self) -> None:
        """Test is_validatable with object having validate method."""

        class MockObject:
            def validate(self) -> FlextResult[None]:
                return FlextResult.ok(None)

        obj = MockObject()
        assert is_validatable(obj)

    def test_is_validatable_without_validate(self) -> None:
        """Test is_validatable with object lacking validate method."""

        class MockObject:
            pass

        obj = MockObject()
        assert not is_validatable(obj)


class TestConversionUtilities:
    """Test conversion utility functions."""

    def test_ensure_result_with_result(self) -> None:
        """Test ensure_result with existing FlextResult."""
        original = FlextResult.ok("test")
        result = ensure_result(original)
        assert result is original
        assert result.is_success
        assert result.data == "test"

    def test_ensure_result_with_regular_value(self) -> None:
        """Test ensure_result with regular value."""
        result = ensure_result("test")
        assert result.is_success
        assert result.data == "test"

    def test_ensure_list_with_list(self) -> None:
        """Test ensure_list with existing list."""
        original = ["a", "b", "c"]
        result = ensure_list(original)
        assert result is original
        assert result == ["a", "b", "c"]

    def test_ensure_list_with_tuple(self) -> None:
        """Test ensure_list with tuple."""
        result = ensure_list(("a", "b", "c"))
        assert isinstance(result, list)
        assert result == ["a", "b", "c"]

    def test_ensure_list_with_set(self) -> None:
        """Test ensure_list with set."""
        result = ensure_list({"a", "b", "c"})
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(item in result for item in ["a", "b", "c"])

    def test_ensure_list_with_single_value(self) -> None:
        """Test ensure_list with single value."""
        result = ensure_list("single")
        assert result == ["single"]

    def test_ensure_dict_with_dict(self) -> None:
        """Test ensure_dict with existing dict."""
        original = {"key": "value"}
        result = ensure_dict(original)
        assert result is original
        assert result == {"key": "value"}

    def test_ensure_dict_with_to_dict_method(self) -> None:
        """Test ensure_dict with object having to_dict method."""

        class MockObject:
            def to_dict(self) -> dict[str, str]:
                return {"mock": "data"}

        obj = MockObject()
        result = ensure_dict(obj)
        assert result == {"mock": "data"}

    def test_ensure_dict_with_dict_attr(self) -> None:
        """Test ensure_dict with object having __dict__ attribute."""

        class MockObject:
            def __init__(self) -> None:
                self.attr1 = "value1"
                self.attr2 = "value2"

        obj = MockObject()
        result = ensure_dict(obj)
        assert result == {"attr1": "value1", "attr2": "value2"}

    def test_ensure_dict_fallback(self) -> None:
        """Test ensure_dict fallback behavior."""
        result = ensure_dict("simple")
        assert result == {"value": "simple"}
