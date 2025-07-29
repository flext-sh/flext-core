"""Tests for FLEXT Core types module."""

from __future__ import annotations

import pytest

from flext_core.types import FlextTypes

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextTypesTypeGuards:
    """Test FlextTypes.TypeGuards utility class."""

    def test_is_instance_of_with_exception_handling(self) -> None:
        """Test is_instance_of method with exception handling."""
        # Test normal case
        assert FlextTypes.TypeGuards.is_instance_of("test", str) is True
        assert FlextTypes.TypeGuards.is_instance_of(123, str) is False

        # Test exception handling with problematic type
        class ProblematicType:
            def __instancecheck__(self, instance: object) -> bool:
                error_message = "Cannot check instance"
                raise TypeError(error_message)

        # Should return False when TypeError/AttributeError occurs
        problematic = ProblematicType()
        assert FlextTypes.TypeGuards.is_instance_of("test", problematic) is False

    def test_is_callable(self) -> None:
        """Test is_callable method."""
        # Callable objects
        assert FlextTypes.TypeGuards.is_callable(lambda x: x) is True
        assert FlextTypes.TypeGuards.is_callable(str) is True
        assert FlextTypes.TypeGuards.is_callable(print) is True

        # Non-callable objects
        assert FlextTypes.TypeGuards.is_callable("string") is False
        assert FlextTypes.TypeGuards.is_callable(123) is False
        assert FlextTypes.TypeGuards.is_callable([1, 2, 3]) is False

    def test_is_dict_like(self) -> None:
        """Test is_dict_like method."""
        # Dict-like objects
        assert FlextTypes.TypeGuards.is_dict_like({}) is True
        assert FlextTypes.TypeGuards.is_dict_like({"key": "value"}) is True

        # Custom dict-like class
        class DictLike:
            def keys(self) -> list[str]:
                return []

            def values(self) -> list[object]:
                return []

            def items(self) -> list[tuple[str, object]]:
                return []

        assert FlextTypes.TypeGuards.is_dict_like(DictLike()) is True

        # Non-dict-like objects
        assert FlextTypes.TypeGuards.is_dict_like("string") is False
        assert FlextTypes.TypeGuards.is_dict_like([1, 2, 3]) is False
        assert FlextTypes.TypeGuards.is_dict_like(123) is False

    def test_is_list_like(self) -> None:
        """Test is_list_like method."""
        # List-like objects
        assert FlextTypes.TypeGuards.is_list_like([1, 2, 3]) is True
        assert FlextTypes.TypeGuards.is_list_like((1, 2, 3)) is True
        assert FlextTypes.TypeGuards.is_list_like({1, 2, 3}) is True

        # Custom list-like class
        class ListLike:
            def __iter__(self) -> object:
                return iter([1, 2, 3])

            def __len__(self) -> int:
                return 3

        assert FlextTypes.TypeGuards.is_list_like(ListLike()) is True

        # Non-list-like objects (strings and bytes are excluded)
        assert FlextTypes.TypeGuards.is_list_like("string") is False
        assert FlextTypes.TypeGuards.is_list_like(b"bytes") is False
        assert FlextTypes.TypeGuards.is_list_like(123) is False
