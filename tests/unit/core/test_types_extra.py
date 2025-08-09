"""Tests for FLEXT Core types module."""

from __future__ import annotations

import pytest

from flext_core.typings import FlextTypes

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextTypesTypeGuards:
    """Test FlextTypes.TypeGuards utility class."""

    def test_is_instance_of_with_exception_handling(self) -> None:
        """Test is_instance_of method with exception handling."""
        # Test normal case
        if not (FlextTypes.TypeGuards.is_instance_of("test", str)):
            raise AssertionError(
                f"Expected True, got {FlextTypes.TypeGuards.is_instance_of('test', str)}"
            )
        if FlextTypes.TypeGuards.is_instance_of(123, str):
            raise AssertionError(
                f"Expected False, got {FlextTypes.TypeGuards.is_instance_of(123, str)}"
            )

        # Test exception handling with problematic type
        class ProblematicTypeMeta(type):
            def __instancecheck__(cls, instance: object) -> bool:
                error_message = "Cannot check instance"
                raise TypeError(error_message)

        class ProblematicType(metaclass=ProblematicTypeMeta):
            pass

        # Should return False when TypeError/AttributeError occurs
        if FlextTypes.TypeGuards.is_instance_of("test", ProblematicType):
            raise AssertionError(
                f"Expected False, got {FlextTypes.TypeGuards.is_instance_of('test', ProblematicType)}"
            )

    def test_is_callable(self) -> None:
        """Test is_callable method."""
        # Callable objects
        if not (FlextTypes.TypeGuards.is_callable(lambda x: x)):
            raise AssertionError(
                f"Expected True, got {FlextTypes.TypeGuards.is_callable(lambda x: x)}"
            )
        assert FlextTypes.TypeGuards.is_callable(str) is True
        if not (FlextTypes.TypeGuards.is_callable(print)):
            raise AssertionError(
                f"Expected True, got {FlextTypes.TypeGuards.is_callable(print)}"
            )

        # Non-callable objects
        if FlextTypes.TypeGuards.is_callable("string"):
            raise AssertionError(
                f"Expected False, got {FlextTypes.TypeGuards.is_callable('string')}"
            )
        assert FlextTypes.TypeGuards.is_callable(123) is False
        if FlextTypes.TypeGuards.is_callable([1, 2, 3]):
            raise AssertionError(
                f"Expected False, got {FlextTypes.TypeGuards.is_callable([1, 2, 3])}"
            )

    def test_is_dict_like(self) -> None:
        """Test is_dict_like method."""
        # Dict-like objects
        if not (FlextTypes.TypeGuards.is_dict_like({})):
            raise AssertionError(
                f"Expected True, got {FlextTypes.TypeGuards.is_dict_like({})}"
            )
        assert FlextTypes.TypeGuards.is_dict_like({"key": "value"}) is True

        # Custom dict-like class
        class DictLike:
            def keys(self) -> list[str]:
                return []

            def values(self) -> list[object]:
                return []

            def items(self) -> list[tuple[str, object]]:
                return []

        if not (FlextTypes.TypeGuards.is_dict_like(DictLike())):
            raise AssertionError(
                f"Expected True, got {FlextTypes.TypeGuards.is_dict_like(DictLike())}"
            )

        # Non-dict-like objects
        if FlextTypes.TypeGuards.is_dict_like("string"):
            raise AssertionError(
                f"Expected False, got {FlextTypes.TypeGuards.is_dict_like('string')}"
            )
        assert FlextTypes.TypeGuards.is_dict_like([1, 2, 3]) is False
        if FlextTypes.TypeGuards.is_dict_like(123):
            raise AssertionError(
                f"Expected False, got {FlextTypes.TypeGuards.is_dict_like(123)}"
            )

    def test_is_list_like(self) -> None:
        """Test is_list_like method."""
        # List-like objects
        if not (FlextTypes.TypeGuards.is_list_like([1, 2, 3])):
            raise AssertionError(
                f"Expected True, got {FlextTypes.TypeGuards.is_list_like([1, 2, 3])}"
            )
        assert FlextTypes.TypeGuards.is_list_like((1, 2, 3)) is True
        if not (FlextTypes.TypeGuards.is_list_like({1, 2, 3})):
            raise AssertionError(
                f"Expected True, got {FlextTypes.TypeGuards.is_list_like({1, 2, 3})}"
            )

        # Custom list-like class
        class ListLike:
            def __iter__(self) -> object:
                return iter([1, 2, 3])

            def __len__(self) -> int:
                return 3

        if not (FlextTypes.TypeGuards.is_list_like(ListLike())):
            raise AssertionError(
                f"Expected True, got {FlextTypes.TypeGuards.is_list_like(ListLike())}"
            )

        # Non-list-like objects (strings and bytes are excluded)
        if FlextTypes.TypeGuards.is_list_like("string"):
            raise AssertionError(
                f"Expected False, got {FlextTypes.TypeGuards.is_list_like('string')}"
            )
        assert FlextTypes.TypeGuards.is_list_like(b"bytes") is False
        if FlextTypes.TypeGuards.is_list_like(123):
            raise AssertionError(
                f"Expected False, got {FlextTypes.TypeGuards.is_list_like(123)}"
            )
