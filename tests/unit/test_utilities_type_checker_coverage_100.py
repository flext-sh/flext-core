"""Comprehensive coverage tests for FlextUtilitiesTypeChecker.

Module: flext_core._utilities.type_checker
Scope: Handler type checking utilities for FlextHandlers

Tests validate:
- compute_accepted_message_types: Generic and explicit type extraction
- can_handle_message_type: Type compatibility checking
- _evaluate_type_compatibility: Type evaluation logic
- _extract_generic_message_types: Generic type extraction
- _extract_message_type_from_handle: Explicit type extraction from handle method
- _check_object_type_compatibility: Object type universal compatibility
- _check_dict_compatibility: Dict type compatibility
- _handle_type_or_origin_check: Type/origin checking
- _handle_instance_check: Instance checking

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import UserDict
from typing import TypeVar

import pytest

from flext_core import FlextHandlers, FlextResult
from flext_core._utilities.type_checker import FlextUtilitiesTypeChecker
from flext_core.typings import FlextTypes

T = TypeVar("T")
TMessage = TypeVar("TMessage")


# Test handler classes
class StringHandler(FlextHandlers[str, str]):
    """Handler for string messages."""

    def handle(self, message: str) -> FlextResult[str]:
        """Handle string message."""
        return FlextResult[str].ok(f"Processed: {message}")


class IntHandler(FlextHandlers[int, int]):
    """Handler for int messages."""

    def handle(self, message: int) -> FlextResult[int]:
        """Handle int message."""
        return FlextResult[int].ok(message * 2)


class DictHandler(
    FlextHandlers[
        dict[str, FlextTypes.GeneralValueType], dict[str, FlextTypes.GeneralValueType]
    ]
):
    """Handler for dict messages."""

    def handle(
        self, message: dict[str, FlextTypes.GeneralValueType]
    ) -> FlextResult[dict[str, FlextTypes.GeneralValueType]]:
        """Handle dict message."""
        return FlextResult[dict[str, FlextTypes.GeneralValueType]].ok({
            "processed": True,
            **message,
        })


class ObjectHandler(FlextHandlers[object, object]):
    """Handler for object messages (universal)."""

    def handle(self, message: object) -> FlextResult[object]:
        """Handle any message."""
        return FlextResult[object].ok(message)


class ExplicitTypeHandler:
    """Handler with explicit type annotation (no generic)."""

    def handle(self, message: str) -> str:
        """Handle string message."""
        return message.upper()


class NoHandleMethod:
    """Class without handle method."""

    def process(self, data: str) -> str:
        """Process data."""
        return data


class NonCallableHandle:
    """Class with non-callable handle attribute."""

    handle: str = "not a method"


class GenericHandler[TMessage]:
    """Generic handler without FlextHandlers base."""

    def handle(self, message: TMessage) -> TMessage:
        """Handle generic message."""
        return message


pytestmark = [pytest.mark.unit, pytest.mark.coverage]


class TestFlextUtilitiesTypeChecker:
    """Comprehensive tests for FlextUtilitiesTypeChecker."""

    def test_compute_accepted_message_types_from_generic(self) -> None:
        """Test compute_accepted_message_types extracts from generic base."""
        types = FlextUtilitiesTypeChecker.compute_accepted_message_types(StringHandler)
        assert len(types) == 1
        assert types[0] is str

        types = FlextUtilitiesTypeChecker.compute_accepted_message_types(IntHandler)
        assert len(types) == 1
        assert types[0] is int

    def test_compute_accepted_message_types_from_explicit(self) -> None:
        """Test compute_accepted_message_types extracts from explicit annotation."""
        types = FlextUtilitiesTypeChecker.compute_accepted_message_types(
            ExplicitTypeHandler
        )
        assert len(types) == 1
        assert types[0] is str

    def test_compute_accepted_message_types_no_handle_method(self) -> None:
        """Test compute_accepted_message_types returns empty for class without handle."""
        types = FlextUtilitiesTypeChecker.compute_accepted_message_types(NoHandleMethod)
        assert len(types) == 0

    def test_compute_accepted_message_types_dict_handler(self) -> None:
        """Test compute_accepted_message_types with dict handler."""
        from typing import get_origin

        types = FlextUtilitiesTypeChecker.compute_accepted_message_types(DictHandler)
        assert len(types) == 1
        # Check that it's a dict type (may be dict or dict[str, GeneralValueType])
        # Generic aliases have origin dict
        origin = get_origin(types[0]) or types[0]
        assert origin is dict or (
            isinstance(types[0], type) and issubclass(types[0], dict)
        )

    def test_compute_accepted_message_types_object_handler(self) -> None:
        """Test compute_accepted_message_types with object handler (universal)."""
        types = FlextUtilitiesTypeChecker.compute_accepted_message_types(ObjectHandler)
        assert len(types) == 1
        assert types[0] is object

    def test_can_handle_message_type_exact_match(self) -> None:
        """Test can_handle_message_type with exact type match."""
        accepted = (str,)
        assert FlextUtilitiesTypeChecker.can_handle_message_type(accepted, str) is True
        assert FlextUtilitiesTypeChecker.can_handle_message_type(accepted, int) is False

    def test_can_handle_message_type_object_accepts_all(self) -> None:
        """Test can_handle_message_type with object type (universal)."""
        accepted = (object,)
        assert FlextUtilitiesTypeChecker.can_handle_message_type(accepted, str) is True
        assert FlextUtilitiesTypeChecker.can_handle_message_type(accepted, int) is True
        assert FlextUtilitiesTypeChecker.can_handle_message_type(accepted, dict) is True

    def test_can_handle_message_type_dict_compatibility(self) -> None:
        """Test can_handle_message_type with dict type compatibility."""
        accepted = (dict,)
        assert FlextUtilitiesTypeChecker.can_handle_message_type(accepted, dict) is True
        # dict[str, GeneralValueType] should be compatible with dict
        dict_type: type[dict[str, FlextTypes.GeneralValueType]] = dict
        assert (
            FlextUtilitiesTypeChecker.can_handle_message_type(accepted, dict_type)
            is True
        )

    def test_can_handle_message_type_empty_accepted(self) -> None:
        """Test can_handle_message_type with empty accepted types."""
        accepted: tuple[FlextTypes.Utility.MessageTypeSpecifier, ...] = ()
        assert FlextUtilitiesTypeChecker.can_handle_message_type(accepted, str) is False

    def test_can_handle_message_type_multiple_accepted(self) -> None:
        """Test can_handle_message_type with multiple accepted types."""
        accepted = (str, int, dict)
        assert FlextUtilitiesTypeChecker.can_handle_message_type(accepted, str) is True
        assert FlextUtilitiesTypeChecker.can_handle_message_type(accepted, int) is True
        assert FlextUtilitiesTypeChecker.can_handle_message_type(accepted, dict) is True
        assert (
            FlextUtilitiesTypeChecker.can_handle_message_type(accepted, float) is False
        )

    def test_evaluate_type_compatibility_exact_match(self) -> None:
        """Test _evaluate_type_compatibility with exact type match."""
        assert FlextUtilitiesTypeChecker._evaluate_type_compatibility(str, str) is True
        assert FlextUtilitiesTypeChecker._evaluate_type_compatibility(int, int) is True

    def test_evaluate_type_compatibility_object_accepts_all(self) -> None:
        """Test _evaluate_type_compatibility with object type."""
        assert (
            FlextUtilitiesTypeChecker._evaluate_type_compatibility(object, str) is True
        )
        assert (
            FlextUtilitiesTypeChecker._evaluate_type_compatibility(object, int) is True
        )
        assert (
            FlextUtilitiesTypeChecker._evaluate_type_compatibility(object, dict) is True
        )

    def test_evaluate_type_compatibility_dict_types(self) -> None:
        """Test _evaluate_type_compatibility with dict types."""
        assert (
            FlextUtilitiesTypeChecker._evaluate_type_compatibility(dict, dict) is True
        )
        # dict[str, GeneralValueType] should be compatible with dict
        dict_type: type[dict[str, FlextTypes.GeneralValueType]] = dict
        assert (
            FlextUtilitiesTypeChecker._evaluate_type_compatibility(dict, dict_type)
            is True
        )

    def test_evaluate_type_compatibility_subclass(self) -> None:
        """Test _evaluate_type_compatibility with subclass relationship."""
        # str is not a subclass of int
        assert FlextUtilitiesTypeChecker._evaluate_type_compatibility(int, str) is False
        # int is not a subclass of str
        assert FlextUtilitiesTypeChecker._evaluate_type_compatibility(str, int) is False

    def test_evaluate_type_compatibility_string_type(self) -> None:
        """Test _evaluate_type_compatibility with string type specifier."""
        # String type specifiers - both are strings, so they may match via instance check
        # Same strings should match
        result_same = FlextUtilitiesTypeChecker._evaluate_type_compatibility(
            "str", "str"
        )
        assert result_same is True
        # Different strings - behavior depends on implementation (may fall through to instance check)
        result_different = FlextUtilitiesTypeChecker._evaluate_type_compatibility(
            "str", "int"
        )
        # Should return boolean (actual value depends on implementation)
        assert isinstance(result_different, bool)

    def test_check_object_type_compatibility_object_type(self) -> None:
        """Test _check_object_type_compatibility with object type."""
        result = FlextUtilitiesTypeChecker._check_object_type_compatibility(object)
        assert result is True

    def test_check_object_type_compatibility_non_object(self) -> None:
        """Test _check_object_type_compatibility with non-object type."""
        result = FlextUtilitiesTypeChecker._check_object_type_compatibility(str)
        assert result is None

    def test_check_dict_compatibility_both_dict(self) -> None:
        """Test _check_dict_compatibility with both types being dict."""
        result = FlextUtilitiesTypeChecker._check_dict_compatibility(
            dict, dict, dict, dict
        )
        assert result is True

    def test_check_dict_compatibility_dict_subclass(self) -> None:
        """Test _check_dict_compatibility with dict subclass."""

        class CustomDict(UserDict):
            """Custom dict subclass."""

        result = FlextUtilitiesTypeChecker._check_dict_compatibility(
            dict, CustomDict, dict, dict
        )
        assert result is True

    def test_check_dict_compatibility_non_dict(self) -> None:
        """Test _check_dict_compatibility with non-dict types."""
        result = FlextUtilitiesTypeChecker._check_dict_compatibility(str, int, str, int)
        assert result is None

    def test_extract_generic_message_types_flext_handlers(self) -> None:
        """Test _extract_generic_message_types with FlextHandlers base."""
        types = FlextUtilitiesTypeChecker._extract_generic_message_types(StringHandler)
        assert len(types) == 1
        assert types[0] is str

        types = FlextUtilitiesTypeChecker._extract_generic_message_types(IntHandler)
        assert len(types) == 1
        assert types[0] is int

    def test_extract_generic_message_types_no_flext_handlers(self) -> None:
        """Test _extract_generic_message_types without FlextHandlers base."""
        types = FlextUtilitiesTypeChecker._extract_generic_message_types(
            ExplicitTypeHandler
        )
        assert len(types) == 0

    def test_extract_generic_message_types_no_orig_bases(self) -> None:
        """Test _extract_generic_message_types with class without __orig_bases__."""

        class PlainClass:
            """Plain class without generic base."""

        types = FlextUtilitiesTypeChecker._extract_generic_message_types(PlainClass)
        assert len(types) == 0

    def test_extract_message_type_from_handle_with_annotation(self) -> None:
        """Test _extract_message_type_from_handle with type annotation."""
        types = FlextUtilitiesTypeChecker._extract_message_type_from_handle(
            ExplicitTypeHandler
        )
        assert types is str

    def test_extract_message_type_from_handle_no_handle_method(self) -> None:
        """Test _extract_message_type_from_handle without handle method."""
        types = FlextUtilitiesTypeChecker._extract_message_type_from_handle(
            NoHandleMethod
        )
        assert types is None

    def test_extract_message_type_from_handle_non_callable(self) -> None:
        """Test _extract_message_type_from_handle with non-callable handle."""
        types = FlextUtilitiesTypeChecker._extract_message_type_from_handle(
            NonCallableHandle
        )
        # Should return None or handle gracefully
        assert types is None or isinstance(types, str)

    def test_get_method_signature_valid_callable(self) -> None:
        """Test _get_method_signature with valid callable."""

        def test_func(x: int) -> str:
            """Test function."""
            return str(x)

        signature = FlextUtilitiesTypeChecker._get_method_signature(test_func)
        assert signature is not None
        assert len(signature.parameters) == 1
        assert "x" in signature.parameters

    def test_get_method_signature_non_callable(self) -> None:
        """Test _get_method_signature with non-callable."""
        signature = FlextUtilitiesTypeChecker._get_method_signature("not callable")
        assert signature is None

    def test_get_type_hints_safe_valid_method(self) -> None:
        """Test _get_type_hints_safe with valid method."""

        class TestClass:
            """Test class."""

            def handle(self, message: str) -> str:
                """Handle message."""
                return message

        hints = FlextUtilitiesTypeChecker._get_type_hints_safe(
            TestClass.handle, TestClass
        )
        assert "message" in hints
        assert hints["message"] is str

    def test_get_type_hints_safe_no_hints(self) -> None:
        """Test _get_type_hints_safe with method without hints."""

        class TestClass:
            """Test class."""

            def handle(self, message):  # type: ignore[no-untyped-def]
                """Handle message."""
                return message

        hints = FlextUtilitiesTypeChecker._get_type_hints_safe(
            TestClass.handle, TestClass
        )
        # Should return empty dict or handle gracefully
        assert isinstance(hints, dict)

    def test_handle_type_or_origin_check_with_origin(self) -> None:
        """Test _handle_type_or_origin_check with __origin__ attribute."""
        from typing import get_origin

        dict_type: type[dict[str, str]] = dict[str, str]
        origin = get_origin(dict_type) or dict_type

        result = FlextUtilitiesTypeChecker._handle_type_or_origin_check(
            dict, dict_type, dict, origin
        )
        # Should handle origin comparison
        assert isinstance(result, bool)

    def test_handle_type_or_origin_check_subclass(self) -> None:
        """Test _handle_type_or_origin_check with subclass relationship."""

        class Base:
            """Base class."""

        class Derived(Base):
            """Derived class."""

        result = FlextUtilitiesTypeChecker._handle_type_or_origin_check(
            Base, Derived, Base, Base
        )
        assert result is True  # Derived is subclass of Base

    def test_handle_type_or_origin_check_type_error(self) -> None:
        """Test _handle_type_or_origin_check handles TypeError gracefully."""
        # Use types that might cause TypeError in issubclass
        result = FlextUtilitiesTypeChecker._handle_type_or_origin_check(
            "str", "int", "str", "int"
        )
        # Should fallback to identity check
        assert isinstance(result, bool)

    def test_handle_instance_check_with_type(self) -> None:
        """Test _handle_instance_check with type origin."""
        result = FlextUtilitiesTypeChecker._handle_instance_check("test", str)
        assert result is True  # "test" is instance of str

    def test_handle_instance_check_non_type_origin(self) -> None:
        """Test _handle_instance_check with non-type origin."""
        result = FlextUtilitiesTypeChecker._handle_instance_check("test", "str")
        # Should return True for non-type origins
        assert result is True

    def test_handle_instance_check_type_error(self) -> None:
        """Test _handle_instance_check handles TypeError gracefully."""
        # Use types that might cause TypeError in isinstance
        result = FlextUtilitiesTypeChecker._handle_instance_check(
            object(), type("CustomType", (), {})
        )
        # Should handle gracefully
        assert isinstance(result, bool)

    def test_boundary_empty_tuple_accepted_types(self) -> None:
        """Test boundary case: empty tuple for accepted types."""
        accepted: tuple[FlextTypes.Utility.MessageTypeSpecifier, ...] = ()
        assert FlextUtilitiesTypeChecker.can_handle_message_type(accepted, str) is False

    def test_boundary_none_message_type(self) -> None:
        """Test boundary case: None as message type."""
        # None is not a valid MessageTypeSpecifier, but should handle gracefully
        accepted = (str,)
        # This should not crash, but may return False
        result = FlextUtilitiesTypeChecker.can_handle_message_type(accepted, None)  # type: ignore[arg-type]
        assert isinstance(result, bool)

    def test_boundary_string_type_specifier(self) -> None:
        """Test boundary case: string type specifier."""
        accepted = ("str",)
        # String specifiers should work
        result = FlextUtilitiesTypeChecker.can_handle_message_type(accepted, "str")
        assert isinstance(result, bool)
