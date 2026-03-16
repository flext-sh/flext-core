"""Comprehensive coverage tests for u.

Module: flext_core._utilities.checker
Scope: Handler type checking utilities for h

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

from collections import UserDict as BaseUserDict
from typing import TypeVar, cast, get_origin, override

import pytest
from flext_tests import t, tm

from flext_core import h, r, u

T = TypeVar("T")
TMessage = TypeVar("TMessage")


def _type_origin(value: t.TypeHintSpecifier) -> t.TypeOriginSpecifier:
    return value


def _message_type(
    value: type[dict[str, t.Tests.object] | int | str],
) -> t.MessageTypeSpecifier:
    return cast("t.MessageTypeSpecifier", value)


class StringHandler(h[str, str]):
    """Handler for string messages."""

    @override
    def handle(self, message: str) -> r[str]:
        """Handle string message."""
        return r[str].ok(f"Processed: {message}")


class IntHandler(h[int, int]):
    """Handler for int messages."""

    @override
    def handle(self, message: int) -> r[int]:
        """Handle int message."""
        return r[int].ok(message * 2)


class DictHandler(h[dict[str, t.Tests.object], dict[str, t.Tests.object]]):
    """Handler for dict messages."""

    @override
    def handle(
        self,
        message: dict[str, t.Tests.object],
    ) -> r[dict[str, t.Tests.object]]:
        """Handle dict message."""
        return r[dict[str, t.Tests.object]].ok({
            "processed": True,
            **message,
        })


class ObjectHandler(h[object, t.Container]):
    """Handler for object messages (universal)."""

    @override
    def handle(self, message: object) -> r[t.Container]:
        """Handle any message."""
        if isinstance(message, (str, int, float, bool)):
            return r[t.Container].ok(message)
        return r[t.Container].fail("unsupported message")


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


class _BadSignatureCallable:
    __signature__ = "bad-signature"

    def __call__(self, x: int) -> str:
        return str(x)


class GenericHandler[TMessage]:
    """Generic handler without h base."""

    def handle(self, message: TMessage) -> TMessage:
        """Handle generic message."""
        return message


pytestmark = [pytest.mark.unit, pytest.mark.coverage]


class TestuTypeChecker:
    """Comprehensive tests for u."""

    def test_compute_accepted_message_types_from_generic(self) -> None:
        """Test compute_accepted_message_types extracts from generic base."""
        types = u.compute_accepted_message_types(StringHandler)
        tm.that(len(types), eq=1)
        tm.that(types[0], eq=str)
        types = u.compute_accepted_message_types(IntHandler)
        tm.that(len(types), eq=1)
        tm.that(types[0], eq=int)

    def test_compute_accepted_message_types_from_explicit(self) -> None:
        """Test compute_accepted_message_types extracts from explicit annotation."""
        types = u.compute_accepted_message_types(ExplicitTypeHandler)
        tm.that(len(types), eq=1)
        tm.that(types[0], eq=str)

    def test_compute_accepted_message_types_no_handle_method(self) -> None:
        """Test compute_accepted_message_types returns empty for class without handle."""
        types = u.compute_accepted_message_types(NoHandleMethod)
        tm.that(len(types), eq=0)

    def test_compute_accepted_message_types_dict_handler(self) -> None:
        """Test compute_accepted_message_types with dict handler."""
        types = u.compute_accepted_message_types(DictHandler)
        tm.that(len(types), eq=1)
        origin = get_origin(types[0])
        if origin is None:
            type_str = str(types[0])
            assert type_str.startswith("dict[") or types[0] is dict
        else:
            tm.that(origin, eq=dict)

    def test_compute_accepted_message_types_object_handler(self) -> None:
        """Test compute_accepted_message_types with object handler (universal)."""
        types = u.compute_accepted_message_types(ObjectHandler)
        tm.that(len(types), eq=1)
        tm.that(types[0], eq=object)

    def test_can_handle_message_type_exact_match(self) -> None:
        """Test can_handle_message_type with exact type match."""
        accepted = (str,)
        tm.that(u.can_handle_message_type(accepted, str), eq=True)
        tm.that(u.can_handle_message_type(accepted, int), eq=False)

    def test_can_handle_message_type_object_accepts_all(self) -> None:
        """Test can_handle_message_type with object type (universal)."""
        accepted: tuple[t.MessageTypeSpecifier, ...] = (
            cast("t.MessageTypeSpecifier", object),
        )
        tm.that(u.can_handle_message_type(accepted, str), eq=True)
        tm.that(u.can_handle_message_type(accepted, int), eq=True)
        tm.that(u.can_handle_message_type(accepted, dict), eq=True)

    def test_can_handle_message_type_dict_compatibility(self) -> None:
        """Test can_handle_message_type with dict type compatibility."""
        accepted: tuple[t.MessageTypeSpecifier, ...] = (_message_type(dict),)
        tm.that(u.can_handle_message_type(accepted, str), eq=False)
        tm.that(u.can_handle_message_type(accepted, dict), eq=True)
        dict_type: type[dict[str, t.Tests.object]] = dict
        tm.that(u.can_handle_message_type(accepted, dict_type), eq=True)

    def test_can_handle_message_type_empty_accepted(self) -> None:
        """Test can_handle_message_type with empty accepted types."""
        accepted: tuple[t.MessageTypeSpecifier, ...] = ()
        tm.that(u.can_handle_message_type(accepted, str), eq=False)

    def test_can_handle_message_type_multiple_accepted(self) -> None:
        """Test can_handle_message_type with multiple accepted types."""
        accepted: tuple[t.MessageTypeSpecifier, ...] = (
            _message_type(str),
            _message_type(int),
            _message_type(dict),
        )
        tm.that(u.can_handle_message_type(accepted, str), eq=True)
        tm.that(u.can_handle_message_type(accepted, int), eq=True)
        tm.that(u.can_handle_message_type(accepted, dict), eq=True)
        tm.that(u.can_handle_message_type(accepted, float), eq=False)

    def test_evaluate_type_compatibility_exact_match(self) -> None:
        """Test _evaluate_type_compatibility with exact type match."""
        tm.that(u._evaluate_type_compatibility(str, str), eq=True)
        tm.that(u._evaluate_type_compatibility(int, int), eq=True)

    def test_evaluate_type_compatibility_object_accepts_all(self) -> None:
        """Test _evaluate_type_compatibility with object type."""
        object_type: t.TypeHintSpecifier = cast("t.TypeOriginSpecifier", object)
        tm.that(u._evaluate_type_compatibility(object_type, str), eq=True)
        tm.that(u._evaluate_type_compatibility(object_type, int), eq=True)
        tm.that(u._evaluate_type_compatibility(object_type, dict), eq=True)

    def test_evaluate_type_compatibility_dict_types(self) -> None:
        """Test _evaluate_type_compatibility with dict types."""
        tm.that(u._evaluate_type_compatibility(_type_origin(dict), dict), eq=True)
        dict_type: type[dict[str, t.Tests.object]] = dict
        tm.that(u._evaluate_type_compatibility(_type_origin(dict), dict_type), eq=True)

    def test_evaluate_type_compatibility_subclass(self) -> None:
        """Test _evaluate_type_compatibility with subclass relationship."""
        tm.that(u._evaluate_type_compatibility(int, str), eq=False)
        tm.that(u._evaluate_type_compatibility(str, int), eq=False)

    def test_evaluate_type_compatibility_string_type(self) -> None:
        """Test _evaluate_type_compatibility with string type specifier."""
        result_same = u._evaluate_type_compatibility("str", "str")
        tm.that(result_same, eq=True)
        result_different = u._evaluate_type_compatibility("str", "int")
        tm.that(result_different, is_=bool)

    def test_check_object_type_compatibility_object_type(self) -> None:
        """Test _check_object_type_compatibility with object type."""
        object_type: t.TypeHintSpecifier = cast("t.TypeOriginSpecifier", object)
        result = u._check_object_type_compatibility(object_type)
        tm.that(result, eq=True)

    def test_check_object_type_compatibility_non_object(self) -> None:
        """Test _check_object_type_compatibility with non-object type."""
        result = u._check_object_type_compatibility(str)
        tm.that(result, eq=False)

    def test_check_dict_compatibility_both_dict(self) -> None:
        """Test _check_dict_compatibility with both types being dict."""
        result = u._check_dict_compatibility(
            _type_origin(dict),
            dict,
            _type_origin(dict),
            _type_origin(dict),
        )
        tm.that(result, eq=True)

    def test_check_dict_compatibility_dict_subclass(self) -> None:
        """Test _check_dict_compatibility with dict subclass."""

        class CustomDict(BaseUserDict[str, object]):
            """Custom dict subclass."""

        result = u._check_dict_compatibility(
            _type_origin(dict),
            CustomDict,
            _type_origin(dict),
            _type_origin(dict),
        )
        tm.that(result, eq=True)

    def test_check_dict_compatibility_non_dict(self) -> None:
        """Test _check_dict_compatibility with non-dict types."""
        result = u._check_dict_compatibility(str, int, str, int)
        tm.that(result, eq=False)

    def test_extract_generic_message_types_flext_handlers(self) -> None:
        """Test _extract_generic_message_types with h base."""
        types = u._extract_generic_message_types(StringHandler)
        tm.that(len(types), eq=1)
        tm.that(types[0], eq=str)
        types = u._extract_generic_message_types(IntHandler)
        tm.that(len(types), eq=1)
        tm.that(types[0], eq=int)

    def test_extract_generic_message_types_no_flext_handlers(self) -> None:
        """Test _extract_generic_message_types without h base."""
        types = u._extract_generic_message_types(ExplicitTypeHandler)
        tm.that(len(types), eq=0)

    def test_extract_generic_message_types_no_orig_bases(self) -> None:
        """Test _extract_generic_message_types with class without __orig_bases__."""

        class PlainClass:
            """Plain class without generic base."""

        types = u._extract_generic_message_types(PlainClass)
        tm.that(len(types), eq=0)

    def test_extract_message_type_from_handle_with_annotation(self) -> None:
        """Test _extract_message_type_from_handle with type annotation."""
        message_type_result = u._extract_message_type_from_handle(
            ExplicitTypeHandler,
        )
        tm.ok(message_type_result)
        tm.that(message_type_result.value, eq=str)

    def test_extract_message_type_from_handle_no_handle_method(self) -> None:
        """Test _extract_message_type_from_handle without handle method."""
        message_type_result = u._extract_message_type_from_handle(
            NoHandleMethod,
        )
        assert message_type_result.is_failure

    def test_extract_message_type_from_handle_non_callable(self) -> None:
        """Test _extract_message_type_from_handle with non-callable handle."""
        message_type_result = u._extract_message_type_from_handle(
            NonCallableHandle,
        )
        assert message_type_result.is_failure

    def test_get_method_signature_valid_callable(self) -> None:
        """Test _get_method_signature with valid callable."""

        def test_func(x: int) -> str:
            """Test function."""
            return str(x)

        signature = u._get_method_signature(test_func)
        tm.ok(signature)
        signature_value = signature.value
        tm.that(len(signature_value.parameters), eq=1)
        assert "x" in signature_value.parameters

    def test_get_method_signature_non_callable(self) -> None:
        """Test _get_method_signature with non-callable."""
        signature = u._get_method_signature(_BadSignatureCallable())
        assert signature.is_failure

    def test_get_type_hints_safe_valid_method(self) -> None:
        """Test _get_type_hints_safe with valid method."""

        class TestClass:
            """Test class."""

            def handle(self, message: str) -> str:
                """Handle message."""
                return message

        hints = u._get_type_hints_safe(TestClass.handle, TestClass)
        assert "message" in hints
        message_type = hints.get("message")
        assert message_type is not None
        assert isinstance(message_type, type) and message_type.__name__ == "str"

    def test_get_type_hints_safe_no_hints(self) -> None:
        """Test _get_type_hints_safe with method without hints."""

        class TestClass:
            """Test class."""

            def handle(self, message: str) -> str:
                """Handle message."""
                return message

        hints = u._get_type_hints_safe(TestClass.handle, TestClass)
        tm.that(hints, is_=dict)

    def test_handle_type_or_origin_check_with_origin(self) -> None:
        """Test _handle_type_or_origin_check with __origin__ attribute."""
        dict_type: type[dict[str, str]] = dict[str, str]
        origin = get_origin(dict_type) or dict_type
        result = u._handle_type_or_origin_check(
            _type_origin(dict),
            _type_origin(dict_type),
            _type_origin(dict),
            _type_origin(origin),
        )
        tm.that(result, is_=bool)

    def test_handle_type_or_origin_check_subclass(self) -> None:
        """Test _handle_type_or_origin_check with subclass relationship."""

        class Base:
            """Base class."""

        class Derived(Base):
            """Derived class."""

        base_type: t.TypeHintSpecifier = cast("t.TypeOriginSpecifier", Base)
        derived_type: t.TypeHintSpecifier = cast("t.TypeOriginSpecifier", Derived)
        result = u._handle_type_or_origin_check(
            base_type,
            derived_type,
            base_type,
            base_type,
        )
        tm.that(result, eq=True)

    def test_handle_type_or_origin_check_type_error(self) -> None:
        """Test _handle_type_or_origin_check handles TypeError gracefully."""
        result = u._handle_type_or_origin_check("str", "int", "str", "int")
        tm.that(result, is_=bool)

    def test_handle_instance_check_with_type(self) -> None:
        """Test _handle_instance_check with type origin."""
        result = u._handle_instance_check("test", str)
        tm.that(result, eq=True)

    def test_handle_instance_check_non_type_origin(self) -> None:
        """Test _handle_instance_check with non-type origin."""
        result = u._handle_instance_check("test", "str")
        tm.that(result, eq=True)

    def test_handle_instance_check_type_error(self) -> None:
        """Test _handle_instance_check handles TypeError gracefully."""
        custom_type = type("CustomType", (), {})
        custom_type_spec: t.TypeHintSpecifier = cast(
            "t.TypeOriginSpecifier",
            custom_type,
        )
        result = u._handle_instance_check(custom_type_spec, custom_type_spec)
        tm.that(result, is_=bool)

    def test_boundary_empty_tuple_accepted_types(self) -> None:
        """Test boundary case: empty tuple for accepted types."""
        accepted: tuple[t.MessageTypeSpecifier, ...] = ()
        tm.that(u.can_handle_message_type(accepted, str), eq=False)

    def test_boundary_none_message_type(self) -> None:
        """Test boundary case: None as message type."""
        accepted = (str,)
        result = u.can_handle_message_type(
            accepted,
            cast("str | type", None),
        )
        tm.that(result, is_=bool)

    def test_boundary_string_type_specifier(self) -> None:
        """Test boundary case: string type specifier."""
        accepted = ("str",)
        result = u.can_handle_message_type(accepted, "str")
        tm.that(result, is_=bool)
