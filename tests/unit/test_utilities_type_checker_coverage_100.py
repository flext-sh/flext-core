"""Comprehensive coverage tests for u.

Module: flext_core
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
from collections.abc import MutableMapping
from typing import TypeVar, get_origin, override

import pytest

from flext_core import h
from flext_tests import tm
from tests import p, r, t, u

TMessage = TypeVar("TMessage")


pytestmark = [pytest.mark.unit, pytest.mark.coverage]


class TestsFlextCoreUtilitiesTypeChecker:
    """Comprehensive tests for u."""

    @staticmethod
    def _type_origin(value: t.TypeHintSpecifier) -> t.TypeOriginSpecifier:
        return value

    @staticmethod
    def _message_type(value: t.MessageTypeSpecifier) -> t.MessageTypeSpecifier:
        return value

    class StringHandler(h[str, str]):
        """Handler for string messages."""

        @override
        def handle(self, message: str) -> p.Result[str]:
            return r[str].ok(f"Processed: {message}")

    class IntHandler(h[int, int]):
        """Handler for integer messages."""

        @override
        def handle(self, message: int) -> p.Result[int]:
            return r[int].ok(message * 2)

    class DictHandler(
        h[
            t.MutableRecursiveContainerMapping,
            t.MutableRecursiveContainerMapping,
        ],
    ):
        """Handler for dictionary messages."""

        @override
        def handle(
            self,
            message: t.MutableRecursiveContainerMapping,
        ) -> p.Result[t.MutableRecursiveContainerMapping]:
            result: t.MutableRecursiveContainerMapping = {"processed": True, **message}
            return r[t.MutableRecursiveContainerMapping].ok(result)

    class ObjectHandler(h[t.RecursiveContainer, t.Container]):
        """Handler for t.RecursiveContainer messages."""

        @override
        def handle(self, message: t.RecursiveContainer) -> p.Result[t.Container]:
            if isinstance(message, (str, int, float, bool)):
                return r[t.Container].ok(message)
            return r[t.Container].fail("unsupported message")

    class ExplicitTypeHandler:
        """Handler with explicit type annotation."""

        def handle(self, message: str) -> str:
            return message.upper()

    class NoHandleMethod:
        """Class without handle method."""

        def process(self, data: str) -> str:
            return data

    class NonCallableHandle:
        """Class with non-callable handle attribute."""

        handle: str = "not a method"

    class _BadSignatureCallable:
        """Callable with bad signature attribute."""

        __signature__ = "bad-signature"

        def __call__(self, x: int) -> str:
            return str(x)

    class GenericHandler[TMessage]:
        """Generic handler with type parameter."""

        def handle(self, message: TMessage) -> TMessage:
            return message

    def test_compute_accepted_message_types_from_generic(self) -> None:
        """Test compute_accepted_message_types extracts from generic base."""
        types = u.compute_accepted_message_types(self.StringHandler)
        tm.that(len(types), eq=1)
        assert types[0] is str
        types = u.compute_accepted_message_types(self.IntHandler)
        tm.that(len(types), eq=1)
        assert types[0] is int

    def test_compute_accepted_message_types_from_explicit(self) -> None:
        """Test compute_accepted_message_types extracts from explicit annotation."""
        types = u.compute_accepted_message_types(self.ExplicitTypeHandler)
        tm.that(len(types), eq=1)
        assert types[0] is str

    def test_compute_accepted_message_types_no_handle_method(self) -> None:
        """Test compute_accepted_message_types returns empty for class without handle."""
        types = u.compute_accepted_message_types(self.NoHandleMethod)
        tm.that(len(types), eq=0)

    def test_compute_accepted_message_types_dict_handler(self) -> None:
        """Test compute_accepted_message_types with dict handler."""
        types = u.compute_accepted_message_types(self.DictHandler)
        tm.that(len(types), eq=1)
        origin = get_origin(types[0])
        if origin is None:
            type_str = str(types[0])
            assert (
                type_str.startswith("Mapping[")
                or types[0] is dict
                or "Mapping" in type_str
            )
        else:
            assert origin in {dict, MutableMapping}

    def test_compute_accepted_message_types_object_handler(self) -> None:
        """Test compute_accepted_message_types with t.RecursiveContainer handler (universal)."""
        types = u.compute_accepted_message_types(self.ObjectHandler)
        tm.that(len(types), eq=1)
        assert types[0] == t.RecursiveContainer

    def test_can_handle_message_type_exact_match(self) -> None:
        """Test can_handle_message_type with exact type match."""
        accepted = (str,)
        tm.that(u.can_handle_message_type(accepted, str), eq=True)
        tm.that(not u.can_handle_message_type(accepted, int), eq=True)

    def test_can_handle_message_type_object_accepts_all(self) -> None:
        """Test can_handle_message_type with t.RecursiveContainer type (universal)."""
        accepted: tuple[t.TypeHintSpecifier, ...] = (t.RecursiveContainer,)
        tm.that(u.can_handle_message_type(accepted, str), eq=True)
        tm.that(u.can_handle_message_type(accepted, int), eq=True)
        tm.that(u.can_handle_message_type(accepted, dict), eq=True)

    def test_can_handle_message_type_dict_compatibility(self) -> None:
        """Test can_handle_message_type with dict type compatibility."""
        accepted: tuple[t.MessageTypeSpecifier, ...] = (self._message_type(dict),)
        tm.that(not u.can_handle_message_type(accepted, str), eq=True)
        tm.that(u.can_handle_message_type(accepted, dict), eq=True)
        dict_type: type[t.RecursiveContainerMapping] = dict
        tm.that(u.can_handle_message_type(accepted, dict_type), eq=True)

    def test_can_handle_message_type_empty_accepted(self) -> None:
        """Test can_handle_message_type with empty accepted types."""
        accepted: tuple[t.MessageTypeSpecifier, ...] = ()
        tm.that(not u.can_handle_message_type(accepted, str), eq=True)

    def test_can_handle_message_type_multiple_accepted(self) -> None:
        """Test can_handle_message_type with multiple accepted types."""
        accepted: tuple[t.MessageTypeSpecifier, ...] = (
            self._message_type(str),
            self._message_type(int),
            self._message_type(dict),
        )
        tm.that(u.can_handle_message_type(accepted, str), eq=True)
        tm.that(u.can_handle_message_type(accepted, int), eq=True)
        tm.that(u.can_handle_message_type(accepted, dict), eq=True)
        tm.that(not u.can_handle_message_type(accepted, float), eq=True)

    def test_evaluate_type_compatibility_exact_match(self) -> None:
        """Test _evaluate_type_compatibility with exact type match."""
        tm.that(u._evaluate_type_compatibility(str, str), eq=True)
        tm.that(u._evaluate_type_compatibility(int, int), eq=True)

    def test_evaluate_type_compatibility_object_accepts_all(self) -> None:
        """Test _evaluate_type_compatibility with t.RecursiveContainer type."""
        object_type: t.TypeHintSpecifier = t.RecursiveContainer
        tm.that(u._evaluate_type_compatibility(object_type, str), eq=True)
        tm.that(u._evaluate_type_compatibility(object_type, int), eq=True)
        tm.that(u._evaluate_type_compatibility(object_type, dict), eq=True)

    def test_evaluate_type_compatibility_dict_types(self) -> None:
        """Test _evaluate_type_compatibility with dict types."""
        tm.that(u._evaluate_type_compatibility(self._type_origin(dict), dict), eq=True)
        dict_type: type[t.RecursiveContainerMapping] = dict
        tm.that(
            u._evaluate_type_compatibility(self._type_origin(dict), dict_type),
            eq=True,
        )

    def test_evaluate_type_compatibility_subclass(self) -> None:
        """Test _evaluate_type_compatibility with subclass relationship."""
        tm.that(not u._evaluate_type_compatibility(int, str), eq=True)
        tm.that(not u._evaluate_type_compatibility(str, int), eq=True)

    def test_evaluate_type_compatibility_string_type(self) -> None:
        """Test _evaluate_type_compatibility with string type specifier."""
        result_same = u._evaluate_type_compatibility("str", "str")
        tm.that(result_same, eq=True)
        result_different = u._evaluate_type_compatibility("str", "int")
        tm.that(result_different, is_=bool)

    def test_check_object_type_compatibility_object_type(self) -> None:
        """Test _check_object_type_compatibility with t.RecursiveContainer type."""
        object_type: t.TypeHintSpecifier = t.RecursiveContainer
        result = u._check_object_type_compatibility(object_type)
        tm.that(result, eq=True)

    def test_check_object_type_compatibility_non_object(self) -> None:
        """Test _check_object_type_compatibility with non-t.RecursiveContainer type."""
        result = u._check_object_type_compatibility(str)
        tm.that(not result, eq=True)

    def test_check_dict_compatibility_both_dict(self) -> None:
        """Test _check_dict_compatibility with both types being dict."""
        result = u._check_dict_compatibility(
            self._type_origin(dict),
            dict,
            self._type_origin(dict),
            self._type_origin(dict),
        )
        tm.that(result, eq=True)

    def test_check_dict_compatibility_dict_subclass(self) -> None:
        """Test _check_dict_compatibility with dict subclass."""

        class CustomDict(BaseUserDict[str, t.RecursiveContainer]):
            """Custom dict subclass."""

        result = u._check_dict_compatibility(
            self._type_origin(dict),
            CustomDict,
            self._type_origin(dict),
            self._type_origin(dict),
        )
        tm.that(result, eq=True)

    def test_check_dict_compatibility_non_dict(self) -> None:
        """Test _check_dict_compatibility with non-dict types."""
        result = u._check_dict_compatibility(str, int, str, int)
        tm.that(not result, eq=True)

    def test_extract_generic_message_types_flext_handlers(self) -> None:
        """Test _extract_generic_message_types with h base."""
        types = u._extract_generic_message_types(self.StringHandler)
        tm.that(len(types), eq=1)
        assert types[0] is str
        types = u._extract_generic_message_types(self.IntHandler)
        tm.that(len(types), eq=1)
        assert types[0] is int

    def test_extract_generic_message_types_no_flext_handlers(self) -> None:
        """Test _extract_generic_message_types without h base."""
        types = u._extract_generic_message_types(self.ExplicitTypeHandler)
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
            self.ExplicitTypeHandler,
        )
        tm.ok(message_type_result)
        assert message_type_result.value is str

    def test_extract_message_type_from_handle_no_handle_method(self) -> None:
        """Test _extract_message_type_from_handle without handle method."""
        message_type_result = u._extract_message_type_from_handle(
            self.NoHandleMethod,
        )
        assert message_type_result.failure

    def test_extract_message_type_from_handle_non_callable(self) -> None:
        """Test _extract_message_type_from_handle with non-callable handle."""
        message_type_result = u._extract_message_type_from_handle(
            self.NonCallableHandle,
        )
        assert message_type_result.failure

    def test_get_method_signature_valid_callable(self) -> None:
        """Test _get_method_signature with valid callable."""

        def test_func(x: int) -> str:
            """Test function."""
            return str(x)

        signature = u._get_method_signature(test_func)
        assert signature.success
        signature_value = signature.value
        tm.that(len(signature_value.parameters), eq=1)
        assert "x" in signature_value.parameters

    def test_get_method_signature_non_callable(self) -> None:
        """Test _get_method_signature with non-callable."""
        signature = u._get_method_signature(self._BadSignatureCallable())
        assert signature.failure

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
        assert isinstance(hints, dict)

    def test_handle_type_or_origin_check_with_origin(self) -> None:
        """Test _handle_type_or_origin_check with __origin__ attribute."""
        dict_type = t.MutableStrMapping
        origin = get_origin(dict_type) or dict_type
        result = u._handle_type_or_origin_check(
            self._type_origin(dict),
            self._type_origin(dict_type),
            self._type_origin(dict),
            self._type_origin(origin),
        )
        tm.that(result, is_=bool)

    def test_handle_type_or_origin_check_subclass(self) -> None:
        """Test _handle_type_or_origin_check with subclass relationship."""

        class Base:
            """Base class."""

        class Derived(Base):
            """Derived class."""

        base_type: t.TypeHintSpecifier = Base
        derived_type: t.TypeHintSpecifier = Derived
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
        custom_type_spec: t.TypeHintSpecifier = custom_type
        result = u._handle_instance_check(custom_type_spec, custom_type_spec)
        tm.that(result, is_=bool)

    def test_boundary_empty_tuple_accepted_types(self) -> None:
        """Test boundary case: empty tuple for accepted types."""
        accepted: tuple[t.MessageTypeSpecifier, ...] = ()
        tm.that(not u.can_handle_message_type(accepted, str), eq=True)

    def test_boundary_none_message_type(self) -> None:
        """Test boundary case: None as message type."""
        accepted = (str,)
        result = u.can_handle_message_type(accepted, None)
        tm.that(result, is_=bool)

    def test_boundary_string_type_specifier(self) -> None:
        """Test boundary case: string type specifier."""
        accepted = ("str",)
        result = u.can_handle_message_type(accepted, "str")
        tm.that(result, is_=bool)
