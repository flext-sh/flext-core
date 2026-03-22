"""Tests for type checker utility.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
from collections import UserDict
from typing import cast, override

from flext_core import r
from tests import c, m, t, u


class TestUtilitiesCheckerFullCoverage:
    class _OnlySelfHandler:
        def handle(self) -> None:
            return None

    class _MissingType:
        """Class to simulate a missing type."""

    class _UnknownHintHandler:
        def handle(self, message: TestUtilitiesCheckerFullCoverage._MissingType) -> str:
            _ = message
            return "ok"

    class _ExplodingSubclassMeta(type):
        @override
        def __subclasscheck__(cls, subclass: type) -> bool:
            _ = subclass
            msg = "no subclass check"
            raise TypeError(msg)

    class _ExplodingInstanceMeta(type):
        @override
        def __instancecheck__(cls, instance: t.NormalizedValue) -> bool:
            _ = instance
            msg = "no instance check"
            raise TypeError(msg)

    class _ExplodingExpected(metaclass=_ExplodingSubclassMeta):
        pass

    class _ExplodingOrigin(metaclass=_ExplodingInstanceMeta):
        pass

    class _FakeObjectName:
        __name__ = "t.NormalizedValue"

    class _DictChild(UserDict[str, str]):
        pass

    class _ExpectedDict(UserDict[str, str]):
        pass

    class _MessageDict(UserDict[str, str]):
        pass

    def test_checker_logger_and_safe_type_hints_fallback(self) -> None:
        assert c.UNKNOWN_ERROR
        assert isinstance(m.Categories(), m.Categories)
        assert r[int].ok(1).is_success
        assert isinstance(t.ConfigMap({"a": 1}), t.ConfigMap)
        checker = u()
        logger = checker.logger
        assert hasattr(logger, "info")
        hints = u._get_type_hints_safe(
            self._UnknownHintHandler.handle,
            self._UnknownHintHandler,
        )
        assert hints == {"message": self._MissingType, "return": str}

    def test_extract_message_type_from_parameter_branches(self) -> None:
        param = inspect.Parameter("message", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        none_hint = u._extract_message_type_from_parameter(
            param,
            {"message": None},
            "message",
        )
        assert none_hint.is_failure
        str_hint = u._extract_message_type_from_parameter(
            param,
            {"message": "abc"},
            "message",
        )
        assert str_hint.is_success and str_hint.value == "abc"
        generic_hint = u._extract_message_type_from_parameter(
            param,
            {"message": str(list[int])},
            "message",
        )
        assert generic_hint.is_success and generic_hint.value == str(list[int])

    def test_extract_message_type_from_handle_with_only_self(self) -> None:
        assert u._extract_message_type_from_handle(self._OnlySelfHandler).is_failure

    def test_object_dict_and_type_error_fallback_paths(self) -> None:
        assert (
            u._check_object_type_compatibility(
                cast("t.TypeOriginSpecifier", t.NormalizedValue),
            )
            is True
        )
        fake_object = cast("t.TypeOriginSpecifier", self._FakeObjectName())
        assert u._check_object_type_compatibility(fake_object) is False
        dict_match = u._check_dict_compatibility(
            cast("t.TypeOriginSpecifier", dict),
            self._DictChild,
            cast("t.TypeOriginSpecifier", dict),
            cast("t.TypeOriginSpecifier", self._DictChild),
        )
        assert dict_match is False
        assert (
            u._handle_type_or_origin_check(
                cast("t.TypeOriginSpecifier", self._ExplodingExpected),
                cast("t.TypeOriginSpecifier", type("Sub", (), {})),
                cast("t.TypeOriginSpecifier", self._ExplodingExpected),
                cast("t.TypeOriginSpecifier", t.NormalizedValue),
            )
            is False
        )
        assert (
            u._handle_instance_check(
                cast("t.TypeOriginSpecifier", t.NormalizedValue()),
                cast("t.TypeOriginSpecifier", self._ExplodingOrigin),
            )
            is True
        )

    def test_extract_message_type_annotation_and_dict_subclass_paths(self) -> None:
        param_typed = inspect.Parameter(
            "message",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=list[int],
        )
        typed_hint = u._extract_message_type_from_parameter(
            param_typed,
            {},
            "message",
        )
        assert typed_hint.is_success and typed_hint.value == str(list[int])
        param_empty = inspect.Parameter(
            "message",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=inspect.Signature.empty,
        )
        empty_hint = u._extract_message_type_from_parameter(
            param_empty,
            {},
            "message",
        )
        assert empty_hint.is_failure
        param_str = inspect.Parameter(
            "message",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation="MyType",
        )
        string_hint = u._extract_message_type_from_parameter(
            param_str,
            {},
            "message",
        )
        assert string_hint.is_success and string_hint.value == "MyType"
        param_type = inspect.Parameter(
            "message",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=int,
        )
        type_hint = u._extract_message_type_from_parameter(
            param_type,
            {},
            "message",
        )
        assert type_hint.is_success and type_hint.value is int
        assert (
            u._check_dict_compatibility(
                cast("t.TypeOriginSpecifier", self._ExpectedDict),
                self._MessageDict,
                cast("t.TypeOriginSpecifier", self._ExpectedDict),
                cast("t.TypeOriginSpecifier", t.NormalizedValue),
            )
            is False
        )
