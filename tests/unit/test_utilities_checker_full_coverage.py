"""Tests for type checker utility.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
from collections import UserDict
from typing import cast

from flext_core import c, m, r, t, u


class _OnlySelfHandler:
    def handle(self) -> None:
        return None


class _UnknownHintHandler:
    def handle(self, message: MissingType) -> None:
        return None


class MissingType:
    """Class to simulate a missing type."""


class _ExplodingSubclassMeta(type):
    def __subclasscheck__(cls, subclass: type) -> bool:
        _ = subclass
        msg = "no subclass check"
        raise TypeError(msg)


class _ExplodingInstanceMeta(type):
    def __instancecheck__(cls, instance: object) -> bool:
        _ = instance
        msg = "no instance check"
        raise TypeError(msg)


class _ExplodingExpected(metaclass=_ExplodingSubclassMeta):
    pass


class _ExplodingOrigin(metaclass=_ExplodingInstanceMeta):
    pass


def test_checker_logger_and_safe_type_hints_fallback() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"a": 1}), t.ConfigMap)

    checker = u.Checker()
    logger = checker.logger
    assert hasattr(logger, "info")

    hints = u.Checker._get_type_hints_safe(
        cast("t.HandlerCallable", _UnknownHintHandler.handle),
        _UnknownHintHandler,
    )
    assert hints == {"message": MissingType, "return": type(None)}


def test_extract_message_type_from_parameter_branches() -> None:
    param = inspect.Parameter("message", inspect.Parameter.POSITIONAL_OR_KEYWORD)

    assert (
        u.Checker._extract_message_type_from_parameter(
            param, {"message": None}, "message",
        )
        is None
    )
    assert (
        u.Checker._extract_message_type_from_parameter(
            param, {"message": "abc"}, "message",
        )
        == "abc"
    )
    assert u.Checker._extract_message_type_from_parameter(
        param,
        {"message": str(list[int])},
        "message",
    ) == str(list[int])


def test_extract_message_type_from_handle_with_only_self() -> None:
    assert u.Checker._extract_message_type_from_handle(_OnlySelfHandler) is None


def test_object_dict_and_type_error_fallback_paths() -> None:
    # _check_object_type_compatibility uses `is object` identity check
    assert (
        u.Checker._check_object_type_compatibility(
            cast("t.TypeOriginSpecifier", object),
        )
        is True
    )

    # Non-object type returns None
    class _FakeObjectName:
        __name__ = "object"

    fake_object = cast("t.TypeOriginSpecifier", _FakeObjectName())
    assert u.Checker._check_object_type_compatibility(fake_object) is None

    class _DictChild(UserDict[str, str]):
        pass

    dict_match = u.Checker._check_dict_compatibility(
        cast("t.TypeOriginSpecifier", dict),
        _DictChild,
        cast("t.TypeOriginSpecifier", dict),
        cast("t.TypeOriginSpecifier", _DictChild),
    )
    assert dict_match is None

    assert (
        u.Checker._handle_type_or_origin_check(
            cast("t.TypeOriginSpecifier", _ExplodingExpected),
            cast("t.TypeOriginSpecifier", type("Sub", (), {})),
            cast("t.TypeOriginSpecifier", _ExplodingExpected),
            cast("t.TypeOriginSpecifier", object),
        )
        is False
    )

    assert (
        u.Checker._handle_instance_check(
            cast("t.TypeOriginSpecifier", object()),
            cast("t.TypeOriginSpecifier", _ExplodingOrigin),
        )
        is True
    )


def test_extract_message_type_annotation_and_dict_subclass_paths() -> None:
    param_typed = inspect.Parameter(
        "message",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=list[int],
    )
    assert u.Checker._extract_message_type_from_parameter(
        param_typed, {}, "message",
    ) == str(list[int])

    param_empty = inspect.Parameter(
        "message",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=inspect.Signature.empty,
    )
    assert (
        u.Checker._extract_message_type_from_parameter(param_empty, {}, "message")
        is None
    )

    param_str = inspect.Parameter(
        "message",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation="MyType",
    )
    assert (
        u.Checker._extract_message_type_from_parameter(param_str, {}, "message")
        == "MyType"
    )

    param_type = inspect.Parameter(
        "message",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=int,
    )
    assert (
        u.Checker._extract_message_type_from_parameter(param_type, {}, "message") is int
    )

    class _ExpectedDict(UserDict[str, str]):
        pass

    class _MessageDict(UserDict[str, str]):
        pass

    assert (
        u.Checker._check_dict_compatibility(
            cast("t.TypeOriginSpecifier", _ExpectedDict),
            _MessageDict,
            cast("t.TypeOriginSpecifier", _ExpectedDict),
            cast("t.TypeOriginSpecifier", object),
        )
        is None
    )
