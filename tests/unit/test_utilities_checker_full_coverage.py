from __future__ import annotations

import inspect
from typing import Any

from flext_core import c, m, r, t, u


class _OnlySelfHandler:
    def handle(self) -> None:
        return None


class _UnknownHintHandler:
    def handle(self, message: "MissingType") -> None:
        return None


class _ExplodingSubclassMeta(type):
    def __subclasscheck__(cls, subclass: Any) -> bool:
        _ = subclass
        raise TypeError("no subclass check")


class _ExplodingInstanceMeta(type):
    def __instancecheck__(cls, instance: Any) -> bool:
        _ = instance
        raise TypeError("no instance check")


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
        _UnknownHintHandler.handle, _UnknownHintHandler
    )
    assert hints == {}


def test_extract_message_type_from_parameter_branches() -> None:
    param = inspect.Parameter("message", inspect.Parameter.POSITIONAL_OR_KEYWORD)

    assert (
        u.Checker._extract_message_type_from_parameter(
            param, {"message": None}, "message"
        )
        is None
    )
    assert (
        u.Checker._extract_message_type_from_parameter(
            param, {"message": "abc"}, "message"
        )
        == "abc"
    )
    assert u.Checker._extract_message_type_from_parameter(
        param, {"message": list[int]}, "message"
    ) == str(list[int])


def test_extract_message_type_from_handle_with_only_self() -> None:
    assert u.Checker._extract_message_type_from_handle(_OnlySelfHandler) is None


def test_object_dict_and_type_error_fallback_paths() -> None:
    class _FakeObjectName:
        __name__ = "object"

    fake_object = _FakeObjectName()
    assert u.Checker._check_object_type_compatibility(fake_object) is True

    class _DictChild(dict):
        pass

    dict_match = u.Checker._check_dict_compatibility(dict, _DictChild, dict, _DictChild)
    assert dict_match is True

    assert (
        u.Checker._handle_type_or_origin_check(
            _ExplodingExpected,
            type("Sub", (), {}),
            _ExplodingExpected,
            object,
        )
        is False
    )

    assert u.Checker._handle_instance_check(object(), _ExplodingOrigin) is True


def test_extract_message_type_annotation_and_dict_subclass_paths() -> None:
    param_typed = inspect.Parameter(
        "message",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=list[int],
    )
    assert u.Checker._extract_message_type_from_parameter(
        param_typed, {}, "message"
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

    class _ExpectedDict(dict):
        pass

    class _MessageDict(dict):
        pass

    assert (
        u.Checker._check_dict_compatibility(
            _ExpectedDict,
            _MessageDict,
            _ExpectedDict,
            object,
        )
        is True
    )
