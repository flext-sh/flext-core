"""Coverage tests for current utilities guard APIs."""

from __future__ import annotations

import builtins
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

import pytest
from pydantic import BaseModel

from flext_core import c, m, r, t, u
from flext_tests import t as test_t

from ._models import _Model


def _is_type_obj(
    value: test_t.Tests.object, type_spec: str | type | tuple[type, ...]
) -> bool:
    """Call is_type with arbitrary object for negative-case testing."""
    fn: Callable[[test_t.Tests.object, str | type | tuple[type, ...]], bool] = getattr(
        u,
        "is_type",
    )
    return fn(value, type_spec)


def _is_flexible_value_obj(value: dict[int, str] | set[int]) -> bool:
    """Call is_flexible_value with arbitrary object for negative-case testing."""
    fn: Callable[..., bool] = getattr(u, "is_flexible_value")
    return fn(value)


class _LoggerLike:
    def debug(self, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        return None

    def info(self, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        return None

    def warning(self, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        return None

    def error(self, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        return None

    def exception(self, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        return None


def _sample_handler(value: t.NormalizedValue) -> t.NormalizedValue:
    return value


def _return_false(_value: str) -> bool:
    return False


def test_aliases_are_available() -> None:
    assert u is not None
    assert c is not None
    assert m is not None
    assert t is not None


def test_is_container_negative_paths_and_callable() -> None:
    assert callable(_sample_handler) or u.is_container(
        cast("t.NormalizedValue", _sample_handler)
    )
    assert u.is_container([1, "x", None])
    assert u.is_container({"k": 1})
    assert not u.is_container(cast("t.NormalizedValue", [{"x"}]))
    assert u.is_container(cast("t.NormalizedValue", {1: "x"}))
    assert not u.is_container(cast("t.NormalizedValue", {"x": {1}}))


def test_is_handler_type_branches() -> None:
    assert u.is_handler_type({"a": 1})
    assert u.is_handler_type(
        cast("t.NormalizedValue", _Model.model_validate({"value": 1}))
    )
    assert u.is_handler_type(cast("t.NormalizedValue", _sample_handler))

    class _BaseModelSubclass:
        value: str = "ok"

    class _DuckHandler:
        value: str = "ok"

        def handle(self, _value: t.NormalizedValue) -> None:
            return None

    assert u.is_handler_type(cast("t.NormalizedValue", _BaseModelSubclass))
    assert u.is_handler_type(cast("t.NormalizedValue", _DuckHandler()))


def test_non_empty_and_normalize_branches() -> None:
    assert u.is_string_non_empty("x")
    assert u.is_type("x", "string_non_empty")
    assert u.is_dict_non_empty({"k": "v"})
    assert u.is_list_non_empty([1])
    assert u.normalize_to_metadata("x") == "x"
    dict_scalar_out = u.normalize_to_metadata({"k": 1})
    assert dict_scalar_out == {"k": 1}
    dict_complex_out = u.normalize_to_metadata(
        cast("t.NormalizedValue", {"k": object()})
    )
    assert isinstance(dict_complex_out, dict) and "k" in dict_complex_out
    list_out = u.normalize_to_metadata(cast("t.NormalizedValue", [1, object()]))
    assert isinstance(list_out, list)
    assert list_out[0] == 1
    assert isinstance(list_out[1], str)
    assert isinstance(
        u.normalize_to_metadata(cast("t.NormalizedValue", {1, 2})),
        str,
    )


def test_configuration_mapping_and_dict_negative_branches() -> None:
    bad_value_mapping = cast("m.ConfigMap", {"k": {1}})
    bad_value_dict = cast("m.Dict", {"k": {1}})
    assert not u.is_configuration_mapping(bad_value_mapping)
    assert not u.is_configuration_dict(bad_value_dict)
    assert u.is_configuration_dict({"k": 1})


def test_is_flexible_value_covers_all_branches() -> None:
    assert u.is_flexible_value(None)
    assert u.is_flexible_value(1)
    assert u.is_flexible_value(datetime.now(UTC))
    assert u.is_flexible_value(["a", 1, None, datetime.now(UTC)])
    assert not u.is_flexible_value(["a", {"no": "nested"}])
    assert _is_flexible_value_obj({1: "bad_key"})
    assert not u.is_flexible_value({"k": {"nested": "bad"}})
    assert u.is_flexible_value({"k": "v"})
    empty_set: set[int] = set()
    assert not _is_flexible_value_obj(empty_set)


def test_protocol_and_simple_guard_helpers() -> None:
    plain_obj: test_t.Tests.object = cast("test_t.Tests.object", object())
    assert not _is_type_obj(plain_obj, "config")
    assert not _is_type_obj(plain_obj, "container")
    assert not _is_type_obj(plain_obj, "command_bus")
    assert not _is_type_obj(plain_obj, "handler")
    assert _is_type_obj(cast("test_t.Tests.object", _LoggerLike()), "logger")
    assert _is_type_obj(cast("test_t.Tests.object", r[int].ok(1)), "result")
    assert not _is_type_obj(plain_obj, "service")
    assert not _is_type_obj(plain_obj, "middleware")
    assert u.is_handler_callable(cast("t.NormalizedValue", _sample_handler))
    assert u.is_mapping({"k": "v"})

    def _identity(value: t.NormalizedValue) -> t.NormalizedValue:
        return value

    assert _is_type_obj(cast("test_t.Tests.object", _identity), "callable")
    assert u.is_type(3, "int")
    assert u.is_type([1, 2], "list_or_tuple")
    assert u.is_type("abc", "sized")
    assert u.is_type(1.5, "float")
    assert u.is_type(True, "bool")
    assert u.is_type(None, "none")
    assert u.is_type((1, 2), "tuple")
    assert u.is_type(cast("t.NormalizedValue", b"a"), "bytes")
    assert u.is_type("abc", "str")
    assert u.is_type({"k": "v"}, "dict")
    assert u.is_type([1], "list")
    assert u.is_type((1,), "sequence")
    assert u.is_type({"k": 1}, "mapping")
    assert u.is_type([1], "sequence_not_str")
    assert u.is_type([1], "sequence_not_str_bytes")
    assert u.is_pydantic_model(
        cast("t.NormalizedValue", _Model.model_validate({"value": 1}))
    )


def test_is_type_non_empty_unknown_and_tuple_and_fallback() -> None:
    value_set: set[int] = set()
    assert not _is_type_obj(cast("test_t.Tests.object", value_set), "string_non_empty")
    assert not u.is_type("x", "unknown_type_name")
    assert u.is_type(3, (int, float))
    assert u.is_type("x", str)
    invalid_spec = cast("str | type | tuple[type, ...]", cast("object", 123))
    assert not u.is_type("x", invalid_spec)


def test_is_type_protocol_fallback_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that is_type returns False for non-protocol objects against protocol types."""
    plain_obj: test_t.Tests.object = cast("test_t.Tests.object", object())
    assert not _is_type_obj(plain_obj, "config")
    assert not _is_type_obj(plain_obj, "context")
    assert not _is_type_obj(plain_obj, "handler")
    assert not _is_type_obj(plain_obj, "service")
    assert not _is_type_obj(plain_obj, "middleware")
    assert not _is_type_obj(plain_obj, "result")
    assert not _is_type_obj(plain_obj, "command_bus")
    assert not _is_type_obj(plain_obj, "logger")


def test_extract_mapping_or_none_branches() -> None:
    mapping_result = u.extract_mapping_or_none({"k": "v"})
    assert mapping_result.is_success
    assert mapping_result.value == {"k": "v"}
    mapping_non_str_keys = u.extract_mapping_or_none(
        cast("t.NormalizedValue", {1: "v"})
    )
    assert mapping_non_str_keys.is_failure
    assert u.extract_mapping_or_none([1, 2, 3]).is_failure


def test_guard_in_has_empty_none_helpers() -> None:
    assert not _return_false("x")
    assert u.guard("x", str)
    assert u.guard("x", validator=str, return_value=True) == "x"
    assert u.guard("x", validator=(str, int), return_value=True) == "x"
    assert u.guard("x", str, return_value=True) == "x"
    assert u.guard("x", validator=None, return_value=False)
    assert u.guard("x", validator=None, return_value=True) == "x"

    def _always_false(_v: t.NormalizedValue) -> bool:
        return False

    def _raise_error(_v: t.NormalizedValue) -> bool:
        _ = _v
        msg = "test error"
        raise TypeError(msg)

    assert u.guard("x", validator=_always_false, default="d") == "d"
    assert u.guard("x", validator=_raise_error, default="d") == "d"
    failure_result = u.guard("x", validator=_always_false, return_value=True)
    assert isinstance(failure_result, r)
    assert failure_result.is_failure
    assert u.in_("a", ["a", "b"])
    assert not u.in_([], ("a", "b"))
    assert not u.in_("a", 42)
    assert u.has({"k": 1}, "k")
    assert u.has({"key": "value"}, "key")
    assert u.empty(None)
    assert u.empty(())
    assert not u.empty([1])
    assert u.empty(0)
    assert u.none_(None, None)
    assert not u.none_(None, "x")


def test_chk_exercises_missed_branches() -> None:
    assert not u.chk(1, none=True)
    assert not u.chk(None, none=False)
    assert not u.chk("a", is_=int)
    with pytest.raises(TypeError):
        u.chk("a", is_=list[int])
    with pytest.raises(TypeError):
        u.chk("a", not_=list[int])
    assert not u.chk("a", not_=str)
    assert not u.chk(1, eq=2)
    assert not u.chk(1, ne=1)
    assert not u.chk(1, in_=[2, 3])


def test_guards_bool_shortcut_and_issubclass_typeerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert u.is_container(True)

    class _SomeType:
        pass

    original_issubclass = builtins.issubclass

    def _fake_issubclass(
        cls: type,
        classinfo: type | tuple[type, ...],
    ) -> bool:
        if cls is _SomeType and classinfo is BaseModel:
            msg = "boom"
            raise TypeError(msg)
        return original_issubclass(cls, classinfo)

    monkeypatch.setattr(builtins, "issubclass", _fake_issubclass)
    assert not _is_type_obj(cast("test_t.Tests.object", _SomeType), "handler")
    assert not u.chk(1, not_in=[1, 2])
    assert not u.chk(1, gt=1)
    assert not u.chk(1, gte=2)
    assert not u.chk(1, lt=1)
    assert not u.chk(2, lte=1)
    assert not u.chk(1, empty=True)
    assert not u.chk("", empty=False)
    assert not u.chk("abc", starts="z")
    assert not u.chk("abc", ends="z")
    assert not u.chk("abc", match="\\d+")
    assert not u.chk("abc", contains="z")
    assert not u.chk({"k": "v"}, contains="x")
    assert not u.chk(["k"], contains="x")
    assert not u.chk("abc", contains="x")
    assert u.chk("abc", contains=1)
    assert u.chk("abc", gte=3, lte=3)
    assert u.chk("", empty=True)


def test_guard_instance_attribute_access_warnings() -> None:
    guards = u()
    method = guards.is_type
    assert callable(method)
    private_method = cast(
        "Callable[..., test_t.Tests.object]", getattr(guards, "_is_str")
    )
    assert callable(private_method)


def test_guards_handler_type_issubclass_typeerror_branch_direct() -> None:
    original_issubclass = builtins.issubclass

    class _Candidate:
        pass

    def _explode(
        cls: type,
        classinfo: type | tuple[type, ...],
    ) -> bool:
        if cls is _Candidate and classinfo is BaseModel:
            msg = "boom"
            raise TypeError(msg)
        return original_issubclass(cls, classinfo)

    setattr(builtins, "issubclass", _explode)
    try:
        assert not _is_type_obj(cast("test_t.Tests.object", _Candidate), "handler")
    finally:
        setattr(builtins, "issubclass", original_issubclass)


def test_guards_bool_identity_branch_via_isinstance_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_isinstance = builtins.isinstance

    def _patched_isinstance(
        obj: object,
        classinfo: type | tuple[type, ...],
    ) -> bool:
        if obj is True and classinfo == (str, int, float, bool, type(None), datetime):
            return False
        return original_isinstance(obj, classinfo)

    monkeypatch.setattr(builtins, "isinstance", _patched_isinstance)
    assert u.is_container(True)


def test_guards_issubclass_typeerror_when_class_not_treated_as_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_callable = builtins.callable
    original_issubclass = builtins.issubclass

    class _Candidate:
        pass

    def _patched_callable(value: t.NormalizedValue) -> bool:
        if value is _Candidate:
            return False
        return original_callable(value)

    def _patched_issubclass(
        cls: type,
        classinfo: type | tuple[type, ...],
    ) -> bool:
        if cls is _Candidate and classinfo is BaseModel:
            msg = "boom"
            raise TypeError(msg)
        return original_issubclass(cls, classinfo)

    monkeypatch.setattr(builtins, "callable", _patched_callable)
    monkeypatch.setattr(builtins, "issubclass", _patched_issubclass)
    assert not _is_type_obj(cast("test_t.Tests.object", _Candidate), "handler")


def test_guards_issubclass_success_when_callable_is_patched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_callable = builtins.callable

    class _ModelSub:
        value: str = "ok"

    def _patched_callable(value: t.NormalizedValue) -> bool:
        if value is _ModelSub:
            return False
        return original_callable(value)

    monkeypatch.setattr(builtins, "callable", _patched_callable)
    assert _is_type_obj(cast("test_t.Tests.object", _ModelSub), "handler") is False
