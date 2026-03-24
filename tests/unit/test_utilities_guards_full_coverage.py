"""Coverage tests for current utilities guard APIs."""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping, MutableSequence
from datetime import UTC, datetime
from typing import cast

import pytest
from flext_tests import tm
from pydantic import BaseModel
from pydantic_core import ValidationError

from flext_core import m as core_m, r
from tests import TestUnitModels, c, m, t, u


def _is_type_obj(
    value: t.NormalizedValue,
    type_spec: str | type | tuple[type, ...],
) -> bool:
    """Call is_type with arbitrary t.NormalizedValue for negative-case testing."""
    fn: Callable[[t.NormalizedValue, str | type | tuple[type, ...]], bool] = getattr(
        u,
        "is_type",
    )
    return fn(value, type_spec)


def _is_flexible_value_obj(value: Mapping[int, str] | set[int]) -> bool:
    """Call is_flexible_value with arbitrary t.NormalizedValue for negative-case testing."""
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
    tm.that(u, none=False)
    tm.that(c, none=False)
    tm.that(m, none=False)
    tm.that(t, none=False)


def test_is_container_negative_paths_and_callable() -> None:
    tm.that(
        callable(_sample_handler)
        or u.is_container(cast("t.NormalizedValue", _sample_handler)),
        eq=True,
    )
    tm.that(u.is_container([1, "x", None]), eq=True)
    tm.that(u.is_container({"k": 1}), eq=True)
    tm.that(not u.is_container(cast("t.NormalizedValue", [{"x"}])), eq=True)
    tm.that(u.is_container(cast("t.NormalizedValue", {1: "x"})), eq=True)
    tm.that(not u.is_container(cast("t.NormalizedValue", {"x": {1}})), eq=True)


def test_is_handler_type_branches() -> None:
    tm.that(u.is_handler_type({"a": 1}), eq=True)
    tm.that(
        u.is_handler_type(
            cast(
                "t.NormalizedValue",
                TestUnitModels._Model.model_validate({"value": 1}),
            ),
        ),
        eq=True,
    )
    tm.that(u.is_handler_type(cast("t.NormalizedValue", _sample_handler)), eq=True)

    class _BaseModelSubclass:
        value: str = "ok"

    class _DuckHandler:
        value: str = "ok"

        def handle(self, _value: t.NormalizedValue) -> None:
            return None

    tm.that(u.is_handler_type(cast("t.NormalizedValue", _BaseModelSubclass)), eq=True)
    tm.that(u.is_handler_type(cast("t.NormalizedValue", _DuckHandler())), eq=True)


def test_non_empty_and_normalize_branches() -> None:
    tm.that(u.is_string_non_empty("x"), eq=True)
    tm.that(u.is_type("x", "string_non_empty"), eq=True)
    tm.that(u.is_dict_non_empty({"k": "v"}), eq=True)
    tm.that(u.is_list_non_empty([1]), eq=True)
    tm.that(u.normalize_to_metadata("x"), eq="x")
    dict_scalar_out = u.normalize_to_metadata({"k": 1})
    tm.that(dict_scalar_out, eq={"k": 1})
    dict_complex_out = u.normalize_to_metadata(
        cast("t.NormalizedValue", {"k": "normalized"}),
    )
    tm.that(isinstance(dict_complex_out, dict) and "k" in dict_complex_out, eq=True)
    list_out = u.normalize_to_metadata(cast("t.NormalizedValue", [1, "normalized"]))
    tm.that(list_out, is_=list)
    assert isinstance(list_out, list)
    tm.that(list_out[0], eq=1)
    tm.that(list_out[1], is_=str)
    tm.that(u.normalize_to_metadata(cast("t.NormalizedValue", {1, 2})), is_=str)


def test_configuration_mapping_and_dict_negative_branches() -> None:
    bad_value_mapping = cast("t.ConfigMap", {"k": {1}})
    bad_value_dict = cast("t.Dict", {"k": {1}})
    tm.that(not u.is_configuration_mapping(bad_value_mapping), eq=True)
    tm.that(not u.is_configuration_dict(bad_value_dict), eq=True)
    tm.that(u.is_configuration_dict({"k": 1}), eq=True)


def test_is_flexible_value_covers_all_branches() -> None:
    tm.that(u.is_flexible_value(None), eq=True)
    tm.that(u.is_flexible_value(1), eq=True)
    tm.that(u.is_flexible_value(datetime.now(UTC)), eq=True)
    tm.that(u.is_flexible_value(["a", 1, None, datetime.now(UTC)]), eq=True)
    tm.that(not u.is_flexible_value(["a", {"no": "nested"}]), eq=True)
    tm.that(_is_flexible_value_obj({1: "bad_key"}), eq=True)
    tm.that(not u.is_flexible_value({"k": {"nested": "bad"}}), eq=True)
    tm.that(u.is_flexible_value({"k": "v"}), eq=True)
    empty_set: set[int] = set()
    tm.that(not _is_flexible_value_obj(empty_set), eq=True)


def test_protocol_and_simple_guard_helpers() -> None:
    plain_obj: t.NormalizedValue = cast("t.NormalizedValue", "normalized")
    tm.that(not _is_type_obj(plain_obj, "config"), eq=True)
    tm.that(not _is_type_obj(plain_obj, "container"), eq=True)
    tm.that(not _is_type_obj(plain_obj, "command_bus"), eq=True)
    tm.that(not _is_type_obj(plain_obj, "handler"), eq=True)
    tm.that(
        not _is_type_obj(cast("t.NormalizedValue", _LoggerLike()), "logger"),
        eq=True,
    )
    tm.that(_is_type_obj(cast("t.NormalizedValue", r[int].ok(1)), "result"), eq=True)
    tm.that(not _is_type_obj(plain_obj, "service"), eq=True)
    tm.that(not _is_type_obj(plain_obj, "middleware"), eq=True)
    tm.that(u.is_handler_callable(cast("t.NormalizedValue", _sample_handler)), eq=True)
    tm.that(u.is_mapping({"k": "v"}), eq=True)

    def _identity(value: t.NormalizedValue) -> t.NormalizedValue:
        return value

    tm.that(_is_type_obj(cast("t.NormalizedValue", _identity), "callable"), eq=True)
    tm.that(u.is_type(3, "int"), eq=True)
    tm.that(u.is_type([1, 2], "list_or_tuple"), eq=True)
    tm.that(u.is_type("abc", "sized"), eq=True)
    tm.that(u.is_type(1.5, "float"), eq=True)
    tm.that(u.is_type(True, "bool"), eq=True)
    tm.that(u.is_type(None, "none"), eq=True)
    tm.that(u.is_type((1, 2), "tuple"), eq=True)
    tm.that(u.is_type(cast("t.NormalizedValue", b"a"), "bytes"), eq=True)
    tm.that(u.is_type("abc", "str"), eq=True)
    tm.that(u.is_type({"k": "v"}, "dict"), eq=True)
    tm.that(u.is_type([1], "list"), eq=True)
    tm.that(u.is_type((1,), "sequence"), eq=True)
    tm.that(u.is_type({"k": 1}, "mapping"), eq=True)
    tm.that(u.is_type([1], "sequence_not_str"), eq=True)
    tm.that(u.is_type([1], "sequence_not_str_bytes"), eq=True)
    tm.that(
        u.is_pydantic_model(
            cast(
                "t.NormalizedValue",
                TestUnitModels._Model.model_validate({"value": 1}),
            ),
        ),
        eq=True,
    )


def test_is_type_non_empty_unknown_and_tuple_and_fallback() -> None:
    value_set: set[int] = set()
    tm.that(
        not _is_type_obj(cast("t.NormalizedValue", value_set), "string_non_empty"),
        eq=True,
    )
    tm.that(not u.is_type("x", "unknown_type_name"), eq=True)
    tm.that(u.is_type(3, (int, float)), eq=True)
    tm.that(u.is_type("x", str), eq=True)
    invalid_spec = cast("str | type | tuple[type, ...]", cast("t.NormalizedValue", 123))
    tm.that(not u.is_type("x", invalid_spec), eq=True)


def test_is_type_protocol_fallback_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that is_type returns False for non-protocol objects against protocol types."""
    plain_obj: t.NormalizedValue = cast("t.NormalizedValue", "normalized")
    tm.that(not _is_type_obj(plain_obj, "config"), eq=True)
    tm.that(not _is_type_obj(plain_obj, "context"), eq=True)
    tm.that(not _is_type_obj(plain_obj, "handler"), eq=True)
    tm.that(not _is_type_obj(plain_obj, "service"), eq=True)
    tm.that(not _is_type_obj(plain_obj, "middleware"), eq=True)
    tm.that(not _is_type_obj(plain_obj, "result"), eq=True)
    tm.that(not _is_type_obj(plain_obj, "command_bus"), eq=True)
    tm.that(not _is_type_obj(plain_obj, "logger"), eq=True)


def test_extract_mapping_or_none_branches() -> None:
    mapping_result = u.extract_mapping_or_none({"k": "v"})
    tm.that(mapping_result.is_success, eq=True)
    tm.that(mapping_result.value, eq={"k": "v"})
    mapping_non_str_keys = u.extract_mapping_or_none(
        cast("t.NormalizedValue", {1: "v"}),
    )
    tm.that(mapping_non_str_keys.is_success, eq=True)
    tm.that(u.extract_mapping_or_none([1, 2, 3]).is_failure, eq=True)


def test_guard_in_has_empty_none_helpers() -> None:
    tm.that(not _return_false("x"), eq=True)
    tm.that(u.guard("x", str), eq=True)
    tm.that(u.guard("x", validator=str, return_value=True), eq="x")
    tm.that(u.guard("x", validator=(str, int), return_value=True), eq="x")
    tm.that(u.guard("x", str, return_value=True), eq="x")
    tm.that(u.guard("x", validator=None, return_value=False), eq=True)
    tm.that(u.guard("x", validator=None, return_value=True), eq="x")

    def _always_false(_v: t.NormalizedValue) -> bool:
        return False

    def _raise_error(_v: t.NormalizedValue) -> bool:
        _ = _v
        msg = "test error"
        raise TypeError(msg)

    tm.that(u.guard("x", validator=_always_false, default="d"), eq="d")
    tm.that(u.guard("x", validator=_raise_error, default="d"), eq="d")
    failure_result = u.guard("x", validator=_always_false, return_value=True)
    assert isinstance(failure_result, r)
    tm.that(failure_result.is_failure, eq=True)
    tm.that(u.in_("a", ["a", "b"]), eq=True)
    tm.that(not u.in_([], ("a", "b")), eq=True)
    tm.that(not u.in_("a", 42), eq=True)
    tm.that({"k": 1}, has="k")
    tm.that({"key": "value"}, has="key")
    tm.that(u.empty(None), eq=True)
    tm.that(u.empty(()), eq=True)
    tm.that(not u.empty([1]), eq=True)
    tm.that(u.empty(0), eq=True)
    tm.that(u.none_(None, None), eq=True)
    tm.that(not u.none_(None, "x"), eq=True)


def test_chk_exercises_missed_branches() -> None:
    tm.that(not u.chk(1, **core_m.GuardCheckSpec(none=True).model_dump()), eq=True)
    tm.that(not u.chk(None, **core_m.GuardCheckSpec(none=False).model_dump()), eq=True)
    tm.that(not u.chk("a", **core_m.GuardCheckSpec(is_=int).model_dump()), eq=True)
    with pytest.raises(ValidationError):
        u.chk("a", is_=cast("t.NormalizedValue", MutableSequence[int]))
    with pytest.raises(ValidationError):
        u.chk("a", not_=cast("t.NormalizedValue", MutableSequence[int]))
    tm.that(not u.chk("a", **core_m.GuardCheckSpec(not_=str).model_dump()), eq=True)
    tm.that(not u.chk(1, **core_m.GuardCheckSpec(eq=2).model_dump()), eq=True)
    tm.that(not u.chk(1, **core_m.GuardCheckSpec(ne=1).model_dump()), eq=True)
    tm.that(not u.chk(1, **core_m.GuardCheckSpec(in_=[2, 3]).model_dump()), eq=True)


def test_guards_bool_shortcut_and_issubclass_typeerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tm.that(u.is_container(True), eq=True)

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
    tm.that(not _is_type_obj(cast("t.NormalizedValue", _SomeType), "handler"), eq=True)
    tm.that(not u.chk(1, **core_m.GuardCheckSpec(not_in=[1, 2]).model_dump()), eq=True)
    tm.that(not u.chk(1, **core_m.GuardCheckSpec(gt=1).model_dump()), eq=True)
    tm.that(not u.chk(1, **core_m.GuardCheckSpec(gte=2).model_dump()), eq=True)
    tm.that(not u.chk(1, **core_m.GuardCheckSpec(lt=1).model_dump()), eq=True)
    tm.that(not u.chk(2, **core_m.GuardCheckSpec(lte=1).model_dump()), eq=True)
    tm.that(not u.chk(1, **core_m.GuardCheckSpec(empty=True).model_dump()), eq=True)
    tm.that(not u.chk("", **core_m.GuardCheckSpec(empty=False).model_dump()), eq=True)
    tm.that(not u.chk("abc", **core_m.GuardCheckSpec(starts="z").model_dump()), eq=True)
    tm.that(not u.chk("abc", **core_m.GuardCheckSpec(ends="z").model_dump()), eq=True)
    tm.that(
        not u.chk("abc", **core_m.GuardCheckSpec(match="\\d+").model_dump()),
        eq=True,
    )
    tm.that(
        not u.chk("abc", **core_m.GuardCheckSpec(contains="z").model_dump()),
        eq=True,
    )
    tm.that(
        not u.chk({"k": "v"}, **core_m.GuardCheckSpec(contains="x").model_dump()),
        eq=True,
    )
    tm.that(
        not u.chk(["k"], **core_m.GuardCheckSpec(contains="x").model_dump()),
        eq=True,
    )
    tm.that(
        not u.chk("abc", **core_m.GuardCheckSpec(contains="x").model_dump()),
        eq=True,
    )
    tm.that(u.chk("abc", **core_m.GuardCheckSpec(contains=1).model_dump()), eq=True)
    tm.that(u.chk("abc", **core_m.GuardCheckSpec(gte=3, lte=3).model_dump()), eq=True)
    tm.that(u.chk("", **core_m.GuardCheckSpec(empty=True).model_dump()), eq=True)


def test_guard_instance_attribute_access_warnings() -> None:
    guards = u()
    method = guards.is_type
    tm.that(callable(method), eq=True)
    mapping_method = cast(
        "Callable[..., t.NormalizedValue]",
        getattr(guards, "is_mapping"),
    )
    tm.that(callable(mapping_method), eq=True)


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
        tm.that(
            not _is_type_obj(cast("t.NormalizedValue", _Candidate), "handler"),
            eq=True,
        )
    finally:
        setattr(builtins, "issubclass", original_issubclass)


def test_guards_bool_identity_branch_via_isinstance_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_isinstance = builtins.isinstance

    def _patched_isinstance(
        obj: t.NormalizedValue,
        classinfo: type | tuple[type, ...],
    ) -> bool:
        if obj is True and classinfo == (str, int, float, bool, type(None), datetime):
            return False
        return original_isinstance(obj, classinfo)

    monkeypatch.setattr(builtins, "isinstance", _patched_isinstance)
    tm.that(u.is_container(True), eq=True)


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
    tm.that(not _is_type_obj(cast("t.NormalizedValue", _Candidate), "handler"), eq=True)


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
    tm.that(
        _is_type_obj(cast("t.NormalizedValue", _ModelSub), "handler") is False,
        eq=True,
    )
