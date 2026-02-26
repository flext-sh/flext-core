"""Coverage tests for current utilities guard APIs."""

from __future__ import annotations

import builtins
from collections.abc import Callable
from datetime import UTC, datetime
from typing import ClassVar, cast

import pytest
from flext_core import c, m, r, t, u
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core.protocols import p
from pydantic import BaseModel


class _Model(BaseModel):
    name: str = "x"


class _LoggerLike:
    def debug(self, *_args: object, **_kwargs: object) -> None:
        return None

    def info(self, *_args: object, **_kwargs: object) -> None:
        return None

    def warning(self, *_args: object, **_kwargs: object) -> None:
        return None

    def error(self, *_args: object, **_kwargs: object) -> None:
        return None

    def exception(self, *_args: object, **_kwargs: object) -> None:
        return None


def _sample_handler(value: t.GeneralValueType) -> t.GeneralValueType:
    return value


def _return_false(_value: object) -> bool:
    return False


def test_aliases_are_available() -> None:
    assert u is not None
    assert c is not None
    assert m is not None
    assert t is not None


def test_is_general_value_type_negative_paths_and_callable() -> None:
    assert u.Guards.is_general_value_type(_sample_handler)
    assert u.Guards.is_general_value_type([1, "x", None])
    assert u.Guards.is_general_value_type({"k": 1})
    assert not u.Guards.is_general_value_type([{"x"}])
    assert not u.Guards.is_general_value_type({1: "x"})
    assert not u.Guards.is_general_value_type({"x": {1}})


def test_is_handler_type_branches() -> None:
    assert u.Guards.is_handler_type({"a": 1})
    assert u.Guards.is_handler_type(_Model())
    assert u.Guards.is_handler_type(cast("t.GuardInputValue", _sample_handler))

    class _BaseModelSubclass(BaseModel):
        value: str = "ok"

    class _DuckHandler(BaseModel):
        value: str = "ok"

        def handle(self, _value: object) -> object:
            return None

    assert u.Guards.is_handler_type(cast("t.GuardInputValue", _BaseModelSubclass))
    assert u.Guards.is_handler_type(cast("t.GuardInputValue", _DuckHandler()))


def test_non_empty_and_normalize_branches() -> None:
    assert u.Guards.is_string_non_empty("x")
    assert u.is_type("x", "string_non_empty")
    assert u.Guards.is_dict_non_empty({"k": "v"})
    assert u.Guards.is_list_non_empty([1])

    assert u.normalize_to_metadata_value("x") == "x"
    dict_scalar_out = u.normalize_to_metadata_value({"k": 1})
    assert dict_scalar_out == {"k": 1}
    dict_out = u.normalize_to_metadata_value(
        cast("t.GeneralValueType", {"k": object()}),
    )
    assert isinstance(dict_out, dict)
    assert "k" in dict_out
    list_out = u.normalize_to_metadata_value(
        cast("t.GeneralValueType", [1, object()]),
    )
    assert isinstance(list_out, list)
    assert list_out[0] == 1
    assert isinstance(list_out[1], str)
    assert isinstance(
        u.normalize_to_metadata_value(
            cast("t.GeneralValueType", cast("object", {1, 2})),
        ),
        str,
    )


def test_configuration_mapping_and_dict_negative_branches() -> None:
    assert not u.Guards.is_configuration_mapping(1)
    bad_key_mapping: dict[object, object] = {1: "ok"}
    bad_value_mapping: dict[str, object] = {"k": {1}}
    bad_value_dict: dict[str, object] = {"k": {1}}

    assert not u.Guards.is_configuration_mapping(
        cast("t.GuardInputValue", bad_key_mapping)
    )
    assert not u.Guards.is_configuration_mapping(
        cast("t.GuardInputValue", bad_value_mapping)
    )
    assert not u.Guards.is_configuration_dict([])
    assert not u.Guards.is_configuration_dict(cast("t.GuardInputValue", {1: "v"}))
    assert not u.Guards.is_configuration_dict(cast("t.GuardInputValue", bad_value_dict))
    assert u.Guards.is_configuration_dict({"k": 1})


def test_is_flexible_value_covers_all_branches() -> None:
    assert u.Guards.is_flexible_value(None)
    assert u.Guards.is_flexible_value(1)
    assert u.Guards.is_flexible_value(datetime.now(UTC))
    assert u.Guards.is_flexible_value(["a", 1, None, datetime.now(UTC)])
    assert not u.Guards.is_flexible_value(["a", {"no": "nested"}])
    assert not u.Guards.is_flexible_value({1: "bad_key"})
    assert not u.Guards.is_flexible_value({"k": {"nested": "bad"}})
    assert u.Guards.is_flexible_value({"k": "v"})
    assert not u.Guards.is_flexible_value(cast("object", set()))


def test_protocol_and_simple_guard_helpers() -> None:
    class _ContextLike:
        def clone(self) -> _ContextLike:
            return self

        def set(self, key: str, value: t.GeneralValueType, scope: str = "") -> object:
            return {"key": key, "value": value, "scope": scope}

        def get(self, key: str, scope: str = "") -> object:
            return {"key": key, "scope": scope}

    assert u.Guards.is_context(_ContextLike())
    assert u.is_type(_ContextLike(), "context")

    assert not u.is_type(object(), "config")
    assert not u.is_type(object(), "container")
    assert not u.is_type(object(), "command_bus")
    assert not u.is_type(object(), "handler")
    assert u.is_type(_LoggerLike(), "logger")
    assert u.is_type(r[int].ok(1), "result")
    assert not u.is_type(object(), "service")
    assert not u.is_type(object(), "middleware")

    assert u.Guards.is_handler_callable(cast("t.GuardInputValue", _sample_handler))

    assert u.Guards.is_mapping({"k": "v"})

    def _identity(value: t.GeneralValueType) -> t.GeneralValueType:
        return value

    assert u.is_type(_identity, "callable")
    assert u.is_type(3, "int")
    assert u.is_type([1, 2], "list_or_tuple")
    assert u.is_type("abc", "sized")
    assert u.is_type(1.5, "float")
    assert u.is_type(True, "bool")
    assert u.is_type(None, "none")
    assert u.is_type((1, 2), "tuple")
    assert u.is_type(b"a", "bytes")
    assert u.is_type("abc", "str")
    assert u.is_type({"k": "v"}, "dict")
    assert u.is_type([1], "list")
    assert u.is_type((1,), "sequence")
    assert u.is_type({"k": 1}, "mapping")
    assert u.is_type([1], "sequence_not_str")
    assert u.is_type([1], "sequence_not_str_bytes")
    assert u.Guards.is_pydantic_model(_Model())


def test_is_type_non_empty_unknown_and_tuple_and_fallback() -> None:
    assert not u.is_type(cast("object", set()), "string_non_empty")
    assert not u.is_type("x", "unknown_type_name")
    assert u.is_type(3, (int, float))
    assert u.is_type("x", str)
    invalid_spec = cast(
        "str | type | tuple[type, ...]",
        cast("object", 123),
    )
    assert not u.is_type("x", invalid_spec)


def test_is_type_protocol_fallback_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ProtocolLike:
        __protocol_attrs__: ClassVar[set[str]] = {"x"}

    original_isinstance = builtins.isinstance

    def _fake_isinstance(
        obj: object,
        classinfo: type[object] | tuple[type[object], ...],
    ) -> bool:
        if classinfo is _ProtocolLike:
            msg = "forced protocol isinstance error"
            raise TypeError(msg)
        return original_isinstance(obj, classinfo)

    targets: list[tuple[str, type[object]]] = [
        ("Config", p.Config),
        ("Context", p.Context),
        ("DI", p.DI),
        ("CommandBus", p.CommandBus),
        ("Handler", p.Handler),
        ("Result", p.Result),
        ("Service", p.Service),
        ("Middleware", p.Middleware),
    ]
    fallback_methods: dict[str, str] = {
        "Config": "_is_config",
        "Context": "_is_context",
        "DI": "_is_container",
        "CommandBus": "_is_command_bus",
        "Handler": "_is_handler",
        "Result": "_is_result",
        "Service": "_is_service",
        "Middleware": "_is_middleware",
    }

    for attr_name, original in targets:
        monkeypatch.setattr(p, attr_name, _ProtocolLike)
        with monkeypatch.context() as local_patch:
            local_patch.setattr(
                FlextUtilitiesGuards,
                fallback_methods[attr_name],
                staticmethod(_return_false),
            )
            local_patch.setattr("builtins.isinstance", _fake_isinstance)
            assert not u.is_type(object(), _ProtocolLike)
        monkeypatch.setattr(p, attr_name, original)

    class _LogNamespace:
        StructlogLogger: type[object] = _ProtocolLike

    original_log = p.Log
    monkeypatch.setattr(p, "Log", _LogNamespace)
    with monkeypatch.context() as local_patch:
        local_patch.setattr(
            FlextUtilitiesGuards,
            "_is_logger",
            staticmethod(_return_false),
        )
        local_patch.setattr("builtins.isinstance", _fake_isinstance)
        assert not u.is_type(object(), _ProtocolLike)
    monkeypatch.setattr(p, "Log", original_log)

    with monkeypatch.context() as local_patch:
        local_patch.setattr("builtins.isinstance", _fake_isinstance)
        assert not u.is_type(object(), _ProtocolLike)


def test_extract_mapping_or_none_branches() -> None:
    assert u.extract_mapping_or_none({"k": "v"}) == {"k": "v"}
    assert u.extract_mapping_or_none(cast("t.GuardInputValue", {1: "v"})) is None
    assert u.extract_mapping_or_none([1, 2, 3]) is None


def test_guard_in_has_empty_none_helpers() -> None:
    assert u.guard("x", str)
    assert u.guard("x", validator=str, return_value=True) == "x"
    assert u.guard("x", validator=(str, int), return_value=True) == "x"
    assert u.guard("x", str, return_value=True) == "x"
    assert u.guard("x", validator=None, return_value=False)
    assert u.guard("x", validator=None, return_value=True) == "x"

    def _always_false(_v: object) -> bool:
        return False

    def _raise_error(_v: object) -> bool:
        _ = _v
        return cast("bool", 1 / 0)

    assert u.guard("x", validator=_always_false, default="d") == "d"
    assert u.guard("x", validator=_raise_error, default="d") == "d"

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
    assert u.chk("a", is_=list[int])
    assert u.chk("a", not_=list[int])
    assert not u.chk("a", not_=str)
    assert not u.chk(1, eq=2)
    assert not u.chk(1, ne=1)
    assert not u.chk(1, in_=[2, 3])


def test_guards_bool_shortcut_and_issubclass_typeerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert u.Guards.is_general_value_type(True)

    class _SomeType:
        pass

    original_issubclass = builtins.issubclass

    def _fake_issubclass(
        cls: type[object],
        classinfo: type[object] | tuple[type[object], ...],
    ) -> bool:
        if cls is _SomeType and classinfo is BaseModel:
            msg = "boom"
            raise TypeError(msg)
        return original_issubclass(cls, classinfo)

    monkeypatch.setattr(builtins, "issubclass", _fake_issubclass)
    assert u.is_type(_SomeType, "handler")
    assert not u.chk(1, not_in=[1, 2])
    assert not u.chk(1, gt=1)
    assert not u.chk(1, gte=2)
    assert not u.chk(1, lt=1)
    assert not u.chk(2, lte=1)
    assert not u.chk(1, empty=True)
    assert not u.chk("", empty=False)
    assert not u.chk("abc", starts="z")
    assert not u.chk("abc", ends="z")
    assert not u.chk("abc", match=r"\d+")
    assert not u.chk("abc", contains="z")
    assert not u.chk({"k": "v"}, contains="x")
    assert not u.chk(["k"], contains="x")
    assert not u.chk("abc", contains="x")
    assert not u.chk("abc", contains=1)
    assert u.chk("abc", gte=3, lte=3)
    assert u.chk("", empty=True)


def test_guard_instance_attribute_access_warnings() -> None:
    guards = u.Guards()

    with pytest.deprecated_call():
        method = guards.is_type
    assert callable(method)

    private_method = cast("Callable[..., object]", getattr(guards, "_is_str"))
    assert callable(private_method)


def test_guards_handler_type_issubclass_typeerror_branch_direct() -> None:
    original_issubclass = builtins.issubclass

    class _Candidate:
        pass

    def _explode(
        cls: type[object],
        classinfo: type[object] | tuple[type[object], ...],
    ) -> bool:
        if cls is _Candidate and classinfo is BaseModel:
            msg = "boom"
            raise TypeError(msg)
        return original_issubclass(cls, classinfo)

    setattr(builtins, "issubclass", _explode)
    try:
        assert u.is_type(_Candidate, "handler")
    finally:
        setattr(builtins, "issubclass", original_issubclass)


def test_guards_bool_identity_branch_via_isinstance_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_isinstance = builtins.isinstance

    def _patched_isinstance(
        obj: object,
        classinfo: type[object] | tuple[type[object], ...],
    ) -> bool:
        if obj is True and classinfo == (str, int, float, bool, type(None), datetime):
            return False
        return original_isinstance(obj, classinfo)

    monkeypatch.setattr(builtins, "isinstance", _patched_isinstance)
    assert u.Guards.is_general_value_type(True)


def test_guards_issubclass_typeerror_when_class_not_treated_as_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_callable = builtins.callable
    original_issubclass = builtins.issubclass

    class _Candidate:
        pass

    def _patched_callable(value: object) -> bool:
        if value is _Candidate:
            return False
        return original_callable(value)

    def _patched_issubclass(
        cls: type[object],
        classinfo: type[object] | tuple[type[object], ...],
    ) -> bool:
        if cls is _Candidate and classinfo is BaseModel:
            msg = "boom"
            raise TypeError(msg)
        return original_issubclass(cls, classinfo)

    monkeypatch.setattr(builtins, "callable", _patched_callable)
    monkeypatch.setattr(builtins, "issubclass", _patched_issubclass)
    assert not u.is_type(_Candidate, "handler")


def test_guards_issubclass_success_when_callable_is_patched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_callable = builtins.callable

    class _ModelSub(BaseModel):
        value: str = "ok"

    def _patched_callable(value: object) -> bool:
        if value is _ModelSub:
            return False
        return original_callable(value)

    monkeypatch.setattr(builtins, "callable", _patched_callable)
    assert u.is_type(_ModelSub, "handler")
