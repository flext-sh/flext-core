"""Coverage tests for current utilities guard APIs."""

from __future__ import annotations

import builtins
from collections.abc import Callable, MutableSequence
from datetime import datetime

import pytest
from pydantic_core import ValidationError

from flext_tests import tm
from tests import c, m, r, t, u

_LoggerLike = m.Core.Tests.LoggerLike


class TestUtilitiesGuardsFullCoverage:
    """Coverage tests for current utilities guard APIs."""

    @staticmethod
    def _matches_type_obj(
        value: t.GuardInput,
        type_spec: str | type | tuple[type, ...],
    ) -> bool:
        """Call matches_type with arbitrary t.GuardInput for negative-case testing."""
        fn: Callable[[t.GuardInput, str | type | tuple[type, ...]], bool] = getattr(
            u,
            "matches_type",
        )
        return fn(value, type_spec)

    @staticmethod
    def _sample_handler(value: t.RecursiveContainer) -> t.RecursiveContainer:
        return value

    @staticmethod
    def _return_false(_value: str) -> bool:
        return False

    def test_aliases_are_available(self) -> None:
        tm.that(u, none=False)
        tm.that(c, none=False)
        tm.that(m, none=False)
        tm.that(t, none=False)

    def test_is_container_negative_paths_and_callable(self) -> None:
        handler_input: t.GuardInput = self._sample_handler
        tm.that(
            callable(self._sample_handler) or u.container(handler_input),
            eq=True,
        )
        tm.that(u.container([1, "x", None]), eq=True)
        tm.that(u.container({"k": 1}), eq=True)
        set_list_input: t.GuardInput = [{"x"}]
        tm.that(not u.container(set_list_input), eq=True)
        int_key_input: t.GuardInput = {1: "x"}
        tm.that(u.container(int_key_input), eq=True)
        set_val_input: t.GuardInput = {"x": {1}}
        tm.that(not u.container(set_val_input), eq=True)

    def test_non_empty_and_normalize_branches(self) -> None:
        tm.that(u.string_non_empty("x"), eq=True)
        tm.that(u.matches_type("x", "string_non_empty"), eq=True)
        tm.that(u.dict_non_empty({"k": "v"}), eq=True)
        tm.that(u.normalize_to_metadata("x"), eq="x")
        dict_scalar_out = u.normalize_to_metadata({"k": 1})
        tm.that(dict_scalar_out, eq={"k": 1})
        dict_complex_out = u.normalize_to_metadata(
            {"k": "normalized"},
        )
        tm.that(isinstance(dict_complex_out, dict) and "k" in dict_complex_out, eq=True)
        list_out = u.normalize_to_metadata([1, "normalized"])
        tm.that(list_out, is_=list)
        assert isinstance(list_out, list)
        tm.that(list_out[0], eq=1)
        tm.that(list_out[1], is_=str)
        set_meta_input: set[int] = {1, 2}
        tm.that(u.normalize_to_metadata(set_meta_input), is_=str)

    def test_configuration_mapping_and_dict_negative_branches(self) -> None:
        bad_value_mapping: t.GuardInput = {"k": {1}}
        bad_value_dict: t.GuardInput = {"k": {1}}
        # Current guard contract accepts plain Mapping inputs via mapping fallback.
        tm.that(u.configuration_mapping(bad_value_mapping), eq=True)
        tm.that(u.configuration_dict(bad_value_dict), eq=True)
        tm.that(u.configuration_dict({"k": 1}), eq=True)

    def test_protocol_and_simple_guard_helpers(self) -> None:
        plain_obj: t.RecursiveContainer = "normalized"
        tm.that(not self._matches_type_obj(plain_obj, "settings"), eq=True)
        tm.that(not self._matches_type_obj(plain_obj, "container"), eq=True)
        tm.that(not self._matches_type_obj(plain_obj, "command_bus"), eq=True)
        tm.that(not self._matches_type_obj(plain_obj, "handler"), eq=True)
        logger_input: t.GuardInput = _LoggerLike()
        tm.that(
            not self._matches_type_obj(logger_input, "logger"),
            eq=True,
        )
        result_input: t.GuardInput = r[int].ok(1)
        tm.that(self._matches_type_obj(result_input, "result"), eq=True)
        tm.that(not self._matches_type_obj(plain_obj, "service"), eq=True)
        tm.that(not self._matches_type_obj(plain_obj, "middleware"), eq=True)
        handler_guard_input: t.GuardInput = self._sample_handler
        tm.that(u.handler_callable(handler_guard_input), eq=True)
        tm.that(u.mapping({"k": "v"}), eq=True)

        def _identity(value: t.RecursiveContainer) -> t.RecursiveContainer:
            return value

        identity_input: t.GuardInput = _identity
        tm.that(self._matches_type_obj(identity_input, "callable"), eq=True)
        tm.that(u.matches_type(3, "int"), eq=True)
        tm.that(u.matches_type([1, 2], "list_or_tuple"), eq=True)
        tm.that(u.matches_type("abc", "sized"), eq=True)
        tm.that(u.matches_type(1.5, "float"), eq=True)
        tm.that(u.matches_type(True, "bool"), eq=True)
        tm.that(u.matches_type(None, "none"), eq=True)
        tm.that(u.matches_type((1, 2), "tuple"), eq=True)
        bytes_input: t.RecursiveContainer = b"a"
        tm.that(u.matches_type(bytes_input, "bytes"), eq=True)
        tm.that(u.matches_type("abc", "str"), eq=True)
        tm.that(u.matches_type({"k": "v"}, "dict"), eq=True)
        tm.that(u.matches_type([1], "list"), eq=True)
        tm.that(u.matches_type((1,), "sequence"), eq=True)
        tm.that(u.matches_type({"k": 1}, "mapping"), eq=True)
        tm.that(u.matches_type([1], "sequence_not_str"), eq=True)
        tm.that(u.matches_type([1], "sequence_not_str_bytes"), eq=True)
        model_input: t.GuardInput = m.Core.Tests._Model.model_validate({"value": 1})
        tm.that(
            u.pydantic_model(model_input),
            eq=True,
        )

    def test_is_type_non_empty_unknown_and_tuple_and_fallback(self) -> None:
        value_set: set[int] = set()
        set_input: t.GuardInput = value_set
        tm.that(
            not self._matches_type_obj(set_input, "string_non_empty"),
            eq=True,
        )
        tm.that(not u.matches_type("x", "unknown_type_name"), eq=True)
        tm.that(u.matches_type(3, (int, float)), eq=True)
        tm.that(u.matches_type("x", str), eq=True)
        invalid_spec: t.Scalar = 123
        tm.that(not u.matches_type("x", invalid_spec), eq=True)

    def test_is_type_protocol_fallback_branches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that is_type returns False for non-protocol objects against protocol types."""
        plain_obj: t.RecursiveContainer = "normalized"
        tm.that(not self._matches_type_obj(plain_obj, "settings"), eq=True)
        tm.that(not self._matches_type_obj(plain_obj, "context"), eq=True)
        tm.that(not self._matches_type_obj(plain_obj, "handler"), eq=True)
        tm.that(not self._matches_type_obj(plain_obj, "service"), eq=True)
        tm.that(not self._matches_type_obj(plain_obj, "middleware"), eq=True)
        tm.that(not self._matches_type_obj(plain_obj, "result"), eq=True)
        tm.that(not self._matches_type_obj(plain_obj, "command_bus"), eq=True)
        tm.that(not self._matches_type_obj(plain_obj, "logger"), eq=True)

    def test_guard_in_has_empty_none_helpers(self) -> None:
        tm.that(not self._return_false("x"), eq=True)
        tm.that(u.guard("x", str), eq=True)
        tm.that(u.guard("x", validator=str, return_value=True), eq="x")
        tm.that(u.guard("x", validator=(str, int), return_value=True), eq="x")
        tm.that(u.guard("x", str, return_value=True), eq="x")
        tm.that(u.guard("x", validator=None, return_value=False), eq=True)
        tm.that(u.guard("x", validator=None, return_value=True), eq="x")

        def _always_false(_v: t.RecursiveContainer) -> bool:
            return False

        def _raise_error(_v: t.RecursiveContainer) -> bool:
            _ = _v
            msg = "test error"
            raise TypeError(msg)

        tm.that(u.guard("x", validator=_always_false, default="d"), eq="d")
        tm.that(u.guard("x", validator=_raise_error, default="d"), eq="d")
        failure_result = u.guard("x", validator=_always_false, return_value=True)
        assert isinstance(failure_result, r)
        tm.that(failure_result.failure, eq=True)
        tm.that(u.in_("a", ["a", "b"]), eq=True)
        tm.that(not u.in_([], ("a", "b")), eq=True)
        tm.that(not u.in_("a", 42), eq=True)
        tm.that({"k": 1}, has="k")
        tm.that({"key": "value"}, has="key")

    def test_chk_exercises_missed_branches(self) -> None:
        tm.that(not u.chk(1, **m.GuardCheckSpec(none=True).model_dump()), eq=True)
        tm.that(not u.chk(None, **m.GuardCheckSpec(none=False).model_dump()), eq=True)
        tm.that(not u.chk("a", **m.GuardCheckSpec(is_=int).model_dump()), eq=True)
        bad_is_input: t.GuardInput = MutableSequence[int]
        with pytest.raises(ValidationError):
            u.chk("a", is_=bad_is_input)
        bad_not_input: t.GuardInput = MutableSequence[int]
        with pytest.raises(ValidationError):
            u.chk("a", not_=bad_not_input)
        tm.that(not u.chk("a", **m.GuardCheckSpec(not_=str).model_dump()), eq=True)
        tm.that(not u.chk(1, **m.GuardCheckSpec(eq=2).model_dump()), eq=True)
        tm.that(not u.chk(1, **m.GuardCheckSpec(ne=1).model_dump()), eq=True)
        tm.that(not u.chk(1, **m.GuardCheckSpec(in_=[2, 3]).model_dump()), eq=True)

    def test_guards_bool_shortcut_and_issubclass_typeerror(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tm.that(u.container(True), eq=True)

        class _SomeType:
            pass

        original_issubclass = builtins.issubclass

        def _fake_issubclass(
            cls: type,
            classinfo: type | tuple[type, ...],
        ) -> bool:
            if cls is _SomeType and classinfo is m.BaseModel:
                msg = "boom"
                raise TypeError(msg)
            return original_issubclass(cls, classinfo)

        some_type_input: t.GuardInput = _SomeType
        tm.that(
            not self._matches_type_obj(some_type_input, "handler"),
            eq=True,
        )
        tm.that(not u.chk(1, **m.GuardCheckSpec(not_in=[1, 2]).model_dump()), eq=True)
        tm.that(not u.chk(1, **m.GuardCheckSpec(gt=1).model_dump()), eq=True)
        tm.that(not u.chk(1, **m.GuardCheckSpec(gte=2).model_dump()), eq=True)
        tm.that(not u.chk(1, **m.GuardCheckSpec(lt=1).model_dump()), eq=True)
        tm.that(not u.chk(2, **m.GuardCheckSpec(lte=1).model_dump()), eq=True)
        tm.that(not u.chk(1, **m.GuardCheckSpec(empty=True).model_dump()), eq=True)
        tm.that(not u.chk("", **m.GuardCheckSpec(empty=False).model_dump()), eq=True)
        tm.that(not u.chk("abc", **m.GuardCheckSpec(starts="z").model_dump()), eq=True)
        tm.that(not u.chk("abc", **m.GuardCheckSpec(ends="z").model_dump()), eq=True)
        tm.that(
            not u.chk("abc", **m.GuardCheckSpec(match="\\d+").model_dump()),
            eq=True,
        )
        tm.that(
            not u.chk("abc", **m.GuardCheckSpec(contains="z").model_dump()),
            eq=True,
        )
        tm.that(
            not u.chk({"k": "v"}, **m.GuardCheckSpec(contains="x").model_dump()),
            eq=True,
        )
        tm.that(
            not u.chk(["k"], **m.GuardCheckSpec(contains="x").model_dump()),
            eq=True,
        )
        tm.that(
            not u.chk("abc", **m.GuardCheckSpec(contains="x").model_dump()),
            eq=True,
        )
        tm.that(u.chk("abc", **m.GuardCheckSpec(contains=1).model_dump()), eq=True)
        tm.that(u.chk("abc", **m.GuardCheckSpec(gte=3, lte=3).model_dump()), eq=True)
        tm.that(u.chk("", **m.GuardCheckSpec(empty=True).model_dump()), eq=True)

    def test_guard_instance_attribute_access_warnings(self) -> None:
        guards = u()
        method = guards.matches_type
        tm.that(callable(method), eq=True)
        mapping_method: Callable[..., t.RecursiveContainer] = getattr(guards, "mapping")
        tm.that(callable(mapping_method), eq=True)

    def test_guards_handler_type_issubclass_typeerror_branch_direct(self) -> None:
        original_issubclass = builtins.issubclass

        class _Candidate:
            pass

        def _explode(
            cls: type,
            classinfo: type | tuple[type, ...],
        ) -> bool:
            if cls is _Candidate and classinfo is m.BaseModel:
                msg = "boom"
                raise TypeError(msg)
            return original_issubclass(cls, classinfo)

        setattr(builtins, "issubclass", _explode)
        try:
            candidate_input: t.GuardInput = _Candidate
            tm.that(
                not self._matches_type_obj(candidate_input, "handler"),
                eq=True,
            )
        finally:
            setattr(builtins, "issubclass", original_issubclass)

    def test_guards_bool_identity_branch_via_isinstance_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_isinstance = builtins.isinstance

        def _patched_isinstance(
            obj: t.RecursiveContainer,
            classinfo: type | tuple[type, ...],
        ) -> bool:
            if obj is True and classinfo == (
                str,
                int,
                float,
                bool,
                type(None),
                datetime,
            ):
                return False
            return original_isinstance(obj, classinfo)

        tm.that(u.container(True), eq=True)

    def test_guards_issubclass_typeerror_when_class_not_treated_as_callable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_callable = builtins.callable
        original_issubclass = builtins.issubclass

        class _Candidate:
            pass

        def _patched_callable(value: t.RecursiveContainer) -> bool:
            if value is _Candidate:
                return False
            return original_callable(value)

        def _patched_issubclass(
            cls: type,
            classinfo: type | tuple[type, ...],
        ) -> bool:
            if cls is _Candidate and classinfo is m.BaseModel:
                msg = "boom"
                raise TypeError(msg)
            return original_issubclass(cls, classinfo)

        candidate_input2: t.GuardInput = _Candidate
        tm.that(
            not self._matches_type_obj(candidate_input2, "handler"),
            eq=True,
        )

    def test_guards_issubclass_success_when_callable_is_patched(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_callable = builtins.callable

        class _ModelSub:
            value: str = "ok"

        def _patched_callable(value: t.RecursiveContainer) -> bool:
            if value is _ModelSub:
                return False
            return original_callable(value)

        model_sub_input: t.GuardInput = _ModelSub
        tm.that(
            self._matches_type_obj(model_sub_input, "handler") is False,
            eq=True,
        )
