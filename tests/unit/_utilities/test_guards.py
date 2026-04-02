"""Tests for FlextUtilitiesGuards — type guards, ensure guards, and chk."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel

from flext_core import r
from flext_tests import tm
from tests import m, t, u

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SampleModel(BaseModel):
    name: str = "test"


class _NoModelDump:
    """Object without model_dump — should fail is_pydantic_model."""


def _always_true(_v: t.GuardInput) -> bool:
    return True


def _always_false(_v: t.GuardInput) -> bool:
    return False


def _raise_type_error(_v: t.GuardInput) -> bool:
    msg = "boom"
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Parametrize data
# ---------------------------------------------------------------------------

_PRIMITIVE_CASES: list[tuple[t.GuardInput, bool]] = [
    ("hello", True),
    (42, True),
    (math.pi, True),
    (True, True),
    (None, False),
    ([1], False),
    ({"k": "v"}, False),
]

_SCALAR_CASES: list[tuple[t.GuardInput, bool]] = [
    ("s", True),
    (1, True),
    (1.0, True),
    (False, True),
    (datetime(2025, 1, 1, tzinfo=UTC), True),
    (None, False),
    ([1, 2], False),
]

_IS_STR_CASES: list[tuple[str, str, bool]] = [
    ("str", "hello", True),
    ("int", 42, True),
    ("float", 1.5, True),
    ("bool", True, True),
    ("none", None, True),
    ("dict", {"k": 1}, True),
    ("list", [1], True),
    ("tuple", (1,), True),
    ("sequence", [1, 2], True),
    ("sequence", (1,), True),
    ("mapping", {"k": "v"}, True),
    ("list_or_tuple", [1], True),
    ("list_or_tuple", (1,), True),
    ("sequence_not_str", [1], True),
    ("sequence_not_str", "hi", False),
    ("sequence_not_str_bytes", [1], True),
    ("sequence_not_str_bytes", b"x", False),
    ("sized", "abc", True),
    ("callable", lambda: None, True),
    ("bytes", b"x", True),
    ("string_non_empty", "x", True),
    ("string_non_empty", "", False),
    ("string_non_empty", "  ", False),
]


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestFlextUtilitiesGuards:
    """Comprehensive tests for guard utilities accessed through u.* facade."""

    # -----------------------------------------------------------------------
    # Core type guards — is_primitive / is_scalar
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(("value", "expected"), _PRIMITIVE_CASES)
    def test_is_primitive(self, value: t.GuardInput, expected: bool) -> None:
        tm.that(u.is_primitive(value), eq=expected)

    @pytest.mark.parametrize(("value", "expected"), _SCALAR_CASES)
    def test_is_scalar(self, value: t.GuardInput, expected: bool) -> None:
        tm.that(u.is_scalar(value), eq=expected)

    # -----------------------------------------------------------------------
    # is_string_non_empty
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("hello", True),
            ("  x  ", True),
            ("", False),
            ("   ", False),
            ("\t", False),
            ("\n", False),
        ],
    )
    def test_is_string_non_empty(self, value: str, expected: bool) -> None:
        tm.that(u.is_string_non_empty(value), eq=expected)

    def test_is_string_non_empty_non_string_input(self) -> None:
        tm.that(u.is_string_non_empty(42), eq=False)

    # -----------------------------------------------------------------------
    # is_list / is_mapping / is_dict_non_empty
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("value", "expected"),
        [([1, 2], True), ([], True), ((1,), False), ("abc", False), (None, False)],
    )
    def test_is_list(self, value: t.GuardInput, expected: bool) -> None:
        tm.that(u.is_list(value), eq=expected)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [({"k": 1}, True), ({}, True), ([1], False), ("abc", False)],
    )
    def test_is_mapping(self, value: t.GuardInput, expected: bool) -> None:
        tm.that(u.is_mapping(value), eq=expected)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [({"k": 1}, True), ({}, False), ([1], False), (None, False)],
    )
    def test_is_dict_non_empty(self, value: t.ValueOrModel, expected: bool) -> None:
        tm.that(u.is_dict_non_empty(value), eq=expected)

    # -----------------------------------------------------------------------
    # is_container — recursive validation
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, True),
            ("text", True),
            (42, True),
            (math.pi, True),
            (True, True),
            (datetime.now(tz=UTC), True),
            (Path("/tmp"), True),
            ([1, "a", None], True),
            ({"k": 1}, True),
            ((1, "x"), True),
            ([{"nested": 1}], True),
            ({"k": [1, 2]}, True),
        ],
    )
    def test_is_container_valid(self, value: t.GuardInput, expected: bool) -> None:
        tm.that(u.is_container(value), eq=expected)

    def test_is_container_rejects_set_inside_list(self) -> None:
        """Sets are not containers — list containing a set should fail."""
        tm.that(u.is_container([{1, 2}]), eq=False)

    def test_is_container_rejects_set_in_mapping_value(self) -> None:
        tm.that(u.is_container({"k": {1}}), eq=False)

    # -----------------------------------------------------------------------
    # is_object_list / is_object_tuple
    # -----------------------------------------------------------------------

    def test_is_object_list(self) -> None:
        tm.that(u.is_object_list([1, 2]), eq=True)
        tm.that(u.is_object_list((1, 2)), eq=False)
        tm.that(u.is_object_list("abc"), eq=False)

    def test_is_object_tuple(self) -> None:
        tm.that(u.is_object_tuple((1, 2)), eq=True)
        tm.that(u.is_object_tuple([1, 2]), eq=False)

    # -----------------------------------------------------------------------
    # is_pydantic_model
    # -----------------------------------------------------------------------

    def test_is_pydantic_model_with_model(self) -> None:
        tm.that(u.is_pydantic_model(_SampleModel()), eq=True)

    def test_is_pydantic_model_with_non_model(self) -> None:
        tm.that(u.is_pydantic_model("not a model"), eq=False)
        tm.that(u.is_pydantic_model(42), eq=False)
        tm.that(u.is_pydantic_model(None), eq=False)

    def test_is_pydantic_model_with_no_model_dump(self) -> None:
        tm.that(u.is_pydantic_model(_NoModelDump()), eq=False)

    # -----------------------------------------------------------------------
    # is_configuration_dict / is_configuration_mapping
    # -----------------------------------------------------------------------

    def test_is_configuration_dict_valid_mapping(self) -> None:
        tm.that(u.is_configuration_dict({"k": 1, "j": "v"}), eq=True)

    def test_is_configuration_dict_rejects_basemodel_values(self) -> None:
        tm.that(u.is_configuration_dict({"k": _SampleModel()}), eq=False)

    def test_is_configuration_mapping_rejects_non_container_values(self) -> None:
        bad = {"k": {1, 2}}
        tm.that(u.is_configuration_mapping(bad), eq=False)

    # -----------------------------------------------------------------------
    # is_instance_of — generic isinstance
    # -----------------------------------------------------------------------

    def test_is_instance_of(self) -> None:
        tm.that(u.is_instance_of("hello", str), eq=True)
        tm.that(u.is_instance_of(42, int), eq=True)
        tm.that(u.is_instance_of("hello", int), eq=False)

    # -----------------------------------------------------------------------
    # require_initialized
    # -----------------------------------------------------------------------

    def test_require_initialized_with_value(self) -> None:
        result = u.require_initialized("ok", "field")
        tm.that(result, eq="ok")

    def test_require_initialized_with_none(self) -> None:
        with pytest.raises(AttributeError, match="field_name is not initialized"):
            u.require_initialized(None, "field_name")

    # -----------------------------------------------------------------------
    # in_
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        ("value", "container", "expected"),
        [
            ("a", ["a", "b"], True),
            ("c", ["a", "b"], False),
            (1, {1, 2, 3}, True),
            ("k", {"k": "v"}, True),
            ("a", (1, 2), False),
            ("a", "not a container", False),
            ("a", 42, False),
        ],
    )
    def test_in(
        self, value: t.GuardInput, container: t.GuardInput, expected: bool
    ) -> None:
        tm.that(u.in_(value, container), eq=expected)

    # -----------------------------------------------------------------------
    # is_type — string type specs
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(("type_name", "value", "expected"), _IS_STR_CASES)
    def test_is_type_string_spec(
        self, type_name: str, value: t.GuardInput, expected: bool
    ) -> None:
        tm.that(u.is_type(value, type_name), eq=expected)

    def test_is_type_unknown_string_returns_false(self) -> None:
        tm.that(u.is_type("x", "nonexistent_type"), eq=False)

    # -----------------------------------------------------------------------
    # is_type — type and tuple specs
    # -----------------------------------------------------------------------

    def test_is_type_with_type_class(self) -> None:
        tm.that(u.is_type("hello", str), eq=True)
        tm.that(u.is_type(42, str), eq=False)

    def test_is_type_with_type_tuple(self) -> None:
        tm.that(u.is_type(42, (int, float)), eq=True)
        tm.that(u.is_type("x", (int, float)), eq=False)

    def test_is_type_invalid_spec_returns_false(self) -> None:
        """Non-type, non-string, non-tuple spec should return False."""
        tm.that(u.is_type("x", 123), eq=False)

    # -----------------------------------------------------------------------
    # is_type — dict_non_empty / list_non_empty
    # -----------------------------------------------------------------------

    def test_is_type_dict_non_empty(self) -> None:
        tm.that(u.is_type({"k": 1}, "dict_non_empty"), eq=True)
        tm.that(u.is_type({}, "dict_non_empty"), eq=False)

    def test_is_type_list_non_empty(self) -> None:
        tm.that(u.is_type([1], "list_non_empty"), eq=True)
        tm.that(u.is_type([], "list_non_empty"), eq=False)
        tm.that(u.is_type("string", "list_non_empty"), eq=False)

    def test_is_type_non_empty_rejects_basemodel(self) -> None:
        """BaseModel instances should not pass *_non_empty checks."""
        model = _SampleModel()
        tm.that(u.is_type(model, "string_non_empty"), eq=False)
        tm.that(u.is_type(model, "dict_non_empty"), eq=False)
        tm.that(u.is_type(model, "list_non_empty"), eq=False)

    # -----------------------------------------------------------------------
    # Protocol-based type checks via is_type
    # -----------------------------------------------------------------------

    def test_is_type_result_protocol(self) -> None:
        tm.that(u.is_type(r[int].ok(1), "result"), eq=True)
        tm.that(u.is_type("not a result", "result"), eq=False)

    def test_is_type_protocol_names_reject_plain_string(self) -> None:
        """All protocol names should reject plain string values."""
        for name in ("config", "context", "handler", "service", "middleware", "logger"):
            tm.that(u.is_type("plain", name), eq=False)

    # -----------------------------------------------------------------------
    # is_handler_callable / is_factory / is_resource / is_result_like
    # -----------------------------------------------------------------------

    def test_is_handler_callable(self) -> None:
        tm.that(u.is_handler_callable(_always_true), eq=True)
        tm.that(u.is_handler_callable("not callable"), eq=False)

    def test_is_factory(self) -> None:
        tm.that(u.is_factory(lambda: None), eq=True)
        tm.that(u.is_factory(42), eq=False)

    def test_is_resource(self) -> None:
        tm.that(u.is_resource(lambda: None), eq=True)
        tm.that(u.is_resource("nope"), eq=False)

    def test_is_result_like(self) -> None:
        tm.that(u.is_result_like(r[str].ok("ok")), eq=True)
        tm.that(u.is_result_like("not result"), eq=False)

    # -----------------------------------------------------------------------
    # is_registerable_service
    # -----------------------------------------------------------------------

    @pytest.mark.parametrize(
        "value",
        [
            None,
            "string",
            42,
            math.pi,
            True,
            _SampleModel(),
            Path("/tmp"),
            {"k": "v"},
            lambda: None,
            [1, 2, 3],
            (1, 2),
        ],
    )
    def test_is_registerable_service_accepts(self, value: t.GuardInput) -> None:
        tm.that(u.is_registerable_service(value), eq=True)

    # -----------------------------------------------------------------------
    # filter_registerable_services
    # -----------------------------------------------------------------------

    def test_filter_registerable_services_none(self) -> None:
        tm.that(u.filter_registerable_services(None), none=True)

    def test_filter_registerable_services_filters(self) -> None:
        services = {"a": "val", "b": 42, "c": lambda: None}
        result = u.filter_registerable_services(services)
        tm.that(result, none=False)
        tm.that(len(result), eq=3)

    # -----------------------------------------------------------------------
    # is_settings_type
    # -----------------------------------------------------------------------

    def test_is_settings_type_rejects_non_class(self) -> None:
        tm.that(u.is_settings_type("not a type"), eq=False)

    def test_is_settings_type_rejects_class_without_get_global(self) -> None:
        tm.that(u.is_settings_type(str), eq=False)

    def test_is_settings_type_accepts_class_with_get_global(self) -> None:
        class _FakeSettings:
            @staticmethod
            def get_global() -> None:
                pass

        tm.that(u.is_settings_type(_FakeSettings), eq=True)

    # -----------------------------------------------------------------------
    # guard — validator dispatch
    # -----------------------------------------------------------------------

    def test_guard_type_validator_passes(self) -> None:
        tm.that(u.guard("hello", str), eq=True)

    def test_guard_type_validator_fails(self) -> None:
        tm.that(u.guard(42, str), eq=False)

    def test_guard_callable_validator(self) -> None:
        tm.that(u.guard("x", _always_true), eq=True)
        tm.that(u.guard("x", _always_false), eq=False)

    def test_guard_tuple_validator(self) -> None:
        tm.that(u.guard("x", (str, int)), eq=True)
        tm.that(u.guard([], (str, int)), eq=False)

    def test_guard_none_validator_truthy_value(self) -> None:
        """None validator delegates to bool(value)."""
        tm.that(u.guard("x", validator=None), eq=True)

    def test_guard_none_validator_falsy_value(self) -> None:
        tm.that(u.guard("", validator=None), eq=False)

    def test_guard_return_value(self) -> None:
        result = u.guard("hello", str, return_value=True)
        tm.that(result, eq="hello")

    def test_guard_return_value_failure_gives_result(self) -> None:
        result = u.guard(42, str, return_value=True)
        tm.that(isinstance(result, r), eq=True)

    def test_guard_default_on_failure(self) -> None:
        result = u.guard(42, str, default="fallback")
        tm.that(result, eq="fallback")

    def test_guard_exception_uses_default(self) -> None:
        result = u.guard("x", _raise_type_error, default="safe")
        tm.that(result, eq="safe")

    def test_guard_exception_without_default(self) -> None:
        result = u.guard("x", _raise_type_error, return_value=True)
        tm.that(isinstance(result, r), eq=True)

    # -----------------------------------------------------------------------
    # chk — GuardCheckSpec-based checking
    # -----------------------------------------------------------------------

    def test_chk_equality(self) -> None:
        tm.that(u.chk(1, **m.GuardCheckSpec(eq=1).model_dump()), eq=True)
        tm.that(u.chk(1, **m.GuardCheckSpec(eq=2).model_dump()), eq=False)

    def test_chk_inequality(self) -> None:
        tm.that(u.chk(1, **m.GuardCheckSpec(ne=2).model_dump()), eq=True)
        tm.that(u.chk(1, **m.GuardCheckSpec(ne=1).model_dump()), eq=False)

    def test_chk_none_checks(self) -> None:
        tm.that(u.chk(None, **m.GuardCheckSpec(none=True).model_dump()), eq=True)
        tm.that(u.chk(1, **m.GuardCheckSpec(none=True).model_dump()), eq=False)
        tm.that(u.chk(1, **m.GuardCheckSpec(none=False).model_dump()), eq=True)
        tm.that(u.chk(None, **m.GuardCheckSpec(none=False).model_dump()), eq=False)

    def test_chk_type_checks(self) -> None:
        tm.that(u.chk("hello", **m.GuardCheckSpec(is_=str).model_dump()), eq=True)
        tm.that(u.chk("hello", **m.GuardCheckSpec(is_=int).model_dump()), eq=False)
        tm.that(u.chk("hello", **m.GuardCheckSpec(not_=int).model_dump()), eq=True)
        tm.that(u.chk("hello", **m.GuardCheckSpec(not_=str).model_dump()), eq=False)

    def test_chk_membership(self) -> None:
        tm.that(u.chk(1, **m.GuardCheckSpec(in_=[1, 2, 3]).model_dump()), eq=True)
        tm.that(u.chk(4, **m.GuardCheckSpec(in_=[1, 2, 3]).model_dump()), eq=False)
        tm.that(u.chk(4, **m.GuardCheckSpec(not_in=[1, 2]).model_dump()), eq=True)
        tm.that(u.chk(1, **m.GuardCheckSpec(not_in=[1, 2]).model_dump()), eq=False)

    def test_chk_numeric_comparisons(self) -> None:
        tm.that(u.chk(5, **m.GuardCheckSpec(gt=4).model_dump()), eq=True)
        tm.that(u.chk(5, **m.GuardCheckSpec(gt=5).model_dump()), eq=False)
        tm.that(u.chk(5, **m.GuardCheckSpec(gte=5).model_dump()), eq=True)
        tm.that(u.chk(5, **m.GuardCheckSpec(gte=6).model_dump()), eq=False)
        tm.that(u.chk(5, **m.GuardCheckSpec(lt=6).model_dump()), eq=True)
        tm.that(u.chk(5, **m.GuardCheckSpec(lt=5).model_dump()), eq=False)
        tm.that(u.chk(5, **m.GuardCheckSpec(lte=5).model_dump()), eq=True)
        tm.that(u.chk(5, **m.GuardCheckSpec(lte=4).model_dump()), eq=False)

    def test_chk_numeric_uses_len_for_strings(self) -> None:
        """For strings, numeric checks operate on len()."""
        tm.that(u.chk("abc", **m.GuardCheckSpec(gte=3, lte=3).model_dump()), eq=True)
        tm.that(u.chk("ab", **m.GuardCheckSpec(gt=2).model_dump()), eq=False)

    def test_chk_empty(self) -> None:
        tm.that(u.chk("", **m.GuardCheckSpec(empty=True).model_dump()), eq=True)
        tm.that(u.chk("x", **m.GuardCheckSpec(empty=True).model_dump()), eq=False)
        tm.that(u.chk("x", **m.GuardCheckSpec(empty=False).model_dump()), eq=True)
        tm.that(u.chk("", **m.GuardCheckSpec(empty=False).model_dump()), eq=False)

    @pytest.mark.parametrize(
        ("value", "spec_kw", "expected"),
        [
            ("hello world", {"starts": "hello"}, True),
            ("hello world", {"starts": "world"}, False),
            ("hello world", {"ends": "world"}, True),
            ("hello world", {"ends": "hello"}, False),
            ("hello world", {"contains": "lo wo"}, True),
            ("hello world", {"contains": "xyz"}, False),
            ("abc123", {"match": r"\d+"}, True),
            ("abcdef", {"match": r"\d+"}, False),
        ],
    )
    def test_chk_string_operations(
        self, value: str, spec_kw: dict[str, str], expected: bool
    ) -> None:
        spec = m.GuardCheckSpec(**spec_kw)
        tm.that(u.chk(value, **spec.model_dump()), eq=expected)

    def test_chk_iterable_contains(self) -> None:
        """Non-string iterables use element-wise containment check."""
        tm.that(
            u.chk(["a", "b"], **m.GuardCheckSpec(contains="a").model_dump()),
            eq=True,
        )
        tm.that(
            u.chk(["a", "b"], **m.GuardCheckSpec(contains="z").model_dump()),
            eq=False,
        )
        tm.that(
            u.chk({"k": "v"}, **m.GuardCheckSpec(contains="k").model_dump()),
            eq=True,
        )

    def test_chk_with_keyword_criteria(self) -> None:
        """Chk accepts keyword criteria in addition to spec."""
        tm.that(u.chk(5, gt=4), eq=True)
        tm.that(u.chk(5, lt=4), eq=False)

    def test_chk_non_string_contains_skips_integer(self) -> None:
        """Non-string contains with integer value in string spec is a no-op."""
        tm.that(u.chk("abc", **m.GuardCheckSpec(contains=1).model_dump()), eq=True)

    # -----------------------------------------------------------------------
    # _resolve_numeric edge cases
    # -----------------------------------------------------------------------

    def test_resolve_numeric_with_sized_object(self) -> None:
        """Objects with __len__ get resolved via len()."""

        class _Sized:
            def __len__(self) -> int:
                return 7

        tm.that(
            u.chk(_Sized(), **m.GuardCheckSpec(gte=7, lte=7).model_dump()),
            eq=True,
        )

    def test_resolve_numeric_fallback_to_zero(self) -> None:
        """Objects without len or numeric value resolve to 0."""

        class _NoLen:
            pass

        tm.that(
            u.chk(_NoLen(), **m.GuardCheckSpec(gte=0, lte=0).model_dump()),
            eq=True,
        )

    # -----------------------------------------------------------------------
    # is_handler (protocol-based)
    # -----------------------------------------------------------------------

    def test_is_handler_rejects_plain_object(self) -> None:
        tm.that(u.is_handler(_SampleModel()), eq=False)


__all__ = ["TestFlextUtilitiesGuards"]
