"""Coverage tests for current utilities parser APIs."""

from __future__ import annotations

from collections import UserString
from enum import StrEnum, unique
from typing import cast, override

import pytest
from flext_tests import t as test_t, tm

from flext_core import r
from flext_core._utilities.parser import FlextUtilitiesParser
from tests import c, m, t, u

from ._models import _Model


class _LenRaises(UserString):
    @override
    def __len__(self) -> int:
        msg = "len boom"
        raise TypeError(msg)


class _BoolRaises:
    def __bool__(self) -> bool:
        msg = "bool boom"
        raise TypeError(msg)


class _StrRaises:
    @override
    def __str__(self) -> str:
        msg = "str boom"
        raise TypeError(msg)


@unique
class _Status(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"


def _raise_type_error_value(_value: t.Scalar) -> str:
    msg = "x"
    raise TypeError(msg)


def _fail_components(*_args: t.Scalar, **_kwargs: t.Scalar) -> r[list[str]]:
    return r[list[str]].fail("forced")


def _safe_length_abc(_value: t.Scalar) -> str:
    return "abc"


def _fail_escape_split(*_args: t.Scalar) -> r[tuple[list[str], int]]:
    return r[tuple[list[str], int]].fail("split fail")


def _fail_pipeline_continue(*_args: t.Scalar, **_kwargs: t.Scalar) -> r[str]:
    return r[str].fail("Continue pipeline", error_code="PIPELINE_CONTINUE")


def _raise_runtime_boom(*_args: t.Scalar, **_kwargs: t.Scalar) -> str:
    msg = "boom"
    raise RuntimeError(msg)


def _raise_value_error_float(_value: t.Scalar) -> r[float]:
    msg = "boom"
    raise ValueError(msg)


def _raise_type_error_bool(_value: t.Scalar) -> r[bool]:
    msg = "boom"
    raise TypeError(msg)


def _raise_value_error_int(_value: t.Scalar) -> r[int]:
    msg = "boom"
    raise ValueError(msg)


def _raise_type_error_str(_value: t.Scalar) -> r[str]:
    msg = "boom"
    raise TypeError(msg)


def _norm_list_dict(*_args: t.Scalar, **_kwargs: t.Scalar) -> dict[str, str]:
    return {"k": "v"}


def test_parser_safe_length_and_parse_delimited_error_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = u()
    sample = "ok"
    tm.that(sample, eq="ok")
    tm.that(c.Processing.PATTERN_TUPLE_MIN_LENGTH, is_=int)
    tm.that(
        parser._safe_text_length(cast("t.NormalizedValue", _LenRaises("x"))),
        eq="unknown",
    )
    monkeypatch.setattr(parser, "_safe_text_length", _raise_type_error_value)
    result = parser.parse_delimited("a,b", ",")
    tm.ok(result)
    monkeypatch.setattr(parser, "_process_components", _fail_components)
    forced_failure = parser.parse_delimited("a,b", ",")
    tm.fail(forced_failure)

    class _SplitRaises:
        def split(self, _delimiter: str) -> list[str]:
            msg = "split boom"
            raise RuntimeError(msg)

    monkeypatch.setattr(parser, "_safe_text_length", _raise_type_error_value)
    split_failure = parser.parse_delimited(
        cast("str", cast("object", _SplitRaises())),
        ",",
    )
    tm.fail(split_failure)


def test_parser_split_and_normalize_exception_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = u()
    monkeypatch.setattr(parser, "_safe_text_length", _safe_length_abc)
    tm.that(parser._get_safe_text_length("abc"), eq=-1)
    monkeypatch.setattr(parser, "_process_escape_splitting", _fail_escape_split)
    split_result = parser._execute_escape_splitting("a,b", ",", "\\")
    tm.fail(split_result)
    text_obj = cast("str", cast("object", _BoolRaises()))
    normal_split = u().split_on_char_with_escape(text_obj, ",")
    tm.fail(normal_split)
    parser2 = u()
    monkeypatch.setattr(parser2, "_safe_text_length", _raise_type_error_value)
    normalized = parser2.normalize_whitespace("x")
    tm.ok(normalized)
    regex_fail = u().normalize_whitespace("abc", pattern="[", replacement="x")
    tm.fail(regex_fail)


def test_parser_pipeline_and_pattern_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = u()
    monkeypatch.setattr(parser, "_safe_text_length", _raise_type_error_value)
    ok = parser.apply_regex_pipeline("abc", [("a", "b")])
    tm.ok(ok)
    monkeypatch.setattr(parser, "_handle_pipeline_edge_cases", _fail_pipeline_continue)
    none_text = parser.apply_regex_pipeline(None, [("a", "b")])
    tm.fail(none_text)
    parser2 = u()
    monkeypatch.setattr(parser2, "_process_all_patterns", _raise_runtime_boom)
    monkeypatch.setattr(parser2, "_safe_text_length", _raise_type_error_value)
    fail = parser2.apply_regex_pipeline("abc", [("a", "b")])
    tm.fail(fail)
    str_conversion_result = u()._extract_key_from_str_conversion(
        cast("t.NormalizedValue", _StrRaises()),
    )
    tm.fail(str_conversion_result)

    class _OddNoStr:
        @override
        def __str__(self) -> str:
            msg = "bad"
            raise TypeError(msg)

    parser3 = u()
    original_hasattr = hasattr

    def _patched_hasattr(obj: test_t.Tests.object, name: str) -> bool:
        if name == "__class__":
            return False
        return original_hasattr(obj, name)

    monkeypatch.setattr("builtins.hasattr", _patched_hasattr)
    tm.that(
        parser3.get_object_key(cast("t.NormalizedValue", object())),
        has="<object object",
    )
    tm.that(
        parser3.get_object_key(cast("t.NormalizedValue", _OddNoStr())),
        eq="_OddNoStr",
    )
    invalid_type = parser3._extract_pattern_components(
        cast("tuple[str, str, int]", cast("object", ("a", 1, 0))),
    )
    invalid_flag = parser3._extract_pattern_components(
        cast("tuple[str, str, int]", cast("object", ("a", "b", "x"))),
    )
    invalid_len = parser3._extract_pattern_components(
        cast("tuple[str, str]", ("only",)),
    )
    tm.fail(invalid_type)
    tm.fail(invalid_flag)
    tm.fail(invalid_len)
    bad_tuple = parser3._process_all_patterns("x", [cast("tuple[str, str]", ("x",))])
    tm.fail(bad_tuple)


def test_parser_parse_helpers_and_primitive_coercion_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = u()
    tm.fail(parser._parse_find_first([1, 2], lambda v: v > 5))
    tm.that(parser._parse_normalize_compare("x", 1), eq=False)
    tm.that(parser._parse_normalize_str(123, case="lower"), eq="123")
    tm.that(parser._parse_normalize_str("abc", case="upper"), eq="ABC")
    tm.that(parser._parse_normalize_str("abc", case="none"), eq="abc")
    tm.that(parser._parse_result_error(r[int].ok(1), default="fallback"), eq="fallback")
    model_result = parser._parse_model(
        cast("t.NormalizedValue", {"name": "ok", "count": 2, "payload": "obj"}),
        _Model,
        "field: ",
        strict=False,
    )
    tm.that(hasattr(model_result, "is_success"), eq=True)
    tm.fail(parser._coerce_to_int([]))
    tm.fail(parser._coerce_to_float("bad"))
    tm.fail(parser._coerce_to_bool("maybe"))
    bool_from_int = parser._coerce_to_bool(0)
    tm.ok(bool_from_int)
    tm.ok(parser._coerce_to_str(42))
    enum_ci = parser._parse_try_enum(
        "inactive",
        _Status,
        case_insensitive=True,
        default=None,
        default_factory=None,
        field_prefix="",
    )
    enum_value = parser._parse_try_enum(
        "inactive",
        _Status,
        case_insensitive=False,
        default=None,
        default_factory=None,
        field_prefix="",
    )
    tm.ok(enum_ci)
    tm.ok(enum_value)
    primitive_float = parser._parse_try_primitive("2.2", float, 0.0, None, "")
    primitive_str = parser._parse_try_primitive(5, str, "x", None, "")
    tm.ok(primitive_float)
    tm.ok(primitive_str)
    monkeypatch.setattr(
        FlextUtilitiesParser,
        "_coerce_to_float",
        staticmethod(_raise_value_error_float),
    )
    failed_float = parser._parse_try_primitive("x", float, 1.2, None, "field: ")
    tm.fail(failed_float)
    monkeypatch.setattr(
        FlextUtilitiesParser,
        "_coerce_to_bool",
        staticmethod(_raise_type_error_bool),
    )
    failed_bool = parser._parse_try_primitive("x", bool, True, None, "field: ")
    tm.fail(failed_bool)

    class _DefaultBox:
        pass

    parsed = parser.parse(
        "x",
        int,
        default=cast("int", cast("object", _DefaultBox())),
        default_factory=lambda: 9,
    )
    tm.ok(parsed)


def test_parser_convert_and_norm_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = u()

    class _BadStr:
        @override
        def __str__(self) -> str:
            msg = "nope"
            raise TypeError(msg)

    class _BadConv:
        @override
        def __str__(self) -> str:
            msg = "nope"
            raise TypeError(msg)

    tm.that(parser.convert("10", int, 0), eq=10)
    tm.that(parser._convert_to_int(True, default=7), eq=7)
    parsed_float = parser._convert_to_float(1.5, default=0.0)
    tm.that(abs(parsed_float - 1.5), lt=1e-09)
    tm.that(parser._convert_to_str("x", default=""), eq="x")
    tm.that(parser._convert_to_str(None, default="d"), eq="d")
    tm.that(
        parser._convert_to_str(
            cast("t.NormalizedValue", _BadStr()),
            default="d",
        ),
        eq="d",
    )
    tm.that(parser._convert_to_bool(True, default=False), eq=True)
    tm.that(
        parser._convert_to_bool(cast("t.NormalizedValue", object()), default=True),
        eq=True,
    )
    tm.that(
        parser.conv_str(
            cast("t.NormalizedValue", _BadConv()),
            default="d",
        ),
        eq="d",
    )
    tm.that(parser.conv_str_list(5), eq=["5"])
    tm.that(parser.norm_str("abc"), eq="abc")
    normalized_map = parser.norm_list(
        t.ConfigMap(root={"a": "", "b": "B"}),
        case="lower",
        filter_truthy=True,
    )
    tm.that(len(normalized_map), eq=1)
    tm.that(normalized_map == {"b": "b"}, eq=True)
    normalized_set = parser.norm_list(["A", "b"], case="lower", to_set=True)
    tm.that(len(normalized_set), eq=2)
    tm.that(normalized_set == {"a", "b"}, eq=True)
    tm.that(parser.norm_join(["A", "B"], sep="-"), eq="A-B")
    mapping_result = parser.norm_in("a", t.ConfigMap(root={"A": "1"}), case="lower")
    config_map_result = parser.norm_in("a", t.ConfigMap(root={"A": "x"}), case="lower")
    with pytest.raises(TypeError):
        parser.norm_in("a", cast("list[str]", object()), case="lower")
    tm.that(mapping_result, eq=True)
    tm.that(config_map_result, eq=True)
    original_norm_list = u.norm_list

    monkeypatch.setattr(
        FlextUtilitiesParser, "norm_list", staticmethod(_norm_list_dict)
    )
    try:
        tm.that(parser.norm_in("v", ["x"], case="lower"), eq=False)
    finally:
        monkeypatch.setattr(FlextUtilitiesParser, "norm_list", original_norm_list)


def test_parser_success_and_edge_paths_cover_major_branches() -> None:
    parser = u()
    opts = m.ParseOptions(
        strip=True,
        remove_empty=True,
        validator=lambda value: len(value) > 1,
    )
    processed = parser.parse_delimited(" a, b, cc ,, ddd ", ",", options=opts)
    tm.ok(processed)
    tm.ok(processed, eq=["cc", "ddd"])
    tm.ok(parser.parse_delimited("", ","), eq=[])
    tm.fail(parser.parse_delimited("a,b", ""))
    tm.fail(parser.parse_delimited("a,b", " "))
    escaped = parser.split_on_char_with_escape("a\\,b,c", ",")
    tm.ok(escaped, eq=["a,b", "c"])
    tm.ok(parser.split_on_char_with_escape("", ","), eq=[""])
    normalized_empty = parser.normalize_whitespace("")
    tm.ok(normalized_empty, eq="")
    tm.ok(parser.apply_regex_pipeline("", [("a", "b")]), eq="")
    tm.ok(parser.apply_regex_pipeline("abc", []), eq="abc")
    tm.fail(parser.apply_regex_pipeline(None, [("a", "b")]))


def test_parser_internal_helpers_additional_coverage() -> None:
    parser = u()
    mapped = parser._extract_key_from_mapping({"name": "n1", "id": "i1"})
    attrs = parser._extract_key_from_attributes(
        cast("t.NormalizedValue", type("Obj", (), {"id": "x1"})()),
    )
    tm.ok(mapped, eq="n1")
    tm.fail(attrs)
    tm.that(attrs.error, eq="No key attribute found")
    split = parser._process_escape_splitting("a\\,b,c", ",", "\\")
    tm.ok(split)
    split_val: tuple[list[str], int] = split.value
    tm.that(split_val[0], eq=["a,b", "c"])
    tm.that(split_val[1], eq=1)
    handled_none = parser._handle_pipeline_edge_cases(None, [("a", "b")])
    handled_empty = parser._handle_pipeline_edge_cases("", [("a", "b")])
    handled_patterns = parser._handle_pipeline_edge_cases("abc", [])
    tm.fail(handled_none)
    tm.ok(handled_empty)
    tm.ok(handled_patterns)
    tm.that(parser._parse_with_default(None, lambda: 3, "err").value, eq=3)
    tm.that(parser._parse_with_default(None, None, "err").is_failure, eq=True)
    enum_exact = parser._parse_enum("ACTIVE", _Status, case_insensitive=False)
    enum_by_value = parser._parse_enum("inactive", _Status, case_insensitive=False)
    tm.ok(enum_exact)
    tm.ok(enum_by_value)


def test_parser_remaining_branch_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = u()
    tm.fail(parser._coerce_to_float([]))

    @unique
    class _ValueEnum(StrEnum):
        FIRST = "v-1"

    enum_by_member_value = parser._parse_try_enum(
        "V-1",
        _ValueEnum,
        case_insensitive=True,
        default=None,
        default_factory=None,
        field_prefix="",
    )
    tm.ok(enum_by_member_value)

    monkeypatch.setattr(
        FlextUtilitiesParser,
        "_coerce_to_int",
        staticmethod(_raise_value_error_int),
    )
    failed_int = parser._parse_try_primitive("x", int, 1, None, "field: ")
    tm.fail(failed_int)
    monkeypatch.setattr(
        FlextUtilitiesParser,
        "_coerce_to_str",
        staticmethod(_raise_type_error_str),
    )
    failed_str = parser._parse_try_primitive("x", str, "d", None, "field: ")
    tm.fail(failed_str)
    tm.that(parser.convert("x", bool, cast("bool", cast("object", "d"))), eq="d")
    tm.that(parser._convert_to_int(5, default=7), eq=5)
    tm.that(
        parser._convert_to_float(cast("t.NormalizedValue", object()), default=1.5),
        eq=1.5,
    )
    tm.that(
        abs(
            parser._convert_to_float(cast("t.NormalizedValue", object()), default=1.5)
            - 1.5,
        ),
        lt=1e-09,
    )
    tm.that(parser.norm_in("a", t.ConfigMap(root={"A": "1"}), case="lower"), eq=True)
