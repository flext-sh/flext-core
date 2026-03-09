"""Coverage tests for current utilities parser APIs."""

from __future__ import annotations

from collections import UserString
from enum import StrEnum
from typing import cast, override

import pytest

from flext_core import c, m, r, t, u

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


class _Status(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"


def _raise_type_error_value(_value: object) -> str:
    msg = "x"
    raise TypeError(msg)


def _fail_components(*_args: object, **_kwargs: object) -> r[list[str]]:
    return r[list[str]].fail("forced")


def _safe_length_abc(_value: object) -> str:
    return "abc"


def _fail_escape_split(*_args: object) -> r[tuple[list[str], int]]:
    return r[tuple[list[str], int]].fail("split fail")


def _fail_pipeline_continue(*_args: object, **_kwargs: object) -> r[str]:
    return r[str].fail("Continue pipeline", error_code="PIPELINE_CONTINUE")


def _raise_runtime_boom(*_args: object, **_kwargs: object) -> str:
    msg = "boom"
    raise RuntimeError(msg)


def _raise_value_error_float(_value: t.ContainerValue) -> r[float]:
    msg = "boom"
    raise ValueError(msg)


def _raise_type_error_bool(_value: t.ContainerValue) -> r[bool]:
    msg = "boom"
    raise TypeError(msg)


def _raise_value_error_int(_value: t.ContainerValue) -> r[int]:
    msg = "boom"
    raise ValueError(msg)


def _raise_type_error_str(_value: t.ContainerValue) -> r[str]:
    msg = "boom"
    raise TypeError(msg)


def _norm_list_dict(*_args: object, **_kwargs: object) -> dict[str, str]:
    return {"k": "v"}


def test_parser_safe_length_and_parse_delimited_error_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = u.Parser()
    sample: t.ContainerValue = "ok"
    assert sample == "ok"
    assert isinstance(c.Processing.PATTERN_TUPLE_MIN_LENGTH, int)
    assert parser._safe_text_length(_LenRaises("x")) == "unknown"
    monkeypatch.setattr(parser, "_safe_text_length", _raise_type_error_value)
    result = parser.parse_delimited("a,b", ",")
    assert result.is_success
    monkeypatch.setattr(parser, "_process_components", _fail_components)
    forced_failure = parser.parse_delimited("a,b", ",")
    assert forced_failure.is_failure

    class _SplitRaises:
        def split(self, _delimiter: str) -> list[str]:
            msg = "split boom"
            raise RuntimeError(msg)

    monkeypatch.setattr(parser, "_safe_text_length", _raise_type_error_value)
    split_failure = parser.parse_delimited(
        cast("str", cast("object", _SplitRaises())),
        ",",
    )
    assert split_failure.is_failure


def test_parser_split_and_normalize_exception_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = u.Parser()
    monkeypatch.setattr(parser, "_safe_text_length", _safe_length_abc)
    assert parser._get_safe_text_length("abc") == -1
    monkeypatch.setattr(parser, "_process_escape_splitting", _fail_escape_split)
    split_result = parser._execute_escape_splitting("a,b", ",", "\\")
    assert split_result.is_failure
    text_obj = cast("str", cast("object", _BoolRaises()))
    normal_split = u.Parser().split_on_char_with_escape(text_obj, ",")
    assert normal_split.is_failure
    parser2 = u.Parser()
    monkeypatch.setattr(parser2, "_safe_text_length", _raise_type_error_value)
    normalized = parser2.normalize_whitespace("x")
    assert normalized.is_success
    regex_fail = u.Parser().normalize_whitespace("abc", pattern="[", replacement="x")
    assert regex_fail.is_failure


def test_parser_pipeline_and_pattern_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = u.Parser()
    monkeypatch.setattr(parser, "_safe_text_length", _raise_type_error_value)
    ok = parser.apply_regex_pipeline("abc", [("a", "b")])
    assert ok.is_success
    monkeypatch.setattr(parser, "_handle_pipeline_edge_cases", _fail_pipeline_continue)
    none_text = parser.apply_regex_pipeline(None, [("a", "b")])
    assert none_text.is_failure
    parser2 = u.Parser()
    monkeypatch.setattr(parser2, "_process_all_patterns", _raise_runtime_boom)
    monkeypatch.setattr(parser2, "_safe_text_length", _raise_type_error_value)
    fail = parser2.apply_regex_pipeline("abc", [("a", "b")])
    assert fail.is_failure
    str_conversion_result = u.Parser()._extract_key_from_str_conversion(
        cast("t.ContainerValue", cast("object", _StrRaises())),
    )
    assert str_conversion_result.is_failure

    class _OddNoStr:
        @override
        def __str__(self) -> str:
            msg = "bad"
            raise TypeError(msg)

    parser3 = u.Parser()
    original_hasattr = hasattr

    def _patched_hasattr(obj: object, name: str) -> bool:
        if name == "__class__":
            return False
        return original_hasattr(obj, name)

    monkeypatch.setattr("builtins.hasattr", _patched_hasattr)
    assert "<object object" in parser3.get_object_key(
        cast("t.ContainerValue", object()),
    )
    assert (
        parser3.get_object_key(cast("t.ContainerValue", cast("object", _OddNoStr())))
        == "_OddNoStr"
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
    assert invalid_type.is_failure
    assert invalid_flag.is_failure
    assert invalid_len.is_failure
    bad_tuple = parser3._process_all_patterns("x", [cast("tuple[str, str]", ("x",))])
    assert bad_tuple.is_failure


def test_parser_parse_helpers_and_primitive_coercion_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = u.Parser()
    assert parser._parse_find_first([1, 2], lambda v: v > 5).is_failure
    assert parser._parse_normalize_compare("x", 1) is False
    assert parser._parse_normalize_str(123, case="lower") == "123"
    assert parser._parse_normalize_str("abc", case="upper") == "ABC"
    assert parser._parse_normalize_str("abc", case="none") == "abc"
    assert parser._parse_result_error(r[int].ok(1), default="fallback") == "fallback"
    model_result = parser._parse_model(
        cast("t.ContainerValue", {"name": "ok", "count": 2, "payload": "obj"}),
        _Model,
        "field: ",
        strict=False,
    )
    assert model_result is not None
    assert parser._coerce_to_int([]).is_failure
    assert parser._coerce_to_float("bad").is_failure
    assert parser._coerce_to_bool("maybe").is_failure
    bool_from_int = parser._coerce_to_bool(0)
    assert bool_from_int is not None and bool_from_int.is_success
    assert parser._coerce_to_str(42).is_success
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
    assert enum_ci is not None and enum_ci.is_success
    assert enum_value is not None and enum_value.is_success
    primitive_float = parser._parse_try_primitive("2.2", float, 0.0, None, "")
    primitive_str = parser._parse_try_primitive(5, str, "x", None, "")
    assert primitive_float is not None and primitive_float.is_success
    assert primitive_str is not None and primitive_str.is_success
    monkeypatch.setattr(
        u.Parser.__mro__[1],
        "_coerce_to_float",
        staticmethod(_raise_value_error_float),
    )
    failed_float = parser._parse_try_primitive("x", float, 1.2, None, "field: ")
    assert failed_float is not None and failed_float.is_failure
    monkeypatch.setattr(
        u.Parser.__mro__[1],
        "_coerce_to_bool",
        staticmethod(_raise_type_error_bool),
    )
    failed_bool = parser._parse_try_primitive("x", bool, True, None, "field: ")
    assert failed_bool is not None and failed_bool.is_failure

    class _DefaultBox:
        pass

    parsed = parser.parse(
        "x",
        int,
        default=cast("int", cast("object", _DefaultBox())),
        default_factory=lambda: 9,
    )
    assert parsed.is_success


def test_parser_convert_and_norm_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = u.Parser()

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

    assert parser.convert("10", int, 0) == 10
    assert parser._convert_to_int(True, default=7) == 7
    parsed_float = parser._convert_to_float(1.5, default=0.0)
    assert abs(parsed_float - 1.5) < 1e-09
    assert parser._convert_to_str("x", default="") == "x"
    assert parser._convert_to_str(None, default="d") == "d"
    assert (
        parser._convert_to_str(
            cast("t.ContainerValue", cast("object", _BadStr())),
            default="d",
        )
        == "d"
    )
    assert parser._convert_to_bool(True, default=False) is True
    assert (
        parser._convert_to_bool(cast("t.ContainerValue", object()), default=True)
        is True
    )
    assert (
        parser.conv_str(
            cast("t.ContainerValue", cast("object", _BadConv())),
            default="d",
        )
        == "d"
    )
    assert parser.conv_str_list(5) == ["5"]
    assert parser.norm_str("abc") == "abc"
    assert parser.norm_list({"a": "", "b": "B"}, case="lower", filter_truthy=True) == {
        "b": "b",
    }
    assert parser.norm_list(["A", "b"], case="lower", to_set=True) == {"a", "b"}
    assert parser.norm_join(["A", "B"], sep="-") == "A-B"
    mapping_result = parser.norm_in("a", m.ConfigMap(root={"A": "1"}), case="lower")
    config_map_result = parser.norm_in("a", m.ConfigMap(root={"A": "x"}), case="lower")
    with pytest.raises(TypeError):
        parser.norm_in("a", cast("list[str]", object()), case="lower")
    assert mapping_result is True
    assert config_map_result is True
    original_norm_list = u.Parser.norm_list
    monkeypatch.setattr(u.Parser.__mro__[1], "norm_list", staticmethod(_norm_list_dict))
    try:
        assert parser.norm_in("v", ["x"], case="lower") is False
    finally:
        monkeypatch.setattr(u.Parser.__mro__[1], "norm_list", original_norm_list)


def test_parser_success_and_edge_paths_cover_major_branches() -> None:
    parser = u.Parser()
    opts = m.CollectionsParseOptions(
        strip=True,
        remove_empty=True,
        validator=lambda value: len(value) > 1,
    )
    processed = parser.parse_delimited(" a, b, cc ,, ddd ", ",", options=opts)
    assert processed.is_success
    assert processed.value == ["cc", "ddd"]
    assert parser.parse_delimited("", ",").value == []
    assert parser.parse_delimited("a,b", "").is_failure
    assert parser.parse_delimited("a,b", " ").is_failure
    escaped = parser.split_on_char_with_escape("a\\,b,c", ",")
    assert escaped.is_success and escaped.value == ["a,b", "c"]
    assert parser.split_on_char_with_escape("", ",").value == [""]
    normalized_empty = parser.normalize_whitespace("")
    assert normalized_empty.is_success and normalized_empty.value == ""
    assert parser.apply_regex_pipeline("", [("a", "b")]).value == ""
    assert parser.apply_regex_pipeline("abc", []).value == "abc"
    assert parser.apply_regex_pipeline(None, [("a", "b")]).is_failure


def test_parser_internal_helpers_additional_coverage() -> None:
    parser = u.Parser()
    mapped = parser._extract_key_from_mapping({"name": "n1", "id": "i1"})
    attrs = parser._extract_key_from_attributes(
        cast("t.ContainerValue", cast("object", type("Obj", (), {"id": "x1"})())),
    )
    assert mapped.is_success and mapped.value == "n1"
    assert attrs.is_success and attrs.value == "x1"
    split = parser._process_escape_splitting("a\\,b,c", ",", "\\")
    assert split.is_success
    assert split.value[0] == ["a,b", "c"]
    assert split.value[1] == 1
    handled_none = parser._handle_pipeline_edge_cases(None, [("a", "b")])
    handled_empty = parser._handle_pipeline_edge_cases("", [("a", "b")])
    handled_patterns = parser._handle_pipeline_edge_cases("abc", [])
    assert handled_none is not None and handled_none.is_failure
    assert handled_empty is not None and handled_empty.is_success
    assert handled_patterns is not None and handled_patterns.is_success
    assert parser._parse_with_default(None, lambda: 3, "err").value == 3
    assert parser._parse_with_default(None, None, "err").is_failure
    enum_exact = parser._parse_enum("ACTIVE", _Status, case_insensitive=False)
    enum_by_value = parser._parse_enum("inactive", _Status, case_insensitive=False)
    assert enum_exact is not None and enum_exact.is_success
    assert enum_by_value is not None and enum_by_value.is_success


def test_parser_remaining_branch_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = u.Parser()
    assert parser._coerce_to_float([]).is_failure

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
    assert enum_by_member_value is not None and enum_by_member_value.is_success
    monkeypatch.setattr(
        u.Parser.__mro__[1],
        "_coerce_to_int",
        staticmethod(_raise_value_error_int),
    )
    failed_int = parser._parse_try_primitive("x", int, 1, None, "field: ")
    assert failed_int is not None and failed_int.is_failure
    monkeypatch.setattr(
        u.Parser.__mro__[1],
        "_coerce_to_str",
        staticmethod(_raise_type_error_str),
    )
    failed_str = parser._parse_try_primitive("x", str, "d", None, "field: ")
    assert failed_str is not None and failed_str.is_failure
    assert parser.convert("x", bool, cast("bool", cast("object", "d"))) == "d"
    assert parser._convert_to_int(5, default=7) == 5
    assert parser._convert_to_float(cast("t.ContainerValue", object()), default=1.5)
    assert (
        abs(
            parser._convert_to_float(cast("t.ContainerValue", object()), default=1.5)
            - 1.5,
        )
        < 1e-09
    )
    assert (
        parser.norm_in("a", cast("list[str]", cast("object", {"A": "1"})), case="lower")
        is True
    )
