from __future__ import annotations

from enum import StrEnum
from typing import cast

from pydantic import BaseModel

from flext_core import c, m, r, t, u
from flext_core._utilities.parser import FlextUtilitiesParser


class _LenRaises(str):
    def __len__(self) -> int:
        raise TypeError("len boom")


class _BoolRaises:
    def __bool__(self) -> bool:
        raise TypeError("bool boom")


class _StrRaises:
    def __str__(self) -> str:
        raise TypeError("str boom")


class _Status(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class _Model(BaseModel):
    name: str
    count: int


def test_parser_safe_length_and_parse_delimited_error_paths(monkeypatch) -> None:
    parser = u.Parser()
    sample: t.GeneralValueType = "ok"
    assert sample == "ok"
    assert isinstance(c.Processing.PATTERN_TUPLE_MIN_LENGTH, int)

    assert parser._safe_text_length(_LenRaises("x")) == "unknown"

    monkeypatch.setattr(
        parser, "_safe_text_length", lambda _v: (_ for _ in ()).throw(TypeError("x"))
    )
    result = parser.parse_delimited("a,b", ",")
    assert result.is_success

    monkeypatch.setattr(
        parser,
        "_process_components",
        lambda *_args, **_kwargs: r[list[str]].fail("forced"),
    )
    forced_failure = parser.parse_delimited("a,b", ",")
    assert forced_failure.is_failure

    class _SplitRaises:
        def split(self, _delimiter: str) -> list[str]:
            raise RuntimeError("split boom")

    monkeypatch.setattr(
        parser, "_safe_text_length", lambda _v: (_ for _ in ()).throw(TypeError("x"))
    )
    split_failure = parser.parse_delimited(
        cast("str", cast("object", _SplitRaises())),
        ",",
    )
    assert split_failure.is_failure


def test_parser_split_and_normalize_exception_paths(monkeypatch) -> None:
    parser = u.Parser()

    monkeypatch.setattr(parser, "_safe_text_length", lambda _v: "abc")
    assert parser._get_safe_text_length("abc") == -1

    monkeypatch.setattr(
        parser,
        "_process_escape_splitting",
        lambda *_args: r[tuple[list[str], int]].fail("split fail"),
    )
    split_result = parser._execute_escape_splitting("a,b", ",", "\\")
    assert split_result.is_failure

    text_obj = cast("str", cast("object", _BoolRaises()))
    normal_split = u.Parser().split_on_char_with_escape(text_obj, ",")
    assert normal_split.is_failure

    parser2 = u.Parser()
    monkeypatch.setattr(
        parser2, "_safe_text_length", lambda _v: (_ for _ in ()).throw(TypeError("x"))
    )
    normalized = parser2.normalize_whitespace("x")
    assert normalized.is_success

    regex_fail = u.Parser().normalize_whitespace("abc", pattern="[", replacement="x")
    assert regex_fail.is_failure


def test_parser_pipeline_and_pattern_branches(monkeypatch) -> None:
    parser = u.Parser()

    monkeypatch.setattr(
        parser, "_safe_text_length", lambda _v: (_ for _ in ()).throw(TypeError("x"))
    )
    ok = parser.apply_regex_pipeline("abc", [("a", "b")])
    assert ok.is_success

    monkeypatch.setattr(
        parser, "_handle_pipeline_edge_cases", lambda *_args, **_kwargs: None
    )
    none_text = parser.apply_regex_pipeline(None, [("a", "b")])
    assert none_text.is_failure

    parser2 = u.Parser()
    monkeypatch.setattr(
        parser2,
        "_process_all_patterns",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        parser2, "_safe_text_length", lambda _v: (_ for _ in ()).throw(TypeError("x"))
    )
    fail = parser2.apply_regex_pipeline("abc", [("a", "b")])
    assert fail.is_failure

    assert (
        u.Parser()._extract_key_from_str_conversion(
            cast("t.GeneralValueType", _StrRaises())
        )
        is None
    )

    class _OddNoStr:
        def __str__(self) -> str:
            raise TypeError("bad")

    parser3 = u.Parser()
    original_hasattr = (
        __builtins__["hasattr"] if isinstance(__builtins__, dict) else hasattr
    )

    def _patched_hasattr(obj: object, name: str) -> bool:
        if name == "__class__":
            return False
        return original_hasattr(obj, name)

    monkeypatch.setattr("builtins.hasattr", _patched_hasattr)
    assert "<object object" in parser3.get_object_key(
        cast("t.GeneralValueType", object())
    )
    assert (
        parser3.get_object_key(cast("t.GeneralValueType", _OddNoStr())) == "_OddNoStr"
    )

    invalid_type = parser3._extract_pattern_components(
        cast("tuple[str, str, int]", cast("object", ("a", 1, 0)))
    )
    invalid_flag = parser3._extract_pattern_components(
        cast("tuple[str, str, int]", cast("object", ("a", "b", "x")))
    )
    invalid_len = parser3._extract_pattern_components(
        cast("tuple[str, str]", ("only",))
    )
    assert invalid_type.is_failure
    assert invalid_flag.is_failure
    assert invalid_len.is_failure

    bad_tuple = parser3._process_all_patterns("x", [cast("tuple[str, str]", ("x",))])
    assert bad_tuple.is_failure


def test_parser_parse_helpers_and_primitive_coercion_branches(monkeypatch) -> None:
    parser = u.Parser()

    assert parser._parse_find_first([1, 2], lambda v: v > 5) is None
    assert parser._parse_normalize_compare("x", 1) is False
    assert parser._parse_normalize_str(123, case="lower") == "123"
    assert parser._parse_normalize_str("abc", case="upper") == "ABC"
    assert parser._parse_normalize_str("abc", case="none") == "abc"
    assert parser._parse_result_error(r[int].ok(1), default="fallback") == "fallback"

    model_result = parser._parse_model(
        {"name": "ok", "count": 2, "payload": cast("t.GeneralValueType", "obj")},
        _Model,
        "field: ",
        strict=False,
    )
    assert model_result is not None

    assert parser._coerce_to_int([]) is None
    assert parser._coerce_to_float("bad") is None
    assert parser._coerce_to_bool("maybe") is None
    bool_from_int = parser._coerce_to_bool(0)
    assert bool_from_int is not None and bool_from_int.is_success
    assert parser._coerce_to_str(42).is_success

    assert parser._coerce_primitive("1", int) is not None
    assert parser._coerce_primitive("1.2", float) is not None
    assert parser._coerce_primitive(1, str) is not None
    assert parser._coerce_primitive("true", bool) is not None

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

    assert parser._try_coerce_to_primitive("1", int) is not None
    assert parser._try_coerce_to_primitive("1.1", float) is not None
    assert parser._try_coerce_to_primitive(1, str) is not None
    assert parser._try_coerce_to_primitive("true", bool) is not None

    primitive_float = parser._parse_try_primitive("2.2", float, 0.0, None, "")
    primitive_str = parser._parse_try_primitive(5, str, "x", None, "")
    assert primitive_float is not None and primitive_float.is_success
    assert primitive_str is not None and primitive_str.is_success

    monkeypatch.setattr(
        FlextUtilitiesParser,
        "_coerce_to_float",
        staticmethod(lambda _v: (_ for _ in ()).throw(ValueError("boom"))),
    )
    failed_float = parser._parse_try_primitive("x", float, 1.2, None, "field: ")
    assert failed_float is not None and failed_float.is_failure

    monkeypatch.setattr(
        FlextUtilitiesParser,
        "_coerce_to_bool",
        staticmethod(lambda _v: (_ for _ in ()).throw(TypeError("boom"))),
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


def test_parser_convert_and_norm_branches(monkeypatch) -> None:
    parser = u.Parser()

    class _BadStr:
        def __str__(self) -> str:
            raise TypeError("nope")

    class _BadConv:
        def __str__(self) -> str:
            raise TypeError("nope")

    assert parser.convert("10", int, 0) == 10
    assert parser._convert_to_int(True, default=7) == 7
    assert parser._convert_to_float(1.5, default=0.0) == 1.5
    assert parser._convert_to_str("x", default="") == "x"
    assert parser._convert_to_str(None, default="d") == "d"
    assert (
        parser._convert_to_str(cast("t.GeneralValueType", _BadStr()), default="d")
        == "d"
    )
    assert parser._convert_to_bool(True, default=False) is True
    assert (
        parser._convert_to_bool(cast("t.GeneralValueType", object()), default=True)
        is True
    )
    assert parser._convert_fallback("x", object, "d") == "d"

    assert parser.conv_str(cast("t.GeneralValueType", _BadConv()), default="d") == "d"
    assert parser.conv_str_list(5) == ["5"]
    assert parser.norm_str("abc") == "abc"
    assert parser.norm_list({"a": "", "b": "B"}, case="lower", filter_truthy=True) == {
        "b": "b"
    }
    assert parser.norm_list(["A", "b"], case="lower", to_set=True) == {"a", "b"}
    assert parser.norm_join(["A", "B"], sep="-") == "A-B"

    mapping_result = parser.norm_in("a", m.ConfigMap(root={"A": "1"}), case="lower")
    config_map_result = parser.norm_in("a", m.ConfigMap(root={"A": "x"}), case="lower")
    unsupported_result = parser.norm_in("a", cast("list[str]", object()), case="lower")
    assert mapping_result is True
    assert config_map_result is True
    assert unsupported_result is False

    original_norm_list = FlextUtilitiesParser.norm_list
    monkeypatch.setattr(
        FlextUtilitiesParser,
        "norm_list",
        staticmethod(lambda *_args, **_kwargs: {"k": "v"}),
    )
    try:
        assert parser.norm_in("v", ["x"], case="lower") is True
    finally:
        monkeypatch.setattr(FlextUtilitiesParser, "norm_list", original_norm_list)


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
    attrs = parser._extract_key_from_attributes(type("Obj", (), {"id": "x1"})())
    assert mapped == "n1"
    assert attrs == "x1"

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


def test_parser_remaining_branch_paths(monkeypatch) -> None:
    parser = u.Parser()

    assert parser._coerce_to_float([]) is None
    assert (
        parser._coerce_primitive("x", cast("type[int | float | str | bool]", object))
        is None
    )

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

    assert (
        parser._try_coerce_to_primitive(
            "x",
            cast("type[int | float | str | bool]", object),
        )
        is None
    )

    monkeypatch.setattr(
        FlextUtilitiesParser,
        "_coerce_to_int",
        staticmethod(lambda _v: (_ for _ in ()).throw(ValueError("boom"))),
    )
    failed_int = parser._parse_try_primitive("x", int, 1, None, "field: ")
    assert failed_int is not None and failed_int.is_failure

    monkeypatch.setattr(
        FlextUtilitiesParser,
        "_coerce_to_str",
        staticmethod(lambda _v: (_ for _ in ()).throw(TypeError("boom"))),
    )
    failed_str = parser._parse_try_primitive("x", str, "d", None, "field: ")
    assert failed_str is not None and failed_str.is_failure

    assert parser.convert("x", bool, cast("bool", cast("object", "d"))) == "d"
    assert parser._convert_to_int(5, default=7) == 5
    assert (
        parser._convert_to_float(cast("t.GeneralValueType", object()), default=1.5)
        == 1.5
    )
    assert parser._convert_fallback("x", str, "d") == "d"

    assert (
        parser.norm_in(
            "a",
            cast("list[str]", cast("object", {"A": "1"})),
            case="lower",
        )
        is True
    )
