"""Coverage tests for current utilities parser APIs."""

from __future__ import annotations

from enum import StrEnum, unique
from typing import cast, override

import pytest
from flext_tests import tm

from flext_core import FlextUtilitiesParser, r
from tests import TestUnitModels, t, u


class TestUtilitiesParserFullCoverage:
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

    @unique
    class _Status(StrEnum):
        ACTIVE = "active"
        INACTIVE = "inactive"

    @staticmethod
    def _raise_value_error_float(_value: t.Scalar) -> r[float]:
        msg = "boom"
        raise ValueError(msg)

    @staticmethod
    def _raise_type_error_bool(_value: t.Scalar) -> r[bool]:
        msg = "boom"
        raise TypeError(msg)

    @staticmethod
    def _raise_value_error_int(_value: t.Scalar) -> r[int]:
        msg = "boom"
        raise ValueError(msg)

    @staticmethod
    def _raise_type_error_str(_value: t.Scalar) -> r[str]:
        msg = "boom"
        raise TypeError(msg)

    def test_parser_parse_helpers_and_primitive_coercion_branches(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        parser = u()
        tm.that(parser._parse_normalize_str(123, case="lower"), eq="123")
        tm.that(parser._parse_normalize_str("abc", case="upper"), eq="ABC")
        tm.that(parser._parse_normalize_str("abc", case="none"), eq="abc")
        tm.that(
            parser._parse_result_error(r[int].ok(1), default="fallback"),
            eq="fallback",
        )
        model_result = parser._parse_model(
            cast("t.NormalizedValue", {"name": "ok", "count": 2, "payload": "obj"}),
            TestUnitModels._Model,
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
            self._Status,
            case_insensitive=True,
            default=None,
            default_factory=None,
            field_prefix="",
        )
        enum_value = parser._parse_try_enum(
            "inactive",
            self._Status,
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
            staticmethod(self._raise_value_error_float),
        )
        failed_float = parser._parse_try_primitive("x", float, 1.2, None, "field: ")
        tm.fail(failed_float)
        monkeypatch.setattr(
            FlextUtilitiesParser,
            "_coerce_to_bool",
            staticmethod(self._raise_type_error_bool),
        )
        failed_bool = parser._parse_try_primitive("x", bool, True, None, "field: ")
        tm.fail(failed_bool)

        class _DefaultBox:
            pass

        parsed = parser.parse(
            "x",
            int,
            default=cast("int", cast("t.NormalizedValue", _DefaultBox())),
            default_factory=lambda: 9,
        )
        tm.ok(parsed)

    def test_parser_convert_and_norm_branches(self) -> None:
        parser = u()
        tm.that(parser.convert("10", int, 0), eq=10)
        tm.that(parser._convert_to_int(True, default=7), eq=7)
        parsed_float = parser._convert_to_float(1.5, default=0.0)
        tm.that(abs(parsed_float - 1.5), lt=1e-09)
        tm.that(parser._convert_to_str("x", default=""), eq="x")
        tm.that(parser._convert_to_str(None, default="d"), eq="d")
        tm.that(
            parser._convert_to_str(
                cast("t.NormalizedValue", self._BadStr()),
                default="d",
            ),
            eq="d",
        )
        tm.that(parser._convert_to_bool(True, default=False), eq=True)
        tm.that(
            parser._convert_to_bool(
                cast("t.NormalizedValue", self._BadConv()),
                default=True,
            ),
            eq=True,
        )
        tm.that(parser.norm_str("abc"), eq="abc")
        tm.that(parser.norm_join(["A", "B"], sep="-"), eq="A-B")
        mapping_result = parser.norm_in(
            "a",
            t.ConfigMap(root={"A": "1"}),
            case="lower",
        )
        config_map_result = parser.norm_in(
            "a",
            t.ConfigMap(root={"A": "x"}),
            case="lower",
        )

        class _NotSequence:
            pass

        not_seq_result = parser.norm_in(
            "a", cast("t.StrSequence", _NotSequence()), case="lower"
        )
        tm.that(not_seq_result, eq=False)
        tm.that(mapping_result, eq=True)
        tm.that(config_map_result, eq=True)

    def test_parser_internal_helpers_parse_with_default(self) -> None:
        parser = u()
        tm.that(parser._parse_with_default(None, lambda: 3, "err").value, eq=3)
        tm.that(parser._parse_with_default(None, None, "err").is_failure, eq=True)

    def test_parser_remaining_branch_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
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
            staticmethod(self._raise_value_error_int),
        )
        failed_int = parser._parse_try_primitive("x", int, 1, None, "field: ")
        tm.fail(failed_int)
        monkeypatch.setattr(
            FlextUtilitiesParser,
            "_coerce_to_str",
            staticmethod(self._raise_type_error_str),
        )
        failed_str = parser._parse_try_primitive("x", str, "d", None, "field: ")
        tm.fail(failed_str)
        tm.that(
            parser.convert("x", bool, cast("bool", cast("t.NormalizedValue", "d"))),
            eq="d",
        )
        tm.that(parser._convert_to_int(5, default=7), eq=5)
        tm.that(
            parser._convert_to_float(
                cast("t.NormalizedValue", "normalized"),
                default=1.5,
            ),
            eq=1.5,
        )
        tm.that(
            abs(
                parser._convert_to_float(
                    cast("t.NormalizedValue", "normalized"),
                    default=1.5,
                )
                - 1.5,
            ),
            lt=1e-09,
        )
        tm.that(
            parser.norm_in("a", t.ConfigMap(root={"A": "1"}), case="lower"),
            eq=True,
        )
