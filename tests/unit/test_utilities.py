"""Behavioral tests for the stable public FlextUtilities helper surface.

Every test asserts an observable contract of a public helper (return value for a
given input) — never an internal collaborator, private attribute, or call spy.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import uuid

import pytest

from tests import c, t, tm, u

type _ConvValue = str | float | int | bool | None


class TestsFlextCoreUtilities:
    """Public-contract tests for the shared utilities facade."""

    @staticmethod
    @pytest.mark.parametrize(
        ("value", "type_name", "expected"),
        [
            ("abc", "str", True),
            (1, "str", False),
            (1, "int", True),
            (True, "bool", True),
            ([1], c.Action.LIST, True),
            ({}, "dict", True),
            ("abc", "int", False),
        ],
    )
    def test_matches_type_reports_runtime_type_membership(
        value: t.JsonValue,
        type_name: str,
        *,
        expected: bool,
    ) -> None:
        """Test matches type reports runtime type membership."""
        tm.that(u.matches_type(value, type_name) is expected, truth=True)

    @staticmethod
    @pytest.mark.parametrize("kind", ["ulid", "uuid4", "uuid", "id", "hex", "short"])
    def test_generate_returns_non_empty_string(kind: str) -> None:
        """Test generate returns non empty string."""
        generated = u.generate(kind)
        tm.that(generated, is_=str)
        tm.that(generated, truth=True)

    @staticmethod
    def test_generate_produces_distinct_values_across_calls() -> None:
        """Test generate produces distinct values across calls."""
        tm.that(u.generate("uuid4"), ne=u.generate("uuid4"))

    @staticmethod
    def test_generate_uuid4_is_a_valid_uuid() -> None:
        # Parsing through uuid.UUID proves the output honors the v4 format.
        """Test generate uuid4 is a valid uuid."""
        parsed = uuid.UUID(u.generate("uuid4"))
        tm.that(parsed.version, eq=4)

    @staticmethod
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (c.ParserBooleanToken.TRUE, True),
            (c.ParserBooleanToken.FALSE, True),
            ("non-empty", True),
            (True, True),
            (False, False),
            ("", False),
            (None, False),
        ],
    )
    def test_to_bool_follows_truthiness(
        value: _ConvValue,
        *,
        expected: bool,
    ) -> None:
        """Test to bool follows truthiness."""
        tm.that(u.to_bool(value) is expected, truth=True)

    @staticmethod
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("5", 5), (3.9, 3), ("x", 0), ("", 0), (None, 0)],
    )
    def test_to_int_parses_or_defaults_to_zero(
        value: _ConvValue,
        expected: int,
    ) -> None:
        """Test to int parses or defaults to zero."""
        tm.that(u.to_int(value), eq=expected)

    @staticmethod
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("1.5", 1.5), ("2", 2.0), ("x", 0.0)],
    )
    def test_to_float_parses_or_defaults_to_zero(
        value: _ConvValue,
        expected: float,
    ) -> None:
        """Test to float parses or defaults to zero."""
        tm.that(u.to_float(value), eq=expected)

    @staticmethod
    @pytest.mark.parametrize(("value", "expected"), [(5, 5), (0, 0), (-1, 0)])
    def test_to_positive_int_clamps_negatives_to_zero(
        value: int,
        expected: int,
    ) -> None:
        """Test to positive int clamps negatives to zero."""
        tm.that(u.to_positive_int(value), eq=expected)

    @staticmethod
    @pytest.mark.parametrize(("value", "expected"), [(5, "5"), (None, "")])
    def test_to_str_stringifies_with_empty_default(
        value: _ConvValue,
        expected: str,
    ) -> None:
        """Test to str stringifies with empty default."""
        tm.that(u.to_str(value), eq=expected)

    @staticmethod
    @pytest.mark.parametrize(("value", "expected"), [("x", "x"), (None, None)])
    def test_to_optional_str_preserves_none(
        value: str | None,
        expected: str | None,
    ) -> None:
        """Test to optional str preserves none."""
        tm.that(u.to_optional_str(value), eq=expected)

    @staticmethod
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("a", ["a"]), ("a,b", ["a,b"]), (["x", "y"], ["x", "y"])],
    )
    def test_to_str_list_wraps_scalars_and_preserves_lists(
        value: str | list[str],
        expected: list[str],
    ) -> None:
        """Test to str list wraps scalars and preserves lists."""
        tm.that(u.to_str_list(value), eq=expected)

    @staticmethod
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("Aa9!@ x", "aa9x"), ("abc", "abc"), ("!!!", "")],
    )
    def test_normalize_alnum_lowercases_and_drops_non_alnum(
        value: str,
        expected: str,
    ) -> None:
        """Test normalize alnum lowercases and drops non alnum."""
        tm.that(u.normalize_alnum(value), eq=expected)

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [({"a": 1}, {"a": 1}, True), ({"a": 1}, {"a": 2}, False), ({}, {}, True)],
    )
    def test_deep_eq_compares_mapping_contents(
        left: t.MappingKV[str, int],
        right: t.MappingKV[str, int],
        *,
        expected: bool,
    ) -> None:
        """Test deep eq compares mapping contents."""
        tm.that(u.deep_eq(left, right) is expected, truth=True)

    @staticmethod
    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [(1, 2, True), (1, "a", False)],
    )
    def test_same_type_compares_runtime_types(
        left: int | str,
        right: int | str,
        *,
        expected: bool,
    ) -> None:
        """Test same type compares runtime types."""
        tm.that(u.same_type(left, right) is expected, truth=True)

    @staticmethod
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(1, "int"), ("a", "str"), ([], c.Action.LIST)],
    )
    def test_type_name_reports_runtime_type_name(
        value: int | str | list[int],
        expected: str,
    ) -> None:
        """Test type name reports runtime type name."""
        tm.that(u.type_name(value), eq=expected)
