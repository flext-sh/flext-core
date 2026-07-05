"""Behavioral tests for the stable public FlextUtilities helper surface.

Every test asserts an observable contract of a public helper (return value for a
given input) — never an internal collaborator, private attribute, or call spy.
"""

from __future__ import annotations

import uuid

import pytest

from tests.utilities import u

type MatchValue = str | int | bool | list[int] | dict[str, int]
type ConvValue = str | float | int | bool | None


class TestsFlextCoreUtilities:
    """Public-contract tests for the shared utilities facade."""

    @pytest.mark.parametrize(
        ("value", "type_name", "expected"),
        [
            ("abc", "str", True),
            (1, "str", False),
            (1, "int", True),
            (True, "bool", True),
            ([1], "list", True),
            ({}, "dict", True),
            ("abc", "int", False),
        ],
    )
    def test_matches_type_reports_runtime_type_membership(
        self, value: MatchValue, type_name: str, expected: bool
    ) -> None:
        assert u.matches_type(value, type_name) is expected

    @pytest.mark.parametrize("kind", ["ulid", "uuid4", "uuid", "id", "hex", "short"])
    def test_generate_returns_non_empty_string(self, kind: str) -> None:
        generated = u.generate(kind)
        assert isinstance(generated, str)
        assert generated

    def test_generate_produces_distinct_values_across_calls(self) -> None:
        assert u.generate("uuid4") != u.generate("uuid4")

    def test_generate_uuid4_is_a_valid_uuid(self) -> None:
        # Parsing through uuid.UUID proves the output honors the v4 format.
        parsed = uuid.UUID(u.generate("uuid4"))
        assert parsed.version == 4

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("true", True),
            ("false", True),
            ("non-empty", True),
            (True, True),
            (False, False),
            ("", False),
            (None, False),
        ],
    )
    def test_to_bool_follows_truthiness(
        self, value: ConvValue, expected: bool
    ) -> None:
        assert u.to_bool(value) is expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("5", 5),
            (3.9, 3),
            ("x", 0),
            ("", 0),
            (None, 0),
        ],
    )
    def test_to_int_parses_or_defaults_to_zero(
        self, value: ConvValue, expected: int
    ) -> None:
        assert u.to_int(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("1.5", 1.5),
            ("2", 2.0),
            ("x", 0.0),
        ],
    )
    def test_to_float_parses_or_defaults_to_zero(
        self, value: ConvValue, expected: float
    ) -> None:
        assert u.to_float(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (5, 5),
            (0, 0),
            (-1, 0),
        ],
    )
    def test_to_positive_int_clamps_negatives_to_zero(
        self, value: int, expected: int
    ) -> None:
        assert u.to_positive_int(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (5, "5"),
            (None, ""),
        ],
    )
    def test_to_str_stringifies_with_empty_default(
        self, value: ConvValue, expected: str
    ) -> None:
        assert u.to_str(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("x", "x"),
            (None, None),
        ],
    )
    def test_to_optional_str_preserves_none(
        self, value: str | None, expected: str | None
    ) -> None:
        assert u.to_optional_str(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("a", ["a"]),
            ("a,b", ["a,b"]),
            (["x", "y"], ["x", "y"]),
        ],
    )
    def test_to_str_list_wraps_scalars_and_preserves_lists(
        self, value: str | list[str], expected: list[str]
    ) -> None:
        assert u.to_str_list(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("foo_bar", "FooBar"),
            ("foo-bar", "FooBar"),
            ("fooBar", "FooBar"),
            ("", ""),
        ],
    )
    def test_pascalize_converts_delimited_tokens(
        self, value: str, expected: str
    ) -> None:
        assert u.pascalize(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("Aa9!@ x", "aa9x"),
            ("abc", "abc"),
            ("!!!", ""),
        ],
    )
    def test_normalize_alnum_lowercases_and_drops_non_alnum(
        self, value: str, expected: str
    ) -> None:
        assert u.normalize_alnum(value) == expected

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ({"a": 1}, {"a": 1}, True),
            ({"a": 1}, {"a": 2}, False),
            ({}, {}, True),
        ],
    )
    def test_deep_eq_compares_mapping_contents(
        self, left: dict[str, int], right: dict[str, int], expected: bool
    ) -> None:
        assert u.deep_eq(left, right) is expected

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            (1, 2, True),
            (1, "a", False),
        ],
    )
    def test_same_type_compares_runtime_types(
        self, left: int | str, right: int | str, expected: bool
    ) -> None:
        assert u.same_type(left, right) is expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1, "int"),
            ("a", "str"),
            ([], "list"),
        ],
    )
    def test_type_name_reports_runtime_type_name(
        self, value: int | str | list[int], expected: str
    ) -> None:
        assert u.type_name(value) == expected
