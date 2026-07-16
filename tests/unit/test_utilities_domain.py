"""Behavioral contract tests for FlextUtilitiesConversion (via the `u` facade).

These tests assert only the OBSERVABLE public behavior of the conversion
utilities: given an input, what value comes back. No private attributes, no
internal collaborators, no implementation spying.
"""

from __future__ import annotations

import pytest

from tests.utilities import u


class TestsFlextCoreUtilitiesDomain:
    """Public-contract behavior of the value-conversion utilities."""

    # ----------------------------------------------------------------- join
    @pytest.mark.parametrize(
        ("values", "separator", "case", "expected"),
        [
            (["a", "b"], " ", None, "a b"),
            (["a", "b"], " ", "lower", "a b"),
            (["A", "B"], " ", "lower", "a b"),
            (["a", "b"], " ", "upper", "A B"),
            (["a", "b"], "-", None, "a-b"),
            (["Mixed", "CASE"], ",", "lower", "mixed,case"),
            ([], " ", None, ""),
            (["solo"], " ", None, "solo"),
        ],
    )
    def test_join_produces_expected_string(
        self,
        values: list[str],
        separator: str,
        case: str | None,
        expected: str,
    ) -> None:
        assert u.join(values, separator=separator, case=case) == expected

    def test_join_empty_sequence_is_empty_string(self) -> None:
        assert u.join([]) == ""

    # ------------------------------------------------------------ normalize
    @pytest.mark.parametrize(
        ("value", "case", "expected"),
        [
            ("abc", None, "abc"),
            ("AbC", "lower", "abc"),
            ("AbC", "upper", "ABC"),
            (10, None, "10"),
            (10.0, None, "10"),
            (10.5, None, "10.50"),
            (True, None, "True"),
        ],
    )
    def test_normalize_returns_expected_string(
        self,
        value: str | float | bool,
        case: str | None,
        expected: str,
    ) -> None:
        assert u.normalize(value, case=case) == expected

    # --------------------------------------------------------------- to_str
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("hello", "hello"),
            (42, "42"),
            (42.0, "42"),
            (9.876, "9.88"),
            (None, ""),
        ],
    )
    def test_to_str_converts_value(
        self, value: str | float | None, expected: str
    ) -> None:
        assert u.to_str(value) == expected

    def test_to_str_none_uses_default(self) -> None:
        assert u.to_str(None, default="fallback") == "fallback"

    def test_to_str_present_value_ignores_default(self) -> None:
        assert u.to_str("real", default="fallback") == "real"

    # ---------------------------------------------------------- to_str_list
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("single", ["single"]),
            (["a", "b"], ["a", "b"]),
            ([1, 2], ["1", "2"]),
            (None, []),
        ],
    )
    def test_to_str_list_converts_value(
        self,
        value: str | list[str] | list[int] | None,
        expected: list[str],
    ) -> None:
        assert u.to_str_list(value) == expected

    def test_to_str_list_none_uses_default(self) -> None:
        assert u.to_str_list(None, default=["x"]) == ["x"]

    # --------------------------------------------------------------- to_int
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (5, 5),
            (5.9, 5),
            ("7", 7),
            ("7.8", 7),
            ("not-a-number", 0),
            (None, 0),
            (True, 0),
            (False, 0),
        ],
    )
    def test_to_int_converts_value(
        self, value: float | str | bool | None, expected: int
    ) -> None:
        assert u.to_int(value) == expected

    def test_to_int_invalid_uses_default(self) -> None:
        assert u.to_int("nope", default=99) == 99

    # ------------------------------------------------------------- to_float
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (5, 5.0),
            (5.5, 5.5),
            ("2.5", 2.5),
            ("bad", 0.0),
            (None, 0.0),
            (True, 0.0),
        ],
    )
    def test_to_float_converts_value(
        self, value: float | str | bool | None, expected: float
    ) -> None:
        assert u.to_float(value) == expected

    def test_to_float_invalid_uses_default(self) -> None:
        assert u.to_float("bad", default=1.5) == pytest.approx(1.5)

    # -------------------------------------------------------------- to_bool
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (True, True),
            (False, False),
            (1, True),
            (0, False),
            ("text", True),
            ("", False),
            (None, False),
        ],
    )
    def test_to_bool_converts_value(
        self, value: bool | int | str | None, expected: bool
    ) -> None:
        assert u.to_bool(value) is expected

    def test_to_bool_none_uses_default(self) -> None:
        assert u.to_bool(None, default=True) is True

    # ------------------------------------------------------ to_positive_int
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (5, 5),
            (0, 0),
            (-3, 0),
            (4.0, 4),
            (4.5, 0),
            ("6", 6),
            ("-6", 0),
            ("abc", 0),
            (None, 0),
            (True, 0),
        ],
    )
    def test_to_positive_int_rejects_non_positive(
        self, value: float | str | bool | None, expected: int
    ) -> None:
        assert u.to_positive_int(value) == expected

    def test_to_positive_int_non_positive_uses_default(self) -> None:
        assert u.to_positive_int(-1, default=10) == 10

    # ------------------------------------------------------ to_optional_str
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("value", "value"),
            ("", None),
            (None, None),
            (123, None),
        ],
    )
    def test_to_optional_str_returns_non_empty_string_only(
        self, value: str | int | None, expected: str | None
    ) -> None:
        assert u.to_optional_str(value) == expected
