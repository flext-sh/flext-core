"""Behavioral tests for FlextUtilities value-conversion and text helpers.

Tested module: flext_core._utilities.conversion (FlextUtilitiesConversion) and
flext_core._utilities.text (FlextUtilitiesText), exposed via the flat runtime
facade ``u`` (FlextUtilities).

Scope: public contract only. Each test asserts an observable return value or a
raised exception from the public API surface - never a private attribute,
internal collaborator, or implementation structure.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from tests import t, u


class TestsFlextCoreUtilitiesCoverage:
    """Behavioral contract of the FlextUtilities conversion and text helpers."""

    @pytest.mark.parametrize(
        ("value", "default", "expected"),
        [
            ("42", 0, 42),
            ("3.9", 0, 3),
            (7, 0, 7),
            (2.5, 0, 2),
            (None, 9, 9),
            (True, 5, 5),
            (False, 5, 5),
            ("not-a-number", 4, 4),
        ],
    )
    def test_to_int_returns_int_or_default(
        self, value: t.JsonPayload | None, default: int, expected: int
    ) -> None:
        """to_int coerces numeric inputs and falls back to default otherwise."""
        assert u.to_int(value, default=default) == expected

    @pytest.mark.parametrize(
        ("value", "default", "expected"),
        [
            ("1.5", 0.0, 1.5),
            (3, 0.0, 3.0),
            (None, 2.0, 2.0),
            (True, 2.0, 2.0),
            ("nan-text", 8.0, 8.0),
        ],
    )
    def test_to_float_returns_float_or_default(
        self, value: t.JsonPayload | None, default: float, expected: float
    ) -> None:
        """to_float coerces numeric inputs and falls back to default otherwise."""
        assert u.to_float(value, default=default) == expected

    @pytest.mark.parametrize(
        ("value", "default", "expected"),
        [
            (1, False, True),
            (0, True, False),
            ("", False, False),
            ("text", False, True),
            (None, True, True),
            (None, False, False),
        ],
    )
    def test_to_bool_returns_truthiness_or_default(
        self, value: t.JsonPayload | None, default: bool, expected: bool
    ) -> None:
        """to_bool reflects truthiness and uses default only for None."""
        assert u.to_bool(value, default=default) is expected

    @pytest.mark.parametrize(
        ("value", "default", "expected"),
        [
            (5, 0, 5),
            (-3, 7, 7),
            (0, 7, 7),
            (4.0, 0, 4),
            (4.5, 1, 1),
            ("10", 0, 10),
            ("-10", 0, 0),
            (True, 9, 9),
            (None, 9, 9),
        ],
    )
    def test_to_positive_int_accepts_only_positive_values(
        self, value: t.JsonPayload | None, default: int, expected: int
    ) -> None:
        """to_positive_int returns strictly positive ints, else the default."""
        assert u.to_positive_int(value, default=default) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(None, ""), ("hello", "hello"), (3.0, "3"), (2.5, "2.50"), (10, "10")],
    )
    def test_to_str_formats_value(
        self, value: t.JsonPayload | None, expected: str
    ) -> None:
        """to_str renders integral floats without decimals and keeps strings."""
        assert u.to_str(value) == expected

    def test_to_str_uses_default_for_none(self) -> None:
        """to_str returns the supplied default when the value is None."""
        assert u.to_str(None, default="fallback") == "fallback"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("", None), ("value", "value"), (None, None), (123, None)],
    )
    def test_to_optional_str_only_returns_non_empty_strings(
        self, value: t.JsonPayload | None, expected: str | None
    ) -> None:
        """to_optional_str yields the string only when it is a non-empty str."""
        assert u.to_optional_str(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"), [("solo", ["solo"]), ([1, 2], ["1", "2"]), (None, [])]
    )
    def test_to_str_list_produces_list_of_strings(
        self, value: t.StrictValue | None, expected: list[str]
    ) -> None:
        """to_str_list normalizes scalars and sequences into lists of strings."""
        assert u.to_str_list(value) == expected

    def test_to_str_list_uses_default_for_none(self) -> None:
        """to_str_list returns the provided default when value is None."""
        assert u.to_str_list(None, default=["x"]) == ["x"]

    @pytest.mark.parametrize(
        ("values", "separator", "case", "expected"),
        [
            (["A", "b"], " ", "lower", "a b"),
            (["A", "b"], " ", "upper", "A B"),
            (["a", "b", "c"], "-", None, "a-b-c"),
            ([], " ", None, ""),
        ],
    )
    def test_join_concatenates_with_separator_and_case(
        self, values: list[str], separator: str, case: str | None, expected: str
    ) -> None:
        """Join applies the separator and optional case transform."""
        assert u.join(values, separator=separator, case=case) == expected

    @pytest.mark.parametrize(
        ("value", "case", "expected"),
        [
            ("Hi", "upper", "HI"),
            ("Hi", "lower", "hi"),
            ("Hi", None, "Hi"),
            (5, None, "5"),
        ],
    )
    def test_normalize_stringifies_with_case(
        self, value: t.StrictValue, case: str | None, expected: str
    ) -> None:
        """Normalize converts to string then applies the optional case."""
        assert u.normalize(value, case=case) == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("My App", "my-app"),
            ("My_App_Name", "my-app-name"),
            ("Already-lower", "already-lower"),
            ("Mixed Case_Id", "mixed-case-id"),
        ],
    )
    def test_format_app_id_normalizes_to_hyphenated_lowercase(
        self, name: str, expected: str
    ) -> None:
        """format_app_id lowercases and replaces spaces/underscores with hyphens."""
        assert u.format_app_id(name) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("  hi  ", "hi"), ("value", "value"), ("\ttrimmed\n", "trimmed")],
    )
    def test_safe_string_strips_and_returns_non_empty(
        self, text: str, expected: str
    ) -> None:
        """safe_string trims surrounding whitespace and returns the content."""
        assert u.safe_string(text) == expected

    @pytest.mark.parametrize("text", [None, "", "   ", "\t\n"])
    def test_safe_string_rejects_empty_input(self, text: str | None) -> None:
        """safe_string raises ValueError for None, empty, or whitespace input."""
        with pytest.raises(ValueError, match="Text"):
            u.safe_string(text)


__all__: list[str] = ["TestsFlextCoreUtilitiesCoverage"]
