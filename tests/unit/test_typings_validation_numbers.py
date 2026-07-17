"""Behavioral tests for numeric/string constrained validation typings.

Every assertion targets the observable public contract of the constrained
type aliases exposed through ``t.*``: a ``m.TypeAdapter`` either returns the
validated value unchanged (accept path) or raises ``c.ValidationError``
(reject path). No implementation internals are inspected.
"""

from __future__ import annotations

import pytest
from flext_tests import tm

from tests.constants import c
from tests.models import m
from tests.typings import p, t


class TestsFlextCoreTypingsValidationNumbers:
    """Public validation behavior of constrained scalar typings."""

    @pytest.mark.parametrize(
        ("alias", "value"),
        [
            (t.NonEmptyStr, "hello"),
            (t.NonEmptyStr, "x" * 255),
            (t.BoundedStr, "valid"),
            (t.BoundedStr, "a"),
            (t.BoundedStr, "y" * 255),
            (t.PositiveInt, 1),
            (t.PositiveInt, 42),
            (t.NonNegativeInt, 0),
            (t.NonNegativeInt, 100),
            (t.PortNumber, 1),
            (t.PortNumber, 8080),
            (t.PortNumber, 65535),
            (t.RetryCount, 0),
            (t.RetryCount, 10),
            (t.WorkerCount, 1),
            (t.WorkerCount, 100),
            (t.HttpStatusCode, 100),
            (t.HttpStatusCode, 200),
            (t.HttpStatusCode, 404),
            (t.HttpStatusCode, 599),
        ],
    )
    def test_accepts_valid_value_returns_input_unchanged(
        self, alias: type[str | int], value: str | int
    ) -> None:
        """A value inside the constraint validates to itself unchanged."""
        adapter: p.TypeAdapter[str | int] = m.TypeAdapter(alias)

        result = adapter.validate_python(value)

        tm.that(result, eq=value)

    @pytest.mark.parametrize(
        ("alias", "value"),
        [
            (t.NonEmptyStr, ""),
            (t.BoundedStr, ""),
            (t.BoundedStr, "x" * 256),
            (t.PositiveInt, 0),
            (t.PositiveInt, -1),
            (t.NonNegativeInt, -1),
            (t.PortNumber, 0),
            (t.PortNumber, 65536),
            (t.RetryCount, -1),
            (t.RetryCount, 11),
            (t.WorkerCount, 0),
            (t.WorkerCount, 101),
            (t.HttpStatusCode, 99),
            (t.HttpStatusCode, 600),
        ],
    )
    def test_rejects_out_of_bound_value_raises_validation_error(
        self, alias: type[str | int], value: str | int
    ) -> None:
        """A value outside the constraint raises the public ValidationError."""
        adapter: p.TypeAdapter[str | int] = m.TypeAdapter(alias)

        with pytest.raises(c.ValidationError):
            adapter.validate_python(value)

    @pytest.mark.parametrize(
        "alias", [t.PositiveInt, t.NonNegativeInt, t.PortNumber, t.HttpStatusCode]
    )
    def test_validation_is_idempotent_for_accepted_values(
        self, alias: type[int]
    ) -> None:
        """Re-validating an already-valid value yields the same result."""
        adapter: p.TypeAdapter[int] = m.TypeAdapter(alias)

        once = adapter.validate_python(100)
        twice = adapter.validate_python(once)

        tm.that(twice, eq=once)


class TestsFlextCoreTypingsStrippedStr:
    """Public contract of t.StrippedStr: strip surrounding whitespace, reject blank."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("hello", "hello"), ("  hello  ", "hello"), ("\tspaced\n", "spaced")],
    )
    def test_strips_surrounding_whitespace(self, value: str, expected: str) -> None:
        """A non-blank value is returned with surrounding whitespace removed."""
        adapter: p.TypeAdapter[str] = m.TypeAdapter(t.StrippedStr)

        result = adapter.validate_python(value)

        tm.that(result, eq=expected)

    @pytest.mark.parametrize("value", ["", "   ", "\t\n"])
    def test_rejects_blank_or_whitespace_only(self, value: str) -> None:
        """An empty or whitespace-only value raises the public ValidationError."""
        adapter: p.TypeAdapter[str] = m.TypeAdapter(t.StrippedStr)

        with pytest.raises(c.ValidationError):
            adapter.validate_python(value)
