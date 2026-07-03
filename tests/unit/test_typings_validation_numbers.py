"""Numeric validation type tests."""

from __future__ import annotations

import pytest
from flext_tests import tm

from tests.constants import c
from tests.models import m
from tests.typings import t


class TestsFlextTypesValidationNumbers:
    def test_non_empty_str_valid(self) -> None:
        """NonEmptyStr accepts non-empty strings."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.NonEmptyStr)
        result = adapter.validate_python("hello")
        tm.that(result, eq="hello")

    def test_non_empty_str_rejects_empty(self) -> None:
        """NonEmptyStr rejects empty strings."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.NonEmptyStr)
        with pytest.raises(c.ValidationError):
            adapter.validate_python("")

    def test_bounded_str_valid(self) -> None:
        """BoundedStr accepts strings between 1-255 chars."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.BoundedStr)
        result = adapter.validate_python("valid")
        tm.that(result, eq="valid")

    def test_bounded_str_rejects_too_long(self) -> None:
        """BoundedStr rejects strings longer than 255 chars."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.BoundedStr)
        with pytest.raises(c.ValidationError):
            adapter.validate_python("x" * 256)

    def test_bounded_str_rejects_empty(self) -> None:
        """BoundedStr rejects empty strings."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.BoundedStr)
        with pytest.raises(c.ValidationError):
            adapter.validate_python("")

    def test_positive_int_valid(self) -> None:
        """PositiveInt accepts positive integers."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.PositiveInt)
        result = adapter.validate_python(42)
        tm.that(result, eq=42)

    def test_positive_int_rejects_zero(self) -> None:
        """PositiveInt rejects zero."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.PositiveInt)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(0)

    def test_positive_int_rejects_negative(self) -> None:
        """PositiveInt rejects negative numbers."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.PositiveInt)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(-1)

    def test_non_negative_int_valid(self) -> None:
        """NonNegativeInt accepts zero and positive."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.NonNegativeInt)
        tm.that(adapter.validate_python(0), eq=0)
        tm.that(adapter.validate_python(100), eq=100)

    def test_non_negative_int_rejects_negative(self) -> None:
        """NonNegativeInt rejects negative numbers."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.NonNegativeInt)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(-1)

    def test_port_number_valid(self) -> None:
        """PortNumber accepts 1-65535."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.PortNumber)
        tm.that(adapter.validate_python(1), eq=1)
        tm.that(adapter.validate_python(8080), eq=8080)
        tm.that(adapter.validate_python(65535), eq=65535)

    def test_port_number_rejects_zero(self) -> None:
        """PortNumber rejects 0."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.PortNumber)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(0)

    def test_port_number_rejects_too_high(self) -> None:
        """PortNumber rejects values above 65535."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.PortNumber)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(65536)

    def test_retry_count_valid(self) -> None:
        """RetryCount accepts 0-10."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.RetryCount)
        tm.that(adapter.validate_python(0), eq=0)
        tm.that(adapter.validate_python(10), eq=10)

    def test_retry_count_rejects_too_high(self) -> None:
        """RetryCount rejects values above 10."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.RetryCount)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(11)

    def test_worker_count_valid(self) -> None:
        """WorkerCount accepts 1-100."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.WorkerCount)
        tm.that(adapter.validate_python(1), eq=1)
        tm.that(adapter.validate_python(100), eq=100)

    def test_worker_count_rejects_zero(self) -> None:
        """WorkerCount rejects 0."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.WorkerCount)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(0)

    def test_http_status_code_valid(self) -> None:
        """HttpStatusCode accepts 100-599."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.HttpStatusCode)
        tm.that(adapter.validate_python(200), eq=200)
        tm.that(adapter.validate_python(404), eq=404)
        tm.that(adapter.validate_python(599), eq=599)

    def test_http_status_code_rejects_invalid(self) -> None:
        """HttpStatusCode rejects values outside 100-599."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.HttpStatusCode)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(99)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(600)
