"""Scalar validation type tests."""

from __future__ import annotations

import math

import pytest
from flext_tests import tm

from tests.constants import c
from tests.models import m
from tests.typings import t


class TestsFlextTypesValidationScalars:
    def test_batch_size_valid(self) -> None:
        """BatchSize accepts 1-10000."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.BatchSize)
        tm.that(adapter.validate_python(1), eq=1)
        tm.that(adapter.validate_python(10000), eq=10000)

    def test_batch_size_rejects_zero(self) -> None:
        """BatchSize rejects 0."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.BatchSize)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(0)

    def test_max_length_valid(self) -> None:
        """MaxLength accepts positive integers."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.MaxLength)
        tm.that(adapter.validate_python(1), eq=1)
        tm.that(adapter.validate_python(9999), eq=9999)

    def test_max_length_rejects_zero(self) -> None:
        """MaxLength rejects 0."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.MaxLength)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(0)

    def test_positive_float_valid(self) -> None:
        """PositiveFloat accepts positive floats."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.PositiveFloat)
        tm.that(adapter.validate_python(0.1), eq=0.1)

    def test_positive_float_rejects_zero(self) -> None:
        """PositiveFloat rejects 0.0."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.PositiveFloat)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(0.0)

    def test_non_negative_float_valid(self) -> None:
        """NonNegativeFloat accepts 0.0 and positive."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.NonNegativeFloat)
        tm.that(adapter.validate_python(0.0), eq=0.0)
        tm.that(adapter.validate_python(math.pi), eq=math.pi)

    def test_non_negative_float_rejects_negative(self) -> None:
        """NonNegativeFloat rejects negative floats."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.NonNegativeFloat)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(-0.1)

    def test_positive_timeout_valid(self) -> None:
        """PositiveTimeout accepts 0 < x <= 300."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.PositiveTimeout)
        tm.that(adapter.validate_python(1.0), eq=1.0)
        tm.that(adapter.validate_python(300.0), eq=300.0)

    def test_positive_timeout_rejects_zero(self) -> None:
        """PositiveTimeout rejects 0.0."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.PositiveTimeout)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(0.0)

    def test_positive_timeout_rejects_too_high(self) -> None:
        """PositiveTimeout rejects > 300."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.PositiveTimeout)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(300.1)

    def test_backoff_multiplier_valid(self) -> None:
        """BackoffMultiplier accepts >= 1.0."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.BackoffMultiplier)
        tm.that(adapter.validate_python(1.0), eq=1.0)
        tm.that(adapter.validate_python(2.5), eq=2.5)

    def test_backoff_multiplier_rejects_below_one(self) -> None:
        """BackoffMultiplier rejects < 1.0."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.BackoffMultiplier)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(0.9)

    def test_percentage_valid(self) -> None:
        """Percentage accepts 0.0-100.0."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.Percentage)
        tm.that(adapter.validate_python(0.0), eq=0.0)
        tm.that(adapter.validate_python(50.0), eq=50.0)
        tm.that(adapter.validate_python(100.0), eq=100.0)

    def test_percentage_rejects_over_100(self) -> None:
        """Percentage rejects > 100."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.Percentage)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(100.1)

    def test_decimal_fraction_valid(self) -> None:
        """DecimalFraction accepts 0.0-1.0."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.DecimalFraction)
        tm.that(adapter.validate_python(0.0), eq=0.0)
        tm.that(adapter.validate_python(0.5), eq=0.5)
        tm.that(adapter.validate_python(1.0), eq=1.0)

    def test_decimal_fraction_rejects_over_one(self) -> None:
        """DecimalFraction rejects > 1.0."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.DecimalFraction)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(1.1)

    def test_hostname_str_valid(self) -> None:
        """HostnameStr accepts non-empty strings."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.HostnameStr)
        result = adapter.validate_python(c.LOCALHOST)
        tm.that(result, eq=c.LOCALHOST)

    def test_hostname_str_rejects_empty(self) -> None:
        """HostnameStr rejects empty strings."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.HostnameStr)
        with pytest.raises(c.ValidationError):
            adapter.validate_python("")

    def test_uri_string_valid(self) -> None:
        """UriString accepts non-empty strings."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.UriString)
        result = adapter.validate_python("https://example.com")
        tm.that(result, eq="https://example.com")

    def test_uri_string_rejects_empty(self) -> None:
        """UriString rejects empty strings."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.UriString)
        with pytest.raises(c.ValidationError):
            adapter.validate_python("")

    def test_timestamp_str_valid(self) -> None:
        """TimestampStr accepts non-empty strings."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(t.TimestampStr)
        result = adapter.validate_python("2025-01-01T00:00:00Z")
        tm.that(result, eq="2025-01-01T00:00:00Z")
