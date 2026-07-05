"""Behavioral tests for constrained scalar typing aliases.

Each alias is a public ``Annotated`` contract exposed through the ``t`` facade.
Tests assert the observable validation behavior of ``TypeAdapter`` against the
public boundary rules (accepted values round-trip unchanged; out-of-range values
raise the public ``ValidationError``) -- never how the constraint is implemented.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math

import pytest
from flext_tests import tm

from tests.constants import c
from tests.models import m
from tests.typings import t


class TestsFlextCoreTypingsValidationScalars:
    """Public validation contract of constrained scalar aliases."""

    # -- integer constraints -------------------------------------------------

    @pytest.mark.parametrize("value", [1, 2, 5000, 9999, 10000])
    def test_batch_size_accepts_in_range(self, value: int) -> None:
        """BatchSize round-trips integers within its inclusive 1..10000 range."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.BatchSize)
        tm.that(adapter.validate_python(value), eq=value)

    @pytest.mark.parametrize("value", [0, -1, 10001, 20000])
    def test_batch_size_rejects_out_of_range(self, value: int) -> None:
        """BatchSize rejects integers below 1 or above 10000."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.BatchSize)
        with pytest.raises(c.ValidationError) as exc:
            adapter.validate_python(value)
        tm.that(exc.value.error_count(), gte=1)

    @pytest.mark.parametrize("value", [1, 2, 9999, 1_000_000])
    def test_max_length_accepts_positive(self, value: int) -> None:
        """MaxLength round-trips any integer >= 1."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.MaxLength)
        tm.that(adapter.validate_python(value), eq=value)

    @pytest.mark.parametrize("value", [0, -1, -100])
    def test_max_length_rejects_non_positive(self, value: int) -> None:
        """MaxLength rejects integers below 1."""
        adapter: m.TypeAdapter[int] = m.TypeAdapter(t.MaxLength)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(value)

    # -- float constraints ---------------------------------------------------

    @pytest.mark.parametrize("value", [0.1, 1.0, math.pi, 1e6])
    def test_positive_float_accepts_positive(self, value: float) -> None:
        """PositiveFloat round-trips strictly positive floats."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.PositiveFloat)
        tm.that(adapter.validate_python(value), eq=value)

    @pytest.mark.parametrize("value", [0.0, -0.1, -100.0])
    def test_positive_float_rejects_non_positive(self, value: float) -> None:
        """PositiveFloat rejects zero and negative floats."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.PositiveFloat)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(value)

    @pytest.mark.parametrize("value", [0.0, 0.5, math.pi, 1e9])
    def test_non_negative_float_accepts_zero_and_positive(
        self, value: float
    ) -> None:
        """NonNegativeFloat round-trips zero and positive floats."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.NonNegativeFloat)
        tm.that(adapter.validate_python(value), eq=value)

    @pytest.mark.parametrize("value", [-0.1, -1.0, -1e6])
    def test_non_negative_float_rejects_negative(self, value: float) -> None:
        """NonNegativeFloat rejects any negative float."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.NonNegativeFloat)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(value)

    @pytest.mark.parametrize("value", [0.001, 1.0, 150.0, 300.0])
    def test_positive_timeout_accepts_in_range(self, value: float) -> None:
        """PositiveTimeout round-trips values in the exclusive-lower 0..300 range."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.PositiveTimeout)
        tm.that(adapter.validate_python(value), eq=value)

    @pytest.mark.parametrize("value", [0.0, -1.0, 300.1, 1000.0])
    def test_positive_timeout_rejects_out_of_range(self, value: float) -> None:
        """PositiveTimeout rejects zero, negatives, and values above 300."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.PositiveTimeout)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(value)

    @pytest.mark.parametrize("value", [1.0, 1.5, 2.5, 100.0])
    def test_backoff_multiplier_accepts_at_or_above_one(
        self, value: float
    ) -> None:
        """BackoffMultiplier round-trips floats >= 1.0 (lower bound inclusive)."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.BackoffMultiplier)
        tm.that(adapter.validate_python(value), eq=value)

    @pytest.mark.parametrize("value", [0.9, 0.0, -1.0])
    def test_backoff_multiplier_rejects_below_one(self, value: float) -> None:
        """BackoffMultiplier rejects floats below 1.0."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.BackoffMultiplier)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(value)

    @pytest.mark.parametrize("value", [0.0, 50.0, 99.9, 100.0])
    def test_percentage_accepts_in_range(self, value: float) -> None:
        """Percentage round-trips values in the inclusive 0..100 range."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.Percentage)
        tm.that(adapter.validate_python(value), eq=value)

    @pytest.mark.parametrize("value", [-0.1, 100.1, 1000.0])
    def test_percentage_rejects_out_of_range(self, value: float) -> None:
        """Percentage rejects values below 0 or above 100."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.Percentage)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(value)

    @pytest.mark.parametrize("value", [0.0, 0.25, 0.5, 1.0])
    def test_decimal_fraction_accepts_in_range(self, value: float) -> None:
        """DecimalFraction round-trips values in the inclusive 0..1 range."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.DecimalFraction)
        tm.that(adapter.validate_python(value), eq=value)

    @pytest.mark.parametrize("value", [-0.1, 1.1, 2.0])
    def test_decimal_fraction_rejects_out_of_range(self, value: float) -> None:
        """DecimalFraction rejects values below 0 or above 1."""
        adapter: m.TypeAdapter[float] = m.TypeAdapter(t.DecimalFraction)
        with pytest.raises(c.ValidationError):
            adapter.validate_python(value)

    # -- non-empty string constraints ----------------------------------------

    @pytest.mark.parametrize(
        "alias",
        [t.HostnameStr, t.UriString, t.TimestampStr],
        ids=["hostname", "uri", "timestamp"],
    )
    @pytest.mark.parametrize(
        "value",
        [c.LOCALHOST, "https://example.com", "2025-01-01T00:00:00Z"],
    )
    def test_non_empty_string_aliases_accept_non_empty(
        self, alias: type[str], value: str
    ) -> None:
        """Non-empty string aliases round-trip any string of length >= 1."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(alias)
        tm.that(adapter.validate_python(value), eq=value)

    @pytest.mark.parametrize(
        "alias",
        [t.HostnameStr, t.UriString, t.TimestampStr],
        ids=["hostname", "uri", "timestamp"],
    )
    def test_non_empty_string_aliases_reject_empty(
        self, alias: type[str]
    ) -> None:
        """Non-empty string aliases reject the empty string."""
        adapter: m.TypeAdapter[str] = m.TypeAdapter(alias)
        with pytest.raises(c.ValidationError):
            adapter.validate_python("")
