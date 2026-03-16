"""Tests for r and FlextExceptions coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_tests import tm

from flext_core import FlextExceptions, m, r


class TestResultBasics:
    """Tests for r basic operations."""

    def test_result_bool_true(self) -> None:
        """Test __bool__ returns True for success."""
        result: r[int] = r[int].ok(42)
        tm.that(bool(result), eq=True)

    def test_result_bool_false(self) -> None:
        """Test __bool__ returns False for failure."""
        result: r[int] = r[int].fail("error")
        tm.that(bool(result), eq=False)

    def test_result_or_operator(self) -> None:
        """Test __or__ operator for default value."""
        result: r[int] = r[int].fail("error")
        defaulted = result | 99
        tm.that(defaulted, eq=99)

    def test_result_or_operator_success(self) -> None:
        """Test __or__ operator on success."""
        result: r[int] = r[int].ok(42)
        defaulted = result | 99
        tm.that(defaulted, eq=42)

    def test_result_repr_format(self) -> None:
        """Test __repr__ format."""
        result = r[int].ok(42)
        repr_str = repr(result)
        tm.that(repr_str, has="r[T].ok")

    def test_result_repr_failure(self) -> None:
        """Test __repr__ on failure."""
        result: r[int] = r[int].fail("error")
        repr_str = repr(result)
        tm.that("error" in repr_str.lower() or "r[T].fail" in repr_str, eq=True)

    def test_result_value_property(self) -> None:
        """Test .value property."""
        result = r[str].ok("test")
        tm.that(result.value, eq="test")

    def test_result_is_success_property(self) -> None:
        """Test .is_success property."""
        result = r[int].ok(42)
        tm.ok(result)

    def test_result_is_failure_property(self) -> None:
        """Test .is_failure property."""
        result: r[int] = r[int].fail("error")
        tm.fail(result)

    def test_exception_str_representation(self) -> None:
        """Test exception string representation."""
        exc = FlextExceptions.ValidationError("test message")
        tm.that(str(exc), has="test message")

    def test_exception_base_error_str(self) -> None:
        """Test BaseError string representation."""
        exc = FlextExceptions.BaseError("base message")
        exc_str = str(exc)
        tm.that(len(exc_str), gt=0)

    def test_exception_connection_error(self) -> None:
        """Test ConnectionError exception."""
        exc = FlextExceptions.ConnectionError("connection lost")
        tm.that(str(exc), has="connection lost")

    def test_exception_rate_limit_error(self) -> None:
        """Test RateLimitError exception."""
        exc = FlextExceptions.RateLimitError("rate limited")
        tm.that(str(exc), has="rate limited")

    def test_exception_circuit_breaker_error(self) -> None:
        """Test CircuitBreakerError exception."""
        exc = FlextExceptions.CircuitBreakerError("circuit open")
        tm.that(str(exc), has="circuit open")


class TestResultTransformations:
    """Tests for r transformation methods."""

    def test_result_lash_recovery(self) -> None:
        """Test lash recovers from failure."""
        result: r[int] = r[int].fail("error")
        r2 = result.lash(lambda e: r[int].ok(42))
        tm.ok(r2)
        tm.that(r2.value, eq=42)

    def test_result_unwrap_or_default(self) -> None:
        """Test unwrap_or returns default on failure."""
        result: r[int] = r[int].fail("error")
        tm.that(result.unwrap_or(999), eq=999)

    def test_result_unwrap_or_value(self) -> None:
        """Test unwrap_or returns value on success."""
        result: r[int] = r[int].ok(42)
        tm.that(result.unwrap_or(999), eq=42)

    def test_result_map_transforms_type(self) -> None:
        """Test map transforms value type."""
        result = r[int].ok(42)
        r2 = result.map(str)
        tm.that(r2.value, eq="42")

    def test_result_flat_map_chains_results(self) -> None:
        """Test flat_map chains multiple results."""

        def increment_wrapper(x: int) -> r[int]:
            result = r[int].ok(x + 1)
            if result.is_success:
                return r[int].ok(result.value)
            return r[int].fail(result.error or "Increment failed")

        result = r[int].ok(1).flat_map(increment_wrapper)
        tm.that(result.value, eq=2)

    def test_result_error_property(self) -> None:
        """Test .error property."""
        result: r[int] = r[int].fail("error message")
        tm.fail(result, has="error message")

    def test_result_error_code_property(self) -> None:
        """Test .error_code property."""
        result: r[int] = r[int].fail("error", error_code="E001")
        tm.that(result.error_code, eq="E001")

    def test_result_error_code_none(self) -> None:
        """Test .error_code property when not provided."""
        result: r[int] = r[int].fail("error")
        tm.that(result.error_code, none=True)

    def test_result_error_data_property(self) -> None:
        """Test .error_data property."""
        r_no_data: r[int] = r[int].fail("error")
        tm.that(r_no_data.error_data, none=True)
        r_with_data: r[int] = r[int].fail(
            "error",
            error_data=m.ConfigMap(root={"detail": "info"}),
        )
        tm.that(isinstance(r_with_data.error_data, m.ConfigMap), eq=True)
        tm.that(str(r_with_data.error_data), has="info")

    def test_result_value_on_success(self) -> None:
        """Test .value on success."""
        result = r[str].ok("value")
        tm.that(result.value, eq="value")

    def test_result_unwrap_or_on_failure(self) -> None:
        """Test .unwrap_or on failure."""
        result: r[str] = r[str].fail("error")
        tm.that(result.unwrap_or("default"), eq="default")

    def test_result_alt_on_failure(self) -> None:
        """Test .alt provides alternative on failure."""
        result: r[int] = r[int].fail("error")
        r2 = result.map_error(lambda _: "recovered")
        tm.fail(r2)
        tm.fail(r2, has="recovered")

    def test_result_alt_on_success(self) -> None:
        """Test .alt passes through on success."""
        result = r[int].ok(42)
        r2 = result.map_error(lambda _: "should not be called")
        tm.ok(r2)
        tm.that(r2.value, eq=42)

    def test_result_filter_passes(self) -> None:
        """Test filter passes when predicate is true."""
        result = r[int].ok(42)
        r2 = result.filter(lambda x: x > 0)
        tm.ok(r2)
        tm.that(r2.value, eq=42)

    def test_result_filter_fails(self) -> None:
        """Test filter fails when predicate is false."""
        result = r[int].ok(42)
        r2 = result.filter(lambda x: x > 100)
        tm.fail(r2)


__all__ = ["TestResultBasics", "TestResultTransformations"]
