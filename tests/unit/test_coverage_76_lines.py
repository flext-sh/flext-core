"""Tests for FlextResult and FlextExceptions coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_core import (
    FlextExceptions,
    FlextResult,
)


class TestResultBasics:
    """Tests for FlextResult basic operations."""

    def test_result_bool_true(self) -> None:
        """Test __bool__ returns True for success."""
        r = FlextResult[int].ok(42)
        assert bool(r) is True

    def test_result_bool_false(self) -> None:
        """Test __bool__ returns False for failure."""
        r = FlextResult[int].fail("error")
        assert bool(r) is False

    def test_result_or_operator(self) -> None:
        """Test __or__ operator for default value."""
        r = FlextResult[int].fail("error")
        result = r | 99
        assert result == 99

    def test_result_or_operator_success(self) -> None:
        """Test __or__ operator on success."""
        r = FlextResult[int].ok(42)
        result = r | 99
        assert result == 42

    def test_result_repr_format(self) -> None:
        """Test __repr__ format."""
        r = FlextResult[int].ok(42)
        repr_str = repr(r)
        assert "FlextResult" in repr_str

    def test_result_repr_failure(self) -> None:
        """Test __repr__ on failure."""
        r = FlextResult[int].fail("error")
        repr_str = repr(r)
        assert "error" in repr_str.lower() or "FlextResult" in repr_str

    def test_result_value_property(self) -> None:
        """Test .value property."""
        r = FlextResult[str].ok("test")
        assert r.value == "test"

    def test_result_is_success_property(self) -> None:
        """Test .is_success property."""
        r = FlextResult[int].ok(42)
        assert r.is_success is True

    def test_result_is_failure_property(self) -> None:
        """Test .is_failure property."""
        r = FlextResult[int].fail("error")
        assert r.is_failure is True

    # Tests for FlextExceptions

    def test_exception_str_representation(self) -> None:
        """Test exception string representation."""
        exc = FlextExceptions.ValidationError("test message")
        assert "test message" in str(exc)

    def test_exception_base_error_str(self) -> None:
        """Test BaseError string representation."""
        exc = FlextExceptions.BaseError("base message")
        exc_str = str(exc)
        assert len(exc_str) > 0

    def test_exception_connection_error(self) -> None:
        """Test ConnectionError exception."""
        exc = FlextExceptions.ConnectionError("connection lost")
        assert isinstance(exc, Exception)

    def test_exception_rate_limit_error(self) -> None:
        """Test RateLimitError exception."""
        exc = FlextExceptions.RateLimitError("rate limited")
        assert isinstance(exc, Exception)

    def test_exception_circuit_breaker_error(self) -> None:
        """Test CircuitBreakerError exception."""
        exc = FlextExceptions.CircuitBreakerError("circuit open")
        assert isinstance(exc, Exception)


class TestResultTransformations:
    """Tests for FlextResult transformation methods."""

    def test_result_lash_recovery(self) -> None:
        """Test lash recovers from failure."""
        r = FlextResult[int].fail("error")
        r2 = r.lash(lambda e: FlextResult[int].ok(42))
        assert r2.is_success
        assert r2.value == 42

    def test_result_unwrap_or_default(self) -> None:
        """Test unwrap_or returns default on failure."""
        r = FlextResult[int].fail("error")
        assert r.unwrap_or(999) == 999

    def test_result_unwrap_or_value(self) -> None:
        """Test unwrap_or returns value on success."""
        r = FlextResult[int].ok(42)
        assert r.unwrap_or(999) == 42

    def test_result_map_transforms_type(self) -> None:
        """Test map transforms value type."""
        r = FlextResult[int].ok(42)
        r2 = r.map(str)
        assert r2.value == "42"

    def test_result_flat_map_chains_results(self) -> None:
        """Test flat_map chains multiple results."""

        def increment_wrapper(x: object) -> FlextResult[object]:
            if isinstance(x, int):
                result = FlextResult[int].ok(x + 1)
                if result.is_success:
                    return FlextResult[object].ok(result.value)
                return FlextResult[object].fail(result.error or "Increment failed")
            return FlextResult[object].fail("Invalid input")

        r = FlextResult[int].ok(1).flat_map(increment_wrapper)
        assert r.value == 2

    def test_result_error_property(self) -> None:
        """Test .error property."""
        r = FlextResult[int].fail("error message")
        assert r.error == "error message"

    def test_result_error_code_property(self) -> None:
        """Test .error_code property."""
        r = FlextResult[int].fail("error", error_code="E001")
        assert r.error_code == "E001"

    def test_result_error_code_none(self) -> None:
        """Test .error_code property when not provided."""
        r = FlextResult[int].fail("error")
        assert r.error_code is None

    def test_result_error_data_property(self) -> None:
        """Test .error_data property."""
        # Without error_data, it's None
        r_no_data = FlextResult[int].fail("error")
        assert r_no_data.error_data is None
        # With error_data, it's a dict
        r_with_data = FlextResult[int].fail("error", error_data={"detail": "info"})
        assert isinstance(r_with_data.error_data, dict)
        assert r_with_data.error_data == {"detail": "info"}

    def test_result_value_on_success(self) -> None:
        """Test .value on success."""
        r = FlextResult[str].ok("value")
        assert r.value == "value"

    def test_result_unwrap_or_on_failure(self) -> None:
        """Test .unwrap_or on failure."""
        r = FlextResult[str].fail("error")
        assert r.unwrap_or("default") == "default"

    def test_result_alt_on_failure(self) -> None:
        """Test .alt provides alternative on failure."""
        r = FlextResult[int].fail("error")
        r2 = r.alt(lambda _: "recovered")  # alt transforms error, not value
        assert r2.is_failure  # still failure but error is transformed
        assert r2.error == "recovered"

    def test_result_alt_on_success(self) -> None:
        """Test .alt passes through on success."""
        r = FlextResult[int].ok(42)
        r2 = r.alt(lambda _: "should not be called")
        assert r2.is_success
        assert r2.value == 42

    def test_result_filter_passes(self) -> None:
        """Test filter passes when predicate is true."""
        r = FlextResult[int].ok(42)
        r2 = r.filter(lambda x: x > 0)
        assert r2.is_success
        assert r2.value == 42

    def test_result_filter_fails(self) -> None:
        """Test filter fails when predicate is false."""
        r = FlextResult[int].ok(42)
        r2 = r.filter(lambda x: x > 100)
        assert r2.is_failure


__all__ = ["TestResultBasics", "TestResultTransformations"]
