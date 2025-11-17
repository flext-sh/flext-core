"""Final surgical strike: Targeting exact 76 uncovered lines for 75% coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import pytest

from flext_core import (
    FlextExceptions,
    FlextResult,
)


class TestCoverage76Lines:
    """Surgical tests for exact uncovered lines."""

    # Result.py uncovered lines: 459, 650, 679-680, 745-746, 761-768, etc.

    def test_result_enter_exit_context(self) -> None:
        """Test __enter__ and __exit__ for context manager."""
        r = FlextResult[str].ok("value")
        with r as val:
            assert val == "value"

    def test_result_getitem_index_0(self) -> None:
        """Test __getitem__ with index 0."""
        r = FlextResult[int].ok(42)
        assert r[0] == 42

    def test_result_getitem_index_1(self) -> None:
        """Test __getitem__ with index 1."""
        r = FlextResult[int].ok(42)
        assert r[1] == ""  # Success case returns empty string for error index

    def test_result_getitem_on_failure(self) -> None:
        """Test __getitem__ on failure raises exception (fast fail pattern)."""
        r = FlextResult[int].fail("error")
        # Fast fail: accessing data on failure raises exception
        with pytest.raises(FlextExceptions.BaseError):
            _ = r[0]

    def test_result_bool_true(self) -> None:
        """Test __bool__ returns True for success."""
        r = FlextResult[int].ok(42)
        assert bool(r) is True

    def test_result_bool_false(self) -> None:
        """Test __bool__ returns False for failure."""
        r = FlextResult[int].fail("error")
        assert bool(r) is False

    def test_result_iter_success(self) -> None:
        """Test __iter__ on success."""
        r = FlextResult[str].ok("value")
        items = list(r)
        assert len(items) == 2
        assert "value" in items

    def test_result_iter_failure(self) -> None:
        """Test __iter__ on failure raises exception (fast fail pattern)."""
        r = FlextResult[str].fail("error message")
        # Fast fail: iterating on failure raises exception
        with pytest.raises(FlextExceptions.BaseError):
            _ = list(r)

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

    def test_result_xor_operator(self) -> None:
        """Test __xor__ operator for recovery."""
        r = FlextResult[int].fail("error")
        r2 = r ^ (lambda e: 99)
        assert r2.is_success
        assert r2.value == 99

    def test_result_hash_consistency(self) -> None:
        """Test __hash__ consistency."""
        r1 = FlextResult[int].ok(42)
        r2 = FlextResult[int].ok(42)
        assert hash(r1) == hash(r2)

    def test_result_eq_success(self) -> None:
        """Test __eq__ for equal success results."""
        r1 = FlextResult[int].ok(42)
        r2 = FlextResult[int].ok(42)
        assert r1 == r2

    def test_result_eq_failure(self) -> None:
        """Test __eq__ for equal failure results."""
        r1 = FlextResult[int].fail("error")
        r2 = FlextResult[int].fail("error")
        assert r1 == r2

    def test_result_ne_different_values(self) -> None:
        """Test __ne__ for different values."""
        r1 = FlextResult[int].ok(42)
        r2 = FlextResult[int].ok(99)
        assert r1 != r2

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

    def test_result_data_property(self) -> None:
        """Test .data property (backward compat)."""
        r = FlextResult[str].ok("test")
        assert r.data == "test"

    def test_result_value_property(self) -> None:
        """Test .value property."""
        r = FlextResult[str].ok("test")
        assert r.value == "test"

    def test_result_success_alias(self) -> None:
        """Test .success property alias for .is_success."""
        r = FlextResult[int].ok(42)
        assert r.success is True

    def test_result_failed_alias(self) -> None:
        """Test .failed property alias for .is_failure."""
        r = FlextResult[int].fail("error")
        assert r.failed is True

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

    def test_result_tap_side_effect(self) -> None:
        """Test tap applies side effects."""
        side_effects = []
        r = FlextResult[int].ok(42)
        r.tap(lambda x: side_effects.append(x * 2))
        assert 84 in side_effects

    def test_result_lash_recovery(self) -> None:
        """Test lash recovers from failure."""
        r = FlextResult[int].fail("error")
        r2 = r.lash(lambda e: FlextResult[int].ok(42))
        assert r2.is_success
        assert r2.value == 42

    def test_result_recover_on_failure(self) -> None:
        """Test recover transforms error into value."""
        r = FlextResult[int].fail("error")
        r2 = r.recover(lambda e: 100)
        assert r2.is_success
        assert r2.value == 100

    def test_result_unwrap_or_default(self) -> None:
        """Test unwrap_or returns default on failure."""
        r = FlextResult[int].fail("error")
        assert r.unwrap_or(999) == 999

    def test_result_unwrap_or_value(self) -> None:
        """Test unwrap_or returns value on success."""
        r = FlextResult[int].ok(42)
        assert r.unwrap_or(999) == 42

    def test_result_expect_success(self) -> None:
        """Test expect returns value on success."""
        r = FlextResult[str].ok("value")
        assert r.expect("failed") == "value"

    def test_result_or_else_get_failure(self) -> None:
        """Test or_else_get with failure."""
        r = FlextResult[int].fail("error")
        r2 = r.or_else_get(lambda: FlextResult[int].ok(42))
        assert r2.is_success
        assert r2.value == 42

    def test_result_or_else_get_success(self) -> None:
        """Test or_else_get passes through success."""
        r = FlextResult[int].ok(42)
        r2 = r.or_else_get(lambda: FlextResult[int].ok(999))
        assert r2.value == 42

    def test_result_map_transforms_type(self) -> None:
        """Test map transforms value type."""
        r = FlextResult[int].ok(42)
        r2 = r.map(str)
        assert r2.value == "42"

    def test_result_flat_map_chains_results(self) -> None:
        """Test flat_map chains multiple results."""
        r = FlextResult[int].ok(1).flat_map(lambda x: FlextResult[int].ok(x + 1))
        assert r.value == 2

    def test_result_error_property(self) -> None:
        """Test .error property."""
        r = FlextResult[int].fail("error message")
        assert r.error == "error message"

    def test_result_error_code_property(self) -> None:
        """Test .error_code property."""
        r = FlextResult[int].fail("error")
        # error_code can be None or a string
        assert r.error_code is None or isinstance(r.error_code, str)

    def test_result_error_data_property(self) -> None:
        """Test .error_data property."""
        r = FlextResult[int].fail("error")
        assert isinstance(r.error_data, dict)

    def test_result_value_or_none_success(self) -> None:
        """Test .value on success - value_or_none removed."""
        r = FlextResult[str].ok("value")
        assert r.value == "value"

    def test_result_value_or_none_failure(self) -> None:
        """Test .unwrap_or on failure - value_or_none removed."""
        r = FlextResult[str].fail("error")
        assert r.unwrap_or("default") == "default"


__all__ = ["TestCoverage76Lines"]
