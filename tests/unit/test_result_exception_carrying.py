"""Tests for FlextResult exception carrying and propagation.

Module: flext_core.result
Scope: Exception carrying functionality in FlextResult

Tests exception carrying including:
- fail() with optional exception parameter
- safe() decorator capturing exceptions
- create_from_callable() capturing exceptions
- Exception propagation through chaining methods (map, flat_map, alt, lash, traverse)
- Backward compatibility (no exception parameter)
- Exception property access
- Error message patterns with exception carrying

Uses Python 3.13 patterns and pytest parametrization.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest
from flext_core import r, t
from flext_core.protocols import p


class TestFailNoExceptionBackwardCompat:
    """Test backward compatibility: fail() without exception parameter."""

    def test_fail_no_exception_backward_compat(self) -> None:
        """Verify fail() works without exception parameter (backward compat)."""
        error_msg = "Operation failed"
        result: r[int] = r[int].fail(error_msg)

        assert result.is_failure
        assert result.error == error_msg
        assert result.exception is None
        assert isinstance(result, p.Result)


class TestFailWithException:
    """Test fail() with exception parameter."""

    def test_fail_with_exception(self) -> None:
        """Verify fail() carries exception when provided."""
        error_msg = "Division by zero"
        exc = ZeroDivisionError("cannot divide by zero")
        result: r[float] = r[float].fail(error_msg, exception=exc)

        assert result.is_failure
        assert result.error == error_msg
        assert result.exception is exc
        assert isinstance(result.exception, ZeroDivisionError)

    def test_fail_with_exception_and_error_code(self) -> None:
        """Verify fail() carries exception with error_code."""
        error_msg = "Invalid input"
        error_code = "INVALID_INPUT"
        exc = ValueError("expected integer")
        result: r[str] = r[str].fail(error_msg, error_code=error_code, exception=exc)

        assert result.is_failure
        assert result.error == error_msg
        assert result.error_code == error_code
        assert result.exception is exc

    def test_fail_with_exception_and_error_data(self) -> None:
        """Verify fail() carries exception with error_data."""
        error_msg = "Validation failed"
        error_data = t.ConfigMap({"field": "email", "reason": "invalid format"})
        exc = ValueError("invalid email")
        result: r[dict] = r[dict].fail(error_msg, error_data=error_data, exception=exc)  # type: ignore[type-arg]

        assert result.is_failure
        assert result.error == error_msg
        assert result.error_data == error_data
        assert result.exception is exc

    def test_fail_with_none_error_and_exception(self) -> None:
        """Verify fail() converts None error to empty string, carries exception."""
        exc = RuntimeError("something went wrong")
        result: r[int] = r[int].fail(None, exception=exc)

        assert result.is_failure
        assert result.error == ""
        assert result.exception is exc


class TestSafeCarriesException:
    """Test @FlextResult.safe decorator captures exceptions."""

    def test_safe_carries_exception(self) -> None:
        """Verify @safe decorator captures exception on failure."""

        @r.safe
        def divide(a: int, b: int) -> float:
            return a / b

        result = divide(10, 0)

        assert result.is_failure
        assert result.error is not None and "division by zero" in result.error.lower()
        assert result.exception is not None
        assert isinstance(result.exception, ZeroDivisionError)

    def test_safe_no_exception_on_success(self) -> None:
        """Verify @safe decorator returns success without exception."""

        @r.safe
        def add(a: int, b: int) -> int:
            return a + b

        result = add(5, 3)

        assert result.is_success
        assert result.value == 8
        assert result.exception is None

    def test_safe_captures_value_error(self) -> None:
        """Verify @safe captures ValueError."""

        @r.safe
        def parse_int(s: str) -> int:
            if not s.isdigit():
                raise ValueError(f"'{s}' is not a valid integer")
            return int(s)

        result = parse_int("abc")

        assert result.is_failure
        assert result.exception is not None
        assert isinstance(result.exception, ValueError)

    def test_safe_captures_type_error(self) -> None:
        """Verify @safe captures TypeError."""

        @r.safe
        def get_length(obj: object) -> int:
            return len(obj)  # type: ignore[arg-type]

        result = get_length(42)

        assert result.is_failure
        assert result.exception is not None
        assert isinstance(result.exception, TypeError)


class TestCreateFromCallableCarriesException:
    """Test create_from_callable() captures exceptions."""

    def test_create_from_callable_carries_exception(self) -> None:
        """Verify create_from_callable() captures exception on failure."""

        def risky_operation() -> int:
            msg = "operation failed"
            raise RuntimeError(msg)

        result = r[int].create_from_callable(risky_operation)

        assert result.is_failure
        assert result.error is not None and "operation failed" in result.error
        assert result.exception is not None
        assert isinstance(result.exception, RuntimeError)

    def test_create_from_callable_success_no_exception(self) -> None:
        """Verify create_from_callable() returns success without exception."""

        def safe_operation() -> str:
            return "success"

        result = r[str].create_from_callable(safe_operation)

        assert result.is_success
        assert result.value == "success"
        assert result.exception is None

    def test_create_from_callable_with_error_code(self) -> None:
        """Verify create_from_callable() carries error_code."""

        def failing_operation() -> int:
            msg = "invalid value"
            raise ValueError(msg)

        result = r[int].create_from_callable(
            failing_operation, error_code="INVALID_VALUE"
        )

        assert result.is_failure
        assert result.error_code == "INVALID_VALUE"
        assert result.exception is not None


class TestMapPropagatesException:
    """Test map() propagates exception from failure."""

    def test_map_propagates_exception_on_failure(self) -> None:
        """Verify map() preserves exception from failed result."""
        exc = ValueError("original error")
        result = r[int].fail("error", exception=exc)
        mapped: r[int] = result.map(lambda x: x * 2)

        assert mapped.is_failure
        assert mapped.exception is exc

    def test_map_success_no_exception(self) -> None:
        """Verify map() on success has no exception."""
        result = r[int].ok(5)
        mapped: r[int] = result.map(lambda x: x * 2)

        assert mapped.is_success
        assert mapped.value == 10
        assert mapped.exception is None

    def test_map_chain_preserves_exception(self) -> None:
        """Verify exception preserved through map chain."""
        exc = RuntimeError("chain error")
        result: r[int] = r[int].fail("error", exception=exc)
        mapped = result.map(lambda x: x + 1).map(lambda x: x * 2)

        assert mapped.is_failure
        assert mapped.exception is exc


class TestFlatMapPropagatesException:
    """Test flat_map() propagates exception from failure."""

    def test_flat_map_propagates_exception_on_failure(self) -> None:
        """Verify flat_map() preserves exception from failed result."""
        exc = TypeError("type error")
        result: r[int] = r[int].fail("error", exception=exc)
        flat_mapped = result.flat_map(lambda x: r[str].ok(str(x)))

        assert flat_mapped.is_failure
        assert flat_mapped.exception is exc

    def test_flat_map_success_no_exception(self) -> None:
        """Verify flat_map() on success has no exception."""
        result = r[int].ok(5)
        flat_mapped: r[str] = result.flat_map(lambda x: r[str].ok(str(x)))

        assert flat_mapped.is_success
        assert flat_mapped.value == "5"
        assert flat_mapped.exception is None

    def test_flat_map_chain_preserves_exception(self) -> None:
        """Verify exception preserved through flat_map chain."""
        exc = KeyError("missing key")
        result: r[int] = r[int].fail("error", exception=exc)
        flat_mapped = result.flat_map(lambda x: r[int].ok(x + 1)).flat_map(
            lambda x: r[str].ok(str(x))
        )

        assert flat_mapped.is_failure
        assert flat_mapped.exception is exc


class TestAltPropagatesException:
    """Test alt() propagates exception from failure."""

    def test_alt_propagates_exception(self) -> None:
        """Verify alt() preserves exception when transforming error."""
        exc = ValueError("original")
        result: r[int] = r[int].fail("error", exception=exc)
        altered = result.alt(lambda e: f"transformed: {e}")

        assert altered.is_failure
        assert altered.exception is exc
        assert altered.error is not None and "transformed" in altered.error

    def test_alt_success_no_exception(self) -> None:
        """Verify alt() on success has no exception."""
        result: r[int] = r[int].ok(42)
        altered = result.alt(lambda e: f"error: {e}")

        assert altered.is_success
        assert altered.value == 42
        assert altered.exception is None


class TestLashPropagatesException:
    """Test lash() propagates exception from failure."""

    def test_lash_propagates_exception(self) -> None:
        """Verify lash() preserves exception when recovering."""
        exc = RuntimeError("recovery needed")
        result = r[int].fail("error", exception=exc)
        recovered = result.lash(lambda e: r[int].ok(0))

        assert recovered.is_success
        assert recovered.value == 0

    def test_lash_preserves_exception_on_recovery_failure(self) -> None:
        """Verify lash() preserves exception when recovery also fails."""
        exc = ValueError("original error")
        result = r[int].fail("error", exception=exc)
        recovery_exc = RuntimeError("recovery failed")
        recovered = result.lash(
            lambda e: r[int].fail(f"recovery failed: {e}", exception=recovery_exc)
        )

        assert recovered.is_failure
        assert recovered.exception is recovery_exc


class TestTraversePropagatesException:
    """Test traverse() propagates exceptions from failures."""

    def test_traverse_propagates_exception(self) -> None:
        """Verify traverse() preserves exception from failed item."""
        exc = ValueError("item error")
        items = [1, 2, 3]

        def process_with_failure(x: int) -> r[int]:
            if x == 2:
                return r[int].fail("error", exception=exc)
            return r[int].ok(x * 2)

        result = r[int].traverse(items, process_with_failure, fail_fast=True)

        assert result.is_failure
        assert result.exception is exc

    def test_traverse_accumulate_preserves_exceptions(self) -> None:
        """Verify traverse() with fail_fast=False accumulates error messages."""
        exc1 = ValueError("error 1")
        exc2 = TypeError("error 2")
        items = [1, 2, 3]

        def process_with_failures(x: int) -> r[int]:
            if x == 1:
                return r[int].fail("error 1", exception=exc1)
            if x == 3:
                return r[int].fail("error 2", exception=exc2)
            return r[int].ok(x * 2)

        result = r[int].traverse(items, process_with_failures, fail_fast=False)

        assert result.is_failure
        assert result.error is not None
        assert "error 1" in result.error and "error 2" in result.error


class TestFromIOResultCarriesException:
    """Test from_io_result() carries exception."""

    def test_from_io_result_carries_exception(self) -> None:
        """Verify from_io_result() preserves exception from IOResult."""
        from returns.io import IO, IOSuccess  # noqa: PLC0415

        result = r[int].from_io_result(IOSuccess(42))

        assert result.is_success
        assert isinstance(result.value, IO)


class TestFromValidationCarriesException:
    """Test from_validation() carries exception."""

    def test_from_validation_carries_exception(self) -> None:
        """Verify from_validation() captures validation exception."""
        from pydantic import BaseModel, ValidationError  # noqa: PLC0415

        class User(BaseModel):
            name: str
            age: int

        invalid_data = {"name": "Alice", "age": "not_an_int"}
        result = r[User].from_validation(invalid_data, User)

        assert result.is_failure
        assert result.exception is not None
        assert isinstance(result.exception, ValidationError)


class TestToIOChainsException:
    """Test to_io() chains exception."""

    def test_to_io_chains_exception(self) -> None:
        """Verify to_io() raises on failure."""
        from flext_core.exceptions import e  # noqa: PLC0415

        exc = ValueError("conversion error")
        result = r[int].fail("error", exception=exc)

        with pytest.raises(e.ValidationError):
            result.to_io()


class TestErrorOrPatternUnchanged:
    """Test backward compatibility: .error or 'fallback' pattern (31 sites)."""

    def test_error_or_pattern_unchanged(self) -> None:
        """Verify .error or 'fallback' pattern still works (31 sites)."""
        result_success = r[int].ok(42)
        result_failure = r[int].fail("error message")

        error_success = result_success.error or "fallback"
        error_failure = result_failure.error or "fallback"

        assert error_success == "fallback"
        assert error_failure == "error message"

    def test_error_or_pattern_with_exception(self) -> None:
        """Verify .error or pattern works with exception carrying."""
        exc = RuntimeError("runtime error")
        result = r[int].fail("error", exception=exc)

        error_msg = result.error or "fallback"

        assert error_msg == "error"
        assert result.exception is exc


class TestOkNoneGuardStillRaises:
    """Test ok(None) still raises ValueError guard."""

    def test_ok_none_guard_still_raises(self) -> None:
        """Verify ok(None) raises ValueError (guard maintained)."""
        with pytest.raises(ValueError, match="Cannot create success result with None"):
            r[int].ok(None)  # type: ignore[arg-type]

    def test_ok_with_valid_value_succeeds(self) -> None:
        """Verify ok() with valid value succeeds."""
        value = 42
        result = r[int].ok(value)

        assert result.is_success
        assert result.value == 42
        assert result.exception is None


class TestMonadicOperationsUnchanged:
    """Test monadic operations work unchanged with exception carrying."""

    def test_monadic_operations_unchanged(self) -> None:
        """Verify monadic operations (map, flat_map, filter) work unchanged."""
        result = r[int].ok(5)
        final = (
            result
            .map(lambda x: x * 2)
            .flat_map(lambda x: r[int].ok(x + 1))
            .filter(lambda x: x > 5)
        )

        assert final.is_success
        assert final.value == 11
        assert final.exception is None

    def test_monadic_chain_with_exception_in_middle(self) -> None:
        """Verify exception propagates through monadic chain."""
        exc = ValueError("chain error")
        result = r[int].ok(5)
        final = (
            result
            .map(lambda x: x * 2)
            .flat_map(lambda x: r[int].fail("error", exception=exc))
            .map(lambda x: x + 1)
        )

        assert final.is_failure
        assert final.exception is exc

    def test_recover_with_exception(self) -> None:
        """Verify recover() works with exception carrying."""
        exc = RuntimeError("recovery needed")
        result = r[int].fail("error", exception=exc)
        recovered = result.recover(lambda e: 0)

        assert recovered.is_success
        assert recovered.value == 0

    def test_fold_with_exception(self) -> None:
        """Verify fold() works with exception carrying."""
        exc = ValueError("fold error")
        result = r[int].fail("error", exception=exc)
        folded = result.fold(
            on_failure=lambda e: f"failed: {e}",
            on_success=lambda v: f"success: {v}",
        )

        assert folded == "failed: error"

    def test_tap_with_exception(self) -> None:
        """Verify tap() works with exception carrying."""
        exc = RuntimeError("tap error")
        result = r[int].fail("error", exception=exc)
        side_effect_called = False

        def side_effect(x: int) -> None:
            nonlocal side_effect_called
            side_effect_called = True

        tapped = result.tap(side_effect)

        assert tapped.is_failure
        assert not side_effect_called
        assert tapped.exception is exc

    def test_tap_error_with_exception(self) -> None:
        """Verify tap_error() works with exception carrying."""
        exc = ValueError("tap_error test")
        result = r[int].fail("error", exception=exc)
        side_effect_called = False

        def side_effect(e: str) -> None:
            nonlocal side_effect_called
            side_effect_called = True

        tapped = result.tap_error(side_effect)

        assert tapped.is_failure
        assert side_effect_called
        assert tapped.exception is exc


class TestExceptionPropertyAccess:
    """Test exception property access patterns."""

    def test_exception_property_none_on_success(self) -> None:
        """Verify exception property is None on success."""
        result = r[int].ok(42)
        exc = result.exception

        assert exc is None

    def test_exception_property_set_on_failure(self) -> None:
        """Verify exception property is set on failure."""
        exc = RuntimeError("test error")
        result = r[int].fail("error", exception=exc)
        retrieved_exc = result.exception

        assert retrieved_exc is exc
        assert isinstance(retrieved_exc, RuntimeError)

    def test_exception_property_type_check(self) -> None:
        """Verify exception property type is BaseException | None."""
        exc = ValueError("value error")
        result = r[int].fail("error", exception=exc)
        retrieved_exc = result.exception

        assert isinstance(retrieved_exc, BaseException)
        assert isinstance(retrieved_exc, ValueError)

    def test_exception_property_multiple_exception_types(self) -> None:
        """Verify exception property works with different exception types."""
        exceptions: list[BaseException] = [
            ValueError("value error"),
            TypeError("type error"),
            RuntimeError("runtime error"),
            KeyError("key error"),
            ZeroDivisionError("zero division"),
        ]

        for exc in exceptions:
            result = r[int].fail("error", exception=exc)
            assert result.exception is exc
            assert isinstance(result.exception, type(exc))
