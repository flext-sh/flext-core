"""Tests for r exception carrying and propagation.

Module: flext_core.result
Scope: Exception carrying functionality in r

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

from collections.abc import Sized
from typing import cast

from flext_tests import tm
from pydantic import BaseModel, ValidationError

from flext_core import m, r, t


class TestFailNoExceptionBackwardCompat:
    """Test backward compatibility: fail() without exception parameter."""

    def test_fail_no_exception_backward_compat(self) -> None:
        """Verify fail() works without exception parameter (backward compat)."""
        error_msg = "Operation failed"
        result: r[int] = r[int].fail(error_msg)
        tm.that(result.is_failure, eq=True)
        tm.that(result.error == error_msg, eq=True)
        tm.that(result.exception is None, eq=True)
        result_as_obj = result
        tm.that(hasattr(result_as_obj, "is_success"), eq=True)
        tm.that(hasattr(result_as_obj, "is_failure"), eq=True)


class TestFailWithException:
    """Test fail() with exception parameter."""

    def test_fail_with_exception(self) -> None:
        """Verify fail() carries exception when provided."""
        error_msg = "Division by zero"
        exc = ZeroDivisionError("cannot divide by zero")
        result: r[float] = r[float].fail(error_msg, exception=exc)
        tm.that(result.is_failure, eq=True)
        tm.that(result.error == error_msg, eq=True)
        tm.that(result.exception is exc, eq=True)
        tm.that(isinstance(result.exception, ZeroDivisionError), eq=True)

    def test_fail_with_exception_and_error_code(self) -> None:
        """Verify fail() carries exception with error_code."""
        error_msg = "Invalid input"
        error_code = "INVALID_INPUT"
        exc = ValueError("expected integer")
        result: r[str] = r[str].fail(error_msg, error_code=error_code, exception=exc)
        tm.that(result.is_failure, eq=True)
        tm.that(result.error == error_msg, eq=True)
        tm.that(result.error_code == error_code, eq=True)
        tm.that(result.exception is exc, eq=True)

    def test_fail_with_exception_and_error_data(self) -> None:
        """Verify fail() carries exception with error_data."""
        error_msg = "Validation failed"
        error_data: dict[str, t.Scalar] = {
            "field": "email",
            "reason": "invalid format",
        }
        exc = ValueError("invalid email")
        result: r[dict[str, str]] = r[dict[str, str]].fail(
            error_msg,
            error_data=error_data,
            exception=exc,
        )
        tm.that(result.is_failure, eq=True)
        tm.that(result.error == error_msg, eq=True)
        tm.that(
            result.error_data
            == m.ConfigMap(
                root=cast("dict[str, t.NormalizedValue | BaseModel]", error_data)
            ),
            eq=True,
        )
        tm.that(result.exception is exc, eq=True)

    def test_fail_with_none_error_and_exception(self) -> None:
        """Verify fail() converts None error to empty string, carries exception."""
        exc = RuntimeError("something went wrong")
        result: r[int] = r[int].fail(None, exception=exc)
        tm.that(result.is_failure, eq=True)
        tm.that(result.error == "", eq=True)
        tm.that(result.exception is exc, eq=True)


class TestSafeCarriesException:
    """Test @r.safe decorator captures exceptions."""

    def test_safe_carries_exception(self) -> None:
        """Verify @safe decorator captures exception on failure."""

        @r.safe
        def divide(a: int, b: int) -> float:
            return a / b

        result: r[float] = divide(10, 0)
        tm.that(result.is_failure, eq=True)
        tm.that(
            result.error is not None and "division by zero" in result.error.lower(),
            eq=True,
        )
        tm.that(result.exception is not None, eq=True)
        tm.that(isinstance(result.exception, ZeroDivisionError), eq=True)

    def test_safe_no_exception_on_success(self) -> None:
        """Verify @safe decorator returns success without exception."""

        @r.safe
        def add(a: int, b: int) -> int:
            return a + b

        result: r[int] = add(5, 3)
        tm.that(result.is_success, eq=True)
        tm.that(result.value == 8, eq=True)
        tm.that(result.exception is None, eq=True)

    def test_safe_captures_value_error(self) -> None:
        """Verify @safe captures ValueError."""

        @r.safe
        def parse_int(s: str) -> int:
            if not s.isdigit():
                raise ValueError(f"'{s}' is not a valid integer")
            return int(s)

        result: r[int] = parse_int("abc")
        tm.that(result.is_failure, eq=True)
        tm.that(result.exception is not None, eq=True)
        tm.that(isinstance(result.exception, ValueError), eq=True)

    def test_safe_captures_type_error(self) -> None:
        """Verify @safe captures TypeError."""

        class _BrokenSized:
            """Test helper: implements __len__ but raises TypeError."""

            def __len__(self) -> int:
                msg = "no length"
                raise TypeError(msg)

        @r.safe
        def get_length(obj: Sized) -> int:
            return len(obj)

        result: r[int] = get_length(_BrokenSized())
        tm.that(result.is_failure, eq=True)
        tm.that(result.exception is not None, eq=True)
        tm.that(isinstance(result.exception, TypeError), eq=True)


class TestCreateFromCallableCarriesException:
    """Test create_from_callable() captures exceptions."""

    def test_create_from_callable_carries_exception(self) -> None:
        """Verify create_from_callable() captures exception on failure."""

        def risky_operation() -> int:
            msg = "operation failed"
            raise RuntimeError(msg)

        result: r[int] = r[int].create_from_callable(risky_operation)
        tm.that(result.is_failure, eq=True)
        tm.that(
            result.error is not None and "operation failed" in result.error, eq=True
        )
        tm.that(result.exception is not None, eq=True)
        tm.that(isinstance(result.exception, RuntimeError), eq=True)

    def test_create_from_callable_success_no_exception(self) -> None:
        """Verify create_from_callable() returns success without exception."""

        def safe_operation() -> str:
            return "success"

        result: r[str] = r[str].create_from_callable(safe_operation)
        tm.that(result.is_success, eq=True)
        tm.that(result.value == "success", eq=True)
        tm.that(result.exception is None, eq=True)

    def test_create_from_callable_with_error_code(self) -> None:
        """Verify create_from_callable() carries error_code."""

        def failing_operation() -> int:
            msg = "invalid value"
            raise ValueError(msg)

        result: r[int] = r[int].create_from_callable(
            failing_operation,
            error_code="INVALID_VALUE",
        )
        tm.that(result.is_failure, eq=True)
        tm.that(result.error_code == "INVALID_VALUE", eq=True)
        tm.that(result.exception is not None, eq=True)


class TestMapPropagatesException:
    """Test map() propagates exception from failure."""

    def test_map_propagates_exception_on_failure(self) -> None:
        """Verify map() preserves exception from failed result."""
        exc = ValueError("original error")
        result: r[int] = r[int].fail("error", exception=exc)
        mapped: r[int] = result.map(lambda x: x * 2)
        tm.that(mapped.is_failure, eq=True)
        tm.that(mapped.exception is exc, eq=True)

    def test_map_success_no_exception(self) -> None:
        """Verify map() on success has no exception."""
        result: r[int] = r[int].ok(5)
        mapped: r[int] = result.map(lambda x: x * 2)
        tm.that(mapped.is_success, eq=True)
        tm.that(mapped.value == 10, eq=True)
        tm.that(mapped.exception is None, eq=True)

    def test_map_chain_preserves_exception(self) -> None:
        """Verify exception preserved through map chain."""
        exc = RuntimeError("chain error")
        result: r[int] = r[int].fail("error", exception=exc)
        mapped: r[int] = result.map(lambda x: x + 1).map(lambda x: x * 2)
        tm.that(mapped.is_failure, eq=True)
        tm.that(mapped.exception is exc, eq=True)


class TestFlatMapPropagatesException:
    """Test flat_map() propagates exception from failure."""

    def test_flat_map_propagates_exception_on_failure(self) -> None:
        """Verify flat_map() preserves exception from failed result."""
        exc = TypeError("type error")
        result: r[int] = r[int].fail("error", exception=exc)
        flat_mapped: r[str] = result.flat_map(lambda x: r[str].ok(str(x)))
        tm.that(flat_mapped.is_failure, eq=True)
        tm.that(flat_mapped.exception is exc, eq=True)

    def test_flat_map_success_no_exception(self) -> None:
        """Verify flat_map() on success has no exception."""
        result: r[int] = r[int].ok(5)
        flat_mapped: r[str] = result.flat_map(lambda x: r[str].ok(str(x)))
        tm.that(flat_mapped.is_success, eq=True)
        tm.that(flat_mapped.value == "5", eq=True)
        tm.that(flat_mapped.exception is None, eq=True)

    def test_flat_map_chain_preserves_exception(self) -> None:
        """Verify exception preserved through flat_map chain."""
        exc = KeyError("missing key")
        result: r[int] = r[int].fail("error", exception=exc)
        flat_mapped: r[str] = result.flat_map(lambda x: r[int].ok(x + 1)).flat_map(
            lambda x: r[str].ok(str(x)),
        )
        tm.that(flat_mapped.is_failure, eq=True)
        tm.that(flat_mapped.exception is exc, eq=True)


class TestAltPropagatesException:
    """Test alt() propagates exception from failure."""

    def test_alt_propagates_exception(self) -> None:
        """Verify alt() preserves exception when transforming error."""
        exc = ValueError("original")
        result: r[int] = r[int].fail("error", exception=exc)
        altered: r[int] = result.map_error(lambda e: f"transformed: {e}")
        tm.that(altered.is_failure, eq=True)
        tm.that(altered.exception is exc, eq=True)
        tm.that(altered.error is not None and "transformed" in altered.error, eq=True)

    def test_alt_success_no_exception(self) -> None:
        """Verify alt() on success has no exception."""
        result: r[int] = r[int].ok(42)
        altered: r[int] = result.map_error(lambda e: f"error: {e}")
        tm.that(altered.is_success, eq=True)
        tm.that(altered.value == 42, eq=True)
        tm.that(altered.exception is None, eq=True)


class TestLashPropagatesException:
    """Test lash() propagates exception from failure."""

    def test_lash_propagates_exception(self) -> None:
        """Verify lash() preserves exception when recovering."""
        exc = RuntimeError("recovery needed")
        result: r[int] = r[int].fail("error", exception=exc)
        recovered = result.lash(lambda e: r[int].ok(0))
        tm.that(recovered.is_success, eq=True)
        tm.that(recovered.value == 0, eq=True)

    def test_lash_preserves_exception_on_recovery_failure(self) -> None:
        """Verify lash() preserves exception when recovery also fails."""
        exc = ValueError("original error")
        result: r[int] = r[int].fail("error", exception=exc)
        recovery_exc = RuntimeError("recovery failed")
        recovered: r[int] = result.lash(
            lambda e: r[int].fail(f"recovery failed: {e}", exception=recovery_exc),
        )
        tm.that(recovered.is_failure, eq=True)
        tm.that(recovered.exception is recovery_exc, eq=True)


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
        tm.that(result.is_failure, eq=True)
        tm.that(result.exception is exc, eq=True)

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
        tm.that(result.is_failure, eq=True)
        tm.that(result.error is not None, eq=True)
        tm.that("error 1" in result.error and "error 2" in result.error, eq=True)


class TestFromValidationCarriesException:
    """Test from_validation() carries exception."""

    def test_from_validation_carries_exception(self) -> None:
        """Verify from_validation() captures validation exception."""

        class User(BaseModel):
            name: str
            age: int

        invalid_data = {"name": "Alice", "age": "not_an_int"}
        result = r[User].from_validation(invalid_data, User)
        tm.that(result.is_failure, eq=True)
        tm.that(result.exception is not None, eq=True)
        tm.that(isinstance(result.exception, ValidationError), eq=True)


class TestErrorOrPatternUnchanged:
    """Test backward compatibility: .error or 'fallback' pattern (31 sites)."""

    def test_error_or_pattern_unchanged(self) -> None:
        """Verify .error or 'fallback' pattern still works (31 sites)."""
        result_success = r[int].ok(42)
        result_failure: r[int] = r[int].fail("error message")
        error_success = result_success.error or "fallback"
        error_failure = result_failure.error or "fallback"
        tm.that(error_success == "fallback", eq=True)
        tm.that(error_failure == "error message", eq=True)

    def test_error_or_pattern_with_exception(self) -> None:
        """Verify .error or pattern works with exception carrying."""
        exc = RuntimeError("runtime error")
        result: r[int] = r[int].fail("error", exception=exc)
        error_msg = result.error or "fallback"
        tm.that(error_msg == "error", eq=True)
        tm.that(result.exception is exc, eq=True)


class TestOkNoneGuardStillRaises:
    """Test ok(None) still raises ValueError guard."""

    def test_ok_none_succeeds(self) -> None:
        """Verify ok(None) creates valid success result."""
        result = r[int | None].ok(None)
        tm.that(result.is_success, eq=True)
        tm.that(result.value is None, eq=True)

    def test_ok_with_valid_value_succeeds(self) -> None:
        """Verify ok() with valid value succeeds."""
        value = 42
        result: r[int] = r[int].ok(value)
        tm.that(result.is_success, eq=True)
        tm.that(result.value == 42, eq=True)
        tm.that(result.exception is None, eq=True)


class TestMonadicOperationsUnchanged:
    """Test monadic operations work unchanged with exception carrying."""

    def test_monadic_operations_unchanged(self) -> None:
        """Verify monadic operations (map, flat_map, filter) work unchanged."""
        result: r[int] = r[int].ok(5)
        final: r[int] = (
            result
            .map(lambda x: x * 2)
            .flat_map(lambda x: r[int].ok(x + 1))
            .filter(lambda x: x > 5)
        )
        tm.that(final.is_success, eq=True)
        tm.that(final.value == 11, eq=True)
        tm.that(final.exception is None, eq=True)

    def test_monadic_chain_with_exception_in_middle(self) -> None:
        """Verify exception propagates through monadic chain."""
        exc = ValueError("chain error")
        result = r[int].ok(5)
        final: r[int] = (
            result
            .map(lambda x: x * 2)
            .flat_map(lambda x: r[int].fail("error", exception=exc))
            .map(lambda x: x + 1)
        )
        tm.that(final.is_failure, eq=True)
        tm.that(final.exception is exc, eq=True)

    def test_recover_with_exception(self) -> None:
        """Verify recover() works with exception carrying."""
        exc = RuntimeError("recovery needed")
        result: r[int] = r[int].fail("error", exception=exc)
        recovered: r[int] = result.recover(lambda e: 0)
        tm.that(recovered.is_success, eq=True)
        tm.that(recovered.value == 0, eq=True)

    def test_fold_with_exception(self) -> None:
        """Verify fold() works with exception carrying."""
        exc = ValueError("fold error")
        result: r[int] = r[int].fail("error", exception=exc)
        folded: str = result.fold(
            on_failure=lambda e: f"failed: {e}",
            on_success=lambda v: f"success: {v}",
        )
        tm.that(folded == "failed: error", eq=True)

    def test_tap_with_exception(self) -> None:
        """Verify tap() works with exception carrying."""
        exc = RuntimeError("tap error")
        result: r[int] = r[int].fail("error", exception=exc)
        side_effect_called = False

        def side_effect(x: int) -> None:
            nonlocal side_effect_called
            side_effect_called = True

        tapped: r[int] = result.tap(side_effect)
        tm.that(tapped.is_failure, eq=True)
        tm.that(not side_effect_called, eq=True)
        tm.that(tapped.exception is exc, eq=True)

    def test_tap_error_with_exception(self) -> None:
        """Verify tap_error() works with exception carrying."""
        exc = ValueError("tap_error test")
        result: r[int] = r[int].fail("error", exception=exc)
        side_effect_called = False

        def side_effect(e: str) -> None:
            nonlocal side_effect_called
            side_effect_called = True

        tapped: r[int] = result.tap_error(side_effect)
        tm.that(tapped.is_failure, eq=True)
        tm.that(side_effect_called, eq=True)
        tm.that(tapped.exception is exc, eq=True)


class TestExceptionPropertyAccess:
    """Test exception property access patterns."""

    def test_exception_property_none_on_success(self) -> None:
        """Verify exception property is None on success."""
        result: r[int] = r[int].ok(42)
        exc = result.exception
        tm.that(exc is None, eq=True)

    def test_exception_property_set_on_failure(self) -> None:
        """Verify exception property is set on failure."""
        exc = RuntimeError("test error")
        result: r[int] = r[int].fail("error", exception=exc)
        retrieved_exc = result.exception
        tm.that(retrieved_exc is exc, eq=True)
        tm.that(isinstance(retrieved_exc, RuntimeError), eq=True)

    def test_exception_property_type_check(self) -> None:
        """Verify exception property type is BaseException | None."""
        exc = ValueError("value error")
        result: r[int] = r[int].fail("error", exception=exc)
        retrieved_exc = result.exception
        tm.that(isinstance(retrieved_exc, BaseException), eq=True)
        tm.that(isinstance(retrieved_exc, ValueError), eq=True)

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
            result: r[int] = r[int].fail("error", exception=exc)
            tm.that(result.exception is exc, eq=True)
            tm.that(isinstance(result.exception, type(exc)), eq=True)
