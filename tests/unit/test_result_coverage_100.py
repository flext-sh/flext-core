"""Comprehensive tests for r module - Real functionality only.

Module: flext_core.result.r[T]
Scope: All public r operations, creation, transformation, chaining patterns
Pattern: Railway-Oriented, Monadic operations, Type validation, Error handling

Tests validate:
- Result creation (ok/fail) with all valid types
- Monadic operations (map, flat_map, lash, alt)
- Railroad pattern (flow_through, chaining)
- Error handling (lash for recovery, alt for error mapping)
- Type checking and validation
- Edge cases with None/empty values
- Integration with returns library (Maybe, IO, IOResult)
- Boolean and operator overloads
- Context manager support
- Decorator and utility methods

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math

import pytest

from flext_core import m, r, t


class _ResultAssertions:
    @staticmethod
    def assert_success[T](result: r[T]) -> None:
        assert isinstance(result, r)
        assert result.is_success

    @staticmethod
    def assert_failure[T](result: r[T]) -> None:
        assert isinstance(result, r)
        assert not result.is_success

    @staticmethod
    def assert_success_with_value[T](result: r[T], expected: T) -> None:
        _ResultAssertions.assert_success(result)
        assert result.value == expected

    @staticmethod
    def assert_failure_with_error[T](
        result: r[T],
        expected_error: str,
    ) -> None:
        _ResultAssertions.assert_failure(result)
        assert isinstance(result, r)
        assert result.error == expected_error


class TestrCoverage:
    """Comprehensive test suite for r - ALL REAL FUNCTIONALITY ONLY."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("test_string", "test_string"),
            (42, 42),
            (math.pi, math.pi),
            ({"key": "value"}, {"key": "value"}),
            ([1, 2, 3], [1, 2, 3]),
            (True, True),
        ],
    )
    def test_ok_creates_success_with_various_types(
        self,
        value: t.GeneralValueType,
        expected: t.GeneralValueType,
    ) -> None:
        """Test creating success results with different value types."""
        result = r[t.GeneralValueType].ok(value)
        _ResultAssertions.assert_success(result)
        assert result.value == expected

    def test_ok_accepts_none_value(self) -> None:
        """Test that ok(None) creates valid success."""
        result = r[t.GeneralValueType].ok(None)
        assert result.is_success
        with pytest.raises(RuntimeError, match="Invariant violation"):
            _ = result.value

    def test_fail_creates_failure_with_message(self) -> None:
        """Test creating failure results."""
        result: r[str] = r[str].fail("Test error")
        _ResultAssertions.assert_failure_with_error(result, expected_error="Test error")

    def test_fail_with_error_code(self) -> None:
        """Test creating failure with error code."""
        result: r[str] = r[str].fail("Error", error_code="TEST_CODE")
        _ResultAssertions.assert_failure(result)
        assert result.error_code == "TEST_CODE"

    def test_fail_with_error_data(self) -> None:
        """Test creating failure with error data."""
        error_data: m.ConfigMap = m.ConfigMap(root={"status": "failed", "count": 5})
        result: r[str] = r[str].fail("Error", error_data=error_data)
        _ResultAssertions.assert_failure(result)
        assert result.error_data == error_data

    def test_value_property_on_success(self) -> None:
        """Test accessing value on success result."""
        result = r[str].ok("test")
        _ResultAssertions.assert_success_with_value(result, "test")

    def test_value_property_on_failure_raises(self) -> None:
        """Test that value property raises on failure."""
        result: r[str] = r[str].fail("error")
        with pytest.raises(RuntimeError, match="Cannot access value of failed result"):
            _ = result.value

    def test_value_property(self) -> None:
        """Test that value property works correctly."""
        result = r[str].ok("test")
        _ResultAssertions.assert_success_with_value(result, "test")

    def test_error_property_on_failure(self) -> None:
        """Test accessing error on failure result."""
        result: r[str] = r[str].fail("test_error")
        _ResultAssertions.assert_failure_with_error(result, expected_error="test_error")

    def test_error_property_on_success_returns_none(self) -> None:
        """Test that error property returns None on success."""
        result = r[str].ok("test")
        assert result.error is None

    def test_unwrap_success(self) -> None:
        """Test unwrap on success result."""
        result = r[str].ok("test")
        assert result.value == "test"

    def test_unwrap_failure_raises(self) -> None:
        """Test that unwrap raises on failure."""
        result: r[str] = r[str].fail("error")
        with pytest.raises(RuntimeError, match="Cannot access value of failed result"):
            result.value

    def test_unwrap_or_success(self) -> None:
        """Test unwrap_or returns value on success."""
        result = r[str].ok("test")
        assert result.unwrap_or("default") == "test"

    def test_unwrap_or_failure(self) -> None:
        """Test unwrap_or returns default on failure."""
        result: r[str] = r[str].fail("error")
        assert result.unwrap_or("default") == "default"

    def test_map_success(self) -> None:
        """Test map operation on success."""
        result = r[int].ok(5).map(lambda x: x * 2)
        _ResultAssertions.assert_success_with_value(result, 10)

    def test_map_failure_skips_function(self) -> None:
        """Test that map skips function on failure."""
        result = r[int].fail("error").map(lambda x: x * 2)
        _ResultAssertions.assert_failure_with_error(result, expected_error="error")

    def test_map_chain_multiple(self) -> None:
        """Test chaining multiple map operations."""

        def double(x: int) -> int:
            return x * 2

        def add_three(x: int) -> int:
            return x + 3

        def to_str(x: int) -> str:
            return str(x)

        result = r[int].ok(5).map(double).map(add_three).map(to_str)
        _ResultAssertions.assert_success_with_value(result, "13")

    def test_flat_map_success(self) -> None:
        """Test flat_map chaining results."""

        def double_in_result(x: int) -> r[int]:
            return r[int].ok(x * 2)

        result = r[int].ok(5).flat_map(double_in_result)
        _ResultAssertions.assert_success(result)
        assert result.value == 10

    def test_flat_map_failure_propagates(self) -> None:
        """Test that flat_map propagates failure from inner result."""

        def failing_op(x: int) -> r[str]:
            return r[str].fail("Inner failed")

        result = r[int].ok(5).flat_map(failing_op)
        _ResultAssertions.assert_failure(result)
        assert result.error == "Inner failed"

    def test_flat_map_initial_failure_skips(self) -> None:
        """Test that flat_map skips on initial failure."""

        def double_in_result(x: int) -> r[int]:
            return r[int].ok(x * 2)

        result = r[int].fail("error").flat_map(double_in_result)
        _ResultAssertions.assert_failure(result)
        assert result.error == "error"

    def test_filter_success_when_predicate_true(self) -> None:
        """Test filter passes when predicate is true."""
        result = r[int].ok(5).filter(lambda x: x > 3)
        _ResultAssertions.assert_success_with_value(result, 5)

    def test_filter_failure_when_predicate_false(self) -> None:
        """Test filter fails when predicate is false."""
        result = r[int].ok(5).filter(lambda x: x > 10)
        _ResultAssertions.assert_failure_with_error(
            result,
            "Value did not pass filter predicate",
        )

    def test_filter_failure_skips_predicate(self) -> None:
        """Test that filter skips on failure."""
        result: r[int] = r[int].fail("error").filter(lambda x: x > 3)
        _ResultAssertions.assert_failure(result)
        assert result.error == "error"

    def test_alt_maps_error_message(self) -> None:
        """Test alt maps error message on failure."""
        result: r[str] = r[str].fail("original").map_error(lambda e: f"Modified: {e}")
        _ResultAssertions.assert_failure_with_error(result, "Modified: original")

    def test_alt_skips_on_success(self) -> None:
        """Test that alt skips on success."""
        result = r[str].ok("test").map_error(lambda e: f"Modified: {e}")
        _ResultAssertions.assert_success_with_value(result, "test")

    def test_lash_recovery_on_failure(self) -> None:
        """Test lash recovers from failure."""

        def recovery(error: str) -> r[str]:
            return r[str].ok(f"Recovered from: {error}")

        failed: r[str] = r[str].fail("error")
        result: r[str] = failed.lash(recovery)
        _ResultAssertions.assert_success_with_value(result, "Recovered from: error")

    def test_lash_skips_on_success(self) -> None:
        """Test that lash skips on success."""

        def recovery(error: str) -> r[str]:
            return r[str].fail("recovery failed")

        result = r[str].ok("test").lash(recovery)
        _ResultAssertions.assert_success_with_value(result, "test")

    def test_lash_failure_in_recovery(self) -> None:
        """Test that lash failure in recovery returns new failure."""

        def failing_recovery(error: str) -> r[str]:
            return r[str].fail("recovery also failed")

        failed: r[str] = r[str].fail("original")
        result: r[str] = failed.lash(failing_recovery)
        _ResultAssertions.assert_failure_with_error(result, "recovery also failed")

    def test_flow_through_chain_success(self) -> None:
        """Test flow_through chains multiple operations."""

        def double(x: int) -> r[int]:
            return r[int].ok(x * 2)

        def add_ten(x: int) -> r[int]:
            return r[int].ok(x + 10)

        result = r[int].ok(5).flow_through(double, add_ten)
        _ResultAssertions.assert_success(result)
        assert result.value == 20

    def test_flow_through_stops_on_failure(self) -> None:
        """Test flow_through stops processing on failure."""

        def double(x: str) -> r[str]:
            if x.isdigit():
                return r[str].ok(str(int(x) * 2))
            return r[str].fail("Not a numeric string")

        def add_ten(x: str) -> r[str]:
            return r[str].fail("Should not reach here")

        result = r[str].ok("test").flow_through(double, add_ten)
        _ResultAssertions.assert_failure(result)
        assert result.error == "Not a numeric string"

    def test_safe_decorator_success(self) -> None:
        """Test safe decorator wraps successful function."""

        def success_func() -> str:
            return "success"

        wrapped_func = r.safe(success_func)
        result: r[str] = wrapped_func()
        _ResultAssertions.assert_success_with_value(result, "success")

    def test_safe_decorator_catches_exception(self) -> None:
        """Test safe decorator catches exceptions."""
        error_msg = "Function failed"

        def failing_func() -> str:
            raise ValueError(error_msg)

        wrapped_func = r.safe(failing_func)
        result: r[str] = wrapped_func()
        _ResultAssertions.assert_failure(result)
        assert result.error is not None and error_msg in result.error

    def test_create_from_callable_success(self) -> None:
        """Test create_from_callable with successful callable."""

        def success_func() -> str:
            return "success"

        result = r[str].create_from_callable(success_func)
        _ResultAssertions.assert_success_with_value(result, "success")

    def test_create_from_callable_exception(self) -> None:
        """Test create_from_callable catches exceptions."""
        error_msg = "Function failed"

        def failing_func() -> str:
            raise ValueError(error_msg)

        result = r[str].create_from_callable(failing_func)
        _ResultAssertions.assert_failure(result)
        assert result.error is not None and error_msg in result.error

    def test_create_from_callable_with_error_code(self) -> None:
        """Test create_from_callable with error code."""
        error_msg = "Error"

        def failing_func() -> str:
            raise ValueError(error_msg)

        result = r[str].create_from_callable(failing_func, error_code="TEST_ERROR")
        _ResultAssertions.assert_failure(result)
        assert result.error_code == "TEST_ERROR"

    def test_traverse_success(self) -> None:
        """Test traverse with successful mapping."""

        def double(x: int) -> r[int]:
            return r[int].ok(x * 2)

        items = [1, 2, 3]
        result = r[list[int]].traverse(items, double)
        _ResultAssertions.assert_success_with_value(result, [2, 4, 6])

    def test_traverse_failure_propagates(self) -> None:
        """Test traverse stops on first failure."""

        def double(x: int) -> r[int]:
            if x == 2:
                return r[int].fail("Found 2")
            return r[int].ok(x * 2)

        items = [1, 2, 3]
        result = r[list[int]].traverse(items, double)
        _ResultAssertions.assert_failure_with_error(result, "Found 2")

    def test_accumulate_errors_all_success(self) -> None:
        """Test accumulate_errors with all successes."""
        results = [r[int].ok(1), r[int].ok(2), r[int].ok(3)]
        combined = r[list[int]].accumulate_errors(*results)
        _ResultAssertions.assert_success_with_value(combined, [1, 2, 3])

    def test_accumulate_errors_with_failures(self) -> None:
        """Test accumulate_errors collects all error messages."""
        results = [r[int].ok(1), r[int].fail("error1"), r[int].fail("error2")]
        combined = r[list[int]].accumulate_errors(*results)
        _ResultAssertions.assert_failure(combined)
        assert combined.error is not None
        assert "error1" in combined.error
        assert "error2" in combined.error

    def test_with_resource_success(self) -> None:
        """Test with_resource executes operation."""
        resources_created: list[m.ConfigMap] = []

        def factory() -> m.ConfigMap:
            resource: m.ConfigMap = m.ConfigMap(root={"id": 1})
            resources_created.append(resource)
            return resource

        def operation(resource: m.ConfigMap) -> r[str]:
            return r[str].ok("success")

        result = r[str].with_resource(factory, operation)
        _ResultAssertions.assert_success_with_value(result, "success")
        assert len(resources_created) == 1

    def test_with_resource_with_cleanup(self) -> None:
        """Test with_resource executes cleanup even on success."""
        cleanups_called: list[bool] = []

        def factory() -> m.ConfigMap:
            return m.ConfigMap(root={"id": 1})

        def operation(resource: m.ConfigMap) -> r[str]:
            return r[str].ok("success")

        def cleanup(resource: m.ConfigMap) -> None:
            cleanups_called.append(True)

        result = r[str].with_resource(factory, operation, cleanup=cleanup)
        _ResultAssertions.assert_success(result)
        assert len(cleanups_called) == 1

    def test_bool_success_is_true(self) -> None:
        """Test that success result is truthy."""
        result = r[str].ok("test")
        assert bool(result) is True

    def test_bool_failure_is_false(self) -> None:
        """Test that failure result is falsy."""
        result: r[str] = r[str].fail("error")
        assert bool(result) is False

    def test_or_operator(self) -> None:
        """Test | operator for default values."""
        result: r[str] = r[str].fail("error")
        value = result | "default"
        assert value == "default"

    def test_or_operator_success(self) -> None:
        """Test | operator returns value on success."""
        result = r[str].ok("test")
        value = result | "default"
        assert value == "test"

    def test_context_manager_entry(self) -> None:
        """Test context manager __enter__ returns self."""
        result = r[str].ok("test")
        with result as ctx:
            assert ctx is result

    def test_context_manager_exit_success(self) -> None:
        """Test context manager __exit__ succeeds."""
        result = r[str].ok("test")
        with result:
            _ResultAssertions.assert_success(result)

    def test_context_manager_exit_failure(self) -> None:
        """Test context manager __exit__ on failure."""
        result: r[str] = r[str].fail("error")
        with result:
            _ResultAssertions.assert_failure(result)

    def test_repr_success(self) -> None:
        """Test __repr__ for success."""
        result = r[str].ok("test")
        assert repr(result) == "r[T].ok('test')"

    def test_repr_failure(self) -> None:
        """Test __repr__ for failure."""
        result: r[str] = r[str].fail("error")
        assert repr(result) == "r[T].fail('error')"

    def test_error_codes_metadata(self) -> None:
        """Test error code and error data metadata."""
        error_data: m.ConfigMap = m.ConfigMap(root={"details": "something"})
        result: r[str] = r[str].fail(
            "Error",
            error_code="CODE_123",
            error_data=error_data,
        )
        assert result.error_code == "CODE_123"
        assert result.error_data == error_data

    def test_empty_string_vs_none_error(self) -> None:
        """Test empty string error vs None."""
        result: r[str] = r[str].fail("")
        _ResultAssertions.assert_failure_with_error(result, "")

    def test_large_value_handling(self) -> None:
        """Test handling of large values."""
        large_list = list(range(1000))
        result = r[list[int]].ok(large_list)
        _ResultAssertions.assert_success(result)
        assert len(result.value) == 1000

    def test_complex_chaining_scenario(self) -> None:
        """Test complex chaining of operations."""

        def double(x: int) -> int:
            return x * 2

        def add_three(x: int) -> r[int]:
            return r[int].ok(x + 3)

        def is_gt_10(x: int) -> bool:
            return x > 10

        def to_str(x: int) -> str:
            return str(x)

        result = (
            r[int].ok(5).map(double).flat_map(add_three).filter(is_gt_10).map(to_str)
        )
        _ResultAssertions.assert_success_with_value(result, "13")


__all__ = ["TestrCoverage"]
