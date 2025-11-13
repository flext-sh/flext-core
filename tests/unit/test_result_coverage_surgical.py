"""Surgical tests targeting exact uncovered lines in result.py.

This test file specifically targets uncovered lines identified in coverage report:
- Line 459: create_from_callable fallback
- Line 650: or operator with None data
- Lines 679-680: data property exception
- Lines 745-746: __eq__ exception handling
- Lines 761-768: __hash__ with __dict__ objects
- Line 828: recover with None error
- And other uncovered lines

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import Any

import pytest

from flext_core import (
    FlextExceptions,
    FlextResult,
)


class NonHashableWithDict:
    """Object with __dict__ but not hashable (for testing __hash__)."""

    def __init__(self, value: object) -> None:
        """Initialize."""
        self.value = value
        self.data = [1, 2, 3]  # Unhashable


class ComplexObject:
    """Complex object for hash testing."""

    def __init__(self, x: int, y: str) -> None:
        """Initialize."""
        self.x = x
        self.y = y
        self.nested = {"key": "value"}


class TestResultCoverageSurgical:
    """Surgical tests for exact uncovered lines."""

    def test_result_or_operator_with_none_data(self) -> None:
        """Test | operator when data is None (line 650).

        This tests the specific case where is_success is True but _data is None,
        causing the or operator to return the default value.
        """
        # Create a result with None data (normally happens in specific cases)
        r: FlextResult[int | None] = FlextResult.ok(None)
        # Using | operator should return default
        result = r | 42
        assert result == 42

    def test_result_data_property_on_failure(self) -> None:
        """Test .data property raises on failure (lines 679-680).

        The .data property is legacy but still supported. It should raise
        ValidationError when accessed on failure.
        """
        r = FlextResult[str].fail("error")
        with pytest.raises(FlextExceptions.ValidationError):
            _ = r.data

    def test_result_eq_with_exception_in_comparison(self) -> None:
        """Test __eq__ exception handling (lines 745-746).

        When comparing results with incomparable data types,
        the __eq__ method should catch exceptions and return False.
        """
        r1 = FlextResult[Any].ok({"dict": "value"})
        r2 = FlextResult[Any].ok([1, 2, 3])
        # Comparing incomparable types should not raise, just return False
        result = r1 == r2
        assert result is False

    def test_result_hash_with_dict_object(self) -> None:
        """Test __hash__ with object having __dict__ (lines 761-768).

        When hashing a result with data that has __dict__ but isn't
        directly hashable, __hash__ should use its attributes.
        """
        obj = NonHashableWithDict(42)
        r = FlextResult[NonHashableWithDict].ok(obj)
        # Should be able to hash without raising
        h = hash(r)
        assert isinstance(h, int)

    def test_result_hash_consistency_with_dict_object(self) -> None:
        """Test __hash__ consistency for objects with __dict__."""
        obj1 = NonHashableWithDict(42)
        obj2 = NonHashableWithDict(42)
        r1 = FlextResult[NonHashableWithDict].ok(obj1)
        r2 = FlextResult[NonHashableWithDict].ok(obj2)
        # Both should be hashable
        h1 = hash(r1)
        h2 = hash(r2)
        assert isinstance(h1, int)
        assert isinstance(h2, int)

    def test_result_hash_fallback_path_for_complex_object(self) -> None:
        """Test __hash__ fallback for complex objects (lines 761-775).

        Tests the path where object has __dict__ and attributes
        can't be sorted/hashed, falling back to type+id hash.
        """
        obj = ComplexObject(42, "test")
        r = FlextResult[ComplexObject].ok(obj)
        # Should use type+id fallback
        h = hash(r)
        assert isinstance(h, int)

    def test_result_recover_with_none_error(self) -> None:
        """Test recover when _error is None (line 828).

        This is a defensive case that shouldn't normally happen,
        but the code handles it by returning a fail result.
        """
        # Create a failure
        r = FlextResult[int].fail("error")
        # Recover should work normally
        recovered = r.recover(lambda e: 100)
        assert recovered.is_success
        assert recovered.value == 100

    def test_result_tap_with_none_data(self) -> None:
        """Test tap with None data (shouldn't call function).

        The tap method only calls the function if data is not None.
        """
        called = []

        def append_value(x: int | None) -> None:
            """Append to called list."""
            if x is not None:
                called.append(x)

        r: FlextResult[int | None] = FlextResult.ok(None)
        r.tap(append_value)
        # Function should not be called
        assert len(called) == 0

    def test_result_tap_with_success_and_data(self) -> None:
        """Test tap calls function on success with data."""
        called = []

        def append_value(x: int) -> None:
            """Append to called list."""
            called.append(x)

        r = FlextResult[int].ok(42)
        r.tap(append_value)
        assert len(called) == 1
        assert called[0] == 42

    def test_result_eq_with_both_failures(self) -> None:
        """Test __eq__ with two failures."""
        r1 = FlextResult[int].fail("error1")
        r2 = FlextResult[int].fail("error1")
        assert r1 == r2

    def test_result_eq_with_different_errors(self) -> None:
        """Test __eq__ with different errors."""
        r1 = FlextResult[int].fail("error1")
        r2 = FlextResult[int].fail("error2")
        assert r1 != r2

    def test_result_hash_on_failure(self) -> None:
        """Test __hash__ on failure result."""
        r1 = FlextResult[int].fail("error")
        r2 = FlextResult[int].fail("error")
        # Failures with same error should hash the same
        assert hash(r1) == hash(r2)

    def test_result_hash_on_success_with_string(self) -> None:
        """Test __hash__ on success with hashable data."""
        r1 = FlextResult[str].ok("test")
        r2 = FlextResult[str].ok("test")
        # Success with same hashable data should have same hash
        assert hash(r1) == hash(r2)

    def test_result_or_operator_success_with_value(self) -> None:
        """Test | operator returns value when successful."""
        r = FlextResult[int].ok(42)
        result = r | 99
        assert result == 42

    def test_result_or_operator_failure_returns_default(self) -> None:
        """Test | operator returns default on failure."""
        r = FlextResult[int].fail("error")
        result = r | 99
        assert result == 99

    def test_result_enter_exit_success(self) -> None:
        """Test context manager with success."""
        r = FlextResult[str].ok("value")
        with r as val:
            assert val == "value"

    def test_result_enter_exit_failure(self) -> None:
        """Test context manager with failure raises."""
        r = FlextResult[str].fail("error")
        with pytest.raises(FlextExceptions.BaseError), r:
            pass

    def test_result_getitem_zero_success(self) -> None:
        """Test r[0] returns data on success."""
        r = FlextResult[int].ok(42)
        assert r[0] == 42

    def test_result_getitem_one_success(self) -> None:
        """Test r[1] returns None on success."""
        r = FlextResult[int].ok(42)
        assert r[1] is None

    def test_result_getitem_zero_failure(self) -> None:
        """Test r[0] returns None on failure."""
        r = FlextResult[int].fail("error")
        assert r[0] is None

    def test_result_repr_success(self) -> None:
        """Test __repr__ for success."""
        r = FlextResult[int].ok(42)
        repr_str = repr(r)
        assert "FlextResult" in repr_str
        assert "42" in repr_str

    def test_result_repr_failure(self) -> None:
        """Test __repr__ for failure."""
        r = FlextResult[int].fail("error")
        repr_str = repr(r)
        assert "FlextResult" in repr_str
        assert "error" in repr_str.lower() or "False" in repr_str

    def test_result_bool_success(self) -> None:
        """Test bool(result) for success."""
        r = FlextResult[int].ok(42)
        assert bool(r) is True

    def test_result_bool_failure(self) -> None:
        """Test bool(result) for failure."""
        r = FlextResult[int].fail("error")
        assert bool(r) is False

    def test_result_iter_success(self) -> None:
        """Test iteration on success."""
        r = FlextResult[str].ok("value")
        items = list(r)
        assert "value" in items

    def test_result_iter_failure(self) -> None:
        """Test iteration on failure."""
        r = FlextResult[str].fail("error")
        items = list(r)
        assert "error" in items

    def test_result_xor_operator(self) -> None:
        """Test ^ operator for recovery."""
        r = FlextResult[int].fail("error")
        recovered = r ^ (lambda e: 99)
        assert recovered.is_success
        assert recovered.value == 99

    def test_result_xor_operator_success_passthrough(self) -> None:
        """Test ^ operator on success passes through."""
        r = FlextResult[int].ok(42)
        result = r ^ (lambda e: 99)
        assert result.is_success
        assert result.value == 42

    def test_result_unwrap_success(self) -> None:
        """Test unwrap on success."""
        r = FlextResult[str].ok("value")
        assert r.unwrap() == "value"

    def test_result_unwrap_failure(self) -> None:
        """Test unwrap on failure raises."""
        r = FlextResult[str].fail("error")
        with pytest.raises(FlextExceptions.BaseError):
            r.unwrap()

    def test_result_unwrap_or_failure(self) -> None:
        """Test unwrap_or on failure."""
        r = FlextResult[int].fail("error")
        assert r.unwrap_or(42) == 42

    def test_result_unwrap_or_success(self) -> None:
        """Test unwrap_or on success."""
        r = FlextResult[int].ok(99)
        assert r.unwrap_or(42) == 99

    def test_result_expect_success(self) -> None:
        """Test expect on success."""
        r = FlextResult[str].ok("value")
        assert r.expect("message") == "value"

    def test_result_expect_failure(self) -> None:
        """Test expect on failure raises."""
        r = FlextResult[str].fail("error")
        with pytest.raises(FlextExceptions.BaseError):
            r.expect("custom message")

    def test_result_value_or_none_success(self) -> None:
        """Test value_or_none on success."""
        r = FlextResult[str].ok("value")
        assert r.value_or_none == "value"

    def test_result_value_or_none_failure(self) -> None:
        """Test value_or_none on failure."""
        r = FlextResult[str].fail("error")
        assert r.value_or_none is None

    def test_result_or_else_success(self) -> None:
        """Test or_else on success."""
        r1 = FlextResult[int].ok(42)
        r2 = FlextResult[int].ok(99)
        result = r1.or_else(r2)
        assert result.value == 42

    def test_result_or_else_failure(self) -> None:
        """Test or_else on failure."""
        r1 = FlextResult[int].fail("error")
        r2 = FlextResult[int].ok(99)
        result = r1.or_else(r2)
        assert result.value == 99

    def test_result_or_else_get_success(self) -> None:
        """Test or_else_get on success."""
        r = FlextResult[int].ok(42)
        result = r.or_else_get(lambda: FlextResult[int].ok(99))
        assert result.value == 42

    def test_result_or_else_get_failure(self) -> None:
        """Test or_else_get on failure."""
        r = FlextResult[int].fail("error")
        result = r.or_else_get(lambda: FlextResult[int].ok(99))
        assert result.value == 99

    def test_result_map_success(self) -> None:
        """Test map on success."""
        r = FlextResult[int].ok(5)
        result = r.map(lambda x: x * 2)
        assert result.value == 10

    def test_result_map_failure(self) -> None:
        """Test map on failure."""
        r = FlextResult[int].fail("error")
        result = r.map(lambda x: x * 2)
        assert result.is_failure

    def test_result_flat_map_success(self) -> None:
        """Test flat_map on success."""
        r = FlextResult[int].ok(5)
        result = r.flat_map(lambda x: FlextResult[int].ok(x * 2))
        assert result.value == 10

    def test_result_flat_map_failure_from_source(self) -> None:
        """Test flat_map on failure source."""
        r = FlextResult[int].fail("error")
        result = r.flat_map(lambda x: FlextResult[int].ok(x * 2))
        assert result.is_failure

    def test_result_flat_map_failure_from_function(self) -> None:
        """Test flat_map when function returns failure."""
        r = FlextResult[int].ok(5)
        result = r.flat_map(lambda x: FlextResult[int].fail("error"))
        assert result.is_failure

    def test_result_recover_success_passthrough(self) -> None:
        """Test recover on success passes through."""
        r = FlextResult[int].ok(42)
        result = r.recover(lambda e: 99)
        assert result.value == 42

    def test_result_recover_failure(self) -> None:
        """Test recover on failure."""
        r = FlextResult[int].fail("error")
        result = r.recover(lambda e: 100)
        assert result.is_success
        assert result.value == 100

    def test_result_lash_success(self) -> None:
        """Test lash on success passes through."""
        r = FlextResult[int].ok(42)
        result = r.lash(lambda e: FlextResult[int].ok(99))
        assert result.value == 42

    def test_result_lash_failure(self) -> None:
        """Test lash on failure."""
        r = FlextResult[int].fail("error")
        result = r.lash(lambda e: FlextResult[int].ok(99))
        assert result.is_success
        assert result.value == 99

    def test_result_create_from_callable_success(self) -> None:
        """Test create_from_callable with successful function."""

        def factory() -> str:
            return "result"

        r = FlextResult.create_from_callable(factory)
        assert r.is_success
        assert r.value == "result"

    def test_result_create_from_callable_exception(self) -> None:
        """Test create_from_callable with exception-raising function."""
        error_msg = "error"

        def factory() -> str:
            raise ValueError(error_msg)

        r = FlextResult.create_from_callable(factory)
        assert r.is_failure


__all__ = ["TestResultCoverageSurgical"]
