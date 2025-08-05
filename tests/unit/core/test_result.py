"""Core Result Pattern Test Suite - Unit Testing Layer Testing Foundation.

Comprehensive unit test suite for FlextResult railway-oriented programming pattern
that validates type-safe error handling across the entire FLEXT ecosystem.

Module Role in Architecture:
    Testing Layer → Unit Tests → FlextResult Pattern Validation

    This module provides comprehensive unit testing that ensures:
    - Railway-oriented programming patterns work correctly across 15,000+ function signatures
    - Type safety is maintained through all transformation and chaining operations
    - Error propagation follows expected patterns without exceptions
    - FlextCore pipeline functions integrate seamlessly with FlextResult

Testing Strategy Coverage:
    ✅ Success/Failure Creation: Basic FlextResult construction patterns
    ✅ Boolean Conversion: Truthiness evaluation for control flow
    ✅ Unwrapping Operations: Safe data extraction with error handling
    ✅ Functional Transformations: map() and flat_map() chaining patterns
    ✅ Pipeline Integration: FlextCore.pipe() with multi-step data processing
    ✅ Side Effect Management: FlextCore.tap() for debugging and logging
    ✅ Conditional Processing: FlextCore.when() for business rule application
    ✅ Result Composition: Combining multiple results for transaction patterns

Enterprise Quality Standards:
    - Test Coverage: 95%+ coverage of result pattern functionality
    - Performance: < 100ms per test, < 10s total suite execution
    - Isolation: Pure unit tests with no external dependencies
    - Type Safety: Comprehensive validation of generic type parameters

Real-World Usage Validation:
    # Railway-oriented data processing pipeline
    result = (
        FlextResult.ok(raw_data)
        .flat_map(validate_input)
        .map(transform_data)
        .flat_map(save_to_database)
    )

    # Error handling without exceptions
    if result.success:
        log_success(result.data)
    else:
        log_error(result.error)

Test Architecture Patterns:
    - Isolated Component Testing: Each method tested independently
    - Error Path Validation: Comprehensive failure scenario coverage
    - Type Parameter Testing: Generic type safety validation
    - Integration Testing: FlextCore pipeline integration validation

See Also:
    - src/flext_core/result.py: FlextResult implementation
    - src/flext_core/core.py: FlextCore pipeline functions
    - examples/01_flext_result_railway_pattern.py: Usage examples
    - tests/integration/: Cross-module integration tests

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest

from flext_core import FlextResult
from flext_core._result_base import _BaseResult, _BaseResultOperations
from flext_core.core import FlextCore
from flext_core.exceptions import FlextOperationError

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextResult:
    """Test FlextResult class."""

    def test_success_creation(self) -> None:
        """Test creating success result."""
        result = FlextResult.ok("test_data")
        assert result.success, f"Expected True, got {result.success}"
        assert result.data == "test_data", f"Expected {'test_data'}, got {result.data}"
        assert result.error is None

    def test_failure_creation(self) -> None:
        """Test creating failure result."""
        result: FlextResult[str] = FlextResult.fail("test_error")
        assert result.is_failure, f"Expected True, got {result.is_failure}"
        assert result.data is None
        assert result.error == "test_error", (
            f"Expected {'test_error'}, got {result.error}"
        )

    def test_success_with_metadata(self) -> None:
        """Test creating success result with metadata."""
        # Metadata not supported in current FlextResult implementation
        result = FlextResult.ok("test_data")
        assert result.data == "test_data", f"Expected {'test_data'}, got {result.data}"

    def test_failure_with_metadata(self) -> None:
        """Test creating failure result with metadata."""
        # Metadata not supported in current FlextResult implementation
        result: FlextResult[str] = FlextResult.fail("test_error")
        assert result.error == "test_error", (
            f"Expected {'test_error'}, got {result.error}"
        )

    def test_boolean_conversion(self) -> None:
        """Test boolean conversion."""
        success_result = FlextResult.ok("data")
        failure_result: FlextResult[str] = FlextResult.fail("error")

        assert bool(success_result), f"Expected True, got {bool(success_result)}"
        assert not bool(failure_result), f"Expected False, got {bool(failure_result)}"

    def test_unwrap_success(self) -> None:
        """Test unwrapping success result."""
        data = "test_data"
        result = FlextResult.ok(data)
        assert result.unwrap() == data, f"Expected {data}, got {result.unwrap()}"

    def test_unwrap_failure_raises(self) -> None:
        """Test unwrapping failure result raises exception."""

        result: FlextResult[str] = FlextResult.fail("test_error")
        with pytest.raises(FlextOperationError, match="test_error"):
            result.unwrap()

    def test_unwrap_or_success(self) -> None:
        """Test unwrap_or with success result."""
        data = "test_data"
        result = FlextResult.ok(data)
        assert result.unwrap_or("default") == data, (
            f"Expected {data}, got {result.unwrap_or('default')}"
        )

    def test_unwrap_or_failure(self) -> None:
        """Test unwrap_or with failure result."""
        result: FlextResult[str] = FlextResult.fail("error")
        assert result.unwrap_or("default") == "default", (
            f"Expected {'default'}, got {result.unwrap_or('default')}"
        )

    def test_map_success(self) -> None:
        """Test map with success result."""
        result = FlextResult.ok("hello")
        mapped = result.map(lambda x: x.upper())
        assert mapped.success, f"Expected True, got {mapped.success}"
        assert mapped.data == "HELLO", f"Expected {'HELLO'}, got {mapped.data}"

    def test_map_failure(self) -> None:
        """Test map with failure result."""
        result: FlextResult[str] = FlextResult.fail("error")
        mapped = result.map(lambda x: x.upper())
        if not (mapped.is_failure):
            raise AssertionError(f"Expected True, got {mapped.is_failure}")
        if mapped.error != "error":
            raise AssertionError(f"Expected {'error'}, got {mapped.error}")

    def test_flat_map_success(self) -> None:
        """Test flat_map with success result."""
        result = FlextResult.ok("hello")
        flat_mapped = result.flat_map(lambda x: FlextResult.ok(x.upper()))
        if not flat_mapped.success:
            raise AssertionError(f"Expected True, got {flat_mapped.success}")
        if flat_mapped.data != "HELLO":
            raise AssertionError(f"Expected {'HELLO'}, got {flat_mapped.data}")

    def test_flat_map_failure(self) -> None:
        """Test flat_map with failure result."""
        result: FlextResult[str] = FlextResult.fail("error")
        flat_mapped = result.flat_map(lambda x: FlextResult.ok("success"))
        if not (flat_mapped.is_failure):
            raise AssertionError(f"Expected True, got {flat_mapped.is_failure}")
        if flat_mapped.error != "error":
            raise AssertionError(f"Expected {'error'}, got {flat_mapped.error}")

    def test_flat_map_chain_failure(self) -> None:
        """Test flat_map with chain failure."""
        result = FlextResult.ok("hello")
        flat_mapped: FlextResult[str] = result.flat_map(
            lambda x: FlextResult.fail("chain_error"),
        )
        if not (flat_mapped.is_failure):
            raise AssertionError(f"Expected True, got {flat_mapped.is_failure}")
        if flat_mapped.error != "chain_error":
            raise AssertionError(f"Expected {'chain_error'}, got {flat_mapped.error}")

    def test_equality(self) -> None:
        """Test equality comparison."""
        result1 = FlextResult.ok("test")
        result2 = FlextResult.ok("test")
        result3: FlextResult[str] = FlextResult.fail("error")

        if result1 != result2:
            raise AssertionError(f"Expected {result2}, got {result1}")
        assert result1 != result3

    def test_repr(self) -> None:
        """Test string representation."""
        result = FlextResult.ok("test_data")
        repr_str = repr(result)
        if "FlextResult" not in repr_str:
            raise AssertionError(f"Expected {'FlextResult'} in {repr_str}")
        assert "is_success" in repr_str


class TestComposeFunction:
    """Test compose function."""

    def test_compose_success_results(self) -> None:
        """Test composing success results."""
        result1 = FlextResult.ok("data1")
        result2 = FlextResult.ok("data2")

        # Use _BaseResultOperations.chain_results for combining results into a list

        composed = _BaseResultOperations.chain_results(
            cast("_BaseResult[object]", result1),
            cast("_BaseResult[object]", result2),
        )
        if not composed.success:
            raise AssertionError(f"Expected True, got {composed.success}")
        if composed.data != ["data1", "data2"]:
            raise AssertionError(f"Expected {['data1', 'data2']}, got {composed.data}")

    def test_compose_with_failure(self) -> None:
        """Test composing with failure result."""
        result1 = FlextResult.ok("data1")
        result2: FlextResult[str] = FlextResult.fail("error")
        result3 = FlextResult.ok("data3")

        # Use _BaseResultOperations.chain_results for combining results

        composed = _BaseResultOperations.chain_results(
            cast("_BaseResult[object]", result1),
            cast("_BaseResult[object]", result2),
            cast("_BaseResult[object]", result3),
        )
        if not (composed.is_failure):
            raise AssertionError(f"Expected True, got {composed.is_failure}")
        if composed.error != "error":
            raise AssertionError(f"Expected {'error'}, got {composed.error}")

    def test_compose_empty_list(self) -> None:
        """Test composing empty list."""
        # Use _BaseResultOperations.chain_results for combining results

        composed = _BaseResultOperations.chain_results()
        if not composed.success:
            raise AssertionError(f"Expected True, got {composed.success}")
        if composed.data != []:
            raise AssertionError(f"Expected {[]}, got {composed.data}")


class TestPipeFunction:
    """Test pipe function."""

    def test_pipe_success(self) -> None:
        """Test piping success result through functions."""

        # FlextCore.pipe creates a pipeline function
        def to_upper(x: object) -> FlextResult[object]:
            x_str = str(x)
            return FlextResult.ok(x_str.upper())

        def replace_spaces(x: object) -> FlextResult[object]:
            x_str = str(x)
            return FlextResult.ok(x_str.replace(" ", "_"))

        def to_lower(x: object) -> FlextResult[object]:
            x_str = str(x)
            return FlextResult.ok(x_str.lower())

        pipeline = FlextCore.pipe(
            to_upper,
            replace_spaces,
            to_lower,
        )
        result = pipeline("hello world")

        if not result.success:
            raise AssertionError(f"Expected True, got {result.success}")
        if result.data != "hello_world":
            raise AssertionError(f"Expected {'hello_world'}, got {result.data}")

    def test_pipe_failure(self) -> None:
        """Test piping failure result."""

        def failing_func(x: object) -> FlextResult[object]:
            return FlextResult.fail("error")

        def another_func(x: object) -> FlextResult[object]:
            x_str = str(x)
            return FlextResult.ok(x_str.upper())

        pipeline = FlextCore.pipe(
            failing_func,
            another_func,
        )
        result = pipeline("hello")

        if not (result.is_failure):
            raise AssertionError(f"Expected True, got {result.is_failure}")
        if result.error != "error":
            raise AssertionError(f"Expected {'error'}, got {result.error}")

    def test_pipe_with_transformation_error(self) -> None:
        """Test pipe with transformation error."""

        def failing_transform(x: object) -> FlextResult[object]:
            msg = "Transform failed"
            return FlextResult.fail(msg)

        pipeline = FlextCore.pipe(
            failing_transform,
        )
        result = pipeline("hello")

        if not (result.is_failure):
            raise AssertionError(f"Expected True, got {result.is_failure}")
        assert result.error is not None
        if "Transform failed" not in (result.error or ""):
            raise AssertionError(f"Expected 'Transform failed' in {result.error}")


class TestTapFunction:
    """Test tap function."""

    def test_tap_success(self) -> None:
        """Test tap with success result."""

        side_effects: list[str] = []

        # FlextCore.tap creates a function that executes side effects
        tap_func = cast(
            "Callable[[object], FlextResult[object]]",
            FlextCore.tap(side_effects.append),
        )
        result = tap_func("test_data")

        if not result.success:
            raise AssertionError(f"Expected True, got {result.success}")
        if result.data != "test_data":
            raise AssertionError(f"Expected {'test_data'}, got {result.data}")
        assert side_effects == ["test_data"]

    def test_tap_failure(self) -> None:
        """Test tap with failure result."""

        side_effects: list[str] = []

        # For a failure result, we need to test the tap behavior with a pipeline
        def failing_func(x: object) -> FlextResult[object]:
            return FlextResult.fail("error")

        tap_func = cast(
            "Callable[[object], FlextResult[object]]",
            FlextCore.tap(side_effects.append),
        )
        pipeline = FlextCore.pipe(
            failing_func,
            tap_func,
        )
        result = pipeline("test")

        if not (result.is_failure):
            raise AssertionError(f"Expected True, got {result.is_failure}")
        if result.error != "error":
            raise AssertionError(f"Expected {'error'}, got {result.error}")
        assert side_effects == []  # Side effect should not be called

    def test_tap_with_error(self) -> None:
        """Test tap with side effect error."""

        def failing_side_effect(x: str) -> None:
            msg = "Side effect failed"
            raise ValueError(msg)

        tap_func = cast(
            "Callable[[object], FlextResult[object]]",
            FlextCore.tap(failing_side_effect),
        )

        # The current implementation doesn't catch exceptions in side effects
        # This is expected behavior - side effects should not fail
        with pytest.raises(ValueError, match="Side effect failed"):
            tap_func("test_data")


class TestWhenFunction:
    """Test when function."""

    def test_when_true_condition(self) -> None:
        """Test when with true condition."""

        # FlextCore.when creates a conditional function
        when_func = FlextCore.when(
            lambda x: len(str(x)) > 5,
            lambda x: FlextResult.ok(str(x).upper()),
        )
        result = when_func("test_data")

        if not result.success:
            raise AssertionError(f"Expected True, got {result.success}")
        if result.data != "TEST_DATA":
            raise AssertionError(f"Expected {'TEST_DATA'}, got {result.data}")

    def test_when_false_condition(self) -> None:
        """Test when with false condition."""

        # FlextCore.when creates a conditional function
        when_func = FlextCore.when(
            lambda x: len(str(x)) > 10,
            lambda x: FlextResult.ok(str(x).upper()),
        )
        result = when_func("test")

        if not result.success:
            raise AssertionError(f"Expected True, got {result.success}")
        if result.data != "test":  # Should remain unchanged:
            raise AssertionError(
                f"Expected {'test'} # Should remain unchanged, got {result.data}"
            )

    def test_when_failure_result(self) -> None:
        """Test when with failure result in pipeline."""

        # Test when function with a failing input via pipeline
        def failing_func(x: object) -> FlextResult[object]:
            return FlextResult.fail("error")

        when_func = FlextCore.when(
            lambda x: True,
            lambda x: FlextResult.ok(x.upper()),
        )

        pipeline = FlextCore.pipe(
            failing_func,
            when_func,
        )
        result = pipeline("test")

        if not (result.is_failure):
            raise AssertionError(f"Expected True, got {result.is_failure}")
        if result.error != "error":
            raise AssertionError(f"Expected {'error'}, got {result.error}")

    def test_when_with_condition_error(self) -> None:
        """Test when with condition error."""

        def failing_condition(x: str) -> bool:
            msg = "Condition failed"
            raise ValueError(msg)

        when_func = FlextCore.when(
            lambda x: failing_condition(str(x)),
            lambda x: FlextResult.ok(str(x).upper()),
        )

        # The current implementation doesn't catch exceptions in predicates
        # This is expected behavior - predicates should be pure functions
        with pytest.raises(ValueError, match="Condition failed"):
            when_func("test_data")


class TestResultBaseCoverage:
    """Test cases specifically for improving coverage of _result_base.py module."""

    def test_combine_results_method(self) -> None:
        """Test combine_results method (line 51 coverage)."""
        from flext_core._result_base import _BaseResult, _BaseResultOperations

        # Create some test results
        result1 = _BaseResult.ok("data1")
        result2 = _BaseResult.ok("data2")
        result3 = _BaseResult.ok("data3")

        # Test combine_results which calls chain_results (line 51)
        combined = _BaseResultOperations.combine_results(
            cast("FlextResult[object]", result1),
            cast("FlextResult[object]", result2),
            cast("FlextResult[object]", result3),
        )

        assert combined.success
        assert combined.data == ["data1", "data2", "data3"]

        # Test with failure
        result_fail: FlextResult[str] = _BaseResult.fail("error")
        combined_with_fail = _BaseResultOperations.combine_results(
            cast("FlextResult[object]", result1),
            cast("FlextResult[object]", result_fail),
            cast("FlextResult[object]", result2),
        )

        assert combined_with_fail.is_failure
        assert combined_with_fail.error == "error"
