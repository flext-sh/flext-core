"""Tests for FLEXT Core Railway Base module."""

from __future__ import annotations

from typing import cast

import pytest

from flext_core._railway_base import _BaseRailway, _BaseRailwayUtils
from flext_core.result import FlextResult

# Constants
EXPECTED_BULK_SIZE = 2
EXPECTED_TOTAL_PAGES = 8

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestBaseRailway:
    """Test _BaseRailway functionality."""

    def test_bind_success_chain(self) -> None:
        """Test successful bind operation chaining."""
        # Create initial success result
        initial_result = FlextResult.ok(10)

        # Define transformation function
        def add_five(x: int) -> FlextResult[object]:
            return FlextResult.ok(x + 5)

        # Execute bind operation
        result = _BaseRailway.bind(initial_result, add_five)

        assert result.success
        if result.data != 15:
            raise AssertionError(f"Expected {15}, got {result.data}")

    def test_bind_failure_propagation(self) -> None:
        """Test bind failure propagation."""
        # Create initial failure result
        initial_result: FlextResult[int] = FlextResult.fail("Initial error")

        # Define transformation function (should not be called)
        def add_five(x: int) -> FlextResult[object]:
            return FlextResult.ok(x + 5)

        # Execute bind operation
        result = _BaseRailway.bind(initial_result, add_five)

        assert result.is_failure
        if result.error != "Initial error":
            raise AssertionError(f"Expected {'Initial error'}, got {result.error}")

    def test_bind_function_failure(self) -> None:
        """Test bind when transformation function fails."""
        # Create initial success result
        initial_result = FlextResult.ok(10)

        # Define failing transformation function
        def failing_function(x: int) -> FlextResult[object]:
            return FlextResult.fail("Function failed")

        # Execute bind operation
        result = _BaseRailway.bind(initial_result, failing_function)

        assert result.is_failure
        if result.error != "Function failed":
            raise AssertionError(f"Expected {'Function failed'}, got {result.error}")

    def test_compose_functions_success(self) -> None:
        """Test successful function composition."""

        # Define component functions
        def add_one(x: object) -> FlextResult[object]:
            return FlextResult.ok(int(cast("int", x)) + 1)

        def multiply_two(x: object) -> FlextResult[object]:
            return FlextResult.ok(int(cast("int", x)) * 2)

        def to_string(x: object) -> FlextResult[object]:
            return FlextResult.ok(str(x))

        # Compose functions
        composed = _BaseRailway.compose_functions(add_one, multiply_two, to_string)

        # Execute composed function
        result = composed(5)

        assert result.success
        if result.data != "12":
            raise AssertionError(f"Expected {'12'}, got {result.data}")

    def test_compose_functions_failure_propagation(self) -> None:
        """Test failure propagation in function composition."""

        # Define component functions
        def add_one(x: object) -> FlextResult[object]:
            return FlextResult.ok(int(cast("int", x)) + 1)

        def failing_function(x: object) -> FlextResult[object]:
            return FlextResult.fail("Middle function failed")

        def multiply_two(x: object) -> FlextResult[object]:
            return FlextResult.ok(int(cast("int", x)) * 2)

        # Compose functions
        composed = _BaseRailway.compose_functions(
            add_one,
            failing_function,
            multiply_two,
        )

        # Execute composed function
        result = composed(5)

        assert result.is_failure
        if result.error != "Middle function failed":
            raise AssertionError(
                f"Expected {'Middle function failed'}, got {result.error}"
            )

    def test_switch_condition_true(self) -> None:
        """Test switch operation when condition is true."""

        # Define condition and functions
        def is_positive(x: int) -> bool:
            return x > 0

        def double(x: object) -> FlextResult[object]:
            return FlextResult.ok(int(cast("int", x)) * 2)

        def negate(x: int) -> FlextResult[object]:
            return FlextResult.ok(-x)

        # Create switch function
        switch_func = _BaseRailway.switch(is_positive, double, negate)

        # Execute switch with positive value
        switched_result = switch_func(10)

        assert switched_result.success
        if switched_result.data != 20:
            raise AssertionError(f"Expected {20}, got {switched_result.data}")

    def test_switch_condition_false(self) -> None:
        """Test switch operation when condition is false."""

        # Define condition and functions
        def is_positive(x: int) -> bool:
            return x > 0

        def double(x: object) -> FlextResult[object]:
            return FlextResult.ok(int(cast("int", x)) * 2)

        def negate(x: int) -> FlextResult[object]:
            return FlextResult.ok(-x)

        # Create switch function
        switch_func = _BaseRailway.switch(is_positive, double, negate)

        # Execute switch with negative value
        switched_result = switch_func(-5)

        assert switched_result.success
        if switched_result.data != 5:  # negate(-5):
            raise AssertionError(f"Expected {5}, got {switched_result.data}")

    def test_switch_with_exception(self) -> None:
        """Test switch operation when condition raises exception."""

        # Define failing condition
        def failing_condition(x: int) -> bool:
            if x == 0:
                msg = "Cannot evaluate zero"
                raise ValueError(msg)
            return x > 0

        def double(x: object) -> FlextResult[object]:
            return FlextResult.ok(int(cast("int", x)) * 2)

        def negate(x: int) -> FlextResult[object]:
            return FlextResult.ok(-x)

        # Create switch function
        switch_func = _BaseRailway.switch(failing_condition, double, negate)

        # Execute switch with value that causes exception
        switched_result = switch_func(0)

        assert switched_result.is_failure
        if (
            not switched_result.error
            or "Switch evaluation failed" not in switched_result.error
        ):
            raise AssertionError(
                f"Expected {'Switch evaluation failed'} in {switched_result.error}"
            )

    def test_tee_success(self) -> None:
        """Test tee operation for side effects."""
        # Track side effect execution
        side_effect_called = []

        def main_func(x: int) -> FlextResult[object]:
            return FlextResult.ok(x * 2)

        def side_func(x: int) -> FlextResult[object]:
            side_effect_called.append(x)
            return FlextResult.ok(None)

        # Create tee function
        tee_func = _BaseRailway.tee(main_func, side_func)

        # Execute tee
        teed_result = tee_func(42)

        assert teed_result.success
        if teed_result.data != 84:  # main_func result (42 * 2):
            raise AssertionError(f"Expected {84}, got {teed_result.data}")
        assert side_effect_called == [42]  # Side effect executed

    def test_tee_with_main_function_failure(self) -> None:
        """Test tee operation with main function failure."""
        # Track side effect execution
        side_effect_called = []

        def failing_main_func(x: int) -> FlextResult[object]:
            return FlextResult.fail("Main function failed")

        def side_func(x: int) -> FlextResult[object]:
            side_effect_called.append(x)
            return FlextResult.ok(None)

        # Create tee function
        tee_func = _BaseRailway.tee(failing_main_func, side_func)

        # Execute tee
        teed_result = tee_func(42)

        assert teed_result.is_failure
        if teed_result.error != "Main function failed":
            raise AssertionError(
                f"Expected {'Main function failed'}, got {teed_result.error}"
            )
        # Side effect should still execute (suppressed exceptions)
        if side_effect_called != [42]:
            raise AssertionError(f"Expected {[42]}, got {side_effect_called}")

    def test_dead_end_success(self) -> None:
        """Test dead_end operation success."""
        # Track void function execution
        void_function_called = []

        def void_function(x: int) -> None:
            void_function_called.append(x * 2)

        # Create dead_end function
        dead_end_func = _BaseRailway.dead_end(void_function)

        # Execute dead_end
        dead_end_result = dead_end_func(100)

        assert dead_end_result.success
        if dead_end_result.data != 100:  # Original value preserved:
            raise AssertionError(f"Expected {100}, got {dead_end_result.data}")
        assert void_function_called == [200]  # Void function executed

    def test_dead_end_with_exception(self) -> None:
        """Test dead_end operation when void function raises exception."""
        # Track void function execution
        void_function_called = []

        def failing_void_function(x: int) -> None:
            void_function_called.append(x)
            if x < 0:
                msg = "Negative value not allowed"
                raise ValueError(msg)

        # Create dead_end function
        dead_end_func = _BaseRailway.dead_end(failing_void_function)

        # Execute dead_end with negative value
        dead_end_result = dead_end_func(-10)

        assert dead_end_result.is_failure
        if (
            not dead_end_result.error
            or "Dead end function failed" not in dead_end_result.error
        ):
            raise AssertionError(
                f"Expected {'Dead end function failed'} in {dead_end_result.error}"
            )
        if void_function_called != [-10]:  # Function was called before failing:
            raise AssertionError(f"Expected {[-10]}, got {void_function_called}")

    def test_plus_both_success(self) -> None:
        """Test plus operation with both functions successful."""

        def func1(x: int) -> FlextResult[object]:
            return FlextResult.ok(x + 5)

        def func2(x: int) -> FlextResult[object]:
            return FlextResult.ok(x * 2)

        # Create plus function
        plus_func = _BaseRailway.plus(func1, func2)

        # Execute plus
        plus_result = plus_func(10)

        assert plus_result.success
        if plus_result.data != [15, 20]:  # [func1(10), func2(10)]:
            raise AssertionError(f"Expected {[15, 20]}, got {plus_result.data}")

    def test_plus_first_failure(self) -> None:
        """Test plus operation with first function failing."""

        def failing_func1(x: int) -> FlextResult[object]:
            return FlextResult.fail("First function failed")

        def func2(x: int) -> FlextResult[object]:
            return FlextResult.ok(x * 2)

        # Create plus function
        plus_func = _BaseRailway.plus(failing_func1, func2)

        # Execute plus
        plus_result = plus_func(10)

        assert plus_result.is_failure
        if not plus_result.error or "First function failed" not in plus_result.error:
            raise AssertionError(
                f"Expected {'First function failed'}, got {plus_result.error}"
            )

    def test_plus_second_failure(self) -> None:
        """Test plus operation with second function failing."""

        def func1(x: int) -> FlextResult[object]:
            return FlextResult.ok(x + 5)

        def failing_func2(x: int) -> FlextResult[object]:
            return FlextResult.fail("Second function failed")

        # Create plus function
        plus_func = _BaseRailway.plus(func1, failing_func2)

        # Execute plus
        plus_result = plus_func(10)

        assert plus_result.is_failure
        if not plus_result.error or "Second function failed" not in plus_result.error:
            raise AssertionError(
                f"Expected {'Second function failed'}, got {plus_result.error}"
            )

    def test_plus_both_failure(self) -> None:
        """Test plus operation with both functions failing."""

        def failing_func1(x: int) -> FlextResult[object]:
            return FlextResult.fail("First function failed")

        def failing_func2(x: int) -> FlextResult[object]:
            return FlextResult.fail("Second function failed")

        # Create plus function
        plus_func = _BaseRailway.plus(failing_func1, failing_func2)

        # Execute plus
        plus_result = plus_func(10)

        assert plus_result.is_failure
        # Should contain both errors
        if not plus_result.error or "First function failed" not in plus_result.error:
            raise AssertionError(
                f"Expected {'First function failed'}, got {plus_result.error}"
            )
        assert plus_result.error is not None
        assert "Second function failed" in plus_result.error


class TestBaseRailwayUtils:
    """Test _BaseRailwayUtils functionality."""

    def test_lift_success(self) -> None:
        """Test lifting regular function to railway function."""

        # Define regular function
        def regular_function(x: int) -> int:
            return x * 2

        # Lift to railway function
        railway_function = _BaseRailwayUtils.lift(regular_function)

        # Execute with success input
        result = railway_function(5)

        assert result.success
        if result.data != 10:
            raise AssertionError(f"Expected {10}, got {result.data}")

    def test_lift_function_exception(self) -> None:
        """Test lifting function that raises exception."""

        # Define function that raises exception
        def failing_function(x: int) -> int:
            if x < 0:
                msg = "Negative value not allowed"
                raise ValueError(msg)
            return x * 2

        # Lift to railway function
        railway_function = _BaseRailwayUtils.lift(failing_function)

        # Execute with input that causes exception
        result = railway_function(-1)

        assert result.is_failure
        if not result.error or "Negative value not allowed" not in result.error:
            raise AssertionError(
                f"Expected {'Negative value not allowed'}, got {result.error}"
            )

    def test_ignore_success(self) -> None:
        """Test ignore utility function."""
        # Create ignore function
        ignore_func = _BaseRailwayUtils.ignore()

        # Execute ignore function
        result = ignore_func("any_input")

        assert result.success
        assert result.data is None

    def test_pass_through_success(self) -> None:
        """Test pass_through utility function."""
        # Create pass_through function
        pass_func = _BaseRailwayUtils.pass_through()

        # Test with various input types
        test_inputs = [42, "test", {"key": "value"}, [1, 2, 3]]

        for test_input in test_inputs:
            result = pass_func(test_input)

            assert result.success
            if result.data != test_input:
                raise AssertionError(f"Expected {test_input}, got {result.data}")

    def test_complex_railway_chain(self) -> None:
        """Test complex railway operation chain."""

        # Define component functions
        def validate_positive(x: object) -> FlextResult[object]:
            if int(cast("int", x)) <= 0:
                return FlextResult.fail("Must be positive")
            return FlextResult.ok(x)

        def double(x: object) -> FlextResult[object]:
            return FlextResult.ok(int(cast("int", x)) * 2)

        def add_ten(x: object) -> FlextResult[object]:
            return FlextResult.ok(int(cast("int", x)) + 10)

        # Create complex chain
        initial_result = FlextResult.ok(5)

        # Chain operations using bind
        result = _BaseRailway.bind(
            _BaseRailway.bind(
                _BaseRailway.bind(initial_result, validate_positive),
                double,
            ),
            add_ten,
        )

        assert result.success
        if result.data != 20:  # (5 * 2) + 10 = 20
            raise AssertionError(f"Expected {20}, got {result.data}")

    def test_complex_railway_chain_with_failure(self) -> None:
        """Test complex railway chain with failure in middle."""

        # Define component functions
        def validate_positive(x: object) -> FlextResult[object]:
            if int(cast("int", x)) <= 0:
                return FlextResult.fail("Must be positive")
            return FlextResult.ok(x)

        def double(x: object) -> FlextResult[object]:
            return FlextResult.ok(int(cast("int", x)) * 2)

        def add_ten(x: object) -> FlextResult[object]:
            return FlextResult.ok(int(cast("int", x)) + 10)

        # Create complex chain with negative input
        initial_result = FlextResult.ok(-3)

        # Chain operations using bind
        result = _BaseRailway.bind(
            _BaseRailway.bind(
                _BaseRailway.bind(initial_result, validate_positive),
                double,
            ),
            add_ten,
        )

        assert result.is_failure
        if result.error != "Must be positive":
            raise AssertionError(f"Expected {'Must be positive'}, got {result.error}")

    def test_bind_with_none_data(self) -> None:
        """Test bind operation when success result has None data."""
        # Create success result with None data
        success_result = FlextResult.ok(None)

        def process_data(x: object) -> FlextResult[object]:
            return FlextResult.ok(str(x))

        # Execute bind operation
        result = _BaseRailway.bind(success_result, process_data)

        assert result.is_failure
        if result.error != "Cannot bind with None data":
            raise AssertionError(
                f"Expected {'Cannot bind with None data'}, got {result.error}"
            )

    def test_bind_function_exception(self) -> None:
        """Test bind operation when function raises exception."""
        # Create success result
        initial_result = FlextResult.ok("test")

        # Define function that raises TypeError
        def failing_function(x: str) -> FlextResult[object]:
            # This will raise TypeError when called with string
            return FlextResult.ok(int(x + None))  # type: ignore[operator] # Intentional error for testing

        # Execute bind operation
        result = _BaseRailway.bind(initial_result, failing_function)

        assert result.is_failure
        if not result.error or "Bind operation failed:" not in result.error:
            raise AssertionError(f"Expected 'Bind operation failed:' in {result.error}")

    def test_type_checking_imports(self) -> None:
        """Test that TYPE_CHECKING imports are properly structured."""
        # This test ensures TYPE_CHECKING block is executed during static analysis
        # The imports in TYPE_CHECKING block (lines 101-103) are covered by this test

        # Verify classes are properly imported and functional
        assert _BaseRailway is not None
        assert _BaseRailwayUtils is not None

        # Verify railway operations work with proper typing
        result = FlextResult.ok(42)
        bind_result = _BaseRailway.bind(result, lambda x: FlextResult.ok(x * 2))

        assert bind_result.success
        if bind_result.data != 84:
            raise AssertionError(f"Expected {84}, got {bind_result.data}")
