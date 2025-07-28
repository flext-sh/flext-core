"""Comprehensive tests for FlextResult consolidated functionality.

Tests all consolidated features following "entregar mais com muito menos" approach:
- Factory methods (ok, fail, from_callable, conditional)
- Transformation methods (map, chain, where)
- Railway patterns (then, recover, tap)
- Combination methods (combine, sequence)
- Error handling patterns
- Type safety and edge cases
"""

from __future__ import annotations

import operator

import pytest

from flext_core import FlextResult

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextResultFactoryMethods:
    """Test factory methods for FlextResult creation."""

    def test_ok_creation_with_various_types(self) -> None:
        """Test ok() factory with different data types."""
        # String data
        string_result = FlextResult.ok("test_string")
        assert string_result.is_success
        assert string_result.data == "test_string"
        assert string_result.error is None

        # Integer data
        int_result = FlextResult.ok(42)
        assert int_result.is_success
        assert int_result.data == 42

        # Dictionary data
        dict_data = {"key": "value", "number": 123}
        dict_result = FlextResult.ok(dict_data)
        assert dict_result.is_success
        assert dict_result.data == dict_data

        # List data
        list_data = [1, 2, 3, "test"]
        list_result = FlextResult.ok(list_data)
        assert list_result.is_success
        assert list_result.data == list_data

        # None data (should be valid)
        none_result = FlextResult.ok(None)
        assert none_result.is_success
        assert none_result.data is None

    def test_fail_creation_with_comprehensive_errors(self) -> None:
        """Test fail() factory with comprehensive error information."""
        # Simple error message
        simple_fail = FlextResult.fail("Simple error")
        assert simple_fail.is_failure
        assert simple_fail.error == "Simple error"
        assert simple_fail.error_code is None
        assert simple_fail.data is None

        # Error with code
        coded_fail = FlextResult.fail("Validation error", error_code="VALIDATION_001")
        assert coded_fail.is_failure
        assert coded_fail.error == "Validation error"
        assert coded_fail.error_code == "VALIDATION_001"

        # Error with code and data
        complex_fail = FlextResult.fail(
            "Complex error",
            error_code="COMPLEX_001",
            error_data={"field": "email", "value": "invalid@", "reason": "malformed"},
        )
        assert complex_fail.is_failure
        assert complex_fail.error == "Complex error"
        assert complex_fail.error_code == "COMPLEX_001"
        assert complex_fail.error_data["field"] == "email"
        assert complex_fail.error_data["value"] == "invalid@"
        assert complex_fail.error_data["reason"] == "malformed"

    def test_from_callable_success_scenarios(self) -> None:
        """Test from_callable() with successful callable scenarios."""

        # Simple function returning string
        def simple_function() -> str:
            return "success_result"

        result = FlextResult.from_callable(simple_function)
        assert result.is_success
        assert result.data == "success_result"

        # Function with complex computation
        def complex_computation() -> dict[str, object]:
            return {
                "computed_value": sum(range(100)),
                "metadata": {"timestamp": "2025-01-01", "version": "1.0"},
            }

        complex_result = FlextResult.from_callable(complex_computation)
        assert complex_result.is_success
        assert complex_result.data["computed_value"] == 4950
        assert complex_result.data["metadata"]["version"] == "1.0"

        # Lambda function
        lambda_result = FlextResult.from_callable(lambda: 42 * 2)
        assert lambda_result.is_success
        assert lambda_result.data == 84

    def test_from_callable_failure_scenarios(self) -> None:
        """Test from_callable() with exception scenarios."""

        # Function that raises ValueError
        def value_error_function() -> str:
            msg = "Invalid value provided"
            raise ValueError(msg)

        result = FlextResult.from_callable(
            value_error_function,
            "Custom error message",
        )
        assert result.is_failure
        assert result.error == "Custom error message"
        assert "Invalid value provided" in str(result.error_data.get("exception", ""))

        # Function that raises TypeError
        def type_error_function() -> str:
            return "test" + 42

        type_result = FlextResult.from_callable(type_error_function)
        assert type_result.is_failure
        assert type_result.error == "Operation failed"

        # Function that raises RuntimeError
        def runtime_error_function() -> str:
            msg = "System failure"
            raise RuntimeError(msg)

        runtime_result = FlextResult.from_callable(runtime_error_function)
        assert runtime_result.is_failure

    def test_conditional_factory_patterns(self) -> None:
        """Test conditional() factory with various condition patterns."""
        # True condition - success
        true_result = FlextResult.conditional(
            condition=True,
            success_data="operation_succeeded",
            failure_message="Should not see this",
        )
        assert true_result.is_success
        assert true_result.data == "operation_succeeded"

        # False condition - failure
        false_result = FlextResult.conditional(
            condition=False,
            success_data="will_not_be_used",
            failure_message="Condition failed",
            failure_code="CONDITION_FALSE",
        )
        assert false_result.is_failure
        assert false_result.error == "Condition failed"
        assert false_result.error_code == "CONDITION_FALSE"

        # Complex condition evaluation
        test_value = 100
        range_result = FlextResult.conditional(
            condition=50 <= test_value <= 150,
            success_data=f"Value {test_value} is in range",
            failure_message=f"Value {test_value} is out of range",
        )
        assert range_result.is_success
        assert "Value 100 is in range" in range_result.data

        # Boundary condition testing
        boundary_result = FlextResult.conditional(
            condition=test_value == 100,
            success_data="exact_match",
            failure_message="no_match",
        )
        assert boundary_result.is_success
        assert boundary_result.data == "exact_match"


class TestFlextResultTransformations:
    """Test transformation methods: map, chain, where."""

    def test_map_success_transformations(self) -> None:
        """Test map() with successful transformations."""
        # String transformation
        string_result = FlextResult.ok("hello")
        upper_result = string_result.map(lambda s: s.upper())
        assert upper_result.is_success
        assert upper_result.data == "HELLO"

        # Numeric transformation
        number_result = FlextResult.ok(5)
        doubled_result = number_result.map(lambda x: x * 2)
        assert doubled_result.is_success
        assert doubled_result.data == 10

        # Type transformation (int to string)
        type_change_result = number_result.map(lambda x: f"Number: {x}")
        assert type_change_result.is_success
        assert type_change_result.data == "Number: 5"

        # Complex object transformation
        user_data = {"name": "John", "age": 30}
        user_result = FlextResult.ok(user_data)
        formatted_result = user_result.map(
            lambda u: f"{u['name']} ({u['age']} years old)",
        )
        assert formatted_result.is_success
        assert formatted_result.data == "John (30 years old)"

        # Chained map transformations
        chained_result = (
            FlextResult.ok(10)
            .map(lambda x: x * 2)
            .map(lambda x: x + 5)
            .map(lambda x: f"Final: {x}")
        )
        assert chained_result.is_success
        assert chained_result.data == "Final: 25"

    def test_map_failure_propagation(self) -> None:
        """Test map() properly propagates failures."""
        failed_result = FlextResult.fail("Original error")

        # Map should not execute on failure
        mapped_result = failed_result.map(lambda x: x * 2)
        assert mapped_result.is_failure
        assert mapped_result.error == "Original error"

        # Chained maps on failure
        chained_failed = (
            failed_result.map(lambda x: x * 2).map(str).map(lambda x: x.upper())
        )
        assert chained_failed.is_failure
        assert chained_failed.error == "Original error"

    def test_chain_success_scenarios(self) -> None:
        """Test chain() with successful result-returning functions."""

        def validate_positive(x: int) -> FlextResult[int]:
            if x > 0:
                return FlextResult.ok(x)
            return FlextResult.fail("Number must be positive")

        def double_if_even(x: int) -> FlextResult[int]:
            if x % 2 == 0:
                return FlextResult.ok(x * 2)
            return FlextResult.fail("Number must be even")

        # Successful chain
        result = FlextResult.ok(4).chain(validate_positive).chain(double_if_even)
        assert result.is_success
        assert result.data == 8

        # Chain with database-like operation simulation
        def simulate_user_lookup(user_id: str) -> FlextResult[dict[str, object]]:
            if user_id == "123":
                return FlextResult.ok({"id": "123", "name": "John", "active": True})
            return FlextResult.fail("User not found")

        def check_user_active(
            user: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            if user.get("active"):
                return FlextResult.ok(user)
            return FlextResult.fail("User is inactive")

        user_result = (
            FlextResult.ok("123").chain(simulate_user_lookup).chain(check_user_active)
        )
        assert user_result.is_success
        assert user_result.data["name"] == "John"

    def test_chain_failure_scenarios(self) -> None:
        """Test chain() with failure scenarios."""

        def always_fail(x: object) -> FlextResult[object]:
            return FlextResult.fail("Always fails")

        # Chain that fails at first step
        result = (
            FlextResult.ok(5)
            .chain(always_fail)
            .chain(lambda x: FlextResult.ok(x * 2))  # Should not execute
        )
        assert result.is_failure
        assert result.error == "Always fails"

        # Starting with failure
        failed_start = FlextResult.fail("Initial failure").chain(
            lambda x: FlextResult.ok(x * 2),
        )
        assert failed_start.is_failure
        assert failed_start.error == "Initial failure"

    def test_where_filtering_patterns(self) -> None:
        """Test where() with various filtering patterns."""
        # Successful filter
        result = FlextResult.ok(10).where(lambda x: x > 5)
        assert result.is_success
        assert result.data == 10

        # Failed filter with default message
        failed_filter = FlextResult.ok(3).where(lambda x: x > 5)
        assert failed_filter.is_failure
        assert failed_filter.error == "Filter condition not met"

        # Failed filter with custom message
        custom_message_filter = FlextResult.ok(3).where(
            lambda x: x > 5,
            "Number must be greater than 5",
        )
        assert custom_message_filter.is_failure
        assert custom_message_filter.error == "Number must be greater than 5"

        # Complex filtering
        user_data = {"name": "John", "age": 25, "verified": True}
        user_result = FlextResult.ok(user_data)

        verified_user = user_result.where(
            lambda u: u["verified"] and u["age"] >= 18,
            "User must be verified and adult",
        )
        assert verified_user.is_success

        # Chained filtering
        filtered_chain = (
            FlextResult.ok(100)
            .where(lambda x: x > 50, "Must be > 50")
            .where(lambda x: x < 200, "Must be < 200")
            .where(lambda x: x % 10 == 0, "Must be divisible by 10")
        )
        assert filtered_chain.is_success
        assert filtered_chain.data == 100

    def test_where_with_failure_propagation(self) -> None:
        """Test where() properly propagates existing failures."""
        failed_result = FlextResult.fail("Original failure")

        # Where should not execute on existing failure
        filtered_result = failed_result.where(lambda x: True)
        assert filtered_result.is_failure
        assert filtered_result.error == "Original failure"


class TestFlextResultRailwayPatterns:
    """Test railway patterns: then, recover, tap."""

    def test_then_sequential_operations(self) -> None:
        """Test then() for sequential operations with type flexibility."""

        def process_string(s: str) -> FlextResult[object]:
            return FlextResult.ok(f"Processed: {s}")

        def add_metadata(processed: object) -> FlextResult[object]:
            return FlextResult.ok(
                {
                    "data": processed,
                    "timestamp": "2025-01-01",
                    "version": "1.0",
                },
            )

        # Successful then chain
        result = FlextResult.ok("input_data").then(process_string).then(add_metadata)
        assert result.is_success
        result_data = result.data
        assert isinstance(result_data, dict)
        assert result_data["data"] == "Processed: input_data"
        assert result_data["version"] == "1.0"

    def test_then_failure_scenarios(self) -> None:
        """Test then() with failure scenarios."""

        def fail_operation(x: object) -> FlextResult[object]:
            return FlextResult.fail("Operation failed")

        # Failure in then chain
        result = (
            FlextResult.ok("test")
            .then(fail_operation)
            .then(lambda x: FlextResult.ok(f"After: {x}"))  # Should not execute
        )
        assert result.is_failure
        assert result.error == "Operation failed"

    def test_recover_patterns(self) -> None:
        """Test recover() for error recovery patterns."""

        def provide_default(error: str) -> FlextResult[str]:
            return FlextResult.ok("default_value")

        def provide_cached_value(error: str) -> FlextResult[str]:
            if "not found" in error.lower():
                return FlextResult.ok("cached_value")
            return FlextResult.fail("No recovery possible")

        # Successful recovery
        recovered_result = FlextResult.fail("Data not found").recover(
            provide_cached_value,
        )
        assert recovered_result.is_success
        assert recovered_result.data == "cached_value"

        # No recovery needed (success)
        no_recovery_needed = FlextResult.ok("original_data").recover(provide_default)
        assert no_recovery_needed.is_success
        assert no_recovery_needed.data == "original_data"

        # Failed recovery
        failed_recovery = FlextResult.fail("Different error").recover(
            provide_cached_value,
        )
        assert failed_recovery.is_failure
        assert failed_recovery.error == "No recovery possible"

        # Chained recovery
        multiple_recovery = (
            FlextResult.fail("Initial error")
            .recover(lambda e: FlextResult.fail("First recovery failed"))
            .recover(provide_default)
        )
        assert multiple_recovery.is_success
        assert multiple_recovery.data == "default_value"

    def test_tap_side_effects(self) -> None:
        """Test tap() for side effects without changing result."""
        side_effects: list[str] = []

        def log_success(data: str) -> None:
            side_effects.append(f"Logged: {data}")

        def update_metrics(data: str) -> None:
            side_effects.append(f"Metrics: {data}")

        # Successful tap chain
        result = (
            FlextResult.ok("test_data")
            .tap(log_success)
            .tap(update_metrics)
            .map(lambda x: x.upper())
        )

        assert result.is_success
        assert result.data == "TEST_DATA"
        assert "Logged: test_data" in side_effects
        assert "Metrics: test_data" in side_effects

        # Tap on failure (should not execute)
        side_effects.clear()
        failed_result = FlextResult.fail("Error occurred").tap(log_success)
        assert failed_result.is_failure
        assert len(side_effects) == 0  # No side effects on failure

    def test_tap_exception_safety(self) -> None:
        """Test tap() handles exceptions in side effects safely."""

        def failing_side_effect(data: str) -> None:
            msg = "Side effect failed"
            raise ValueError(msg)

        # Tap with failing side effect should not break the chain
        result = (
            FlextResult.ok("test")
            .tap(failing_side_effect)  # This should not break the chain
            .map(lambda x: x.upper())
        )
        assert result.is_success
        assert result.data == "TEST"


class TestFlextResultCombinations:
    """Test combination methods: combine, sequence."""

    def test_combine_successful_results(self) -> None:
        """Test combine() with successful results."""
        user_result = FlextResult.ok({"name": "John", "id": "123"})
        posts_result = FlextResult.ok([{"id": "1", "title": "Post 1"}])

        combined = FlextResult.combine(
            user_result,
            posts_result,
            lambda user, posts: {"user": user, "posts": posts},
        )

        assert combined.is_success
        combined_data = combined.data
        assert combined_data["user"]["name"] == "John"
        assert len(combined_data["posts"]) == 1

        # Simple value combination
        num1_result = FlextResult.ok(10)
        num2_result = FlextResult.ok(20)

        sum_result = FlextResult.combine(
            num1_result,
            num2_result,
            operator.add,
        )
        assert sum_result.is_success
        assert sum_result.data == 30

    def test_combine_failure_scenarios(self) -> None:
        """Test combine() with failure scenarios."""
        success_result = FlextResult.ok("success_data")
        failure_result = FlextResult.fail("Failed operation")

        # First result fails
        first_fail = FlextResult.combine(
            failure_result,
            success_result,
            lambda a, b: f"{a}-{b}",
        )
        assert first_fail.is_failure
        assert first_fail.error == "Failed operation"

        # Second result fails
        second_fail = FlextResult.combine(
            success_result,
            failure_result,
            lambda a, b: f"{a}-{b}",
        )
        assert second_fail.is_failure
        assert second_fail.error == "Failed operation"

        # Both results fail (should return first failure)
        both_fail = FlextResult.combine(
            FlextResult.fail("First error"),
            FlextResult.fail("Second error"),
            lambda a, b: f"{a}-{b}",
        )
        assert both_fail.is_failure
        assert both_fail.error == "First error"

    def test_sequence_successful_operations(self) -> None:
        """Test sequence() with successful operations."""
        # Empty sequence
        empty_sequence = FlextResult.sequence()
        assert empty_sequence.is_success
        assert empty_sequence.data == []

        # Single result
        single_sequence = FlextResult.sequence(FlextResult.ok("single"))
        assert single_sequence.is_success
        assert single_sequence.data == ["single"]

        # Multiple successful results
        multiple_sequence = FlextResult.sequence(
            FlextResult.ok("first"),
            FlextResult.ok(42),
            FlextResult.ok({"key": "value"}),
            FlextResult.ok([1, 2, 3]),
        )
        assert multiple_sequence.is_success
        sequence_data = multiple_sequence.data
        assert len(sequence_data) == 4
        assert sequence_data[0] == "first"
        assert sequence_data[1] == 42
        assert sequence_data[2] == {"key": "value"}
        assert sequence_data[3] == [1, 2, 3]

    def test_sequence_failure_scenarios(self) -> None:
        """Test sequence() with failure scenarios."""
        # First result fails
        first_fail_sequence = FlextResult.sequence(
            FlextResult.fail("First failed"),
            FlextResult.ok("second"),
            FlextResult.ok("third"),
        )
        assert first_fail_sequence.is_failure
        assert first_fail_sequence.error == "First failed"

        # Middle result fails
        middle_fail_sequence = FlextResult.sequence(
            FlextResult.ok("first"),
            FlextResult.fail("Middle failed"),
            FlextResult.ok("third"),
        )
        assert middle_fail_sequence.is_failure
        assert middle_fail_sequence.error == "Middle failed"

        # Last result fails
        last_fail_sequence = FlextResult.sequence(
            FlextResult.ok("first"),
            FlextResult.ok("second"),
            FlextResult.fail("Last failed"),
        )
        assert last_fail_sequence.is_failure
        assert last_fail_sequence.error == "Last failed"


class TestFlextResultEdgeCases:
    """Test edge cases and error conditions."""

    def test_result_state_consistency(self) -> None:
        """Test result state consistency."""
        # Success result properties
        success = FlextResult.ok("data")
        assert success.is_success
        assert not success.is_failure
        assert success.data == "data"
        assert success.error is None

        # Failure result properties
        failure = FlextResult.fail("error")
        assert failure.is_failure
        assert not failure.is_success
        assert failure.data is None
        assert failure.error == "error"

    def test_nested_result_handling(self) -> None:
        """Test handling of nested operations."""
        # Complex nested transformations
        nested_result = (
            FlextResult.ok(
                {"users": [{"id": "1", "active": True}, {"id": "2", "active": False}]},
            )
            .map(lambda data: [user for user in data["users"] if user["active"]])
            .where(lambda users: len(users) > 0, "No active users found")
            .map(lambda users: users[0]["id"])
        )
        assert nested_result.is_success
        assert nested_result.data == "1"

    def test_type_safety_preservation(self) -> None:
        """Test that type information is preserved through operations."""
        # String type preservation
        string_result = FlextResult.ok("test").map(lambda s: s.upper())
        assert isinstance(string_result.data, str)

        # Dictionary type preservation
        dict_result = FlextResult.ok({"key": "value"}).map(
            lambda d: {**d, "new_key": "new_value"},
        )
        assert isinstance(dict_result.data, dict)
        assert dict_result.data["key"] == "value"
        assert dict_result.data["new_key"] == "new_value"

    def test_error_data_preservation(self) -> None:
        """Test that error data is preserved through operations."""
        complex_error = FlextResult.fail(
            "Validation failed",
            error_code="VALIDATION_001",
            error_data={"field": "email", "value": "invalid"},
        )

        # Error should be preserved through transformations
        transformed = complex_error.map(lambda x: x.upper())
        assert transformed.is_failure
        assert transformed.error == "Validation failed"
        assert transformed.error_code == "VALIDATION_001"
        assert transformed.error_data["field"] == "email"

        # Error should be preserved through chains
        chained = complex_error.chain(FlextResult.ok)
        assert chained.is_failure
        assert chained.error_code == "VALIDATION_001"
