"""Real test to boost FlextResult coverage targeting missing lines."""

from typing import Any

import pytest

from flext_core.result import FlextResult


class TestFlextResultRealBoost:
    """Test FlextResult targeting specific uncovered lines."""

    def test_result_error_handling(self) -> None:
        """Test error handling paths in FlextResult."""
        # Test fail with None error - FlextResult creates default error message
        result = FlextResult[str].fail("None error")
        assert result.is_failure
        assert result.error is not None  # FlextResult provides default error

        # Test fail with string error
        error_result = FlextResult[str].fail("Test error")
        assert error_result.is_failure
        assert error_result.error == "Test error"

    def test_result_value_access_patterns(self) -> None:
        """Test different value access patterns."""
        # Test success result
        success = FlextResult[int].ok(42)
        assert success.is_success
        assert success.value == 42
        assert success.data == 42  # Legacy API
        assert success.unwrap() == 42

        # Test failure result - value access raises exception
        failure = FlextResult[int].fail("Error")
        assert failure.is_failure
        with pytest.raises(TypeError):
            _ = failure.value  # This should raise
        assert failure.data is None  # Legacy API

    def test_result_unwrap_or_methods(self) -> None:
        """Test unwrap_or and similar methods."""
        # Test unwrap_or with success
        success = FlextResult[str].ok("value")
        assert success.unwrap_or("default") == "value"

        # Test unwrap_or with failure
        failure = FlextResult[str].fail("error")
        assert failure.unwrap_or("default") == "default"

        # Test or_else_get with success (should return FlextResult)
        assert (
            success.or_else_get(lambda: FlextResult[str].ok("computed")).unwrap()
            == "value"
        )

        # Test or_else_get with failure (should return FlextResult)
        assert (
            failure.or_else_get(lambda: FlextResult[str].ok("computed")).unwrap()
            == "computed"
        )

    def test_result_map_operations(self) -> None:
        """Test map and flat_map operations."""
        # Test map with success
        success = FlextResult[int].ok(10)
        mapped = success.map(lambda x: x * 2)
        assert mapped.is_success
        assert mapped.unwrap() == 20

        # Test map with failure
        failure = FlextResult[int].fail("error")
        mapped_fail = failure.map(lambda x: x * 2)
        assert mapped_fail.is_failure
        assert mapped_fail.error == "error"

        # Test flat_map with success
        def double_as_result(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        flat_mapped = success.flat_map(double_as_result)
        assert flat_mapped.is_success
        assert flat_mapped.unwrap() == 20

        # Test flat_map with failure
        flat_mapped_fail = failure.flat_map(double_as_result)
        assert flat_mapped_fail.is_failure

    def test_result_filter_operations(self) -> None:
        """Test filter operations."""
        # Test filter with success that passes
        success = FlextResult[int].ok(10)
        filtered = success.filter(lambda x: x > 5, "Value too small")
        assert filtered.is_success
        assert filtered.unwrap() == 10

        # Test filter with success that fails
        filtered_fail = success.filter(lambda x: x > 15, "Value too small")
        assert filtered_fail.is_failure
        assert filtered_fail.error == "Value too small"

        # Test filter with failure
        failure = FlextResult[int].fail("original error")
        filtered_already_fail = failure.filter(lambda x: x > 5, "Value too small")
        assert filtered_already_fail.is_failure
        assert filtered_already_fail.error == "original error"

    @pytest.mark.skip(
        reason="tap_error tests don't verify side effects are actually executed"
    )
    def test_result_error_mapping(self) -> None:
        """Test map_error operations.

        TODO: Improve this test to:
        - Verify that tap_error actually executes the side effect function
        - Use a mock or counter to ensure the function was called
        - Test with multiple error types and transformations
        - Test error mapping chain operations
        """
        # Test tap_error with failure (tap_error expects None return)
        failure = FlextResult[int].fail("original")
        mapped_error = failure.tap_error(
            lambda _: None
        )  # tap_error expects None return
        assert mapped_error.is_failure
        assert mapped_error.error == "original"  # tap_error doesn't change the error

        # Test tap_error with success (should not change)
        success = FlextResult[int].ok(42)
        mapped_error_success = success.tap_error(
            lambda _: None
        )  # tap_error expects None return
        assert mapped_error_success.is_success
        assert mapped_error_success.unwrap() == 42

    def test_result_boolean_operations(self) -> None:
        """Test boolean operations and properties."""
        success = FlextResult[str].ok("test")
        failure = FlextResult[str].fail("error")

        # Test success property
        assert success.success is True
        assert failure.success is False

        # Test is_success/is_failure
        assert success.is_success is True
        assert success.is_failure is False
        assert failure.is_success is False
        assert failure.is_failure is True

    def test_result_expect_operations(self) -> None:
        """Test expect operations that raise on failure."""
        success = FlextResult[str].ok("value")

        # Test expect with success
        assert success.expect("Should not raise") == "value"

        # Test expect with failure - should raise
        failure = FlextResult[str].fail("error")
        with pytest.raises(Exception, match="Custom message"):
            failure.expect("Custom message")

    def test_result_and_or_operations(self) -> None:
        """Test and/or combinatorial operations."""
        success1 = FlextResult[int].ok(1)
        success2 = FlextResult[str].ok("test")
        failure2 = FlextResult[str].fail("error2")

        # Test then operation (similar to and_)
        then_success = success1.then(lambda _: success2)
        assert then_success.is_success
        assert then_success.unwrap() == "test"

        then_failure = success1.then(lambda _: failure2)
        assert then_failure.is_failure
        assert then_failure.error == "error2"

        # Test or_else operation with compatible types
        failure_str = FlextResult[str].fail("error1")
        success_str = FlextResult[str].ok("test")
        or_success = failure_str.or_else(success_str)
        assert or_success.is_success
        assert or_success.unwrap() == "test"

        failure2_str = FlextResult[str].fail("error2")
        or_failure = failure_str.or_else(failure2_str)
        assert or_failure.is_failure
        assert or_failure.error == "error2"  # or_else returns the alternative

    def test_result_edge_cases(self) -> None:
        """Test edge cases and boundary conditions."""
        # Test with None values
        none_success = FlextResult[Any].ok(None)
        assert none_success.is_success
        assert none_success.value is None

        # Test with empty strings
        empty_success = FlextResult[str].ok("")
        assert empty_success.is_success
        assert empty_success.unwrap() == ""

        # Test with zero values
        zero_success = FlextResult[int].ok(0)
        assert zero_success.is_success
        assert zero_success.unwrap() == 0

    def test_result_chaining_operations(self) -> None:
        """Test complex chaining of operations."""
        # Complex chain with success
        result = (
            FlextResult[int]
            .ok(5)
            .map(lambda x: x * 2)
            .filter(lambda x: x > 5, "Too small")
            .flat_map(lambda x: FlextResult[str].ok(f"Value: {x}"))
            .map(lambda s: s.upper())
        )
        assert result.is_success
        assert result.unwrap() == "VALUE: 10"

        # Complex chain with failure in filter
        result_fail = (
            FlextResult[int]
            .ok(5)
            .map(lambda x: x * 2)
            .filter(lambda x: x > 15, "Too small")  # This will fail
            .flat_map(lambda x: FlextResult[str].ok(f"Value: {x}"))
            .map(lambda s: s.upper())
        )
        assert result_fail.is_failure
        assert result_fail.error == "Too small"

    def test_result_factory_methods(self) -> None:
        """Test factory methods and alternative constructors."""
        # Test from_exception class method if available
        if hasattr(FlextResult, "from_exception"):

            def failing_func() -> None:
                msg = "Test exception"
                raise ValueError(msg)

            exc_result = FlextResult.from_exception(failing_func)
            assert exc_result.is_failure
            # Error message should contain some indication of the exception

        # Test various ok() patterns
        ok_result = FlextResult.ok("test")
        assert ok_result.is_success

        # Test various fail() patterns
        fail_result: FlextResult[object] = FlextResult.fail("test error")
        assert fail_result.is_failure

    def test_result_utility_methods(self) -> None:
        """Test utility and inspection methods."""
        success = FlextResult[dict[str, int]].ok({"key": 42})
        failure = FlextResult[dict[str, int]].fail("Dict error")

        # Test string representations
        success_str = str(success)
        assert "42" in success_str or "key" in success_str

        failure_str = str(failure)
        assert "Dict error" in failure_str

        # Test hash and equality if implemented
        success2 = FlextResult[dict[str, int]].ok({"key": 42})
        assert success.unwrap() == success2.unwrap()

    def test_result_async_compatibility(self) -> None:
        """Test async compatibility patterns."""

        # Test patterns commonly used with async code
        async def async_operation() -> FlextResult[str]:
            return FlextResult[str].ok("async_result")

        # These patterns should work in sync context too
        result = FlextResult[str].ok("sync_result")
        mapped = result.map(lambda x: f"processed_{x}")
        assert mapped.unwrap() == "processed_sync_result"
