"""FlextResult uncovered gaps tests targeting specific uncovered lines.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Any

import pytest

from flext_core import FlextResult
from flext_tests import FlextTestsMatchers


class TestFlextResultUncoveredGaps:
    """Targeted tests for likely uncovered FlextResult functionality."""

    def test_unwrap_or_raise_success_case(self) -> None:
        """Test FlextResult.unwrap_or_raise with successful result."""
        result = FlextResult[str].ok("success_value")
        unwrapped = FlextResult.unwrap_or_raise(result)
        assert unwrapped == "success_value"

    def test_unwrap_or_raise_failure_default_exception(self) -> None:
        """Test FlextResult.unwrap_or_raise with failure using default exception type."""
        result = FlextResult[str].fail("operation failed")

        with pytest.raises(RuntimeError, match="operation failed"):
            FlextResult.unwrap_or_raise(result)

    def test_unwrap_or_raise_failure_custom_exception(self) -> None:
        """Test FlextResult.unwrap_or_raise with failure using custom exception type."""
        result = FlextResult[str].fail("custom error")

        with pytest.raises(ValueError, match="custom error"):
            FlextResult.unwrap_or_raise(result, exception_type=ValueError)

    def test_unwrap_or_raise_failure_no_error_message(self) -> None:
        """Test FlextResult.unwrap_or_raise with failure having no error message."""
        # Create failure with empty error message
        result = FlextResult[str](error="")

        with pytest.raises(RuntimeError, match="Operation failed"):
            FlextResult.unwrap_or_raise(result)

    def test_from_exception_success_case(self) -> None:
        """Test FlextResult.from_exception with function that succeeds."""

        def successful_func() -> str:
            return "success_result"

        result = FlextResult.from_exception(successful_func)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "success_result"

    def test_from_exception_type_error(self) -> None:
        """Test FlextResult.from_exception with TypeError."""

        def failing_func() -> str:
            msg = "Type error occurred"
            raise TypeError(msg)

        result = FlextResult.from_exception(failing_func)
        FlextTestsMatchers.assert_result_failure(result)
        assert "Type error occurred" in result.error

    def test_from_exception_value_error(self) -> None:
        """Test FlextResult.from_exception with ValueError."""

        def failing_func() -> int:
            msg = "Invalid value provided"
            raise ValueError(msg)

        result = FlextResult.from_exception(failing_func)
        FlextTestsMatchers.assert_result_failure(result)
        assert "Invalid value provided" in result.error

    def test_from_exception_attribute_error(self) -> None:
        """Test FlextResult.from_exception with AttributeError."""

        def failing_func() -> object:
            obj = None
            return obj.missing_attribute

        result = FlextResult.from_exception(failing_func)
        FlextTestsMatchers.assert_result_failure(result)
        assert "attribute" in result.error.lower()

    def test_from_exception_runtime_error(self) -> None:
        """Test FlextResult.from_exception with RuntimeError."""

        def failing_func() -> str:
            msg = "Runtime failure"
            raise RuntimeError(msg)

        result = FlextResult.from_exception(failing_func)
        FlextTestsMatchers.assert_result_failure(result)
        assert "Runtime failure" in result.error

    def test_from_exception_unhandled_exception(self) -> None:
        """Test FlextResult.from_exception with unhandled exception type."""

        def failing_func() -> str:
            msg = "Unhandled IO error"
            raise OSError(msg)

        # Should re-raise unhandled exceptions
        with pytest.raises(IOError, match="Unhandled IO error"):
            FlextResult.from_exception(failing_func)

    def test_or_else_success_case(self) -> None:
        """Test FlextResult.or_else with successful result."""
        success_result = FlextResult[str].ok("original")
        alternative_result = FlextResult[str].ok("alternative")

        result = success_result.or_else(alternative_result)
        assert result is success_result
        assert result.value == "original"

    def test_or_else_failure_case(self) -> None:
        """Test FlextResult.or_else with failed result."""
        failure_result = FlextResult[str].fail("original failed")
        alternative_result = FlextResult[str].ok("alternative")

        result = failure_result.or_else(alternative_result)
        assert result is alternative_result
        assert result.value == "alternative"

    def test_hash_success_with_hashable_data(self) -> None:
        """Test FlextResult.__hash__ with successful result containing hashable data."""
        result1 = FlextResult[str].ok("test_data")
        result2 = FlextResult[str].ok("test_data")
        result3 = FlextResult[str].ok("different_data")

        # Same data should have same hash
        assert hash(result1) == hash(result2)
        # Different data should have different hash (likely)
        assert hash(result1) != hash(result3)

    def test_hash_success_with_non_hashable_data(self) -> None:
        """Test FlextResult.__hash__ with successful result containing non-hashable data."""
        non_hashable_data = {"key": "value", "nested": [1, 2, 3]}
        result = FlextResult[dict].ok(non_hashable_data)

        # Should not raise exception and should return consistent hash
        hash_value = hash(result)
        assert isinstance(hash_value, int)
        assert hash(result) == hash_value  # Consistent

    def test_hash_success_with_object_having_dict(self) -> None:
        """Test FlextResult.__hash__ with object having __dict__ attribute."""

        class TestObject:
            def __init__(self) -> None:
                self.attr1 = "value1"
                self.attr2 = 42

        obj = TestObject()
        result = FlextResult[TestObject].ok(obj)

        # Should handle object with __dict__
        hash_value = hash(result)
        assert isinstance(hash_value, int)

    def test_hash_success_with_complex_unhashable_object(self) -> None:
        """Test FlextResult.__hash__ with complex object that can't be hashed by attributes."""

        class ComplexObject:
            def __init__(self) -> None:
                self.unhashable_attr = [{"nested": "dict"}, {"in": "list"}]

        obj = ComplexObject()
        result = FlextResult[ComplexObject].ok(obj)

        # Should fall back to type name + id approach
        hash_value = hash(result)
        assert isinstance(hash_value, int)

    def test_hash_failure_case(self) -> None:
        """Test FlextResult.__hash__ with failed result."""
        result1 = FlextResult[str].fail("error message", error_code="ERR001")
        result2 = FlextResult[str].fail("error message", error_code="ERR001")
        result3 = FlextResult[str].fail("different error", error_code="ERR002")

        # Same error should have same hash
        assert hash(result1) == hash(result2)
        # Different error should have different hash
        assert hash(result1) != hash(result3)

    def test_safe_call_success(self) -> None:
        """Test FlextResult.safe_call with successful function."""

        def safe_func() -> str:
            return "safe_result"

        result = FlextResult.safe_call(safe_func)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "safe_result"

    def test_safe_call_exception_handling(self) -> None:
        """Test FlextResult.safe_call with function that raises exception."""

        def unsafe_func() -> str:
            msg = "Function failed"
            raise ValueError(msg)

        result = FlextResult.safe_call(unsafe_func)
        FlextTestsMatchers.assert_result_failure(result)
        assert "Function failed" in result.error

    def test_safe_call_generic_exception(self) -> None:
        """Test FlextResult.safe_call with generic exception type."""

        class TestError(Exception):
            """Custom test exception."""

        def failing_func() -> int:
            msg = "Generic error"
            raise TestError(msg)

        result = FlextResult.safe_call(failing_func)
        FlextTestsMatchers.assert_result_failure(result)
        assert "Generic error" in result.error

    def test_safe_call_with_complex_return_type(self) -> None:
        """Test FlextResult.safe_call with complex return type."""

        def complex_func() -> dict[str, Any]:
            return {"result": "complex", "data": [1, 2, 3]}

        result = FlextResult.safe_call(complex_func)
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == {"result": "complex", "data": [1, 2, 3]}

    def test_edge_case_error_codes_and_data(self) -> None:
        """Test edge cases with error codes and error data."""
        # Test with None error code
        result1 = FlextResult[str].fail("error", error_code=None)
        assert result1.error_code is None

        # Test with empty error data
        result2 = FlextResult[str].fail("error", error_data={})
        assert result2.error_data == {}

        # Test with complex error data
        error_data = {"context": {"user_id": 123}, "details": ["validation", "failed"]}
        result3 = FlextResult[str].fail("error", error_data=error_data)
        assert result3.error_data == error_data

    def test_metadata_edge_cases(self) -> None:
        """Test metadata handling edge cases."""
        # Test with success result - FlextResult doesn't seem to have metadata parameter
        # Testing error_data instead which is available
        result1 = FlextResult[str](data="test")
        # FlextResult doesn't have metadata attribute, test error_data instead
        assert result1.error_data == {}

        # Test with error data in failure case
        result2 = FlextResult[str](error="test error", error_data={"context": "test"})
        assert result2.error_data == {"context": "test"}

        # Test with complex error data
        error_data = {
            "timestamp": "2023-01-01",
            "source": "test",
            "tags": ["unit", "test"],
        }
        result3 = FlextResult[str](error="complex error", error_data=error_data)
        assert result3.error_data == error_data

    def test_result_construction_edge_cases(self) -> None:
        """Test FlextResult construction with edge case parameters."""
        # Test with failure case having None error_code and empty error_data
        result = FlextResult[Any](error="test error", error_code=None, error_data=None)
        assert result.is_failure
        assert result.error == "test error"
        assert result.error_code is None
        assert result.error_data == {}  # error_data defaults to empty dict

    def test_ensure_success_data_edge_cases(self) -> None:
        """Test _ensure_success_data method edge cases."""
        # Create success result with None data
        result = FlextResult[Any](data=None)

        # The _ensure_success_data method should handle None data in success state
        # This should not raise an exception
        assert result.is_success
        assert result.value is None  # Should return None for success with None data

    def test_integration_with_flext_tests_matchers(self) -> None:
        """Test integration with FlextTestsMatchers for comprehensive validation."""
        # Test complex success case
        success_result = FlextResult[dict].ok({"processed": True, "items": [1, 2, 3]})
        FlextTestsMatchers.assert_result_success(success_result)

        # Test complex failure case
        failure_result = FlextResult[dict].fail(
            "Processing failed",
            error_code="PROC_001",
            error_data={"failed_items": [1, 3], "retries": 3},
        )
        FlextTestsMatchers.assert_result_failure(failure_result)

        # Verify error details are preserved
        assert failure_result.error_code == "PROC_001"
        assert failure_result.error_data == {"failed_items": [1, 3], "retries": 3}
