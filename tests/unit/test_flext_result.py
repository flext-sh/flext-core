"""Comprehensive tests for FlextResult using all test infrastructure."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TypeVar, cast
from unittest.mock import Mock

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from flext_core import FlextResult
from flext_tests import (
    FlextResultFactory,
    TestBuilders,
)

# Type variable for generic test utilities
T = TypeVar("T")


# Test Utilities and Matchers
class TestUtilities:
    """Test utilities for FlextResult validation."""

    @staticmethod
    def is_valid_result(result: FlextResult[T]) -> bool:
        """Check if result is valid (either success or failure)."""
        return result.is_success or result.is_failure

    @staticmethod
    def result_has_data(result: FlextResult[T]) -> bool:
        """Check if result has data."""
        return result.is_success and result.value is not None


class FlextMatchers:
    """Custom matchers for FlextResult testing."""

    @staticmethod
    def is_successful_result(result: FlextResult[T]) -> bool:
        """Check if result is successful."""
        return result.is_success

    @staticmethod
    def is_failed_result(result: FlextResult[T]) -> bool:
        """Check if result failed."""
        return result.is_failure

    @staticmethod
    def result_contains_data(
        result: FlextResult[T],
        expected_data: T,
    ) -> bool:
        """Check if result contains expected data."""
        return result.is_success and result.value == expected_data

    @staticmethod
    def result_has_value_type(result: FlextResult[T], expected_type: type) -> bool:
        """Check if result value has expected type."""
        return result.is_success and isinstance(result.value, expected_type)

    @staticmethod
    def result_has_error_code(result: FlextResult[T], expected_code: str) -> bool:
        """Check if result has expected error code."""
        return result.is_failure and result.error_code == expected_code

    @staticmethod
    def result_has_error_message(
        result: FlextResult[T],
        expected_message: str,
    ) -> bool:
        """Check if result has expected error message."""
        return result.is_failure and expected_message in (result.error or "")


class TestFlextResultComprehensive:
    """Comprehensive FlextResult testing with full coverage."""

    # =========================================================================
    # SUCCESS SCENARIOS - Using TestBuilders and Factories
    # =========================================================================

    def test_success_result_creation_with_builders(self) -> None:
        """Test success result creation using TestBuilders."""
        # Use factory to create test data
        test_data = {"user_id": "123", "name": "test", "active": True}

        # Use builder pattern
        result = TestBuilders.result().with_success_data(test_data).build()

        # Use custom assertions
        assert result.success
        assert result.success
        assert result.value == test_data
        assert result.error is None
        assert result.error_code is None

    def test_success_result_with_different_data_types(
        self,
        user_factory: FlextResultFactory,  # From conftest.py
    ) -> None:
        """Test success results with various data types."""
        # String result
        string_result = FlextResult[str].ok("success message")
        assert FlextMatchers.is_successful_result(string_result)
        assert string_result.value == "success message"

        # Integer result
        int_result = FlextResult[int].ok(42)
        assert int_result.success
        assert int_result.value == 42

        # List result
        list_result = FlextResult[list[str]].ok(["a", "b", "c"])
        assert list_result.success
        assert list_result.value == ["a", "b", "c"]

        # Dict result
        dict_result = FlextResult[dict[str, object]].ok({"key": "value"})
        assert dict_result.success
        assert dict_result.value == {"key": "value"}

        # None result (valid for operations that don't return data)
        none_result = FlextResult[None].ok(None)
        assert none_result.success
        assert none_result.value is None

    # =========================================================================
    # FAILURE SCENARIOS - Using TestBuilders and Edge Cases
    # =========================================================================

    def test_failure_result_creation_comprehensive(self) -> None:
        """Test failure result creation with all parameters."""
        # Simple failure
        simple_fail = FlextResult[str].fail("Operation failed")
        assert simple_fail.is_failure
        assert simple_fail.error == "Operation failed"
        assert simple_fail.error_code is None
        # Accessing .value on failed result should raise TypeError
        with pytest.raises(
            TypeError,
            match="Attempted to access value on failed result",
        ):
            _ = simple_fail.value

        # Failure with error code
        coded_fail = FlextResult[str].fail(
            "Validation error",
            error_code="VALIDATION_001",
        )
        assert coded_fail.is_failure
        assert coded_fail.error == "Validation error"
        assert coded_fail.error_code == "VALIDATION_001"

        # Failure with error data
        data_fail = FlextResult[str].fail(
            "Complex error",
            error_code="COMPLEX_001",
            error_data={"field": "email", "value": "invalid"},
        )
        assert data_fail.is_failure
        assert data_fail.error_data == {"field": "email", "value": "invalid"}

    def test_failure_result_with_builder_pattern(self) -> None:
        """Test failure results using builder pattern."""
        # Use TestBuilders for failure scenarios
        failure_result = (
            TestBuilders.result()
            .with_failure(
                "Database connection failed",
                "DB_CONN_001",
                {"host": "localhost", "port": 5432, "timeout": True},
            )
            .build()
        )

        assert FlextMatchers.is_failed_result(failure_result)
        assert failure_result.error == "Database connection failed"
        assert failure_result.error_code == "DB_CONN_001"
        assert failure_result.error_data == {
            "host": "localhost",
            "port": 5432,
            "timeout": True,
        }

    # =========================================================================
    # EDGE CASES AND ERROR SCENARIOS
    # =========================================================================

    def test_result_with_large_data_structures(
        self,
        benchmark: BenchmarkFixture,
    ) -> None:
        """Test FlextResult with large data structures and performance."""
        # Create large data structure
        large_data = {f"key_{i}": f"value_{i}" for i in range(10000)}

        # Benchmark result creation
        def create_large_result() -> FlextResult[dict[str, str]]:
            return FlextResult[dict[str, str]].ok(large_data)

        result = benchmark(create_large_result)

        assert result.success
        assert len(result.value) == 10000
        assert FlextMatchers.is_successful_result(result)

    def test_result_serialization_edge_cases(self) -> None:
        """Test FlextResult with complex serialization scenarios."""
        # Test with non-serializable object (Mock)
        mock_object = Mock()
        mock_object.name = "test_mock"

        result_with_mock = FlextResult[Mock].ok(mock_object)
        assert result_with_mock.success
        assert result_with_mock.value.name == "test_mock"

        # Test with nested complex data
        complex_data: dict[str, object] = {
            "nested": {"level": 2, "data": [1, 2, 3]},
            "timestamp": "2024-01-01T00:00:00Z",
            "metadata": {"source": "test", "version": 1.0},
        }

        complex_result = FlextResult[dict[str, object]].ok(complex_data)
        assert complex_result.success
        nested = complex_result.value
        assert cast("dict[str, object]", nested["nested"])["level"] == 2

    def test_result_chaining_patterns(self) -> None:
        """Test FlextResult chaining and composition patterns."""

        # Success chain
        def process_data(data: str) -> FlextResult[str]:
            if not data:
                return FlextResult[str].fail("Empty data")
            return FlextResult[str].ok(data.upper())

        def validate_data(data: str) -> FlextResult[str]:
            if len(data) < 3:
                return FlextResult[str].fail("Data too short")
            return FlextResult[str].ok(data)

        # Test successful chain
        initial_result = FlextResult[str].ok("hello")
        if initial_result.success and initial_result.value:
            processed = process_data(initial_result.value)
            if processed.success and processed.value:
                final = validate_data(processed.value)
                assert final.success
                assert final.value == "HELLO"

        # Test failure chain
        empty_result = FlextResult[str].ok("")
        if empty_result.success and empty_result.value is not None:
            processed = process_data(empty_result.value)
            assert processed.is_failure
            assert "Empty data" in (processed.error or "")

    # =========================================================================
    # INTEGRATION WITH TEST UTILITIES
    # =========================================================================

    def test_result_with_test_utilities_validation(self) -> None:
        """Test FlextResult integration with TestUtilities."""
        # Create test result
        result = FlextResult[dict[str, str]].ok({"status": "active", "role": "REDACTED_LDAP_BIND_PASSWORD"})

        # Use TestUtilities for validation
        assert TestUtilities.is_valid_result(result)
        assert TestUtilities.result_has_data(result)

        # Test with failure result
        failure = FlextResult[dict[str, str]].fail(
            "Access denied",
            error_code="AUTH_001",
        )
        assert TestUtilities.is_valid_result(failure)  # Valid result, but failed
        assert not TestUtilities.result_has_data(failure)

    def test_result_with_custom_matchers_comprehensive(self) -> None:
        """Test FlextResult with all custom matchers."""
        success_result = FlextResult[str].ok("test_data")
        failure_result = FlextResult[str].fail("test_error", error_code="ERR_001")

        # Success matchers
        assert FlextMatchers.is_successful_result(success_result)
        assert FlextMatchers.result_contains_data(success_result, "test_data")
        assert FlextMatchers.result_has_value_type(success_result, str)

        # Failure matchers
        assert FlextMatchers.is_failed_result(failure_result)
        assert FlextMatchers.result_has_error_code(failure_result, "ERR_001")
        assert FlextMatchers.result_has_error_message(failure_result, "test_error")

        # Negative assertions
        assert not FlextMatchers.is_successful_result(failure_result)
        assert not FlextMatchers.is_failed_result(success_result)

    # =========================================================================
    # PERFORMANCE AND MEMORY TESTING
    # =========================================================================

    def test_result_memory_efficiency(self) -> None:
        """Test FlextResult memory usage patterns."""
        # Test that results properly hold references to data
        large_data = list(range(1000))
        result = FlextResult[list[int]].ok(large_data)

        # Reference should be maintained
        assert len(result.value) == 1000

        # Modifying original data should affect result (shared reference is expected behavior)
        original_length = len(result.value)
        assert original_length == 1000

        # Test memory cleanup on failure - failed results don't have accessible values
        failure_result = FlextResult[list[int]].fail("Test error")
        # Failed results raise TypeError when accessing .value (no data reference)
        with pytest.raises(
            TypeError,
            match="Attempted to access value on failed result",
        ):
            _ = failure_result.value

    def test_result_concurrent_access_safety(self) -> None:
        """Test FlextResult thread safety characteristics."""
        # FlextResult should be immutable after creation
        result = FlextResult[dict[str, str]].ok({"key": "value"})

        original_value = result.value
        original_success = result.success
        original_error = result.error

        # These should remain unchanged
        assert result.value is original_value
        assert result.success == original_success
        assert result.error == original_error

        # Properties should be consistent
        assert result.is_failure == (not result.success)

    # =========================================================================
    # FILE I/O AND PERSISTENCE EDGE CASES
    # =========================================================================

    def test_result_with_file_operations(self) -> None:
        """Test FlextResult with file I/O edge cases."""
        # Test with temporary file
        with tempfile.NamedTemporaryFile(encoding="utf-8", mode="w", delete=False) as f:
            f.write("test content")
            temp_path = f.name

        # Successful file read
        try:
            path_obj = Path(temp_path)
            if path_obj.exists():
                content = path_obj.read_text(encoding="utf-8")
                result = FlextResult[str].ok(content)
                assert result.success
                assert result.value == "test content"
        finally:
            Path(temp_path).unlink(missing_ok=True)

        # Failed file operation
        nonexistent_path = Path("/nonexistent/path/file.txt")
        failure_result = FlextResult[str].fail(
            f"File not found: {nonexistent_path}",
            error_code="FILE_NOT_FOUND",
            error_data={"path": str(nonexistent_path), "operation": "read"},
        )

        assert failure_result.is_failure
        assert "File not found" in (failure_result.error or "")
        assert failure_result.error_code == "FILE_NOT_FOUND"

    def test_result_json_serialization_comprehensive(self) -> None:
        """Test FlextResult JSON serialization edge cases."""
        # Test success result with JSON-serializable data
        json_data: dict[str, object] = {
            "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        }
        result = FlextResult[dict[str, object]].ok(json_data)

        # Should be able to serialize the data
        serialized = json.dumps(result.value)
        deserialized = json.loads(serialized)
        assert deserialized == json_data

        # Test failure result serialization
        failure = FlextResult[dict[str, object]].fail(
            "JSON serialization error",
            error_code="JSON_001",
            error_data={
                "invalid_type": str(type(object())),
            },  # Convert to string for JSON
        )

        error_data_json = json.dumps(failure.error_data)
        assert "invalid_type" in error_data_json


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-skip"])
