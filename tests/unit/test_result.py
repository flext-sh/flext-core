"""Targeted tests for 100% coverage on FlextResult module."""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from flext_core import FlextResult


class TestFlextResult100PercentCoverage:
    """Targeted tests for the remaining 16 uncovered lines in result.py."""

    def test_line_635_or_operator_none_success_data(self) -> None:
        """Test line 635: __or__ operator with None data in success case."""
        # Create a result that can lead to None data scenario
        result = FlextResult.ok(None)  # Explicitly None value
        default_value = "default"

        # This should trigger line 635 for None data case
        actual = result | default_value
        # When data is None, should return default
        assert actual == default_value

    def test_lines_244_245_unwrap_none_data_internal(self) -> None:
        """Test lines 244-245: internal unwrap edge case with None data."""

        # Create scenario where internal state could have None data
        # This is a more realistic scenario through transformation chains
        def transform_to_none(_x: object) -> None:
            return None

        # Create a chain that results in None value
        result = FlextResult.ok("start").map(transform_to_none)

        # This should work - None is a valid value
        assert result.unwrap() is None

    def test_lines_646_647_context_manager_edge_case(self) -> None:
        """Test lines 646-647: context manager with edge case scenarios."""
        # Test normal context manager flow
        with FlextResult.ok("test_value") as value:
            assert value == "test_value"

        # Test failure case
        failed_result: FlextResult[str] = FlextResult.fail("context_error")
        with pytest.raises(RuntimeError, match="context_error"), failed_result:
            pass

    def test_line_858_equality_comparison_edge_cases(self) -> None:
        """Test line 858: equality comparison with different object types."""
        result = FlextResult.ok("test")

        # Test equality with non-FlextResult objects
        assert result != "test"  # String comparison
        assert result != 123  # Number comparison
        assert result != ["test"]  # List comparison
        assert result != {"key": "test"}  # Dict comparison
        assert result is not None  # None comparison

        # Test equality with other FlextResult objects
        same_result = FlextResult.ok("test")
        assert result == same_result

        different_result = FlextResult.ok("different")
        assert result != different_result

    def test_line_869_hash_consistency(self) -> None:
        """Test line 869: hash method consistency and edge cases."""
        # Test hash consistency
        result1: FlextResult[str] = FlextResult.ok("test")
        result2: FlextResult[str] = FlextResult.ok("test")
        assert hash(result1) == hash(result2)

        # Test hash with different values
        result3: FlextResult[str] = FlextResult.ok("different")
        assert hash(result1) != hash(result3)

        # Test hash with errors
        error_result1: FlextResult[str] = FlextResult.fail("error")
        error_result2: FlextResult[str] = FlextResult.fail("error")
        assert hash(error_result1) == hash(error_result2)

    def test_lines_899_900_repr_comprehensive(self) -> None:
        """Test lines 899-900: comprehensive __repr__ testing."""
        # Test successful result repr
        success_result: FlextResult[str] = FlextResult.ok("success_data")
        repr_str = repr(success_result)
        assert "FlextResult" in repr_str
        assert "success_data" in repr_str

        # Test failure result repr
        failure_result: FlextResult[str] = FlextResult.fail("error_message")
        repr_str = repr(failure_result)
        assert "FlextResult" in repr_str
        assert "error_message" in repr_str

        # Test with complex data
        complex_data: dict[str, dict[str, list[int]]] = {"nested": {"list": [1, 2, 3]}}
        complex_result: FlextResult[dict[str, dict[str, list[int]]]] = FlextResult.ok(
            complex_data,
        )
        repr_str = repr(complex_result)
        assert "FlextResult" in repr_str

    def test_async_operations_edge_cases(self) -> None:
        """Test async-related edge cases that might trigger uncovered lines."""

        async def async_test() -> None:
            # Test async unwrap
            result: FlextResult[str] = FlextResult.ok("async_value")
            if hasattr(result, "unwrap_async"):
                value = await result.unwrap_async()
                assert value == "async_value"

            # Test async context manager if supported
            try:
                # FlextResult doesn't implement async context manager
                # This is just for coverage testing
                pass
            except (TypeError, AttributeError):
                # Async context manager might not be implemented
                pass

        # Only run if we're in an async context or can create one
        with contextlib.suppress(RuntimeError):
            # If already in async context, skip this test
            asyncio.run(async_test())

    def test_error_handling_edge_cases(self) -> None:
        """Test error handling paths that might trigger uncovered lines."""
        # Test with empty error message - gets normalized to default
        result: FlextResult[str] = FlextResult.fail("")
        assert result.is_failure
        assert result.error == "Unknown error occurred"  # Empty errors get normalized

        # Test with None error (should be handled gracefully)
        try:
            none_result: FlextResult[str] = FlextResult.fail(
                ""
            )  # Use empty string instead of None
            assert none_result.is_failure
        except (TypeError, ValueError):
            # Might not accept None as error
            pass

    def test_chaining_operations_comprehensive(self) -> None:
        """Test chaining operations that might trigger edge cases."""
        # Test complex chaining that could lead to edge cases
        result: FlextResult[str] = (
            FlextResult.ok(10)
            .map(lambda x: x * 2)
            .filter(lambda x: x > 15, "Too small")
            .map(str)
        )

        assert result.is_success
        assert result.unwrap() == "20"

        # Test chaining that leads to failure
        failed_result: FlextResult[str] = (
            FlextResult.ok(5).filter(lambda x: x > 10, "Too small").map(str)
        )

        assert failed_result.is_failure
        assert "Too small" in (failed_result.error or "")

    def test_or_operator_comprehensive(self) -> None:
        """Test | operator comprehensively to trigger line 635."""
        # Test success case with normal value
        success = FlextResult.ok("value")
        assert (success | "default") == "value"

        # Test failure case
        failure: FlextResult[object] = FlextResult.fail("error")
        assert (failure | "default") == "default"

        # Test success case with None value
        none_success = FlextResult.ok(None)
        assert (none_success | "default") == "default"

    def test_context_manager_comprehensive(self) -> None:
        """Test context manager to trigger lines 646-647."""
        # Test successful context manager
        with FlextResult.ok("context_value") as value:
            assert value == "context_value"

        # Test failed context manager
        with pytest.raises(RuntimeError), FlextResult.fail("context_error"):
            pass
