"""Comprehensive tests for FlextGuards backward compatibility layer.

This module provides comprehensive test coverage for guards.py using extensive
flext_tests standardization patterns to achieve near 100% coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextGuards
from flext_tests import FlextTestsMatchers


class TestFlextGuardsComprehensive:
    """Comprehensive tests for FlextGuards backward compatibility layer."""

    # Test the nested _ValidationUtils class methods
    def test_validation_utils_require_not_none_success(self) -> None:
        """Test ValidationUtils.require_not_none with valid value."""
        result = FlextGuards._ValidationUtils.require_not_none(
            "valid_value", "test_param"
        )
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "valid_value"

    def test_validation_utils_require_not_none_failure(self) -> None:
        """Test ValidationUtils.require_not_none with None value."""
        result = FlextGuards._ValidationUtils.require_not_none(None, "test_param")
        FlextTestsMatchers.assert_result_failure(result)
        assert "test_param cannot be None" in result.error

    def test_validation_utils_require_not_none_default_name(self) -> None:
        """Test ValidationUtils.require_not_none with default parameter name."""
        result = FlextGuards._ValidationUtils.require_not_none(None)
        FlextTestsMatchers.assert_result_failure(result)
        assert "value cannot be None" in result.error

    def test_validation_utils_require_positive_success(self) -> None:
        """Test ValidationUtils.require_positive with positive value."""
        result = FlextGuards._ValidationUtils.require_positive(42, "test_number")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 42

    def test_validation_utils_require_positive_failure(self) -> None:
        """Test ValidationUtils.require_positive with negative value."""
        result = FlextGuards._ValidationUtils.require_positive(-5, "test_number")
        FlextTestsMatchers.assert_result_failure(result)
        assert "test_number must be positive" in result.error

    def test_validation_utils_require_positive_default_name(self) -> None:
        """Test ValidationUtils.require_positive with default parameter name."""
        result = FlextGuards._ValidationUtils.require_positive(-1)
        FlextTestsMatchers.assert_result_failure(result)
        assert "value must be positive" in result.error

    def test_validation_utils_require_in_range_success(self) -> None:
        """Test ValidationUtils.require_in_range with value in range."""
        result = FlextGuards._ValidationUtils.require_in_range(
            50, 10, 100, "test_value"
        )
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 50

    def test_validation_utils_require_in_range_failure_below(self) -> None:
        """Test ValidationUtils.require_in_range with value below range."""
        result = FlextGuards._ValidationUtils.require_in_range(5, 10, 100, "test_value")
        FlextTestsMatchers.assert_result_failure(result)
        assert "test_value out of range" in result.error

    def test_validation_utils_require_in_range_failure_above(self) -> None:
        """Test ValidationUtils.require_in_range with value above range."""
        result = FlextGuards._ValidationUtils.require_in_range(
            150, 10, 100, "test_value"
        )
        FlextTestsMatchers.assert_result_failure(result)
        assert "test_value out of range" in result.error

    def test_validation_utils_require_in_range_default_name(self) -> None:
        """Test ValidationUtils.require_in_range with default parameter name."""
        result = FlextGuards._ValidationUtils.require_in_range(5, 10, 100)
        FlextTestsMatchers.assert_result_failure(result)
        assert "value out of range" in result.error

    def test_validation_utils_require_non_empty_success_string(self) -> None:
        """Test ValidationUtils.require_non_empty with non-empty string."""
        result = FlextGuards._ValidationUtils.require_non_empty("hello", "test_string")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "hello"

    def test_validation_utils_require_non_empty_success_list(self) -> None:
        """Test ValidationUtils.require_non_empty with non-empty list."""
        test_list = [1, 2, 3]
        result = FlextGuards._ValidationUtils.require_non_empty(test_list, "test_list")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == test_list

    def test_validation_utils_require_non_empty_failure_empty_string(self) -> None:
        """Test ValidationUtils.require_non_empty with empty string."""
        result = FlextGuards._ValidationUtils.require_non_empty("", "test_string")
        FlextTestsMatchers.assert_result_failure(result)
        assert "test_string cannot be empty" in result.error

    def test_validation_utils_require_non_empty_failure_empty_list(self) -> None:
        """Test ValidationUtils.require_non_empty with empty list."""
        result = FlextGuards._ValidationUtils.require_non_empty([], "test_list")
        FlextTestsMatchers.assert_result_failure(result)
        assert "test_list cannot be empty" in result.error

    def test_validation_utils_require_non_empty_default_name(self) -> None:
        """Test ValidationUtils.require_non_empty with default parameter name."""
        result = FlextGuards._ValidationUtils.require_non_empty("")
        FlextTestsMatchers.assert_result_failure(result)
        assert "value cannot be empty" in result.error

    # Test the direct static methods
    def test_is_dict_of_success_string_values(self) -> None:
        """Test is_dict_of with dictionary of string values."""
        test_dict = {"key1": "value1", "key2": "value2"}
        result = FlextGuards.is_dict_of(test_dict, str)
        assert result is True

    def test_is_dict_of_success_int_values(self) -> None:
        """Test is_dict_of with dictionary of integer values."""
        test_dict = {"key1": 1, "key2": 2, "key3": 3}
        result = FlextGuards.is_dict_of(test_dict, int)
        assert result is True

    def test_is_dict_of_failure_mixed_types(self) -> None:
        """Test is_dict_of with mixed value types."""
        test_dict = {"key1": "string", "key2": 42}
        result = FlextGuards.is_dict_of(test_dict, str)
        assert result is False

    def test_is_dict_of_failure_not_dict(self) -> None:
        """Test is_dict_of with non-dictionary input."""
        result = FlextGuards.is_dict_of([1, 2, 3], str)
        assert result is False

    def test_is_dict_of_empty_dict(self) -> None:
        """Test is_dict_of with empty dictionary."""
        result = FlextGuards.is_dict_of({}, str)
        assert result is True  # Empty dict is valid for any type

    def test_is_list_of_success_string_items(self) -> None:
        """Test is_list_of with list of string items."""
        test_list = ["item1", "item2", "item3"]
        result = FlextGuards.is_list_of(test_list, str)
        assert result is True

    def test_is_list_of_success_int_items(self) -> None:
        """Test is_list_of with list of integer items."""
        test_list = [1, 2, 3, 4]
        result = FlextGuards.is_list_of(test_list, int)
        assert result is True

    def test_is_list_of_failure_mixed_types(self) -> None:
        """Test is_list_of with mixed item types."""
        test_list = ["string", 42, "another_string"]
        result = FlextGuards.is_list_of(test_list, str)
        assert result is False

    def test_is_list_of_failure_not_list(self) -> None:
        """Test is_list_of with non-list input."""
        result = FlextGuards.is_list_of({"key": "value"}, str)
        assert result is False

    def test_is_list_of_empty_list(self) -> None:
        """Test is_list_of with empty list."""
        result = FlextGuards.is_list_of([], str)
        assert result is True  # Empty list is valid for any type

    def test_direct_require_not_none_success(self) -> None:
        """Test direct require_not_none with valid value."""
        result = FlextGuards.require_not_none("valid_value", "direct_param")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "valid_value"

    def test_direct_require_not_none_failure(self) -> None:
        """Test direct require_not_none with None value."""
        result = FlextGuards.require_not_none(None, "direct_param")
        FlextTestsMatchers.assert_result_failure(result)
        assert "direct_param cannot be None" in result.error

    def test_direct_require_positive_success(self) -> None:
        """Test direct require_positive with positive value."""
        result = FlextGuards.require_positive(100, "direct_number")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 100

    def test_direct_require_positive_failure(self) -> None:
        """Test direct require_positive with negative value."""
        result = FlextGuards.require_positive(-10, "direct_number")
        FlextTestsMatchers.assert_result_failure(result)
        assert "direct_number must be positive" in result.error

    def test_direct_require_in_range_success(self) -> None:
        """Test direct require_in_range with value in range."""
        result = FlextGuards.require_in_range(75, 50, 100, "direct_value")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == 75

    def test_direct_require_in_range_failure(self) -> None:
        """Test direct require_in_range with value out of range."""
        result = FlextGuards.require_in_range(25, 50, 100, "direct_value")
        FlextTestsMatchers.assert_result_failure(result)
        assert "direct_value out of range" in result.error

    def test_direct_require_non_empty_success(self) -> None:
        """Test direct require_non_empty with non-empty value."""
        result = FlextGuards.require_non_empty("direct_string", "direct_param")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "direct_string"

    def test_direct_require_non_empty_failure(self) -> None:
        """Test direct require_non_empty with empty value."""
        result = FlextGuards.require_non_empty("", "direct_param")
        FlextTestsMatchers.assert_result_failure(result)
        assert "direct_param cannot be empty" in result.error

    # Test ValidationUtils compatibility
    def test_validation_utils_compatibility_access(self) -> None:
        """Test ValidationUtils can be accessed as class attribute."""
        # Test that ValidationUtils is accessible
        assert hasattr(FlextGuards, "ValidationUtils")
        assert FlextGuards.ValidationUtils is FlextGuards._ValidationUtils

        # Test methods are callable through compatibility layer
        result = FlextGuards.ValidationUtils.require_not_none("test", "compat_param")
        FlextTestsMatchers.assert_result_success(result)
        assert result.value == "test"

    # Test edge cases for comprehensive coverage
    def test_edge_cases_zero_values(self) -> None:
        """Test edge cases with zero values."""
        # Zero is not positive
        result = FlextGuards.require_positive(0, "zero_value")
        FlextTestsMatchers.assert_result_failure(result)
        assert "zero_value must be positive" in result.error

    def test_edge_cases_boundary_values(self) -> None:
        """Test edge cases with boundary values."""
        # Test exact boundaries
        result_min = FlextGuards.require_in_range(10, 10, 100, "boundary_min")
        FlextTestsMatchers.assert_result_success(result_min)
        assert result_min.value == 10

        result_max = FlextGuards.require_in_range(100, 10, 100, "boundary_max")
        FlextTestsMatchers.assert_result_success(result_max)
        assert result_max.value == 100

    def test_comprehensive_type_checking(self) -> None:
        """Test comprehensive type checking scenarios."""
        # Test with various types for is_dict_of
        assert FlextGuards.is_dict_of({"a": 1.0, "b": 2.0}, float) is True
        assert FlextGuards.is_dict_of({"a": True, "b": False}, bool) is True
        assert FlextGuards.is_dict_of({}, object) is True  # Empty dict

        # Test with various types for is_list_of
        assert FlextGuards.is_list_of([1.1, 2.2, 3.3], float) is True
        assert FlextGuards.is_list_of([True, False, True], bool) is True
        assert FlextGuards.is_list_of([], object) is True  # Empty list
