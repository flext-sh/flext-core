"""Comprehensive tests for FlextStandardization - Data Standardization.

This module tests the data standardization functionality provided by FlextStandardization.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time

from flext_core.standardization import FlextStandardization


class TestFlextStandardization:
    """Test suite for FlextStandardization data standardization."""

    def test_standardization_initialization(self) -> None:
        """Test standardization initialization."""
        std = FlextStandardization()

        assert std is not None
        assert hasattr(std, "standardize_data")

    def test_data_standardization(self) -> None:
        """Test basic data standardization."""
        # Test string standardization
        result = FlextStandardization.standardize_data("test string")
        assert result.is_success
        assert result.value == "test string"

    def test_numeric_standardization(self) -> None:
        """Test numeric data standardization."""
        # Test numeric standardization
        result = FlextStandardization.standardize_data(123.45)
        assert result.is_success
        assert result.value == 123.45

    def test_boolean_standardization(self) -> None:
        """Test boolean data standardization."""
        # Test boolean standardization
        result = FlextStandardization.standardize_data(True)
        assert result.is_success
        assert result.value is True

    def test_list_standardization(self) -> None:
        """Test list data standardization."""
        # Test list standardization
        test_list = [1, 2, 3, 4]
        result = FlextStandardization.standardize_data(test_list)
        assert result.is_success
        assert result.value == test_list

    def test_dict_standardization(self) -> None:
        """Test dictionary data standardization."""
        # Test dict standardization
        test_dict = {"key1": "value1", "key2": 123}
        result = FlextStandardization.standardize_data(test_dict)
        assert result.is_success
        assert result.value == test_dict

    def test_nested_data_standardization(self) -> None:
        """Test nested data standardization."""
        # Test nested data standardization
        test_data = {
            "users": [{"name": "John", "age": 25}, {"name": "Jane", "age": 30}],
            "metadata": {"created": "2023-01-01"},
        }

        result = FlextStandardization.standardize_data(test_data)
        assert result.is_success
        assert result.value == test_data

    def test_standardization_error_handling(self) -> None:
        """Test standardization error handling."""
        # Test with None data
        result = FlextStandardization.standardize_data(None)
        assert result.is_failure
        assert "Data cannot be empty" in result.error

    def test_standardization_performance(self) -> None:
        """Test standardization performance."""
        # Create large dataset
        large_data = [
            {
                "id": i,
                "name": f"User {i}",
                "email": f"user{i}@example.com",
                "active": i % 2 == 0,
            }
            for i in range(1000)
        ]

        start_time = time.time()
        result = FlextStandardization.standardize_data(large_data)
        end_time = time.time()

        execution_time = end_time - start_time

        # Should complete quickly (less than 1 second)
        assert execution_time < 1.0
        assert result.is_success

    def test_standardization_type_preservation(self) -> None:
        """Test that standardization preserves data types."""
        test_data = {
            "string": "test",
            "integer": 123,
            "float": 123.45,
            "boolean": True,
            "list": [1, 2, 3],
            "nested_dict": {"key": "value"},
        }

        result = FlextStandardization.standardize_data(test_data)
        assert result.is_success

        standardized = result.value
        assert isinstance(standardized["string"], str)
        assert isinstance(standardized["integer"], int)
        assert isinstance(standardized["float"], float)
        assert isinstance(standardized["boolean"], bool)
        assert isinstance(standardized["list"], list)
        assert isinstance(standardized["nested_dict"], dict)

    def test_standardization_edge_cases(self) -> None:
        """Test standardization edge cases."""
        # Test empty data (considered invalid by the current implementation)
        result = FlextStandardization.standardize_data({})
        assert result.is_failure
        assert "Data cannot be empty" in result.error

        # Test empty list (considered invalid by the current implementation)
        result = FlextStandardization.standardize_data([])
        assert result.is_failure
        assert "Data cannot be empty" in result.error

        # Test single value
        result = FlextStandardization.standardize_data("single value")
        assert result.is_success
        assert result.value == "single value"
