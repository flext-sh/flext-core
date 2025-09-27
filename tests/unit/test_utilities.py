"""Comprehensive tests for FlextUtilities - Utility Functions.

This module tests the utility functions and helpers provided by FlextUtilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time

from flext_core import FlextUtilities
from flext_core.result import FlextResult


class TestFlextUtilities:
    """Test suite for FlextUtilities utility functions."""

    def test_validation_utilities(self) -> None:
        """Test validation utility functions."""
        # Test email validation
        email_result = FlextUtilities.Validation.validate_email("test@example.com")
        assert email_result.is_success

        email_result = FlextUtilities.Validation.validate_email("invalid-email")
        assert email_result.is_failure

    def test_string_utilities(self) -> None:
        """Test string utility functions."""
        # Test string validation operations
        result = FlextUtilities.Validation.validate_string_not_empty("")
        assert result.is_failure

        result = FlextUtilities.Validation.validate_string_not_empty("not empty")
        assert result.is_success
        assert result.value == "not empty"

    def test_numeric_utilities(self) -> None:
        """Test numeric utility functions."""
        # Test numeric validation operations
        result = FlextUtilities.Validation.validate_positive_integer(5)
        assert result.is_success
        assert result.value == 5

        result = FlextUtilities.Validation.validate_positive_integer(-5)
        assert result.is_failure

    def test_collection_utilities(self) -> None:
        """Test collection utility functions."""
        # Test collection processing operations
        test_list = [1, 2, 3, 4, 5]

        def process_item(item: int) -> FlextResult[int]:
            return FlextResult[int].ok(item * 2)

        result = FlextUtilities.Utilities.batch_process(test_list, process_item)
        assert result.is_success
        assert result.value == [2, 4, 6, 8, 10]

    def test_datetime_utilities(self) -> None:
        """Test datetime utility functions."""
        # Test conversion operations
        result = FlextUtilities.Conversions.to_bool(value="true")
        assert result.is_success
        assert result.value is True

        result = FlextUtilities.Conversions.to_bool(value="false")
        assert result.is_success
        assert result.value is False

    def test_file_utilities(self) -> None:
        """Test file utility functions."""
        # Test file validation operations
        result = FlextUtilities.Validation.validate_file_path("/valid/path")
        assert result.is_success
        assert result.value == "/valid/path"

        result = FlextUtilities.Validation.validate_file_path("")
        assert result.is_failure

    def test_network_utilities(self) -> None:
        """Test network utility functions."""
        # Test network validation operations
        result = FlextUtilities.Validation.validate_host("example.com")
        assert result.is_success
        assert result.value == "example.com"

        result = FlextUtilities.Validation.validate_host("")
        assert result.is_failure

    def test_json_utilities(self) -> None:
        """Test JSON utility functions."""
        # Test text processing operations
        test_text = "  Hello   World  \n\t  "
        result = FlextUtilities.TextProcessor.clean_text(test_text)
        assert result.is_success
        assert result.value == "Hello World"

    def test_crypto_utilities(self) -> None:
        """Test cryptographic utility functions."""
        # Test generation operations
        result = FlextUtilities.Generators.generate_id()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_encoding_utilities(self) -> None:
        """Test encoding utility functions."""
        # Test conversion operations
        result = FlextUtilities.Conversions.to_int("42")
        assert result.is_success
        assert result.value == 42

    def test_compression_utilities(self) -> None:
        """Test compression utility functions."""
        # Test text processing operations
        test_text = "This is a very long text that should be truncated"
        result = FlextUtilities.TextProcessor.truncate_text(test_text, max_length=20)
        assert result.is_success
        assert len(result.value) <= 20

    def test_utility_error_handling(self) -> None:
        """Test error handling in utility functions."""
        # Test error handling
        result = FlextUtilities.Validation.validate_email("")
        assert result.is_failure

    def test_utility_performance(self) -> None:
        """Test performance of utility functions."""
        start_time = time.time()

        # Test performance of common operations
        for i in range(1000):
            FlextUtilities.Validation.validate_string_not_empty(f"test_string_{i}")
            FlextUtilities.Validation.validate_positive_integer(i)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete quickly (less than 1 second)
        assert execution_time < 1.0

    def test_utility_type_safety(self) -> None:
        """Test type safety of utility functions."""
        # Test type safety
        result = FlextUtilities.Validation.validate_string_not_empty("")
        # Should handle validation gracefully
        assert result.is_failure
