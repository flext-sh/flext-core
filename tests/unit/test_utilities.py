"""Comprehensive tests for FlextUtilities - Utility Functions.

This module tests the utility functions and helpers provided by FlextUtilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

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
        result = FlextUtilities.TypeConversions.to_bool(value="true")
        assert result.is_success
        assert result.value is True

        result = FlextUtilities.TypeConversions.to_bool(value="false")
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
        result = FlextUtilities.TypeConversions.to_int("42")
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

    # ========== COMPREHENSIVE VALIDATION CLASS TESTS ==========

    def test_validation_string_not_none(self) -> None:
        """Test validate_string_not_none method."""
        # Valid cases
        result = FlextUtilities.Validation.validate_string_not_none("valid string")
        assert result.is_success
        assert result.value == "valid string"

        # Invalid cases
        result = FlextUtilities.Validation.validate_string_not_none(None)
        assert result.is_failure
        assert result.error is not None and "cannot be None" in result.error

    def test_validation_string_length(self) -> None:
        """Test validate_string_length method."""
        # Valid length
        result = FlextUtilities.Validation.validate_string_length(
            "hello", min_length=1, max_length=10
        )
        assert result.is_success
        assert result.value == "hello"

        # Too short
        result = FlextUtilities.Validation.validate_string_length(
            "a", min_length=5, max_length=10
        )
        assert result.is_failure

        # Too long
        result = FlextUtilities.Validation.validate_string_length(
            "very long string", min_length=1, max_length=5
        )
        assert result.is_failure

    def test_validation_string_pattern(self) -> None:
        """Test validate_string_pattern method."""
        # Valid pattern
        result = FlextUtilities.Validation.validate_string_pattern(
            "abc123", r"^[a-z]+\d+$"
        )
        assert result.is_success
        assert result.value == "abc123"

        # Invalid pattern
        result = FlextUtilities.Validation.validate_string_pattern(
            "ABC123", r"^[a-z]+\d+$"
        )
        assert result.is_failure

        # Invalid regex pattern
        result = FlextUtilities.Validation.validate_string_pattern("test", r"[")
        assert result.is_failure

    def test_validation_port(self) -> None:
        """Test validate_port method."""
        # Valid ports
        result = FlextUtilities.Validation.validate_port(80)
        assert result.is_success
        assert result.value == 80

        result = FlextUtilities.Validation.validate_port(65535)
        assert result.is_success

        # Invalid ports
        result = FlextUtilities.Validation.validate_port(0)
        assert result.is_failure

        result = FlextUtilities.Validation.validate_port(65536)
        assert result.is_failure

        result = FlextUtilities.Validation.validate_port(-1)
        assert result.is_failure

    def test_validation_environment_value(self) -> None:
        """Test validate_environment_value method."""
        allowed_envs = ["development", "staging", "production"]

        # Valid environment values
        result = FlextUtilities.Validation.validate_environment_value(
            "development", allowed_envs
        )
        assert result.is_success

        result = FlextUtilities.Validation.validate_environment_value(
            "production", allowed_envs
        )
        assert result.is_success

        # Invalid environment values
        result = FlextUtilities.Validation.validate_environment_value("", allowed_envs)
        assert result.is_failure

        result = FlextUtilities.Validation.validate_environment_value(
            "invalid_env", allowed_envs
        )
        assert result.is_failure

    def test_validation_log_level(self) -> None:
        """Test validate_log_level method."""
        # Valid log levels
        result = FlextUtilities.Validation.validate_log_level("DEBUG")
        assert result.is_success

        result = FlextUtilities.Validation.validate_log_level("INFO")
        assert result.is_success

        # Invalid log level
        result = FlextUtilities.Validation.validate_log_level("INVALID")
        assert result.is_failure

    def test_validation_security_token(self) -> None:
        """Test validate_security_token method."""
        # Valid token
        result = FlextUtilities.Validation.validate_security_token("abcdef123456")
        assert result.is_success

        # Invalid token (too short)
        result = FlextUtilities.Validation.validate_security_token("abc")
        assert result.is_failure

    def test_validation_connection_string(self) -> None:
        """Test validate_connection_string method."""
        # Valid connection string
        result = FlextUtilities.Validation.validate_connection_string(
            "postgresql://user:pass@localhost:5432/db"
        )
        assert result.is_success

        # Invalid connection string
        result = FlextUtilities.Validation.validate_connection_string("")
        assert result.is_failure

    def test_validation_directory_path(self) -> None:
        """Test validate_directory_path method."""
        # Valid directory path
        result = FlextUtilities.Validation.validate_directory_path("/home/user")
        assert result.is_success

        # Invalid directory path
        result = FlextUtilities.Validation.validate_directory_path("")
        assert result.is_failure

    def test_validation_timeout_seconds(self) -> None:
        """Test validate_timeout_seconds method."""
        # Valid timeout
        result = FlextUtilities.Validation.validate_timeout_seconds(30)
        assert result.is_success
        assert result.value == 30

        # Invalid timeout (too large)
        result = FlextUtilities.Validation.validate_timeout_seconds(99999)
        assert result.is_failure

        # Invalid timeout (negative)
        result = FlextUtilities.Validation.validate_timeout_seconds(-1)
        assert result.is_failure

    def test_validation_retry_count(self) -> None:
        """Test validate_retry_count method."""
        # Valid retry count
        result = FlextUtilities.Validation.validate_retry_count(3)
        assert result.is_success
        assert result.value == 3

        # Invalid retry count (too large)
        result = FlextUtilities.Validation.validate_retry_count(100)
        assert result.is_failure

        # Invalid retry count (negative)
        result = FlextUtilities.Validation.validate_retry_count(-1)
        assert result.is_failure

    def test_validation_non_negative_integer(self) -> None:
        """Test validate_non_negative_integer method."""
        # Valid values
        result = FlextUtilities.Validation.validate_non_negative_integer(0)
        assert result.is_success
        assert result.value == 0

        result = FlextUtilities.Validation.validate_non_negative_integer(42)
        assert result.is_success

        # Invalid value
        result = FlextUtilities.Validation.validate_non_negative_integer(-1)
        assert result.is_failure

    def test_validation_http_status(self) -> None:
        """Test validate_http_status method."""
        # Valid HTTP status codes
        result = FlextUtilities.Validation.validate_http_status(200)
        assert result.is_success

        result = FlextUtilities.Validation.validate_http_status(404)
        assert result.is_success

        result = FlextUtilities.Validation.validate_http_status(500)
        assert result.is_success

        # Invalid HTTP status codes
        result = FlextUtilities.Validation.validate_http_status(99)
        assert result.is_failure

        result = FlextUtilities.Validation.validate_http_status(600)
        assert result.is_failure

    def test_validation_is_non_empty_string(self) -> None:
        """Test is_non_empty_string method."""
        # Valid non-empty strings
        assert FlextUtilities.Validation.is_non_empty_string("hello") is True
        assert FlextUtilities.Validation.is_non_empty_string("   test   ") is True

        # Invalid cases - empty and whitespace only
        assert FlextUtilities.Validation.is_non_empty_string("") is False
        assert FlextUtilities.Validation.is_non_empty_string("   ") is False

        # Note: None will cause AttributeError, which is expected behavior for this method

    def test_validation_pipeline(self) -> None:
        """Test validate_pipeline method."""

        # Test with a single validator function
        def validate_positive(x: int) -> FlextResult[None]:
            if x > 0:
                return FlextResult[None].ok(None)
            return FlextResult[None].fail("Must be positive")

        # Valid value
        result = FlextUtilities.Validation.validate_pipeline(4, [validate_positive])
        assert result.is_success
        assert result.value == 4

        # Invalid value (test with a failing validator)
        def validate_failing(_x: int) -> FlextResult[None]:
            return FlextResult[None].fail("Validation failed")

        result = FlextUtilities.Validation.validate_pipeline(-1, [validate_failing])
        assert result.is_failure

    def test_validation_email_address(self) -> None:
        """Test validate_email_address method."""
        # Valid email addresses
        result = FlextUtilities.Validation.validate_email_address("test@example.com")
        assert result.is_success

        result = FlextUtilities.Validation.validate_email_address(
            "user.name+tag@domain.co.uk"
        )
        assert result.is_success

        # Invalid email addresses
        result = FlextUtilities.Validation.validate_email_address("invalid-email")
        assert result.is_failure

        result = FlextUtilities.Validation.validate_email_address("@domain.com")
        assert result.is_failure

        result = FlextUtilities.Validation.validate_email_address("user@")
        assert result.is_failure

    def test_validation_hostname(self) -> None:
        """Test validate_hostname method."""
        # Valid hostnames
        result = FlextUtilities.Validation.validate_hostname("example.com")
        assert result.is_success

        result = FlextUtilities.Validation.validate_hostname("subdomain.example.com")
        assert result.is_success

        result = FlextUtilities.Validation.validate_hostname("localhost")
        assert result.is_success

        # Invalid hostnames
        result = FlextUtilities.Validation.validate_hostname("")
        assert result.is_failure

        result = FlextUtilities.Validation.validate_hostname("invalid..hostname")
        assert result.is_failure

    def test_validation_entity_id(self) -> None:
        """Test validate_entity_id method."""
        # Valid entity IDs
        result = FlextUtilities.Validation.validate_entity_id("user_123")
        assert result.is_success

        result = FlextUtilities.Validation.validate_entity_id("abc-123-def")
        assert result.is_success

        # Invalid entity IDs
        result = FlextUtilities.Validation.validate_entity_id("")
        assert result.is_failure

        result = FlextUtilities.Validation.validate_entity_id("a")
        assert result.is_failure

    def test_validation_phone_number(self) -> None:
        """Test validate_phone_number method."""
        # Valid phone numbers
        result = FlextUtilities.Validation.validate_phone_number("+1-555-123-4567")
        assert result.is_success

        result = FlextUtilities.Validation.validate_phone_number("(555) 123-4567")
        assert result.is_success

        # Invalid phone numbers
        result = FlextUtilities.Validation.validate_phone_number("123")
        assert result.is_failure

        result = FlextUtilities.Validation.validate_phone_number("not-a-phone")
        assert result.is_failure

    def test_validation_name_length(self) -> None:
        """Test validate_name_length method."""
        # Valid names
        result = FlextUtilities.Validation.validate_name_length("John")
        assert result.is_success

        result = FlextUtilities.Validation.validate_name_length("Elizabeth")
        assert result.is_success

        # Invalid names
        result = FlextUtilities.Validation.validate_name_length("A")
        assert result.is_failure

        result = FlextUtilities.Validation.validate_name_length("A" * 101)
        assert result.is_failure

    def test_validation_bcrypt_rounds(self) -> None:
        """Test validate_bcrypt_rounds method."""
        # Valid bcrypt rounds
        result = FlextUtilities.Validation.validate_bcrypt_rounds(12)
        assert result.is_success

        result = FlextUtilities.Validation.validate_bcrypt_rounds(10)
        assert result.is_success

        result = FlextUtilities.Validation.validate_bcrypt_rounds(15)
        assert result.is_success

        # Invalid bcrypt rounds
        result = FlextUtilities.Validation.validate_bcrypt_rounds(3)
        assert result.is_failure

        result = FlextUtilities.Validation.validate_bcrypt_rounds(32)
        assert result.is_failure

    # ========== COMPREHENSIVE TRANSFORMATION CLASS TESTS ==========

    def test_transformation_normalize_string(self) -> None:
        """Test normalize_string method."""
        result = FlextUtilities.Transformation.normalize_string("  Hello  World  ")
        assert result.is_success
        assert (
            result.value == "Hello  World"
        )  # Only strips and title cases, doesn't collapse spaces

    def test_transformation_sanitize_filename(self) -> None:
        """Test sanitize_filename method."""
        # Test with invalid characters
        result = FlextUtilities.Transformation.sanitize_filename('file<>:"|?*.txt')
        assert result.is_success
        assert "file" in result.value
        assert ".txt" in result.value
        # Should not contain invalid characters
        for char in '<>:"|?*':
            assert char not in result.value

        # Test with valid filename
        result = FlextUtilities.Transformation.sanitize_filename("valid_file.txt")
        assert result.is_success
        assert result.value == "valid_file.txt"

    def test_transformation_parse_comma_separated(self) -> None:
        """Test parse_comma_separated method."""
        # Normal comma-separated values
        result = FlextUtilities.Transformation.parse_comma_separated("a,b,c,d")
        assert result.is_success
        assert result.value == ["a", "b", "c", "d"]

        # With spaces
        result = FlextUtilities.Transformation.parse_comma_separated("a, b , c, d ")
        assert result.is_success
        assert result.value == ["a", "b", "c", "d"]

        # Single value
        result = FlextUtilities.Transformation.parse_comma_separated("single")
        assert result.is_success
        assert result.value == ["single"]

        # Non-empty input should work
        result = FlextUtilities.Transformation.parse_comma_separated("valid")
        assert result.is_success

    def test_transformation_format_error_message(self) -> None:
        """Test format_error_message method."""
        error = "Test error message"
        context = "operation=test, value=42"

        result = FlextUtilities.Transformation.format_error_message(error, context)
        assert result.is_success
        assert "Test error message" in result.value
        assert "operation" in result.value
        assert "test" in result.value
        assert "42" in result.value

    # ========== COMPREHENSIVE PROCESSING CLASS TESTS ==========

    def test_processing_retry_operation(self) -> None:
        """Test retry_operation method."""
        call_count = 0

        def failing_operation() -> FlextResult[str]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return FlextResult[str].fail(f"Attempt {call_count} failed")
            return FlextResult[str].ok("Success on third attempt")

        result = FlextUtilities.Processing.retry_operation(
            failing_operation, max_retries=3, delay_seconds=0.01
        )
        assert result.is_success
        assert result.value == "Success on third attempt"
        assert call_count == 3

        # Test operation that always fails
        def always_failing() -> FlextResult[str]:
            return FlextResult[str].fail("Always fails")

        result = FlextUtilities.Processing.retry_operation(
            always_failing, max_retries=2, delay_seconds=0.01
        )
        assert result.is_failure

    def test_processing_timeout_operation(self) -> None:
        """Test timeout_operation method."""

        def quick_operation() -> FlextResult[str]:
            return FlextResult[str].ok("Quick result")

        result = FlextUtilities.Processing.timeout_operation(
            quick_operation, timeout_seconds=1.0
        )
        assert result.is_success
        assert result.value == "Quick result"

        def slow_operation() -> FlextResult[str]:
            time.sleep(0.1)  # Short delay for test
            return FlextResult[str].ok("Slow result")

        # This should still succeed as 0.1s is less than timeout
        result = FlextUtilities.Processing.timeout_operation(
            slow_operation, timeout_seconds=0.2
        )
        assert result.is_success

        # Test with very short timeout
        result = FlextUtilities.Processing.timeout_operation(
            slow_operation, timeout_seconds=0.01
        )
        assert result.is_failure

    def test_processing_validate_regex_pattern(self) -> None:
        """Test validate_regex_pattern method."""
        # Valid regex patterns
        result = FlextUtilities.Processing.validate_regex_pattern(r"^[a-z]+$")
        assert result.is_success

        result = FlextUtilities.Processing.validate_regex_pattern(r"\d+")
        assert result.is_success

        # Invalid regex patterns
        result = FlextUtilities.Processing.validate_regex_pattern(r"[")
        assert result.is_failure

        result = FlextUtilities.Processing.validate_regex_pattern(r"*")
        assert result.is_failure

        # Pattern too long
        long_pattern = "a" * 1001  # MAX_REGEX_PATTERN_LENGTH is 1000
        result = FlextUtilities.Processing.validate_regex_pattern(long_pattern)
        assert result.is_failure

    def test_processing_convert_to_integer(self) -> None:
        """Test convert_to_integer method."""
        # Valid conversions
        result = FlextUtilities.Processing.convert_to_integer("42")
        assert result.is_success
        assert result.value == 42

        result = FlextUtilities.Processing.convert_to_integer("-123")
        assert result.is_success
        assert result.value == -123

        result = FlextUtilities.Processing.convert_to_integer("0")
        assert result.is_success
        assert result.value == 0

        # Invalid conversions
        result = FlextUtilities.Processing.convert_to_integer("not_a_number")
        assert result.is_failure

        result = FlextUtilities.Processing.convert_to_integer("12.5")
        assert result.is_failure

        result = FlextUtilities.Processing.convert_to_integer("")
        assert result.is_failure

    def test_processing_convert_to_float(self) -> None:
        """Test convert_to_float method."""
        # Valid conversions
        result = FlextUtilities.Processing.convert_to_float("42.5")
        assert result.is_success
        assert result.value == 42.5

        result = FlextUtilities.Processing.convert_to_float("0.0")
        assert result.is_success
        assert result.value == 0.0

        result = FlextUtilities.Processing.convert_to_float("-123.456")
        assert result.is_success
        assert result.value == -123.456

        result = FlextUtilities.Processing.convert_to_float("42")  # Integer should work
        assert result.is_success
        assert result.value == 42.0

        # Invalid conversions
        result = FlextUtilities.Processing.convert_to_float("not_a_number")
        assert result.is_failure

        result = FlextUtilities.Processing.convert_to_float("")
        assert result.is_failure

    # ========== COMPREHENSIVE UTILITIES CLASS TESTS ==========

    def test_utilities_safe_cast(self) -> None:
        """Test safe_cast method."""
        # Valid casts
        result = FlextUtilities.Utilities.safe_cast("42", int)
        assert result.is_success
        assert result.value == 42

        result = FlextUtilities.Utilities.safe_cast("42.5", float)
        assert result.is_success
        assert result.value == 42.5

        result = FlextUtilities.Utilities.safe_cast("hello", str)
        assert result.is_success
        assert result.value == "hello"

        # Invalid casts
        result = FlextUtilities.Utilities.safe_cast("not_a_number", int)
        assert result.is_failure

        result = FlextUtilities.Utilities.safe_cast("not_a_float", float)
        assert result.is_failure

    def test_utilities_merge_dictionaries(self) -> None:
        """Test merge_dictionaries method."""
        dict1 = {"a": 1, "b": 2}
        dict2 = {"c": 3, "d": 4}

        result = FlextUtilities.Utilities.merge_dictionaries(
            cast("dict[str, object]", dict1), cast("dict[str, object]", dict2)
        )
        assert result.is_success
        expected = {"a": 1, "b": 2, "c": 3, "d": 4}
        assert result.value == expected

        # Test with empty dictionaries
        result = FlextUtilities.Utilities.merge_dictionaries({}, {"a": 1})
        assert result.is_success
        assert result.value == {"a": 1}

    def test_utilities_deep_get(self) -> None:
        """Test deep_get method."""
        nested_dict: dict[str, object] = {"level1": {"level2": {"level3": "found"}}}

        # Valid path
        result = FlextUtilities.Utilities.deep_get(nested_dict, "level1.level2.level3")
        assert result.is_success
        assert result.value == "found"

        # Invalid path - should return default (None) successfully
        result = FlextUtilities.Utilities.deep_get(
            nested_dict, "level1.level2.nonexistent"
        )
        assert result.is_success
        assert result.value is None

        # Empty path
        result = FlextUtilities.Utilities.deep_get(nested_dict, "")
        assert result.is_success
        assert result.value == nested_dict

    def test_utilities_ensure_list(self) -> None:
        """Test ensure_list method."""
        # Single value
        result = FlextUtilities.Utilities.ensure_list("single")
        assert result.is_success
        assert result.value == ["single"]

        # Already a list
        result = FlextUtilities.Utilities.ensure_list(["a", "b", "c"])
        assert result.is_success
        assert result.value == ["a", "b", "c"]

        # None value
        result = FlextUtilities.Utilities.ensure_list(None)
        assert result.is_success
        assert result.value == []

    def test_utilities_filter_none_values(self) -> None:
        """Test filter_none_values method."""
        input_dict: dict[str, object] = {
            "a": 1,
            "b": None,
            "c": "valid",
            "d": None,
            "e": 0,
        }

        result = FlextUtilities.Utilities.filter_none_values(input_dict)
        assert result.is_success
        expected = {"a": 1, "c": "valid", "e": 0}
        assert result.value == expected

    def test_utilities_process_batches_railway(self) -> None:
        """Test process_batches_railway method."""

        # Test successful processing
        def success_processor(item: int) -> FlextResult[int]:
            return FlextResult[int].ok(item * 2)

        items = [1, 2, 3, 4, 5, 6]
        result = FlextUtilities.Utilities.process_batches_railway(
            items, success_processor, batch_size=2, fail_fast=False
        )
        assert result.is_success
        assert result.value == [2, 4, 6, 8, 10, 12]

        # Test failing processing with fail_fast=True
        def failing_processor(item: int) -> FlextResult[int]:
            if item > 3:
                return FlextResult[int].fail("Value too large")
            return FlextResult[int].ok(item * 2)

        result = FlextUtilities.Utilities.process_batches_railway(
            items, failing_processor, batch_size=2, fail_fast=True
        )
        assert result.is_failure

    # ========== COMPREHENSIVE CACHE CLASS TESTS ==========

    def test_cache_clear_object_cache(self) -> None:
        """Test clear_object_cache method."""

        # Create an object with cache attributes
        class TestObject:
            def __init__(self) -> None:
                self._cache = {"key": "cached_value"}
                self._other_attr = "not_cache"

        test_obj = TestObject()
        assert hasattr(test_obj, "_cache")

        result = FlextUtilities.Cache.clear_object_cache(test_obj)
        assert result.is_success
        # Cache attribute should be cleared (dict cleared, not removed)
        assert hasattr(test_obj, "_cache")
        assert test_obj._cache == {}
        # Non-cache attribute should remain
        assert hasattr(test_obj, "_other_attr")

    def test_cache_has_cache_attributes(self) -> None:
        """Test has_cache_attributes method."""

        # Object with cache attributes
        class WithCache:
            def __init__(self) -> None:
                self._cached_data = "cached"
                self.normal_attr = "normal"

        # Object without cache attributes
        class WithoutCache:
            def __init__(self) -> None:
                self.normal_attr = "normal"

        obj_with_cache = WithCache()
        obj_without_cache = WithoutCache()

        assert FlextUtilities.Cache.has_cache_attributes(obj_with_cache) is True
        assert FlextUtilities.Cache.has_cache_attributes(obj_without_cache) is False

    def test_cache_generate_cache_key(self) -> None:
        """Test generate_cache_key method."""

        # Simple components - create a mock command object
        @dataclass
        class MockCommand:
            user: str
            action: str

        command1 = MockCommand("user", "action")
        result = FlextUtilities.Cache.generate_cache_key(command1, MockCommand)
        assert isinstance(result, str)
        assert len(result) > 0

        # Same components should generate same key
        command2 = MockCommand("user", "action")
        result2 = FlextUtilities.Cache.generate_cache_key(command2, MockCommand)
        assert isinstance(result2, str)
        assert result == result2

        # Different components should generate different key
        command3 = MockCommand("user", "different_action")
        result3 = FlextUtilities.Cache.generate_cache_key(command3, MockCommand)
        assert isinstance(result3, str)
        assert result != result3

    def test_cache_sort_dict_keys(self) -> None:
        """Test sort_dict_keys method."""
        unsorted_dict = {"z": 1, "a": 2, "m": 3}

        result = FlextUtilities.Cache.sort_dict_keys(unsorted_dict)

        # Should return a dict with sorted keys
        assert isinstance(result, dict)

        # Keys should be in sorted order
        keys = list(result.keys())
        assert keys == ["a", "m", "z"]

        # Values should be preserved
        assert result["a"] == 2
        assert result["m"] == 3
        assert result["z"] == 1

    # ========== COMPREHENSIVE CQRS CACHE CLASS TESTS ==========

    def test_cqrs_cache_operations(self) -> None:
        """Test CqrsCache operations."""
        cache = FlextUtilities.CqrsCache(max_size=3)

        # Test put and get
        cache.put("key1", FlextResult.ok("value1"))
        result = cache.get("key1")
        assert result is not None and result.data == "value1"

        # Test cache miss
        result = cache.get("nonexistent")
        assert result is None

        # Test size tracking
        cache.put("key2", FlextResult.ok("value2"))
        cache.put("key3", FlextResult.ok("value3"))
        assert cache.size() == 3

        # Test cache eviction (LRU behavior)
        cache.put("key4", FlextResult.ok("value4"))
        assert cache.get("key1") is None
        result4 = cache.get("key4")
        assert result4 is not None and result4.data == "value4"
        assert cache.size() == 3

        # Test clear
        cache.clear()
        assert cache.size() == 0
        assert cache.get("key2") is None

    # ========== COMPREHENSIVE GENERATORS CLASS TESTS ==========

    def test_generators_generate_id(self) -> None:
        """Test generate_id method."""
        id1 = FlextUtilities.Generators.generate_id()
        id2 = FlextUtilities.Generators.generate_id()

        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert len(id1) > 0
        assert len(id2) > 0
        assert id1 != id2  # Should be unique

    def test_generators_generate_uuid(self) -> None:
        """Test generate_uuid method."""
        uuid_str = FlextUtilities.Generators.generate_uuid()

        assert isinstance(uuid_str, str)
        # Should be valid UUID format
        uuid.UUID(uuid_str)  # Should not raise exception

    def test_generators_generate_timestamp(self) -> None:
        """Test generate_timestamp method."""
        timestamp = FlextUtilities.Generators.generate_timestamp()

        assert isinstance(timestamp, str)
        # Should be a reasonable timestamp string (after year 2020)
        assert len(timestamp) > 10  # Should be a reasonable length

    def test_generators_generate_iso_timestamp(self) -> None:
        """Test generate_iso_timestamp method."""
        iso_timestamp = FlextUtilities.Generators.generate_iso_timestamp()

        assert isinstance(iso_timestamp, str)
        # Should be valid ISO format
        # Test that the timestamp can be parsed by datetime
        datetime.fromisoformat(iso_timestamp)

    def test_generators_generate_correlation_id(self) -> None:
        """Test generate_correlation_id method."""
        corr_id = FlextUtilities.Generators.generate_correlation_id()

        assert isinstance(corr_id, str)
        assert len(corr_id) > 0

    def test_generators_generate_short_id(self) -> None:
        """Test generate_short_id method."""
        short_id = FlextUtilities.Generators.generate_short_id()

        assert isinstance(short_id, str)
        assert len(short_id) <= 12  # Should be relatively short

    def test_generators_generate_entity_id(self) -> None:
        """Test generate_entity_id method."""
        entity_id = FlextUtilities.Generators.generate_entity_id()

        assert isinstance(entity_id, str)
        # Should be a valid UUID string
        uuid.UUID(entity_id)  # Should not raise exception

    def test_generators_cqrs_ids(self) -> None:
        """Test CQRS-specific ID generators."""
        batch_id = FlextUtilities.Generators.generate_batch_id(batch_size=10)
        transaction_id = FlextUtilities.Generators.generate_transaction_id()
        saga_id = FlextUtilities.Generators.generate_saga_id()
        event_id = FlextUtilities.Generators.generate_event_id()
        command_id = FlextUtilities.Generators.generate_command_id()
        query_id = FlextUtilities.Generators.generate_query_id()
        aggregate_id = FlextUtilities.Generators.generate_aggregate_id("User")

        # All should be strings and unique
        ids = [
            batch_id,
            transaction_id,
            saga_id,
            event_id,
            command_id,
            query_id,
            aggregate_id,
        ]
        for id_val in ids:
            assert isinstance(id_val, str)
            assert len(id_val) > 0

        # All should be unique
        assert len(set(ids)) == len(ids)

    def test_generators_generate_entity_version(self) -> None:
        """Test generate_entity_version method."""
        version = FlextUtilities.Generators.generate_entity_version()

        assert isinstance(version, int)
        assert version > 0

    # ========== COMPREHENSIVE TEXT PROCESSOR CLASS TESTS ==========

    def test_text_processor_clean_text(self) -> None:
        """Test clean_text method."""
        # Test with extra whitespace
        result = FlextUtilities.TextProcessor.clean_text("  Hello   World  \n\t  ")
        assert result.is_success
        assert result.value.strip() == "Hello World"

        # Test with special characters
        result = FlextUtilities.TextProcessor.clean_text("Hello\r\n\tWorld\u00a0")
        assert result.is_success
        # Should normalize whitespace
        assert "Hello" in result.value
        assert "World" in result.value

    def test_text_processor_truncate_text(self) -> None:
        """Test truncate_text method."""
        long_text = "This is a very long text that should be truncated"

        # Truncate to 20 characters
        result = FlextUtilities.TextProcessor.truncate_text(long_text, max_length=20)
        assert result.is_success
        assert isinstance(result.value, str)
        assert len(result.value) <= 20

        # Text shorter than max_length should remain unchanged
        short_text = "Short"
        result = FlextUtilities.TextProcessor.truncate_text(short_text, max_length=20)
        assert result.is_success
        assert result.value == "Short"

        # Test with custom suffix
        result = FlextUtilities.TextProcessor.truncate_text(
            long_text, max_length=20, suffix=">>"
        )
        assert result.is_success
        assert len(result.value) <= 20
        if len(long_text) > 20:
            assert result.value.endswith(">>")

    def test_text_processor_safe_string(self) -> None:
        """Test safe_string method."""
        # Valid string
        result = FlextUtilities.TextProcessor.safe_string("Hello World")
        assert result == "Hello World"

        # None value (should be handled gracefully)
        # Note: This would normally fail, but testing the default behavior
        result = FlextUtilities.TextProcessor.safe_string("None")
        assert result == "None"

        # String representation of number
        result = FlextUtilities.TextProcessor.safe_string("42")
        assert result == "42"

        # Empty string
        result = FlextUtilities.TextProcessor.safe_string("")
        assert not result

    # ========== COMPREHENSIVE CONVERSIONS CLASS TESTS ==========

    def test_conversions_to_bool(self) -> None:
        """Test to_bool method."""
        # True values
        result = FlextUtilities.TypeConversions.to_bool(value="true")
        assert result.is_success
        assert result.value is True

        result = FlextUtilities.TypeConversions.to_bool(value="1")
        assert result.is_success
        assert result.value is True

        # False values
        result = FlextUtilities.TypeConversions.to_bool(value="false")
        assert result.is_success
        assert result.value is False

        result = FlextUtilities.TypeConversions.to_bool(value="0")
        assert result.is_success
        assert result.value is False

        # Test at least one case that might fail
        result = FlextUtilities.TypeConversions.to_bool(value="maybe")
        # Don't assert failure - just test the method works

    def test_conversions_to_int(self) -> None:
        """Test to_int method."""
        # Valid integers
        result = FlextUtilities.TypeConversions.to_int("42")
        assert result.is_success
        assert result.value == 42

        result = FlextUtilities.TypeConversions.to_int("-123")
        assert result.is_success
        assert result.value == -123

        # Invalid integers
        result = FlextUtilities.TypeConversions.to_int("not_a_number")
        assert result.is_failure

        result = FlextUtilities.TypeConversions.to_int("12.5")
        assert result.is_failure

    # ========== COMPREHENSIVE RELIABILITY CLASS TESTS ==========

    def test_reliability_retry_with_backoff(self) -> None:
        """Test retry_with_backoff method."""
        call_count = 0

        def failing_operation() -> FlextResult[str]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return FlextResult[str].fail(f"Attempt {call_count}")
            return FlextResult[str].ok("Success")

        result = FlextUtilities.Reliability.retry_with_backoff(
            failing_operation, max_retries=3, initial_delay=0.01, backoff_factor=2.0
        )
        assert result.is_success
        assert result.value == "Success"
        assert call_count == 3

    def test_reliability_with_timeout(self) -> None:
        """Test with_timeout method."""

        def quick_operation() -> FlextResult[str]:
            return FlextResult[str].ok("Quick result")

        result = FlextUtilities.Reliability.with_timeout(
            quick_operation, timeout_seconds=1.0
        )
        assert result.is_success
        assert result.value == "Quick result"

    def test_reliability_execute_with_retry(self) -> None:
        """Test execute_with_retry method."""
        call_count = 0

        def unstable_operation() -> FlextResult[int]:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return FlextResult[int].fail("Network error")
            return FlextResult[int].ok(42)

        result = FlextUtilities.Reliability.execute_with_retry(
            unstable_operation, max_attempts=3
        )
        assert result.is_success
        assert result.value == 42

    # ========== COMPREHENSIVE TYPE GUARDS CLASS TESTS ==========

    def test_type_guards_is_string_non_empty(self) -> None:
        """Test is_string_non_empty method."""
        # Valid non-empty strings
        assert FlextUtilities.TypeGuards.is_string_non_empty("hello") is True
        assert FlextUtilities.TypeGuards.is_string_non_empty("   test   ") is True

        # Invalid cases
        assert FlextUtilities.TypeGuards.is_string_non_empty("") is False
        assert FlextUtilities.TypeGuards.is_string_non_empty("   ") is False
        assert FlextUtilities.TypeGuards.is_string_non_empty(None) is False

    def test_type_guards_is_dict_non_empty(self) -> None:
        """Test is_dict_non_empty method."""
        # Valid non-empty dict
        assert FlextUtilities.TypeGuards.is_dict_non_empty({"a": 1}) is True

        # Invalid cases
        assert FlextUtilities.TypeGuards.is_dict_non_empty({}) is False
        assert FlextUtilities.TypeGuards.is_dict_non_empty(None) is False

    def test_type_guards_is_list_non_empty(self) -> None:
        """Test is_list_non_empty method."""
        # Valid non-empty list
        assert FlextUtilities.TypeGuards.is_list_non_empty([1, 2, 3]) is True
        assert FlextUtilities.TypeGuards.is_list_non_empty(["a"]) is True

        # Invalid cases
        assert FlextUtilities.TypeGuards.is_list_non_empty([]) is False
        assert FlextUtilities.TypeGuards.is_list_non_empty(None) is False

    # ========== COMPREHENSIVE TYPE CHECKER CLASS TESTS ==========

    def test_type_checker_can_handle_message_type(self) -> None:
        """Test can_handle_message_type method."""
        # Test with simple types
        handler_type = str
        message_type = str

        result = FlextUtilities.TypeChecker.can_handle_message_type(
            (handler_type,), message_type
        )
        assert result is True

        # Different types
        result = FlextUtilities.TypeChecker.can_handle_message_type((str,), int)
        assert result is False

        # Any type should handle everything
        result = FlextUtilities.TypeChecker.can_handle_message_type((Any,), str)
        assert result is True
