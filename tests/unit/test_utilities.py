"""Comprehensive coverage tests for FlextUtilities - High Impact Coverage.

This module provides extensive test coverage for FlextUtilities to achieve ~100% coverage
by testing all major utility classes and methods with real functionality.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from flext_core import FlextTypes, FlextUtilities
from flext_core.result import FlextResult


class TestFlextUtilitiesComprehensive:
    """Comprehensive test suite for FlextUtilities with high coverage."""

    def test_validation_string_operations(self) -> None:
        """Test all string validation operations comprehensively."""
        # Test validate_string_not_none
        assert FlextUtilities.Validation.validate_string_not_none("test").is_success
        assert FlextUtilities.Validation.validate_string_not_none(None).is_failure

        # Test validate_string_not_empty
        assert FlextUtilities.Validation.validate_string_not_empty("test").is_success
        assert FlextUtilities.Validation.validate_string_not_empty("").is_failure

        # Test validate_string_length
        result = FlextUtilities.Validation.validate_string_length(
            "test",
            min_length=2,
            max_length=10,
        )
        assert result.is_success

        result = FlextUtilities.Validation.validate_string_length(
            "a",
            min_length=2,
            max_length=10,
        )
        assert result.is_failure

        result = FlextUtilities.Validation.validate_string_length(
            "very_long_string",
            min_length=2,
            max_length=10,
        )
        assert result.is_failure

        # Test validate_string_pattern
        result = FlextUtilities.Validation.validate_string_pattern(
            "test123",
            r"^[a-z]+\d+$",
        )
        assert result.is_success

        result = FlextUtilities.Validation.validate_string_pattern("TEST", r"^[a-z]+$")
        assert result.is_failure

        # Test validate_string_pattern with None pattern (should pass)
        result = FlextUtilities.Validation.validate_string_pattern("any_string", None)
        assert result.is_success

        # Test validate_string_pattern with invalid regex pattern (should fail)
        result = FlextUtilities.Validation.validate_string_pattern("test", "[invalid")
        assert result.is_failure

        # Test validate_string (comprehensive validation)
        result = FlextUtilities.Validation.validate_string(
            "valid_string",
            min_length=5,
            max_length=20,
            pattern=r"^[a-z_]+$",
        )
        assert result.is_success

    def test_validation_network_operations(self) -> None:
        """Test network-related validation operations."""
        # Test email validation
        assert FlextUtilities.Validation.validate_email("test@example.com").is_success
        assert FlextUtilities.Validation.validate_email("invalid-email").is_failure

        # Test URL validation
        assert FlextUtilities.Validation.validate_url("https://example.com").is_success
        assert FlextUtilities.Validation.validate_url("not_a_url").is_failure

        # Test port validation
        assert FlextUtilities.Validation.validate_port(8080).is_success
        assert FlextUtilities.Validation.validate_port("8080").is_success
        assert FlextUtilities.Validation.validate_port(99999).is_failure
        assert FlextUtilities.Validation.validate_port(-1).is_failure

        # Test host validation
        assert FlextUtilities.Validation.validate_host("localhost").is_success
        assert FlextUtilities.Validation.validate_host("192.168.1.1").is_success
        assert FlextUtilities.Validation.validate_host("").is_failure

        # Test hostname validation
        assert FlextUtilities.Validation.validate_hostname("example.com").is_success
        assert FlextUtilities.Validation.validate_hostname("localhost").is_success
        assert FlextUtilities.Validation.validate_hostname("").is_failure

        # Test comprehensive email validation
        assert FlextUtilities.Validation.validate_email_address(
            "user@example.com",
        ).is_success
        assert FlextUtilities.Validation.validate_email_address(
            "invalid.email",
        ).is_failure

    def test_validation_numeric_operations(self) -> None:
        """Test numeric validation operations."""
        # Test positive integer validation
        assert FlextUtilities.Validation.validate_positive_integer(5).is_success
        assert FlextUtilities.Validation.validate_positive_integer(0).is_failure
        assert FlextUtilities.Validation.validate_positive_integer(-1).is_failure

        # Test non-negative integer validation
        assert FlextUtilities.Validation.validate_non_negative_integer(0).is_success
        assert FlextUtilities.Validation.validate_non_negative_integer(5).is_success
        assert FlextUtilities.Validation.validate_non_negative_integer(-1).is_failure

        # Test timeout validation
        assert FlextUtilities.Validation.validate_timeout_seconds(30.0).is_success
        assert FlextUtilities.Validation.validate_timeout_seconds(-1.0).is_failure
        assert FlextUtilities.Validation.validate_timeout_seconds(4000.0).is_failure

        # Test retry count validation
        assert FlextUtilities.Validation.validate_retry_count(3).is_success
        assert FlextUtilities.Validation.validate_retry_count(-1).is_failure
        assert FlextUtilities.Validation.validate_retry_count(20).is_failure

        # Test HTTP status validation
        assert FlextUtilities.Validation.validate_http_status(200).is_success
        assert FlextUtilities.Validation.validate_http_status(404).is_success
        assert FlextUtilities.Validation.validate_http_status(99).is_failure
        assert FlextUtilities.Validation.validate_http_status(600).is_failure

    def test_validation_specialized_operations(self) -> None:
        """Test specialized validation operations."""
        # Test environment value validation
        assert FlextUtilities.Validation.validate_environment_value(
            "production",
            ["production", "development"],
        ).is_success
        assert FlextUtilities.Validation.validate_environment_value(
            "invalid",
            ["production", "development"],
        ).is_failure

        # Test log level validation
        assert FlextUtilities.Validation.validate_log_level("INFO").is_success
        assert FlextUtilities.Validation.validate_log_level("DEBUG").is_success
        assert FlextUtilities.Validation.validate_log_level("INVALID").is_failure

        # Test security token validation
        assert FlextUtilities.Validation.validate_security_token(
            "token123456",
        ).is_success
        assert FlextUtilities.Validation.validate_security_token("short").is_failure

        # Test connection string validation
        assert FlextUtilities.Validation.validate_connection_string(
            "host=localhost;port=5432",
        ).is_success
        assert FlextUtilities.Validation.validate_connection_string("").is_failure

        # Test entity ID validation
        assert FlextUtilities.Validation.validate_entity_id("entity_123").is_success
        assert FlextUtilities.Validation.validate_entity_id("").is_failure

        # Test phone number validation
        assert FlextUtilities.Validation.validate_phone_number(
            "+1-555-123-4567",
        ).is_success
        assert FlextUtilities.Validation.validate_phone_number(
            "555-123-4567",
        ).is_success
        assert FlextUtilities.Validation.validate_phone_number("invalid").is_failure

        # Test name length validation
        assert FlextUtilities.Validation.validate_name_length("John Doe").is_success
        assert FlextUtilities.Validation.validate_name_length("").is_failure
        assert FlextUtilities.Validation.validate_name_length("x" * 300).is_failure

        # Test bcrypt rounds validation
        assert FlextUtilities.Validation.validate_bcrypt_rounds(12).is_success
        assert FlextUtilities.Validation.validate_bcrypt_rounds(3).is_failure
        assert FlextUtilities.Validation.validate_bcrypt_rounds(32).is_failure

    def test_validation_path_operations(self) -> None:
        """Test file and directory path validation."""
        # Test directory path validation with secure temp directory
        with tempfile.TemporaryDirectory() as secure_temp_dir:
            result = FlextUtilities.Validation.validate_directory_path(secure_temp_dir)
            assert result.is_success

        result = FlextUtilities.Validation.validate_directory_path("")
        assert result.is_failure

        # Test file path validation with secure temp file
        with tempfile.NamedTemporaryFile(
            encoding="utf-8",
            mode="w",
            delete=False,
        ) as secure_temp_file:
            secure_temp_file.write("test content")
            temp_file_path = secure_temp_file.name

        try:
            result = FlextUtilities.Validation.validate_file_path(temp_file_path)
            assert result.is_success
        finally:
            Path(temp_file_path).unlink()

        result = FlextUtilities.Validation.validate_file_path("")
        assert result.is_failure

        # Test existing file path validation with temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"test content")
            temp_path = tmp.name

        try:
            result = FlextUtilities.Validation.validate_existing_file_path(temp_path)
            assert result.is_success
        finally:
            Path(temp_path).unlink(missing_ok=True)

        # Test non-existing file
        result = FlextUtilities.Validation.validate_existing_file_path(
            "/non/existent/file.txt",
        )
        assert result.is_failure

    def test_validation_helper_functions(self) -> None:
        """Test validation helper functions."""
        # Test is_non_empty_string
        assert FlextUtilities.Validation.is_non_empty_string("test") is True
        assert FlextUtilities.Validation.is_non_empty_string("") is False

        # Test validation pipeline
        def validator1(value: str) -> FlextResult[None]:
            return FlextUtilities.Validation.validate_string_not_empty(value).map(
                lambda _: None,
            )

        def validator2(value: str) -> FlextResult[None]:
            return FlextUtilities.Validation.validate_string_length(
                value,
                min_length=3,
                max_length=20,
            ).map(lambda _: None)

        result = FlextUtilities.Validation.validate_pipeline(
            "test",
            [validator1, validator2],
        )
        assert result.is_success

        result = FlextUtilities.Validation.validate_pipeline(
            "",
            [validator1, validator2],
        )
        assert result.is_failure

        # Test validation with context

        def context_validator(value: str) -> FlextResult[None]:
            # Use context from closure
            if len(value) < 8:
                return FlextResult[None].fail("String too short")
            return FlextResult[None].ok(None)

        result = FlextUtilities.Validation.validate_with_context(
            "short",
            "context_test",
            context_validator,
        )
        assert result.is_failure

        result = FlextUtilities.Validation.validate_with_context(
            "long_enough",
            "context_test",
            context_validator,
        )
        assert result.is_success

    def test_transformation_operations(self) -> None:
        """Test transformation utility operations."""
        # Test string normalization
        result = FlextUtilities.Transformation.normalize_string("  Test String  ")
        assert result.is_success
        assert result.value == "Test String"

        # Test filename sanitization
        result = FlextUtilities.Transformation.sanitize_filename("file<name>.txt")
        assert result.is_success
        assert "<" not in result.value
        assert ">" not in result.value

        # Test comma-separated parsing
        comma_result: FlextResult[FlextTypes.StringList] = (
            FlextUtilities.Transformation.parse_comma_separated("a,b,c")
        )
        assert comma_result.is_success
        assert comma_result.value == ["a", "b", "c"]

        result2 = FlextUtilities.Transformation.parse_comma_separated("a, b , c ")
        assert result2.is_success
        assert result2.value == ["a", "b", "c"]

        # Test error message formatting
        error_result = FlextUtilities.Transformation.format_error_message(
            "Error",
            "Additional context",
        )
        assert error_result.is_success
        assert "Error" in error_result.value
        assert "Additional context" in error_result.value

    def test_processing_operations(self) -> None:
        """Test processing utility operations."""
        # Test regex pattern validation
        pattern_result: FlextResult[re.Pattern[str]] = (
            FlextUtilities.Processing.validate_regex_pattern(r"^[a-zA-Z]+$")
        )
        assert pattern_result.is_success
        assert isinstance(pattern_result.value, re.Pattern)

        pattern_result2 = FlextUtilities.Processing.validate_regex_pattern("[")
        assert pattern_result2.is_failure

        # Test integer conversion
        int_result = FlextUtilities.Processing.convert_to_integer("123")
        assert int_result.is_success
        assert int_result.value == 123

        int_result2 = FlextUtilities.Processing.convert_to_integer(123)
        assert int_result2.is_success
        assert int_result2.value == 123

        int_result3 = FlextUtilities.Processing.convert_to_integer("not_a_number")
        assert int_result3.is_failure

        # Test float conversion
        float_result = FlextUtilities.Processing.convert_to_float("123.45")
        assert float_result.is_success
        assert float_result.value == 123.45

        float_result2 = FlextUtilities.Processing.convert_to_float(123.45)
        assert float_result2.is_success
        assert float_result2.value == 123.45

        float_result3 = FlextUtilities.Processing.convert_to_float("not_a_number")
        assert float_result3.is_failure

    def test_processing_retry_operations(self) -> None:
        """Test retry and reliability operations."""
        # Test retry operation
        attempt_count = 0

        def failing_operation() -> FlextResult[str]:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                return FlextResult[str].fail("Temporary failure")
            return FlextResult[str].ok("Success")

        result = FlextUtilities.Processing.retry_operation(
            failing_operation,
            max_retries=3,
        )
        assert result.is_success
        assert result.value == "Success"
        assert attempt_count == 3

        # Test timeout operation
        def quick_operation() -> FlextResult[str]:
            return FlextResult[str].ok("Quick result")

        result = FlextUtilities.Processing.timeout_operation(
            quick_operation,
            timeout_seconds=1.0,
        )
        assert result.is_success
        assert result.value == "Quick result"

    def test_utilities_operations(self) -> None:
        """Test general utility operations."""
        # Test safe cast
        cast_result = FlextUtilities.Utilities.safe_cast("123", int)
        assert cast_result.is_success
        assert cast_result.value == 123

        cast_result2 = FlextUtilities.Utilities.safe_cast("not_a_number", int)
        assert cast_result2.is_failure

        # Test dictionary merging with no conflicts
        dict1: FlextTypes.Dict = {"a": 1, "b": 2}
        dict2: FlextTypes.Dict = {"c": 4, "d": 5}
        result: FlextResult[FlextTypes.Dict] = (
            FlextUtilities.Utilities.merge_dictionaries(dict1, dict2)
        )
        assert result.is_success
        assert result.value["a"] == 1
        assert result.value["b"] == 2
        assert result.value["c"] == 4
        assert result.value["d"] == 5

        # Test dictionary merging with conflicts (should fail)
        dict3: FlextTypes.Dict = {"a": 1, "b": 2}
        dict4: FlextTypes.Dict = {"b": 3, "c": 4}
        result2: FlextResult[FlextTypes.Dict] = (
            FlextUtilities.Utilities.merge_dictionaries(dict3, dict4)
        )
        assert result2.is_failure

        # Test deep get
        data: FlextTypes.Dict = {"a": {"b": {"c": "value"}}}
        deep_result = FlextUtilities.Utilities.deep_get(data, "a.b.c")
        assert deep_result.is_success
        assert deep_result.value == "value"

        deep_result2 = FlextUtilities.Utilities.deep_get(data, "a.b.d")
        assert (
            deep_result2.is_success
        )  # Returns default value None for non-existent path

        # Test ensure list
        list_result = FlextUtilities.Utilities.ensure_list([1, 2, 3])
        assert list_result.is_success
        assert list_result.value == [1, 2, 3]

        list_result2 = FlextUtilities.Utilities.ensure_list("single_value")
        assert list_result2.is_success
        assert list_result2.value == ["single_value"]

        list_result3 = FlextUtilities.Utilities.ensure_list(None)
        assert list_result3.is_success
        assert list_result3.value == []

        # Test filter none values
        data_with_none: FlextTypes.Dict = {"a": 1, "b": None, "c": 3}
        filter_result = FlextUtilities.Utilities.filter_none_values(data_with_none)
        assert filter_result.is_success
        assert "a" in filter_result.value
        assert "b" not in filter_result.value
        assert "c" in filter_result.value

        # Test batch processing
        items = [1, 2, 3, 4, 5]
        batch_result = FlextUtilities.Utilities.batch_process(
            items,
            batch_size=2,
            processor=lambda x: FlextResult[int].ok(x * 2),
        )
        assert batch_result.is_success
        assert len(batch_result.value) == 5
        assert all(isinstance(x, int) for x in batch_result.value)

    def test_generators_operations(self) -> None:
        """Test ID and data generation operations."""
        # Test basic ID generation
        id1 = FlextUtilities.Generators.generate_id()
        id2 = FlextUtilities.Generators.generate_id()
        assert id1 != id2
        assert len(id1) > 0

        # Test UUID generation
        uuid1 = FlextUtilities.Generators.generate_uuid()
        uuid2 = FlextUtilities.Generators.generate_uuid()
        assert uuid1 != uuid2
        assert len(uuid1) > 0

        # Test timestamp generation
        ts1 = FlextUtilities.Generators.generate_timestamp()
        time.sleep(0.01)
        ts2 = FlextUtilities.Generators.generate_timestamp()
        assert ts1 != ts2

        # Test ISO timestamp generation
        iso_ts = FlextUtilities.Generators.generate_iso_timestamp()
        assert "T" in iso_ts

        # Test correlation ID generation
        corr_id = FlextUtilities.Generators.generate_correlation_id()
        assert len(corr_id) > 0

        # Test short ID generation
        short_id = FlextUtilities.Generators.generate_short_id(length=10)
        assert len(short_id) == 10

        # Test entity ID generation
        entity_id = FlextUtilities.Generators.generate_entity_id()
        assert len(entity_id) > 0

        # Test specific ID types
        batch_id = FlextUtilities.Generators.generate_batch_id(100)
        assert len(batch_id) > 0

        transaction_id = FlextUtilities.Generators.generate_transaction_id()
        assert len(transaction_id) > 0

        saga_id = FlextUtilities.Generators.generate_saga_id()
        assert len(saga_id) > 0

        event_id = FlextUtilities.Generators.generate_event_id()
        assert len(event_id) > 0

        command_id = FlextUtilities.Generators.generate_command_id()
        assert len(command_id) > 0

        query_id = FlextUtilities.Generators.generate_query_id()
        assert len(query_id) > 0

        aggregate_id = FlextUtilities.Generators.generate_aggregate_id("User")
        assert "User" in aggregate_id

        version = FlextUtilities.Generators.generate_entity_version()
        assert isinstance(version, int)
        assert version > 0

        # Test correlation with context
        corr_with_ctx = FlextUtilities.Generators.generate_correlation_id_with_context(
            "test_context",
        )
        assert "test_context" in corr_with_ctx

    def test_text_processor_operations(self) -> None:
        """Test text processing operations."""
        # Test text cleaning
        result = FlextUtilities.TextProcessor.clean_text("  Test\n\r\tText  ")
        assert result.is_success
        assert result.value.strip() == "Test Text"

        # Test text truncation
        result = FlextUtilities.TextProcessor.truncate_text(
            "Long text here",
            max_length=8,
        )
        assert result.is_success
        assert len(result.value) <= 11  # 8 + "..." (3)

        result = FlextUtilities.TextProcessor.truncate_text("Short", max_length=10)
        assert result.is_success
        assert result.value == "Short"

        # Test safe string
        result_str = FlextUtilities.TextProcessor.safe_string("test")
        assert result_str == "test"

        result_str = FlextUtilities.TextProcessor.safe_string("")
        assert not result_str

        result_str = FlextUtilities.TextProcessor.safe_string("", default="default")
        assert result_str == "default"

    def test_conversions_operations(self) -> None:
        """Test type conversion operations."""
        # Test boolean conversion
        assert FlextUtilities.TypeConversions.to_bool(value=True).is_success
        assert FlextUtilities.TypeConversions.to_bool(value=True).value is True

        assert FlextUtilities.TypeConversions.to_bool(value="true").is_success
        assert FlextUtilities.TypeConversions.to_bool(value="true").value is True

        assert FlextUtilities.TypeConversions.to_bool(value="false").is_success
        assert FlextUtilities.TypeConversions.to_bool(value="false").value is False

        assert FlextUtilities.TypeConversions.to_bool(value=1).is_success
        assert FlextUtilities.TypeConversions.to_bool(value=1).value is True

        assert FlextUtilities.TypeConversions.to_bool(value=0).is_success
        assert FlextUtilities.TypeConversions.to_bool(value=0).value is False

        assert FlextUtilities.TypeConversions.to_bool(value=None).is_success
        assert FlextUtilities.TypeConversions.to_bool(value=None).value is False

        assert FlextUtilities.TypeConversions.to_bool(value="invalid").is_failure

        # Test integer conversion
        assert FlextUtilities.TypeConversions.to_int("123").is_success
        assert FlextUtilities.TypeConversions.to_int("123").value == 123

        assert FlextUtilities.TypeConversions.to_int(123.0).is_success
        assert FlextUtilities.TypeConversions.to_int(123.0).value == 123

        assert FlextUtilities.TypeConversions.to_int(None).is_failure
        assert FlextUtilities.TypeConversions.to_int("not_a_number").is_failure

    def test_type_guards_operations(self) -> None:
        """Test type guard operations."""
        # Test string non-empty check
        assert FlextUtilities.TypeGuards.is_string_non_empty("test") is True
        assert FlextUtilities.TypeGuards.is_string_non_empty("") is False
        assert FlextUtilities.TypeGuards.is_string_non_empty(None) is False
        assert FlextUtilities.TypeGuards.is_string_non_empty(123) is False

        # Test dict non-empty check
        assert FlextUtilities.TypeGuards.is_dict_non_empty({"a": 1}) is True
        assert FlextUtilities.TypeGuards.is_dict_non_empty({}) is False
        assert FlextUtilities.TypeGuards.is_dict_non_empty(None) is False
        assert FlextUtilities.TypeGuards.is_dict_non_empty("not_dict") is False

        # Test list non-empty check
        assert FlextUtilities.TypeGuards.is_list_non_empty([1, 2, 3]) is True
        assert FlextUtilities.TypeGuards.is_list_non_empty([]) is False
        assert FlextUtilities.TypeGuards.is_list_non_empty(None) is False
        assert FlextUtilities.TypeGuards.is_list_non_empty("not_list") is False

    def test_cache_operations(self) -> None:
        """Test cache utility operations."""

        # Create a simple object to test cache operations
        class TestObj:
            def __init__(self) -> None:
                self._cache: FlextTypes.Dict = {}

        test_obj = TestObj()

        # Test cache clearing
        result = FlextUtilities.Cache.clear_object_cache(test_obj)
        assert result.is_success

        # Test cache attribute detection
        has_cache = FlextUtilities.Cache.has_cache_attributes(test_obj)
        assert has_cache is True

        simple_obj = object()
        has_cache = FlextUtilities.Cache.has_cache_attributes(simple_obj)
        assert has_cache is False

        # Test cache key generation
        cache_key = FlextUtilities.Cache.generate_cache_key("test_command", str)
        assert len(cache_key) > 0
        assert isinstance(cache_key, str)

        # Test sorting utilities
        sort_key = FlextUtilities.Cache.sort_key({"b": 2, "a": 1})
        assert isinstance(sort_key, str)

        normalized = FlextUtilities.Cache.normalize_component({"b": 2, "a": 1})
        assert isinstance(normalized, (dict, str))

        sorted_dict = FlextUtilities.Cache.sort_dict_keys({"b": 2, "a": 1})
        assert isinstance(sorted_dict, dict)

    def test_cqrs_cache_operations(self) -> None:
        """Test CQRS-specific cache operations."""
        cache = FlextUtilities.CqrsCache(max_size=3)

        # Test initial state
        assert cache.size() == 0

        # Test putting and getting results
        success_result: FlextResult[object] = FlextResult[object].ok("test_value")
        cache.put("test_key", success_result)
        assert cache.size() == 1

        retrieved = cache.get("test_key")
        assert retrieved is not None
        assert retrieved.is_success
        assert retrieved.value == "test_value"

        # Test getting non-existent key
        non_existent = cache.get("non_existent")
        assert non_existent is None

        # Test cache size limit
        cache.put("key1", FlextResult[object].ok("value1"))
        cache.put("key2", FlextResult[object].ok("value2"))
        cache.put("key3", FlextResult[object].ok("value3"))
        assert cache.size() == 3

        # Adding one more should not increase size (eviction)
        cache.put("key4", FlextResult[object].ok("value4"))
        assert cache.size() == 3

        # Test cache clearing
        cache.clear()
        assert cache.size() == 0

    def test_reliability_operations(self) -> None:
        """Test reliability utility operations."""
        # Test retry with backoff
        attempt_count = 0

        def flaky_operation() -> FlextResult[str]:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                return FlextResult[str].fail("Temporary failure")
            return FlextResult[str].ok("Success after retry")

        result = FlextUtilities.Reliability.retry_with_backoff(
            operation=flaky_operation,
            max_retries=3,
            initial_delay=0.01,
            backoff_factor=2.0,
        )
        assert result.is_success
        assert result.value == "Success after retry"

        # Test timeout wrapper
        def quick_operation() -> FlextResult[str]:
            return FlextResult[str].ok("Quick result")

        result = FlextUtilities.Reliability.with_timeout(
            quick_operation,
            timeout_seconds=1.0,
        )
        assert result.is_success
        assert result.value == "Quick result"

        # Test execute with retry
        attempt_count = 0

        def retry_operation() -> FlextResult[str]:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                return FlextResult[str].fail("Retry needed")
            return FlextResult[str].ok("Retry successful")

        result = FlextUtilities.Reliability.execute_with_retry(
            retry_operation,
            max_attempts=3,
        )
        assert result.is_success
        assert result.value == "Retry successful"

    def test_type_checker_operations(self) -> None:
        """Test type checking operations."""
        # Test message type compatibility
        can_handle = FlextUtilities.TypeChecker.can_handle_message_type((str,), str)
        assert can_handle is True

        can_handle = FlextUtilities.TypeChecker.can_handle_message_type((str,), int)
        assert can_handle is False

    def test_message_validator_operations(self) -> None:
        """Test message validation operations."""

        @dataclass
        class TestCommand:
            name: str

        @dataclass
        class TestQuery:
            id: str

        # Test command validation
        command = TestCommand("test_command")
        result = FlextUtilities.MessageValidator.validate_command(command)
        assert result.is_success

        # Test query validation
        query = TestQuery("test_query")
        result = FlextUtilities.MessageValidator.validate_query(query)
        assert result.is_success

        # Test generic message validation
        result = FlextUtilities.MessageValidator.validate_message(
            command,
            operation="command",
        )
        assert result.is_success

        # Test message payload building
        payload = FlextUtilities.MessageValidator.build_serializable_message_payload(
            command,
        )
        assert isinstance(payload, dict)
        assert "name" in payload

    def test_composition_operations(self) -> None:
        """Test composition utility operations."""

        # Test pipeline composition
        def step1(value: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"{value}_step1")

        def step2(value: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"{value}_step2")

        result = FlextUtilities.Composition.compose_pipeline("initial", [step1, step2])
        assert result.is_success
        assert result.value == "initial_step1_step2"

        # Test parallel composition
        def operation_a(value: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"{value}_a")

        def operation_b(value: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"{value}_b")

        parallel_func = FlextUtilities.Composition.compose_parallel(
            operation_a,
            operation_b,
        )
        parallel_result = parallel_func("input")
        assert parallel_result.is_success
        assert len(parallel_result.value) == 2
        assert "input_a" in parallel_result.value
        assert "input_b" in parallel_result.value

        # Test conditional composition
        def condition(value: str) -> bool:
            return len(value) > 5

        def true_operation(value: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"{value}_long")

        def false_operation(value: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"{value}_short")

        conditional_func = FlextUtilities.Composition.compose_conditional(
            condition,
            true_operation,
            false_operation,
        )
        result = conditional_func("short")
        assert result.is_success
        assert result.value == "short_short"

        result = conditional_func("long_input")
        assert result.is_success
        assert result.value == "long_input_long"

    @pytest.mark.skip(
        reason="TableConversion class does not exist - needs implementation"
    )
    def test_conversion_operations(self) -> None:
        """Test data conversion operations."""
        # Test data normalization for table
        data: FlextTypes.List = [
            {"name": "John", "age": 30, "city": "NYC"},
            {"name": "Jane", "age": 25, "city": "LA"},
        ]

        result = FlextUtilities.TableConversion.normalize_data_for_table(data)
        assert result.is_success
        normalized_data = result.value
        assert isinstance(normalized_data, list)
        assert len(normalized_data) == 2
        assert all(isinstance(row, dict) for row in normalized_data)
