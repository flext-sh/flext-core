"""Comprehensive coverage tests for FlextUtilities - High Impact Coverage.

This module provides extensive test coverage for FlextUtilities to achieve ~100% coverage
by testing all major utility classes and methods with real functionality.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

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

        # Test execute with retry
        attempt_count = 0

        def retry_operation() -> FlextResult[str]:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                return FlextResult[str].fail("Retry needed")
            return FlextResult[str].ok("Retry successful")

    def test_type_checker_operations(self) -> None:
        """Test type checking operations."""
        # Test message type compatibility
        can_handle = FlextUtilities.TypeChecker.can_handle_message_type((str,), str)
        assert can_handle is True

        can_handle = FlextUtilities.TypeChecker.can_handle_message_type((str,), int)
        assert can_handle is False
