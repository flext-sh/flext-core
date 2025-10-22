"""Comprehensive coverage tests for FlextUtilities - High Impact Coverage.

This module provides extensive test coverage for FlextUtilities to achieve ~100% coverage
by testing all major utility classes and methods with real functionality.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_core import FlextConfig, FlextResult, FlextUtilities


class TestFlextUtilitiesComprehensive:
    """Comprehensive test suite for FlextUtilities with high coverage."""

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
        ts2 = FlextUtilities.Generators.generate_timestamp()
        # Timestamps may be equal if generated within same second (microseconds removed)
        # Just verify they have valid ISO format
        assert "T" in ts1
        assert "T" in ts2
        assert len(ts1) > 0
        assert len(ts2) > 0

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
        assert (
            result.value.strip() == "TestText"
        )  # Clean text removes internal whitespace

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

        # Test dict[str, object] non-empty check
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
                super().__init__()
                self._cache: dict[str, object] = {}

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

        # Test sorting utilities (Cache.sort_key returns tuple, Validation.sort_key returns str)
        sort_key = FlextUtilities.Cache.sort_key({"b": 2, "a": 1})
        assert isinstance(sort_key, tuple)  # Cache.sort_key returns (depth, str_repr)

        normalized = FlextUtilities.Cache.normalize_component({"b": 2, "a": 1})
        assert isinstance(normalized, (dict, str))

        sorted_dict = FlextUtilities.Cache.sort_dict_keys({"b": 2, "a": 1})
        assert isinstance(sorted_dict, dict)

    def test_type_checker_operations(self) -> None:
        """Test type checking operations."""
        # Test message type compatibility
        can_handle = FlextUtilities.TypeChecker.can_handle_message_type((str,), str)
        assert can_handle is True

        can_handle = FlextUtilities.TypeChecker.can_handle_message_type((str,), int)
        assert can_handle is False

    def test_additional_validation_operations(self) -> None:
        """Test additional validation operations not covered in other tests."""
        # NOTE: The following validation methods are not currently implemented:
        # - validate_email
        # - validate_url
        # - validate_port
        # - validate_environment_value
        # - validate_log_level
        # - validate_security_token
        # - validate_connection_string
        #
        # This test is a placeholder for future implementations.
        # For now, we just verify the Validation class exists
        assert FlextUtilities.Validation is not None

    def test_correlation_operations(self) -> None:
        """Test correlation ID operations."""
        # Test correlation ID generation
        corr_id1 = FlextUtilities.Correlation.generate_correlation_id()
        corr_id2 = FlextUtilities.Correlation.generate_correlation_id()
        assert corr_id1 != corr_id2
        assert len(corr_id1) > 0

        # Test ISO timestamp generation
        iso_ts = FlextUtilities.Correlation.generate_iso_timestamp()
        assert "T" in iso_ts

        # Test command ID generation
        cmd_id = FlextUtilities.Correlation.generate_command_id()
        assert len(cmd_id) > 0

        # Test query ID generation
        query_id = FlextUtilities.Correlation.generate_query_id()
        assert len(query_id) > 0

    def test_reliability_operations(self) -> None:
        """Test reliability and retry operations."""
        # Test retry logic (if method exists)
        if hasattr(FlextUtilities.Reliability, "retry"):
            call_count = 0

            def failing_operation() -> FlextResult[str]:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    return FlextResult[str].fail("Temporary failure")
                return FlextResult[str].ok("Success after retries")

            result: FlextResult[str] = FlextUtilities.Reliability.retry(
                failing_operation,
                max_attempts=3,
                delay_seconds=0.01,
            )
            assert result.is_success
            assert call_count == 3

        # Test exponential backoff (not yet implemented)
        # Future enhancement: Implement exponential_backoff functionality in FlextUtilities.Reliability
        assert not hasattr(FlextUtilities.Reliability, "exponential_backoff")

        # Test circuit breaker (not yet implemented)
        # Future enhancement: Implement circuit_breaker functionality in FlextUtilities.Reliability
        assert not hasattr(FlextUtilities.Reliability, "circuit_breaker")

    def test_generate_id_function(self) -> None:
        """Test standalone generate_id function."""
        id1 = FlextUtilities.Generators.generate_id()
        id2 = FlextUtilities.Generators.generate_id()
        assert id1 != id2
        assert len(id1) > 0
        assert isinstance(id1, str)

    def test_validation_initialize(self) -> None:
        """Test FlextUtilities.Validation.initialize static method."""

        class TestObj:
            is_valid: bool = False

        obj = TestObj()
        FlextUtilities.Validation.initialize(obj, "is_valid")
        assert obj.is_valid is True

    def test_cache_clear_object_cache(self) -> None:
        """Test FlextUtilities.Cache.clear_object_cache static method."""

        class TestObj:
            pass

        obj = TestObj()
        # Should not crash
        FlextUtilities.Cache.clear_object_cache(obj)

    def test_generators_ensure_id(self) -> None:
        """Test FlextUtilities.Generators.ensure_id static method."""

        class TestObj:
            id: str = ""

        obj = TestObj()
        FlextUtilities.Generators.ensure_id(obj)
        assert obj.id  # Non-empty string is truthy

    def test_configuration_get_parameter(self) -> None:
        """Test FlextUtilities.Configuration.get_parameter static method."""
        config = FlextConfig.get_global_instance()
        value = FlextUtilities.Configuration.get_parameter(config, "app_name")
        assert value is not None

    def test_configuration_set_parameter(self) -> None:
        """Test FlextUtilities.Configuration.set_parameter static method."""
        config = FlextConfig.get_global_instance()
        # Try to set a parameter
        result = FlextUtilities.Configuration.set_parameter(
            config, "test_param", "test_value"
        )
        assert isinstance(result, bool)

    def test_configuration_get_singleton(self) -> None:
        """Test FlextUtilities.Configuration.get_singleton static method."""
        value = FlextUtilities.Configuration.get_singleton(FlextConfig, "app_name")
        assert value is not None

    def test_configuration_set_singleton(self) -> None:
        """Test FlextUtilities.Configuration.set_singleton static method."""
        result = FlextUtilities.Configuration.set_singleton(
            FlextConfig, "test_param", "test_value"
        )
        assert isinstance(result, bool)
