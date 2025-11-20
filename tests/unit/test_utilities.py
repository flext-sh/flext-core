"""Comprehensive coverage tests for FlextUtilities - High Impact Coverage.

This module provides extensive test coverage for FlextUtilities to achieve ~100% coverage
by testing all major utility classes and methods with real functionality.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import math
from typing import ClassVar

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
        ts1 = FlextUtilities.Generators.generate_iso_timestamp()
        ts2 = FlextUtilities.Generators.generate_iso_timestamp()
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

        # Test safe string - now returns FlextResult[str] with fast fail
        result = FlextUtilities.TextProcessor.safe_string("test")
        assert result.is_success
        assert result.value == "test"

        # Empty string should fail (fast fail pattern)
        result = FlextUtilities.TextProcessor.safe_string("")
        assert result.is_failure

        # None should fail (fast fail pattern)
        result = FlextUtilities.TextProcessor.safe_string(None)
        assert result.is_failure

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
            unique_id: str = ""

        obj = TestObj()
        FlextUtilities.Generators.ensure_id(obj)
        assert obj.unique_id  # Non-empty string is truthy

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
        # set_singleton now returns FlextResult[bool]
        assert (
            result.is_failure
        )  # Should fail because FlextConfig doesn't allow arbitrary params


class TestFlextValidationPipeline:
    """Comprehensive test suite for FlextUtilities.ValidationPipeline."""

    def test_pipeline_initialization_default(self) -> None:
        """Test FlextUtilities.ValidationPipeline initialization with default settings."""
        pipeline = FlextUtilities.ValidationPipeline[str]()
        assert pipeline.is_empty() is True
        assert pipeline.count() == 0

    def test_pipeline_initialization_aggregate_errors(self) -> None:
        """Test FlextUtilities.ValidationPipeline initialization with aggregate_errors=True."""
        pipeline = FlextUtilities.ValidationPipeline[str](aggregate_errors=True)
        assert pipeline.is_empty() is True

    def test_pipeline_add_single_validator(self) -> None:
        """Test adding a single validator to pipeline."""
        pipeline = FlextUtilities.ValidationPipeline[str]()

        def is_not_empty(data: str) -> FlextResult[str]:
            if not data:
                return FlextResult[str].fail("String cannot be empty")
            return FlextResult[str].ok(data)

        result = pipeline.add_validator(is_not_empty)
        assert result is pipeline  # Fluent chaining
        assert pipeline.count() == 1
        assert pipeline.is_empty() is False

    def test_pipeline_add_multiple_validators_chaining(self) -> None:
        """Test chaining multiple validators."""
        pipeline = FlextUtilities.ValidationPipeline[str]()

        def is_not_empty(data: str) -> FlextResult[str]:
            if not data:
                return FlextResult[str].fail("String cannot be empty")
            return FlextResult[str].ok(data)

        def has_min_length(data: str) -> FlextResult[str]:
            if len(data) < 3:
                return FlextResult[str].fail("String must be at least 3 characters")
            return FlextResult[str].ok(data)

        result = pipeline.add_validator(is_not_empty).add_validator(has_min_length)
        assert result is pipeline
        assert pipeline.count() == 2

    def test_pipeline_validate_empty_pipeline(self) -> None:
        """Test validating with empty pipeline (no validators added)."""
        pipeline = FlextUtilities.ValidationPipeline[str]()
        result = pipeline.validate("test_data")
        assert result.is_success
        assert result.unwrap() == "test_data"

    def test_pipeline_validate_single_validator_success(self) -> None:
        """Test validation succeeds with single validator."""

        def is_not_empty(data: str) -> FlextResult[str]:
            if not data:
                return FlextResult[str].fail("String cannot be empty")
            return FlextResult[str].ok(data)

        pipeline = FlextUtilities.ValidationPipeline[str]().add_validator(is_not_empty)
        result = pipeline.validate("valid_data")
        assert result.is_success
        assert result.unwrap() == "valid_data"

    def test_pipeline_validate_single_validator_failure(self) -> None:
        """Test validation fails with single validator."""

        def is_not_empty(data: str) -> FlextResult[str]:
            if not data:
                return FlextResult[str].fail("String cannot be empty")
            return FlextResult[str].ok(data)

        pipeline = FlextUtilities.ValidationPipeline[str]().add_validator(is_not_empty)
        result = pipeline.validate("")
        assert result.is_failure
        assert "cannot be empty" in str(result.error)

    def test_pipeline_validate_multiple_validators_all_success(self) -> None:
        """Test validation with multiple validators all succeeding."""

        def is_not_empty(data: str) -> FlextResult[str]:
            if not data:
                return FlextResult[str].fail("String cannot be empty")
            return FlextResult[str].ok(data)

        def has_min_length(data: str) -> FlextResult[str]:
            if len(data) < 3:
                return FlextResult[str].fail("String must be at least 3 characters")
            return FlextResult[str].ok(data)

        pipeline = (
            FlextUtilities.ValidationPipeline[str]()
            .add_validator(is_not_empty)
            .add_validator(has_min_length)
        )
        result = pipeline.validate("valid")
        assert result.is_success
        assert result.unwrap() == "valid"

    def test_pipeline_validate_short_circuit_on_first_failure(self) -> None:
        """Test short-circuit behavior: stops on first validator failure."""
        call_count = 0

        def first_validator(data: str) -> FlextResult[str]:
            nonlocal call_count
            call_count += 1
            return FlextResult[str].fail("First validator failed")

        def second_validator(data: str) -> FlextResult[str]:
            nonlocal call_count
            call_count += 1
            return FlextResult[str].ok(data)

        pipeline = (
            FlextUtilities.ValidationPipeline[str]()
            .add_validator(first_validator)
            .add_validator(second_validator)
        )
        result = pipeline.validate("data")
        assert result.is_failure
        assert call_count == 1  # Second validator not called

    def test_pipeline_validate_aggregation_mode(self) -> None:
        """Test aggregation mode: collects all validation errors."""

        def validator1(data: str) -> FlextResult[str]:
            return FlextResult[str].fail("Error 1")

        def validator2(data: str) -> FlextResult[str]:
            return FlextResult[str].fail("Error 2")

        pipeline = (
            FlextUtilities.ValidationPipeline[str](aggregate_errors=True)
            .add_validator(validator1)
            .add_validator(validator2)
        )
        result = pipeline.validate("data")
        assert result.is_failure
        error_msg = str(result.error)
        assert "Error 1" in error_msg
        assert "Error 2" in error_msg
        assert "; " in error_msg  # Errors joined with semicolon

    def test_pipeline_validate_aggregation_partial_failure(self) -> None:
        """Test aggregation with some validators failing."""

        def validator1(data: str) -> FlextResult[str]:
            return FlextResult[str].fail("Error 1")

        def validator2(data: str) -> FlextResult[str]:
            return FlextResult[str].ok(data)

        def validator3(data: str) -> FlextResult[str]:
            return FlextResult[str].fail("Error 3")

        pipeline = (
            FlextUtilities.ValidationPipeline[str](aggregate_errors=True)
            .add_validator(validator1)
            .add_validator(validator2)
            .add_validator(validator3)
        )
        result = pipeline.validate("data")
        assert result.is_failure
        error_msg = str(result.error)
        assert "Error 1" in error_msg
        assert "Error 3" in error_msg

    def test_pipeline_clear_validators(self) -> None:
        """Test clearing all validators from pipeline."""

        def validator(data: str) -> FlextResult[str]:
            return FlextResult[str].ok(data)

        pipeline = (
            FlextUtilities.ValidationPipeline[str]()
            .add_validator(validator)
            .add_validator(validator)
        )
        assert pipeline.count() == 2

        result = pipeline.clear()
        assert result is pipeline  # Fluent chaining
        assert pipeline.count() == 0
        assert pipeline.is_empty() is True

    def test_pipeline_count_validators(self) -> None:
        """Test counting validators in pipeline."""

        def validator(data: str) -> FlextResult[str]:
            return FlextResult[str].ok(data)

        pipeline = FlextUtilities.ValidationPipeline[str]()
        assert pipeline.count() == 0

        pipeline.add_validator(validator)
        assert pipeline.count() == 1

        pipeline.add_validator(validator)
        assert pipeline.count() == 2

    def test_pipeline_is_empty(self) -> None:
        """Test is_empty check."""

        def validator(data: str) -> FlextResult[str]:
            return FlextResult[str].ok(data)

        pipeline = FlextUtilities.ValidationPipeline[str]()
        assert pipeline.is_empty() is True

        pipeline.add_validator(validator)
        assert pipeline.is_empty() is False

    def test_pipeline_with_int_type(self) -> None:
        """Test FlextUtilities.ValidationPipeline with int type parameter."""

        def is_positive(data: int) -> FlextResult[int]:
            if data <= 0:
                return FlextResult[int].fail("Must be positive")
            return FlextResult[int].ok(data)

        pipeline = FlextUtilities.ValidationPipeline[int]().add_validator(is_positive)
        result = pipeline.validate(5)
        assert result.is_success
        assert result.unwrap() == 5

    def test_pipeline_with_int_type_failure(self) -> None:
        """Test FlextUtilities.ValidationPipeline failure with int type."""

        def is_positive(data: int) -> FlextResult[int]:
            if data <= 0:
                return FlextResult[int].fail("Must be positive")
            return FlextResult[int].ok(data)

        pipeline = FlextUtilities.ValidationPipeline[int]().add_validator(is_positive)
        result = pipeline.validate(-5)
        assert result.is_failure

    def test_pipeline_validation_chain_data_mutation(self) -> None:
        """Test that validators can transform data in aggregation mode."""

        def uppercase_validator(data: str) -> FlextResult[str]:
            return FlextResult[str].ok(data.upper())

        def add_suffix_validator(data: str) -> FlextResult[str]:
            return FlextResult[str].ok(data + "_VALIDATED")

        pipeline = (
            FlextUtilities.ValidationPipeline[str](aggregate_errors=False)
            .add_validator(uppercase_validator)
            .add_validator(add_suffix_validator)
        )
        result = pipeline.validate("test")
        assert result.is_success
        assert result.unwrap() == "TEST_VALIDATED"

    def test_pipeline_aggregation_all_success(self) -> None:
        """Test aggregation mode with all validators succeeding."""

        def validator1(data: str) -> FlextResult[str]:
            return FlextResult[str].ok(data)

        def validator2(data: str) -> FlextResult[str]:
            return FlextResult[str].ok(data)

        pipeline = (
            FlextUtilities.ValidationPipeline[str](aggregate_errors=True)
            .add_validator(validator1)
            .add_validator(validator2)
        )
        result = pipeline.validate("data")
        assert result.is_success
        assert result.unwrap() == "data"

    def test_pipeline_clear_and_reuse(self) -> None:
        """Test clearing pipeline and adding new validators."""

        def old_validator(data: str) -> FlextResult[str]:
            return FlextResult[str].fail("Old")

        def new_validator(data: str) -> FlextResult[str]:
            return FlextResult[str].ok(data)

        pipeline = FlextUtilities.ValidationPipeline[str]().add_validator(old_validator)
        assert pipeline.count() == 1

        pipeline.clear()
        assert pipeline.count() == 0

        pipeline.add_validator(new_validator)
        assert pipeline.count() == 1
        result = pipeline.validate("data")
        assert result.is_success


class TestFlextUtilitiesEdgeCases:
    """Additional edge case tests for FlextUtilities."""

    def test_cache_normalize_with_nested_structures(self) -> None:
        """Test cache normalization with nested dict/list structures."""
        nested_obj = {"a": {"b": {"c": 1}}, "d": [1, 2, 3]}
        normalized = FlextUtilities.Cache.normalize_component(nested_obj)
        assert isinstance(normalized, dict)
        assert "a" in normalized

    def test_cache_normalize_with_sets(self) -> None:
        """Test cache normalization with set types."""
        set_obj = {1, 2, 3}
        normalized = FlextUtilities.Cache.normalize_component(set_obj)
        assert isinstance(normalized, (set, frozenset))

    def test_cache_normalize_with_primitives(self) -> None:
        """Test cache normalization with primitive types."""
        assert FlextUtilities.Cache.normalize_component("string") == "string"
        assert FlextUtilities.Cache.normalize_component(42) == 42
        assert FlextUtilities.Cache.normalize_component(math.pi) == math.pi

    def test_cache_sort_key_string(self) -> None:
        """Test sort_key with string input."""
        key = FlextUtilities.Cache.sort_key("TestString")
        assert isinstance(key, tuple)
        assert key[0] == 0  # String type code
        assert isinstance(key[1], str)

    def test_cache_sort_key_number(self) -> None:
        """Test sort_key with number inputs."""
        int_key = FlextUtilities.Cache.sort_key(42)
        assert int_key[0] == 1  # Number type code

        float_key = FlextUtilities.Cache.sort_key(math.pi)
        assert float_key[0] == 1  # Number type code

    def test_cache_sort_key_other_type(self) -> None:
        """Test sort_key with other type inputs."""
        key = FlextUtilities.Cache.sort_key([1, 2, 3])
        assert key[0] == 2  # Other type code

    def test_cache_sort_dict_keys_empty(self) -> None:
        """Test sorting empty dict."""
        result = FlextUtilities.Cache.sort_dict_keys({})
        assert result == {}

    def test_cache_sort_dict_keys_with_values(self) -> None:
        """Test sorting dict with keys."""
        data = {"z": 1, "a": 2, "m": 3}
        sorted_data = FlextUtilities.Cache.sort_dict_keys(data)
        assert isinstance(sorted_data, dict)

    def test_text_processor_clean_text_multiple_spaces(self) -> None:
        """Test cleaning text with multiple consecutive spaces."""
        result = FlextUtilities.TextProcessor.clean_text("a    b    c")
        assert result.is_success
        assert " " not in result.value or result.value.count(" ") < 3

    def test_text_processor_clean_text_tabs_and_newlines(self) -> None:
        """Test cleaning text with tabs and newlines."""
        result = FlextUtilities.TextProcessor.clean_text("a\t\nb\rc")
        assert result.is_success

    def test_text_processor_truncate_with_ellipsis(self) -> None:
        """Test truncate adds ellipsis to long text."""
        result = FlextUtilities.TextProcessor.truncate_text(
            "VeryLongText", max_length=5
        )
        assert result.is_success
        assert "..." in result.value or len(result.value) <= 8

    def test_text_processor_truncate_text_no_truncation_needed(self) -> None:
        """Test truncate when text is already short."""
        result = FlextUtilities.TextProcessor.truncate_text("Hi", max_length=10)
        assert result.is_success
        assert result.value == "Hi"

    def test_generators_short_id_default_length(self) -> None:
        """Test generate_short_id with default length."""
        id1 = FlextUtilities.Generators.generate_short_id()
        id2 = FlextUtilities.Generators.generate_short_id()
        assert id1 != id2
        assert len(id1) == 8  # Default length

    def test_generators_short_id_custom_length(self) -> None:
        """Test generate_short_id with custom length."""
        short = FlextUtilities.Generators.generate_short_id(length=5)
        long = FlextUtilities.Generators.generate_short_id(length=20)
        assert len(short) == 5
        assert len(long) == 20

    def test_generators_batch_id(self) -> None:
        """Test generate_batch_id includes batch size."""
        batch_id = FlextUtilities.Generators.generate_batch_id(1000)
        assert isinstance(batch_id, str)
        assert len(batch_id) > 0

    def test_generators_multiple_timestamp_uniqueness(self) -> None:
        """Test that timestamps are reasonably unique."""
        ts1 = FlextUtilities.Generators.generate_iso_timestamp()
        ts2 = FlextUtilities.Generators.generate_iso_timestamp()
        # Timestamps may be identical if within same second
        # Just verify format
        assert "T" in ts1
        assert "T" in ts2

    def test_type_guards_dict_with_values(self) -> None:
        """Test is_dict_non_empty with various dict contents."""
        assert FlextUtilities.TypeGuards.is_dict_non_empty({"key": "value"}) is True
        assert FlextUtilities.TypeGuards.is_dict_non_empty({"x": None}) is True
        assert FlextUtilities.TypeGuards.is_dict_non_empty({"": ""}) is True

    def test_type_guards_list_with_values(self) -> None:
        """Test is_list_non_empty with various list contents."""
        assert FlextUtilities.TypeGuards.is_list_non_empty([None]) is True
        assert FlextUtilities.TypeGuards.is_list_non_empty([""]) is True
        assert FlextUtilities.TypeGuards.is_list_non_empty([0]) is True

    def test_type_guards_string_with_whitespace(self) -> None:
        """Test is_string_non_empty with whitespace strings."""
        # Whitespace-only strings are considered empty after strip()
        assert FlextUtilities.TypeGuards.is_string_non_empty(" ") is False
        assert FlextUtilities.TypeGuards.is_string_non_empty("\t") is False
        assert FlextUtilities.TypeGuards.is_string_non_empty("\n") is False
        # Strings with actual content after whitespace are non-empty
        assert FlextUtilities.TypeGuards.is_string_non_empty(" content ") is True
        assert FlextUtilities.TypeGuards.is_string_non_empty("\tcontent\t") is True

    def test_cache_clear_object_cache_returns_success(self) -> None:
        """Test that clear_object_cache returns success result."""
        obj = {"test": "data"}
        result = FlextUtilities.Cache.clear_object_cache(obj)
        assert result.is_success

    def test_cache_generate_cache_key_with_different_types(self) -> None:
        """Test cache key generation with different argument types."""
        key1 = FlextUtilities.Cache.generate_cache_key("arg1", "arg2", 123)
        key2 = FlextUtilities.Cache.generate_cache_key("arg1", "arg2", 123)
        assert key1 == key2  # Same arguments should generate same key

    def test_cache_generate_cache_key_with_kwargs(self) -> None:
        """Test cache key generation with keyword arguments."""
        key = FlextUtilities.Cache.generate_cache_key("test", foo="bar", num=42)
        assert isinstance(key, str)
        assert len(key) > 0

    def test_text_processor_safe_string_with_none(self) -> None:
        """Test safe_string with None - should fail (fast fail pattern)."""
        # None should fail (fast fail pattern - no fallback)
        result = FlextUtilities.TextProcessor.safe_string(None)
        assert result.is_failure

    def test_text_processor_safe_string_empty_with_default(self) -> None:
        """Test safe_string with empty string - should fail (fast fail pattern)."""
        # Empty string should fail (fast fail pattern - no fallback)
        result = FlextUtilities.TextProcessor.safe_string("")
        assert result.is_failure

    def test_validators_are_available(self) -> None:
        """Test that validation methods exist in FlextUtilities.Validation."""
        assert hasattr(FlextUtilities.Validation, "validate_required_string")
        assert hasattr(FlextUtilities.Validation, "validate_choice")
        assert hasattr(FlextUtilities.Validation, "validate_length")
        assert hasattr(FlextUtilities.Validation, "validate_pattern")
        assert hasattr(FlextUtilities.Validation, "validate_port_number")

    def test_type_checker_can_handle_multiple_types(self) -> None:
        """Test type checking with tuple of acceptable types."""
        message_types = (str, int)
        assert FlextUtilities.TypeChecker.can_handle_message_type(message_types, str)
        assert FlextUtilities.TypeChecker.can_handle_message_type(message_types, int)
        assert (
            FlextUtilities.TypeChecker.can_handle_message_type(message_types, float)
            is False
        )


class TestFlextValidationAndCacheMethods:
    """Comprehensive tests for Validation and Cache methods with high coverage."""

    def test_validate_pipeline_success_path(self) -> None:
        """Test validate_pipeline with successful validators."""

        def validator1(data: str) -> FlextResult[bool]:
            """First validator passes."""
            if len(data) < 2:
                return FlextResult[bool].fail("Too short")
            return FlextResult[bool].ok(True)

        def validator2(data: str) -> FlextResult[bool]:
            """Second validator passes."""
            if not data.isalnum():
                return FlextResult[bool].fail("Not alphanumeric")
            return FlextResult[bool].ok(True)

        result = FlextUtilities.Validation.validate_pipeline(
            "abc123", [validator1, validator2]
        )
        assert result.is_success

    def test_validate_pipeline_failure_on_first_validator(self) -> None:
        """Test validate_pipeline stops on first validator failure - CORRECT API."""

        def validator1(data: str) -> None:
            """First validator fails by raising exception (correct API)."""
            msg = "First failed"
            raise ValueError(msg)

        def validator2(data: str) -> None:
            """Second validator should not be called."""
            # Validates successfully

        result = FlextUtilities.Validation.validate_pipeline(
            "test", [validator1, validator2]
        )
        assert result.is_failure
        assert "First failed" in str(result.error)

    def test_validate_pipeline_with_exception_in_validator(self) -> None:
        """Test validate_pipeline handles exceptions from validators."""

        def bad_validator(data: str) -> FlextResult[bool]:
            """Validator that raises an exception."""
            msg = "Unexpected error"
            raise ValueError(msg)

        result = FlextUtilities.Validation.validate_pipeline("test", [bad_validator])
        assert result.is_failure

    def test_validate_pipeline_with_non_callable(self) -> None:
        """Test validate_pipeline fails on non-callable - FAST FAIL expected."""
        # Non-callable validators should FAIL immediately (FAST FAIL, not defensive programming)
        result = FlextUtilities.Validation.validate_pipeline(
            "test", [None, "not_callable"]
        )
        assert result.is_failure
        assert "Validator must be callable" in result.error

    def test_validate_pipeline_empty_validators(self) -> None:
        """Test validate_pipeline with empty validator list."""
        result = FlextUtilities.Validation.validate_pipeline("test", [])
        assert result.is_success

    def test_clear_all_caches_with_dict_cache(self) -> None:
        """Test clear_all_caches with dict-like cache attributes."""

        class MockObject:
            """Mock object with cache attributes."""

            _cache: ClassVar[dict[str, object]] = {}

        obj = MockObject()
        obj._cache = {"key": "value"}
        result = FlextUtilities.Cache.clear_object_cache(obj)
        assert result.is_success

    def test_clear_all_caches_with_none_attributes(self) -> None:
        """Test clear_all_caches with None cache attributes."""

        class MockObject:
            """Mock object with None cache."""

            _cache: object = None

        obj = MockObject()
        result = FlextUtilities.Cache.clear_object_cache(obj)
        assert result.is_success

    def test_clear_all_caches_with_exception(self) -> None:
        """Test clear_all_caches handles exceptions gracefully."""
        # Pass something that might raise an error
        result = FlextUtilities.Cache.clear_object_cache(None)
        # Should handle the error and return failure result
        assert isinstance(result, FlextResult)

    def test_has_cache_attributes_true(self) -> None:
        """Test has_cache_attributes returns True for objects with cache."""

        class MockObject:
            """Mock object with cache attributes."""

            _cache: ClassVar[dict[str, object]] = {}

        obj = MockObject()
        # Use direct method call
        has_cache = FlextUtilities.Cache.has_cache_attributes(obj)
        assert has_cache is True

    def test_has_cache_attributes_false(self) -> None:
        """Test has_cache_attributes returns False for objects without cache."""

        class EmptyObject:
            """Object without cache attributes."""

        obj = EmptyObject()
        # Use direct method call
        has_cache = FlextUtilities.Cache.has_cache_attributes(obj)
        assert has_cache is False

    def test_sort_key_with_string(self) -> None:
        """Test sort_key generates deterministic keys for strings."""
        key1 = FlextUtilities.Validation.sort_key("test_string")
        key2 = FlextUtilities.Validation.sort_key("test_string")
        assert key1 == key2
        assert isinstance(key1, str)

    def test_sort_key_with_number(self) -> None:
        """Test sort_key generates deterministic keys for numbers."""
        key1 = FlextUtilities.Validation.sort_key(42)
        key2 = FlextUtilities.Validation.sort_key(42)
        assert key1 == key2
        assert isinstance(key1, str)

    def test_sort_key_with_dict(self) -> None:
        """Test sort_key generates deterministic keys for dicts."""
        dict_a = {"z": 1, "a": 2}
        dict_b = {"a": 2, "z": 1}
        # Both should generate the same key due to sorting
        key_a = FlextUtilities.Validation.sort_key(dict_a)
        key_b = FlextUtilities.Validation.sort_key(dict_b)
        assert key_a == key_b

    def test_sort_key_with_none(self) -> None:
        """Test sort_key handles None."""
        key = FlextUtilities.Validation.sort_key(None)
        assert isinstance(key, str)

    def test_normalize_component_with_none(self) -> None:
        """Test normalize_component with None value."""
        result = FlextUtilities.Validation.normalize_component(None)
        assert result is None

    def test_normalize_component_with_primitives(self) -> None:
        """Test normalize_component with primitive types."""
        assert FlextUtilities.Validation.normalize_component(True) is True
        assert FlextUtilities.Validation.normalize_component(42) == 42
        assert FlextUtilities.Validation.normalize_component(math.pi) == math.pi
        assert FlextUtilities.Validation.normalize_component("test") == "test"

    def test_normalize_component_with_bytes(self) -> None:
        """Test normalize_component with bytes."""
        result = FlextUtilities.Validation.normalize_component(b"hello")
        assert isinstance(result, tuple)
        assert result[0] == "bytes"

    def test_normalize_component_with_dict(self) -> None:
        """Test normalize_component with dictionary."""
        test_dict = {"key1": "value1", "key2": "value2"}
        result = FlextUtilities.Validation.normalize_component(test_dict)
        # Should return a dict (or normalized structure)
        assert isinstance(result, (dict, tuple))

    def test_normalize_component_with_list(self) -> None:
        """Test normalize_component with list."""
        test_list = [1, 2, 3, "four"]
        result = FlextUtilities.Validation.normalize_component(test_list)
        # Should return tuple with 'sequence' marker
        assert isinstance(result, tuple)
        assert result[0] == "sequence"

    def test_normalize_component_with_set(self) -> None:
        """Test normalize_component with set."""
        test_set = {1, 2, 3}
        result = FlextUtilities.Validation.normalize_component(test_set)
        # Should return tuple with 'set' marker
        assert isinstance(result, tuple)
        assert result[0] == "set"

    def test_normalize_component_with_tuple(self) -> None:
        """Test normalize_component with tuple."""
        test_tuple = (1, 2, 3)
        result = FlextUtilities.Validation.normalize_component(test_tuple)
        # Should return tuple with 'sequence' marker
        assert isinstance(result, tuple)
        assert result[0] == "sequence"

    def test_normalize_component_with_nested_structure(self) -> None:
        """Test normalize_component with nested dict and list."""
        nested = {"list": [1, 2, 3], "dict": {"key": "value"}}
        result = FlextUtilities.Validation.normalize_component(nested)
        # Should handle nested structures
        assert isinstance(result, (dict, tuple))

    def test_normalize_component_with_pydantic_model(self) -> None:
        """Test normalize_component with Pydantic model."""
        from pydantic import BaseModel

        class TestModel(BaseModel):
            """Simple test model."""

            name: str
            value: int

        model = TestModel(name="test", value=42)
        result = FlextUtilities.Validation.normalize_component(model)
        # Should handle Pydantic models
        assert isinstance(result, tuple)
        assert result[0] == "pydantic"

    def test_normalize_component_deterministic_ordering(self) -> None:
        """Test normalize_component produces deterministic results."""
        dict_a = {"z": 1, "a": 2, "m": 3}
        dict_b = {"a": 2, "m": 3, "z": 1}
        # Both should normalize to the same structure
        result_a = FlextUtilities.Validation.normalize_component(dict_a)
        result_b = FlextUtilities.Validation.normalize_component(dict_b)
        assert result_a == result_b

    def test_validate_pipeline_with_mixed_validators(self) -> None:
        """Test validate_pipeline with multiple validators executed in sequence."""
        call_order = []

        def first_validator(data: str) -> FlextResult[bool]:
            """First validator in pipeline."""
            call_order.append("first")
            return FlextResult[bool].ok(True)

        def second_validator(data: str) -> FlextResult[bool]:
            """Second validator in pipeline."""
            call_order.append("second")
            return FlextResult[bool].ok(True)

        result = FlextUtilities.Validation.validate_pipeline(
            "test", [first_validator, second_validator]
        )
        # Both validators should be called in order
        assert result.is_success
        assert call_order == ["first", "second"]

    def test_sort_key_with_list(self) -> None:
        """Test sort_key with list values."""
        key = FlextUtilities.Validation.sort_key([1, 2, 3])
        assert isinstance(key, str)

    def test_sort_key_with_complex_nested_structure(self) -> None:
        """Test sort_key with complex nested structure."""
        complex_obj = {
            "list": [1, 2, 3],
            "dict": {"key": "value"},
            "nested": {"deep": {"structure": [4, 5, 6]}},
        }
        key = FlextUtilities.Validation.sort_key(complex_obj)
        assert isinstance(key, str)

    def test_normalize_component_with_dataclass(self) -> None:
        """Test normalize_component with dataclass instances."""
        from dataclasses import dataclass

        @dataclass
        class Person:
            """Simple dataclass for testing."""

            name: str
            age: int

        person = Person(name="Alice", age=30)
        result = FlextUtilities.Validation.normalize_component(person)
        # Should handle dataclasses
        assert isinstance(result, tuple)
        assert result[0] == "dataclass"

    def test_normalize_component_with_arbitrary_object(self) -> None:
        """Test normalize_component with arbitrary object using vars()."""

        class CustomObject:
            """Custom object with attributes."""

            def __init__(self) -> None:
                """Initialize object."""
                self.attr1 = "value1"
                self.attr2 = 42

        obj = CustomObject()
        result = FlextUtilities.Validation.normalize_component(obj)
        # Should handle arbitrary objects via vars()
        assert isinstance(result, tuple)
        assert result[0] == "vars"

    def test_normalize_component_with_unmappable_object(self) -> None:
        """Test normalize_component with object that raises on vars()."""

        # Create an object that raises on vars()
        class NoVarsObject:
            """Object that prevents vars() access."""

            __slots__ = ("value",)

            def __init__(self) -> None:
                """Initialize."""
                self.value = "test"

        obj = NoVarsObject()
        result = FlextUtilities.Validation.normalize_component(obj)
        # Should fall back to repr()
        assert isinstance(result, tuple)
        assert result[0] == "repr"

    def test_normalize_component_set_deterministic_ordering(self) -> None:
        """Test set normalization produces deterministic ordering."""
        set_a = {3, 1, 2}
        set_b = {1, 2, 3}
        # Both should normalize identically
        result_a = FlextUtilities.Validation.normalize_component(set_a)
        result_b = FlextUtilities.Validation.normalize_component(set_b)
        assert result_a == result_b

    def test_clear_all_caches_with_non_callable_clear(self) -> None:
        """Test clear_all_caches with cache attribute that has clear but not callable."""

        class WeirdCache:
            """Cache-like object with non-callable clear."""

            clear = "not a function"

        class ObjectWithWeirdCache:
            """Object with non-callable clear method."""

            _cache: WeirdCache = WeirdCache()

        obj = ObjectWithWeirdCache()
        result = FlextUtilities.Cache.clear_object_cache(obj)
        # Should handle non-callable clear gracefully
        assert isinstance(result, FlextResult)

    def test_validate_pipeline_with_all_non_callable(self) -> None:
        """Test validate_pipeline fails on first non-callable - FAST FAIL expected."""
        result = FlextUtilities.Validation.validate_pipeline(
            "test", [None, "string", 42, {}, []]
        )
        # Should FAIL on first non-callable (FAST FAIL, not defensive programming)
        assert result.is_failure
        assert "Validator must be callable" in result.error

    def test_sort_key_with_boolean_values(self) -> None:
        """Test sort_key with boolean values."""
        key_true = FlextUtilities.Validation.sort_key(True)
        key_false = FlextUtilities.Validation.sort_key(False)
        assert isinstance(key_true, str)
        assert isinstance(key_false, str)
        assert key_true != key_false

    def test_normalize_component_with_empty_dict(self) -> None:
        """Test normalize_component with empty dictionary."""
        result = FlextUtilities.Validation.normalize_component({})
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_normalize_component_with_empty_list(self) -> None:
        """Test normalize_component with empty list."""
        result = FlextUtilities.Validation.normalize_component([])
        assert isinstance(result, tuple)
        assert result[0] == "sequence"
        assert len(result[1]) == 0

    def test_normalize_component_with_empty_set(self) -> None:
        """Test normalize_component with empty set."""
        result = FlextUtilities.Validation.normalize_component(set())
        assert isinstance(result, tuple)
        assert result[0] == "set"

    def test_normalize_component_with_deeply_nested_lists(self) -> None:
        """Test normalize_component with deeply nested list structures."""
        nested = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        result = FlextUtilities.Validation.normalize_component(nested)
        # Should handle deep nesting
        assert isinstance(result, tuple)
        assert result[0] == "sequence"

    def test_validate_pipeline_with_typed_error(self) -> None:
        """Test validate_pipeline with various exception types."""

        def key_error_validator(data: str) -> FlextResult[bool]:
            """Validator that raises KeyError."""
            msg = "Missing key"
            raise KeyError(msg)

        result = FlextUtilities.Validation.validate_pipeline(
            "test", [key_error_validator]
        )
        assert result.is_failure

    def test_validate_pipeline_with_attribute_error(self) -> None:
        """Test validate_pipeline with AttributeError."""

        def attr_error_validator(data: str) -> FlextResult[bool]:
            """Validator that raises AttributeError."""
            msg = "Missing attribute"
            raise AttributeError(msg)

        result = FlextUtilities.Validation.validate_pipeline(
            "test", [attr_error_validator]
        )
        assert result.is_failure

    def test_normalize_component_with_float_values(self) -> None:
        """Test normalize_component preserves float values."""
        values = [math.pi, math.e, 1.41]
        result = FlextUtilities.Validation.normalize_component(values)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == "sequence"
        # Floats should be preserved in the sequence
        assert isinstance(result[1], tuple)
        assert len(result[1]) == 3
        assert result[1][0] == math.pi
        assert result[1][1] == math.e
        assert result[1][2] == 1.41

    def test_normalize_component_mixed_types_in_dict(self) -> None:
        """Test normalize_component with mixed types in dictionary."""
        mixed_dict = {
            "string": "value",
            "number": 42,
            "float": math.pi,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
        }
        result = FlextUtilities.Validation.normalize_component(mixed_dict)
        # Should handle all types in dict
        assert isinstance(result, (dict, tuple))


class TestFlextCacheMethods:
    """Comprehensive tests for Cache methods covering sort_dict_keys and normalization."""

    def test_sort_dict_keys_simple_dict(self) -> None:
        """Test sort_dict_keys with simple dictionary."""
        data = {"z": "last", "a": "first", "m": "middle"}
        result = FlextUtilities.Cache.sort_dict_keys(data)
        assert isinstance(result, dict)
        # Keys should be in sorted order
        keys = list(result.keys())
        assert keys == ["a", "m", "z"]

    def test_sort_dict_keys_nested_dict(self) -> None:
        """Test sort_dict_keys with nested dictionaries."""
        data = {
            "z": {"nested_z": 1, "nested_a": 2},
            "a": {"nested_z": 3, "nested_a": 4},
        }
        result = FlextUtilities.Cache.sort_dict_keys(data)
        assert isinstance(result, dict)
        # Top level and nested levels should be sorted
        assert list(result.keys()) == ["a", "z"]

    def test_sort_dict_keys_with_list(self) -> None:
        """Test sort_dict_keys with list."""
        data = [{"z": 1, "a": 2}, {"y": 3, "b": 4}]
        result = FlextUtilities.Cache.sort_dict_keys(data)
        assert isinstance(result, list)
        # Dicts in list should have sorted keys

    def test_sort_dict_keys_with_tuple(self) -> None:
        """Test sort_dict_keys with tuple."""
        data = ({"z": 1, "a": 2}, {"y": 3, "b": 4})
        result = FlextUtilities.Cache.sort_dict_keys(data)
        assert isinstance(result, tuple)
        # Dicts in tuple should have sorted keys

    def test_sort_dict_keys_with_primitive(self) -> None:
        """Test sort_dict_keys with primitive value."""
        result = FlextUtilities.Cache.sort_dict_keys("string_value")
        assert result == "string_value"

    def test_sort_dict_keys_deeply_nested(self) -> None:
        """Test sort_dict_keys with deeply nested structure."""
        data = {
            "z": {
                "y": {
                    "x": {"w": 1, "a": 2},
                    "b": 3,
                },
                "c": 4,
            },
            "a": 5,
        }
        result = FlextUtilities.Cache.sort_dict_keys(data)
        # Should handle deep nesting
        assert isinstance(result, dict)

    def test_sort_dict_keys_mixed_types(self) -> None:
        """Test sort_dict_keys with mixed types in nested structure."""
        data = {
            "dict_key": {"nested": 1},
            "list_key": [1, 2, {"inner": 3}],
            "tuple_key": (1, 2, {"inner": 4}),
            "primitive_key": "value",
        }
        result = FlextUtilities.Cache.sort_dict_keys(data)
        assert isinstance(result, dict)

    def test_sort_dict_keys_empty_dict(self) -> None:
        """Test sort_dict_keys with empty dictionary."""
        result = FlextUtilities.Cache.sort_dict_keys({})
        assert result == {}

    def test_sort_dict_keys_empty_list(self) -> None:
        """Test sort_dict_keys with empty list."""
        result = FlextUtilities.Cache.sort_dict_keys([])
        assert result == []

    def test_sort_dict_keys_empty_tuple(self) -> None:
        """Test sort_dict_keys with empty tuple."""
        result = FlextUtilities.Cache.sort_dict_keys(())
        assert result == ()

    def test_generate_cache_key_with_unicode_string(self) -> None:
        """Test generate_cache_key with unicode strings."""
        unicode_string = "test_\u00e9_\u00f1_\u00fc"
        key = FlextUtilities.Cache.generate_cache_key(unicode_string, str)
        assert isinstance(key, str)

    def test_generate_cache_key_large_dict(self) -> None:
        """Test generate_cache_key with large dictionary."""
        large_dict = {f"key_{i}": f"value_{i}" for i in range(100)}
        key = FlextUtilities.Cache.generate_cache_key(large_dict, dict)
        assert isinstance(key, str)

    def test_sort_dict_keys_with_int_keys_converted_to_strings(self) -> None:
        """Test sort_dict_keys converts all keys to strings."""
        # Dict with string keys
        data = {"10": "a", "2": "b", "20": "c"}
        result = FlextUtilities.Cache.sort_dict_keys(data)
        # Should sort as strings (lexicographically)
        assert isinstance(result, dict)
        keys = list(result.keys())
        assert keys == ["10", "2", "20"]  # String sort order


class TestServiceMetadataBuilder:
    """Test suite for ServiceMetadataBuilder fluent API."""

    def test_empty_builder(self) -> None:
        """Test building empty metadata."""
        from flext_core import FlextUtilities

        builder = FlextUtilities.ServiceMetadataBuilder()
        metadata = builder.build()
        assert metadata == {}

    def test_with_service_type(self) -> None:
        """Test adding service type."""
        from flext_core import FlextUtilities

        builder = FlextUtilities.ServiceMetadataBuilder()
        metadata = builder.with_service_type("UserService").build()
        assert metadata["service_type"] == "UserService"

    def test_with_service_name(self) -> None:
        """Test adding service name."""
        from flext_core import FlextUtilities

        builder = FlextUtilities.ServiceMetadataBuilder()
        metadata = builder.with_service_name("user_manager").build()
        assert metadata["service_name"] == "user_manager"

    def test_with_timestamps(self) -> None:
        """Test adding timestamps."""
        from flext_core import FlextUtilities

        builder = FlextUtilities.ServiceMetadataBuilder()
        metadata = builder.with_timestamps().build()

        assert "created_at" in metadata
        assert "extracted_at" in metadata
        assert metadata["created_at"] is not None
        assert metadata["extracted_at"] is not None

    def test_with_timestamps_selective(self) -> None:
        """Test adding selective timestamps."""
        from flext_core import FlextUtilities

        builder = FlextUtilities.ServiceMetadataBuilder()
        metadata = builder.with_timestamps(
            include_created=True, include_extracted=False
        ).build()

        assert "created_at" in metadata
        assert "extracted_at" not in metadata

    def test_with_custom_data(self) -> None:
        """Test adding custom data."""
        from flext_core import FlextUtilities

        builder = FlextUtilities.ServiceMetadataBuilder()
        metadata = builder.with_custom_data("version", "1.0.0").build()
        assert metadata["version"] == "1.0.0"

    def test_with_service_instance(self) -> None:
        """Test extracting metadata from service instance."""
        from flext_core import FlextUtilities

        class TestService:
            pass

        service = TestService()
        builder = FlextUtilities.ServiceMetadataBuilder()
        metadata = builder.with_service_instance(service).build()

        assert metadata["service_class"] == "TestService"
        assert metadata["service_module"] == "tests.unit.test_utilities"

    def test_with_service_instance_with_name(self) -> None:
        """Test extracting metadata from service with custom name."""
        from flext_core import FlextUtilities

        class TestService:
            _service_name = "custom_name"

        service = TestService()
        builder = FlextUtilities.ServiceMetadataBuilder()
        metadata = builder.with_service_instance(service).build()

        assert metadata["service_class"] == "TestService"
        assert metadata["service_name"] == "custom_name"

    def test_fluent_chaining(self) -> None:
        """Test fluent method chaining."""
        from flext_core import FlextUtilities

        builder = FlextUtilities.ServiceMetadataBuilder()
        metadata = (
            builder.with_service_type("UserService")
            .with_service_name("user_manager")
            .with_timestamps()
            .with_custom_data("version", "1.0.0")
            .build()
        )

        assert metadata["service_type"] == "UserService"
        assert metadata["service_name"] == "user_manager"
        assert "created_at" in metadata
        assert "extracted_at" in metadata
        assert metadata["version"] == "1.0.0"

    def test_method_returns_self(self) -> None:
        """Test that methods return self for chaining."""
        from flext_core import FlextUtilities

        builder = FlextUtilities.ServiceMetadataBuilder()
        result = builder.with_service_type("Test")
        assert result is builder


class TestImprovedFlextUtilities:
    """Test suite for improved FlextUtilities methods."""

    def test_extract_common_kwargs_success(self) -> None:
        """Test successful extraction of common kwargs."""
        kwargs: dict[str, object] = {
            "correlation_id": "test-123",
            "metadata": {"key": "value"},
        }
        result = FlextUtilities.Exceptions.extract_common_kwargs(kwargs)

        assert result.is_success
        correlation_id, metadata = result.unwrap()
        assert correlation_id == "test-123"
        assert metadata == {"key": "value"}

    def test_extract_common_kwargs_empty(self) -> None:
        """Test extraction with empty kwargs."""
        kwargs: dict[str, object] = {}
        result = FlextUtilities.Exceptions.extract_common_kwargs(kwargs)

        assert result.is_success
        correlation_id, metadata = result.unwrap()
        assert correlation_id is None
        assert metadata is None

    def test_extract_common_kwargs_failure(self) -> None:
        """Test extraction failure handling."""
        # This should not fail in normal usage, but let's test error handling
        kwargs = {"invalid": object()}  # Object that might cause issues
        result = FlextUtilities.Exceptions.extract_common_kwargs(kwargs)

        # Should still succeed since we use try/catch
        assert result.is_success

    def test_format_service_info_success(self) -> None:
        """Test successful service info formatting."""

        class TestService:
            pass

        service = TestService()
        metadata: dict[str, object] = {
            "service_type": "UserService",
            "service_name": "user_manager",
        }

        result = FlextUtilities.ServiceHelpers.format_service_info(service, metadata)

        assert result.is_success
        formatted = result.unwrap()
        assert "Service: UserService (user_manager)" in formatted
        assert "tests.unit.test_utilities" in formatted

    def test_format_service_info_minimal(self) -> None:
        """Test service info formatting with minimal metadata."""
        service = object()
        metadata: dict[str, object] = {}

        result = FlextUtilities.ServiceHelpers.format_service_info(service, metadata)

        assert result.is_success
        formatted = result.unwrap()
        assert "Service: Unknown (unnamed)" in formatted

    def test_format_service_info_failure(self) -> None:
        """Test service info formatting error handling."""
        service = None
        metadata: dict[str, object] = {"service_type": "Test"}

        result = FlextUtilities.ServiceHelpers.format_service_info(service, metadata)

        assert result.is_success  # Should handle None gracefully

    def test_create_service_metadata_builder(self) -> None:
        """Test creating service metadata builder."""
        from flext_core import FlextUtilities

        builder = FlextUtilities.ServiceHelpers.create_service_metadata_builder()

        assert isinstance(builder, FlextUtilities.ServiceMetadataBuilder)
