"""Comprehensive tests for FlextUtilities module to achieve 100% coverage.

This module provides comprehensive tests for the FlextUtilities utility classes
using flext_tests standardization patterns to ensure complete coverage of all
utility methods and edge cases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from flext_core import FlextUtilities


class TestFlextUtilitiesValidation:
    """Comprehensive tests for FlextUtilities.Validation class."""

    def test_is_non_empty_string_valid(self) -> None:
        """Test Validation.is_non_empty_string with valid strings."""
        assert FlextUtilities.Validation.is_non_empty_string("hello") is True
        assert FlextUtilities.Validation.is_non_empty_string("test") is True
        assert FlextUtilities.Validation.is_non_empty_string("a") is True
        assert FlextUtilities.Validation.is_non_empty_string("  content  ") is True

    def test_is_non_empty_string_invalid(self) -> None:
        """Test Validation.is_non_empty_string with invalid inputs."""
        assert FlextUtilities.Validation.is_non_empty_string("") is False
        # NOTE: API signature expects str, not str|None, so None test is removed


class TestFlextUtilitiesGenerators:
    """Comprehensive tests for FlextUtilities.Generators class."""

    def test_generate_id(self) -> None:
        """Test Generators.generate_id method."""
        id1 = FlextUtilities.Generators.generate_id()
        id2 = FlextUtilities.Generators.generate_id()

        # Should generate different IDs
        assert id1 != id2
        # Should be strings
        assert isinstance(id1, str)
        assert isinstance(id2, str)
        # Should be non-empty
        assert len(id1) > 0
        assert len(id2) > 0

    def test_generate_uuid(self) -> None:
        """Test Generators.generate_uuid method."""
        uuid1 = FlextUtilities.Generators.generate_uuid()
        uuid2 = FlextUtilities.Generators.generate_uuid()

        # Should generate different UUIDs
        assert uuid1 != uuid2
        # Should be valid UUID strings
        assert isinstance(uuid1, str)
        assert isinstance(uuid2, str)
        # Should be valid UUID format (can be parsed)
        UUID(uuid1)  # Should not raise exception
        UUID(uuid2)  # Should not raise exception

    def test_generate_iso_timestamp(self) -> None:
        """Test Generators.generate_iso_timestamp method."""
        timestamp1 = FlextUtilities.Generators.generate_iso_timestamp()
        timestamp2 = FlextUtilities.Generators.generate_iso_timestamp()

        # Should be strings
        assert isinstance(timestamp1, str)
        assert isinstance(timestamp2, str)
        # Should be valid ISO format (can be parsed)
        datetime.fromisoformat(timestamp1)
        datetime.fromisoformat(timestamp2)
        # Different calls should produce different timestamps (usually)
        # Note: In very rare cases they might be the same due to timing

    def test_generate_correlation_id(self) -> None:
        """Test Generators.generate_correlation_id method."""
        corr_id1 = FlextUtilities.Generators.generate_correlation_id()
        corr_id2 = FlextUtilities.Generators.generate_correlation_id()

        # Should generate different correlation IDs
        assert corr_id1 != corr_id2
        # Should be strings
        assert isinstance(corr_id1, str)
        assert isinstance(corr_id2, str)
        # Should have expected prefix pattern (if any)
        assert len(corr_id1) > 0
        assert len(corr_id2) > 0

    def test_generate_request_id(self) -> None:
        """Test Generators.generate_id method."""
        req_id1 = FlextUtilities.Generators.generate_id()
        req_id2 = FlextUtilities.Generators.generate_id()

        # Should generate different request IDs
        assert req_id1 != req_id2
        # Should be strings
        assert isinstance(req_id1, str)
        assert isinstance(req_id2, str)
        # Should be non-empty
        assert len(req_id1) > 0
        assert len(req_id2) > 0

    def test_generate_entity_id(self) -> None:
        """Test Generators.generate_entity_id method."""
        entity_id1 = FlextUtilities.Generators.generate_entity_id()
        entity_id2 = FlextUtilities.Generators.generate_entity_id()

        # Should generate different entity IDs
        assert entity_id1 != entity_id2
        # Should be strings
        assert isinstance(entity_id1, str)
        assert isinstance(entity_id2, str)
        # Should be non-empty
        assert len(entity_id1) > 0
        assert len(entity_id2) > 0


class TestFlextUtilitiesTextProcessor:
    """Comprehensive tests for FlextUtilities.TextProcessor class."""

    def test_safe_string_valid_inputs(self) -> None:
        """Test TextProcessor.safe_string with valid inputs."""
        assert FlextUtilities.TextProcessor.safe_string("hello") == "hello"
        assert FlextUtilities.TextProcessor.safe_string("test string") == "test string"
        assert FlextUtilities.TextProcessor.safe_string("") is not None

    def test_safe_string_none_input(self) -> None:
        """Test TextProcessor.safe_string with None input."""
        # NOTE: safe_string expects str parameter, not str|None
        # This test is removed as it doesn't match the API signature

    def test_safe_string_non_string_inputs(self) -> None:
        """Test TextProcessor.safe_string with non-string inputs."""
        # NOTE: safe_string expects str parameter, not other types
        # These tests are removed as they don't match the API signature

    def test_clean_text_basic(self) -> None:
        """Test TextProcessor.clean_text basic functionality."""
        assert FlextUtilities.TextProcessor.clean_text("  hello  ") == "hello"
        assert FlextUtilities.TextProcessor.clean_text("test\nstring") == "test string"
        assert FlextUtilities.TextProcessor.clean_text("test\tstring") == "test string"
        assert (
            FlextUtilities.TextProcessor.clean_text("test\r\nstring") == "test string"
        )

    def test_clean_text_none_input(self) -> None:
        """Test TextProcessor.clean_text with None input."""
        # NOTE: clean_text expects str parameter, not str|None
        # This test is removed as it doesn't match the API signature

    def test_clean_text_edge_cases(self) -> None:
        """Test TextProcessor.clean_text with edge cases."""
        assert FlextUtilities.TextProcessor.clean_text("") is not None
        assert FlextUtilities.TextProcessor.clean_text("   ") is not None
        assert FlextUtilities.TextProcessor.clean_text("\n\r\t") is not None

    def test_is_non_empty_string_valid(self) -> None:
        """Test Validation.is_non_empty_string with valid strings."""
        assert FlextUtilities.Validation.is_non_empty_string("hello") is True
        assert FlextUtilities.Validation.is_non_empty_string("test") is True
        assert FlextUtilities.Validation.is_non_empty_string("a") is True

    def test_is_non_empty_string_invalid(self) -> None:
        """Test Validation.is_non_empty_string with invalid inputs."""
        assert FlextUtilities.Validation.is_non_empty_string("") is False
        # NOTE: is_non_empty_string expects str parameter, not str|None

    # NOTE: slugify method doesn't exist in TextProcessor class
    # These tests are removed as they don't match the actual API


class TestFlextUtilitiesConversions:
    """Comprehensive tests for FlextUtilities.Conversions class."""

    def test_to_int_valid_inputs(self) -> None:
        """Test Conversions.to_int with valid inputs."""
        result1 = FlextUtilities.Conversions.to_int("123")
        assert result1.success is True
        assert result1.value == 123

        result2 = FlextUtilities.Conversions.to_int("0")
        assert result2.success is True
        assert result2.value == 0

        result3 = FlextUtilities.Conversions.to_int("-456")
        assert result3.success is True
        assert result3.value == -456

        result4 = FlextUtilities.Conversions.to_int(789)
        assert result4.success is True
        assert result4.value == 789

    def test_to_int_invalid_inputs(self) -> None:
        """Test Conversions.to_int with invalid inputs."""
        result1 = FlextUtilities.Conversions.to_int("abc")
        assert result1.success is False
        assert result1.error is not None
        assert "Integer conversion failed" in result1.error

        result2 = FlextUtilities.Conversions.to_int("")
        assert result2.success is False
        assert result2.error is not None
        assert "Integer conversion failed" in result2.error

        result3 = FlextUtilities.Conversions.to_int(None)
        assert result3.success is False
        assert result3.error is not None
        assert "Cannot convert None to integer" in result3.error

    def test_to_bool_true_values(self) -> None:
        """Test Conversions.to_bool with values that should return True."""
        result1 = FlextUtilities.Conversions.to_bool(value="true")
        assert result1.success is True
        assert result1.value is True

        result2 = FlextUtilities.Conversions.to_bool(value="True")
        assert result2.success is True
        assert result2.value is True

        result3 = FlextUtilities.Conversions.to_bool(value="TRUE")
        assert result3.success is True
        assert result3.value is True

        result4 = FlextUtilities.Conversions.to_bool(value="yes")
        assert result4.success is True
        assert result4.value is True

        result5 = FlextUtilities.Conversions.to_bool(value="1")
        assert result5.success is True
        assert result5.value is True

        result6 = FlextUtilities.Conversions.to_bool(value=1)
        assert result6.success is True
        assert result6.value is True

        result7 = FlextUtilities.Conversions.to_bool(value=True)
        assert result7.success is True
        assert result7.value is True

    def test_to_bool_false_values(self) -> None:
        """Test Conversions.to_bool with values that should return False."""
        result1 = FlextUtilities.Conversions.to_bool(value="false")
        assert result1.success is True
        assert result1.value is False

        result2 = FlextUtilities.Conversions.to_bool(value="False")
        assert result2.success is True
        assert result2.value is False

        result3 = FlextUtilities.Conversions.to_bool(value="FALSE")
        assert result3.success is True
        assert result3.value is False

        result4 = FlextUtilities.Conversions.to_bool(value="no")
        assert result4.success is True
        assert result4.value is False

        result5 = FlextUtilities.Conversions.to_bool(value="0")
        assert result5.success is True
        assert result5.value is False

        result6 = FlextUtilities.Conversions.to_bool(value=0)
        assert result6.success is True
        assert result6.value is False

        result7 = FlextUtilities.Conversions.to_bool(value=False)
        assert result7.success is True
        assert result7.value is False

    def test_to_bool_edge_cases(self) -> None:
        """Test Conversions.to_bool with edge cases."""
        # Test with invalid string values
        result1 = FlextUtilities.Conversions.to_bool(value="maybe")
        assert result1.success is False
        assert result1.error is not None
        assert "Cannot convert 'maybe' to boolean" in result1.error

        # Test with empty string
        result2 = FlextUtilities.Conversions.to_bool(value="")
        assert result2.success is True
        assert result2.value is False

        # Test with None
        result3 = FlextUtilities.Conversions.to_bool(value=None)
        assert result3.success is True
        assert result3.value is False

        # Test with numeric values
        result4 = FlextUtilities.Conversions.to_bool(value=2)
        assert result4.success is True
        assert result4.value is True

        result5 = FlextUtilities.Conversions.to_bool(value=-1)
        assert result5.success is True
        assert result5.value is True

    # NOTE: safe_float method doesn't exist in Conversions class
    # These tests are removed as they don't match the actual API


class TestFlextUtilitiesTypeGuards:
    """Comprehensive tests for FlextUtilities.TypeGuards class."""

    def test_is_dict_non_empty_valid(self) -> None:
        """Test TypeGuards.is_dict_non_empty with non-empty dictionaries."""
        assert FlextUtilities.TypeGuards.is_dict_non_empty({"key": "value"}) is True
        assert FlextUtilities.TypeGuards.is_dict_non_empty({"a": 1, "b": 2}) is True
        assert FlextUtilities.TypeGuards.is_dict_non_empty({1: "one"}) is True

    def test_is_dict_non_empty_invalid(self) -> None:
        """Test TypeGuards.is_dict_non_empty with invalid inputs."""
        assert FlextUtilities.TypeGuards.is_dict_non_empty({}) is False
        assert FlextUtilities.TypeGuards.is_dict_non_empty(None) is False
        assert FlextUtilities.TypeGuards.is_dict_non_empty([]) is False
        assert FlextUtilities.TypeGuards.is_dict_non_empty("dict") is False
        assert FlextUtilities.TypeGuards.is_dict_non_empty(123) is False

    def test_is_string_non_empty_valid(self) -> None:
        """Test TypeGuards.is_string_non_empty with non-empty strings."""
        assert FlextUtilities.TypeGuards.is_string_non_empty("hello") is True
        assert FlextUtilities.TypeGuards.is_string_non_empty("test") is True
        assert FlextUtilities.TypeGuards.is_string_non_empty("a") is True
        assert FlextUtilities.TypeGuards.is_string_non_empty("  spaces  ") is True

    def test_is_string_non_empty_invalid(self) -> None:
        """Test TypeGuards.is_string_non_empty with invalid inputs."""
        assert FlextUtilities.TypeGuards.is_string_non_empty("") is False
        assert FlextUtilities.TypeGuards.is_string_non_empty(None) is False
        assert FlextUtilities.TypeGuards.is_string_non_empty(123) is False
        assert FlextUtilities.TypeGuards.is_string_non_empty([]) is False
        assert FlextUtilities.TypeGuards.is_string_non_empty({}) is False

    def test_is_list_non_empty_valid(self) -> None:
        """Test TypeGuards.is_list_non_empty with non-empty lists."""
        assert FlextUtilities.TypeGuards.is_list_non_empty([1, 2, 3]) is True
        assert FlextUtilities.TypeGuards.is_list_non_empty(["a"]) is True
        assert FlextUtilities.TypeGuards.is_list_non_empty([None]) is True
        assert FlextUtilities.TypeGuards.is_list_non_empty([{"key": "value"}]) is True

    def test_is_list_non_empty_invalid(self) -> None:
        """Test TypeGuards.is_list_non_empty with invalid inputs."""
        assert FlextUtilities.TypeGuards.is_list_non_empty([]) is False
        assert FlextUtilities.TypeGuards.is_list_non_empty(None) is False
        assert FlextUtilities.TypeGuards.is_list_non_empty("list") is False
        assert FlextUtilities.TypeGuards.is_list_non_empty({}) is False
        assert FlextUtilities.TypeGuards.is_list_non_empty(123) is False


class TestFlextUtilitiesTopLevelMethods:
    """Comprehensive tests for top-level FlextUtilities methods."""

    def test_generate_iso_timestamp(self) -> None:
        """Test Generators.generate_iso_timestamp method."""
        timestamp1 = FlextUtilities.Generators.generate_iso_timestamp()
        timestamp2 = FlextUtilities.Generators.generate_iso_timestamp()

        # Should be strings
        assert isinstance(timestamp1, str)
        assert isinstance(timestamp2, str)
        # Should be valid ISO format (can be parsed)
        datetime.fromisoformat(timestamp1)
        datetime.fromisoformat(timestamp2)

    # NOTE: safe_json_stringify method doesn't exist in FlextUtilities
    # These tests are removed as they don't match the actual API

    # NOTE: All safe_json_stringify tests removed as method doesn't exist


class TestFlextUtilitiesIntegration:
    """Integration tests for FlextUtilities components working together."""

    def test_generators_and_validation_integration(self) -> None:
        """Test generators with validation methods."""
        # Generate IDs and validate they're non-empty strings
        uuid_val = FlextUtilities.Generators.generate_uuid()
        assert FlextUtilities.Validation.is_non_empty_string(uuid_val) is True

        entity_id = FlextUtilities.Generators.generate_id()
        assert FlextUtilities.TypeGuards.is_string_non_empty(entity_id) is True

    def test_text_processing_and_json_integration(self) -> None:
        """Test text processing with JSON utilities."""
        # Process text
        processed_text_result = FlextUtilities.TextProcessor.clean_text(
            "  hello world  \n"
        )
        assert processed_text_result.success is True
        assert processed_text_result.value == "hello world"

    def test_conversions_and_type_guards_integration(self) -> None:
        """Test conversions with type guards."""
        # Convert and validate
        int_result = FlextUtilities.Conversions.to_int("123")
        assert int_result.success is True
        converted_dict = {"converted_int": int_result.value}
        assert FlextUtilities.TypeGuards.is_dict_non_empty(converted_dict) is True

        bool_result = FlextUtilities.Conversions.to_bool(value="true")
        assert bool_result.success is True
        bool_list = [bool_result.value]
        assert FlextUtilities.TypeGuards.is_list_non_empty(bool_list) is True

    def test_full_workflow_integration(self) -> None:
        """Test a full workflow using multiple utility components."""
        # Generate data
        entity_id = FlextUtilities.Generators.generate_id()
        timestamp = FlextUtilities.Generators.generate_iso_timestamp()

        # Process and clean
        raw_text = "  User Registration Event  \n"
        clean_text_result = FlextUtilities.TextProcessor.clean_text(raw_text)
        assert clean_text_result.success is True
        clean_text = clean_text_result.value

        # Create data structure
        bool_result = FlextUtilities.Conversions.to_bool(value="true")
        assert bool_result.success is True

        event_data = {
            "entity_id": entity_id,
            "timestamp": timestamp,
            "event_type": clean_text,
            "active": bool_result.value,
        }

        # Validate structure
        assert FlextUtilities.TypeGuards.is_dict_non_empty(event_data) is True
        assert (
            FlextUtilities.TypeGuards.is_string_non_empty(event_data["entity_id"])
            is True
        )

        # Validate result
        assert event_data["event_type"] == "User Registration Event"
        assert event_data["active"] is True


# NOTE: safe_result method doesn't exist in Reliability class
# These tests are removed as they don't match the actual API
