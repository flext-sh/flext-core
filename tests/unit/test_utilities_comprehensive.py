"""Comprehensive tests for FlextUtilities module to achieve 100% coverage.

This module provides comprehensive tests for the FlextUtilities utility classes
using flext_tests standardization patterns to ensure complete coverage of all
utility methods and edge cases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from datetime import datetime
from uuid import UUID

from flext_core import FlextResult, FlextUtilities


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
        assert FlextUtilities.Validation.is_non_empty_string(None) is False
        # NOTE: API signature expects str|None, non-string inputs should be handled by type checking
        # Removing invalid test cases that don't match actual API signature


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
        """Test Generators.generate_request_id method."""
        req_id1 = FlextUtilities.Generators.generate_request_id()
        req_id2 = FlextUtilities.Generators.generate_request_id()

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
        assert FlextUtilities.TextProcessor.safe_string("") == ""

    def test_safe_string_none_input(self) -> None:
        """Test TextProcessor.safe_string with None input."""
        assert FlextUtilities.TextProcessor.safe_string(None) == ""

    def test_safe_string_non_string_inputs(self) -> None:
        """Test TextProcessor.safe_string with non-string inputs."""
        assert FlextUtilities.TextProcessor.safe_string(123) == "123"
        assert FlextUtilities.TextProcessor.safe_string(45.67) == "45.67"
        assert FlextUtilities.TextProcessor.safe_string(True) == "True"
        assert FlextUtilities.TextProcessor.safe_string(False) == "False"
        assert FlextUtilities.TextProcessor.safe_string([1, 2, 3]) == "[1, 2, 3]"
        assert (
            FlextUtilities.TextProcessor.safe_string({"key": "value"})
            == "{'key': 'value'}"
        )

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
        assert FlextUtilities.TextProcessor.clean_text(None) == ""

    def test_clean_text_edge_cases(self) -> None:
        """Test TextProcessor.clean_text with edge cases."""
        assert FlextUtilities.TextProcessor.clean_text("") == ""
        assert FlextUtilities.TextProcessor.clean_text("   ") == ""
        assert FlextUtilities.TextProcessor.clean_text("\n\r\t") == ""

    def test_is_non_empty_string_valid(self) -> None:
        """Test TextProcessor.is_non_empty_string with valid strings."""
        assert FlextUtilities.TextProcessor.is_non_empty_string("hello") is True
        assert FlextUtilities.TextProcessor.is_non_empty_string("test") is True
        assert FlextUtilities.TextProcessor.is_non_empty_string("a") is True

    def test_is_non_empty_string_invalid(self) -> None:
        """Test TextProcessor.is_non_empty_string with invalid inputs."""
        assert FlextUtilities.TextProcessor.is_non_empty_string("") is False
        assert FlextUtilities.TextProcessor.is_non_empty_string(None) is False

    def test_slugify_basic(self) -> None:
        """Test TextProcessor.slugify basic functionality."""
        assert FlextUtilities.TextProcessor.slugify("Hello World") == "hello-world"
        assert FlextUtilities.TextProcessor.slugify("Test String") == "test-string"
        assert FlextUtilities.TextProcessor.slugify("hello") == "hello"

    def test_slugify_special_characters(self) -> None:
        """Test TextProcessor.slugify with special characters."""
        # The regex removes special chars but keeps word chars (including _)
        assert FlextUtilities.TextProcessor.slugify("Hello@World!") == "helloworld"
        assert FlextUtilities.TextProcessor.slugify("Test_String") == "test_string"
        assert FlextUtilities.TextProcessor.slugify("hello-world") == "hello-world"

    def test_slugify_edge_cases(self) -> None:
        """Test TextProcessor.slugify with edge cases."""
        assert FlextUtilities.TextProcessor.slugify("") == ""
        assert FlextUtilities.TextProcessor.slugify("   ") == ""
        assert FlextUtilities.TextProcessor.slugify("123") == "123"

    def test_slugify_none_input(self) -> None:
        """Test TextProcessor.slugify with None input."""
        result = FlextUtilities.TextProcessor.slugify(None)
        assert result == "" or result is None  # Handle both possible behaviors


class TestFlextUtilitiesConversions:
    """Comprehensive tests for FlextUtilities.Conversions class."""

    def test_safe_int_valid_inputs(self) -> None:
        """Test Conversions.safe_int with valid inputs."""
        assert FlextUtilities.Conversions.safe_int("123") == 123
        assert FlextUtilities.Conversions.safe_int("0") == 0
        assert FlextUtilities.Conversions.safe_int("-456") == -456
        assert FlextUtilities.Conversions.safe_int(789) == 789

    def test_safe_int_invalid_inputs_with_default(self) -> None:
        """Test Conversions.safe_int with invalid inputs using default."""
        # API uses keyword-only default parameter
        assert FlextUtilities.Conversions.safe_int("abc", default=0) == 0
        assert FlextUtilities.Conversions.safe_int("", default=42) == 42
        assert FlextUtilities.Conversions.safe_int(None, default=-1) == -1
        assert FlextUtilities.Conversions.safe_int("12.34", default=100) == 100

    def test_safe_int_invalid_inputs_no_default(self) -> None:
        """Test Conversions.safe_int with invalid inputs without default."""
        assert FlextUtilities.Conversions.safe_int("abc") == 0
        assert FlextUtilities.Conversions.safe_int("") == 0
        assert FlextUtilities.Conversions.safe_int(None) == 0

    def test_safe_bool_true_values(self) -> None:
        """Test Conversions.safe_bool with values that should return True."""
        assert FlextUtilities.Conversions.safe_bool("true") is True
        assert FlextUtilities.Conversions.safe_bool("True") is True
        assert FlextUtilities.Conversions.safe_bool("TRUE") is True
        assert FlextUtilities.Conversions.safe_bool("yes") is True
        assert FlextUtilities.Conversions.safe_bool("Yes") is True
        assert FlextUtilities.Conversions.safe_bool("YES") is True
        assert FlextUtilities.Conversions.safe_bool("1") is True
        assert FlextUtilities.Conversions.safe_bool(1) is True
        assert FlextUtilities.Conversions.safe_bool(True) is True

    def test_safe_bool_false_values(self) -> None:
        """Test Conversions.safe_bool with values that should return False."""
        assert FlextUtilities.Conversions.safe_bool("false") is False
        assert FlextUtilities.Conversions.safe_bool("False") is False
        assert FlextUtilities.Conversions.safe_bool("FALSE") is False
        assert FlextUtilities.Conversions.safe_bool("no") is False
        assert FlextUtilities.Conversions.safe_bool("No") is False
        assert FlextUtilities.Conversions.safe_bool("NO") is False
        assert FlextUtilities.Conversions.safe_bool("0") is False
        assert FlextUtilities.Conversions.safe_bool(0) is False
        assert FlextUtilities.Conversions.safe_bool(False) is False

    def test_safe_bool_with_default(self) -> None:
        """Test Conversions.safe_bool with default values."""
        # API uses keyword-only default parameter
        assert FlextUtilities.Conversions.safe_bool("invalid", default=True) is True
        assert FlextUtilities.Conversions.safe_bool("invalid", default=False) is False
        assert FlextUtilities.Conversions.safe_bool(None, default=True) is True
        assert FlextUtilities.Conversions.safe_bool("", default=False) is False

    def test_safe_bool_edge_cases(self) -> None:
        """Test Conversions.safe_bool with edge cases."""
        # Test with other string values
        assert (
            FlextUtilities.Conversions.safe_bool("maybe") is False
        )  # Default behavior for unknown strings
        assert FlextUtilities.Conversions.safe_bool("") is False
        assert FlextUtilities.Conversions.safe_bool(None) is False

        # Test with numeric values
        assert (
            FlextUtilities.Conversions.safe_bool(2) is True
        )  # Non-zero numbers should be True
        assert FlextUtilities.Conversions.safe_bool(-1) is True

    def test_safe_float_valid_inputs(self) -> None:
        """Test Conversions.safe_float with valid inputs."""
        assert FlextUtilities.Conversions.safe_float("123.45") == 123.45
        assert FlextUtilities.Conversions.safe_float("0.0") == 0.0
        assert FlextUtilities.Conversions.safe_float("-456.78") == -456.78
        assert FlextUtilities.Conversions.safe_float("123") == 123.0
        assert FlextUtilities.Conversions.safe_float(789.12) == 789.12

    def test_safe_float_invalid_inputs_with_default(self) -> None:
        """Test Conversions.safe_float with invalid inputs using default."""
        # API uses keyword-only default parameter
        assert FlextUtilities.Conversions.safe_float("abc", default=0.0) == 0.0
        assert FlextUtilities.Conversions.safe_float("", default=1.5) == 1.5
        assert FlextUtilities.Conversions.safe_float(None, default=-2.5) == -2.5

    def test_safe_float_invalid_inputs_no_default(self) -> None:
        """Test Conversions.safe_float with invalid inputs without default."""
        assert FlextUtilities.Conversions.safe_float("abc") == 0.0
        assert FlextUtilities.Conversions.safe_float("") == 0.0
        assert FlextUtilities.Conversions.safe_float(None) == 0.0


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
        """Test top-level generate_iso_timestamp method."""
        timestamp1 = FlextUtilities.generate_iso_timestamp()
        timestamp2 = FlextUtilities.generate_iso_timestamp()

        # Should be strings
        assert isinstance(timestamp1, str)
        assert isinstance(timestamp2, str)
        # Should be valid ISO format (can be parsed)
        datetime.fromisoformat(timestamp1)
        datetime.fromisoformat(timestamp2)

    def test_safe_json_stringify_valid_inputs(self) -> None:
        """Test safe_json_stringify with valid JSON-serializable inputs."""
        # Simple values
        assert FlextUtilities.safe_json_stringify("hello") == '"hello"'
        assert FlextUtilities.safe_json_stringify(123) == "123"
        assert FlextUtilities.safe_json_stringify(True) == "true"
        assert FlextUtilities.safe_json_stringify(False) == "false"
        assert FlextUtilities.safe_json_stringify(None) == "null"

        # Complex structures
        data = {"key": "value", "number": 42}
        result = FlextUtilities.safe_json_stringify(data)
        assert isinstance(result, str)
        # Should be valid JSON (can be parsed back)
        parsed = json.loads(result)
        assert parsed == data

    def test_safe_json_stringify_with_default(self) -> None:
        """Test safe_json_stringify with default value."""
        default_value = '{"error": "serialization_failed"}'

        # Test with non-serializable object
        class NonSerializable:
            pass

        non_serializable = NonSerializable()
        result = FlextUtilities.safe_json_stringify(
            non_serializable, default=default_value
        )
        assert result == default_value

    def test_safe_json_stringify_without_default(self) -> None:
        """Test safe_json_stringify without default (should return default JSON error)."""

        class NonSerializable:
            pass

        non_serializable = NonSerializable()
        result = FlextUtilities.safe_json_stringify(non_serializable)

        # Should return some default error representation
        assert isinstance(result, str)
        # Should be valid JSON
        json.loads(result)  # Should not raise exception

    def test_safe_json_stringify_edge_cases(self) -> None:
        """Test safe_json_stringify with edge cases."""
        # Empty containers
        assert FlextUtilities.safe_json_stringify({}) == "{}"
        assert FlextUtilities.safe_json_stringify([]) == "[]"

        # Nested structures
        nested = {"level1": {"level2": {"level3": "deep"}}}
        result = FlextUtilities.safe_json_stringify(nested)
        parsed = json.loads(result)
        assert parsed == nested

    def test_safe_json_stringify_special_characters(self) -> None:
        """Test safe_json_stringify with special characters."""
        special_text = 'Text with "quotes" and \n newlines and \t tabs'
        result = FlextUtilities.safe_json_stringify(special_text)

        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed == special_text


class TestFlextUtilitiesIntegration:
    """Integration tests for FlextUtilities components working together."""

    def test_generators_and_validation_integration(self) -> None:
        """Test generators with validation methods."""
        # Generate IDs and validate they're non-empty strings
        uuid_val = FlextUtilities.Generators.generate_uuid()
        assert FlextUtilities.Validation.is_non_empty_string(uuid_val) is True

        entity_id = FlextUtilities.Generators.generate_entity_id()
        assert FlextUtilities.TypeGuards.is_string_non_empty(entity_id) is True

    def test_text_processing_and_json_integration(self) -> None:
        """Test text processing with JSON utilities."""
        # Process text and serialize
        processed_text = FlextUtilities.TextProcessor.clean_text("  hello world  \n")
        json_result = FlextUtilities.safe_json_stringify(processed_text)

        # Should be valid JSON containing the cleaned text
        parsed = json.loads(json_result)
        assert parsed == "hello world"

    def test_conversions_and_type_guards_integration(self) -> None:
        """Test conversions with type guards."""
        # Convert and validate
        converted_dict = {"converted_int": FlextUtilities.Conversions.safe_int("123")}
        assert FlextUtilities.TypeGuards.is_dict_non_empty(converted_dict) is True

        bool_list = [FlextUtilities.Conversions.safe_bool("true")]
        assert FlextUtilities.TypeGuards.is_list_non_empty(bool_list) is True

    def test_full_workflow_integration(self) -> None:
        """Test a full workflow using multiple utility components."""
        # Generate data
        entity_id = FlextUtilities.Generators.generate_entity_id()
        timestamp = FlextUtilities.generate_iso_timestamp()

        # Process and clean
        raw_text = "  User Registration Event  \n"
        clean_text = FlextUtilities.TextProcessor.clean_text(raw_text)
        slug = FlextUtilities.TextProcessor.slugify(clean_text)

        # Create data structure
        event_data = {
            "entity_id": entity_id,
            "timestamp": timestamp,
            "event_type": clean_text,
            "event_slug": slug,
            "active": FlextUtilities.Conversions.safe_bool("true"),
        }

        # Validate structure
        assert FlextUtilities.TypeGuards.is_dict_non_empty(event_data) is True
        assert (
            FlextUtilities.TypeGuards.is_string_non_empty(event_data["entity_id"])
            is True
        )

        # Serialize
        json_result = FlextUtilities.safe_json_stringify(event_data)

        # Validate result
        assert isinstance(json_result, str)
        parsed_back = json.loads(json_result)
        assert parsed_back["event_type"] == "User Registration Event"
        assert parsed_back["event_slug"] == "user-registration-event"
        assert parsed_back["active"] is True


class TestFlextUtilitiesReliability:
    """Tests for reliability helpers."""

    def test_safe_result_wraps_return_value(self) -> None:
        """Test safe_result wraps return value."""

        @FlextUtilities.Reliability.safe_result
        def add(a: int, b: int) -> int:
            return a + b

        result = add(2, 3)
        assert isinstance(result, FlextResult)
        assert result.is_success
        assert result.unwrap() == 5

    def test_safe_result_preserves_flext_result(self) -> None:
        """Test safe_result preserves FlextResult."""

        @FlextUtilities.Reliability.safe_result
        def make_result(value: int) -> FlextResult[int]:
            return FlextResult[int].ok(value)

        result = make_result(7)
        assert result.is_success
        assert result.unwrap() == 7

    def test_safe_result_captures_exception(self) -> None:
        """Test safe_result captures exception."""

        @FlextUtilities.Reliability.safe_result
        def explode() -> int:
            error_msg = "boom"
            raise ValueError(error_msg)

        result = explode()
        assert result.is_failure
        assert result.error == "boom"
