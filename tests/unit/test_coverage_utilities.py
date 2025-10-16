"""Comprehensive coverage tests for FlextUtilities.

This module provides extensive tests for FlextUtilities functionality:
- Validation (15+ validators)
- Generators (14+ ID/timestamp generators)
- Type guards and conversions
- Text processing and reliability patterns
- Type checking for CQRS handlers
- Configuration parameter access
- External command execution

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import math

import pytest

from flext_core import FlextResult, FlextUtilities


class TestValidation:
    """Test FlextUtilities.Validation namespace."""

    def test_validate_string_not_empty(self) -> None:
        """Test string non-empty validation."""
        result = FlextUtilities.Validation.validate_string_not_empty("hello")
        assert result.is_success

        result = FlextUtilities.Validation.validate_string_not_empty("")
        assert result.is_failure

    def test_validate_string_length(self) -> None:
        """Test string length validation."""
        result = FlextUtilities.Validation.validate_string_length(
            "hello", min_length=1, max_length=10
        )
        assert result.is_success

        result = FlextUtilities.Validation.validate_string_length(
            "hello", min_length=10
        )
        assert result.is_failure

    def test_validate_url(self) -> None:
        """Test URL validation."""
        result = FlextUtilities.Validation.validate_url("https://example.com")
        assert result.is_success

        result = FlextUtilities.Validation.validate_url("invalid-url")
        assert result.is_failure

    def test_validate_email(self) -> None:
        """Test email validation."""
        result = FlextUtilities.Validation.validate_email("user@example.com")
        assert result.is_success
        assert result.value == "user@example.com"

        result = FlextUtilities.Validation.validate_email("invalid-email")
        assert result.is_failure

    def test_validate_port(self) -> None:
        """Test port validation."""
        result = FlextUtilities.Validation.validate_port(8080)
        assert result.is_success
        assert result.value == 8080

        result = FlextUtilities.Validation.validate_port("invalid")
        assert result.is_failure

    def test_validate_timeout_seconds(self) -> None:
        """Test timeout seconds validation."""
        result = FlextUtilities.Validation.validate_timeout_seconds(30.0)
        assert result.is_success

        result = FlextUtilities.Validation.validate_timeout_seconds(-1.0)
        assert result.is_failure

    def test_validate_retry_count(self) -> None:
        """Test retry count validation."""
        result = FlextUtilities.Validation.validate_retry_count(3)
        assert result.is_success

        result = FlextUtilities.Validation.validate_retry_count(-1)
        assert result.is_failure

    def test_validate_file_path(self) -> None:
        """Test file path validation."""
        result = FlextUtilities.Validation.validate_file_path("/tmp/test.txt")
        assert result.is_success

        result = FlextUtilities.Validation.validate_file_path("")
        assert result.is_failure

    def test_validate_positive_integer(self) -> None:
        """Test positive integer validation."""
        result = FlextUtilities.Validation.validate_positive_integer(42)
        assert result.is_success

        result = FlextUtilities.Validation.validate_positive_integer(-1)
        assert result.is_failure

    def test_validate_non_negative_integer(self) -> None:
        """Test non-negative integer validation."""
        result = FlextUtilities.Validation.validate_non_negative_integer(0)
        assert result.is_success

        result = FlextUtilities.Validation.validate_non_negative_integer(-1)
        assert result.is_failure


class TestTypeGuards:
    """Test FlextUtilities.TypeGuards namespace."""

    def test_is_string_non_empty(self) -> None:
        """Test non-empty string type guard."""
        assert FlextUtilities.TypeGuards.is_string_non_empty("hello") is True
        assert FlextUtilities.TypeGuards.is_string_non_empty("") is False
        assert FlextUtilities.TypeGuards.is_string_non_empty(123) is False

    def test_is_dict_non_empty(self) -> None:
        """Test non-empty dict type guard."""
        assert FlextUtilities.TypeGuards.is_dict_non_empty({"key": "value"}) is True
        assert FlextUtilities.TypeGuards.is_dict_non_empty({}) is False
        assert FlextUtilities.TypeGuards.is_dict_non_empty(None) is False

    def test_is_list_non_empty(self) -> None:
        """Test non-empty list type guard."""
        assert FlextUtilities.TypeGuards.is_list_non_empty([1, 2, 3]) is True
        assert FlextUtilities.TypeGuards.is_list_non_empty([]) is False
        assert FlextUtilities.TypeGuards.is_list_non_empty(None) is False


class TestGenerators:
    """Test FlextUtilities.Generators namespace."""

    def test_generate_id(self) -> None:
        """Test ID generation."""
        id1 = FlextUtilities.Generators.generate_id()
        id2 = FlextUtilities.Generators.generate_id()

        assert isinstance(id1, str)
        assert len(id1) > 0
        assert id1 != id2

    def test_generate_uuid(self) -> None:
        """Test UUID generation."""
        uuid = FlextUtilities.Generators.generate_uuid()
        assert isinstance(uuid, str)
        assert len(uuid) > 0

    def test_generate_timestamp(self) -> None:
        """Test timestamp generation."""
        ts = FlextUtilities.Generators.generate_timestamp()
        assert isinstance(ts, str)
        assert "T" in ts or "-" in ts  # ISO format check

    def test_generate_correlation_id(self) -> None:
        """Test correlation ID generation."""
        corr_id = FlextUtilities.Generators.generate_correlation_id()
        assert isinstance(corr_id, str)
        assert corr_id.startswith("corr_")

    def test_generate_short_id(self) -> None:
        """Test short ID generation."""
        short_id = FlextUtilities.Generators.generate_short_id(8)
        assert isinstance(short_id, str)
        assert len(short_id) == 8

    def test_generate_entity_id(self) -> None:
        """Test entity ID generation."""
        entity_id = FlextUtilities.Generators.generate_entity_id()
        assert isinstance(entity_id, str)
        assert len(entity_id) > 0

    def test_generate_command_id(self) -> None:
        """Test command ID generation."""
        cmd_id = FlextUtilities.Generators.generate_command_id()
        assert isinstance(cmd_id, str)
        assert cmd_id.startswith("cmd_")

    def test_generate_query_id(self) -> None:
        """Test query ID generation."""
        qry_id = FlextUtilities.Generators.generate_query_id()
        assert isinstance(qry_id, str)
        assert qry_id.startswith("qry_")


class TestTextProcessor:
    """Test FlextUtilities.TextProcessor namespace."""

    def test_clean_text(self) -> None:
        """Test text cleaning."""
        result = FlextUtilities.TextProcessor.clean_text("  hello   world  ")
        assert result.is_success
        assert result.value == "hello world"

    def test_truncate_text(self) -> None:
        """Test text truncation."""
        result = FlextUtilities.TextProcessor.truncate_text("hello world", max_length=8)
        assert result.is_success
        assert len(result.value) <= 8
        assert result.value.endswith("...")

    def test_safe_string(self) -> None:
        """Test safe string conversion."""
        result = FlextUtilities.TextProcessor.safe_string("  hello  ")
        assert result == "hello"

        result = FlextUtilities.TextProcessor.safe_string("")
        assert not result


class TestTypeConversions:
    """Test FlextUtilities.TypeConversions namespace."""

    def test_to_bool_from_string(self) -> None:
        """Test boolean conversion from string."""
        result = FlextUtilities.TypeConversions.to_bool(value="true")
        assert result.is_success
        assert result.value is True

        result = FlextUtilities.TypeConversions.to_bool(value="false")
        assert result.is_success
        assert result.value is False

    def test_to_bool_from_int(self) -> None:
        """Test boolean conversion from int."""
        result = FlextUtilities.TypeConversions.to_bool(value=1)
        assert result.is_success
        assert result.value is True

    def test_to_bool_from_bool(self) -> None:
        """Test boolean conversion from bool."""
        result = FlextUtilities.TypeConversions.to_bool(value=True)
        assert result.is_success
        assert result.value is True

    def test_to_bool_invalid(self) -> None:
        """Test boolean conversion with invalid value."""
        result = FlextUtilities.TypeConversions.to_bool(value="invalid")
        assert result.is_failure

    def test_to_int_from_string(self) -> None:
        """Test integer conversion from string."""
        result = FlextUtilities.TypeConversions.to_int("42")
        assert result.is_success
        assert result.value == 42

    def test_to_int_from_float(self) -> None:
        """Test integer conversion from float."""
        result = FlextUtilities.TypeConversions.to_int(math.pi)
        assert result.is_success
        assert result.value == 3

    def test_to_int_invalid(self) -> None:
        """Test integer conversion with invalid value."""
        result = FlextUtilities.TypeConversions.to_int("not-an-int")
        assert result.is_failure


class TestCache:
    """Test FlextUtilities.Cache namespace."""

    def test_normalize_component_dict(self) -> None:
        """Test component normalization for dict."""
        result = FlextUtilities.Cache.normalize_component({"a": 1, "b": 2})
        assert isinstance(result, dict)
        assert result["a"] == 1

    def test_sort_dict_keys(self) -> None:
        """Test dictionary key sorting."""
        data = {"z": 1, "a": 2, "m": 3}
        result = FlextUtilities.Cache.sort_dict_keys(data)
        assert isinstance(result, dict)
        keys = list(result.keys())
        # Check keys are sorted
        assert keys == sorted(keys)

    def test_generate_cache_key(self) -> None:
        """Test cache key generation."""
        key1 = FlextUtilities.Cache.generate_cache_key(None, None)
        FlextUtilities.Cache.generate_cache_key(None, None)
        assert isinstance(key1, str)
        assert len(key1) > 0

    def test_clear_object_cache(self) -> None:
        """Test clearing object caches."""
        obj = type("TestObj", (), {"_cache": {}})()
        result = FlextUtilities.Cache.clear_object_cache(obj)
        assert result.is_success

    def test_has_cache_attributes(self) -> None:
        """Test checking for cache attributes."""
        obj_with_cache = type("TestObj", (), {"_cache": {}})()
        assert FlextUtilities.Cache.has_cache_attributes(obj_with_cache) is True

        obj_without_cache = type("TestObj", (), {})()
        assert FlextUtilities.Cache.has_cache_attributes(obj_without_cache) is False


class TestReliability:
    """Test FlextUtilities.Reliability namespace."""

    def test_with_timeout_success(self) -> None:
        """Test timeout operation that succeeds."""

        def quick_op() -> FlextResult[str]:
            return FlextResult[str].ok("success")

        result = FlextUtilities.Reliability.with_timeout(quick_op, 5.0)
        assert result.is_success
        assert result.value == "success"

    def test_with_timeout_invalid_timeout(self) -> None:
        """Test timeout with invalid timeout value."""

        def op() -> FlextResult[str]:
            return FlextResult[str].ok("success")

        result = FlextUtilities.Reliability.with_timeout(op, -1.0)
        assert result.is_failure

    def test_retry_success_first_attempt(self) -> None:
        """Test retry that succeeds on first attempt."""

        def op() -> FlextResult[str]:
            return FlextResult[str].ok("success")

        result = FlextUtilities.Reliability.retry(op, max_attempts=3)
        assert result.is_success
        assert result.value == "success"

    def test_retry_eventual_success(self) -> None:
        """Test retry that eventually succeeds."""
        attempt_count = 0

        def flaky_op() -> FlextResult[str]:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                return FlextResult[str].fail("Temporary failure")
            return FlextResult[str].ok("success")

        result = FlextUtilities.Reliability.retry(
            flaky_op, max_attempts=3, delay_seconds=0.01
        )
        assert result.is_success
        assert attempt_count >= 2


class TestTypeChecker:
    """Test FlextUtilities.TypeChecker namespace."""

    def test_can_handle_message_type_with_object(self) -> None:
        """Test type checking with object type."""
        accepted = (object,)
        assert FlextUtilities.TypeChecker.can_handle_message_type(accepted, str) is True

    def test_can_handle_message_type_with_specific_type(self) -> None:
        """Test type checking with specific type."""
        accepted = (str,)
        assert FlextUtilities.TypeChecker.can_handle_message_type(accepted, str) is True
        assert (
            FlextUtilities.TypeChecker.can_handle_message_type(accepted, int) is False
        )

    def test_can_handle_message_type_empty_accepted(self) -> None:
        """Test type checking with no accepted types."""
        accepted: tuple[object, ...] = ()
        assert (
            FlextUtilities.TypeChecker.can_handle_message_type(accepted, str) is False
        )


class TestConfiguration:
    """Test FlextUtilities.Configuration namespace."""

    def test_get_parameter_from_dict(self) -> None:
        """Test parameter retrieval."""

        class MockConfig:
            def model_dump(self) -> dict[str, object]:
                return {"timeout": 30}

        config = MockConfig()
        value = FlextUtilities.Configuration.get_parameter(config, "timeout")
        assert value == 30

    def test_get_parameter_missing(self) -> None:
        """Test parameter retrieval for missing parameter."""

        class MockConfig:
            def model_dump(self) -> dict[str, object]:
                return {"timeout": 30}

        config = MockConfig()
        with pytest.raises(Exception):
            FlextUtilities.Configuration.get_parameter(config, "missing")


class TestExternalCommand:
    """Test FlextUtilities.run_external_command."""

    def test_run_external_command_echo(self) -> None:
        """Test running simple echo command."""
        result = FlextUtilities.run_external_command(
            ["echo", "hello"],
            capture_output=True,
            text=True,
        )
        assert result.is_success

    def test_run_external_command_empty_cmd(self) -> None:
        """Test with empty command."""
        result = FlextUtilities.run_external_command([], capture_output=True)
        assert result.is_failure

    def test_run_external_command_invalid_cmd(self) -> None:
        """Test with invalid command."""
        result = FlextUtilities.run_external_command(
            ["/nonexistent/command/path"],
            capture_output=True,
        )
        assert result.is_failure


class TestCorrelation:
    """Test FlextUtilities.Correlation namespace."""

    def test_generate_correlation_id(self) -> None:
        """Test correlation ID generation via Correlation namespace."""
        corr_id = FlextUtilities.Correlation.generate_correlation_id()
        assert isinstance(corr_id, str)
        assert len(corr_id) > 0

    def test_generate_iso_timestamp(self) -> None:
        """Test ISO timestamp generation via Correlation namespace."""
        ts = FlextUtilities.Correlation.generate_iso_timestamp()
        assert isinstance(ts, str)
        assert len(ts) > 0


__all__ = [
    "TestCache",
    "TestConfiguration",
    "TestCorrelation",
    "TestExternalCommand",
    "TestGenerators",
    "TestReliability",
    "TestTextProcessor",
    "TestTypeChecker",
    "TestTypeConversions",
    "TestTypeGuards",
    "TestValidation",
]
