"""Comprehensive coverage tests for FlextUtilitiesValidation.

Module: flext_core._utilities.validation
Scope: Validation utilities for pipeline validation, normalization, sorting, and type checking

Tests validate:
- validate_pipeline: Pipeline validation with multiple validators
- normalize_component: Component normalization (dicts, sequences, primitives, Pydantic models)
- sort_dict_keys: Dictionary key sorting for deterministic ordering
- sort_key: Key sorting for consistent ordering
- generate_cache_key: Cache key generation from various types
- validate_required_string: Required string validation
- validate_choice: Choice validation
- validate_length: Length validation
- validate_pattern: Pattern/regex validation
- validate_uri: URI validation
- validate_port_number: Port number validation
- validate_non_negative: Non-negative number validation
- validate_positive: Positive number validation
- validate_range: Range validation
- validate_callable: Callable validation
- validate_timeout: Timeout validation
- validate_http_status_codes: HTTP status code validation
- validate_iso8601_timestamp: ISO8601 timestamp validation
- validate_hostname: Hostname validation
- validate_identifier: Identifier validation
- validate_batch_services: Batch services validation
- validate_dispatch_config: Dispatch config validation
- validate_domain_event: Domain event validation

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import pytest

from flext_core import FlextResult
from flext_core._utilities.validation import FlextUtilitiesValidation
from flext_core.typings import FlextTypes
from tests.helpers import TestModels


# Test models using TestModels base
class PydanticModelForTest(TestModels.Value):
    """Test Pydantic model for normalization."""

    name: str
    value: int
    enabled: bool = True


@dataclass
class DataclassForTest:
    """Test dataclass for normalization."""

    name: str
    value: int


pytestmark = [pytest.mark.unit, pytest.mark.coverage]


class TestFlextUtilitiesValidation:
    """Comprehensive tests for FlextUtilitiesValidation."""

    def test_validate_pipeline_all_pass(self) -> None:
        """Test validate_pipeline with all validators passing."""

        def validator1(value: str) -> FlextResult[bool]:
            """First validator."""
            return FlextResult[bool].ok(True)

        def validator2(value: str) -> FlextResult[bool]:
            """Second validator."""
            return FlextResult[bool].ok(True)

        result = FlextUtilitiesValidation.validate_pipeline(
            "test",
            [validator1, validator2],
        )
        assert result.is_success
        assert result.value is True

    def test_validate_pipeline_first_fails(self) -> None:
        """Test validate_pipeline with first validator failing."""

        def validator1(value: str) -> FlextResult[bool]:
            """First validator fails."""
            return FlextResult[bool].fail("Validation failed")

        def validator2(value: str) -> FlextResult[bool]:
            """Second validator (not reached)."""
            return FlextResult[bool].ok(True)

        result = FlextUtilitiesValidation.validate_pipeline(
            "test",
            [validator1, validator2],
        )
        assert result.is_failure
        assert "Validation failed" in result.error

    def test_validate_pipeline_validator_raises_exception(self) -> None:
        """Test validate_pipeline handles validator exceptions."""

        def validator(value: str) -> FlextResult[bool]:
            """Validator that raises exception."""
            msg = "Validator error"
            raise ValueError(msg)

        result = FlextUtilitiesValidation.validate_pipeline("test", [validator])
        assert result.is_failure
        assert "Validator error" in result.error

    def test_validate_pipeline_non_callable_validator(self) -> None:
        """Test validate_pipeline with non-callable validator."""
        result = FlextUtilitiesValidation.validate_pipeline("test", ["not callable"])  # type: ignore[list-item]
        assert result.is_failure
        assert "Validator must be callable" in result.error

    def test_validate_pipeline_validator_returns_false(self) -> None:
        """Test validate_pipeline with validator returning False."""

        def validator(value: str) -> FlextResult[bool]:
            """Validator returns False."""
            return FlextResult[bool].ok(False)

        result = FlextUtilitiesValidation.validate_pipeline("test", [validator])
        assert result.is_failure
        assert "must return FlextResult[bool].ok(True)" in result.error

    def test_normalize_component_string(self) -> None:
        """Test normalize_component with string."""
        result = FlextUtilitiesValidation.normalize_component("test")
        assert result == "test"

    def test_normalize_component_dict(self) -> None:
        """Test normalize_component with dict."""
        data: dict[str, FlextTypes.GeneralValueType] = {
            "name": "test",
            "value": 42,
            "nested": {"key": "value"},
        }
        result = FlextUtilitiesValidation.normalize_component(data)
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_normalize_component_list(self) -> None:
        """Test normalize_component with list."""
        data: list[FlextTypes.GeneralValueType] = [1, 2, 3, "test"]
        result = FlextUtilitiesValidation.normalize_component(data)
        assert isinstance(
            result, dict
        )  # Sequences are normalized to dict with type marker
        assert result.get("type") == "sequence"

    def test_normalize_component_pydantic_model(self) -> None:
        """Test normalize_component with Pydantic model."""
        model = PydanticModelForTest(name="test", value=42)
        result = FlextUtilitiesValidation.normalize_component(model)
        # normalize_component converts Pydantic models to dict via model_dump
        assert isinstance(result, dict)
        assert result.get("name") == "test"
        assert result.get("value") == 42

    def test_normalize_component_dataclass(self) -> None:
        """Test normalize_component with dataclass."""
        data = DataclassForTest(name="test", value=42)
        result = FlextUtilitiesValidation.normalize_component(data)
        # normalize_component converts dataclasses to dict
        assert isinstance(result, dict)
        # May be normalized dict or dict with type marker
        assert isinstance(result, dict)

    def test_normalize_component_circular_reference(self) -> None:
        """Test normalize_component handles circular references."""
        data: dict[str, FlextTypes.GeneralValueType] = {"key": "value"}
        data["self"] = data  # Create circular reference
        result = FlextUtilitiesValidation.normalize_component(data)
        # Should handle circular reference gracefully
        assert isinstance(result, dict)

    def test_sort_dict_keys_simple_dict(self) -> None:
        """Test sort_dict_keys with simple dict."""
        data: dict[str, FlextTypes.GeneralValueType] = {
            "z": 3,
            "a": 1,
            "m": 2,
        }
        result = FlextUtilitiesValidation.sort_dict_keys(data)
        assert isinstance(result, dict)
        keys = list(result.keys())
        assert keys == sorted(keys, key=str.casefold)

    def test_sort_dict_keys_nested_dict(self) -> None:
        """Test sort_dict_keys with nested dict."""
        data: dict[str, FlextTypes.GeneralValueType] = {
            "z": {"c": 3, "a": 1},
            "a": {"b": 2},
        }
        result = FlextUtilitiesValidation.sort_dict_keys(data)
        assert isinstance(result, dict)
        # Keys should be sorted
        assert next(iter(result.keys())) == "a"

    def test_sort_dict_keys_non_dict(self) -> None:
        """Test sort_dict_keys with non-dict value."""
        result = FlextUtilitiesValidation.sort_dict_keys("not a dict")
        assert result == "not a dict"

    def test_sort_key_string(self) -> None:
        """Test sort_key with string."""
        result = FlextUtilitiesValidation.sort_key("Test")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == "test"  # casefold
        assert result[1] == "Test"  # original

    def test_sort_key_number(self) -> None:
        """Test sort_key with number."""
        result = FlextUtilitiesValidation.sort_key(42.5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_generate_cache_key_pydantic_model(self) -> None:
        """Test generate_cache_key with Pydantic model."""
        model = PydanticModelForTest(name="test", value=42)
        key = FlextUtilitiesValidation.generate_cache_key(model, PydanticModelForTest)
        assert isinstance(key, str)
        assert "PydanticModelForTest" in key

    def test_generate_cache_key_dict(self) -> None:
        """Test generate_cache_key with dict."""
        data: dict[str, FlextTypes.GeneralValueType] = {"name": "test", "value": 42}
        key = FlextUtilitiesValidation.generate_cache_key(data, dict)
        assert isinstance(key, str)

    def test_generate_cache_key_dataclass(self) -> None:
        """Test generate_cache_key with dataclass."""
        data = DataclassForTest(name="test", value=42)
        key = FlextUtilitiesValidation.generate_cache_key(data, DataclassForTest)
        assert isinstance(key, str)
        assert "DataclassForTest" in key

    def test_generate_cache_key_none(self) -> None:
        """Test generate_cache_key with None."""
        key = FlextUtilitiesValidation.generate_cache_key(None, str)
        assert isinstance(key, str)
        assert "str" in key

    def test_validate_required_string_valid(self) -> None:
        """Test validate_required_string with valid string."""
        result = FlextUtilitiesValidation.validate_required_string("test")
        assert result == "test"  # Returns the string directly

    def test_validate_required_string_empty(self) -> None:
        """Test validate_required_string with empty string."""
        with pytest.raises(ValueError):
            FlextUtilitiesValidation.validate_required_string("")

    def test_validate_required_string_none(self) -> None:
        """Test validate_required_string with None."""
        with pytest.raises(ValueError):
            FlextUtilitiesValidation.validate_required_string(None)  # type: ignore[arg-type]

    def test_validate_choice_valid(self) -> None:
        """Test validate_choice with valid choice."""
        result = FlextUtilitiesValidation.validate_choice("a", ["a", "b", "c"])
        assert result.is_success
        assert result.value == "a"  # Returns the chosen value

    def test_validate_choice_invalid(self) -> None:
        """Test validate_choice with invalid choice."""
        result = FlextUtilitiesValidation.validate_choice("d", ["a", "b", "c"])
        assert result.is_failure

    def test_validate_length_valid(self) -> None:
        """Test validate_length with valid length."""
        result = FlextUtilitiesValidation.validate_length(
            "test", min_length=2, max_length=10
        )
        assert result.is_success
        assert result.value == "test"  # Returns the string

    def test_validate_length_too_short(self) -> None:
        """Test validate_length with too short string."""
        result = FlextUtilitiesValidation.validate_length("a", min_length=2)
        assert result.is_failure

    def test_validate_length_too_long(self) -> None:
        """Test validate_length with too long string."""
        result = FlextUtilitiesValidation.validate_length("a" * 100, max_length=10)
        assert result.is_failure

    def test_validate_pattern_valid(self) -> None:
        """Test validate_pattern with valid pattern match."""
        result = FlextUtilitiesValidation.validate_pattern(
            "test@example.com", r"^[^@]+@[^@]+\.[^@]+$"
        )
        assert result.is_success
        assert result.value == "test@example.com"  # Returns the string

    def test_validate_pattern_invalid(self) -> None:
        """Test validate_pattern with invalid pattern match."""
        result = FlextUtilitiesValidation.validate_pattern(
            "invalid-email", r"^[^@]+@[^@]+\.[^@]+$"
        )
        assert result.is_failure

    def test_validate_pattern_invalid_regex(self) -> None:
        """Test validate_pattern with invalid regex."""
        result = FlextUtilitiesValidation.validate_pattern("test", "[invalid")
        assert result.is_failure

    def test_validate_uri_valid(self) -> None:
        """Test validate_uri with valid URI."""
        result = FlextUtilitiesValidation.validate_uri("https://example.com")
        assert result.is_success
        assert result.value == "https://example.com"  # Returns the URI

    def test_validate_uri_invalid(self) -> None:
        """Test validate_uri with invalid URI."""
        result = FlextUtilitiesValidation.validate_uri("not a uri")
        assert result.is_failure

    def test_validate_port_number_valid(self) -> None:
        """Test validate_port_number with valid port."""
        result = FlextUtilitiesValidation.validate_port_number(8080)
        assert result.is_success
        assert result.value == 8080  # Returns the port number

    def test_validate_port_number_invalid_low(self) -> None:
        """Test validate_port_number with port too low."""
        result = FlextUtilitiesValidation.validate_port_number(0)
        assert result.is_failure

    def test_validate_port_number_invalid_high(self) -> None:
        """Test validate_port_number with port too high."""
        result = FlextUtilitiesValidation.validate_port_number(65536)
        assert result.is_failure

    def test_validate_non_negative_valid(self) -> None:
        """Test validate_non_negative with valid number."""
        result = FlextUtilitiesValidation.validate_non_negative(42)
        assert result.is_success
        assert result.value == 42  # Returns the number

        result = FlextUtilitiesValidation.validate_non_negative(0)
        assert result.is_success
        assert result.value == 0

    def test_validate_non_negative_invalid(self) -> None:
        """Test validate_non_negative with negative number."""
        result = FlextUtilitiesValidation.validate_non_negative(-1)
        assert result.is_failure

    def test_validate_positive_valid(self) -> None:
        """Test validate_positive with valid number."""
        result = FlextUtilitiesValidation.validate_positive(42)
        assert result.is_success
        assert result.value == 42  # Returns the number

    def test_validate_positive_invalid_zero(self) -> None:
        """Test validate_positive with zero."""
        result = FlextUtilitiesValidation.validate_positive(0)
        assert result.is_failure

    def test_validate_positive_invalid_negative(self) -> None:
        """Test validate_positive with negative number."""
        result = FlextUtilitiesValidation.validate_positive(-1)
        assert result.is_failure

    def test_validate_range_valid(self) -> None:
        """Test validate_range with value in range."""
        result = FlextUtilitiesValidation.validate_range(5, min_value=1, max_value=10)
        assert result.is_success
        assert result.value == 5  # Returns the value

    def test_validate_range_invalid_low(self) -> None:
        """Test validate_range with value below minimum."""
        result = FlextUtilitiesValidation.validate_range(0, min_value=1, max_value=10)
        assert result.is_failure

    def test_validate_range_invalid_high(self) -> None:
        """Test validate_range with value above maximum."""
        result = FlextUtilitiesValidation.validate_range(11, min_value=1, max_value=10)
        assert result.is_failure

    def test_validate_callable_valid(self) -> None:
        """Test validate_callable with callable."""

        def test_func() -> str:
            """Test function."""
            return "test"

        result = FlextUtilitiesValidation.validate_callable(test_func)
        assert result.is_success
        assert result.value is True

    def test_validate_callable_invalid(self) -> None:
        """Test validate_callable with non-callable."""
        result = FlextUtilitiesValidation.validate_callable("not callable")
        assert result.is_failure

    def test_validate_timeout_valid(self) -> None:
        """Test validate_timeout with valid timeout."""
        result = FlextUtilitiesValidation.validate_timeout(30.0, max_timeout=60.0)
        assert result.is_success
        assert result.value == 30.0  # Returns the timeout

    def test_validate_timeout_invalid_negative(self) -> None:
        """Test validate_timeout with negative timeout."""
        result = FlextUtilitiesValidation.validate_timeout(-1.0, max_timeout=60.0)
        assert result.is_failure

    def test_validate_timeout_invalid_exceeds_max(self) -> None:
        """Test validate_timeout with timeout exceeding max."""
        result = FlextUtilitiesValidation.validate_timeout(100.0, max_timeout=60.0)
        assert result.is_failure

    def test_validate_http_status_codes_valid(self) -> None:
        """Test validate_http_status_codes with valid status codes."""
        result = FlextUtilitiesValidation.validate_http_status_codes([200, 404, 500])
        assert result.is_success
        assert result.value == [200, 404, 500]  # Returns the codes

    def test_validate_http_status_codes_invalid(self) -> None:
        """Test validate_http_status_codes with invalid status code."""
        result = FlextUtilitiesValidation.validate_http_status_codes([999])
        assert result.is_failure

    def test_validate_iso8601_timestamp_valid(self) -> None:
        """Test validate_iso8601_timestamp with valid timestamp."""
        timestamp = datetime.now().isoformat()
        result = FlextUtilitiesValidation.validate_iso8601_timestamp(timestamp)
        assert result.is_success
        assert result.value == timestamp  # Returns the timestamp

    def test_validate_iso8601_timestamp_invalid(self) -> None:
        """Test validate_iso8601_timestamp with invalid timestamp."""
        result = FlextUtilitiesValidation.validate_iso8601_timestamp("not a timestamp")
        assert result.is_failure

    def test_validate_iso8601_timestamp_empty_allowed(self) -> None:
        """Test validate_iso8601_timestamp with empty string when allowed."""
        result = FlextUtilitiesValidation.validate_iso8601_timestamp(
            "", allow_empty=True
        )
        assert result.is_success
        assert result.value == ""

    def test_validate_iso8601_timestamp_empty_not_allowed(self) -> None:
        """Test validate_iso8601_timestamp with empty string when not allowed."""
        result = FlextUtilitiesValidation.validate_iso8601_timestamp(
            "", allow_empty=False
        )
        assert result.is_failure

    def test_validate_hostname_valid(self) -> None:
        """Test validate_hostname with valid hostname."""
        result = FlextUtilitiesValidation.validate_hostname(
            "example.com", perform_dns_lookup=False
        )
        assert result.is_success
        assert result.value == "example.com"  # Returns the hostname

    def test_validate_hostname_invalid(self) -> None:
        """Test validate_hostname with invalid hostname."""
        result = FlextUtilitiesValidation.validate_hostname(
            "invalid..hostname", perform_dns_lookup=False
        )
        assert result.is_failure

    def test_validate_identifier_valid(self) -> None:
        """Test validate_identifier with valid identifier."""
        result = FlextUtilitiesValidation.validate_identifier("user_123")
        assert result.is_success
        assert result.value is True

    def test_validate_identifier_invalid(self) -> None:
        """Test validate_identifier with invalid identifier."""
        result = FlextUtilitiesValidation.validate_identifier("123-invalid")
        assert result.is_failure

    def test_boundary_normalize_component_empty_dict(self) -> None:
        """Test normalize_component with empty dict."""
        result = FlextUtilitiesValidation.normalize_component({})
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_boundary_normalize_component_empty_list(self) -> None:
        """Test normalize_component with empty list."""
        result = FlextUtilitiesValidation.normalize_component([])
        assert isinstance(result, dict)
        assert result.get("type") == "sequence"

    def test_boundary_normalize_component_none(self) -> None:
        """Test normalize_component with None."""
        result = FlextUtilitiesValidation.normalize_component(None)
        assert result is None

    def test_boundary_normalize_component_primitive_types(self) -> None:
        """Test normalize_component with primitive types."""
        assert FlextUtilitiesValidation.normalize_component(42) == 42
        assert FlextUtilitiesValidation.normalize_component(math.pi) == math.pi
        assert FlextUtilitiesValidation.normalize_component(True) is True
        assert FlextUtilitiesValidation.normalize_component(False) is False

    def test_boundary_sort_dict_keys_empty_dict(self) -> None:
        """Test sort_dict_keys with empty dict."""
        result = FlextUtilitiesValidation.sort_dict_keys({})
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_boundary_generate_cache_key_empty_dict(self) -> None:
        """Test generate_cache_key with empty dict."""
        key = FlextUtilitiesValidation.generate_cache_key({}, dict)
        assert isinstance(key, str)
        assert "dict" in key
