"""Comprehensive coverage tests for u.Validation.

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
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import cast

import pytest

from flext_core import m, r, t
from flext_tests import u


# Test models using FlextModelsEntity base
class PydanticModelForTest(m.Value):
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


class TestuValidation:
    """Comprehensive tests for u.Validation."""

    def test_validate_pipeline_all_pass(self) -> None:
        """Test validate_pipeline with all validators passing."""

        def validator1(value: str) -> r[bool]:
            """First validator."""
            return r[bool].ok(True)

        def validator2(value: str) -> r[bool]:
            """Second validator."""
            return r[bool].ok(True)

        result = u.Validation.validate_pipeline(
            "test",
            [validator1, validator2],
        )
        u.Tests.Result.assert_success_with_value(result, True)

    def test_validate_pipeline_first_fails(self) -> None:
        """Test validate_pipeline with first validator failing."""

        def validator1(value: str) -> r[bool]:
            """First validator fails."""
            return r[bool].fail("Validation failed")

        def validator2(value: str) -> r[bool]:
            """Second validator (not reached)."""
            return r[bool].ok(True)

        result = u.Validation.validate_pipeline(
            "test",
            [validator1, validator2],
        )
        u.Tests.Result.assert_result_failure_with_error(
            result,
            expected_error="Validation failed",
        )

    def test_validate_pipeline_validator_raises_exception(self) -> None:
        """Test validate_pipeline handles validator exceptions."""

        def validator(value: str) -> r[bool]:
            """Validator that raises exception."""
            msg = "Validator error"
            raise ValueError(msg)

        result = u.Validation.validate_pipeline("test", [validator])
        u.Tests.Result.assert_failure_with_error(
            result,
            "Validator error",
        )

    def test_validate_pipeline_non_callable_validator(self) -> None:
        """Test validate_pipeline with non-callable validator."""
        # Type narrowing: validate_pipeline expects list[Callable] but we test with str
        # This tests runtime error handling - mypy will complain but runtime works
        non_callable_validators: list[object] = ["not callable"]
        result = u.Validation.validate_pipeline(
            "test",
            cast("list[Callable[[str], r[bool]]]", non_callable_validators),
        )
        u.Tests.Result.assert_failure_with_error(
            result,
            "Validator must be callable",
        )

    def test_validate_pipeline_validator_returns_false(self) -> None:
        """Test validate_pipeline with validator returning False."""

        def validator(value: str) -> r[bool]:
            """Validator returns False."""
            return r[bool].ok(False)

        result = u.Validation.validate_pipeline("test", [validator])
        u.Tests.Result.assert_failure_with_error(
            result,
            "must return r[bool].ok(True)",
        )

    def test_normalize_component_string(self) -> None:
        """Test normalize_component with string."""
        result = u.Validation.normalize_component("test")
        assert result == "test"

    def test_normalize_component_dict(self) -> None:
        """Test normalize_component with dict."""
        data: dict[str, t.GeneralValueType] = {
            "name": "test",
            "value": 42,
            "nested": {"key": "value"},
        }
        result = u.Validation.normalize_component(data)
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42

    def test_normalize_component_list(self) -> None:
        """Test normalize_component with list."""
        data: list[t.GeneralValueType] = [1, 2, 3, "test"]
        result = u.Validation.normalize_component(data)
        assert isinstance(
            result,
            dict,
        )  # Sequences are normalized to dict with type marker
        assert result.get("type") == "sequence"

    def test_normalize_component_pydantic_model(self) -> None:
        """Test normalize_component with Pydantic model."""
        model = PydanticModelForTest(name="test", value=42)
        # Convert BaseModel to t.GeneralValueType via model_dump()
        model_dict: t.GeneralValueType = model.model_dump()
        result = u.Validation.normalize_component(model_dict)
        # normalize_component converts Pydantic models to dict via model_dump
        assert isinstance(result, dict)
        assert result.get("name") == "test"
        assert result.get("value") == 42

    def test_normalize_component_dataclass(self) -> None:
        """Test normalize_component with dataclass."""
        data = DataclassForTest(name="test", value=42)
        # Convert dataclass to dict (t.GeneralValueType) for type compatibility
        data_dict: t.GeneralValueType = asdict(data)
        result = u.Validation.normalize_component(data_dict)
        # normalize_component converts dataclasses to dict
        assert isinstance(result, dict)
        # May be normalized dict or dict with type marker
        assert isinstance(result, dict)

    def test_normalize_component_dataclass_direct(self) -> None:
        """Test normalize_component with dataclass instance directly."""
        data = DataclassForTest(name="test", value=42)
        # Pass dataclass instance directly - casts required for type checker
        # but runtime supports it
        result = u.Validation.normalize_component(cast("t.GeneralValueType", data))
        assert isinstance(result, dict)
        assert result.get("type") == "dataclass"
        assert result["data"]["name"] == "test"

    def test_normalize_component_pydantic_exception(self) -> None:
        """Test normalize_component with failing model_dump."""

        class FailingModel:
            def model_dump(self) -> dict[str, int]:
                msg = "Dump failed"
                raise ValueError(msg)

            def __str__(self) -> str:
                return "FailingModelString"

        obj = FailingModel()
        # Should fallback to string representation
        result = u.Validation.normalize_component(cast("t.GeneralValueType", obj))
        assert result == "FailingModelString"

    def test_normalize_component_pydantic_non_dict(self) -> None:
        """Test normalize_component with model_dump returning non-dict."""

        class NonDictModel:
            def model_dump(self) -> str:
                return "not a dict"

            def __str__(self) -> str:
                return "NonDictModelString"

        obj = NonDictModel()
        # Should fallback to string representation
        result = u.Validation.normalize_component(cast("t.GeneralValueType", obj))
        assert result == "NonDictModelString"

    def test_normalize_component_circular_reference_list(self) -> None:
        """Test normalize_component handles circular references in list."""
        data: list[t.GeneralValueType] = []
        data.append(data)  # Create circular reference

        result = u.Validation.normalize_component(data)
        assert isinstance(result, dict)
        assert result.get("type") == "sequence"
        items = cast("list[t.GeneralValueType]", result.get("data"))
        assert len(items) == 1
        assert isinstance(items[0], dict)
        assert items[0].get("type") == "circular_reference"

    def test_normalize_component_pydantic_model_direct(self) -> None:
        """Test normalize_component with Pydantic model instance directly."""
        model = PydanticModelForTest(name="test", value=42)
        # Convert BaseModel to t.GeneralValueType via model_dump() before passing
        model_dict: t.GeneralValueType = model.model_dump()
        result = u.Validation.normalize_component(model_dict)
        assert isinstance(result, dict)
        assert result.get("name") == "test"
        assert result.get("value") == 42

    def test_nested_classes_wrappers_failure(self) -> None:
        """Test nested class wrappers failure cases."""
        # String.validate_required_string failure
        res_req = u.Validation.String.validate_required_string("")
        assert res_req.is_failure
        assert "cannot be empty" in str(res_req.error)

    def test_nested_classes_wrappers(self) -> None:
        """Test nested class wrappers to ensure coverage."""
        # Network
        res_uri = u.Validation.Network.validate_uri("https://example.com")
        assert res_uri.is_success

        res_port = u.Validation.Network.validate_port_number(8080)
        assert res_port.is_success

        res_host = u.Validation.Network.validate_hostname("localhost")
        assert res_host.is_success

        # String
        res_req = u.Validation.String.validate_required_string("test")
        assert res_req.is_success
        assert res_req.value == "test"

        res_choice = u.Validation.String.validate_choice("a", {"a", "b"})
        assert res_choice.is_success

        res_len = u.Validation.String.validate_length("test", min_length=1)
        assert res_len.is_success

        res_pat = u.Validation.String.validate_pattern("test", r"test")
        assert res_pat.is_success

        # Numeric
        res_non_neg = u.Validation.Numeric.validate_non_negative(0)
        assert res_non_neg.is_success

        res_pos = u.Validation.Numeric.validate_positive(1)
        assert res_pos.is_success

        res_range = u.Validation.Numeric.validate_range(5, 1, 10)
        assert res_range.is_success

    def test_normalize_component_circular_reference(self) -> None:
        """Test normalize_component handles circular references."""
        data: dict[str, t.GeneralValueType] = {"key": "value"}
        data["self"] = data  # Create circular reference
        result = u.Validation.normalize_component(data)
        # Should handle circular reference gracefully
        assert isinstance(result, dict)

    def test_sort_dict_keys_simple_dict(self) -> None:
        """Test sort_dict_keys with simple dict."""
        data: dict[str, t.GeneralValueType] = {
            "z": 3,
            "a": 1,
            "m": 2,
        }
        result = u.Validation.sort_dict_keys(data)
        assert isinstance(result, dict)
        keys = list(result.keys())
        assert keys == sorted(keys, key=str.casefold)

    def test_sort_dict_keys_nested_dict(self) -> None:
        """Test sort_dict_keys with nested dict."""
        data: dict[str, t.GeneralValueType] = {
            "z": {"c": 3, "a": 1},
            "a": {"b": 2},
        }
        result = u.Validation.sort_dict_keys(data)
        assert isinstance(result, dict)
        # Keys should be sorted
        assert next(iter(result.keys())) == "a"

    def test_sort_dict_keys_non_dict(self) -> None:
        """Test sort_dict_keys with non-dict value."""
        result = u.Validation.sort_dict_keys("not a dict")
        assert result == "not a dict"

    def test_sort_key_string(self) -> None:
        """Test sort_key with string."""
        result = u.Validation.sort_key("Test")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == "test"  # casefold
        assert result[1] == "Test"  # original

    def test_sort_key_number(self) -> None:
        """Test sort_key with number."""
        result = u.Validation.sort_key(42.5)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_generate_cache_key_pydantic_model(self) -> None:
        """Test generate_cache_key with Pydantic model."""
        model = PydanticModelForTest(name="test", value=42)
        # Convert BaseModel to t.GeneralValueType via model_dump()
        model_dict: t.GeneralValueType = model.model_dump()
        # Business Rule: generate_cache_key accepts model dict and type for cache key generation
        # The type parameter determines the type name in the cache key

        # Pass str as command_type - the key format is "{type_name}_{hash}"
        key = u.Validation.generate_cache_key(model_dict, str)
        assert isinstance(key, str)
        # Key format is "str_{hash}" when using str as command_type
        assert key.startswith("str_")

    def test_generate_cache_key_dict(self) -> None:
        """Test generate_cache_key with dict."""
        data: dict[str, t.GeneralValueType] = {"name": "test", "value": 42}
        key = u.Validation.generate_cache_key(data, dict)
        assert isinstance(key, str)

    def test_generate_cache_key_dataclass(self) -> None:
        """Test generate_cache_key with dataclass."""
        data = DataclassForTest(name="test", value=42)
        # Convert dataclass to t.GeneralValueType via asdict()
        data_dict: t.GeneralValueType = asdict(data)
        # Type narrowing: generate_cache_key accepts type hints, cast for mypy
        # DataclassForTest is compatible at runtime but mypy expects specific types
        key = u.Validation.generate_cache_key(
            data_dict,
            cast("type[str]", DataclassForTest),
        )
        assert isinstance(key, str)
        assert "DataclassForTest" in key

    def test_generate_cache_key_none(self) -> None:
        """Test generate_cache_key with None."""
        key = u.Validation.generate_cache_key(None, str)
        assert isinstance(key, str)
        assert "str" in key

    def test_validate_required_string_valid(self) -> None:
        """Test validate_required_string with valid string."""
        result = u.Validation.validate_required_string("test")
        assert result == "test"  # Returns the string directly

    def test_validate_required_string_empty(self) -> None:
        """Test validate_required_string with empty string."""
        with pytest.raises(ValueError):
            u.Validation.validate_required_string("")

    def test_validate_required_string_none(self) -> None:
        """Test validate_required_string with None."""
        # Business Rule: validate_required_string raises ValueError for None
        # None is not a valid string value - intentionally passed to test error handling
        # Function signature accepts str | None, but raises ValueError for None
        with pytest.raises(ValueError):
            u.Validation.validate_required_string(None)

    @pytest.mark.parametrize(
        ("choice", "choices", "should_succeed", "expected_value"),
        [
            ("a", {"a", "b", "c"}, True, "a"),
            ("b", {"a", "b", "c"}, True, "b"),
            ("d", {"a", "b", "c"}, False, None),
        ],
    )
    def test_validate_choice(
        self,
        choice: str,
        choices: set[str],
        should_succeed: bool,
        expected_value: str | None,
    ) -> None:
        """Test validate_choice with valid and invalid choices."""
        result = u.Validation.validate_choice(choice, choices)
        if should_succeed:
            assert expected_value is not None
            u.Tests.Result.assert_success_with_value(
                result,
                expected_value,
            )
        else:
            u.Tests.Result.assert_result_failure(result)

    @pytest.mark.parametrize(
        ("value", "min_length", "max_length", "should_succeed", "expected_value"),
        [
            ("test", 2, 10, True, "test"),
            ("ab", 2, 10, True, "ab"),
            ("a", 2, 10, False, None),  # Too short
            ("a" * 100, 2, 10, False, None),  # Too long
        ],
    )
    def test_validate_length(
        self,
        value: str,
        min_length: int | None,
        max_length: int | None,
        should_succeed: bool,
        expected_value: str | None,
    ) -> None:
        """Test validate_length with various length constraints."""
        # Type narrowing: validate_length accepts keyword args, unpack dict explicitly
        if min_length is not None and max_length is not None:
            result = u.Validation.validate_length(
                value,
                min_length=min_length,
                max_length=max_length,
            )
        elif min_length is not None:
            result = u.Validation.validate_length(
                value,
                min_length=min_length,
            )
        elif max_length is not None:
            result = u.Validation.validate_length(
                value,
                max_length=max_length,
            )
        else:
            result = u.Validation.validate_length(value)
        if should_succeed:
            assert expected_value is not None
            u.Tests.Result.assert_success_with_value(
                result,
                expected_value,
            )
        else:
            u.Tests.Result.assert_result_failure(result)

    @pytest.mark.parametrize(
        ("value", "pattern", "should_succeed", "expected_value"),
        [
            ("test@example.com", r"^[^@]+@[^@]+\.[^@]+$", True, "test@example.com"),
            ("invalid-email", r"^[^@]+@[^@]+\.[^@]+$", False, None),
            ("test", "[invalid", False, None),  # Invalid regex
        ],
    )
    def test_validate_pattern(
        self,
        value: str,
        pattern: str,
        should_succeed: bool,
        expected_value: str | None,
    ) -> None:
        """Test validate_pattern with valid and invalid patterns."""
        result = u.Validation.validate_pattern(value, pattern)
        if should_succeed:
            assert expected_value is not None
            u.Tests.Result.assert_success_with_value(
                result,
                expected_value,
            )
        else:
            u.Tests.Result.assert_result_failure(result)

    @pytest.mark.parametrize(
        ("uri", "should_succeed", "expected_value"),
        [
            ("https://example.com", True, "https://example.com"),
            ("http://localhost:8000", True, "http://localhost:8000"),
            ("not a uri", False, None),
        ],
    )
    def test_validate_uri(
        self,
        uri: str,
        should_succeed: bool,
        expected_value: str | None,
    ) -> None:
        """Test validate_uri with valid and invalid URIs."""
        result = u.Validation.validate_uri(uri)
        if should_succeed:
            assert expected_value is not None
            u.Tests.Result.assert_success_with_value(
                result,
                expected_value,
            )
        else:
            u.Tests.Result.assert_result_failure(result)

    @pytest.mark.parametrize(
        ("port", "should_succeed", "expected_value"),
        [
            (8080, True, 8080),
            (1, True, 1),
            (65535, True, 65535),
            (0, False, None),  # Too low
            (65536, False, None),  # Too high
        ],
    )
    def test_validate_port_number(
        self,
        port: int,
        should_succeed: bool,
        expected_value: int | None,
    ) -> None:
        """Test validate_port_number with valid and invalid ports."""
        result = u.Validation.validate_port_number(port)
        if should_succeed:
            assert expected_value is not None
            u.Tests.Result.assert_success_with_value(
                result,
                expected_value,
            )
        else:
            u.Tests.Result.assert_result_failure(result)

    @pytest.mark.parametrize(
        ("value", "should_succeed", "expected_value"),
        [
            (42, True, 42),
            (0, True, 0),
            (-1, False, None),
        ],
    )
    def test_validate_non_negative(
        self,
        value: int,
        should_succeed: bool,
        expected_value: int | None,
    ) -> None:
        """Test validate_non_negative with valid and invalid numbers."""
        result = u.Validation.validate_non_negative(value)
        if should_succeed:
            assert expected_value is not None
            u.Tests.Result.assert_success_with_value(
                result,
                expected_value,
            )
        else:
            u.Tests.Result.assert_result_failure(result)

    @pytest.mark.parametrize(
        ("value", "should_succeed", "expected_value"),
        [
            (42, True, 42),
            (1, True, 1),
            (0, False, None),
            (-1, False, None),
        ],
    )
    def test_validate_positive(
        self,
        value: int,
        should_succeed: bool,
        expected_value: int | None,
    ) -> None:
        """Test validate_positive with valid and invalid numbers."""
        result = u.Validation.validate_positive(value)
        if should_succeed:
            assert expected_value is not None
            u.Tests.Result.assert_success_with_value(
                result,
                expected_value,
            )
        else:
            u.Tests.Result.assert_result_failure(result)

    @pytest.mark.parametrize(
        ("value", "min_value", "max_value", "should_succeed", "expected_value"),
        [
            (5, 1, 10, True, 5),
            (1, 1, 10, True, 1),
            (10, 1, 10, True, 10),
            (0, 1, 10, False, None),  # Too low
            (11, 1, 10, False, None),  # Too high
        ],
    )
    def test_validate_range(
        self,
        value: int,
        min_value: int,
        max_value: int,
        should_succeed: bool,
        expected_value: int | None,
    ) -> None:
        """Test validate_range with valid and invalid ranges."""
        result = u.Validation.validate_range(
            value,
            min_value=min_value,
            max_value=max_value,
        )
        if should_succeed:
            assert expected_value is not None
            u.Tests.Result.assert_success_with_value(
                result,
                expected_value,
            )
        else:
            u.Tests.Result.assert_result_failure(result)

    def test_validate_callable_valid(self) -> None:
        """Test validate_callable with callable."""

        def test_func() -> str:
            """Test function."""
            return "test"

        # validate_callable accepts callables at runtime but expects t.GeneralValueType

        result = u.Validation.validate_callable(cast("t.GeneralValueType", test_func))
        u.Tests.Result.assert_result_success(result)
        u.Tests.Result.assert_success_with_value(result, True)

    def test_validate_callable_invalid(self) -> None:
        """Test validate_callable with non-callable."""
        result = u.Validation.validate_callable("not callable")
        u.Tests.Result.assert_result_failure(result)

    def test_validate_timeout_valid(self) -> None:
        """Test validate_timeout with valid timeout."""
        result = u.Validation.validate_timeout(30.0, max_timeout=60.0)
        u.Tests.Result.assert_result_success(result)
        u.Tests.Result.assert_success_with_value(
            result,
            30.0,  # Returns the timeout
        )

    def test_validate_timeout_invalid_negative(self) -> None:
        """Test validate_timeout with negative timeout."""
        result = u.Validation.validate_timeout(-1.0, max_timeout=60.0)
        u.Tests.Result.assert_result_failure(result)

    def test_validate_timeout_invalid_exceeds_max(self) -> None:
        """Test validate_timeout with timeout exceeding max."""
        result = u.Validation.validate_timeout(100.0, max_timeout=60.0)
        u.Tests.Result.assert_result_failure(result)

    def test_validate_http_status_codes_valid(self) -> None:
        """Test validate_http_status_codes with valid status codes."""
        result = u.Validation.validate_http_status_codes([200, 404, 500])
        u.Tests.Result.assert_result_success(result)
        u.Tests.Result.assert_success_with_value(
            result,
            [200, 404, 500],  # Returns the codes
        )

    def test_validate_http_status_codes_invalid(self) -> None:
        """Test validate_http_status_codes with invalid status code."""
        result = u.Validation.validate_http_status_codes([999])
        u.Tests.Result.assert_result_failure(result)

    def test_validate_iso8601_timestamp_valid(self) -> None:
        """Test validate_iso8601_timestamp with valid timestamp."""
        timestamp = datetime.now(UTC).isoformat()
        result = u.Validation.validate_iso8601_timestamp(timestamp)
        u.Tests.Result.assert_result_success(result)
        u.Tests.Result.assert_success_with_value(
            result,
            timestamp,  # Returns the timestamp
        )

    def test_validate_iso8601_timestamp_invalid(self) -> None:
        """Test validate_iso8601_timestamp with invalid timestamp."""
        result = u.Validation.validate_iso8601_timestamp("not a timestamp")
        u.Tests.Result.assert_result_failure(result)

    def test_validate_iso8601_timestamp_empty_allowed(self) -> None:
        """Test validate_iso8601_timestamp with empty string when allowed."""
        result = u.Validation.validate_iso8601_timestamp(
            "",
            allow_empty=True,
        )
        u.Tests.Result.assert_result_success(result)
        u.Tests.Result.assert_success_with_value(result, "")

    def test_validate_iso8601_timestamp_empty_not_allowed(self) -> None:
        """Test validate_iso8601_timestamp with empty string when not allowed."""
        result = u.Validation.validate_iso8601_timestamp(
            "",
            allow_empty=False,
        )
        u.Tests.Result.assert_result_failure(result)

    def test_validate_hostname_valid(self) -> None:
        """Test validate_hostname with valid hostname."""
        # validate_hostname is in Network nested class
        result = u.Validation.Network.validate_hostname(
            "example.com",
        )
        u.Tests.Result.assert_result_success(result)
        u.Tests.Result.assert_success_with_value(
            result,
            "example.com",  # Returns the hostname
        )

    def test_validate_hostname_invalid(self) -> None:
        """Test validate_hostname with invalid hostname."""
        # validate_hostname is in Network nested class
        result = u.Validation.Network.validate_hostname(
            "invalid..hostname",
        )
        u.Tests.Result.assert_result_failure(result)

    def test_validate_identifier_valid(self) -> None:
        """Test validate_identifier with valid identifier."""
        result = u.Validation.validate_identifier("user_123")
        u.Tests.Result.assert_result_success(result)
        u.Tests.Result.assert_success_with_value(
            result,
            "user_123",  # Returns normalized string, not True
        )

    def test_validate_identifier_invalid(self) -> None:
        """Test validate_identifier with invalid identifier."""
        result = u.Validation.validate_identifier("123-invalid")
        u.Tests.Result.assert_result_failure(result)

    def test_boundary_normalize_component_empty_dict(self) -> None:
        """Test normalize_component with empty dict."""
        result = u.Validation.normalize_component({})
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_boundary_normalize_component_empty_list(self) -> None:
        """Test normalize_component with empty list."""
        result = u.Validation.normalize_component([])
        assert isinstance(result, dict)
        assert result.get("type") == "sequence"

    def test_boundary_normalize_component_none(self) -> None:
        """Test normalize_component with None."""
        result = u.Validation.normalize_component(None)
        assert result is None

    def test_boundary_normalize_component_primitive_types(self) -> None:
        """Test normalize_component with primitive types."""
        assert u.Validation.normalize_component(42) == 42
        assert u.Validation.normalize_component(math.pi) == math.pi
        assert u.Validation.normalize_component(True) is True
        assert u.Validation.normalize_component(False) is False

    def test_boundary_sort_dict_keys_empty_dict(self) -> None:
        """Test sort_dict_keys with empty dict."""
        result = u.Validation.sort_dict_keys({})
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_boundary_generate_cache_key_empty_dict(self) -> None:
        """Test generate_cache_key with empty dict."""
        key = u.Validation.generate_cache_key({}, dict)
        assert isinstance(key, str)
        assert "dict" in key
