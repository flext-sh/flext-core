"""Comprehensive coverage tests for FlextUtilities.

Module: flext_core.utilities.FlextUtilities
Scope: Type guards, generators, text processing, caching, reliability, validation, type checking

Tests validate:
- Type guards (string, dict, list non-empty checks)
- ID/timestamp generation (multiple generator types)
- Text processing (cleaning, truncation, safe string conversion)
- Caching utilities (normalization, key generation, cleanup)
- Reliability patterns (timeout, retry)
- Type checking for message handlers
- Validation utilities (sort_key, normalize_component, validate_pipeline)
- Cache utilities (sort_dict_keys, cache operations)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

import pytest
from pydantic import BaseModel

from flext_core import FlextConfig, FlextResult, FlextUtilities

# =========================================================================
# Test Data and Enums
# =========================================================================


class UtilityOperationType(StrEnum):
    """Utility operation types for parametrization."""

    TYPE_GUARD_STRING = "type_guard_string"
    TYPE_GUARD_DICT = "type_guard_dict"
    TYPE_GUARD_LIST = "type_guard_list"
    ID_GENERATION = "id_generation"
    TIMESTAMP_GENERATION = "timestamp_generation"
    TEXT_CLEANING = "text_cleaning"
    TEXT_TRUNCATION = "text_truncation"
    CACHE_NORMALIZE = "cache_normalize"
    CACHE_SORT = "cache_sort"
    CACHE_KEY = "cache_key"
    VALIDATION = "validation"


@dataclass(frozen=True, slots=True)
class UtilityTestCase:
    """Test case for utility operations."""

    operation: UtilityOperationType
    input_data: object = None
    expected_result: object = None
    should_succeed: bool = True


class UtilityScenarios:
    """Centralized utility test scenarios."""

    TYPE_GUARD_STRING_CASES: ClassVar[list[tuple[str, object, bool]]] = [
        ("string_empty", "", False),
        ("string_valid", "test", True),
        ("string_none", None, False),
        ("string_number", 123, False),
    ]

    TYPE_GUARD_DICT_CASES: ClassVar[list[tuple[str, object, bool]]] = [
        ("dict_empty", {}, False),
        ("dict_valid", {"a": 1}, True),
        ("dict_none", None, False),
        ("dict_string", "not_dict", False),
    ]

    TYPE_GUARD_LIST_CASES: ClassVar[list[tuple[str, object, bool]]] = [
        ("list_empty", [], False),
        ("list_valid", [1, 2, 3], True),
        ("list_none", None, False),
        ("list_string", "not_list", False),
    ]

    ID_GENERATOR_CASES: ClassVar[list[str]] = [
        "generate_id",
        "generate_iso_timestamp",
        "generate_correlation_id",
        "generate_entity_id",
        "generate_transaction_id",
        "generate_saga_id",
        "generate_event_id",
    ]

    TEXT_PROCESSOR_CASES: ClassVar[list[tuple[str, str, int]]] = [
        ("clean_spaces", "  Test  Text  ", 2),
        ("clean_newlines", "a\nb\nc", 1),
        ("truncate_long", "VeryLongText", 5),
        ("truncate_short", "Short", 20),
    ]

    CACHE_NORMALIZE_CASES: ClassVar[list[tuple[object, type]]] = [
        ({"a": 1, "b": 2}, dict),
        ([1, 2, 3], list),
        ("string", str),
        (42, int),
    ]

    @staticmethod
    def create_test_service() -> object:
        """Create test service object."""

        class TestService:
            _service_name = "test_service"

        return TestService()

    @staticmethod
    def create_test_model() -> BaseModel:
        """Create test Pydantic model."""

        class TestModel(BaseModel):
            name: str
            value: int

        return TestModel(name="test", value=42)


# =========================================================================
# Test Suite - FlextUtilities Comprehensive Coverage
# =========================================================================


class TestFlextUtilities:
    """Unified test suite for FlextUtilities - ALL REAL FUNCTIONALITY."""

    # =====================================================================
    # Type Guards Tests
    # =====================================================================

    @pytest.mark.parametrize(
        ("description", "value", "expected"),
        UtilityScenarios.TYPE_GUARD_STRING_CASES,
    )
    def test_type_guard_string(
        self, description: str, value: object, expected: bool
    ) -> None:
        """Test string type guards."""
        result = FlextUtilities.TypeGuards.is_string_non_empty(value)
        assert result is expected

    @pytest.mark.parametrize(
        ("description", "value", "expected"),
        UtilityScenarios.TYPE_GUARD_DICT_CASES,
    )
    def test_type_guard_dict(
        self, description: str, value: object, expected: bool
    ) -> None:
        """Test dict type guards."""
        result = FlextUtilities.TypeGuards.is_dict_non_empty(value)
        assert result is expected

    @pytest.mark.parametrize(
        ("description", "value", "expected"),
        UtilityScenarios.TYPE_GUARD_LIST_CASES,
    )
    def test_type_guard_list(
        self, description: str, value: object, expected: bool
    ) -> None:
        """Test list type guards."""
        result = FlextUtilities.TypeGuards.is_list_non_empty(value)
        assert result is expected

    # =====================================================================
    # Generators Tests
    # =====================================================================

    @pytest.mark.parametrize("method_name", UtilityScenarios.ID_GENERATOR_CASES)
    def test_generators_operations(self, method_name: str) -> None:
        """Test ID and timestamp generation operations."""
        method = getattr(FlextUtilities.Generators, method_name)
        result = method()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_generators_short_id_with_length(self) -> None:
        """Test short ID generation with custom length."""
        short = FlextUtilities.Generators.generate_short_id(length=5)
        medium = FlextUtilities.Generators.generate_short_id(length=10)
        long = FlextUtilities.Generators.generate_short_id(length=20)

        assert len(short) == 5
        assert len(medium) == 10
        assert len(long) == 20

    def test_generators_batch_id(self) -> None:
        """Test batch ID generation."""
        batch_id = FlextUtilities.Generators.generate_batch_id(100)
        assert isinstance(batch_id, str)
        assert len(batch_id) > 0

    def test_generators_correlation_id_with_context(self) -> None:
        """Test correlation ID with context."""
        corr_id = FlextUtilities.Generators.generate_correlation_id_with_context(
            "test_ctx"
        )
        assert isinstance(corr_id, str)
        assert "test_ctx" in corr_id

    def test_generators_uniqueness(self) -> None:
        """Test generator uniqueness."""
        id1 = FlextUtilities.Generators.generate_id()
        id2 = FlextUtilities.Generators.generate_id()
        assert id1 != id2

    # =====================================================================
    # Text Processor Tests
    # =====================================================================

    def test_text_processor_clean_text(self) -> None:
        """Test text cleaning - returns str directly."""
        result = FlextUtilities.TextProcessor.clean_text("  Test\n\r\tText  ")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_text_processor_truncate(self) -> None:
        """Test text truncation - returns FlextResult[str]."""
        result = FlextUtilities.TextProcessor.truncate_text(
            "VeryLongText", max_length=5
        )
        assert result.is_success
        assert len(result.value) <= 8  # 5 + "..." (3)

    def test_text_processor_safe_string_success(self) -> None:
        """Test safe string with valid input - returns str directly."""
        result = FlextUtilities.TextProcessor.safe_string("valid")
        assert isinstance(result, str)
        assert result == "valid"

    def test_text_processor_safe_string_empty(self) -> None:
        """Test safe string with empty - raises ValueError."""
        with pytest.raises(ValueError):
            FlextUtilities.TextProcessor.safe_string("")

    # =====================================================================
    # Cache Tests
    # =====================================================================

    def test_cache_normalize_component(self) -> None:
        """Test cache component normalization."""
        normalized_dict = FlextUtilities.Cache.normalize_component({"b": 2, "a": 1})
        assert isinstance(normalized_dict, (dict, str))

        normalized_list = FlextUtilities.Cache.normalize_component([1, 2, 3])
        assert isinstance(normalized_list, list)

    def test_cache_sort_dict_keys(self) -> None:
        """Test dictionary key sorting."""
        data = {"z": 1, "a": 2, "m": 3}
        result = FlextUtilities.Cache.sort_dict_keys(data)
        assert isinstance(result, dict)
        keys = list(result.keys())
        assert keys == ["a", "m", "z"]

    def test_cache_generate_key(self) -> None:
        """Test cache key generation."""
        key1 = FlextUtilities.Cache.generate_cache_key("arg1", "arg2")
        key2 = FlextUtilities.Cache.generate_cache_key("arg1", "arg2")
        assert key1 == key2
        assert isinstance(key1, str)

    def test_cache_clear_object_cache(self) -> None:
        """Test clearing object cache."""
        obj = {"test": "data"}
        result = FlextUtilities.Cache.clear_object_cache(obj)
        assert result.is_success

    def test_cache_has_attributes_true(self) -> None:
        """Test detecting cache attributes - true case."""

        class MockWithCache:
            _cache: ClassVar[dict[str, object]] = {}

        obj = MockWithCache()
        assert FlextUtilities.Cache.has_cache_attributes(obj) is True

    def test_cache_has_attributes_false(self) -> None:
        """Test detecting cache attributes - false case."""

        class MockNoCache:
            pass

        obj = MockNoCache()
        assert FlextUtilities.Cache.has_cache_attributes(obj) is False

    def test_cache_sort_key(self) -> None:
        """Test sort_key returns tuple."""
        key = FlextUtilities.Cache.sort_key("test")
        assert isinstance(key, tuple)
        assert len(key) == 2

    # =====================================================================
    # Type Checker Tests
    # =====================================================================

    def test_type_checker_can_handle_matching(self) -> None:
        """Test type checking with matching types."""
        assert FlextUtilities.TypeChecker.can_handle_message_type((str,), str) is True
        assert (
            FlextUtilities.TypeChecker.can_handle_message_type((str, int), str) is True
        )
        assert (
            FlextUtilities.TypeChecker.can_handle_message_type((str, int), int) is True
        )

    def test_type_checker_can_handle_non_matching(self) -> None:
        """Test type checking with non-matching types."""
        assert FlextUtilities.TypeChecker.can_handle_message_type((str,), int) is False
        assert FlextUtilities.TypeChecker.can_handle_message_type((int,), str) is False

    # =====================================================================
    # Validation Tests
    # =====================================================================

    def test_validation_sort_key_deterministic(self) -> None:
        """Test sort_key produces deterministic results."""
        key1 = FlextUtilities.Validation.sort_key("test")
        key2 = FlextUtilities.Validation.sort_key("test")
        assert key1 == key2
        assert isinstance(key1, (str, tuple))

    def test_validation_sort_key_different(self) -> None:
        """Test sort_key differs for different inputs."""
        key1 = FlextUtilities.Validation.sort_key("test1")
        key2 = FlextUtilities.Validation.sort_key("test2")
        assert key1 != key2

    def test_validation_normalize_component_primitives(self) -> None:
        """Test normalize_component with primitives."""
        assert FlextUtilities.Validation.normalize_component(True) is True
        assert FlextUtilities.Validation.normalize_component(42) == 42
        assert FlextUtilities.Validation.normalize_component("test") == "test"
        assert FlextUtilities.Validation.normalize_component(None) is None

    def test_validation_normalize_component_dict(self) -> None:
        """Test normalize_component with dictionaries."""
        test_dict = {"z": 1, "a": 2}
        result = FlextUtilities.Validation.normalize_component(test_dict)
        assert isinstance(result, (dict, tuple))

    def test_validation_normalize_component_list(self) -> None:
        """Test normalize_component with lists."""
        test_list = [1, 2, 3, "four"]
        result = FlextUtilities.Validation.normalize_component(test_list)
        assert isinstance(result, (tuple, list, str))

    def test_validation_normalize_component_pydantic(self) -> None:
        """Test normalize_component with Pydantic model."""
        model = UtilityScenarios.create_test_model()
        result = FlextUtilities.Validation.normalize_component(model)
        assert result is not None

    def test_validation_validate_pipeline_success(self) -> None:
        """Test validation pipeline with successful validators."""

        def validator1(data: str) -> FlextResult[bool]:
            return (
                FlextResult[bool].ok(True)
                if len(data) > 0
                else FlextResult[bool].fail("Empty")
            )

        def validator2(data: str) -> FlextResult[bool]:
            return (
                FlextResult[bool].ok(True)
                if data.isalnum()
                else FlextResult[bool].fail("Non-alnum")
            )

        result = FlextUtilities.Validation.validate_pipeline(
            "abc123", [validator1, validator2]
        )
        assert result.is_success

    def test_validation_validate_pipeline_failure(self) -> None:
        """Test validation pipeline with validator failure."""

        def validator1(data: str) -> FlextResult[bool]:
            return FlextResult[bool].fail("First failed")

        result = FlextUtilities.Validation.validate_pipeline("test", [validator1])
        assert result.is_failure
        assert "First failed" in str(result.error)

    def test_validation_validate_pipeline_empty(self) -> None:
        """Test validation pipeline with empty validators."""
        result = FlextUtilities.Validation.validate_pipeline("test", [])
        assert result.is_success

    def test_validation_validate_pipeline_exception(self) -> None:
        """Test validation pipeline handles exceptions."""

        def bad_validator(data: str) -> FlextResult[bool]:
            msg = "Unexpected error"
            raise ValueError(msg)

        result = FlextUtilities.Validation.validate_pipeline("test", [bad_validator])
        assert result.is_failure

    # =====================================================================
    # Configuration Tests
    # =====================================================================

    def test_configuration_get_parameter(self) -> None:
        """Test getting configuration parameter."""
        config = FlextConfig.get_global_instance()
        value = FlextUtilities.Configuration.get_parameter(config, "app_name")
        assert value is not None

    def test_configuration_set_parameter(self) -> None:
        """Test setting configuration parameter."""
        config = FlextConfig.get_global_instance()
        result = FlextUtilities.Configuration.set_parameter(
            config, "test_param", "test_value"
        )
        assert isinstance(result, bool)

    def test_configuration_get_singleton(self) -> None:
        """Test getting singleton configuration."""
        value = FlextUtilities.Configuration.get_singleton(FlextConfig, "app_name")
        assert value is not None

    # =====================================================================
    # Reliability Tests
    # =====================================================================

    def test_reliability_retry_immediate_success(self) -> None:
        """Test retry with immediate success."""

        def quick_success() -> FlextResult[str]:
            return FlextResult[str].ok("success")

        result = FlextUtilities.Reliability.retry(quick_success, max_attempts=3)
        assert result.is_success

    def test_reliability_retry_eventual_success(self) -> None:
        """Test retry with eventual success."""
        call_count = [0]

        def flaky_op() -> FlextResult[str]:
            call_count[0] += 1
            if call_count[0] < 3:
                return FlextResult[str].fail("Temporary failure")
            return FlextResult[str].ok("Success")

        result = FlextUtilities.Reliability.retry(
            flaky_op, max_attempts=5, delay_seconds=0.01
        )
        assert result.is_success
        assert call_count[0] >= 3

    # =====================================================================
    # Edge Cases and Advanced Tests
    # =====================================================================

    def test_cache_normalize_nested_structures(self) -> None:
        """Test cache normalization with nested structures."""
        nested = {"a": {"b": {"c": 1}}, "d": [1, 2, 3]}
        result = FlextUtilities.Cache.normalize_component(nested)
        assert isinstance(result, (dict, tuple))

    def test_text_processor_clean_multiple_spaces(self) -> None:
        """Test cleaning text with multiple spaces."""
        result = FlextUtilities.TextProcessor.clean_text("a    b    c")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_text_processor_truncate_no_truncation_needed(self) -> None:
        """Test truncate when not needed."""
        result = FlextUtilities.TextProcessor.truncate_text("Hi", max_length=10)
        assert result.is_success
        assert result.value == "Hi"

    def test_generators_default_short_id(self) -> None:
        """Test short ID with default length."""
        short_id = FlextUtilities.Generators.generate_short_id()
        assert isinstance(short_id, str)
        assert len(short_id) == 8

    def test_validation_normalize_dataclass(self) -> None:
        """Test normalize with dataclass."""

        @dataclass
        class Person:
            name: str
            age: int

        person = Person(name="Alice", age=30)
        result = FlextUtilities.Validation.normalize_component(person)
        assert result is not None

    def test_validation_sort_key_with_dict(self) -> None:
        """Test sort_key with dictionary."""
        dict_a = {"z": 1, "a": 2}
        key = FlextUtilities.Validation.sort_key(dict_a)
        assert isinstance(key, (str, tuple))
        assert len(key) > 0 if isinstance(key, (str, tuple)) else False

    def test_cache_generate_key_with_kwargs(self) -> None:
        """Test cache key generation with kwargs."""
        key = FlextUtilities.Cache.generate_cache_key("test", foo="bar", num=42)
        assert isinstance(key, str)
        assert len(key) > 0

    def test_type_guard_whitespace_strings(self) -> None:
        """Test type guard with whitespace-only strings."""
        assert FlextUtilities.TypeGuards.is_string_non_empty(" ") is False
        assert FlextUtilities.TypeGuards.is_string_non_empty("\t") is False
        assert FlextUtilities.TypeGuards.is_string_non_empty(" content ") is True


__all__ = ["TestFlextUtilities"]
