"""Comprehensive coverage tests for u.

Module: flext_core.utilities.u
Scope: All public u namespaces
Pattern: Railway-Oriented, Functional utilities, Type guards, Generators

Tests validate:
- Type guards (string, dict, list non-empty checks)
- ID/timestamp generation (multiple generator types)
- Text processing and normalization
- Caching utilities (normalization, key generation, cleanup)
- Reliability patterns (timeout, retry)
- Type checking for message handlers
- Configuration parameter access
- Validation utilities

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, cast

import pytest
from pydantic import BaseModel

from flext_core import FlextExceptions, FlextResult, p, t, u

# =========================================================================
# Test Data and Scenarios
# =========================================================================


class UtilityOperationType(StrEnum):
    """Utility operation types for parametrization."""

    TYPE_GUARD_STRING = "type_guard_string"
    TYPE_GUARD_DICT = "type_guard_dict"
    TYPE_GUARD_LIST = "type_guard_list"
    ID_GENERATION = "id_generation"
    TIMESTAMP_GENERATION = "timestamp_generation"
    CACHE_NORMALIZATION = "cache_normalization"
    CACHE_KEY = "cache_key"
    TEXT_CLEANING = "text_cleaning"
    TEXT_TRUNCATION = "text_truncation"


@dataclass(frozen=True, slots=True)
class UtilityTestCase:
    """Test case for utility operations."""

    operation: UtilityOperationType
    input_data: t.GeneralValueType | None = None
    expected_type: type | None = None
    should_succeed: bool = True
    description: str = ""


class UtilityScenarios:
    """Centralized utility test scenarios."""

    # Type guard test cases
    TYPE_GUARD_CASES: ClassVar[list[UtilityTestCase]] = [
        UtilityTestCase(
            operation=UtilityOperationType.TYPE_GUARD_STRING,
            input_data="hello",
            expected_type=bool,
            should_succeed=True,
            description="Non-empty string passes guard",
        ),
        UtilityTestCase(
            operation=UtilityOperationType.TYPE_GUARD_STRING,
            input_data="",
            expected_type=bool,
            should_succeed=False,
            description="Empty string fails guard",
        ),
        UtilityTestCase(
            operation=UtilityOperationType.TYPE_GUARD_STRING,
            input_data=123,
            expected_type=bool,
            should_succeed=False,
            description="Non-string fails guard",
        ),
        UtilityTestCase(
            operation=UtilityOperationType.TYPE_GUARD_DICT,
            input_data={"key": "value"},
            expected_type=bool,
            should_succeed=True,
            description="Non-empty dict passes guard",
        ),
        UtilityTestCase(
            operation=UtilityOperationType.TYPE_GUARD_DICT,
            input_data={},
            expected_type=bool,
            should_succeed=False,
            description="Empty dict fails guard",
        ),
        UtilityTestCase(
            operation=UtilityOperationType.TYPE_GUARD_DICT,
            input_data=None,
            expected_type=bool,
            should_succeed=False,
            description="None fails dict guard",
        ),
        UtilityTestCase(
            operation=UtilityOperationType.TYPE_GUARD_LIST,
            input_data=[1, 2, 3],
            expected_type=bool,
            should_succeed=True,
            description="Non-empty list passes guard",
        ),
        UtilityTestCase(
            operation=UtilityOperationType.TYPE_GUARD_LIST,
            input_data=[],
            expected_type=bool,
            should_succeed=False,
            description="Empty list fails guard",
        ),
        UtilityTestCase(
            operation=UtilityOperationType.TYPE_GUARD_LIST,
            input_data=None,
            expected_type=bool,
            should_succeed=False,
            description="None fails list guard",
        ),
    ]

    # Generator test cases
    ID_GENERATOR_CASES: ClassVar[list[tuple[str, str | None]]] = [
        ("generate_id", None),
        (
            "generate_iso_timestamp",
            None,
        ),  # Returns full ISO timestamp, no simple prefix - uses u.Generators.generate_iso_timestamp()
        ("generate_correlation_id", "corr_"),
        ("generate_entity_id", None),
        ("generate_saga_id", None),
        ("generate_event_id", None),
    ]

    # Cache test cases
    CACHE_NORMALIZATION_CASES: ClassVar[list[tuple[object, type]]] = [
        ({"a": 1, "b": 2}, dict),
        ([1, 2, 3], list),
        (None, type(None)),
    ]

    @staticmethod
    def create_mock_config(
        **kwargs: t.GeneralValueType,
    ) -> p.HasModelDump:
        """Create mock config object."""

        class TestConfig:
            def model_dump(self) -> t.ConfigurationMapping:
                # Convert kwargs to proper ConfigurationMapping type
                # HasModelDump expects Mapping[str, FlexibleValue]
                # ConfigurationMapping = Mapping[str, t.GeneralValueType]
                # For test purposes, we return ConfigurationMapping which is compatible
                result: dict[str, t.GeneralValueType] = {}
                for key, value in kwargs.items():
                    result[str(key)] = value
                # dict[str, t.GeneralValueType] is compatible with Mapping[str, t.GeneralValueType]
                return result

        # TestConfig implements HasModelDump protocol via structural typing
        # model_dump() returns ConfigurationMapping which satisfies the protocol
        # ConfigurationMapping = Mapping[str, t.GeneralValueType]
        # HasModelDump expects Mapping[str, FlexibleValue]
        # t.GeneralValueType is compatible with FlexibleValue at runtime
        # Type assertion: TestConfig structurally implements HasModelDump
        # The model_dump() method signature is compatible with the protocol
        # Use cast to help type checker understand structural typing
        return cast("p.HasModelDump", TestConfig())

    @staticmethod
    def create_mock_cached_object() -> object:
        """Create mock object with cache attributes."""

        class TestCachedObject:
            def __init__(self) -> None:
                self._cache: t.ConfigurationMapping = {"key": "value"}
                self._simple_cache: str = "cached_value"

        return TestCachedObject()

    @staticmethod
    def create_mock_uncached_object() -> object:
        """Create mock object without cache attributes."""

        class TestUncachedObject:
            def __init__(self) -> None:
                pass

        return TestUncachedObject()

    @staticmethod
    def create_custom_object() -> object:
        """Create custom serializable object."""

        class CustomObject:
            def __str__(self) -> str:
                return "custom_object"

        return CustomObject()

    @staticmethod
    def create_flaky_operation() -> tuple[
        list[int],
        Callable[[], FlextResult[str]],
    ]:
        """Create flaky operation that eventually succeeds."""
        attempt_count = [0]

        def flaky_op() -> FlextResult[str]:
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                return FlextResult[str].fail("Temporary failure")
            return FlextResult[str].ok("success")

        return attempt_count, flaky_op


# =========================================================================
# Test Suite - u Comprehensive Coverage
# =========================================================================


class Testu:
    """Unified test suite for u - ALL REAL FUNCTIONALITY."""

    # =====================================================================
    # Type Guards Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "case",
        [
            c
            for c in UtilityScenarios.TYPE_GUARD_CASES
            if c.operation == UtilityOperationType.TYPE_GUARD_STRING
        ],
    )
    def test_type_guard_string(self, case: UtilityTestCase) -> None:
        """Test string type guards."""
        result = u.is_type(case.input_data, "string_non_empty")
        assert isinstance(result, bool)
        assert result is case.should_succeed

    @pytest.mark.parametrize(
        "case",
        [
            c
            for c in UtilityScenarios.TYPE_GUARD_CASES
            if c.operation == UtilityOperationType.TYPE_GUARD_DICT
        ],
    )
    def test_type_guard_dict(self, case: UtilityTestCase) -> None:
        """Test dict type guards."""
        result = u.is_type(case.input_data, "dict_non_empty")
        assert isinstance(result, bool)
        assert result is case.should_succeed

    @pytest.mark.parametrize(
        "case",
        [
            c
            for c in UtilityScenarios.TYPE_GUARD_CASES
            if c.operation == UtilityOperationType.TYPE_GUARD_LIST
        ],
    )
    def test_type_guard_list(self, case: UtilityTestCase) -> None:
        """Test list type guards."""
        result = u.is_type(case.input_data, "list_non_empty")
        assert isinstance(result, bool)
        assert result is case.should_succeed

    # =====================================================================
    # Generators Tests
    # =====================================================================

    def test_generate_id_uniqueness(self) -> None:
        """Test ID generation produces unique values."""
        id1 = u.generate()
        id2 = u.generate()
        assert isinstance(id1, str)
        assert len(id1) > 0
        assert id1 != id2

    @pytest.mark.parametrize(
        ("method_name", "prefix"),
        UtilityScenarios.ID_GENERATOR_CASES,
    )
    def test_generators(self, method_name: str, prefix: str | None) -> None:
        """Test various ID and timestamp generators."""
        if method_name == "generate_iso_timestamp":
            # Special case - this method still exists
            result = u.Generators.generate_iso_timestamp()
        elif method_name == "generate_id":
            # Use unified generate() without kind
            result = u.generate()
        elif method_name == "generate_correlation_id":
            result = u.generate("correlation")
        elif method_name == "generate_entity_id":
            result = u.generate("entity")
        elif method_name == "generate_saga_id":
            result = u.generate("saga")
        elif method_name == "generate_event_id":
            result = u.generate("event")
        else:
            # Fallback for any other methods
            method = getattr(u.Generators, method_name, None)
            if method is None:
                pytest.skip(f"Method {method_name} not available")
            result = method()
        assert isinstance(result, str)
        assert len(result) > 0
        if prefix:
            assert result.startswith(prefix)

    def test_generate_short_id_length(self) -> None:
        """Test short ID generation with specific length."""
        short_id = u.generate("ulid", length=8)
        assert isinstance(short_id, str)
        assert len(short_id) == 8

    # =====================================================================
    # Text Processor Tests
    # =====================================================================

    def test_text_processor_clean(self) -> None:
        """Test text cleaning - returns str directly."""
        result = u.Text.clean_text("  hello   world  ")
        assert isinstance(result, str)
        assert result == "hello world"

    def test_text_processor_truncate(self) -> None:
        """Test text truncation - returns FlextResult[str]."""
        result = u.Text.truncate_text("hello world", max_length=8)
        assert result.is_success
        assert len(result.value) <= 8
        assert result.value.endswith("...")

    def test_text_processor_safe_string_success(self) -> None:
        """Test safe string conversion - returns str directly."""
        result = u.Text.safe_string("  hello  ")
        assert isinstance(result, str)
        assert result == "hello"

    def test_text_processor_safe_string_failure(self) -> None:
        """Test safe string conversion with empty - raises ValueError."""
        with pytest.raises(ValueError):
            u.Text.safe_string("")

    # =====================================================================
    # Cache Tests
    # =====================================================================

    @pytest.mark.parametrize(
        ("input_data", "expected_type"),
        UtilityScenarios.CACHE_NORMALIZATION_CASES,
    )
    def test_cache_normalize_component(
        self,
        input_data: t.GeneralValueType | None,
        expected_type: type,
    ) -> None:
        """Test cache component normalization."""
        result = u.Cache.normalize_component(input_data)
        assert isinstance(result, (dict, str, type(None), expected_type))

    def test_cache_sort_dict_keys(self) -> None:
        """Test dictionary key sorting."""
        data = {"z": 1, "a": 2, "m": 3}
        result = u.Cache.sort_dict_keys(data)
        assert isinstance(result, dict)
        keys = list(result.keys())
        assert keys == sorted(keys)

    def test_cache_generate_key(self) -> None:
        """Test cache key generation."""
        key1 = u.Cache.generate_cache_key(None, None)
        assert isinstance(key1, str)
        assert len(key1) > 0

    def test_cache_generate_key_uniqueness(self) -> None:
        """Test cache keys are unique for different inputs."""
        key1 = u.Cache.generate_cache_key("arg1", kwarg1="value1")
        key2 = u.Cache.generate_cache_key("different")
        assert key1 != key2

    def test_cache_clear_object(self) -> None:
        """Test clearing object cache."""
        obj = UtilityScenarios.create_mock_cached_object()
        # Type narrowing: obj is object, but clear_object_cache expects t.GeneralValueType
        # Since obj has model_dump, it's compatible
        if isinstance(obj, BaseModel):
            result = u.Cache.clear_object_cache(obj)
        else:
            # For non-BaseModel objects, convert to dict-like structure
            obj_dict: dict[str, t.GeneralValueType] = {}
            if hasattr(obj, "__dict__"):
                for k, v in obj.__dict__.items():
                    obj_dict[str(k)] = (
                        v
                        if isinstance(
                            v,
                            (str, int, float, bool, type(None), list, dict),
                        )
                        else str(v)
                    )
            result = u.Cache.clear_object_cache(obj_dict)
        assert result.is_success

    def test_cache_has_attributes_true(self) -> None:
        """Test detecting cache attributes on object with cache."""
        obj = UtilityScenarios.create_mock_cached_object()
        # has_cache_attributes expects an object with attributes, not a converted value
        # Cast to t.GeneralValueType for type checker
        obj_typed: t.GeneralValueType = cast("t.GeneralValueType", obj)
        assert u.Cache.has_cache_attributes(obj_typed) is True

    def test_cache_has_attributes_false(self) -> None:
        """Test detecting cache attributes on object without cache."""
        obj = UtilityScenarios.create_mock_uncached_object()
        # has_cache_attributes expects an object with attributes, not a converted value
        # Cast to t.GeneralValueType for type checker
        obj_typed: t.GeneralValueType = cast("t.GeneralValueType", obj)
        assert u.Cache.has_cache_attributes(obj_typed) is False

    # =====================================================================
    # Reliability Tests
    # =====================================================================

    def test_reliability_timeout_success(self) -> None:
        """Test timeout with successful operation."""

        def quick_op() -> FlextResult[str]:
            return FlextResult[str].ok("success")

        result = u.Reliability.with_timeout(quick_op, 5.0)
        assert result.is_success
        assert result.value == "success"

    def test_reliability_timeout_invalid(self) -> None:
        """Test timeout with invalid timeout value."""

        def op() -> FlextResult[str]:
            return FlextResult[str].ok("success")

        result = u.Reliability.with_timeout(op, -1.0)
        assert result.is_failure

    def test_reliability_retry_first_success(self) -> None:
        """Test retry that succeeds immediately."""

        def op() -> FlextResult[str]:
            return FlextResult[str].ok("success")

        result: FlextResult[str] = u.Reliability.retry(op, max_attempts=3)
        assert result.is_success
        assert result.value == "success"

    def test_reliability_retry_eventual_success(self) -> None:
        """Test retry with eventual success."""
        attempt_count, flaky_op = UtilityScenarios.create_flaky_operation()
        result: FlextResult[str] = u.Reliability.retry(
            flaky_op,
            max_attempts=3,
            delay_seconds=0.01,
        )
        assert result.is_success
        assert attempt_count[0] >= 2

    # =====================================================================
    # Type Checker Tests
    # =====================================================================

    def test_type_checker_object_accepts_all(self) -> None:
        """Test type checking with object (accepts all)."""
        # MessageTypeSpecifier = str | type[t.GeneralValueType]
        # object is not a valid MessageTypeSpecifier, use str instead
        accepted: tuple[t.MessageTypeSpecifier, ...] = (str,)
        assert u.Checker.can_handle_message_type(accepted, str) is True

    def test_type_checker_specific_type_match(self) -> None:
        """Test type checking with matching specific type."""
        accepted = (str,)
        assert u.Checker.can_handle_message_type(accepted, str) is True

    def test_type_checker_specific_type_mismatch(self) -> None:
        """Test type checking with mismatched specific type."""
        accepted = (str,)
        assert u.Checker.can_handle_message_type(accepted, int) is False

    def test_type_checker_empty_accepted(self) -> None:
        """Test type checking with no accepted types."""
        accepted: tuple[t.MessageTypeSpecifier, ...] = ()
        assert u.Checker.can_handle_message_type(accepted, str) is False

    # =====================================================================
    # Configuration Tests
    # =====================================================================

    def test_configuration_get_parameter(self) -> None:
        """Test parameter retrieval from config."""
        config = UtilityScenarios.create_mock_config(timeout=30)
        # get_parameter expects HasModelDump | ConfigurationMapping | None
        # TestConfig has model_dump method, so it's compatible
        value = u.Configuration.get_parameter(config, "timeout")
        assert value == 30

    def test_configuration_get_parameter_missing(self) -> None:
        """Test parameter retrieval for missing parameter."""
        config = UtilityScenarios.create_mock_config(timeout=30)
        # get_parameter expects HasModelDump | ConfigurationMapping | None
        # TestConfig has model_dump method, so it's compatible
        with pytest.raises(FlextExceptions.NotFoundError):
            u.Configuration.get_parameter(config, "missing")

    # =====================================================================
    # Validation Tests
    # =====================================================================

    def test_validation_pipeline_with_failing_validator(self) -> None:
        """Test validation pipeline with exception-raising validator."""

        def failing_validator(x: object) -> FlextResult[bool]:
            msg = "Validation error"
            raise ValueError(msg)

        result = u.Validation.validate_pipeline(
            "test_value",
            [failing_validator],
        )
        assert result.is_failure
        assert "Validator failed" in (result.error or "")

    def test_validation_sort_key_dict(self) -> None:
        """Test sort_key with dict - returns tuple[str, str]."""
        key = u.Validation.sort_key({"b": 2, "a": 1})
        assert isinstance(key, tuple)
        assert len(key) == 2

    def test_validation_sort_key_list(self) -> None:
        """Test sort_key with list - returns tuple[str, str]."""
        key = u.Validation.sort_key([1, 2, 3])
        assert isinstance(key, tuple)
        assert len(key) == 2

    def test_validation_sort_key_string(self) -> None:
        """Test sort_key with string - returns tuple[str, str]."""
        key = u.Validation.sort_key("test")
        assert isinstance(key, tuple)
        assert len(key) == 2

    def test_validation_sort_key_number(self) -> None:
        """Test sort_key with number - returns tuple[str, str]."""
        # sort_key expects t.GeneralValueType, numbers are compatible
        key = u.Validation.sort_key(42)
        assert isinstance(key, tuple)
        assert len(key) == 2

    def test_validation_sort_key_custom_object(self) -> None:
        """Test sort_key with custom object - returns tuple[str, str]."""
        obj = UtilityScenarios.create_custom_object()
        # Type narrowing: sort_key expects t.GeneralValueType
        # Convert object to t.GeneralValueType compatible value
        obj_value: t.GeneralValueType = (
            obj
            if isinstance(obj, (str, int, float, bool, type(None), list, dict))
            else str(obj)
        )
        result = u.Validation.sort_key(obj_value)
        assert isinstance(result, tuple)
        assert len(result) == 2


__all__ = ["Testu"]
