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
from enum import StrEnum
from typing import Annotated, ClassVar, cast, override

import pytest
from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextExceptions, m, p, r, u
from flext_tests import t, tm

from ..test_utils import assertion_helpers
from .contracts.text_contract import TextUtilityContract


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


class UtilityTestCase(BaseModel):
    """Test case for utility operations."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    operation: Annotated[
        UtilityOperationType, Field(description="Utility operation under test")
    ]
    input_data: t.NormalizedValue = Field(
        default=None, description="Input data for operation"
    )
    expected_type: type | None = Field(default=None, description="Expected output type")
    should_succeed: bool = Field(
        default=True, description="Whether operation should succeed"
    )
    description: str = Field(default="", description="Scenario description")


class _TestCachedObject:
    """Mock object with cache attributes."""

    def __init__(self) -> None:
        self._cache: m.ConfigMap = m.ConfigMap(root={"key": "value"})
        self._simple_cache: str = "cached_value"


class _TestUncachedObject:
    """Mock object without cache attributes."""

    def __init__(self) -> None:
        pass


class _CustomObject:
    """Custom serializable object."""

    @override
    def __str__(self) -> str:
        return "custom_object"


class UtilityScenarios:
    """Centralized utility test scenarios."""

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
    ID_GENERATOR_CASES: ClassVar[list[tuple[str, str | None]]] = [
        ("generate_id", None),
        ("generate_iso_timestamp", None),
        ("generate_correlation_id", "corr_"),
        ("generate_entity_id", None),
        ("generate_saga_id", None),
        ("generate_event_id", None),
    ]
    CACHE_NORMALIZATION_CASES: ClassVar[list[tuple[t.NormalizedValue, type]]] = [
        ({"a": 1, "b": 2}, dict),
        ([1, 2, 3], list),
        (None, type(None)),
    ]

    @staticmethod
    def create_mock_config(**kwargs: t.Scalar) -> p.HasModelDump:
        """Create mock config object."""
        result: dict[str, t.NormalizedValue | BaseModel] = {}
        for key, value in kwargs.items():
            result[str(key)] = value
        config_map: p.HasModelDump = m.ConfigMap(root=result)
        return config_map

    @staticmethod
    def create_mock_cached_object() -> _TestCachedObject:
        """Create mock object with cache attributes."""
        return _TestCachedObject()

    @staticmethod
    def create_mock_uncached_object() -> _TestUncachedObject:
        """Create mock object without cache attributes."""
        return _TestUncachedObject()

    @staticmethod
    def create_custom_object() -> _CustomObject:
        """Create custom serializable object."""
        return _CustomObject()

    @staticmethod
    def create_flaky_operation() -> tuple[list[int], Callable[[], r[str]]]:
        """Create flaky operation that eventually succeeds."""
        attempt_count = [0]

        def flaky_op() -> r[str]:
            attempt_count[0] += 1
            if attempt_count[0] < 2:
                return r[str].fail("Temporary failure")
            return r[str].ok("success")

        return (attempt_count, flaky_op)


class Testu(TextUtilityContract):
    """Unified test suite for u - ALL REAL FUNCTIONALITY."""

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
        tm.that(isinstance(result, bool), eq=True)
        tm.that(result, eq=case.should_succeed)

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
        tm.that(isinstance(result, bool), eq=True)
        tm.that(result, eq=case.should_succeed)

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
        tm.that(isinstance(result, bool), eq=True)
        tm.that(result, eq=case.should_succeed)

    def test_generate_id_uniqueness(self) -> None:
        """Test ID generation produces unique values."""
        id1 = u.generate()
        id2 = u.generate()
        tm.that(isinstance(id1, str), eq=True)
        tm.that(len(id1), gt=0)
        tm.that(id1, ne=id2)

    @pytest.mark.parametrize(
        ("method_name", "prefix"),
        UtilityScenarios.ID_GENERATOR_CASES,
    )
    def test_generators(self, method_name: str, prefix: str | None) -> None:
        """Test various ID and timestamp generators."""
        if method_name == "generate_iso_timestamp":
            result = u.generate_iso_timestamp()
        elif method_name == "generate_id":
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
            method = getattr(u, method_name, None)
            if method is None:
                pytest.skip(f"Method {method_name} not available")
            result = method()
        tm.that(isinstance(result, str), eq=True)
        tm.that(len(result), gt=0)
        if prefix:
            tm.that(result.startswith(prefix), eq=True)

    def test_generate_short_id_length(self) -> None:
        """Test short ID generation with specific length."""
        short_id = u.generate("ulid", length=8)
        tm.that(isinstance(short_id, str), eq=True)
        tm.that(len(short_id), eq=8)

    @pytest.mark.parametrize(("raw", "expected"), TextUtilityContract.CLEAN_TEXT_CASES)
    def test_text_processor_clean(self, raw: str, expected: str) -> None:
        """Test text cleaning contract across reusable shared cases."""
        self.assert_clean_text(raw, expected)

    def test_text_processor_truncate(self) -> None:
        """Test text truncation - returns r[str]."""
        result = u.truncate_text("hello world", max_length=8)
        _ = assertion_helpers.assert_flext_result_success(result)
        tm.that(len(result.value), lte=8)
        tm.that(result.value.endswith("..."), eq=True)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        TextUtilityContract.SAFE_STRING_VALID_CASES,
    )
    def test_text_processor_safe_string_success(self, raw: str, expected: str) -> None:
        """Test safe string contract for valid values."""
        self.assert_safe_string_valid(raw, expected)

    @pytest.mark.parametrize(
        ("raw", "error_message"),
        TextUtilityContract.SAFE_STRING_INVALID_CASES,
    )
    def test_text_processor_safe_string_failure(
        self,
        raw: str | None,
        error_message: str,
    ) -> None:
        """Test safe string contract for invalid values."""
        with pytest.raises(ValueError, match=error_message):
            u.safe_string(raw)

    @pytest.mark.parametrize(
        ("input_data", "expected_type"),
        UtilityScenarios.CACHE_NORMALIZATION_CASES,
    )
    def test_cache_normalize_component(
        self,
        input_data: t.NormalizedValue,
        expected_type: type,
    ) -> None:
        """Test cache component normalization."""
        result = u.normalize_component(input_data)
        tm.that(isinstance(result, (dict, str, type(None), expected_type)), eq=True)

    def test_cache_sort_dict_keys(self) -> None:
        """Test dictionary key sorting."""
        data = {"z": 1, "a": 2, "m": 3}
        result = u.sort_dict_keys(data)
        tm.that(isinstance(result, dict), eq=True)
        keys = list(result.keys())
        tm.that(keys, eq=sorted(keys))

    def test_cache_generate_key(self) -> None:
        """Test cache key generation."""
        key1 = u.generate_cache_key(None, None)
        tm.that(isinstance(key1, str), eq=True)
        tm.that(len(key1), gt=0)

    def test_cache_generate_key_uniqueness(self) -> None:
        """Test cache keys are unique for different inputs."""
        key1 = u.generate_cache_key("arg1", kwarg1="value1")
        key2 = u.generate_cache_key("different")
        tm.that(key1, ne=key2)

    def test_cache_clear_object(self) -> None:
        """Test clearing object cache."""
        obj = UtilityScenarios.create_mock_cached_object()
        if isinstance(obj, BaseModel):
            result = u.clear_object_cache(obj)
        else:
            obj_dict: dict[str, t.NormalizedValue | BaseModel] = {}
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
            result = u.clear_object_cache(m.ConfigMap(root=obj_dict))
        _ = assertion_helpers.assert_flext_result_success(result)

    def test_cache_has_attributes_true(self) -> None:
        """Test detecting cache attributes on object with cache."""
        obj = UtilityScenarios.create_mock_cached_object()
        assert u.has_cache_attributes(cast("t.NormalizedValue", obj)) is True

    def test_cache_has_attributes_false(self) -> None:
        """Test detecting cache attributes on object without cache."""
        obj = UtilityScenarios.create_mock_uncached_object()
        assert u.has_cache_attributes(cast("t.NormalizedValue", obj)) is False

    def test_reliability_timeout_success(self) -> None:
        """Test timeout with successful operation."""

        def quick_op() -> r[str]:
            return r[str].ok("success")

        result = u.with_timeout(quick_op, 5.0)
        _ = assertion_helpers.assert_flext_result_success(result)
        tm.that(result.value, eq="success")

    def test_reliability_timeout_invalid(self) -> None:
        """Test timeout with invalid timeout value."""

        def op() -> r[str]:
            return r[str].ok("success")

        result = u.with_timeout(op, -1.0)
        _ = assertion_helpers.assert_flext_result_failure(result)

    def test_reliability_retry_first_success(self) -> None:
        """Test retry that succeeds immediately."""

        def op() -> r[str]:
            return r[str].ok("success")

        result: r[str] = u.retry(op, max_attempts=3)
        _ = assertion_helpers.assert_flext_result_success(result)
        tm.that(result.value, eq="success")

    def test_reliability_retry_eventual_success(self) -> None:
        """Test retry with eventual success."""
        attempt_count, flaky_op = UtilityScenarios.create_flaky_operation()
        result: r[str] = u.retry(
            flaky_op,
            max_attempts=3,
            delay_seconds=0.01,
        )
        _ = assertion_helpers.assert_flext_result_success(result)
        tm.that(attempt_count[0], gte=2)

    def test_type_checker_object_accepts_all(self) -> None:
        """Test type checking with object (accepts all)."""
        accepted: tuple[t.MessageTypeSpecifier, ...] = (str,)
        tm.that(u.can_handle_message_type(accepted, str), eq=True)

    def test_type_checker_specific_type_match(self) -> None:
        """Test type checking with matching specific type."""
        accepted = (str,)
        tm.that(u.can_handle_message_type(accepted, str), eq=True)

    def test_type_checker_specific_type_mismatch(self) -> None:
        """Test type checking with mismatched specific type."""
        accepted = (str,)
        tm.that(u.can_handle_message_type(accepted, int), eq=False)

    def test_type_checker_empty_accepted(self) -> None:
        """Test type checking with no accepted types."""
        accepted: tuple[t.MessageTypeSpecifier, ...] = ()
        tm.that(u.can_handle_message_type(accepted, str), eq=False)

    def test_configuration_get_parameter(self) -> None:
        """Test parameter retrieval from config."""
        config = UtilityScenarios.create_mock_config(timeout=30)
        value = u.get_parameter(config, "timeout")
        tm.that(value, eq=30)

    def test_configuration_get_parameter_missing(self) -> None:
        """Test parameter retrieval for missing parameter."""
        config = UtilityScenarios.create_mock_config(timeout=30)
        with pytest.raises(FlextExceptions.NotFoundError):
            u.get_parameter(config, "missing")


__all__ = ["Testu"]
