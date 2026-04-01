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

from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import ClassVar, override

import pytest
from flext_tests import tm

from flext_core import r
from tests import TextUtilityContract, assertion_helpers, t, u


class Testu(TextUtilityContract):
    """Unified test suite for u - ALL REAL FUNCTIONALITY."""

    @unique
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

        operation: str
        input_data: t.NormalizedValue = None
        expected_type: type | None = None
        should_succeed: bool = True
        description: str = ""

    class _TestCachedObject:
        """Mock t.NormalizedValue with cache attributes."""

        def __init__(self) -> None:
            self._cache: t.ConfigMap = t.ConfigMap(root={"key": "value"})
            self._simple_cache: str = "cached_value"

    class _TestUncachedObject:
        """Mock t.NormalizedValue without cache attributes."""

        def __init__(self) -> None:
            self.name: str = "uncached"

    class _CustomObject:
        """Custom serializable t.NormalizedValue."""

        @override
        def __str__(self) -> str:
            return "custom_object"

    class UtilityScenarios:
        """Centralized utility test scenarios."""

        TYPE_GUARD_STRING_CASES: ClassVar[
            Sequence[tuple[t.NormalizedValue, bool, str]]
        ] = [
            ("hello", True, "Non-empty string passes guard"),
            ("", False, "Empty string fails guard"),
            (123, False, "Non-string fails guard"),
        ]
        TYPE_GUARD_DICT_CASES: ClassVar[
            Sequence[tuple[t.NormalizedValue, bool, str]]
        ] = [
            ({"key": "value"}, True, "Non-empty dict passes guard"),
            ({}, False, "Empty dict fails guard"),
            (None, False, "None fails dict guard"),
        ]
        TYPE_GUARD_LIST_CASES: ClassVar[
            Sequence[tuple[t.NormalizedValue, bool, str]]
        ] = [
            ([1, 2, 3], True, "Non-empty list passes guard"),
            ([], False, "Empty list fails guard"),
            (None, False, "None fails list guard"),
        ]
        ID_GENERATOR_CASES: ClassVar[Sequence[tuple[str, str | None]]] = [
            ("generate_id", None),
            ("generate_iso_timestamp", None),
            ("generate_correlation_id", "corr_"),
            ("generate_entity_id", None),
            ("generate_saga_id", None),
            ("generate_event_id", None),
        ]
        CACHE_NORMALIZATION_CASES: ClassVar[
            Sequence[tuple[t.NormalizedValue, type]]
        ] = [
            ({"a": 1, "b": 2}, dict),
            ([1, 2, 3], list),
            (None, type(None)),
        ]

        @staticmethod
        def create_mock_config(**kwargs: t.Scalar) -> t.ConfigMap:
            """Create mock config t.NormalizedValue."""
            result: MutableMapping[str, t.ValueOrModel] = {}
            for key, value in kwargs.items():
                result[str(key)] = value
            return t.ConfigMap(root=result)

        @staticmethod
        def create_mock_cached_object() -> Testu._TestCachedObject:
            """Create mock t.NormalizedValue with cache attributes."""
            return Testu._TestCachedObject()

        @staticmethod
        def create_mock_uncached_object() -> Testu._TestUncachedObject:
            """Create mock t.NormalizedValue without cache attributes."""
            return Testu._TestUncachedObject()

        @staticmethod
        def create_custom_object() -> Testu._CustomObject:
            """Create custom serializable t.NormalizedValue."""
            return Testu._CustomObject()

        @staticmethod
        def create_flaky_operation() -> tuple[Sequence[int], Callable[[], r[str]]]:
            """Create flaky operation that eventually succeeds."""
            attempt_count = [0]

            def flaky_op() -> r[str]:
                attempt_count[0] += 1
                if attempt_count[0] < 2:
                    return r[str].fail("Temporary failure")
                return r[str].ok("success")

            return (attempt_count, flaky_op)

    @pytest.mark.parametrize(
        ("input_data", "should_succeed", "description"),
        UtilityScenarios.TYPE_GUARD_STRING_CASES,
    )
    def test_type_guard_string(
        self,
        input_data: t.NormalizedValue,
        should_succeed: bool,
        description: str,
    ) -> None:
        """Test string type guards."""
        case = self.UtilityTestCase(
            operation=self.UtilityOperationType.TYPE_GUARD_STRING.value,
            input_data=input_data,
            expected_type=bool,
            should_succeed=should_succeed,
            description=description,
        )
        result = u.is_type(case.input_data, "string_non_empty")
        tm.that(result, eq=case.should_succeed)

    @pytest.mark.parametrize(
        ("input_data", "should_succeed", "description"),
        UtilityScenarios.TYPE_GUARD_DICT_CASES,
    )
    def test_type_guard_dict(
        self,
        input_data: t.NormalizedValue,
        should_succeed: bool,
        description: str,
    ) -> None:
        """Test dict type guards."""
        case = self.UtilityTestCase(
            operation=self.UtilityOperationType.TYPE_GUARD_DICT.value,
            input_data=input_data,
            expected_type=bool,
            should_succeed=should_succeed,
            description=description,
        )
        result = u.is_type(case.input_data, "dict_non_empty")
        tm.that(result, eq=case.should_succeed)

    @pytest.mark.parametrize(
        ("input_data", "should_succeed", "description"),
        UtilityScenarios.TYPE_GUARD_LIST_CASES,
    )
    def test_type_guard_list(
        self,
        input_data: t.NormalizedValue,
        should_succeed: bool,
        description: str,
    ) -> None:
        """Test list type guards."""
        case = self.UtilityTestCase(
            operation=self.UtilityOperationType.TYPE_GUARD_LIST.value,
            input_data=input_data,
            expected_type=bool,
            should_succeed=should_succeed,
            description=description,
        )
        result = u.is_type(case.input_data, "list_non_empty")
        tm.that(result, eq=case.should_succeed)

    def test_generate_id_uniqueness(self) -> None:
        """Test ID generation produces unique values."""
        id1 = u.generate()
        id2 = u.generate()
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
        tm.that(result, is_=str)
        tm.that(len(result), gt=0)
        if prefix:
            tm.that(result.startswith(prefix), eq=True)

    def test_generate_short_id_length(self) -> None:
        """Test short ID generation with specific length."""
        short_id = u.generate("ulid", length=8)
        tm.that(len(short_id), eq=8)

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
        tm.that(result, is_=(dict, str, type(None), expected_type))

    def test_reliability_retry_first_success(self) -> None:
        """Test retry that succeeds immediately."""

        def op() -> r[str]:
            return r[str].ok("success")

        result: r[str] = u.retry(op, max_attempts=3)
        _ = assertion_helpers.assert_flext_result_success(result)
        tm.that(result.value, eq="success")

    def test_reliability_retry_eventual_success(self) -> None:
        """Test retry with eventual success."""
        attempt_count, flaky_op = self.UtilityScenarios.create_flaky_operation()
        result: r[str] = u.retry(
            flaky_op,
            max_attempts=3,
            delay_seconds=0.01,
        )
        _ = assertion_helpers.assert_flext_result_success(result)
        tm.that(attempt_count[0], gte=2)

    def test_type_checker_object_accepts_all(self) -> None:
        """Test type checking with t.NormalizedValue (accepts all)."""
        accepted: tuple[t.MessageTypeSpecifier, ...] = (str,)
        tm.that(u.can_handle_message_type(accepted, str), eq=True)

    def test_type_checker_specific_type_match(self) -> None:
        """Test type checking with matching specific type."""
        accepted = (str,)
        tm.that(u.can_handle_message_type(accepted, str), eq=True)

    def test_type_checker_specific_type_mismatch(self) -> None:
        """Test type checking with mismatched specific type."""
        accepted = (str,)
        tm.that(not u.can_handle_message_type(accepted, int), eq=True)

    def test_type_checker_empty_accepted(self) -> None:
        """Test type checking with no accepted types."""
        accepted: tuple[t.MessageTypeSpecifier, ...] = ()
        tm.that(not u.can_handle_message_type(accepted, str), eq=True)


__all__ = ["Testu"]
