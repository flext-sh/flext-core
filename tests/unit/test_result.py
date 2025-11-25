"""Tests for FlextResult - Core railway pattern implementation.

Module: flext_core.result
Scope: FlextResult - railway-oriented programming core implementation

Tests FlextResult functionality including:
- Result creation (ok/fail)
- Value extraction (unwrap, unwrap_or)
- Railway transformations (map, flat_map, filter)
- Error recovery (alt, lash)
- Operators (|) and boolean conversion
- Railway composition patterns

Uses Python 3.13 patterns (StrEnum, frozen dataclasses with slots),
centralized constants, and parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

import pytest

from flext_core import FlextResult

# =========================================================================
# Result Scenario Type Enumerations
# =========================================================================


class ResultOperationType(StrEnum):
    """Result operation test scenario types."""

    CREATION_SUCCESS = "creation_success"
    CREATION_FAILURE = "creation_failure"
    UNWRAP = "unwrap"
    UNWRAP_OR = "unwrap_or"
    MAP = "map"
    FLAT_MAP = "flat_map"
    FILTER = "filter"
    ALT = "alt"
    LASH = "lash"
    OR_OPERATOR = "or_operator"
    BOOL_CONVERSION = "bool_conversion"
    RAILWAY_COMPOSITION = "railway_composition"


# =========================================================================
# Test Case Structures
# =========================================================================


@dataclass(frozen=True, slots=True)
class StringScenario:
    """String value scenario for FlextResult tests."""

    name: str
    operation_type: ResultOperationType
    value: str
    is_success_expected: bool = True


@dataclass(frozen=True, slots=True)
class IntScenario:
    """Integer value scenario for FlextResult tests."""

    name: str
    operation_type: ResultOperationType
    value: int
    is_success_expected: bool = True


@dataclass(frozen=True, slots=True)
class BoolScenario:
    """Boolean value scenario for FlextResult tests."""

    name: str
    operation_type: ResultOperationType
    value: bool


# =========================================================================
# Test Scenario Factories
# =========================================================================


class StringScenarios:
    """Factory for string-based FlextResult scenarios."""

    SCENARIOS: ClassVar[list[StringScenario]] = [
        StringScenario(
            name="creation_success_string",
            operation_type=ResultOperationType.CREATION_SUCCESS,
            value="success",
        ),
        StringScenario(
            name="creation_failure_message",
            operation_type=ResultOperationType.CREATION_FAILURE,
            value="error message",
            is_success_expected=False,
        ),
        StringScenario(
            name="unwrap_or_success",
            operation_type=ResultOperationType.UNWRAP_OR,
            value="value",
        ),
        StringScenario(
            name="unwrap_or_failure",
            operation_type=ResultOperationType.UNWRAP_OR,
            value="error",
            is_success_expected=False,
        ),
        StringScenario(
            name="map_failure",
            operation_type=ResultOperationType.MAP,
            value="error",
            is_success_expected=False,
        ),
        StringScenario(
            name="flat_map_failure",
            operation_type=ResultOperationType.FLAT_MAP,
            value="error",
            is_success_expected=False,
        ),
        StringScenario(
            name="alt_success",
            operation_type=ResultOperationType.ALT,
            value="success",
        ),
        StringScenario(
            name="alt_failure",
            operation_type=ResultOperationType.ALT,
            value="original_error",
            is_success_expected=False,
        ),
        StringScenario(
            name="lash_success",
            operation_type=ResultOperationType.LASH,
            value="success",
        ),
        StringScenario(
            name="lash_failure",
            operation_type=ResultOperationType.LASH,
            value="error",
            is_success_expected=False,
        ),
        StringScenario(
            name="or_operator_success",
            operation_type=ResultOperationType.OR_OPERATOR,
            value="value",
        ),
        StringScenario(
            name="or_operator_failure",
            operation_type=ResultOperationType.OR_OPERATOR,
            value="error",
            is_success_expected=False,
        ),
    ]


class IntScenarios:
    """Factory for integer-based FlextResult scenarios."""

    SCENARIOS: ClassVar[list[IntScenario]] = [
        IntScenario(
            name="unwrap_success",
            operation_type=ResultOperationType.UNWRAP,
            value=42,
        ),
        IntScenario(
            name="map_success",
            operation_type=ResultOperationType.MAP,
            value=5,
        ),
        IntScenario(
            name="flat_map_success",
            operation_type=ResultOperationType.FLAT_MAP,
            value=5,
        ),
        IntScenario(
            name="filter_passes",
            operation_type=ResultOperationType.FILTER,
            value=10,
        ),
        IntScenario(
            name="filter_fails",
            operation_type=ResultOperationType.FILTER,
            value=3,
            is_success_expected=False,
        ),
        IntScenario(
            name="railway_composition",
            operation_type=ResultOperationType.RAILWAY_COMPOSITION,
            value=5,
        ),
    ]


class BoolScenarios:
    """Factory for boolean-based FlextResult scenarios."""

    SCENARIOS: ClassVar[list[BoolScenario]] = [
        BoolScenario(
            name="bool_conversion_success",
            operation_type=ResultOperationType.BOOL_CONVERSION,
            value=True,
        ),
        BoolScenario(
            name="bool_conversion_failure",
            operation_type=ResultOperationType.BOOL_CONVERSION,
            value=False,
        ),
    ]


# =========================================================================
# Test Suite
# =========================================================================


class TestFlextResult:
    """Comprehensive test suite for FlextResult railway pattern operations."""

    @pytest.mark.parametrize(
        "scenario",
        StringScenarios.SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_result_string_operations(self, scenario: StringScenario) -> None:
        """Test FlextResult with string values across all scenarios."""
        if scenario.operation_type == ResultOperationType.CREATION_SUCCESS:
            result = FlextResult[str].ok(scenario.value)
            assert result.is_success
            assert not result.is_failure
            assert result.value == scenario.value
            assert result.error is None

        elif scenario.operation_type == ResultOperationType.CREATION_FAILURE:
            result = FlextResult[str].fail(scenario.value)
            assert result.is_failure
            assert not result.is_success
            assert result.error == scenario.value

        elif scenario.operation_type == ResultOperationType.UNWRAP_OR:
            if scenario.is_success_expected:
                result = FlextResult[str].ok(scenario.value)
                assert result.unwrap_or("default") == scenario.value
            else:
                result = FlextResult[str].fail(scenario.value)
                assert result.unwrap_or("default") == "default"

        elif scenario.operation_type == ResultOperationType.MAP:
            result = FlextResult[str].fail(scenario.value)
            mapped = result.map(lambda x: x * 2)
            assert mapped.is_failure
            assert mapped.error == scenario.value

        elif scenario.operation_type == ResultOperationType.FLAT_MAP:
            result = FlextResult[str].fail(scenario.value)
            flat_mapped = result.flat_map(
                lambda x: FlextResult[object].ok(f"value_{x}")
            )
            assert flat_mapped.is_failure
            assert flat_mapped.error == scenario.value

        elif scenario.operation_type == ResultOperationType.ALT:
            if scenario.is_success_expected:
                result = FlextResult[str].ok(scenario.value)
                alt_result = result.alt(lambda e: f"alt_{e}")
                assert alt_result.is_success
                assert alt_result.value == scenario.value
            else:
                result = FlextResult[str].fail(scenario.value)
                alt_result = result.alt(lambda e: f"alt_{e}")
                assert alt_result.is_failure
                assert alt_result.error == f"alt_{scenario.value}"

        elif scenario.operation_type == ResultOperationType.LASH:
            if scenario.is_success_expected:
                result = FlextResult[str].ok(scenario.value)
                lash_result = result.lash(
                    lambda e: FlextResult[str].ok(f"recovered_{e}")
                )
                assert lash_result.is_success
                assert lash_result.value == scenario.value
            else:
                result = FlextResult[str].fail(scenario.value)
                lash_result = result.lash(
                    lambda e: FlextResult[str].ok(f"recovered_{e}")
                )
                assert lash_result.is_success
                assert lash_result.value == f"recovered_{scenario.value}"

        elif scenario.operation_type == ResultOperationType.OR_OPERATOR:
            if scenario.is_success_expected:
                result = FlextResult[str].ok(scenario.value)
                assert (result | "default") == scenario.value
            else:
                result = FlextResult[str].fail(scenario.value)
                assert (result | "default") == "default"

    @pytest.mark.parametrize(
        "scenario",
        IntScenarios.SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_result_int_operations(self, scenario: IntScenario) -> None:
        """Test FlextResult with integer values across all scenarios."""
        if scenario.operation_type == ResultOperationType.UNWRAP:
            result = FlextResult[int].ok(scenario.value)
            assert result.unwrap() == scenario.value

        elif scenario.operation_type == ResultOperationType.MAP:
            result = FlextResult[int].ok(scenario.value)
            mapped = result.map(lambda x: x * 2)
            assert mapped.is_success
            assert mapped.value == scenario.value * 2

        elif scenario.operation_type == ResultOperationType.FLAT_MAP:
            result = FlextResult[int].ok(scenario.value)
            flat_mapped = result.flat_map(
                lambda x: FlextResult[object].ok(f"value_{x}")
            )
            assert flat_mapped.is_success
            assert flat_mapped.value == f"value_{scenario.value}"

        elif scenario.operation_type == ResultOperationType.FILTER:
            result = FlextResult[int].ok(scenario.value)
            filtered = result.filter(lambda x: x > 5)
            if scenario.is_success_expected:
                assert filtered.is_success
                assert filtered.value == scenario.value
            else:
                assert filtered.is_failure

        elif scenario.operation_type == ResultOperationType.RAILWAY_COMPOSITION:
            res1 = FlextResult[int].ok(scenario.value)
            res2 = res1.map(lambda v: v * 2)
            res3 = res2.map(lambda v: f"result_{v}")
            assert res3.is_success
            assert res3.value == "result_10"

    @pytest.mark.parametrize(
        "scenario",
        BoolScenarios.SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_result_bool_operations(self, scenario: BoolScenario) -> None:
        """Test FlextResult with boolean values across all scenarios."""
        if scenario.operation_type == ResultOperationType.BOOL_CONVERSION:
            if scenario.value:
                result = FlextResult[str].ok("value")
                assert bool(result) is True
            else:
                result = FlextResult[str].fail("error")
                assert bool(result) is False


__all__ = ["TestFlextResult"]
