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

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

import pytest

from flext_core import FlextResult
from flext_core.constants import FlextConstants
from flext_tests.utilities import FlextTestsUtilities


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


@dataclass(frozen=True, slots=True)
class ResultScenario:
    """Generic result scenario for FlextResult tests."""

    name: str
    operation_type: ResultOperationType
    value: object
    is_success_expected: bool = True
    expected_result: object | None = None


class ResultScenarios:
    """Centralized result test scenarios using FlextConstants."""

    STRING_SCENARIOS: ClassVar[list[ResultScenario]] = [
        ResultScenario(
            "creation_success_string", ResultOperationType.CREATION_SUCCESS, "success",
        ),
        ResultScenario(
            "creation_failure_message",
            ResultOperationType.CREATION_FAILURE,
            "error message",
            False,
        ),
        ResultScenario("unwrap_or_success", ResultOperationType.UNWRAP_OR, "value"),
        ResultScenario(
            "unwrap_or_failure", ResultOperationType.UNWRAP_OR, "error", False,
        ),
        ResultScenario("map_failure", ResultOperationType.MAP, "error", False),
        ResultScenario(
            "flat_map_failure", ResultOperationType.FLAT_MAP, "error", False,
        ),
        ResultScenario("alt_success", ResultOperationType.ALT, "success"),
        ResultScenario("alt_failure", ResultOperationType.ALT, "original_error", False),
        ResultScenario("lash_success", ResultOperationType.LASH, "success"),
        ResultScenario("lash_failure", ResultOperationType.LASH, "error", False),
        ResultScenario("or_operator_success", ResultOperationType.OR_OPERATOR, "value"),
        ResultScenario(
            "or_operator_failure", ResultOperationType.OR_OPERATOR, "error", False,
        ),
    ]

    INT_SCENARIOS: ClassVar[list[ResultScenario]] = [
        ResultScenario("unwrap_success", ResultOperationType.UNWRAP, 42),
        ResultScenario("map_success", ResultOperationType.MAP, 5),
        ResultScenario("flat_map_success", ResultOperationType.FLAT_MAP, 5),
        ResultScenario("filter_passes", ResultOperationType.FILTER, 10),
        ResultScenario("filter_fails", ResultOperationType.FILTER, 3, False),
        ResultScenario(
            "railway_composition", ResultOperationType.RAILWAY_COMPOSITION, 5,
        ),
    ]

    BOOL_SCENARIOS: ClassVar[list[ResultScenario]] = [
        ResultScenario(
            "bool_conversion_success", ResultOperationType.BOOL_CONVERSION, True,
        ),
        ResultScenario(
            "bool_conversion_failure", ResultOperationType.BOOL_CONVERSION, False,
        ),
    ]


class TestFlextResult:
    """Comprehensive test suite for FlextResult using FlextTestsUtilities."""

    @pytest.mark.parametrize(
        "scenario", ResultScenarios.STRING_SCENARIOS, ids=lambda s: s.name,
    )
    def test_result_string_operations(self, scenario: ResultScenario) -> None:
        """Test FlextResult with string values across all scenarios."""
        op_type = scenario.operation_type
        value = scenario.value
        is_success = scenario.is_success_expected

        if op_type == ResultOperationType.CREATION_SUCCESS:
            result = FlextTestsUtilities.ResultHelpers.create_success_result(value)
            FlextTestsUtilities.ResultHelpers.assert_success_with_value(result, value)

        elif op_type == ResultOperationType.CREATION_FAILURE:
            result = FlextTestsUtilities.ResultHelpers.create_failure_result(str(value))
            FlextTestsUtilities.ResultHelpers.assert_failure_with_error(
                result, str(value),
            )

        elif op_type == ResultOperationType.UNWRAP_OR:
            result = (
                FlextTestsUtilities.ResultHelpers.create_success_result(value)
                if is_success
                else FlextTestsUtilities.ResultHelpers.create_failure_result(str(value))
            )
            default = "default"
            assert result.unwrap_or(default) == (value if is_success else default)

        elif op_type == ResultOperationType.MAP:
            map_result: FlextResult[str] = FlextResult[str].fail(str(value))
            mapped = map_result.map(lambda x: str(x) * 2)
            assert mapped.is_failure and mapped.error == str(value)

        elif op_type == ResultOperationType.FLAT_MAP:
            result = FlextTestsUtilities.ResultHelpers.create_failure_result(str(value))
            flat_mapped = result.flat_map(
                lambda x: FlextResult[object].ok(f"value_{x}"),
            )
            FlextTestsUtilities.ResultHelpers.assert_failure_with_error(
                flat_mapped, str(value),
            )

        elif op_type == ResultOperationType.ALT:
            result = (
                FlextTestsUtilities.ResultHelpers.create_success_result(value)
                if is_success
                else FlextTestsUtilities.ResultHelpers.create_failure_result(str(value))
            )
            alt_result = result.alt(lambda e: f"alt_{e}")
            if is_success:
                FlextTestsUtilities.ResultHelpers.assert_success_with_value(
                    alt_result, value,
                )
            else:
                FlextTestsUtilities.ResultHelpers.assert_failure_with_error(
                    alt_result, f"alt_{value}",
                )

        elif op_type == ResultOperationType.LASH:
            lash_result_base: FlextResult[str] = (
                FlextResult[str].ok(str(value))
                if is_success
                else FlextResult[str].fail(str(value))
            )
            lash_result = lash_result_base.lash(
                lambda e: FlextResult[str].ok(f"recovered_{e}"),
            )
            if is_success:
                assert lash_result.is_success and lash_result.value == str(value)
            else:
                expected = f"recovered_{value}"
                assert lash_result.is_success and lash_result.value == expected

        elif op_type == ResultOperationType.OR_OPERATOR:
            result = (
                FlextTestsUtilities.ResultHelpers.create_success_result(value)
                if is_success
                else FlextTestsUtilities.ResultHelpers.create_failure_result(str(value))
            )
            default = "default"
            assert (result | default) == (value if is_success else default)

    @pytest.mark.parametrize(
        "scenario", ResultScenarios.INT_SCENARIOS, ids=lambda s: s.name,
    )
    def test_result_int_operations(self, scenario: ResultScenario) -> None:
        """Test FlextResult with integer values across all scenarios."""
        op_type = scenario.operation_type
        value = scenario.value
        is_success = scenario.is_success_expected

        if op_type == ResultOperationType.UNWRAP:
            assert isinstance(value, int)
            result = FlextResult[int].ok(value)
            assert result.unwrap() == value

        elif op_type == ResultOperationType.MAP:
            assert isinstance(value, int)
            result = FlextResult[int].ok(value)
            mapped = result.map(lambda x: x * 2)
            assert mapped.is_success and mapped.value == value * 2

        elif op_type == ResultOperationType.FLAT_MAP:
            assert isinstance(value, int)
            result = FlextResult[int].ok(value)
            flat_mapped = result.flat_map(
                lambda x: FlextResult[object].ok(f"value_{x}"),
            )
            expected = f"value_{value}"
            assert flat_mapped.is_success and flat_mapped.value == expected

        elif op_type == ResultOperationType.FILTER:
            assert isinstance(value, int)
            result = FlextResult[int].ok(value)
            filtered = result.filter(lambda x: x > 5)
            if is_success:
                assert filtered.is_success and filtered.value == value
            else:
                assert filtered.is_failure

        elif op_type == ResultOperationType.RAILWAY_COMPOSITION:
            assert isinstance(value, int)
            res1 = FlextResult[int].ok(value)
            res2 = res1.map(lambda v: v * 2)
            res3 = res2.map(lambda v: f"result_{v}")
            expected = f"result_{value * 2}"
            assert res3.is_success and res3.value == expected

    @pytest.mark.parametrize(
        "scenario", ResultScenarios.BOOL_SCENARIOS, ids=lambda s: s.name,
    )
    def test_result_bool_operations(self, scenario: ResultScenario) -> None:
        """Test FlextResult with boolean values across all scenarios."""
        if scenario.operation_type == ResultOperationType.BOOL_CONVERSION:
            result = (
                FlextTestsUtilities.ResultHelpers.create_success_result("value")
                if scenario.value
                else FlextTestsUtilities.ResultHelpers.create_failure_result(
                    FlextConstants.Errors.GENERIC_ERROR,
                )
            )
            assert bool(result) is scenario.value


__all__ = ["TestFlextResult"]
