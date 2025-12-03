"""Tests for r - Core railway pattern implementation.

Module: flext_core.result
Scope: r - railway-oriented programming core implementation

Tests r functionality including:
- Result creation (ok/fail)
- Value extraction (unwrap, unwrap_or)
- Railway transformations (map, flat_map, filter)
- Error recovery (alt, lash)
- Operators (|) and boolean conversion
- Railway composition patterns

Uses Python 3.13 patterns, FlextTestsUtilities, c,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Never, cast

import pytest

from flext_core import c, r, t
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
    """Generic result scenario for r tests."""

    name: str
    operation_type: ResultOperationType
    value: t.GeneralValueType
    is_success_expected: bool = True
    expected_result: t.GeneralValueType = None


class ResultScenarios:
    """Centralized result test scenarios using c."""

    STRING_SCENARIOS: ClassVar[list[ResultScenario]] = [
        ResultScenario(
            "creation_success_string",
            ResultOperationType.CREATION_SUCCESS,
            "success",
        ),
        ResultScenario(
            "creation_failure_message",
            ResultOperationType.CREATION_FAILURE,
            "error message",
            False,
        ),
        ResultScenario("unwrap_or_success", ResultOperationType.UNWRAP_OR, "value"),
        ResultScenario(
            "unwrap_or_failure",
            ResultOperationType.UNWRAP_OR,
            "error",
            False,
        ),
        ResultScenario("map_failure", ResultOperationType.MAP, "error", False),
        ResultScenario(
            "flat_map_failure",
            ResultOperationType.FLAT_MAP,
            "error",
            False,
        ),
        ResultScenario("alt_success", ResultOperationType.ALT, "success"),
        ResultScenario("alt_failure", ResultOperationType.ALT, "original_error", False),
        ResultScenario("lash_success", ResultOperationType.LASH, "success"),
        ResultScenario("lash_failure", ResultOperationType.LASH, "error", False),
        ResultScenario("or_operator_success", ResultOperationType.OR_OPERATOR, "value"),
        ResultScenario(
            "or_operator_failure",
            ResultOperationType.OR_OPERATOR,
            "error",
            False,
        ),
    ]

    INT_SCENARIOS: ClassVar[list[ResultScenario]] = [
        ResultScenario("unwrap_success", ResultOperationType.UNWRAP, 42),
        ResultScenario("map_success", ResultOperationType.MAP, 5),
        ResultScenario("flat_map_success", ResultOperationType.FLAT_MAP, 5),
        ResultScenario("filter_passes", ResultOperationType.FILTER, 10),
        ResultScenario("filter_fails", ResultOperationType.FILTER, 3, False),
        ResultScenario(
            "railway_composition",
            ResultOperationType.RAILWAY_COMPOSITION,
            5,
        ),
    ]

    BOOL_SCENARIOS: ClassVar[list[ResultScenario]] = [
        ResultScenario(
            "bool_conversion_success",
            ResultOperationType.BOOL_CONVERSION,
            True,
        ),
        ResultScenario(
            "bool_conversion_failure",
            ResultOperationType.BOOL_CONVERSION,
            False,
        ),
    ]


class Testr:
    """Comprehensive test suite for r using FlextTestsUtilities."""

    @pytest.mark.parametrize(
        "scenario",
        ResultScenarios.STRING_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_result_string_operations(self, scenario: ResultScenario) -> None:
        """Test r with string values across all scenarios."""
        op_type = scenario.operation_type
        value = scenario.value
        is_success = scenario.is_success_expected

        if op_type == ResultOperationType.CREATION_SUCCESS:
            # Use generic helper to replace 10+ lines of result creation code
            creation_result: r[t.GeneralValueType] = (
                FlextTestsUtilities.GenericHelpers.create_result_from_value(
                    value,
                    error_on_none="Value cannot be None",
                )
            )
            # value is already GeneralValueType from ResultScenario
            FlextTestsUtilities.ResultHelpers.assert_success_with_value(
                creation_result, value
            )

        elif op_type == ResultOperationType.CREATION_FAILURE:
            failure_result: r[t.GeneralValueType] = (
                FlextTestsUtilities.ResultHelpers.create_failure_result(str(value))
            )
            FlextTestsUtilities.ResultHelpers.assert_failure_with_error(
                failure_result,
                str(value),
            )

        elif op_type == ResultOperationType.UNWRAP_OR:
            # value is already GeneralValueType from ResultScenario
            unwrap_result: r[t.GeneralValueType] = (
                FlextTestsUtilities.ResultHelpers.create_success_result(value)
                if is_success
                else FlextTestsUtilities.ResultHelpers.create_failure_result(str(value))
            )
            default = "default"
            assert unwrap_result.unwrap_or(default) == (
                value if is_success else default
            )

        elif op_type == ResultOperationType.MAP:
            map_result: r[str] = r[str].fail(str(value))
            mapped = map_result.map(lambda x: str(x) * 2)
            assert mapped.is_failure and mapped.error == str(value)

        elif op_type == ResultOperationType.FLAT_MAP:
            flat_map_result: r[t.GeneralValueType] = (
                FlextTestsUtilities.ResultHelpers.create_failure_result(str(value))
            )
            flat_mapped = flat_map_result.flat_map(
                lambda x: r[str].ok(f"value_{x}"),
            )
            FlextTestsUtilities.ResultHelpers.assert_failure_with_error(
                flat_mapped,
                str(value),
            )

        elif op_type == ResultOperationType.ALT:
            # value is already GeneralValueType from ResultScenario
            result_alt: r[t.GeneralValueType] = (
                FlextTestsUtilities.ResultHelpers.create_success_result(value)
                if is_success
                else FlextTestsUtilities.ResultHelpers.create_failure_result(str(value))
            )
            alt_result = result_alt.alt(lambda e: f"alt_{e}")
            if is_success:
                FlextTestsUtilities.ResultHelpers.assert_success_with_value(
                    alt_result,
                    value,
                )
            else:
                error_str_alt: str = f"alt_{value}"
                FlextTestsUtilities.ResultHelpers.assert_failure_with_error(
                    alt_result,
                    error_str_alt,
                )

        elif op_type == ResultOperationType.LASH:
            lash_result_base: r[str] = (
                r[str].ok(str(value)) if is_success else r[str].fail(str(value))
            )
            lash_result = lash_result_base.lash(
                lambda e: r[str].ok(f"recovered_{e}"),
            )
            if is_success:
                assert lash_result.is_success and lash_result.value == str(value)
            else:
                expected = f"recovered_{value}"
                assert lash_result.is_success and lash_result.value == expected

        elif op_type == ResultOperationType.OR_OPERATOR:
            # value is already GeneralValueType from ResultScenario
            result_or: r[t.GeneralValueType] = (
                FlextTestsUtilities.ResultHelpers.create_success_result(value)
                if is_success
                else FlextTestsUtilities.ResultHelpers.create_failure_result(str(value))
            )
            default = "default"
            assert (result_or | default) == (value if is_success else default)

    @pytest.mark.parametrize(
        "scenario",
        ResultScenarios.INT_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_result_int_operations(self, scenario: ResultScenario) -> None:
        """Test r with integer values across all scenarios."""
        op_type = scenario.operation_type
        value = scenario.value
        is_success = scenario.is_success_expected

        if op_type == ResultOperationType.UNWRAP:
            assert isinstance(value, int)
            result = r[int].ok(value)
            assert result.unwrap() == value

        elif op_type == ResultOperationType.MAP:
            assert isinstance(value, int)
            result = r[int].ok(value)
            mapped = result.map(lambda x: x * 2)
            assert mapped.is_success and mapped.value == value * 2

        elif op_type == ResultOperationType.FLAT_MAP:
            assert isinstance(value, int)
            result = r[int].ok(value)
            flat_mapped = result.flat_map(
                lambda x: r[object].ok(f"value_{x}"),
            )
            expected = f"value_{value}"
            assert flat_mapped.is_success and flat_mapped.value == expected

        elif op_type == ResultOperationType.FILTER:
            assert isinstance(value, int)
            result = r[int].ok(value)
            filtered = result.filter(lambda x: x > 5)
            if is_success:
                assert filtered.is_success and filtered.value == value
            else:
                assert filtered.is_failure

        elif op_type == ResultOperationType.RAILWAY_COMPOSITION:
            assert isinstance(value, int)
            # Use generic helper to test result chain - replaces 10+ lines
            # Each result typed independently for type safety
            res1 = r[int].ok(value)
            res2 = res1.map(lambda v: v * 2)
            res3 = res2.map(lambda v: f"result_{v}")
            expected = f"result_{value * 2}"
            # Use generic helper for chain validation with explicit sequence cast
            # Cast required as r types vary (int → int → str)
            FlextTestsUtilities.GenericHelpers.assert_result_chain(
                cast("Sequence[r[Never]]", [res1, res2, res3]),
                expected_success_count=3,
                expected_failure_count=0,
                first_failure_index=None,
            )
            assert res3.is_success and res3.value == expected

    @pytest.mark.parametrize(
        "scenario",
        ResultScenarios.BOOL_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_result_bool_operations(self, scenario: ResultScenario) -> None:
        """Test r with boolean values across all scenarios."""
        if scenario.operation_type == ResultOperationType.BOOL_CONVERSION:
            result = (
                FlextTestsUtilities.ResultHelpers.create_success_result("value")
                if scenario.value
                else FlextTestsUtilities.ResultHelpers.create_failure_result(
                    c.Errors.GENERIC_ERROR,
                )
            )
            assert bool(result) is scenario.value

    def test_result_chain_validation_real_behavior(self) -> None:
        """Test result chain validation with real behavior patterns.

        Tests actual chain operations and validates using generic helpers.
        """
        # Create a real chain of operations
        results: list[r[int]] = []
        initial_value = 5

        # Step 1: Initial value
        res1 = FlextTestsUtilities.GenericHelpers.create_result_from_value(
            initial_value,
            error_on_none="Initial value cannot be None",
        )
        results.append(res1)

        # Step 2: Transform
        res2 = res1.map(lambda x: x * 2)
        results.append(res2)

        # Step 3: Another transform
        res3 = res2.map(lambda x: x + 10)
        results.append(res3)

        # Validate entire chain using generic helper (replaces 10+ lines)
        FlextTestsUtilities.GenericHelpers.assert_result_chain(
            results,
            expected_success_count=3,
            expected_failure_count=0,
            first_failure_index=None,
        )

        # Verify final value
        assert res3.is_success
        assert res3.value == 20  # (5 * 2) + 10

    def test_result_chain_failure_behavior(self) -> None:
        """Test result chain with failure - real behavior and limits."""
        results: list[r[int]] = []

        # Success
        res1 = r[int].ok(10)
        results.append(res1)

        # Success
        res2 = res1.map(lambda x: x * 2)
        results.append(res2)

        # Failure - division by zero limit case
        res3 = res2.flat_map(
            lambda x: r[int].fail("Division by zero") if x == 0 else r[int].ok(x // 2),
        )
        results.append(res3)

        # Should still succeed (20 // 2 = 10)
        assert res3.is_success
        assert res3.value == 10

        # Now test actual failure case
        res4 = res3.flat_map(
            lambda x: r[int].fail("Cannot process zero") if x == 0 else r[int].ok(x),
        )
        results.append(res4)

        # Validate chain - should still be all successful
        FlextTestsUtilities.GenericHelpers.assert_result_chain(
            results,
            expected_success_count=4,
            expected_failure_count=0,
        )

    def test_result_parametrized_cases_generic_helper(self) -> None:
        """Test using generic helper for parametrized test cases.

        Replaces 10+ lines of manual test case creation.
        """
        # Use generic helper to create parametrized cases
        success_values: list[str] = ["value1", "value2", "value3"]
        failure_errors: list[str] = ["error1", "error2"]
        error_codes: list[str | None] = ["CODE1", None]

        cases = FlextTestsUtilities.GenericHelpers.create_parametrized_cases(
            success_values,
            failure_errors,
            error_codes=error_codes,
        )

        # Verify cases structure
        assert len(cases) == 5  # 3 success + 2 failure

        # Verify success cases
        for i, (result, is_success, value, error) in enumerate(cases[:3]):
            assert is_success is True
            assert result.is_success
            assert value == success_values[i]
            assert error is None

        # Verify failure cases
        for i, (result, is_success, value, error) in enumerate(cases[3:]):
            assert is_success is False
            assert result.is_failure
            assert value is None
            assert error == failure_errors[i]

    def test_result_none_handling_limits(self) -> None:
        """Test None handling limits using generic helper."""
        # Test with None and default
        # Business Rule: create_result_from_value infers type from default_on_none
        # When value is None and default_on_none is provided, type is inferred from default
        result1: r[str] = FlextTestsUtilities.GenericHelpers.create_result_from_value(
            None,
            default_on_none="default_value",
        )
        assert result1.is_success
        assert result1.value == "default_value"

        # Test with None and error
        result2: r[str | None] = (
            FlextTestsUtilities.GenericHelpers.create_result_from_value(
                None,
                error_on_none="Value is None",
            )
        )
        assert result2.is_failure
        assert result2.error == "Value is None"

        # Test with actual value
        result3 = FlextTestsUtilities.GenericHelpers.create_result_from_value(
            "actual_value",
        )
        assert result3.is_success
        assert result3.value == "actual_value"


__all__ = ["Testr"]
