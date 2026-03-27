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

Uses Python 3.13 patterns, u, c,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from enum import StrEnum, unique
from typing import Annotated, ClassVar, cast

import pytest
from flext_tests import t, tm, u
from hypothesis import given, settings, strategies as st
from pydantic import BaseModel, ConfigDict, Field

from flext_core import r

from ..test_utils import assertion_helpers


@unique
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


class Testr:
    ResultOperationType = ResultOperationType

    class ResultScenario(BaseModel):
        """Generic result scenario for r tests."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Result scenario name")]
        operation_type: Annotated[
            ResultOperationType,
            Field(description="Result operation type"),
        ]
        value: Annotated[
            t.NormalizedValue,
            Field(description="Input value for result operation"),
        ]
        is_success_expected: Annotated[
            bool,
            Field(default=True, description="Expected success state"),
        ] = True
        expected_result: Annotated[
            t.NormalizedValue | None,
            Field(default=None, description="Optional expected result payload"),
        ] = None

        def __init__(
            self,
            name: str,
            operation_type: ResultOperationType,
            value: t.NormalizedValue,
            *,
            is_success_expected: bool = True,
            expected_result: t.NormalizedValue | None = None,
        ) -> None:
            super().__init__(
                name=name,
                operation_type=operation_type,
                value=value,
                is_success_expected=is_success_expected,
                expected_result=expected_result,
            )

    STRING_SCENARIOS: ClassVar[Sequence[Testr.ResultScenario]] = [
        ResultScenario(
            name="creation_success_string",
            operation_type=ResultOperationType.CREATION_SUCCESS,
            value="success",
        ),
        ResultScenario(
            name="creation_failure_message",
            operation_type=ResultOperationType.CREATION_FAILURE,
            value="error message",
            is_success_expected=False,
        ),
        ResultScenario(
            name="unwrap_or_success",
            operation_type=ResultOperationType.UNWRAP_OR,
            value="value",
        ),
        ResultScenario(
            name="unwrap_or_failure",
            operation_type=ResultOperationType.UNWRAP_OR,
            value="error",
            is_success_expected=False,
        ),
        ResultScenario(
            name="map_failure",
            operation_type=ResultOperationType.MAP,
            value="error",
            is_success_expected=False,
        ),
        ResultScenario(
            name="flat_map_failure",
            operation_type=ResultOperationType.FLAT_MAP,
            value="error",
            is_success_expected=False,
        ),
        ResultScenario(
            name="alt_success",
            operation_type=ResultOperationType.ALT,
            value="success",
        ),
        ResultScenario(
            name="alt_failure",
            operation_type=ResultOperationType.ALT,
            value="original_error",
            is_success_expected=False,
        ),
        ResultScenario(
            name="lash_success",
            operation_type=ResultOperationType.LASH,
            value="success",
        ),
        ResultScenario(
            name="lash_failure",
            operation_type=ResultOperationType.LASH,
            value="error",
            is_success_expected=False,
        ),
        ResultScenario(
            name="or_operator_success",
            operation_type=ResultOperationType.OR_OPERATOR,
            value="value",
        ),
        ResultScenario(
            name="or_operator_failure",
            operation_type=ResultOperationType.OR_OPERATOR,
            value="error",
            is_success_expected=False,
        ),
    ]
    INT_SCENARIOS: ClassVar[Sequence[Testr.ResultScenario]] = [
        ResultScenario(
            name="unwrap_success",
            operation_type=ResultOperationType.UNWRAP,
            value=42,
        ),
        ResultScenario(
            name="map_success",
            operation_type=ResultOperationType.MAP,
            value=5,
        ),
        ResultScenario(
            name="flat_map_success",
            operation_type=ResultOperationType.FLAT_MAP,
            value=5,
        ),
        ResultScenario(
            name="filter_passes",
            operation_type=ResultOperationType.FILTER,
            value=10,
        ),
        ResultScenario(
            name="filter_fails",
            operation_type=ResultOperationType.FILTER,
            value=3,
            is_success_expected=False,
        ),
        ResultScenario(
            name="railway_composition",
            operation_type=ResultOperationType.RAILWAY_COMPOSITION,
            value=5,
        ),
    ]
    BOOL_SCENARIOS: ClassVar[Sequence[Testr.ResultScenario]] = [
        ResultScenario(
            name="bool_conversion_success",
            operation_type=ResultOperationType.BOOL_CONVERSION,
            value=True,
        ),
        ResultScenario(
            name="bool_conversion_failure",
            operation_type=ResultOperationType.BOOL_CONVERSION,
            value=False,
        ),
    ]

    @pytest.mark.parametrize("scenario", STRING_SCENARIOS, ids=lambda s: s.name)
    def test_result_string_operations(self, scenario: ResultScenario) -> None:
        """Test r with string values across all scenarios."""
        op_type = scenario.operation_type
        value = scenario.value
        is_success = scenario.is_success_expected
        if not isinstance(value, str):
            pytest.fail("Expected string scenario value")
        if op_type == self.ResultOperationType.CREATION_SUCCESS:
            creation_result: r[str] = u.Tests.GenericHelpers.create_result_from_value(
                value,
                error_on_none="Value cannot be None",
            )
            u.Tests.Result.assert_success_with_value(creation_result, value)
        elif op_type == self.ResultOperationType.CREATION_FAILURE:
            failure_result_raw = u.Tests.Result.create_failure_result(str(value))
            failure_result: r[str] = failure_result_raw
            u.Tests.Result.assert_failure_with_error(failure_result, str(value))
        elif op_type == self.ResultOperationType.UNWRAP_OR:
            if is_success:
                unwrap_result: r[str] = u.Tests.Result.create_success_result(value)
            else:
                failure_raw = u.Tests.Result.create_failure_result(str(value))
                unwrap_result = failure_raw
            default = "default"
            tm.that(
                unwrap_result.unwrap_or(default),
                eq=value if is_success else default,
            )
        elif op_type == self.ResultOperationType.MAP:
            map_result: r[str] = r[str].fail(str(value))
            mapped = map_result.map(lambda x: str(x) * 2)
            u.Tests.Result.assert_failure_with_error(mapped, str(value))
        elif op_type == self.ResultOperationType.FLAT_MAP:
            failure_raw = u.Tests.Result.create_failure_result(str(value))
            flat_map_result: r[str] = failure_raw
            flat_mapped = flat_map_result.flat_map(lambda x: r[str].ok(f"value_{x}"))
            u.Tests.Result.assert_failure_with_error(flat_mapped, str(value))
        elif op_type == self.ResultOperationType.ALT:
            if is_success:
                result_alt: r[str] = u.Tests.Result.create_success_result(value)
            else:
                failure_raw = u.Tests.Result.create_failure_result(str(value))
                result_alt = failure_raw
            alt_result = result_alt.map_error(lambda e: f"alt_{e}")
            if is_success:
                u.Tests.Result.assert_success_with_value(alt_result, value)
            else:
                error_str_alt: str = f"alt_{value}"
                u.Tests.Result.assert_failure_with_error(alt_result, error_str_alt)
        elif op_type == self.ResultOperationType.LASH:
            lash_result_base: r[str] = (
                r[str].ok(str(value)) if is_success else r[str].fail(str(value))
            )
            lash_result = lash_result_base.lash(lambda e: r[str].ok(f"recovered_{e}"))
            if is_success:
                u.Tests.Result.assert_success_with_value(lash_result, str(value))
            else:
                expected = f"recovered_{value}"
                u.Tests.Result.assert_success_with_value(lash_result, expected)
        elif op_type == self.ResultOperationType.OR_OPERATOR:
            if is_success:
                result_or: r[str] = u.Tests.Result.create_success_result(value)
            else:
                failure_raw = u.Tests.Result.create_failure_result(str(value))
                result_or = failure_raw
            default = "default"
            tm.that(result_or | default, eq=value if is_success else default)

    @pytest.mark.parametrize("scenario", INT_SCENARIOS, ids=lambda s: s.name)
    def test_result_int_operations(self, scenario: ResultScenario) -> None:
        """Test r with integer values across all scenarios."""
        op_type = scenario.operation_type
        value = scenario.value
        is_success = scenario.is_success_expected
        if op_type == self.ResultOperationType.UNWRAP:
            if not isinstance(value, int):
                pytest.fail("Expected integer scenario value")
            result = r[int].ok(value)
            tm.that(result.value, eq=value)
        elif op_type == self.ResultOperationType.MAP:
            if not isinstance(value, int):
                pytest.fail("Expected integer scenario value")
            result = r[int].ok(value)
            mapped = result.map(lambda x: x * 2)
            u.Tests.Result.assert_success_with_value(mapped, value * 2)
        elif op_type == self.ResultOperationType.FLAT_MAP:
            if not isinstance(value, int):
                pytest.fail("Expected integer scenario value")
            result = r[int].ok(value)
            flat_mapped = result.flat_map(lambda x: r[str].ok(f"value_{x}"))
            expected = f"value_{value}"
            u.Tests.Result.assert_success_with_value(flat_mapped, expected)
        elif op_type == self.ResultOperationType.FILTER:
            if not isinstance(value, int):
                pytest.fail("Expected integer scenario value")
            result = r[int].ok(value)
            filtered = result.filter(lambda x: x > 5)
            if is_success:
                u.Tests.Result.assert_success_with_value(filtered, value)
            else:
                _ = u.Tests.Result.assert_failure(filtered)
        elif op_type == self.ResultOperationType.RAILWAY_COMPOSITION:
            if not isinstance(value, int):
                pytest.fail("Expected integer scenario value")
            res1 = r[int].ok(value)
            res2 = res1.map(lambda v: v * 2)
            res3 = res2.map(lambda v: f"result_{v}")
            expected = f"result_{value * 2}"
            result_list: Sequence[r[str]] = [res1.map(str), res2.map(str), res3]
            u.Tests.GenericHelpers.assert_result_chain(
                result_list,
                expected_success_count=3,
                expected_failure_count=0,
                first_failure_index=None,
            )
            u.Tests.Result.assert_success_with_value(res3, expected)

    @pytest.mark.parametrize("scenario", BOOL_SCENARIOS, ids=lambda s: s.name)
    def test_result_bool_operations(self, scenario: ResultScenario) -> None:
        """Test r with boolean values across all scenarios."""
        if scenario.operation_type == self.ResultOperationType.BOOL_CONVERSION:
            result = (
                u.Tests.Result.create_success_result("value")
                if scenario.value
                else u.Tests.Result.create_failure_result("generic_error")
            )
            tm.that(bool(result), eq=bool(scenario.value))

    def test_result_chain_validation_real_behavior(self) -> None:
        """Test result chain validation with real behavior patterns.

        Tests actual chain operations and validates using generic helpers.
        """
        results: MutableSequence[r[int]] = []
        initial_value = 5
        res1 = u.Tests.GenericHelpers.create_result_from_value(
            initial_value,
            error_on_none="Initial value cannot be None",
        )
        results.append(res1)
        res2 = res1.map(lambda x: x * 2)
        results.append(res2)
        res3 = res2.map(lambda x: x + 10)
        results.append(res3)
        u.Tests.GenericHelpers.assert_result_chain(
            results,
            expected_success_count=3,
            expected_failure_count=0,
            first_failure_index=None,
        )
        u.Tests.Result.assert_success_with_value(res3, 20)

    def test_result_chain_failure_behavior(self) -> None:
        """Test result chain with failure - real behavior and limits."""
        results: MutableSequence[r[int]] = []
        res1 = r[int].ok(10)
        results.append(res1)
        res2 = res1.map(lambda x: x * 2)
        results.append(res2)
        res3 = res2.flat_map(
            lambda x: r[int].fail("Division by zero") if x == 0 else r[int].ok(x // 2),
        )
        results.append(res3)
        u.Tests.Result.assert_success_with_value(res3, 10)
        res4 = res3.flat_map(
            lambda x: r[int].fail("Cannot process zero") if x == 0 else r[int].ok(x),
        )
        results.append(res4)
        u.Tests.GenericHelpers.assert_result_chain(
            results,
            expected_success_count=4,
            expected_failure_count=0,
        )

    def test_result_parametrized_cases_generic_helper(self) -> None:
        """Test using generic helper for parametrized test cases.

        Replaces 10+ lines of manual test case creation.
        """
        success_values: t.ContainerList = ["value1", "value2", "value3"]
        failure_errors: t.StrSequence = ["error1", "error2"]
        error_codes: Sequence[str | None] = ["CODE1", None]
        cases = u.Tests.GenericHelpers.create_parametrized_cases(
            success_values,
            failure_errors,
            error_codes=error_codes,
        )
        tm.that(len(cases), eq=5)
        for i, (result, is_success, _value, error) in enumerate(cases[:3]):
            tm.that(is_success, eq=True)
            u.Tests.Result.assert_success_with_value(result, success_values[i])
            tm.that(error, none=True)
        for i, (result, is_success, _value, error) in enumerate(cases[3:]):
            tm.that(not is_success, eq=True)
            _ = u.Tests.Result.assert_failure(result)
            tm.that(error, eq=failure_errors[i])

    def test_result_none_handling_limits(self) -> None:
        """Test None handling limits using generic helper."""
        result1: r[str] = u.Tests.GenericHelpers.create_result_from_value(
            None,
            default_on_none="default_value",
        )
        u.Tests.Result.assert_success_with_value(result1, "default_value")
        result2: r[str | None] = u.Tests.GenericHelpers.create_result_from_value(
            None,
            error_on_none="Value is None",
        )
        u.Tests.Result.assert_failure_with_error(result2, "Value is None")
        result3 = u.Tests.GenericHelpers.create_result_from_value("actual_value")
        u.Tests.Result.assert_success_with_value(result3, "actual_value")

    def test_safe_decorator(self) -> None:
        """Test safe decorator wraps function in try/except."""

        def divide(a: int, b: int) -> int:
            return a // b

        divide_wrapped = r.safe(divide)
        result: r[int] = divide_wrapped(10, 2)
        _ = assertion_helpers.assert_flext_result_success(result)
        tm.that(result.value, eq=5)
        result_fail: r[int] = divide_wrapped(10, 0)
        tm.fail(result_fail)

    def test_map_error(self) -> None:
        """Test map_error transforms error message."""
        result: r[str] = r[str].fail("original error")
        transformed = result.map_error(lambda e: f"PREFIX: {e}")
        tm.fail(transformed)
        tm.that(transformed.error, eq="PREFIX: original error")
        success = r[str].ok("value")
        unchanged = success.map_error(lambda e: f"PREFIX: {e}")
        tm.ok(unchanged)
        tm.that(unchanged.value, eq="value")

    def test_filter_success(self) -> None:
        """Test filter with success result."""
        result = r[int].ok(10)
        filtered = result.filter(lambda x: x > 5)
        tm.ok(filtered)
        tm.that(filtered.value, eq=10)
        filtered_fail = result.filter(lambda x: x > 20)
        tm.fail(filtered_fail)

    def test_filter_failure(self) -> None:
        """Test filter with failure result returns unchanged."""
        result: r[int] = r[int].fail("error")
        filtered = result.filter(lambda x: x > 5)
        tm.fail(filtered)
        tm.that(filtered.error, eq="error")

    def test_flow_through(self) -> None:
        """Test flow_through chains multiple operations."""

        def add_one(x: int) -> r[int]:
            return r[int].ok(x + 1)

        def multiply_two(x: int) -> r[int]:
            return r[int].ok(x * 2)

        result = r[int].ok(5)
        final = result.flow_through(add_one, multiply_two)
        tm.ok(final)
        tm.that(final.value, eq=12)

    def test_flow_through_failure(self) -> None:
        """Test flow_through stops on first failure."""

        def add_one(x: int) -> r[int]:
            return r[int].ok(x + 1)

        def fail_op(_x: int) -> r[int]:
            return r[int].fail("error")

        def multiply_two(x: int) -> r[int]:
            return r[int].ok(x * 2)

        result = r[int].ok(5)
        final = result.flow_through(add_one, fail_op, multiply_two)
        tm.fail(final)
        tm.that(final.error, eq="error")

    def test_traverse_success(self) -> None:
        """Test traverse maps over sequence successfully."""
        items = [1, 2, 3]
        result = r.traverse(items, lambda x: r[int].ok(x * 2))
        _ = assertion_helpers.assert_flext_result_success(result)
        tm.that(result.value, eq=[2, 4, 6])

    def test_traverse_failure(self) -> None:
        """Test traverse fails fast on first failure."""
        items = [1, 2, 3]
        result = r.traverse(
            items,
            lambda x: r[int].fail("error") if x == 2 else r[int].ok(x),
        )
        _ = assertion_helpers.assert_flext_result_failure(result)
        tm.that(result.error, eq="error")

    def test_accumulate_errors_all_success(self) -> None:
        """Test accumulate_errors with all successes."""
        results = [r[int].ok(1), r[int].ok(2), r[int].ok(3)]
        accumulated = r.accumulate_errors(*results)
        tm.ok(accumulated)
        tm.that(accumulated.value, eq=[1, 2, 3])

    def test_accumulate_errors_with_failures(self) -> None:
        """Test accumulate_errors collects all errors."""
        results = [r[int].ok(1), r[int].fail("error1"), r[int].fail("error2")]
        accumulated = r.accumulate_errors(*results)
        tm.fail(accumulated)
        tm.that(accumulated.error, none=False)
        tm.that(str(accumulated.error), has="error1")
        tm.that(str(accumulated.error), has="error2")

    def test_traverse_fail_fast_true(self) -> None:
        """Test traverse with fail_fast=True (default) stops on first failure."""
        items = [1, 2, 3]
        result = r.traverse(
            items,
            lambda x: r[int].fail("error") if x == 2 else r[int].ok(x),
            fail_fast=True,
        )
        _ = assertion_helpers.assert_flext_result_failure(result)
        tm.that(result.error, eq="error")

    def test_traverse_fail_fast_false(self) -> None:
        """Test traverse with fail_fast=False collects all errors."""
        items = [1, 2, 3]
        result = r.traverse(
            items,
            lambda x: r[int].fail(f"error_{x}") if x in {2, 3} else r[int].ok(x),
            fail_fast=False,
        )
        _ = assertion_helpers.assert_flext_result_failure(result)
        tm.that(result.error, none=False)
        tm.that(str(result.error), has="error_2")
        tm.that(str(result.error), has="error_3")

    def test_with_resource(self) -> None:
        """Test with_resource manages resource lifecycle."""
        resource_created: MutableSequence[str] = []
        resource_cleaned: MutableSequence[str] = []

        def factory() -> MutableSequence[str]:
            resource_created.append("created")
            return ["resource"]

        def op(resource: MutableSequence[str]) -> r[str]:
            resource.append("used")
            return r[str].ok("success")

        def cleanup(resource: MutableSequence[str]) -> None:
            resource_cleaned.append("cleaned")
            resource.clear()

        result: r[str] = r[str].with_resource(factory, op, cleanup)
        _ = assertion_helpers.assert_flext_result_success(result)
        tm.that(result.value, eq="success")
        tm.that(len(resource_created), eq=1)
        tm.that(len(resource_cleaned), eq=1)

    def test_context_manager(self) -> None:
        """Test context manager protocol."""
        result = r[str].ok("value")
        with result as ctx_result:
            tm.that(ctx_result is result, eq=True)
            tm.that(ctx_result.value, eq="value")

    def test_repr_success(self) -> None:
        """Test __repr__ for success result."""
        result = r[str].ok("test")
        repr_str = repr(result)
        tm.that(repr_str, has="r[T].ok")
        tm.that(repr_str, has="test")

    def test_repr_failure(self) -> None:
        """Test __repr__ for failure result."""
        result: r[str] = r[str].fail("error")
        repr_str = repr(result)
        tm.that(repr_str, has="r[T].fail")
        tm.that(repr_str, has="error")

    def test_value_property_failure(self) -> None:
        """Test value property raises RuntimeError on failure."""
        result: r[str] = r[str].fail("error")
        with pytest.raises(RuntimeError, match="Cannot access value of failed result"):
            _ = result.value

    def test_error_property_success(self) -> None:
        """Test error property returns None for success."""
        result = r[str].ok("test")
        tm.that(result.error, none=True)

    def test_error_code_property(self) -> None:
        """Test error_code property."""
        result: r[str] = r[str].fail("error", error_code="TEST_ERROR")
        tm.that(result.error_code, eq="TEST_ERROR")
        success = r[str].ok("test")
        tm.that(success.error_code, none=True)

    def test_error_data_property(self) -> None:
        """Test error_data property."""
        error_data = t.ConfigMap(root={"key": "value"})
        result: r[str] = r[str].fail("error", error_data=error_data)
        tm.that(result.error_data, eq=error_data)
        success = r[str].ok("test")
        tm.that(success.error_data, none=True)

    def test_unwrap_failure(self) -> None:
        """Test unwrap raises RuntimeError on failure."""
        result: r[str] = r[str].fail("error")
        with pytest.raises(RuntimeError, match="Cannot access value of failed result"):
            result.value

    def test_flat_map_inner_failure(self) -> None:
        """Test flat_map inner function returns Failure."""
        result = r[int].ok(5)

        def failing_func(_value: int) -> r[str]:
            return r[str].fail("flat_map failed")

        bound = result.flat_map(failing_func)
        tm.fail(bound)

    def test_flow_through_empty(self) -> None:
        """Test flow_through with no functions."""
        result = r[int].ok(5)
        tm.that(result.flow_through() is result, eq=True)
        tm.that(result.value, eq=5)

    def test_ok_with_none_succeeds(self) -> None:
        """Test ok(None) creates valid success result."""
        result = r[str | None].ok(None)
        tm.that(result.is_success, eq=True)
        tm.that(result.value, none=True)

    def test_flow_through_stops_on_failure(self) -> None:
        """Test flow_through stops when function returns failure."""

        def add_one(x: int) -> r[int]:
            return r[int].ok(x + 1)

        def fail_op(_x: int) -> r[int]:
            return r[int].fail("stopped")

        def never_called(_x: int) -> r[int]:
            return r[int].ok(999)

        result = r[int].ok(5)
        final = result.flow_through(add_one, fail_op, never_called)
        tm.fail(final)
        tm.that(final.error, eq="stopped")

    def test_create_from_callable_success(self) -> None:
        """Test create_from_callable with successful callable."""

        def func() -> str:
            return "success"

        result = r.create_from_callable(func)
        _ = assertion_helpers.assert_flext_result_success(result)
        tm.that(result.value, eq="success")

    def test_create_from_callable_none(self) -> None:
        """Test create_from_callable with callable returning None."""

        def func() -> str | None:
            return None

        result = r.create_from_callable(func)
        _ = assertion_helpers.assert_flext_result_failure(result)
        error_msg = result.error
        tm.that(error_msg, none=False)
        tm.that(error_msg, has="Callable returned None")

    def test_create_from_callable_exception(self) -> None:
        """Test create_from_callable handles exceptions."""

        def func() -> str:
            error_msg = "Callable failed"
            raise ValueError(error_msg)

        result = r.create_from_callable(func)
        _ = assertion_helpers.assert_flext_result_failure(result)
        error_msg = result.error
        tm.that(error_msg, none=False)
        tm.that(error_msg, has="Callable failed")

    def test_map_or_success_without_func(self) -> None:
        """Test map_or returns value on success when func is None."""
        result: r[str] = r[str].ok("hello")
        value = result.map_or(None)
        tm.that(value, eq="hello")

    def test_map_or_failure_without_func(self) -> None:
        """Test map_or returns default on failure when func is None."""
        result: r[str] = r[str].fail("error")
        value = result.map_or("default")
        tm.that(value, eq="default")

    def test_map_or_success_with_func(self) -> None:
        """Test map_or applies func on success."""
        result: r[str] = r[str].ok("hello")
        length = result.map_or(0, len)
        tm.that(length, eq=5)

    def test_map_or_failure_with_func(self) -> None:
        """Test map_or returns default on failure even with func."""
        result: r[str] = r[str].fail("error")
        length = result.map_or(0, len)
        tm.that(length, eq=0)

    def test_fold_success(self) -> None:
        """Test fold applies on_success function."""
        result: r[str] = r[str].ok("hello")
        message = result.fold(
            on_success=lambda v: f"Got: {v}",
            on_failure=lambda e: f"Error: {e}",
        )
        tm.that(message, eq="Got: hello")

    def test_fold_failure(self) -> None:
        """Test fold applies on_failure function."""
        result: r[str] = r[str].fail("something broke")
        message = result.fold(
            on_success=lambda v: f"Got: {v}",
            on_failure=lambda e: f"Error: {e}",
        )
        tm.that(message, eq="Error: something broke")

    def test_fold_different_return_types(self) -> None:
        """Test fold can return different types than input."""
        result: r[str] = r[str].ok("hello")
        response: t.ContainerMapping = result.fold(
            on_success=lambda v: {"status": 200, "data": v},
            on_failure=lambda e: {"status": 400, "error": e},
        )
        tm.that(
            response, eq=cast("t.Tests.Testobject", {"status": 200, "data": "hello"})
        )

    @given(x=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_identity_law(self, x: int) -> None:
        """Functor identity: map(id) == id."""
        left = r[int].ok(x).map(lambda v: v)
        right = r[int].ok(x)
        tm.ok(left, eq=right.value)
        tm.that(left.is_success, eq=right.is_success)

    @given(x=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_composition_law(self, x: int) -> None:
        """Functor composition: map(f).map(g) == map(g . f)."""

        def f(v: int) -> int:
            return v + 3

        def g(v: int) -> int:
            return v * 2

        left = r[int].ok(x).map(f).map(g)
        right = r[int].ok(x).map(lambda v: g(f(v)))
        tm.ok(left, eq=right.value)

    @given(x=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_left_unit_law(self, x: int) -> None:
        """Monad left unit: ok(x).flat_map(f) == f(x)."""

        def f(v: int) -> r[int]:
            return r[int].ok(v * 4)

        left = r[int].ok(x).flat_map(f)
        right = f(x)
        tm.ok(left, eq=right.value)

    @given(err=st.text(min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_error_propagation_property(self, err: str) -> None:
        """Errors propagate through map unchanged."""
        propagated = r[int].fail(err).map(lambda v: v + 1)
        tm.fail(propagated, has=err)
        tm.that(propagated.is_failure, eq=True)

    __all__ = ["Testr"]


__all__ = ["Testr"]
