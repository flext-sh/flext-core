"""Tests for r - Core railway pattern implementation.

Module: flext_core
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

from collections.abc import (
    MutableSequence,
    Sequence,
)
from enum import StrEnum, unique
from typing import Annotated, ClassVar

import pytest
from flext_tests import tm
from hypothesis import given, settings, strategies as st

from tests import m, p, r, t, u


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


class TestsFlextResult:
    ResultOperationType = ResultOperationType

    class ResultScenario(m.BaseModel):
        """Generic result scenario for r tests."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Result scenario name")]
        operation_type: Annotated[
            ResultOperationType,
            m.Field(description="Result operation type"),
        ]
        value: Annotated[
            t.JsonValue,
            m.Field(description="Input value for result operation"),
        ]
        is_success_expected: Annotated[
            bool, m.Field(description="Expected success state")
        ] = True
        expected_result: Annotated[
            t.JsonValue | None,
            m.Field(description="Optional expected result payload"),
        ] = None

    STRING_SCENARIOS: ClassVar[Sequence[TestsFlextResult.ResultScenario]] = [
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
    INT_SCENARIOS: ClassVar[Sequence[TestsFlextResult.ResultScenario]] = [
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
    BOOL_SCENARIOS: ClassVar[Sequence[TestsFlextResult.ResultScenario]] = [
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
        success = scenario.is_success_expected
        if not isinstance(value, str):
            pytest.fail("Expected string scenario value")
        if op_type == self.ResultOperationType.CREATION_SUCCESS:
            creation_result: p.Result[str] = u.Tests.create_result_from_value(
                value,
                error_on_none="Value cannot be None",
            )
            u.Tests.assert_success(creation_result, expected_value=value)
        elif op_type == self.ResultOperationType.CREATION_FAILURE:
            failure_result_raw = r[str].fail(value)
            failure_result: p.Result[str] = failure_result_raw
            u.Tests.assert_failure(failure_result, value)
        elif op_type == self.ResultOperationType.UNWRAP_OR:
            if success:
                unwrap_result: p.Result[str] = r[str].ok(value)
            else:
                failure_raw = r[str].fail(value)
                unwrap_result = failure_raw
            default = "default"
            tm.that(
                unwrap_result.unwrap_or(default),
                eq=value if success else default,
            )
        elif op_type == self.ResultOperationType.MAP:
            map_result: p.Result[str] = r[str].fail(value)
            mapped = map_result.map(lambda x: x * 2)
            u.Tests.assert_failure(mapped, value)
        elif op_type == self.ResultOperationType.FLAT_MAP:
            failure_raw = r[str].fail(value)
            flat_map_result: p.Result[str] = failure_raw
            flat_mapped = flat_map_result.flat_map(lambda x: r[str].ok(f"value_{x}"))
            u.Tests.assert_failure(flat_mapped, value)
        elif op_type == self.ResultOperationType.ALT:
            if success:
                result_alt: p.Result[str] = r[str].ok(value)
            else:
                failure_raw = r[str].fail(value)
                result_alt = failure_raw
            alt_result = result_alt.map_error(lambda e: f"alt_{e}")
            if success:
                u.Tests.assert_success(alt_result, expected_value=value)
            else:
                error_str_alt: str = f"alt_{value}"
                u.Tests.assert_failure(alt_result, error_str_alt)
        elif op_type == self.ResultOperationType.LASH:
            lash_result_base: p.Result[str] = (
                r[str].ok(value) if success else r[str].fail(value)
            )
            lash_result = lash_result_base.lash(lambda e: r[str].ok(f"recovered_{e}"))
            if success:
                u.Tests.assert_success(lash_result, expected_value=value)
            else:
                expected = f"recovered_{value}"
                u.Tests.assert_success(lash_result, expected_value=expected)
        elif op_type == self.ResultOperationType.OR_OPERATOR:
            if success:
                result_or: p.Result[str] = r[str].ok(value)
            else:
                failure_raw = r[str].fail(value)
                result_or = failure_raw
            default = "default"
            tm.that(result_or | default, eq=value if success else default)

    @pytest.mark.parametrize("scenario", INT_SCENARIOS, ids=lambda s: s.name)
    def test_result_int_operations(self, scenario: ResultScenario) -> None:
        """Test r with integer values across all scenarios."""
        op_type = scenario.operation_type
        value = scenario.value
        success = scenario.is_success_expected
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
            u.Tests.assert_success(mapped, expected_value=value * 2)
        elif op_type == self.ResultOperationType.FLAT_MAP:
            if not isinstance(value, int):
                pytest.fail("Expected integer scenario value")
            result = r[int].ok(value)
            flat_mapped = result.flat_map(lambda x: r[str].ok(f"value_{x}"))
            expected = f"value_{value}"
            u.Tests.assert_success(flat_mapped, expected_value=expected)
        elif op_type == self.ResultOperationType.FILTER:
            if not isinstance(value, int):
                pytest.fail("Expected integer scenario value")
            result = r[int].ok(value)
            filtered = result.filter(lambda x: x > 5)
            if success:
                u.Tests.assert_success(filtered, expected_value=value)
            else:
                _ = u.Tests.assert_failure(filtered)
        elif op_type == self.ResultOperationType.RAILWAY_COMPOSITION:
            if not isinstance(value, int):
                pytest.fail("Expected integer scenario value")
            res1 = r[int].ok(value)
            res2 = res1.map(lambda v: v * 2)
            res3 = res2.map(lambda v: f"result_{v}")
            expected = f"result_{value * 2}"
            result_list: Sequence[p.Result[str]] = [res1.map(str), res2.map(str), res3]
            u.Tests.assert_result_chain(
                result_list,
                expected_success_count=3,
                expected_failure_count=0,
                first_failure_index=None,
            )
            u.Tests.assert_success(res3, expected_value=expected)

    @pytest.mark.parametrize("scenario", BOOL_SCENARIOS, ids=lambda s: s.name)
    def test_result_bool_operations(self, scenario: ResultScenario) -> None:
        """Test r with boolean values across all scenarios."""
        if scenario.operation_type == self.ResultOperationType.BOOL_CONVERSION:
            result = (
                r[str].ok("value") if scenario.value else r[str].fail("generic_error")
            )
            tm.that(bool(result), eq=bool(scenario.value))

    def test_result_chain_validation_real_behavior(self) -> None:
        """Test result chain validation with real behavior patterns.

        Tests actual chain operations and validates using generic helpers.
        """
        results: MutableSequence[p.Result[int]] = []
        initial_value = 5
        res1 = u.Tests.create_result_from_value(
            initial_value,
            error_on_none="Initial value cannot be None",
        )
        results.append(res1)
        res2 = res1.map(lambda x: x * 2)
        results.append(res2)
        res3 = res2.map(lambda x: x + 10)
        results.append(res3)
        u.Tests.assert_result_chain(
            results,
            expected_success_count=3,
            expected_failure_count=0,
            first_failure_index=None,
        )
        u.Tests.assert_success(res3, expected_value=20)

    def test_result_chain_failure_behavior(self) -> None:
        """Test result chain with failure - real behavior and limits."""
        results: MutableSequence[p.Result[int]] = []
        res1 = r[int].ok(10)
        results.append(res1)
        res2 = res1.map(lambda x: x * 2)
        results.append(res2)
        res3 = res2.flat_map(
            lambda x: r[int].fail("Division by zero") if x == 0 else r[int].ok(x // 2),
        )
        results.append(res3)
        u.Tests.assert_success(res3, expected_value=10)
        res4 = res3.flat_map(
            lambda x: r[int].fail("Cannot process zero") if x == 0 else r[int].ok(x),
        )
        results.append(res4)
        u.Tests.assert_result_chain(
            results,
            expected_success_count=4,
            expected_failure_count=0,
        )

    def test_result_parametrized_cases_generic_helper(self) -> None:
        """Test using generic helper for parametrized test cases.

        Replaces 10+ lines of manual test case creation.
        """
        success_values: t.JsonList = ["value1", "value2", "value3"]
        failure_errors: t.StrSequence = ["error1", "error2"]
        error_codes: Sequence[str | None] = ["CODE1", None]
        cases = u.Tests.create_parametrized_cases(
            success_values,
            failure_errors,
            error_codes=error_codes,
        )
        tm.that(len(cases), eq=5)
        for i, (result, success, _value, error) in enumerate(cases[:3]):
            tm.that(success, eq=True)
            u.Tests.assert_success(result, expected_value=success_values[i])
            tm.that(error, none=True)
        for i, (result, success, _value, error) in enumerate(cases[3:]):
            tm.that(not success, eq=True)
            _ = u.Tests.assert_failure(result)
            tm.that(error, eq=failure_errors[i])

    def test_result_none_handling_limits(self) -> None:
        """Test None handling limits using generic helper."""
        result1: p.Result[str] = u.Tests.create_result_from_value(
            None,
            default_on_none="default_value",
        )
        u.Tests.assert_success(result1, expected_value="default_value")
        result2: p.Result[str | None] = u.Tests.create_result_from_value(
            None,
            error_on_none="Value is None",
        )
        u.Tests.assert_failure(result2, "Value is None")
        result3 = u.Tests.create_result_from_value("actual_value")
        u.Tests.assert_success(result3, expected_value="actual_value")

    def test_safe_decorator(self) -> None:
        """Test safe decorator wraps function in try/except."""

        def divide(a: int, b: int) -> int:
            return a // b

        divide_wrapped = r.safe(divide)
        result: p.Result[int] = divide_wrapped(10, 2)
        _ = u.Tests.assert_success(result)
        tm.that(result.value, eq=5)
        result_fail: p.Result[int] = divide_wrapped(10, 0)
        tm.fail(result_fail)

    def test_map_error(self) -> None:
        """Test map_error transforms error message."""
        result: p.Result[str] = r[str].fail("original error")
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
        result: p.Result[int] = r[int].fail("error")
        filtered = result.filter(lambda x: x > 5)
        tm.fail(filtered)
        tm.that(filtered.error, eq="error")

    def test_flow_through(self) -> None:
        """Test flow_through chains multiple operations."""

        def add_one(x: int) -> p.Result[int]:
            return r[int].ok(x + 1)

        def multiply_two(x: int) -> p.Result[int]:
            return r[int].ok(x * 2)

        result = r[int].ok(5)
        final = result.flow_through(add_one, multiply_two)
        tm.ok(final)
        tm.that(final.value, eq=12)

    def test_flow_through_failure(self) -> None:
        """Test flow_through stops on first failure."""

        def add_one(x: int) -> p.Result[int]:
            return r[int].ok(x + 1)

        def fail_op(_x: int) -> p.Result[int]:
            return r[int].fail("error")

        def multiply_two(x: int) -> p.Result[int]:
            return r[int].ok(x * 2)

        result = r[int].ok(5)
        final = result.flow_through(add_one, fail_op, multiply_two)
        tm.fail(final)
        tm.that(final.error, eq="error")

    def test_traverse_success(self) -> None:
        """Test traverse maps over sequence successfully."""
        items = [1, 2, 3]
        result = r.traverse(items, lambda x: r[int].ok(x * 2))
        _ = u.Tests.assert_success(result)
        tm.that(result.value, eq=[2, 4, 6])

    def test_traverse_failure(self) -> None:
        """Test traverse fails fast on first failure."""
        items = [1, 2, 3]
        result = r.traverse(
            items,
            lambda x: r[int].fail("error") if x == 2 else r[int].ok(x),
        )
        _ = u.Tests.assert_failure(result)
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
        _ = u.Tests.assert_failure(result)
        tm.that(result.error, eq="error")

    def test_traverse_fail_fast_false(self) -> None:
        """Test traverse with fail_fast=False collects all errors."""
        items = [1, 2, 3]
        result = r.traverse(
            items,
            lambda x: r[int].fail(f"error_{x}") if x in {2, 3} else r[int].ok(x),
            fail_fast=False,
        )
        _ = u.Tests.assert_failure(result)
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

        def op(resource: MutableSequence[str]) -> p.Result[str]:
            resource.append("used")
            return r[str].ok("success")

        def cleanup(resource: MutableSequence[str]) -> None:
            resource_cleaned.append("cleaned")
            resource.clear()

        result: p.Result[str] = r[str].with_resource(factory, op, cleanup)
        _ = u.Tests.assert_success(result)
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
        result: p.Result[str] = r[str].fail("error")
        repr_str = repr(result)
        tm.that(repr_str, has="r[T].fail")
        tm.that(repr_str, has="error")

    def test_value_property_failure(self) -> None:
        """Test value property raises RuntimeError on failure."""
        result: p.Result[str] = r[str].fail("error")
        with pytest.raises(RuntimeError, match="Cannot access value of failed result"):
            _ = result.value

    def test_error_property_success(self) -> None:
        """Test error property returns None for success."""
        result = r[str].ok("test")
        tm.that(result.error, none=True)

    def test_error_code_property(self) -> None:
        """Test error_code property."""
        result: p.Result[str] = r[str].fail("error", error_code="TEST_ERROR")
        tm.that(result.error_code, eq="TEST_ERROR")
        success = r[str].ok("test")
        tm.that(success.error_code, none=True)

    def test_error_data_property(self) -> None:
        """Test error_data property."""
        error_payload: dict[str, t.JsonPayload] = {"key": "value"}
        error_data = m.ConfigMap(root=error_payload)
        result: p.Result[str] = r[str].fail("error", error_data=error_data)
        tm.that(result.error_data, eq=error_payload)
        success = r[str].ok("test")
        tm.that(success.error_data, none=True)

    def test_unwrap_failure(self) -> None:
        """Test unwrap raises RuntimeError on failure."""
        result: p.Result[str] = r[str].fail("error")
        with pytest.raises(RuntimeError, match="Cannot access value of failed result"):
            result.value

    def test_flat_map_inner_failure(self) -> None:
        """Test flat_map inner function returns Failure."""
        result = r[int].ok(5)

        def failing_func(value: int) -> p.Result[str]:
            return r[str].fail("flat_map failed")

        bound = result.flat_map(failing_func)
        tm.fail(bound)

    def test_flow_through_empty(self) -> None:
        """Test flow_through with no functions."""
        result = r[int].ok(5)
        tm.that(result.flow_through() is result, eq=True)
        tm.that(result.value, eq=5)

    def test_ok_with_valid_value_succeeds(self) -> None:
        """Test ok(True) creates valid success result."""
        result = r[bool].ok(True)
        tm.that(result.success, eq=True)
        tm.that(result.value, eq=True)

    def test_flow_through_stops_on_failure(self) -> None:
        """Test flow_through stops when function returns failure."""

        def add_one(x: int) -> p.Result[int]:
            return r[int].ok(x + 1)

        def fail_op(_x: int) -> p.Result[int]:
            return r[int].fail("stopped")

        def never_called(_x: int) -> p.Result[int]:
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
        _ = u.Tests.assert_success(result)
        tm.that(result.value, eq="success")

    def test_create_from_callable_none(self) -> None:
        """Test create_from_callable with callable returning None."""

        def func() -> str | None:
            return None

        result = r.create_from_callable(func)
        _ = u.Tests.assert_failure(result)
        error_msg = result.error
        tm.that(error_msg, none=False)
        tm.that(error_msg, has="Callable returned None")

    def test_create_from_callable_exception(self) -> None:
        """Test create_from_callable handles exceptions."""

        def func() -> str:
            error_msg = "Callable failed"
            raise ValueError(error_msg)

        result = r.create_from_callable(func)
        _ = u.Tests.assert_failure(result)
        error_msg = result.error
        tm.that(error_msg, none=False)
        tm.that(error_msg, has="Callable failed")

    def test_map_or_success_without_func(self) -> None:
        """Test map_or returns value on success when func is None."""
        result: p.Result[str] = r[str].ok("hello")
        value = result.map_or(None)
        tm.that(value, eq="hello")

    def test_map_or_failure_without_func(self) -> None:
        """Test map_or returns default on failure when func is None."""
        result: p.Result[str] = r[str].fail("error")
        value = result.map_or("default")
        tm.that(value, eq="default")

    def test_map_or_success_with_func(self) -> None:
        """Test map_or applies func on success."""
        result: p.Result[str] = r[str].ok("hello")
        length = result.map_or(0, len)
        tm.that(length, eq=5)

    def test_map_or_failure_with_func(self) -> None:
        """Test map_or returns default on failure even with func."""
        result: p.Result[str] = r[str].fail("error")
        length = result.map_or(0, len)
        tm.that(length, eq=0)

    def test_fold_success(self) -> None:
        """Test fold applies on_success function."""
        result: p.Result[str] = r[str].ok("hello")
        message = result.fold(
            on_success=lambda v: f"Got: {v}",
            on_failure=lambda e: f"Error: {e}",
        )
        tm.that(message, eq="Got: hello")

    def test_fold_failure(self) -> None:
        """Test fold applies on_failure function."""
        result: p.Result[str] = r[str].fail("something broke")
        message = result.fold(
            on_success=lambda v: f"Got: {v}",
            on_failure=lambda e: f"Error: {e}",
        )
        tm.that(message, eq="Error: something broke")

    def test_fold_different_return_types(self) -> None:
        """Test fold can return different types than input."""
        result: p.Result[str] = r[str].ok("hello")
        response: t.JsonMapping = result.fold(
            on_success=lambda v: {"status": 200, "data": v},
            on_failure=lambda e: {"status": 400, "error": e},
        )
        tm.that(
            response,
            eq={"status": 200, "data": "hello"},
        )

    @given(x=st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_identity_law(self, x: int) -> None:
        """Functor identity: map(id) == id."""
        left = r[int].ok(x).map(lambda v: v)
        right = r[int].ok(x)
        tm.ok(left, eq=right.value)
        tm.that(left.success, eq=right.success)

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

        def f(v: int) -> p.Result[int]:
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
        tm.that(propagated.failure, eq=True)

    def test_instances_satisfy_success_checkable_runtime_protocol(self) -> None:
        """R instances conform to p.SuccessCheckable structural contract at runtime."""
        assert isinstance(r[str].ok("value"), p.SuccessCheckable)
        assert isinstance(r[str].fail("boom"), p.SuccessCheckable)

    def test_ok_accepts_valid_value(self) -> None:
        result = r[bool].ok(True)
        tm.that(result.success, eq=True)
        tm.that(result.value, eq=True)

    def test_map_error_transforms_failure_and_preserves_code(self) -> None:
        failure: p.Result[int] = r[int].fail(
            "bad",
            error_code="E1",
            error_data=m.ConfigMap(root={"k": "v"}),
        )
        transformed = failure.map_error(lambda msg: f"{msg}_mapped")
        tm.fail(transformed)
        tm.that(transformed.error, contains="bad_mapped")
        tm.that(transformed.error_code, eq="E1")

    def test_map_error_short_circuits_on_success(self) -> None:
        success = r[int].ok(1)
        tm.that(success.map_error(lambda msg: msg + "_x") is success, eq=True)

    def test_flow_through_stops_at_first_failure(self) -> None:
        visited: MutableSequence[int] = []

        def step1(v: int) -> p.Result[int]:
            visited.append(v)
            return r[int].ok(v + 1)

        def fail_step(_: int) -> p.Result[int]:
            return r[int].fail("stop")

        def unreachable(_: int) -> p.Result[int]:
            visited.append(999)
            return r[int].ok(0)

        result = r[int].ok(1).flow_through(step1, fail_step, unreachable)
        tm.fail(result)
        tm.that(result.error, contains="stop")
        tm.that(visited, eq=[1])

    def test_create_from_callable_handles_none_and_exception(self) -> None:
        def none_callable() -> int | None:
            return None

        none_result = r[int].create_from_callable(none_callable)
        tm.fail(none_result)
        tm.that(none_result.error, contains="Callable returned None")

        def error_callable() -> int | None:
            msg = "boom"
            raise ValueError(msg)

        error_result = r[int].create_from_callable(error_callable)
        tm.fail(error_result)
        tm.that(error_result.error, contains="boom")

    def test_with_resource_invokes_cleanup_after_success(self) -> None:
        cleanup_calls: MutableSequence[str] = []

        def factory() -> MutableSequence[int]:
            return []

        def op(resource: MutableSequence[int]) -> p.Result[str]:
            resource.append(1)
            return r[str].ok("done")

        def cleanup(resource: MutableSequence[int]) -> None:
            resource.clear()
            cleanup_calls.append("ran")

        result = r[str].with_resource(factory, op, cleanup)
        tm.ok(result)
        tm.that(result.value, eq="done")
        tm.that(cleanup_calls, eq=["ran"])


__all__: t.MutableSequenceOf[str] = ["TestsFlextResult"]
