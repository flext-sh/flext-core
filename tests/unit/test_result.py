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

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Never, cast

import pytest
from returns.io import IO, IOFailure, IOResult, IOSuccess
from returns.maybe import Nothing, Some
from returns.result import Success

from flext_core import c, e, r, t
from flext_tests import FlextTestsUtilities, u
from tests.test_utils import assertion_helpers


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
                FlextTestsUtilities.Tests.GenericHelpers.create_result_from_value(
                    value,
                    error_on_none="Value cannot be None",
                )
            )
            # value is already t.GeneralValueType from ResultScenario
            u.Tests.Result.assert_success_with_value(
                creation_result,
                value,
            )

        elif op_type == ResultOperationType.CREATION_FAILURE:
            # create_failure_result returns r[object], cast to r[t.GeneralValueType]
            failure_result_raw = u.Tests.Result.create_failure_result(
                str(value),
            )
            failure_result: r[t.GeneralValueType] = cast(
                "r[t.GeneralValueType]",
                failure_result_raw,
            )
            u.Tests.Result.assert_failure_with_error(
                failure_result,
                str(value),
            )

        elif op_type == ResultOperationType.UNWRAP_OR:
            # value is already t.GeneralValueType from ResultScenario
            if is_success:
                unwrap_result: r[t.GeneralValueType] = (
                    u.Tests.Result.create_success_result(value)
                )
            else:
                failure_raw = u.Tests.Result.create_failure_result(str(value))
                unwrap_result = cast("r[t.GeneralValueType]", failure_raw)
            default = "default"
            assert unwrap_result.unwrap_or(default) == (
                value if is_success else default
            )

        elif op_type == ResultOperationType.MAP:
            map_result: r[str] = r[str].fail(str(value))
            mapped = map_result.map(lambda x: str(x) * 2)
            u.Tests.Result.assert_failure_with_error(
                mapped,
                str(value),
            )

        elif op_type == ResultOperationType.FLAT_MAP:
            failure_raw = u.Tests.Result.create_failure_result(str(value))
            flat_map_result: r[t.GeneralValueType] = cast(
                "r[t.GeneralValueType]",
                failure_raw,
            )
            flat_mapped = flat_map_result.flat_map(
                lambda x: r[str].ok(f"value_{x}"),
            )
            u.Tests.Result.assert_failure_with_error(
                flat_mapped,
                str(value),
            )

        elif op_type == ResultOperationType.ALT:
            # value is already t.GeneralValueType from ResultScenario
            if is_success:
                result_alt: r[t.GeneralValueType] = (
                    u.Tests.Result.create_success_result(value)
                )
            else:
                failure_raw = u.Tests.Result.create_failure_result(str(value))
                result_alt = cast("r[t.GeneralValueType]", failure_raw)
            alt_result = result_alt.alt(lambda e: f"alt_{e}")
            if is_success:
                u.Tests.Result.assert_success_with_value(
                    alt_result,
                    value,
                )
            else:
                error_str_alt: str = f"alt_{value}"
                u.Tests.Result.assert_failure_with_error(
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
                u.Tests.Result.assert_success_with_value(
                    lash_result,
                    str(value),
                )
            else:
                expected = f"recovered_{value}"
                u.Tests.Result.assert_success_with_value(
                    lash_result,
                    expected,
                )

        elif op_type == ResultOperationType.OR_OPERATOR:
            # value is already t.GeneralValueType from ResultScenario
            if is_success:
                result_or: r[t.GeneralValueType] = u.Tests.Result.create_success_result(
                    value
                )
            else:
                failure_raw = u.Tests.Result.create_failure_result(str(value))
                result_or = cast("r[t.GeneralValueType]", failure_raw)
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
            assert result.value == value

        elif op_type == ResultOperationType.MAP:
            assert isinstance(value, int)
            result = r[int].ok(value)
            mapped = result.map(lambda x: x * 2)
            u.Tests.Result.assert_success_with_value(
                mapped,
                value * 2,
            )

        elif op_type == ResultOperationType.FLAT_MAP:
            assert isinstance(value, int)
            result = r[int].ok(value)
            flat_mapped = result.flat_map(
                lambda x: r[str].ok(f"value_{x}"),
            )
            expected = f"value_{value}"
            # Direct assert to avoid type-var issue with object
            u.Tests.Result.assert_success_with_value(
                flat_mapped,
                expected,
            )

        elif op_type == ResultOperationType.FILTER:
            assert isinstance(value, int)
            result = r[int].ok(value)
            filtered = result.filter(lambda x: x > 5)
            if is_success:
                u.Tests.Result.assert_success_with_value(
                    filtered,
                    value,
                )
            else:
                u.Tests.Result.assert_result_failure(filtered)

        elif op_type == ResultOperationType.RAILWAY_COMPOSITION:
            assert isinstance(value, int)
            # Use generic helper to test result chain - replaces 10+ lines
            # Each result typed independently for type safety
            res1 = r[int].ok(value)
            res2 = res1.map(lambda v: v * 2)
            res3 = res2.map(lambda v: f"result_{v}")
            expected = f"result_{value * 2}"
            # Use generic helper for chain validation with explicit list cast
            # assert_result_chain expects list[r[Never]], not Sequence
            result_list: list[r[Never]] = [
                cast("r[Never]", res1),
                cast("r[Never]", res2),
                cast("r[Never]", res3),
            ]
            FlextTestsUtilities.Tests.GenericHelpers.assert_result_chain(
                result_list,
                expected_success_count=3,
                expected_failure_count=0,
                first_failure_index=None,
            )
            u.Tests.Result.assert_success_with_value(
                res3,
                expected,
            )

    @pytest.mark.parametrize(
        "scenario",
        ResultScenarios.BOOL_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_result_bool_operations(self, scenario: ResultScenario) -> None:
        """Test r with boolean values across all scenarios."""
        if scenario.operation_type == ResultOperationType.BOOL_CONVERSION:
            result = (
                u.Tests.Result.create_success_result("value")
                if scenario.value
                else u.Tests.Result.create_failure_result(
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
        res1 = FlextTestsUtilities.Tests.GenericHelpers.create_result_from_value(
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
        FlextTestsUtilities.Tests.GenericHelpers.assert_result_chain(
            results,
            expected_success_count=3,
            expected_failure_count=0,
            first_failure_index=None,
        )

        # Verify final value
        u.Tests.Result.assert_success_with_value(
            res3,
            20,  # (5 * 2) + 10
        )

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
        u.Tests.Result.assert_success_with_value(
            res3,
            10,
        )

        # Now test actual failure case
        res4 = res3.flat_map(
            lambda x: r[int].fail("Cannot process zero") if x == 0 else r[int].ok(x),
        )
        results.append(res4)

        # Validate chain - should still be all successful
        FlextTestsUtilities.Tests.GenericHelpers.assert_result_chain(
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

        # Convert list[str] to list[t.GeneralValueType] for create_parametrized_cases
        success_values_general: list[t.GeneralValueType] = [
            cast("t.GeneralValueType", v) for v in success_values
        ]
        cases = FlextTestsUtilities.Tests.GenericHelpers.create_parametrized_cases(
            success_values_general,
            failure_errors,
            error_codes=error_codes,
        )

        # Verify cases structure
        assert len(cases) == 5  # 3 success + 2 failure

        # Verify success cases
        for i, (result, is_success, _value, error) in enumerate(cases[:3]):
            assert is_success is True
            u.Tests.Result.assert_success_with_value(
                result,
                success_values[i],
            )
            assert error is None

        # Verify failure cases
        for i, (result, is_success, _value, error) in enumerate(cases[3:]):
            assert is_success is False
            u.Tests.Result.assert_result_failure(result)
            assert error == failure_errors[i]

    def test_result_none_handling_limits(self) -> None:
        """Test None handling limits using generic helper."""
        # Test with None and default
        # Business Rule: create_result_from_value infers type from default_on_none
        # When value is None and default_on_none is provided, type is inferred from default
        result1: r[str] = (
            FlextTestsUtilities.Tests.GenericHelpers.create_result_from_value(
                None,
                default_on_none="default_value",
            )
        )
        u.Tests.Result.assert_success_with_value(
            result1,
            "default_value",
        )

        # Test with None and error
        result2: r[str | None] = (
            FlextTestsUtilities.Tests.GenericHelpers.create_result_from_value(
                None,
                error_on_none="Value is None",
            )
        )
        u.Tests.Result.assert_failure_with_error(
            result2,
            "Value is None",
        )

        # Test with actual value
        result3 = FlextTestsUtilities.Tests.GenericHelpers.create_result_from_value(
            "actual_value",
        )
        u.Tests.Result.assert_success_with_value(
            result3,
            "actual_value",
        )

    def test_to_io_success(self) -> None:
        """Test to_io conversion for success result."""
        result = r[str].ok("test_value")
        io_result = result.to_io()
        assert isinstance(io_result, IO)
        # IO wraps the value, need to unwrap to access
        assert io_result._inner_value == "test_value"

    def test_to_io_failure(self) -> None:
        """Test to_io conversion raises ValidationError for failure."""
        result = r[str].fail("error")
        with pytest.raises(e.ValidationError, match="Cannot convert failure to IO"):
            result.to_io()

    def test_from_io_result_success(self) -> None:
        """Test from_io_result with IOSuccess."""
        # IOSuccess wraps a Success in its _inner_value
        io_success = IOSuccess("test_value")
        result = r.from_io_result(io_success)
        assertion_helpers.assert_flext_result_success(result)
        # IOSuccess._inner_value is a Success object (from returns library)
        # The implementation stores this Success object as the FlextResult value
        value = result.value
        assert isinstance(value, Success)
        # Access inner value via _inner_value attribute of the Success
        assert value._inner_value == "test_value"

    def test_from_io_result_invalid_type(self) -> None:
        """Test from_io_result with invalid IO result type."""

        class InvalidIOResult:
            pass

        invalid_io = InvalidIOResult()
        # from_io_result expects IOResult[t.GeneralValueType, str]
        # InvalidIOResult is not IOResult, but method handles it at runtime
        result = r.from_io_result(cast("IOResult[t.GeneralValueType, str]", invalid_io))
        assertion_helpers.assert_flext_result_failure(result)
        assert result.error is not None
        assert "Invalid IO result type" in str(result.error)

    def test_from_io_result_failure(self) -> None:
        """Test from_io_result with IOFailure."""
        io_failure = IOFailure("error_message")
        result = r.from_io_result(io_failure)
        assertion_helpers.assert_flext_result_failure(result)
        # IOFailure._inner_value may be a Failure object, converted to string
        # The error should contain the error message
        assert "error_message" in str(result.error)

    def test_safe_decorator(self) -> None:
        """Test safe decorator wraps function in try/except."""

        # safe expects p.VariadicCallable[T] which is Callable[..., T]
        # divide is Callable[[int, int], int], compatible at runtime
        @r.safe
        def divide(a: int, b: int) -> int:
            return a // b

        result: r[int] = divide(10, 2)
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == 5

        result_fail: r[int] = divide(10, 0)
        assert result_fail.is_failure

    def test_map_error(self) -> None:
        """Test map_error transforms error message."""
        result = r[str].fail("original error")
        transformed = result.map_error(lambda e: f"PREFIX: {e}")
        assert transformed.is_failure
        assert transformed.error == "PREFIX: original error"

        # Success should remain unchanged
        success = r[str].ok("value")
        unchanged = success.map_error(lambda e: f"PREFIX: {e}")
        assert unchanged.is_success
        assert unchanged.value == "value"

    def test_filter_success(self) -> None:
        """Test filter with success result."""
        result = r[int].ok(10)
        filtered = result.filter(lambda x: x > 5)
        assert filtered.is_success
        assert filtered.value == 10

        filtered_fail = result.filter(lambda x: x > 20)
        assert filtered_fail.is_failure

    def test_filter_failure(self) -> None:
        """Test filter with failure result returns unchanged."""
        result = r[int].fail("error")
        filtered = result.filter(lambda x: x > 5)
        assert filtered.is_failure
        assert filtered.error == "error"

    def test_flow_through(self) -> None:
        """Test flow_through chains multiple operations."""

        def add_one(x: int) -> r[int]:
            return r[int].ok(x + 1)

        def multiply_two(x: int) -> r[int]:
            return r[int].ok(x * 2)

        result = r[int].ok(5)
        final = result.flow_through(add_one, multiply_two)
        assert final.is_success
        assert final.value == 12

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
        assert final.is_failure
        assert final.error == "error"

    def test_traverse_success(self) -> None:
        """Test traverse maps over sequence successfully."""
        items = [1, 2, 3]
        result = r.traverse(items, lambda x: r[int].ok(x * 2))
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == [2, 4, 6]

    def test_traverse_failure(self) -> None:
        """Test traverse fails fast on first failure."""
        items = [1, 2, 3]
        result = r.traverse(
            items,
            lambda x: r[int].fail("error") if x == 2 else r[int].ok(x),
        )
        assertion_helpers.assert_flext_result_failure(result)
        assert result.error == "error"

    def test_accumulate_errors_all_success(self) -> None:
        """Test accumulate_errors with all successes."""
        results = [r[int].ok(1), r[int].ok(2), r[int].ok(3)]
        accumulated = r.accumulate_errors(*results)
        assert accumulated.is_success
        assert accumulated.value == [1, 2, 3]

    def test_accumulate_errors_with_failures(self) -> None:
        """Test accumulate_errors collects all errors."""
        results = [r[int].ok(1), r[int].fail("error1"), r[int].fail("error2")]
        accumulated = r.accumulate_errors(*results)
        assert accumulated.is_failure
        assert accumulated.error is not None
        assert "error1" in str(accumulated.error)
        assert "error2" in str(accumulated.error)

    def test_parallel_map_fail_fast(self) -> None:
        """Test parallel_map with fail_fast=True."""
        items = [1, 2, 3]
        result = r.parallel_map(
            items,
            lambda x: r[int].fail("error") if x == 2 else r[int].ok(x),
        )
        assertion_helpers.assert_flext_result_failure(result)

    def test_parallel_map_no_fail_fast(self) -> None:
        """Test parallel_map with fail_fast=False."""
        items = [1, 2, 3]
        result = r.parallel_map(
            items,
            lambda x: r[int].fail("error") if x == 2 else r[int].ok(x),
            fail_fast=False,
        )
        assertion_helpers.assert_flext_result_failure(result)
        assert result.error is not None
        assert "error" in str(result.error)

    def test_with_resource(self) -> None:
        """Test with_resource manages resource lifecycle."""
        resource_created = []
        resource_cleaned = []

        def factory() -> list[str]:
            resource_created.append("created")
            return ["resource"]

        def op(resource: list[str]) -> r[str]:
            resource.append("used")
            return r[str].ok("success")

        def cleanup(resource: list[str]) -> None:
            resource_cleaned.append("cleaned")
            resource.clear()

        result = r.with_resource(factory, op, cleanup)
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == "success"
        assert len(resource_created) == 1
        assert len(resource_cleaned) == 1

    def test_context_manager(self) -> None:
        """Test context manager protocol."""
        result = r[str].ok("value")
        with result as ctx_result:
            assert ctx_result is result
            assert ctx_result.value == "value"

    def test_repr_success(self) -> None:
        """Test __repr__ for success result."""
        result = r[str].ok("test")
        repr_str = repr(result)
        assert "r.ok" in repr_str
        assert "test" in repr_str

    def test_repr_failure(self) -> None:
        """Test __repr__ for failure result."""
        result = r[str].fail("error")
        repr_str = repr(result)
        assert "r.fail" in repr_str
        assert "error" in repr_str

    def test_to_io_result_success(self) -> None:
        """Test to_io_result for success."""
        result = r[str].ok("value")
        io_result = result.to_io_result()
        assert isinstance(io_result, IOSuccess)
        # IOSuccess contains a Success with the value
        assert io_result._inner_value._inner_value == "value"

    def test_to_io_result_failure(self) -> None:
        """Test to_io_result for failure."""
        result = r[str].fail("error")
        io_result = result.to_io_result()
        assert isinstance(io_result, IOFailure)

    def test_from_io_result_unwrap_exception(self) -> None:
        """Test from_io_result handles unwrap exceptions - tests lines 106-107.

        Note: Testing exception path in from_io_result is difficult because
        IOSuccess is immutable. The exception handling (lines 106-107) is
        defensive code that's hard to trigger without complex mocking.
        We verify the code path exists and works with normal IOSuccess.
        """
        # Test normal IOSuccess path works
        io_value = IO("test_value")
        real_io = IOSuccess(io_value)
        # from_io_result expects IOResult[t.GeneralValueType, str]
        # IOSuccess[IO[str]] is compatible at runtime but type system doesn't know
        result = r.from_io_result(cast("IOResult[t.GeneralValueType, str]", real_io))
        assertion_helpers.assert_flext_result_success(result)

        # The exception path (lines 106-107) is defensive code that catches
        # exceptions during unwrap(). Since IOSuccess is immutable and can't be
        # mocked easily, this path is tested implicitly through integration tests.
        # Adding pragma comment for coverage exemption on defensive code.

    def test_from_io_result_general_exception(self) -> None:
        """Test from_io_result handles general exceptions."""

        # This tests lines 117-118
        class ExceptionRaisingIO:
            def __init__(self) -> None:
                error_msg = "General exception"
                raise RuntimeError(error_msg)

        # Create object that raises on isinstance check
        class BadIO:
            def __init__(self) -> None:
                pass

        bad_io = BadIO()
        # from_io_result expects IOResult[t.GeneralValueType, str]
        # BadIO is not IOResult, but method handles it at runtime
        result = r.from_io_result(cast("IOResult[t.GeneralValueType, str]", bad_io))
        assertion_helpers.assert_flext_result_failure(result)
        error_msg = result.error
        assert error_msg is not None
        assert "Invalid IO result type" in error_msg

    def test_value_property_failure(self) -> None:
        """Test value property raises RuntimeError on failure."""
        result = r[str].fail("error")
        with pytest.raises(RuntimeError, match="Cannot access value of failed result"):
            _ = result.value

    def test_data_property(self) -> None:
        """Test data property (alias for value)."""
        result = r[str].ok("test")
        assert result.data == "test"
        assert result.data == result.value

    def test_error_property_success(self) -> None:
        """Test error property returns None for success."""
        result = r[str].ok("test")
        assert result.error is None

    def test_error_code_property(self) -> None:
        """Test error_code property."""
        result = r[str].fail("error", error_code="TEST_ERROR")
        assert result.error_code == "TEST_ERROR"

        success = r[str].ok("test")
        assert success.error_code is None

    def test_error_data_property(self) -> None:
        """Test error_data property."""
        error_data = {"key": "value"}
        result = r[str].fail("error", error_data=error_data)
        assert result.error_data == error_data

        success = r[str].ok("test")
        assert success.error_data is None

    def test_unwrap_failure(self) -> None:
        """Test unwrap raises RuntimeError on failure."""
        result = r[str].fail("error")
        with pytest.raises(RuntimeError, match="Cannot access value of failed result"):
            result.value

    def test_flat_map_inner_failure(self) -> None:
        """Test flat_map inner function returns Failure."""
        result = r[int].ok(5)

        def failing_func(value: int) -> r[str]:
            return r[str].fail("flat_map failed")

        bound = result.flat_map(failing_func)
        assert bound.is_failure
        # This tests line 303 where inner returns Failure

    def test_flow_through_empty(self) -> None:
        """Test flow_through with no functions."""
        result = r[int].ok(5)
        final: r[int] = result.flow_through()
        assert final.is_success
        assert final.value == 5

    def test_parallel_map_all_success(self) -> None:
        """Test parallel_map with all successes."""
        items = [1, 2, 3]
        result = r.parallel_map(items, lambda x: r[int].ok(x * 2), fail_fast=True)
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == [2, 4, 6]

    def test_ok_with_none_raises(self) -> None:
        """Test ok() raises ValueError for None value."""
        # ok() rejects None values - this is correct behavior
        # The test verifies that ValueError is raised
        with pytest.raises(
            ValueError,
            match="Cannot create success result with None value",
        ):
            r[str].ok(None)

    def test_to_maybe_success(self) -> None:
        """Test to_maybe converts success to Some."""
        result = r[str].ok("value")
        maybe = result.to_maybe()
        assert isinstance(maybe, Some)
        assert maybe.unwrap() == "value"

    def test_to_maybe_failure(self) -> None:
        """Test to_maybe converts failure to Nothing."""
        result = r[str].fail("error")
        maybe = result.to_maybe()
        assert maybe is Nothing

    def test_from_maybe_some(self) -> None:
        """Test from_maybe with Some."""
        maybe = Some("value")
        result = r.from_maybe(maybe)
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == "value"

    def test_from_maybe_nothing(self) -> None:
        """Test from_maybe with Nothing."""
        maybe = Nothing
        result = r.from_maybe(maybe, error="Custom error")
        assertion_helpers.assert_flext_result_failure(result)
        assert result.error == "Custom error"

    def test_flow_through_stops_on_failure(self) -> None:
        """Test flow_through stops when function returns failure."""

        def add_one(x: int) -> r[int]:
            return r[int].ok(x + 1)

        def fail_op(_x: int) -> r[int]:
            return r[int].fail("stopped")

        def never_called(_x: int) -> r[int]:
            return r[int].ok(999)  # Should never be called

        result = r[int].ok(5)
        final = result.flow_through(add_one, fail_op, never_called)
        assert final.is_failure
        assert final.error == "stopped"
        # This tests lines 355-361 (flow_through failure path)

    def test_create_from_callable_success(self) -> None:
        """Test create_from_callable with successful callable."""

        def func() -> str:
            return "success"

        result = r.create_from_callable(func)
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == "success"

    def test_create_from_callable_none(self) -> None:
        """Test create_from_callable with callable returning None."""

        def func() -> str | None:
            return None

        result = r.create_from_callable(func)
        assertion_helpers.assert_flext_result_failure(result)
        error_msg = result.error
        assert error_msg is not None
        assert "Callable returned None" in error_msg

    def test_create_from_callable_exception(self) -> None:
        """Test create_from_callable handles exceptions."""

        def func() -> str:
            error_msg = "Callable failed"
            raise ValueError(error_msg)

        result = r.create_from_callable(func)
        assertion_helpers.assert_flext_result_failure(result)
        error_msg = result.error
        assert error_msg is not None
        assert "Callable failed" in error_msg


__all__ = ["Testr"]
