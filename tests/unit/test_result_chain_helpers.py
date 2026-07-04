"""Result chain helper tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_tests import r, tm

from tests.unit._result_scenarios import (
    ResultOperationType,
)
from tests.utilities import u

if TYPE_CHECKING:
    from collections.abc import MutableSequence

    from tests.protocols import p
    from tests.typings import t


class TestsFlextResultChainHelpers:
    ResultOperationType = ResultOperationType

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
        error_codes: t.SequenceOf[str | None] = ["CODE1", None]
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
