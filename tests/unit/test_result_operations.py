"""Result operation scenario tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from flext_tests import r, tm

from tests.unit._result_scenarios import (
    BOOL_SCENARIOS,
    INT_SCENARIOS,
    STRING_SCENARIOS,
    ResultOperationType,
    ResultScenario,
)
from tests.utilities import u

if TYPE_CHECKING:
    from tests.protocols import p
    from tests.typings import t


class TestsFlextResultOperations:
    ResultOperationType = ResultOperationType

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
            result_list: t.SequenceOf[p.Result[str]] = [
                res1.map(str),
                res2.map(str),
                res3,
            ]
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
