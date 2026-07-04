"""Decorator combined and integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from flext_tests import d, r
from hypothesis import given, settings, strategies as st

from tests.unit._decorators_support import (
    TestsFlextDecoratorsLegacy,
)
from tests.utilities import u

if TYPE_CHECKING:
    from tests.protocols import p

COMBINED_SCENARIOS = TestsFlextDecoratorsLegacy.COMBINED_SCENARIOS


class TestsFlextDecoratorsCombined(TestsFlextDecoratorsLegacy):
    @pytest.mark.parametrize(
        "test_case",
        COMBINED_SCENARIOS,
        ids=lambda case: case.name,
    )
    def test_combined_decorator(
        self,
        test_case: TestsFlextDecoratorsLegacy.DecoratorTestCase,
    ) -> None:
        if test_case.operation == self.DecoratorOperationType.COMBINED_BASIC:

            @d.combined(operation_name="test_op", track_perf=True)
            def simple_function() -> str:
                return "result"

            assert simple_function() == "result"
        elif test_case.operation == self.DecoratorOperationType.COMBINED_WITH_RAILWAY:

            @d.combined(
                operation_name="wrapped",
                railway_enabled=True,
            )
            def operation() -> str:
                return "success"

            result = operation()
            assert isinstance(result, r)
            _ = u.Tests.assert_success(result)

    def test_railway_with_existing_result(self) -> None:
        @d.railway()
        def returns_result() -> p.Result[str]:
            return r[str].ok("already_wrapped")

        result = returns_result()
        _ = u.Tests.assert_success(result)
        assert result.value.value == "already_wrapped"

    def test_retry_with_class_logger(self) -> None:
        service = self.ServiceWithLogger()
        assert service.logger is not None

        @d.retry(max_attempts=2, delay_seconds=0.001)
        def flaky_method() -> str:
            return service.flaky_method()

        assert flaky_method() == "success"
        assert service.attempts == 2

    def test_integration_manual_stacking(self) -> None:
        @d.log_operation("stacked")
        @d.log_operation("stacked")
        @d.railway()
        def stacked_operation() -> str:
            return "stacked_result"

        result = stacked_operation()
        assert isinstance(result, r)
        _ = u.Tests.assert_success(result)

    def test_integration_retry_with_railway(self) -> None:
        attempts = 0

        @d.railway()
        @d.retry(max_attempts=3, delay_seconds=0.001)
        def flaky_with_railway() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                error_msg = "Retry me"
                raise RuntimeError(error_msg)
            return "success"

        result = flaky_with_railway()
        assert isinstance(result, r)
        _ = u.Tests.assert_success(result)
        assert attempts == 2

    @given(a=st.integers(), b=st.integers(min_value=1, max_value=1000))
    @settings(max_examples=50)
    def test_hypothesis_railway_division_always_returns_result(
        self,
        a: int,
        b: int,
    ) -> None:
        """Property: railway-wrapped division always returns a result."""

        @d.railway(error_code="DIV")
        def divide(x: int, y: int) -> float:
            return x / y

        result = divide(a, b)
        assert result.success or result.failure
