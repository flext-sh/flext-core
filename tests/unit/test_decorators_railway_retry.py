"""Decorator railway, retry, and timeout tests."""

from __future__ import annotations

import time

import pytest

from tests import d, e, r, u
from tests.unit._decorators_support import (
    TestsFlextDecoratorsLegacy,
    capture_stdout,
)

RAILWAY_SCENARIOS = TestsFlextDecoratorsLegacy.RAILWAY_SCENARIOS
RETRY_SCENARIOS = TestsFlextDecoratorsLegacy.RETRY_SCENARIOS
TIMEOUT_SCENARIOS = TestsFlextDecoratorsLegacy.TIMEOUT_SCENARIOS


class TestsFlextDecoratorsRailwayRetry(TestsFlextDecoratorsLegacy):
    @pytest.mark.parametrize("test_case", RAILWAY_SCENARIOS, ids=lambda case: case.name)
    def test_railway_decorator(
        self,
        test_case: TestsFlextDecoratorsLegacy.DecoratorTestCase,
    ) -> None:
        if test_case.operation == self.DecoratorOperationType.RAILWAY_SUCCESS:

            @d.railway()
            def successful_operation() -> str:
                return "success"

            result = successful_operation()
            assert isinstance(result, r)
            u.Tests.assert_success(result, expected_value="success")
        elif test_case.operation == self.DecoratorOperationType.RAILWAY_EXCEPTION:

            @d.railway(error_code="CUSTOM_ERROR")
            def failing_operation() -> str:
                error_msg = "Operation failed"
                raise ValueError(error_msg)

            result = failing_operation()
            assert isinstance(result, r)
            _ = u.Tests.assert_failure(result)
            assert result.error is not None
            assert "Operation failed" in result.error

    @pytest.mark.parametrize("test_case", RETRY_SCENARIOS, ids=lambda case: case.name)
    def test_retry_decorator(
        self,
        test_case: TestsFlextDecoratorsLegacy.DecoratorTestCase,
    ) -> None:
        if test_case.operation == self.DecoratorOperationType.RETRY_SUCCESS_FIRST:

            @d.retry(max_attempts=3)
            def successful_operation() -> str:
                return "success"

            assert successful_operation() == "success"
        elif (
            test_case.operation
            == self.DecoratorOperationType.RETRY_SUCCESS_AFTER_FAILURES
        ):
            attempts = 0

            @d.retry(max_attempts=3, delay_seconds=0.001)
            def flaky_operation() -> str:
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    error_msg = f"Attempt {attempts} failed"
                    raise RuntimeError(error_msg)
                return "success"

            assert flaky_operation() == "success"
            assert attempts == 3
        elif test_case.operation == self.DecoratorOperationType.RETRY_EXHAUSTED:

            @d.retry(max_attempts=2, delay_seconds=0.001)
            def always_fails() -> str:
                error_msg = "Always fails"
                raise ValueError(error_msg)

            def emit() -> None:
                with pytest.raises(
                    e.TimeoutError,
                    match="failed after 2 attempts",
                ):
                    always_fails()

            _ = capture_stdout(
                emit,
                contains="operation_failed_all_retries_exhausted",
            )

    @pytest.mark.parametrize("test_case", TIMEOUT_SCENARIOS, ids=lambda case: case.name)
    def test_timeout_decorator(
        self,
        test_case: TestsFlextDecoratorsLegacy.DecoratorTestCase,
    ) -> None:
        if test_case.operation == self.DecoratorOperationType.TIMEOUT_SUCCESS:

            @d.timeout(timeout_seconds=1.0)
            def fast_operation() -> str:
                time.sleep(0.01)
                return "completed"

            assert fast_operation() == "completed"
        elif test_case.operation == self.DecoratorOperationType.TIMEOUT_EXCEEDED:

            @d.timeout(timeout_seconds=0.005)
            def slow_operation() -> str:
                time.sleep(0.01)
                return "should_not_reach"

            with pytest.raises(e.TimeoutError):
                slow_operation()
