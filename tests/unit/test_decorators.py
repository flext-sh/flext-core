"""Refactored comprehensive tests for d - Automation decorators."""

from __future__ import annotations

import time
from collections.abc import (
    Generator,
    Sequence,
)
from enum import StrEnum, unique
from typing import Annotated, ClassVar

import pytest
from hypothesis import given, settings, strategies as st

from flext_core import FlextContainer
from tests import d, e, m, p, r, t, u


class TestsFlextCoreDecoratorsLegacy:
    @unique
    class DecoratorOperationType(StrEnum):
        """Decorator operation types."""

        INJECT_BASIC = "inject_basic"
        INJECT_MISSING = "inject_missing"
        INJECT_PROVIDED = "inject_provided"
        LOG_OPERATION_BASIC = "log_operation_basic"
        LOG_OPERATION_EXCEPTION = "log_operation_exception"
        TRACK_PERFORMANCE_BASIC = "track_performance_basic"
        TRACK_PERFORMANCE_EXCEPTION = "track_performance_exception"
        RAILWAY_SUCCESS = "railway_success"
        RAILWAY_EXCEPTION = "railway_exception"
        RETRY_SUCCESS_FIRST = "retry_success_first"
        RETRY_SUCCESS_AFTER_FAILURES = "retry_success_after_failures"
        RETRY_EXHAUSTED = "retry_exhausted"
        TIMEOUT_SUCCESS = "timeout_success"
        TIMEOUT_EXCEEDED = "timeout_exceeded"
        COMBINED_BASIC = "combined_basic"
        COMBINED_WITH_RAILWAY = "combined_with_railway"

    class DecoratorTestCase(m.BaseModel):
        """Test case for decorator."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Decorator test case name")]
        operation: Annotated[str, m.Field(description="Decorator operation under test")]

    class TestService:
        """Service for testing."""

        def get_value(self) -> str:
            return "test_value"

    class ServiceWithLogger:
        """Service with logger for testing."""

        def __init__(self) -> None:
            self.logger = u.fetch_logger(__name__)
            self.attempts = 0

        def flaky_method(self) -> str:
            self.attempts += 1
            if self.attempts == 1:
                error_msg = "First attempt fails"
                raise RuntimeError(error_msg)
            return "success"

    INJECT_SCENARIOS: ClassVar[Sequence[DecoratorTestCase]] = [
        DecoratorTestCase(
            name="inject_basic_dependency",
            operation=DecoratorOperationType.INJECT_BASIC,
        ),
        DecoratorTestCase(
            name="inject_missing_dependency",
            operation=DecoratorOperationType.INJECT_MISSING,
        ),
        DecoratorTestCase(
            name="inject_with_provided_kwarg",
            operation=DecoratorOperationType.INJECT_PROVIDED,
        ),
    ]
    LOG_SCENARIOS: ClassVar[Sequence[DecoratorTestCase]] = [
        DecoratorTestCase(
            name="log_operation_basic",
            operation=DecoratorOperationType.LOG_OPERATION_BASIC,
        ),
        DecoratorTestCase(
            name="log_operation_exception",
            operation=DecoratorOperationType.LOG_OPERATION_EXCEPTION,
        ),
    ]
    TRACK_SCENARIOS: ClassVar[Sequence[DecoratorTestCase]] = [
        DecoratorTestCase(
            name="track_performance_basic",
            operation=DecoratorOperationType.TRACK_PERFORMANCE_BASIC,
        ),
        DecoratorTestCase(
            name="track_performance_exception",
            operation=DecoratorOperationType.TRACK_PERFORMANCE_EXCEPTION,
        ),
    ]
    RAILWAY_SCENARIOS: ClassVar[Sequence[DecoratorTestCase]] = [
        DecoratorTestCase(
            name="railway_success",
            operation=DecoratorOperationType.RAILWAY_SUCCESS,
        ),
        DecoratorTestCase(
            name="railway_exception",
            operation=DecoratorOperationType.RAILWAY_EXCEPTION,
        ),
    ]
    RETRY_SCENARIOS: ClassVar[Sequence[DecoratorTestCase]] = [
        DecoratorTestCase(
            name="retry_success_first_attempt",
            operation=DecoratorOperationType.RETRY_SUCCESS_FIRST,
        ),
        DecoratorTestCase(
            name="retry_success_after_failures",
            operation=DecoratorOperationType.RETRY_SUCCESS_AFTER_FAILURES,
        ),
        DecoratorTestCase(
            name="retry_exhausted",
            operation=DecoratorOperationType.RETRY_EXHAUSTED,
        ),
    ]
    TIMEOUT_SCENARIOS: ClassVar[Sequence[DecoratorTestCase]] = [
        DecoratorTestCase(
            name="timeout_success",
            operation=DecoratorOperationType.TIMEOUT_SUCCESS,
        ),
        DecoratorTestCase(
            name="timeout_exceeded",
            operation=DecoratorOperationType.TIMEOUT_EXCEEDED,
        ),
    ]
    COMBINED_SCENARIOS: ClassVar[Sequence[DecoratorTestCase]] = [
        DecoratorTestCase(
            name="combined_basic",
            operation=DecoratorOperationType.COMBINED_BASIC,
        ),
        DecoratorTestCase(
            name="combined_with_railway",
            operation=DecoratorOperationType.COMBINED_WITH_RAILWAY,
        ),
    ]

    @staticmethod
    @pytest.fixture(autouse=True)
    def reset_flext_container_singleton() -> Generator[None]:
        """Isolate FlextContainer singleton state across decorator tests."""
        FlextContainer.reset_for_testing()
        yield
        FlextContainer.reset_for_testing()

    @pytest.mark.parametrize("test_case", INJECT_SCENARIOS, ids=lambda case: case.name)
    def test_inject_decorator(
        self,
        test_case: TestsFlextCoreDecoratorsLegacy.DecoratorTestCase,
    ) -> None:
        if test_case.operation == self.DecoratorOperationType.INJECT_BASIC:

            @d.inject(test_service="test_service")
            def process_data_basic(
                data: str,
                *,
                test_service: TestsFlextCoreDecoratorsLegacy.TestService | None = None,
            ) -> str:
                if test_service is not None:
                    return f"{data}_{test_service.get_value()}"
                return f"{data}_default"

            assert process_data_basic("input") == "input_default"
        elif test_case.operation == self.DecoratorOperationType.INJECT_MISSING:

            @d.inject(missing_service="missing_service")
            def process_data_missing(*, missing_service: str = "default") -> str:
                return missing_service

            assert process_data_missing() == "default"
        elif test_case.operation == self.DecoratorOperationType.INJECT_PROVIDED:

            class TestServiceTyped(m.BaseModel):
                value: str

            @d.inject(service="service")
            def process(*, service: TestServiceTyped) -> str:
                return service.value

            explicit_service = TestServiceTyped.model_validate({"value": "explicit"})
            assert process(service=explicit_service) == "explicit"

    @pytest.mark.parametrize("test_case", LOG_SCENARIOS, ids=lambda case: case.name)
    def test_log_operation_decorator(
        self,
        test_case: TestsFlextCoreDecoratorsLegacy.DecoratorTestCase,
    ) -> None:
        if test_case.operation == self.DecoratorOperationType.LOG_OPERATION_BASIC:

            @d.log_operation("test_operation")
            def simple_function() -> str:
                return "success"

            assert simple_function() == "success"
        elif test_case.operation == self.DecoratorOperationType.LOG_OPERATION_EXCEPTION:

            @d.log_operation("failing_operation")
            def failing_function() -> None:
                error_msg = "Test error"
                raise ValueError(error_msg)

            with pytest.raises(ValueError, match="Test error"):
                failing_function()

    @pytest.mark.parametrize("test_case", TRACK_SCENARIOS, ids=lambda case: case.name)
    def test_track_performance_decorator(
        self,
        test_case: TestsFlextCoreDecoratorsLegacy.DecoratorTestCase,
    ) -> None:
        if test_case.operation == self.DecoratorOperationType.TRACK_PERFORMANCE_BASIC:

            @d.log_operation("timed_operation")
            def timed_function() -> str:
                time.sleep(0.01)
                return "completed"

            assert timed_function() == "completed"
        elif (
            test_case.operation
            == self.DecoratorOperationType.TRACK_PERFORMANCE_EXCEPTION
        ):

            @d.log_operation("failing_operation")
            def failing_function() -> None:
                error_msg = "Timed failure"
                raise RuntimeError(error_msg)

            with pytest.raises(RuntimeError, match="Timed failure"):
                failing_function()

    @pytest.mark.parametrize("test_case", RAILWAY_SCENARIOS, ids=lambda case: case.name)
    def test_railway_decorator(
        self,
        test_case: TestsFlextCoreDecoratorsLegacy.DecoratorTestCase,
    ) -> None:
        if test_case.operation == self.DecoratorOperationType.RAILWAY_SUCCESS:

            @d.railway()
            def successful_operation() -> str:
                return "success"

            result = successful_operation()
            assert isinstance(result, r)
            u.Core.Tests.assert_success_with_value(result, "success")
        elif test_case.operation == self.DecoratorOperationType.RAILWAY_EXCEPTION:

            @d.railway(error_code="CUSTOM_ERROR")
            def failing_operation() -> str:
                error_msg = "Operation failed"
                raise ValueError(error_msg)

            result = failing_operation()
            assert isinstance(result, r)
            _ = u.Core.Tests.assert_failure(result)
            assert result.error is not None
            assert "Operation failed" in result.error

    @pytest.mark.parametrize("test_case", RETRY_SCENARIOS, ids=lambda case: case.name)
    def test_retry_decorator(
        self,
        test_case: TestsFlextCoreDecoratorsLegacy.DecoratorTestCase,
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

            with pytest.raises(
                e.TimeoutError,
                match="failed after 2 attempts",
            ):
                always_fails()

    @pytest.mark.parametrize("test_case", TIMEOUT_SCENARIOS, ids=lambda case: case.name)
    def test_timeout_decorator(
        self,
        test_case: TestsFlextCoreDecoratorsLegacy.DecoratorTestCase,
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

    @pytest.mark.parametrize(
        "test_case",
        COMBINED_SCENARIOS,
        ids=lambda case: case.name,
    )
    def test_combined_decorator(
        self,
        test_case: TestsFlextCoreDecoratorsLegacy.DecoratorTestCase,
    ) -> None:
        if test_case.operation == self.DecoratorOperationType.COMBINED_BASIC:

            @d.combined(operation_name="test_op", track_perf=True)
            def simple_function() -> str:
                return "result"

            assert simple_function() == "result"
        elif test_case.operation == self.DecoratorOperationType.COMBINED_WITH_RAILWAY:

            @d.combined(use_railway=True, operation_name="wrapped")
            def operation() -> str:
                return "success"

            result = operation()
            assert isinstance(result, r)
            _ = u.Core.Tests.assert_success(result)

    def test_railway_with_existing_result(self) -> None:
        @d.railway()
        def returns_result() -> p.Result[str]:
            return r[str].ok("already_wrapped")

        result = returns_result()
        _ = u.Core.Tests.assert_success(result)
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
        _ = u.Core.Tests.assert_success(result)

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
        _ = u.Core.Tests.assert_success(result)
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


__all__: t.MutableSequenceOf[str] = ["TestsFlextCoreDecoratorsLegacy"]
