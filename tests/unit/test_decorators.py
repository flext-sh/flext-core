"""Refactored comprehensive tests for FlextDecorators - Automation decorators."""

from __future__ import annotations

import time
from enum import StrEnum, unique
from typing import Annotated, ClassVar

import pytest
from flext_tests import u
from pydantic import BaseModel, ConfigDict, Field

from flext_core import (
    FlextDecorators,
    FlextExceptions,
    FlextLogger,
    r,
)


class TestFlextDecorators:
    """Unified test suite for FlextDecorators."""

    @unique
    class DecoratorOperationType(StrEnum):
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

    class DecoratorTestCase(BaseModel):
        model_config = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Decorator test case name")]
        operation: Annotated[
            DecoratorOperationType, Field(description="Decorator operation under test")
        ]
        should_succeed: Annotated[
            bool, Field(default=True, description="Whether operation should succeed")
        ] = True
        error_type: Annotated[
            type[Exception] | None,
            Field(default=None, description="Expected exception type"),
        ] = None
        error_pattern: Annotated[
            str | None,
            Field(default=None, description="Expected error message pattern"),
        ] = None
        requires_container_setup: Annotated[
            bool,
            Field(default=False, description="Whether container setup is required"),
        ] = False
        with_exception_handling: Annotated[
            bool,
            Field(
                default=False, description="Whether exception handling path is expected"
            ),
        ] = False
        timeout_duration: Annotated[
            float, Field(default=0.1, description="Timeout duration in seconds")
        ] = 0.1
        retry_attempts: Annotated[
            int, Field(default=3, description="Retry attempts count")
        ] = 3
        retry_delay: Annotated[
            float, Field(default=0.001, description="Retry delay in seconds")
        ] = 0.001

    class TestService:
        def get_value(self) -> str:
            return "test_value"

    class ServiceWithLogger:
        def __init__(self) -> None:
            self.logger = FlextLogger(__name__)
            self.attempts = 0

        def flaky_method(self) -> str:
            self.attempts += 1
            if self.attempts == 1:
                error_msg = "First attempt fails"
                raise RuntimeError(error_msg)
            return "success"

    class DecoratorScenarios:
        INJECT_SCENARIOS: ClassVar[list[DecoratorTestCase]] = [
            DecoratorTestCase(
                name="inject_basic_dependency",
                operation=DecoratorOperationType.INJECT_BASIC,
                should_succeed=True,
                error_type=None,
                error_pattern=None,
                requires_container_setup=True,
            ),
            DecoratorTestCase(
                name="inject_missing_dependency",
                operation=DecoratorOperationType.INJECT_MISSING,
                should_succeed=True,
            ),
            DecoratorTestCase(
                name="inject_with_provided_kwarg",
                operation=DecoratorOperationType.INJECT_PROVIDED,
                should_succeed=True,
                error_type=None,
                error_pattern=None,
                requires_container_setup=True,
            ),
        ]
        LOG_SCENARIOS: ClassVar[list[DecoratorTestCase]] = [
            DecoratorTestCase(
                name="log_operation_basic",
                operation=DecoratorOperationType.LOG_OPERATION_BASIC,
                should_succeed=True,
            ),
            DecoratorTestCase(
                name="log_operation_exception",
                operation=DecoratorOperationType.LOG_OPERATION_EXCEPTION,
                should_succeed=False,
                error_type=ValueError,
                error_pattern="Test error",
                requires_container_setup=False,
                with_exception_handling=True,
            ),
        ]
        TRACK_SCENARIOS: ClassVar[list[DecoratorTestCase]] = [
            DecoratorTestCase(
                name="track_performance_basic",
                operation=DecoratorOperationType.TRACK_PERFORMANCE_BASIC,
                should_succeed=True,
            ),
            DecoratorTestCase(
                name="track_performance_exception",
                operation=DecoratorOperationType.TRACK_PERFORMANCE_EXCEPTION,
                should_succeed=False,
                error_type=RuntimeError,
                error_pattern="Timed failure",
                requires_container_setup=False,
                with_exception_handling=True,
            ),
        ]
        RAILWAY_SCENARIOS: ClassVar[list[DecoratorTestCase]] = [
            DecoratorTestCase(
                name="railway_success",
                operation=DecoratorOperationType.RAILWAY_SUCCESS,
                should_succeed=True,
            ),
            DecoratorTestCase(
                name="railway_exception",
                operation=DecoratorOperationType.RAILWAY_EXCEPTION,
                should_succeed=True,
            ),
        ]
        RETRY_SCENARIOS: ClassVar[list[DecoratorTestCase]] = [
            DecoratorTestCase(
                name="retry_success_first_attempt",
                operation=DecoratorOperationType.RETRY_SUCCESS_FIRST,
                should_succeed=True,
                error_type=None,
                error_pattern=None,
                requires_container_setup=False,
                with_exception_handling=False,
                timeout_duration=0.1,
                retry_attempts=3,
                retry_delay=0.001,
            ),
            DecoratorTestCase(
                name="retry_success_after_failures",
                operation=DecoratorOperationType.RETRY_SUCCESS_AFTER_FAILURES,
                should_succeed=True,
                error_type=None,
                error_pattern=None,
                requires_container_setup=False,
                with_exception_handling=False,
                timeout_duration=0.1,
                retry_attempts=3,
                retry_delay=0.001,
            ),
            DecoratorTestCase(
                name="retry_exhausted",
                operation=DecoratorOperationType.RETRY_EXHAUSTED,
                should_succeed=False,
                error_type=FlextExceptions.TimeoutError,
                error_pattern="failed after 2 attempts",
                requires_container_setup=False,
                with_exception_handling=True,
                timeout_duration=0.1,
                retry_attempts=2,
                retry_delay=0.001,
            ),
        ]
        TIMEOUT_SCENARIOS: ClassVar[list[DecoratorTestCase]] = [
            DecoratorTestCase(
                name="timeout_success",
                operation=DecoratorOperationType.TIMEOUT_SUCCESS,
                should_succeed=True,
                error_type=None,
                error_pattern=None,
                requires_container_setup=False,
                with_exception_handling=False,
                timeout_duration=1.0,
            ),
            DecoratorTestCase(
                name="timeout_exceeded",
                operation=DecoratorOperationType.TIMEOUT_EXCEEDED,
                should_succeed=False,
                error_type=FlextExceptions.TimeoutError,
                error_pattern=None,
                requires_container_setup=False,
                with_exception_handling=True,
                timeout_duration=0.005,
            ),
        ]
        COMBINED_SCENARIOS: ClassVar[list[DecoratorTestCase]] = [
            DecoratorTestCase(
                name="combined_basic",
                operation=DecoratorOperationType.COMBINED_BASIC,
                should_succeed=True,
            ),
            DecoratorTestCase(
                name="combined_with_railway",
                operation=DecoratorOperationType.COMBINED_WITH_RAILWAY,
                should_succeed=True,
            ),
        ]

        @staticmethod
        def create_service_with_logger() -> ServiceWithLogger:
            return TestFlextDecorators.ServiceWithLogger()

    @pytest.mark.parametrize(
        "test_case",
        DecoratorScenarios.INJECT_SCENARIOS,
        ids=lambda case: case.name,
    )
    def test_inject_decorator(self, test_case: DecoratorTestCase) -> None:
        if test_case.operation == self.DecoratorOperationType.INJECT_BASIC:

            @FlextDecorators.inject(test_service="test_service")
            def process_data_basic(
                data: str,
                *,
                test_service: TestFlextDecorators.TestService | None = None,
            ) -> str:
                if test_service is None:
                    return f"{data}_default"
                return f"{data}_{test_service.get_value()}"

            assert process_data_basic("input") == "input_default"
        elif test_case.operation == self.DecoratorOperationType.INJECT_MISSING:

            @FlextDecorators.inject(missing_service="missing_service")
            def process_data_missing(*, missing_service: str = "default") -> str:
                return missing_service

            assert process_data_missing() == "default"
        elif test_case.operation == self.DecoratorOperationType.INJECT_PROVIDED:

            class TestServiceTyped(BaseModel):
                value: str

            @FlextDecorators.inject(service="service")
            def process(*, service: TestServiceTyped) -> str:
                return service.value

            explicit_service = TestServiceTyped.model_validate({"value": "explicit"})
            assert process(service=explicit_service) == "explicit"

    @pytest.mark.parametrize(
        "test_case",
        DecoratorScenarios.LOG_SCENARIOS,
        ids=lambda case: case.name,
    )
    def test_log_operation_decorator(self, test_case: DecoratorTestCase) -> None:
        if test_case.operation == self.DecoratorOperationType.LOG_OPERATION_BASIC:

            @FlextDecorators.log_operation("test_operation")
            def simple_function() -> str:
                return "success"

            assert simple_function() == "success"
        elif test_case.operation == self.DecoratorOperationType.LOG_OPERATION_EXCEPTION:

            @FlextDecorators.log_operation("failing_operation")
            def failing_function() -> None:
                error_msg = "Test error"
                raise ValueError(error_msg)

            with pytest.raises(ValueError, match="Test error"):
                failing_function()

    @pytest.mark.parametrize(
        "test_case",
        DecoratorScenarios.TRACK_SCENARIOS,
        ids=lambda case: case.name,
    )
    def test_track_performance_decorator(self, test_case: DecoratorTestCase) -> None:
        if test_case.operation == self.DecoratorOperationType.TRACK_PERFORMANCE_BASIC:

            @FlextDecorators.log_operation("timed_operation")
            def timed_function() -> str:
                time.sleep(0.01)
                return "completed"

            result = timed_function()
            assert result == "completed"
        elif (
            test_case.operation
            == self.DecoratorOperationType.TRACK_PERFORMANCE_EXCEPTION
        ):

            @FlextDecorators.log_operation("failing_operation")
            def failing_function() -> None:
                error_msg = "Timed failure"
                raise RuntimeError(error_msg)

            with pytest.raises(RuntimeError, match="Timed failure"):
                failing_function()

    @pytest.mark.parametrize(
        "test_case",
        DecoratorScenarios.RAILWAY_SCENARIOS,
        ids=lambda case: case.name,
    )
    def test_railway_decorator(self, test_case: DecoratorTestCase) -> None:
        if test_case.operation == self.DecoratorOperationType.RAILWAY_SUCCESS:

            @FlextDecorators.railway()
            def successful_operation() -> str:
                return "success"

            result = successful_operation()
            assert isinstance(result, r)
            u.Tests.Result.assert_success_with_value(result, "success")
        elif test_case.operation == self.DecoratorOperationType.RAILWAY_EXCEPTION:

            @FlextDecorators.railway(error_code="CUSTOM_ERROR")
            def failing_operation() -> str:
                error_msg = "Operation failed"
                raise ValueError(error_msg)

            result = failing_operation()
            assert isinstance(result, r)
            _ = u.Tests.Result.assert_failure(result)
            assert result.error is not None
            assert "Operation failed" in result.error

    @pytest.mark.parametrize(
        "test_case",
        DecoratorScenarios.RETRY_SCENARIOS,
        ids=lambda case: case.name,
    )
    def test_retry_decorator(self, test_case: DecoratorTestCase) -> None:
        if test_case.operation == self.DecoratorOperationType.RETRY_SUCCESS_FIRST:

            @FlextDecorators.retry(max_attempts=3)
            def successful_operation() -> str:
                return "success"

            assert successful_operation() == "success"
        elif (
            test_case.operation
            == self.DecoratorOperationType.RETRY_SUCCESS_AFTER_FAILURES
        ):
            attempts = 0

            @FlextDecorators.retry(max_attempts=3, delay_seconds=0.001)
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

            @FlextDecorators.retry(max_attempts=2, delay_seconds=0.001)
            def always_fails() -> str:
                error_msg = "Always fails"
                raise ValueError(error_msg)

            with pytest.raises(
                FlextExceptions.TimeoutError, match="failed after 2 attempts"
            ):
                always_fails()

    @pytest.mark.parametrize(
        "test_case",
        DecoratorScenarios.TIMEOUT_SCENARIOS,
        ids=lambda case: case.name,
    )
    def test_timeout_decorator(self, test_case: DecoratorTestCase) -> None:
        if test_case.operation == self.DecoratorOperationType.TIMEOUT_SUCCESS:

            @FlextDecorators.timeout(timeout_seconds=1.0)
            def fast_operation() -> str:
                time.sleep(0.01)
                return "completed"

            result = fast_operation()
            assert result == "completed"
        elif test_case.operation == self.DecoratorOperationType.TIMEOUT_EXCEEDED:

            @FlextDecorators.timeout(timeout_seconds=0.005)
            def slow_operation() -> str:
                time.sleep(0.01)
                return "should_not_reach"

            with pytest.raises(FlextExceptions.TimeoutError) as exc_info:
                slow_operation()
            assert exc_info.value is not None
            assert "timeout" in str(exc_info.value).lower() or "0.005" in str(
                exc_info.value,
            )

    @pytest.mark.parametrize(
        "test_case",
        DecoratorScenarios.COMBINED_SCENARIOS,
        ids=lambda case: case.name,
    )
    def test_combined_decorator(self, test_case: DecoratorTestCase) -> None:
        if test_case.operation == self.DecoratorOperationType.COMBINED_BASIC:

            @FlextDecorators.combined(operation_name="test_op", track_perf=True)
            def simple_function() -> str:
                return "result"

            assert simple_function() == "result"
        elif test_case.operation == self.DecoratorOperationType.COMBINED_WITH_RAILWAY:

            @FlextDecorators.combined(use_railway=True, operation_name="wrapped")
            def operation() -> str:
                return "success"

            result = operation()
            assert isinstance(result, r)
            _ = u.Tests.Result.assert_success(result)

    def test_railway_with_existing_result(self) -> None:
        @FlextDecorators.railway()
        def returns_result() -> r[str]:
            return r[str].ok("already_wrapped")

        result = returns_result()
        _ = u.Tests.Result.assert_success(result)
        unwrapped = result.value
        assert unwrapped.value == "already_wrapped"

    def test_retry_with_class_logger(self) -> None:
        service = type(self).DecoratorScenarios.create_service_with_logger()
        assert hasattr(service, "logger")
        assert service.logger is not None

        @FlextDecorators.retry(max_attempts=2, delay_seconds=0.001)
        def flaky_method() -> str:
            return service.flaky_method()

        result = flaky_method()
        assert result == "success"
        assert service.attempts == 2
        assert hasattr(service, "logger")

    def test_integration_manual_stacking(self) -> None:
        @FlextDecorators.log_operation("stacked")
        @FlextDecorators.log_operation("stacked")
        @FlextDecorators.railway()
        def stacked_operation() -> str:
            return "stacked_result"

        result = stacked_operation()
        assert isinstance(result, r)
        _ = u.Tests.Result.assert_success(result)

    def test_integration_retry_with_railway(self) -> None:
        attempts = 0

        @FlextDecorators.railway()
        @FlextDecorators.retry(max_attempts=3, delay_seconds=0.001)
        def flaky_with_railway() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                error_msg = "Retry me"
                raise RuntimeError(error_msg)
            return "success"

        result = flaky_with_railway()
        assert isinstance(result, r)
        _ = u.Tests.Result.assert_success(result)
        assert attempts == 2


__all__ = ["TestFlextDecorators"]
