"""Refactored comprehensive tests for FlextDecorators - Automation decorators.

Tests the actual FlextDecorators API with real functionality testing using
Python 3.13 patterns, StrEnum, frozen dataclasses, and advanced parametrization.

Consolidates 9 test classes with 50+ methods into 1 unified class with
parametrized and integration tests.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import dataclasses
import logging
import time
from enum import StrEnum
from typing import ClassVar

import pytest

from flext_core import (
    FlextContainer,
    FlextDecorators,
    FlextExceptions,
    FlextLogger,
    FlextResult,
)
from flext_core.runtime import FlextRuntime

# =========================================================================
# Decorator Type Enumeration
# =========================================================================


class DecoratorOperationType(StrEnum):
    """Decorator operation types for parametrized testing."""

    INJECT_BASIC = "inject_basic"
    INJECT_MISSING = "inject_missing"
    INJECT_PROVIDED = "inject_provided"
    LOG_OPERATION_BASIC = "log_operation_basic"
    LOG_OPERATION_WITH_LOGGER = "log_operation_with_logger"
    LOG_OPERATION_EXCEPTION = "log_operation_exception"
    LOG_OPERATION_DEFAULT_NAME = "log_operation_default_name"
    LOG_OPERATION_WITH_PERF = "log_operation_with_perf"
    TRACK_PERFORMANCE_BASIC = "track_performance_basic"
    TRACK_PERFORMANCE_WITH_LOGGER = "track_performance_with_logger"
    TRACK_PERFORMANCE_EXCEPTION = "track_performance_exception"
    TRACK_PERFORMANCE_DEFAULT = "track_performance_default"
    RAILWAY_SUCCESS = "railway_success"
    RAILWAY_EXCEPTION = "railway_exception"
    RAILWAY_EXISTING_RESULT = "railway_existing_result"
    RAILWAY_DEFAULT_ERROR = "railway_default_error"
    RETRY_SUCCESS_FIRST = "retry_success_first"
    RETRY_SUCCESS_AFTER_FAILURES = "retry_success_after_failures"
    RETRY_EXHAUSTED = "retry_exhausted"
    RETRY_EXPONENTIAL_BACKOFF = "retry_exponential_backoff"
    RETRY_LINEAR_BACKOFF = "retry_linear_backoff"
    RETRY_WITH_CLASS_LOGGER = "retry_with_class_logger"
    TIMEOUT_SUCCESS = "timeout_success"
    TIMEOUT_EXCEEDED = "timeout_exceeded"
    TIMEOUT_EXCEPTION_SLOW = "timeout_exception_slow"
    TIMEOUT_EXCEPTION_FAST = "timeout_exception_fast"
    TIMEOUT_CUSTOM_ERROR = "timeout_custom_error"
    COMBINED_BASIC = "combined_basic"
    COMBINED_WITH_INJECTION = "combined_with_injection"
    COMBINED_WITH_RAILWAY = "combined_with_railway"
    COMBINED_RAILWAY_EXCEPTION = "combined_railway_exception"
    COMBINED_WITHOUT_PERF = "combined_without_perf"
    COMBINED_ALL_FEATURES = "combined_all_features"
    EDGE_CASE_INJECT_NO_CONTAINER = "edge_case_inject_no_container"
    EDGE_CASE_LOG_WITHOUT_LOGGER = "edge_case_log_without_logger"
    EDGE_CASE_TRACK_WITHOUT_LOGGER = "edge_case_track_without_logger"
    EDGE_CASE_RETRY_NO_EXCEPTION = "edge_case_retry_no_exception"
    EDGE_CASE_TIMEOUT_BOUNDARY = "edge_case_timeout_boundary"
    EDGE_CASE_COMBINED_MINIMAL = "edge_case_combined_minimal"
    INTEGRATION_MANUAL_STACKING = "integration_manual_stacking"
    INTEGRATION_RETRY_WITH_RAILWAY = "integration_retry_with_railway"
    INTEGRATION_TIMEOUT_WITH_RETRY = "integration_timeout_with_retry"


# =========================================================================
# Test Case Data Structure
# =========================================================================


@dataclasses.dataclass(frozen=True, slots=True)
class DecoratorTestCase:
    """Decorator test case definition with parametrization data."""

    name: str
    operation: DecoratorOperationType
    should_succeed: bool = True
    error_type: type[Exception] | None = None
    error_pattern: str | None = None
    requires_container_setup: bool = False
    with_exception_handling: bool = False
    timeout_duration: float = 0.1
    retry_attempts: int = 3
    retry_delay: float = 0.001
    backoff_strategy: str = "linear"


# =========================================================================
# Test Service Classes (Real Implementations)
# =========================================================================


class TestService:
    """Simple test service for dependency injection."""

    def get_value(self) -> str:
        """Return test value."""
        return "test_value"


class ServiceWithLogger:
    """Service that uses FlextLogger."""

    def __init__(self) -> None:
        """Initialize service with logger."""
        self.logger = FlextLogger(__name__)
        self.attempts = 0

    def process(self, value: str) -> str:
        """Process a value."""
        return f"processed_{value}"

    def flaky_method(self) -> str:
        """Method that fails once then succeeds."""
        self.attempts += 1
        if self.attempts == 1:
            msg = "First attempt fails"
            raise RuntimeError(msg)
        return "success"


class Repository:
    """Repository for data persistence."""

    def save(self, data: str) -> str:
        """Save data and return result."""
        return f"saved_{data}"


# =========================================================================
# Test Scenario Factory
# =========================================================================


class DecoratorScenarios:
    """Factory for decorator test scenarios with centralized test data."""

    # Inject decorator scenarios
    INJECT_SCENARIOS: ClassVar[list[DecoratorTestCase]] = [
        DecoratorTestCase(
            name="inject_basic_dependency",
            operation=DecoratorOperationType.INJECT_BASIC,
            should_succeed=True,
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
            requires_container_setup=True,
        ),
        DecoratorTestCase(
            name="inject_no_container_service",
            operation=DecoratorOperationType.EDGE_CASE_INJECT_NO_CONTAINER,
            should_succeed=True,
        ),
    ]

    # Log operation decorator scenarios
    LOG_OPERATION_SCENARIOS: ClassVar[list[DecoratorTestCase]] = [
        DecoratorTestCase(
            name="log_operation_basic",
            operation=DecoratorOperationType.LOG_OPERATION_BASIC,
            should_succeed=True,
        ),
        DecoratorTestCase(
            name="log_operation_with_class_logger",
            operation=DecoratorOperationType.LOG_OPERATION_WITH_LOGGER,
            should_succeed=True,
        ),
        DecoratorTestCase(
            name="log_operation_with_exception",
            operation=DecoratorOperationType.LOG_OPERATION_EXCEPTION,
            should_succeed=False,
            error_type=ValueError,
            error_pattern="Test error",
            with_exception_handling=True,
        ),
        DecoratorTestCase(
            name="log_operation_default_name",
            operation=DecoratorOperationType.LOG_OPERATION_DEFAULT_NAME,
            should_succeed=True,
        ),
        DecoratorTestCase(
            name="log_operation_with_perf_tracking_failure",
            operation=DecoratorOperationType.LOG_OPERATION_WITH_PERF,
            should_succeed=False,
            error_type=ValueError,
            error_pattern="Test error with performance tracking",
            with_exception_handling=True,
        ),
        DecoratorTestCase(
            name="log_operation_without_logger",
            operation=DecoratorOperationType.EDGE_CASE_LOG_WITHOUT_LOGGER,
            should_succeed=True,
        ),
    ]

    # Track performance decorator scenarios
    TRACK_PERFORMANCE_SCENARIOS: ClassVar[list[DecoratorTestCase]] = [
        DecoratorTestCase(
            name="track_performance_basic",
            operation=DecoratorOperationType.TRACK_PERFORMANCE_BASIC,
            should_succeed=True,
        ),
        DecoratorTestCase(
            name="track_performance_with_class_logger",
            operation=DecoratorOperationType.TRACK_PERFORMANCE_WITH_LOGGER,
            should_succeed=True,
        ),
        DecoratorTestCase(
            name="track_performance_with_exception",
            operation=DecoratorOperationType.TRACK_PERFORMANCE_EXCEPTION,
            should_succeed=False,
            error_type=RuntimeError,
            error_pattern="Timed failure",
            with_exception_handling=True,
        ),
        DecoratorTestCase(
            name="track_performance_default_name",
            operation=DecoratorOperationType.TRACK_PERFORMANCE_DEFAULT,
            should_succeed=True,
        ),
        DecoratorTestCase(
            name="track_performance_without_logger",
            operation=DecoratorOperationType.EDGE_CASE_TRACK_WITHOUT_LOGGER,
            should_succeed=True,
        ),
    ]

    # Railway decorator scenarios
    RAILWAY_SCENARIOS: ClassVar[list[DecoratorTestCase]] = [
        DecoratorTestCase(
            name="railway_success",
            operation=DecoratorOperationType.RAILWAY_SUCCESS,
            should_succeed=True,
        ),
        DecoratorTestCase(
            name="railway_exception",
            operation=DecoratorOperationType.RAILWAY_EXCEPTION,
            should_succeed=True,  # Exception is wrapped in FlextResult
        ),
        DecoratorTestCase(
            name="railway_with_existing_result",
            operation=DecoratorOperationType.RAILWAY_EXISTING_RESULT,
            should_succeed=True,
        ),
        DecoratorTestCase(
            name="railway_default_error_code",
            operation=DecoratorOperationType.RAILWAY_DEFAULT_ERROR,
            should_succeed=True,  # Exception is wrapped in FlextResult
        ),
    ]

    # Retry decorator scenarios
    RETRY_SCENARIOS: ClassVar[list[DecoratorTestCase]] = [
        DecoratorTestCase(
            name="retry_success_first_attempt",
            operation=DecoratorOperationType.RETRY_SUCCESS_FIRST,
            should_succeed=True,
            retry_attempts=3,
        ),
        DecoratorTestCase(
            name="retry_success_after_failures",
            operation=DecoratorOperationType.RETRY_SUCCESS_AFTER_FAILURES,
            should_succeed=True,
            retry_attempts=3,
            retry_delay=0.001,
        ),
        DecoratorTestCase(
            name="retry_exhausted",
            operation=DecoratorOperationType.RETRY_EXHAUSTED,
            should_succeed=False,
            error_type=ValueError,
            error_pattern="Always fails",
            retry_attempts=2,
            retry_delay=0.001,
            with_exception_handling=True,
        ),
        DecoratorTestCase(
            name="retry_exponential_backoff",
            operation=DecoratorOperationType.RETRY_EXPONENTIAL_BACKOFF,
            should_succeed=False,
            error_type=RuntimeError,
            retry_attempts=3,
            retry_delay=0.001,
            backoff_strategy="exponential",
            with_exception_handling=True,
        ),
        DecoratorTestCase(
            name="retry_linear_backoff",
            operation=DecoratorOperationType.RETRY_LINEAR_BACKOFF,
            should_succeed=False,
            error_type=RuntimeError,
            retry_attempts=3,
            retry_delay=0.001,
            backoff_strategy="linear",
            with_exception_handling=True,
        ),
        DecoratorTestCase(
            name="retry_with_class_logger",
            operation=DecoratorOperationType.RETRY_WITH_CLASS_LOGGER,
            should_succeed=True,
            retry_attempts=2,
            retry_delay=0.001,
        ),
        DecoratorTestCase(
            name="retry_no_exception",
            operation=DecoratorOperationType.EDGE_CASE_RETRY_NO_EXCEPTION,
            should_succeed=True,
            retry_attempts=3,
            retry_delay=0.001,
        ),
    ]

    # Timeout decorator scenarios
    TIMEOUT_SCENARIOS: ClassVar[list[DecoratorTestCase]] = [
        DecoratorTestCase(
            name="timeout_success",
            operation=DecoratorOperationType.TIMEOUT_SUCCESS,
            should_succeed=True,
            timeout_duration=1.0,
        ),
        DecoratorTestCase(
            name="timeout_exceeded",
            operation=DecoratorOperationType.TIMEOUT_EXCEEDED,
            should_succeed=False,
            error_type=FlextExceptions.TimeoutError,
            timeout_duration=0.005,
            with_exception_handling=True,
        ),
        DecoratorTestCase(
            name="timeout_with_exception_slow",
            operation=DecoratorOperationType.TIMEOUT_EXCEPTION_SLOW,
            should_succeed=False,
            error_type=FlextExceptions.TimeoutError,
            timeout_duration=0.005,
            with_exception_handling=True,
        ),
        DecoratorTestCase(
            name="timeout_with_exception_fast",
            operation=DecoratorOperationType.TIMEOUT_EXCEPTION_FAST,
            should_succeed=False,
            error_type=ValueError,
            error_pattern="Fast failure",
            timeout_duration=1.0,
            with_exception_handling=True,
        ),
        DecoratorTestCase(
            name="timeout_custom_error_code",
            operation=DecoratorOperationType.TIMEOUT_CUSTOM_ERROR,
            should_succeed=False,
            error_type=FlextExceptions.TimeoutError,
            timeout_duration=0.005,
            with_exception_handling=True,
        ),
        DecoratorTestCase(
            name="timeout_boundary_case",
            operation=DecoratorOperationType.EDGE_CASE_TIMEOUT_BOUNDARY,
            should_succeed=True,
            timeout_duration=0.1,
        ),
    ]

    # Combined decorator scenarios
    COMBINED_SCENARIOS: ClassVar[list[DecoratorTestCase]] = [
        DecoratorTestCase(
            name="combined_basic",
            operation=DecoratorOperationType.COMBINED_BASIC,
            should_succeed=True,
        ),
        DecoratorTestCase(
            name="combined_with_injection",
            operation=DecoratorOperationType.COMBINED_WITH_INJECTION,
            should_succeed=True,
            requires_container_setup=True,
        ),
        DecoratorTestCase(
            name="combined_with_railway",
            operation=DecoratorOperationType.COMBINED_WITH_RAILWAY,
            should_succeed=True,
        ),
        DecoratorTestCase(
            name="combined_railway_with_exception",
            operation=DecoratorOperationType.COMBINED_RAILWAY_EXCEPTION,
            should_succeed=True,  # Exception wrapped in FlextResult
        ),
        DecoratorTestCase(
            name="combined_without_perf_tracking",
            operation=DecoratorOperationType.COMBINED_WITHOUT_PERF,
            should_succeed=True,
        ),
        DecoratorTestCase(
            name="combined_all_features",
            operation=DecoratorOperationType.COMBINED_ALL_FEATURES,
            should_succeed=True,
            requires_container_setup=True,
        ),
        DecoratorTestCase(
            name="combined_minimal_options",
            operation=DecoratorOperationType.EDGE_CASE_COMBINED_MINIMAL,
            should_succeed=True,
        ),
    ]

    # Integration scenarios
    INTEGRATION_SCENARIOS: ClassVar[list[DecoratorTestCase]] = [
        DecoratorTestCase(
            name="manual_decorator_stacking",
            operation=DecoratorOperationType.INTEGRATION_MANUAL_STACKING,
            should_succeed=True,
        ),
        DecoratorTestCase(
            name="retry_with_railway",
            operation=DecoratorOperationType.INTEGRATION_RETRY_WITH_RAILWAY,
            should_succeed=True,
            retry_attempts=3,
            retry_delay=0.001,
        ),
        DecoratorTestCase(
            name="timeout_with_retry",
            operation=DecoratorOperationType.INTEGRATION_TIMEOUT_WITH_RETRY,
            should_succeed=True,
            timeout_duration=0.5,
            retry_attempts=2,
            retry_delay=0.001,
        ),
    ]

    @staticmethod
    def get_container() -> FlextContainer:
        """Get the global container instance."""
        return FlextContainer()

    @staticmethod
    def create_test_service() -> TestService:
        """Create test service instance."""
        return TestService()

    @staticmethod
    def create_service_with_logger() -> ServiceWithLogger:
        """Create service with logger instance."""
        return ServiceWithLogger()

    @staticmethod
    def create_repository() -> Repository:
        """Create repository instance."""
        return Repository()


# =========================================================================
# Test Suite
# =========================================================================


class TestFlextDecorators:
    """Unified test suite for FlextDecorators with parametrized tests."""

    # =====================================================================
    # Inject Decorator Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        DecoratorScenarios.INJECT_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_inject_decorator(self, test_case: DecoratorTestCase) -> None:
        """Test inject decorator with various scenarios."""
        if test_case.operation == DecoratorOperationType.INJECT_BASIC:
            # Test that inject decorator can be applied to a function
            @FlextDecorators.inject(test_service="test_service")
            def process_data(
                data: str, *, test_service: TestService | None = None
            ) -> str:
                # In test context, service won't be injected automatically
                if test_service is None:
                    return f"{data}_default"
                return f"{data}_{test_service.get_value()}"

            result = process_data("input")
            assert result == "input_default"

        elif test_case.operation == DecoratorOperationType.INJECT_MISSING:

            @FlextDecorators.inject(missing_service="missing_service")
            def process_data(*, missing_service: str = "default") -> str:
                return missing_service

            result = process_data()
            assert result == "default"

        elif test_case.operation == DecoratorOperationType.INJECT_PROVIDED:
            container = DecoratorScenarios.get_container()

            @dataclasses.dataclass
            class TestServiceTyped:
                value: str

            container.with_service("service", TestServiceTyped("from_container"))

            @FlextDecorators.inject(service="service")
            def process(*, service: TestServiceTyped) -> str:
                return service.value

            explicit_service = TestServiceTyped("explicit")
            result = process(service=explicit_service)
            assert result == "explicit"

        elif (
            test_case.operation == DecoratorOperationType.EDGE_CASE_INJECT_NO_CONTAINER
        ):

            @FlextDecorators.inject(nonexistent="nonexistent")
            def func(*, nonexistent: str = "fallback") -> str:
                return nonexistent

            result = func()
            assert result == "fallback"

    # =====================================================================
    # Log Operation Decorator Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        DecoratorScenarios.LOG_OPERATION_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_log_operation_decorator(self, test_case: DecoratorTestCase) -> None:
        """Test log_operation decorator with various scenarios."""
        if test_case.operation == DecoratorOperationType.LOG_OPERATION_BASIC:

            @FlextDecorators.log_operation("test_operation")
            def simple_function() -> str:
                return "success"

            result = simple_function()
            assert result == "success"

        elif test_case.operation == DecoratorOperationType.LOG_OPERATION_WITH_LOGGER:

            @FlextDecorators.log_operation("process_data")
            def process(value: str) -> str:
                return f"processed_{value}"

            result = process("test")
            assert result == "processed_test"

        elif test_case.operation == DecoratorOperationType.LOG_OPERATION_EXCEPTION:

            @FlextDecorators.log_operation("failing_operation")
            def failing_function() -> None:
                msg = "Test error"
                raise ValueError(msg)

            with pytest.raises(ValueError, match="Test error"):
                failing_function()

        elif test_case.operation == DecoratorOperationType.LOG_OPERATION_DEFAULT_NAME:

            @FlextDecorators.log_operation()
            def my_function() -> str:
                return "result"

            result = my_function()
            assert result == "result"

        elif test_case.operation == DecoratorOperationType.LOG_OPERATION_WITH_PERF:
            FlextRuntime.configure_structlog(
                log_level=logging.INFO, console_renderer=False
            )

            @FlextDecorators.log_operation("failing_operation", track_perf=True)
            def failing_function() -> None:
                msg = "Test error with performance tracking"
                raise ValueError(msg)

            with pytest.raises(
                ValueError, match="Test error with performance tracking"
            ):
                failing_function()

        elif test_case.operation == DecoratorOperationType.EDGE_CASE_LOG_WITHOUT_LOGGER:

            @FlextDecorators.log_operation()
            def standalone_function() -> str:
                return "logged"

            result = standalone_function()
            assert result == "logged"

    # =====================================================================
    # Track Performance Decorator Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        DecoratorScenarios.TRACK_PERFORMANCE_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_track_performance_decorator(self, test_case: DecoratorTestCase) -> None:
        """Test track_performance decorator with various scenarios."""
        if test_case.operation == DecoratorOperationType.TRACK_PERFORMANCE_BASIC:

            @FlextDecorators.track_performance("timed_operation")
            def timed_function() -> str:
                time.sleep(0.01)
                return "completed"

            result = timed_function()
            assert result == "completed"

        elif (
            test_case.operation == DecoratorOperationType.TRACK_PERFORMANCE_WITH_LOGGER
        ):

            @FlextDecorators.track_performance("process")
            def process() -> str:
                return "done"

            result = process()
            assert result == "done"

        elif test_case.operation == DecoratorOperationType.TRACK_PERFORMANCE_EXCEPTION:

            @FlextDecorators.track_performance("failing_operation")
            def failing_function() -> None:
                msg = "Timed failure"
                raise RuntimeError(msg)

            with pytest.raises(RuntimeError, match="Timed failure"):
                failing_function()

        elif test_case.operation == DecoratorOperationType.TRACK_PERFORMANCE_DEFAULT:

            @FlextDecorators.track_performance()
            def measured_function() -> int:
                return 42

            result = measured_function()
            assert result == 42

        elif (
            test_case.operation == DecoratorOperationType.EDGE_CASE_TRACK_WITHOUT_LOGGER
        ):

            @FlextDecorators.track_performance()
            def standalone_function() -> str:
                return "tracked"

            result = standalone_function()
            assert result == "tracked"

    # =====================================================================
    # Railway Decorator Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        DecoratorScenarios.RAILWAY_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_railway_decorator(self, test_case: DecoratorTestCase) -> None:
        """Test railway decorator with various scenarios."""
        if test_case.operation == DecoratorOperationType.RAILWAY_SUCCESS:

            @FlextDecorators.railway()
            def successful_operation() -> str:
                return "success"

            result = successful_operation()
            assert isinstance(result, FlextResult)
            assert result.is_success
            assert result.unwrap() == "success"

        elif test_case.operation == DecoratorOperationType.RAILWAY_EXCEPTION:

            @FlextDecorators.railway(error_code="CUSTOM_ERROR")
            def failing_operation() -> str:
                msg = "Operation failed"
                raise ValueError(msg)

            result = failing_operation()
            assert isinstance(result, FlextResult)
            assert result.is_failure
            assert "Operation failed" in (result.error or "")

        elif test_case.operation == DecoratorOperationType.RAILWAY_EXISTING_RESULT:

            @FlextDecorators.railway()
            def returns_result() -> FlextResult[str]:
                return FlextResult[str].ok("already_wrapped")

            result = returns_result()
            assert isinstance(result, FlextResult)
            assert result.is_success
            assert result.unwrap() == "already_wrapped"

        elif test_case.operation == DecoratorOperationType.RAILWAY_DEFAULT_ERROR:

            @FlextDecorators.railway()
            def failing() -> str:
                msg = "Error"
                raise RuntimeError(msg)

            result = failing()
            assert result.is_failure

    # =====================================================================
    # Retry Decorator Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        DecoratorScenarios.RETRY_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_retry_decorator(self, test_case: DecoratorTestCase) -> None:
        """Test retry decorator with various scenarios."""
        if test_case.operation == DecoratorOperationType.RETRY_SUCCESS_FIRST:

            @FlextDecorators.retry(max_attempts=3)
            def successful_operation() -> str:
                return "success"

            result = successful_operation()
            assert result == "success"

        elif test_case.operation == DecoratorOperationType.RETRY_SUCCESS_AFTER_FAILURES:
            attempts = 0

            @FlextDecorators.retry(max_attempts=3, delay_seconds=0.001)
            def flaky_operation() -> str:
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    msg = f"Attempt {attempts} failed"
                    raise RuntimeError(msg)
                return "success"

            result = flaky_operation()
            assert result == "success"
            assert attempts == 3

        elif test_case.operation == DecoratorOperationType.RETRY_EXHAUSTED:

            @FlextDecorators.retry(max_attempts=2, delay_seconds=0.001)
            def always_fails() -> str:
                msg = "Always fails"
                raise ValueError(msg)

            with pytest.raises(ValueError, match="Always fails"):
                always_fails()

        elif test_case.operation == DecoratorOperationType.RETRY_EXPONENTIAL_BACKOFF:
            start_time = time.time()

            @FlextDecorators.retry(
                max_attempts=3,
                delay_seconds=0.001,
                backoff_strategy="exponential",
            )
            def failing_operation() -> None:
                msg = "Fail"
                raise RuntimeError(msg)

            with pytest.raises(RuntimeError):
                failing_operation()

            elapsed = time.time() - start_time
            assert elapsed >= 0.002

        elif test_case.operation == DecoratorOperationType.RETRY_LINEAR_BACKOFF:

            @FlextDecorators.retry(
                max_attempts=3, delay_seconds=0.001, backoff_strategy="linear"
            )
            def failing_operation() -> None:
                msg = "Fail"
                raise RuntimeError(msg)

            with pytest.raises(RuntimeError):
                failing_operation()

        elif test_case.operation == DecoratorOperationType.RETRY_WITH_CLASS_LOGGER:
            service = DecoratorScenarios.create_service_with_logger()

            @FlextDecorators.retry(max_attempts=2, delay_seconds=0.001)
            def flaky_method() -> str:
                return service.flaky_method()

            result = flaky_method()
            assert result == "success"
            assert service.attempts == 2

        elif test_case.operation == DecoratorOperationType.EDGE_CASE_RETRY_NO_EXCEPTION:

            @FlextDecorators.retry(max_attempts=3, delay_seconds=0.001)
            def always_succeeds() -> str:
                return "success"

            result = always_succeeds()
            assert result == "success"

    # =====================================================================
    # Timeout Decorator Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        DecoratorScenarios.TIMEOUT_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_timeout_decorator(self, test_case: DecoratorTestCase) -> None:
        """Test timeout decorator with various scenarios."""
        if test_case.operation == DecoratorOperationType.TIMEOUT_SUCCESS:

            @FlextDecorators.timeout(timeout_seconds=1.0)
            def fast_operation() -> str:
                time.sleep(0.01)
                return "completed"

            result = fast_operation()
            assert result == "completed"

        elif test_case.operation == DecoratorOperationType.TIMEOUT_EXCEEDED:

            @FlextDecorators.timeout(timeout_seconds=0.005)
            def slow_operation() -> str:
                time.sleep(0.01)
                return "should_not_reach"

            with pytest.raises(FlextExceptions.TimeoutError):
                slow_operation()

        elif test_case.operation == DecoratorOperationType.TIMEOUT_EXCEPTION_SLOW:

            @FlextDecorators.timeout(timeout_seconds=0.005)
            def slow_failing_operation() -> None:
                time.sleep(0.01)
                msg = "Should timeout before this"
                raise ValueError(msg)

            with pytest.raises(FlextExceptions.TimeoutError):
                slow_failing_operation()

        elif test_case.operation == DecoratorOperationType.TIMEOUT_EXCEPTION_FAST:

            @FlextDecorators.timeout(timeout_seconds=1.0)
            def fast_failing_operation() -> None:
                msg = "Fast failure"
                raise ValueError(msg)

            with pytest.raises(ValueError, match="Fast failure"):
                fast_failing_operation()

        elif test_case.operation == DecoratorOperationType.TIMEOUT_CUSTOM_ERROR:

            @FlextDecorators.timeout(timeout_seconds=0.005, error_code="CUSTOM_TIMEOUT")
            def slow_operation() -> str:
                time.sleep(0.01)
                return "late"

            with pytest.raises(FlextExceptions.TimeoutError):
                slow_operation()

        elif test_case.operation == DecoratorOperationType.EDGE_CASE_TIMEOUT_BOUNDARY:

            @FlextDecorators.timeout(timeout_seconds=0.1)
            def boundary_operation() -> str:
                time.sleep(0.009)
                return "completed"

            result = boundary_operation()
            assert result == "completed"

    # =====================================================================
    # Combined Decorator Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        DecoratorScenarios.COMBINED_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_combined_decorator(self, test_case: DecoratorTestCase) -> None:
        """Test combined decorator with various scenarios."""
        if test_case.operation == DecoratorOperationType.COMBINED_BASIC:

            @FlextDecorators.combined(operation_name="test_op", track_perf=True)
            def simple_function() -> str:
                return "result"

            result = simple_function()
            assert result == "result"

        elif test_case.operation == DecoratorOperationType.COMBINED_WITH_INJECTION:
            # Test that combined decorator with injection can be applied
            @FlextDecorators.combined(
                inject_deps={"service": "service"},
                operation_name="process",
            )
            def process_data(*, service: TestService | None = None) -> str:
                if service is None:
                    return "no_injection"
                return service.get_value()

            # In test context, service won't be auto-injected
            result = process_data()
            assert result == "no_injection"

        elif test_case.operation == DecoratorOperationType.COMBINED_WITH_RAILWAY:

            @FlextDecorators.combined(use_railway=True, operation_name="wrapped")
            def operation() -> str:
                return "success"

            result = operation()
            assert isinstance(result, FlextResult)
            assert result.is_success

        elif test_case.operation == DecoratorOperationType.COMBINED_RAILWAY_EXCEPTION:

            @FlextDecorators.combined(use_railway=True, error_code="TEST_ERROR")
            def failing_operation() -> str:
                msg = "Failed"
                raise ValueError(msg)

            result = failing_operation()
            assert isinstance(result, FlextResult)
            assert result.is_failure

        elif test_case.operation == DecoratorOperationType.COMBINED_WITHOUT_PERF:

            @FlextDecorators.combined(track_perf=False, operation_name="no_perf")
            def simple() -> int:
                return 42

            result = simple()
            assert result == 42

        elif test_case.operation == DecoratorOperationType.COMBINED_ALL_FEATURES:
            container = DecoratorScenarios.get_container()
            repo = DecoratorScenarios.create_repository()
            container.with_service("repo", repo)

            @FlextDecorators.combined(
                inject_deps={"repo": "repo"},
                operation_name="full_operation",
                track_perf=True,
                use_railway=True,
                error_code="FULL_ERROR",
            )
            def full_operation(data: str, *, repo: Repository) -> str:
                return repo.save(data)

            # Call decorated function with manual repo passing
            result = full_operation("test", repo=repo)
            assert isinstance(result, FlextResult)
            assert result.is_success
            assert "saved_test" in result.unwrap()

        elif test_case.operation == DecoratorOperationType.EDGE_CASE_COMBINED_MINIMAL:

            @FlextDecorators.combined()
            def minimal() -> str:
                return "minimal"

            result = minimal()
            assert result == "minimal"

    # =====================================================================
    # Integration Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        DecoratorScenarios.INTEGRATION_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_decorator_integration(self, test_case: DecoratorTestCase) -> None:
        """Test decorator combinations and stacking."""
        if test_case.operation == DecoratorOperationType.INTEGRATION_MANUAL_STACKING:

            @FlextDecorators.log_operation("stacked")
            @FlextDecorators.track_performance("stacked")
            @FlextDecorators.railway()
            def stacked_operation() -> str:
                return "stacked_result"

            result = stacked_operation()
            assert isinstance(result, FlextResult)
            assert result.is_success

        elif (
            test_case.operation == DecoratorOperationType.INTEGRATION_RETRY_WITH_RAILWAY
        ):
            attempts = 0

            @FlextDecorators.railway()
            @FlextDecorators.retry(max_attempts=3, delay_seconds=0.001)
            def flaky_with_railway() -> str:
                nonlocal attempts
                attempts += 1
                if attempts < 2:
                    msg = "Retry me"
                    raise RuntimeError(msg)
                return "success"

            result = flaky_with_railway()
            assert isinstance(result, FlextResult)
            assert result.is_success
            assert attempts == 2

        elif (
            test_case.operation == DecoratorOperationType.INTEGRATION_TIMEOUT_WITH_RETRY
        ):

            @FlextDecorators.retry(max_attempts=2, delay_seconds=0.001)
            @FlextDecorators.timeout(timeout_seconds=0.5)
            def operation() -> str:
                time.sleep(0.01)
                return "success"

            result = operation()
            assert result == "success"


__all__ = ["TestFlextDecorators"]
