"""FLEXT Core Reliability Utilities Tests - Comprehensive Coverage.

Tests for flext_core._utilities.reliability.FlextUtilitiesReliability covering:
- Timeout operations (success, timeout, exception, failure paths)
- Retry mechanisms (success after failure, parameter validation)
- Delay calculations (exponential and linear backoff)
- Controlled retries (should_retry_func, cleanup_func)

Modules tested: flext_core._utilities.reliability.FlextUtilitiesReliability
Scope: All reliability utility methods with 100% coverage including edge cases

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Final, Never

import pytest
from flext_core import FlextRuntime, r
from flext_core.typings import t
from flext_tests import u


class TestFlextUtilitiesReliability:
    """Comprehensive tests for FlextUtilitiesReliability.

    Uses advanced Python 3.13 patterns.

    Uses factories, enums, mappings, and dynamic tests to reduce code while maintaining
    100% coverage. All test constants organized in nested classes.
    """

    class TimeoutScenario(StrEnum):
        """Timeout test scenarios."""

        SUCCESS = "success"
        TIMEOUT = "timeout"
        EXCEPTION = "exception"
        FAILURE = "failure"

    class RetryScenario(StrEnum):
        """Retry test scenarios."""

        SUCCESS_AFTER_FAILURE = "success_after_failure"
        VALIDATION_ERROR = "validation_error"
        CONTROLLED_RETRY = "controlled_retry"
        BLOCKED_RETRY = "blocked_retry"

    class DelayConfig(StrEnum):
        """Delay configuration types."""

        EXPONENTIAL = "exponential"
        LINEAR = "linear"

    class Constants:
        """Test constants for reliability utilities."""

        TIMEOUT_SHORT: Final[float] = 0.01
        TIMEOUT_MEDIUM: Final[float] = 0.05
        TIMEOUT_LONG: Final[float] = 0.1
        DELAY_INITIAL: Final[float] = 0.1
        DELAY_MAX: Final[float] = 1.0
        DELAY_LINEAR_INITIAL: Final[float] = 0.2
        DELAY_LINEAR_MAX: Final[float] = 0.5
        BACKOFF_MULTIPLIER: Final[float] = 1.5
        MAX_ATTEMPTS_VALID: Final[int] = 3
        MAX_ATTEMPTS_INVALID: Final[int] = 0
        SUCCESS_VALUE: Final[int] = 42
        SUCCESS_STRING: Final[str] = "done"
        SUCCESS_RETRY: Final[str] = "ok"

    _constants = Constants()

    class Factories:
        """Factories for creating test operations and configurations."""

        @staticmethod
        def _create_timeout_operation() -> r[str]:
            """Create timeout operation that sleeps then returns result."""
            time.sleep(0.05)
            return r[str].ok("late")

        @staticmethod
        def create_timeout_operation(
            scenario: TestFlextUtilitiesReliability.TimeoutScenario,
        ) -> Callable[[], r[str]]:
            """Create timeout operation for given scenario."""
            scenarios: Mapping[
                TestFlextUtilitiesReliability.TimeoutScenario,
                Callable[[], r[str]],
            ] = {
                TestFlextUtilitiesReliability.TimeoutScenario.SUCCESS: lambda: r[
                    str
                ].ok("done"),
                TestFlextUtilitiesReliability.TimeoutScenario.TIMEOUT: (
                    TestFlextUtilitiesReliability.Factories._create_timeout_operation
                ),
                TestFlextUtilitiesReliability.TimeoutScenario.EXCEPTION: lambda: (
                    _ for _ in ()
                ).throw(ValueError("boom")),
                TestFlextUtilitiesReliability.TimeoutScenario.FAILURE: lambda: r[
                    str
                ].fail("no result"),
            }
            return scenarios[scenario]

        @staticmethod
        def create_retry_operation(
            success_after: int,
            success_value: int = 42,
        ) -> tuple[Callable[[], r[int]], list[int]]:
            """Create retry operation that succeeds after N attempts."""
            attempts: list[int] = []

            def op() -> r[int]:
                attempts.append(len(attempts))
                if len(attempts) >= success_after:
                    return r[int].ok(success_value)
                return r[int].fail("transient")

            return op, attempts

        @staticmethod
        def create_delay_config(
            config_type: TestFlextUtilitiesReliability.DelayConfig,
        ) -> dict[str, t.GeneralValueType]:
            """Create delay configuration for given type."""
            configs: Mapping[
                TestFlextUtilitiesReliability.DelayConfig,
                dict[str, t.GeneralValueType],
            ] = {
                TestFlextUtilitiesReliability.DelayConfig.EXPONENTIAL: {
                    "initial_delay_seconds": 0.1,
                    "max_delay_seconds": 1.0,
                    "exponential_backoff": True,
                    "backoff_multiplier": 1.5,
                },
                TestFlextUtilitiesReliability.DelayConfig.LINEAR: {
                    "initial_delay_seconds": 0.2,
                    "max_delay_seconds": 0.5,
                    "exponential_backoff": False,
                    "backoff_multiplier": None,
                },
            }
            return configs[config_type]

        @staticmethod
        def create_controlled_retry_operation(
            success_after: int,
            constants: TestFlextUtilitiesReliability.Constants,
            retry_error: str = "retry me",
        ) -> tuple[
            Callable[[], r[str]],
            Callable[[int, str | None], bool],
            list[int],
            list[str],
        ]:
            """Create controlled retry operation with should_retry and cleanup."""
            attempts: list[int] = []
            cleanups: list[str] = []

            def op() -> r[str]:
                attempts.append(len(attempts))
                if len(attempts) >= success_after:
                    return r[str].ok(constants.SUCCESS_RETRY)
                return r[str].fail(retry_error)

            def should_retry(attempt: int, error: str | None) -> bool:
                return attempt == 0 and error == retry_error

            def cleanup() -> None:
                cleanups.append("done")

            return op, should_retry, attempts, cleanups

    @pytest.mark.parametrize(
        ("scenario", "timeout", "expected_success", "error_pattern"),
        [
            (
                "success",
                0.1,
                True,
                None,
            ),
            (
                "timeout",
                0.01,
                False,
                "timed out",
            ),
            (
                "exception",
                0.1,
                False,
                "exception",
            ),
            (
                "failure",
                0.1,
                False,
                "no result",
            ),
        ],
    )
    def test_with_timeout_scenarios(
        self,
        scenario: str,
        timeout: float,
        expected_success: bool,
        error_pattern: str | None,
    ) -> None:
        """Test timeout operations covering all scenarios."""
        scenario_enum = self.TimeoutScenario(scenario)
        operation = self.Factories.create_timeout_operation(scenario_enum)
        result = u.Reliability.with_timeout(operation, timeout)

        if expected_success:
            u.Tests.Result.assert_success_with_value(
                result,
                self.Constants.SUCCESS_STRING,
            )
        else:
            assert error_pattern is not None
            u.Tests.Result.assert_result_failure(result)
            assert error_pattern in (result.error or "")

    def test_with_timeout_invalid_timeout(self) -> None:
        """Test timeout validation."""
        result = u.Reliability.with_timeout(
            lambda: r[str].ok("test"),
            -1.0,
        )
        u.Tests.Result.assert_result_failure(result)
        assert "Timeout must be positive" in (result.error or "")

    def test_retry_succeeds_after_failure(self) -> None:
        """Test retry succeeds after initial failure."""
        op, attempts = self.Factories.create_retry_operation(success_after=2)

        result: r[int] = u.Reliability.retry(
            op,
            max_attempts=self.Constants.MAX_ATTEMPTS_VALID,
            delay_seconds=0.0,
        )

        u.Tests.Result.assert_success_with_value(
            result,
            self.Constants.SUCCESS_VALUE,
        )
        assert len(attempts) == 2

    def test_retry_validation_error(self) -> None:
        """Test retry parameter validation."""
        result: r[int] = u.Reliability.retry(
            lambda: r[int].fail("fail"),
            max_attempts=self.Constants.MAX_ATTEMPTS_INVALID,
        )
        u.Tests.Result.assert_result_failure(result)
        assert "Max attempts must be at least" in (result.error or "")

    @pytest.mark.parametrize(
        ("config_type", "attempt", "expected_min", "expected_max", "expected_exact"),
        [
            (
                "exponential",
                0,
                0.1,
                0.2,
                None,
            ),
            (
                "exponential",
                1,
                0.1,
                None,
                None,
            ),
            (
                "linear",
                2,
                None,
                None,
                0.5,
            ),
        ],
    )
    def test_calculate_delay_configs(
        self,
        config_type: str,
        attempt: int,
        expected_min: float | None,
        expected_max: float | None,
        expected_exact: float | None,
    ) -> None:
        """Test delay calculation for exponential and linear configs."""
        config_enum = self.DelayConfig(config_type)
        config = self.Factories.create_delay_config(config_enum)
        delay = u.Reliability.calculate_delay(attempt, config)

        if expected_exact is not None:
            assert delay == expected_exact
        else:
            if expected_min is not None:
                assert delay >= expected_min
            if expected_max is not None:
                assert delay <= expected_max
            if config_type == "exponential" and attempt > 0:
                # Verify exponential growth
                delay0 = u.Reliability.calculate_delay(0, config)
                assert delay > delay0

    def test_with_retry_controlled_retries(self) -> None:
        """Test controlled retries with should_retry_func and cleanup."""
        (
            op,
            should_retry,
            attempts,
            cleanups,
        ) = self.Factories.create_controlled_retry_operation(
            success_after=2,
            constants=self._constants,
        )

        result = u.Reliability.with_retry(
            op,
            max_attempts=self.Constants.MAX_ATTEMPTS_VALID,
            should_retry_func=should_retry,
            cleanup_func=lambda: cleanups.append("done"),
        )

        assert result.is_success
        assert result.value == self.Constants.SUCCESS_RETRY
        assert cleanups == ["done"]
        assert attempts == [0, 1]

    def test_with_retry_blocked(self) -> None:
        """Test retry blocked by should_retry_func."""
        blocked: FlextRuntime.RuntimeResult[Never] = u.Reliability.with_retry(
            lambda: r[str].fail("stop"),
            max_attempts=2,
            should_retry_func=lambda attempt, _error: attempt == 0,
        )
        assert blocked.is_failure
        assert blocked.error is not None and "stop" in blocked.error
