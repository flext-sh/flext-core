"""Dispatcher infrastructure models extracted from dispatcher helpers.

This module contains BaseModel subclasses used by dispatcher reliability and
timeout helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import concurrent.futures
import secrets
import time
from collections.abc import Mapping
from typing import Annotated, override

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from flext_core import c, r, t


class FlextModelsDispatcher:
    """Dispatcher infrastructure models."""

    class TimeoutEnforcer(BaseModel):
        """Manage timeout enforcement and dispatcher thread-pool execution."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        use_timeout_executor: bool = Field(
            description="Whether timeout executor is enabled",
        )
        executor_workers: Annotated[
            t.PositiveInt,
            Field(
                description="Number of worker threads for timeout executor",
            ),
        ]
        _executor: concurrent.futures.ThreadPoolExecutor | None = PrivateAttr(
            default=None,
        )

        def __init__(
            self,
            *,
            use_timeout_executor: bool,
            executor_workers: int,
        ) -> None:
            """Initialize the timeout coordinator.

            Args:
                use_timeout_executor: Whether to route handler execution through a
                    dedicated timeout executor
                executor_workers: Number of worker threads to provision when the
                    executor is enabled

            """
            super().__init__(
                use_timeout_executor=use_timeout_executor,
                executor_workers=executor_workers,
            )

        @override
        def model_post_init(self, __context: dict[str, t.Scalar] | None, /) -> None:
            self.executor_workers = max(
                self.executor_workers,
                c.RETRY_COUNT_MIN,
            )

        def cleanup(self) -> None:
            """Release executor resources used by dispatcher timeout handling."""
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None

        def ensure_executor(self) -> concurrent.futures.ThreadPoolExecutor:
            """Create the shared executor on demand with lazy initialization.

            Returns:
                ThreadPoolExecutor: The shared thread pool executor instance.

            """
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.executor_workers,
                    thread_name_prefix=c.THREAD_NAME_PREFIX,
                )
            return self._executor

        def get_executor_status(self) -> Mapping[str, t.Scalar]:
            """Return executor status metadata for diagnostics and metrics.

            Returns:
                ConfigurationDict: Dictionary with executor status information.

            """
            return {
                "executor_active": self._executor is not None,
                "executor_workers": self.executor_workers if self._executor else 0,
            }

        def reset_executor(self) -> None:
            """Reset executor after shutdown to allow lazy re-creation."""
            self._executor = None

        def resolve_workers(self) -> int:
            """Return the configured worker count for the dispatcher executor.

            Returns:
                int: Number of worker threads configured for the executor.

            """
            return self.executor_workers

        def should_use_executor(self) -> bool:
            """Return ``True`` when a dedicated timeout executor is enabled.

            Returns:
                bool: True if timeout executor is enabled, False otherwise.

            """
            return self.use_timeout_executor

    class CircuitBreakerManager(BaseModel):
        """Manage per-message circuit breaker state for dispatcher executions.

        Handles state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED) with
        configurable thresholds and recovery timeouts to protect downstream
        handlers from cascading failures.
        """

        model_config = ConfigDict(arbitrary_types_allowed=True)

        threshold: Annotated[
            t.PositiveInt,
            Field(description="Failure count before opening circuit"),
        ]
        recovery_timeout: Annotated[
            t.PositiveFloat,
            Field(
                description="Seconds before attempting recovery",
            ),
        ]
        success_threshold: Annotated[
            t.PositiveInt,
            Field(
                description="Successes needed to close from half-open",
            ),
        ]
        _failures: dict[str, int] = PrivateAttr(
            default_factory=lambda: dict[str, int](),
        )
        _states: dict[str, str] = PrivateAttr(default_factory=lambda: dict[str, str]())
        _opened_at: dict[str, float] = PrivateAttr(
            default_factory=lambda: dict[str, float](),
        )
        _success_counts: dict[str, int] = PrivateAttr(
            default_factory=lambda: dict[str, int](),
        )
        _recovery_successes: dict[str, int] = PrivateAttr(
            default_factory=lambda: dict[str, int](),
        )
        _recovery_failures: dict[str, int] = PrivateAttr(
            default_factory=lambda: dict[str, int](),
        )
        _total_successes: dict[str, int] = PrivateAttr(
            default_factory=lambda: dict[str, int](),
        )

        def __init__(
            self,
            threshold: int,
            recovery_timeout: float,
            success_threshold: int,
        ) -> None:
            """Initialize circuit breaker manager.

            Args:
                threshold: Failure count before opening circuit
                recovery_timeout: Seconds before attempting recovery
                success_threshold: Successes needed to close from half-open

            """
            super().__init__(
                threshold=threshold,
                recovery_timeout=recovery_timeout,
                success_threshold=success_threshold,
            )

        def attempt_reset(self, message_type: str) -> None:
            """Attempt recovery if circuit is open."""
            if self.is_open(message_type):
                opened_at = self._opened_at.get(message_type, 0.0)
                if time.time() - opened_at >= self.recovery_timeout:
                    self.transition_to_half_open(message_type)

        def check_before_dispatch(self, message_type: str) -> r[bool]:
            """Return a result indicating whether dispatch can proceed.

            Returns:
                r[bool]: Success with ``True`` when the circuit is closed
                    or half-open; failure with metadata when the circuit remains
                    open.

            """
            self.attempt_reset(message_type)
            if self.is_open(message_type):
                return r[bool].fail(
                    f"Circuit breaker is open for message type '{message_type}'",
                    error_code=c.OPERATION_ERROR,
                    error_data=t.ConfigMap(
                        root={
                            "message_type": message_type,
                            "state": self.get_state(message_type),
                            "failure_count": self.get_failure_count(message_type),
                        },
                    ),
                )
            return r[bool].ok(value=True)

        def cleanup(self) -> None:
            """Clear all state."""
            self._failures.clear()
            self._states.clear()
            self._opened_at.clear()
            self._success_counts.clear()
            self._recovery_successes.clear()
            self._recovery_failures.clear()
            self._total_successes.clear()

        def get_failure_count(self, message_type: str) -> int:
            """Get current failure count.

            Returns:
                Current failure count for the message type.

            """
            return self._failures.get(message_type, 0)

        def get_metrics(self) -> Mapping[str, t.Scalar]:
            """Collect circuit breaker metrics, including recovery statistics.

            Returns:
                Dictionary containing circuit breaker metrics.

            """
            total_recovery_attempts = sum(
                self._recovery_successes.get(mt, 0) + self._recovery_failures.get(mt, 0)
                for mt in self._states
            )
            total_recovery_successes = sum(
                self._recovery_successes.get(mt, 0) for mt in self._states
            )
            recovery_success_rate = (
                total_recovery_successes
                / total_recovery_attempts
                * c.PERCENTAGE_MULTIPLIER
                if total_recovery_attempts > 0
                else 0.0
            )
            total_failures = sum(self._failures.values())
            total_successes = sum(self._total_successes.values())
            total_operations = total_failures + total_successes
            failure_rate = (
                total_failures / total_operations * c.PERCENTAGE_MULTIPLIER
                if total_operations > 0
                else 0.0
            )
            return {
                "failures": len(self._failures),
                "states": len(self._states),
                "open_count": sum(
                    1
                    for state in self._states.values()
                    if state == c.CircuitBreakerState.OPEN
                ),
                "recovery_success_rate": recovery_success_rate,
                "failure_rate": failure_rate,
                "total_recovery_attempts": total_recovery_attempts,
                "total_recovery_successes": total_recovery_successes,
                "total_operations": total_operations,
            }

        def get_state(self, message_type: str) -> str:
            """Get current state for message type.

            Returns:
                Current circuit breaker state for the message type.

            """
            return self._states.get(
                message_type,
                c.CircuitBreakerState.CLOSED,
            )

        def get_threshold(self) -> int:
            """Get circuit breaker threshold."""
            return self.threshold

        def is_open(self, message_type: str) -> bool:
            """Check if circuit breaker is open for message type.

            Returns:
                True if circuit breaker is open, False otherwise.

            """
            return self.get_state(message_type) == c.CircuitBreakerState.OPEN

        def record_failure(self, message_type: str) -> None:
            """Record failed operation and update state."""
            current_state = self.get_state(message_type)
            current_failures = self._failures.get(message_type, 0) + 1
            self._failures[message_type] = current_failures
            if current_state == c.CircuitBreakerState.HALF_OPEN:
                self._recovery_failures[message_type] = (
                    self._recovery_failures.get(message_type, 0) + 1
                )
                self.transition_to_open(message_type)
            elif (
                current_state == c.CircuitBreakerState.CLOSED
                and current_failures >= self.threshold
            ):
                self.transition_to_open(message_type)

        def record_success(self, message_type: str) -> None:
            """Record successful operation and update state."""
            current_state = self.get_state(message_type)
            self._total_successes[message_type] = (
                self._total_successes.get(message_type, 0) + 1
            )
            if current_state == c.CircuitBreakerState.HALF_OPEN:
                success_count = self._success_counts.get(message_type, 0) + 1
                self._success_counts[message_type] = success_count
                if success_count >= self.success_threshold:
                    self._recovery_successes[message_type] = (
                        self._recovery_successes.get(message_type, 0) + 1
                    )
                    self.transition_to_closed(message_type)
            elif current_state == c.CircuitBreakerState.CLOSED:
                self._failures[message_type] = 0

        def set_state(self, message_type: str, state: str) -> None:
            """Set state for message type."""
            self._states[message_type] = state

        def transition_to_closed(self, message_type: str) -> None:
            """Transition to CLOSED state."""
            self.transition_to_state(
                message_type,
                c.CircuitBreakerState.CLOSED,
            )

        def transition_to_half_open(self, message_type: str) -> None:
            """Transition to HALF_OPEN state."""
            self.transition_to_state(
                message_type,
                c.CircuitBreakerState.HALF_OPEN,
            )

        def transition_to_open(self, message_type: str) -> None:
            """Transition to OPEN state."""
            self.transition_to_state(
                message_type,
                c.CircuitBreakerState.OPEN,
            )

        def transition_to_state(self, message_type: str, new_state: str) -> None:
            """Transition to specified state."""
            self.set_state(message_type, new_state)
            if new_state == c.CircuitBreakerState.CLOSED:
                self._failures[message_type] = 0
                self._success_counts[message_type] = 0
                if message_type in self._opened_at:
                    del self._opened_at[message_type]
            elif new_state == c.CircuitBreakerState.OPEN:
                self._opened_at[message_type] = time.time()
                self._success_counts[message_type] = 0
            elif new_state == c.CircuitBreakerState.HALF_OPEN:
                self._success_counts[message_type] = 0

    class RateLimiterManager(BaseModel):
        """Enforce per-message rate limits with a sliding window algorithm."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        max_requests: t.PositiveInt = Field(
            description="Maximum requests allowed per window"
        )
        window_seconds: t.PositiveFloat = Field(
            description="Time window in seconds for rate limiting",
        )
        jitter_factor: t.DecimalFraction = Field(
            default=0.1,
            description="Jitter variance as fraction between 0.0 and 1.0",
        )
        _windows: dict[str, tuple[float, int]] = PrivateAttr(
            default_factory=lambda: dict[str, tuple[float, int]](),
        )

        def __init__(
            self,
            max_requests: int,
            window_seconds: float,
            jitter_factor: float = 0.1,
        ) -> None:
            """Initialize rate limiter manager.

            Args:
                max_requests: Maximum requests allowed per window
                window_seconds: Time window in seconds for rate limiting
                jitter_factor: Jitter variance as fraction (0.1 = +/-10%)

            """
            super().__init__(
                max_requests=max_requests,
                window_seconds=window_seconds,
                jitter_factor=jitter_factor,
            )

        @override
        def model_post_init(self, __context: dict[str, t.Scalar] | None, /) -> None:
            self.jitter_factor = max(0.0, min(self.jitter_factor, 1.0))

        def check_rate_limit(self, message_type: str) -> r[bool]:
            """Return whether dispatch is allowed under the current rate window.

            Returns:
                Success result if rate limit allows dispatch, failure otherwise.

            """
            current_time = time.time()
            window_start, count = self._windows.get(message_type, (current_time, 0))
            if current_time - window_start >= self.window_seconds:
                window_start = current_time
                count = 0
            if count >= self.max_requests:
                elapsed = current_time - window_start
                retry_after = max(0, int(self.window_seconds - elapsed))
                return r[bool].fail(
                    f"Rate limit exceeded for message type '{message_type}'",
                    error_code=c.OPERATION_ERROR,
                    error_data=t.ConfigMap(
                        root={
                            "message_type": message_type,
                            "limit": self.max_requests,
                            "window_seconds": self.window_seconds,
                            "current_count": count,
                            "retry_after": retry_after,
                        },
                    ),
                )
            self._windows[message_type] = (window_start, count + 1)
            return r[bool].ok(value=True)

        def cleanup(self) -> None:
            """Clear all rate limit windows."""
            self._windows.clear()

        def get_max_requests(self) -> int:
            """Get maximum requests per window."""
            return self.max_requests

        def get_window_seconds(self) -> float:
            """Get rate limit window duration in seconds."""
            return self.window_seconds

        def _apply_jitter(self, base_delay: float) -> float:
            """Apply jitter variance to a delay value.

            Returns:
                Jittered delay value.

            """
            if base_delay <= 0.0 or self.jitter_factor <= 0.0:
                return base_delay
            secure_random = secrets.SystemRandom()
            variance = (2.0 * secure_random.random() - 1.0) * self.jitter_factor
            jittered = base_delay * (1.0 + variance)
            return max(0.0, jittered)

    class RetryPolicy(BaseModel):
        """Coordinate retry attempts with configurable backoff for dispatcher steps."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        max_attempts: t.PositiveInt = Field(
            description="Maximum retry attempts allowed"
        )
        retry_delay: t.PositiveFloat = Field(
            description="Base delay in seconds between retry attempts",
        )
        _attempts: dict[str, int] = PrivateAttr(
            default_factory=lambda: dict[str, int](),
        )
        _exponential_factor: float = PrivateAttr(default=2.0)
        _max_delay: float = PrivateAttr(default=c.DEFAULT_MAX_DELAY_SECONDS)

        def __init__(self, max_attempts: int, retry_delay: float) -> None:
            """Initialize retry policy manager.

            Args:
                max_attempts: Maximum retry attempts allowed
                retry_delay: Base delay in seconds between retry attempts

            """
            super().__init__(max_attempts=max_attempts, retry_delay=retry_delay)

        @override
        def model_post_init(self, __context: dict[str, t.Scalar] | None, /) -> None:
            self.max_attempts = max(self.max_attempts, c.RETRY_COUNT_MIN)
            self.retry_delay = max(self.retry_delay, c.INITIAL_TIME)

        @staticmethod
        def is_retriable_error(error: str | None) -> bool:
            """Check if an error is retriable."""
            if error is None:
                return False
            retriable_patterns = (
                "Temporary failure",
                "timeout",
                "transient",
                "temporarily unavailable",
                "try again",
            )
            return any(
                pattern.lower() in error.lower() for pattern in retriable_patterns
            )

        def cleanup(self) -> None:
            """Clear all attempt tracking."""
            self._attempts.clear()

        def get_exponential_delay(self, attempt_number: int) -> float:
            """Calculate exponential backoff delay for given attempt."""
            if self.retry_delay <= 0.0:
                return 0.0
            exponential_delay = (
                self.retry_delay * self._exponential_factor**attempt_number
            )
            return min(exponential_delay, self._max_delay)

        def get_max_attempts(self) -> int:
            """Get maximum retry attempts."""
            return self.max_attempts

        def get_retry_delay(self) -> float:
            """Get base delay between retry attempts."""
            return self.retry_delay

        def record_attempt(self, message_type: str) -> None:
            """Record an attempt for tracking purposes."""
            self._attempts[message_type] = self._attempts.get(message_type, 0) + 1

        def reset(self, message_type: str) -> None:
            """Reset attempt tracking for a message type."""
            _ = self._attempts.pop(message_type, None)

        def should_retry(self, current_attempt: int) -> bool:
            """Check if we should retry the operation."""
            return current_attempt < self.max_attempts - 1


__all__ = ["FlextModelsDispatcher"]
