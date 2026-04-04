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
from typing import Annotated, override

from pydantic import Field, PrivateAttr

from flext_core import FlextModelFoundation, c, r, t


class FlextModelsDispatcher:
    """Dispatcher infrastructure models."""

    class TimeoutEnforcer(FlextModelFoundation.ArbitraryTypesModel):
        """Manage timeout enforcement and dispatcher thread-pool execution."""

        use_timeout_executor: Annotated[
            bool,
            Field(description="Whether timeout executor is enabled"),
        ]
        executor_workers: Annotated[
            t.PositiveInt,
            Field(
                description="Number of worker threads for timeout executor",
            ),
        ]
        _executor: concurrent.futures.ThreadPoolExecutor | None = PrivateAttr(
            default=None,
        )

        @override
        def model_post_init(self, __context: t.ScalarMapping | None, /) -> None:
            self.executor_workers = max(
                self.executor_workers,
                c.DEFAULT_RETRY_DELAY_SECONDS,
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

        def get_executor_status(self) -> t.ScalarMapping:
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

    class CircuitBreakerStateRecord(FlextModelFoundation.FlexibleInternalModel):
        """Per-message-type circuit breaker state."""

        state: Annotated[
            str,
            Field(
                default=c.CircuitBreakerState.CLOSED,
                description="Current circuit breaker state (CLOSED, OPEN, or HALF_OPEN).",
            ),
        ] = c.CircuitBreakerState.CLOSED
        failures: Annotated[
            t.NonNegativeInt,
            Field(
                default=0,
                description="Consecutive failure count since last reset.",
            ),
        ] = 0
        success_count: Annotated[
            t.NonNegativeInt,
            Field(
                default=0,
                description="Consecutive success count during half-open recovery.",
            ),
        ] = 0
        opened_at: Annotated[
            t.NonNegativeFloat,
            Field(
                default=0.0,
                description="Epoch timestamp when the circuit was opened.",
            ),
        ] = 0.0
        recovery_successes: Annotated[
            t.NonNegativeInt,
            Field(
                default=0,
                description="Total successful recovery transitions from half-open to closed.",
            ),
        ] = 0
        recovery_failures: Annotated[
            t.NonNegativeInt,
            Field(
                default=0,
                description="Total failed recovery attempts that re-opened the circuit.",
            ),
        ] = 0
        total_successes: Annotated[
            t.NonNegativeInt,
            Field(
                default=0,
                description="Cumulative successful dispatches tracked for metrics.",
            ),
        ] = 0

    class CircuitBreakerManager(FlextModelFoundation.ArbitraryTypesModel):
        """Manage per-message circuit breaker state for dispatcher executions.

        Handles state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED) with
        configurable thresholds and recovery timeouts to protect downstream
        handlers from cascading failures.
        """

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
        _breakers: dict[str, FlextModelsDispatcher.CircuitBreakerStateRecord] = (
            PrivateAttr(
                default_factory=lambda: dict[
                    str,
                    FlextModelsDispatcher.CircuitBreakerStateRecord,
                ](),
            )
        )

        def _get_record(
            self,
            message_type: str,
        ) -> FlextModelsDispatcher.CircuitBreakerStateRecord:
            """Get or create state record for a message type."""
            if message_type not in self._breakers:
                self._breakers[message_type] = (
                    FlextModelsDispatcher.CircuitBreakerStateRecord()
                )
            return self._breakers[message_type]

        def attempt_reset(self, message_type: str) -> None:
            """Attempt recovery if circuit is open."""
            if self.is_open(message_type):
                rec = self._get_record(message_type)
                if time.time() - rec.opened_at >= self.recovery_timeout:
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
            return r[bool].ok(True)

        def cleanup(self) -> None:
            """Clear all state."""
            self._breakers.clear()

        def get_failure_count(self, message_type: str) -> int:
            """Get current failure count.

            Returns:
                Current failure count for the message type.

            """
            return self._get_record(message_type).failures

        def get_metrics(self) -> t.ScalarMapping:
            """Collect circuit breaker metrics, including recovery statistics.

            Returns:
                Dictionary containing circuit breaker metrics.

            """
            total_recovery_attempts = sum(
                rec.recovery_successes + rec.recovery_failures
                for rec in self._breakers.values()
            )
            total_recovery_successes = sum(
                rec.recovery_successes for rec in self._breakers.values()
            )
            recovery_success_rate = (
                total_recovery_successes / total_recovery_attempts * c.HTTP_STATUS_MIN
                if total_recovery_attempts > 0
                else 0.0
            )
            total_failures = sum(rec.failures for rec in self._breakers.values())
            total_successes = sum(
                rec.total_successes for rec in self._breakers.values()
            )
            total_operations = total_failures + total_successes
            failure_rate = (
                total_failures / total_operations * c.HTTP_STATUS_MIN
                if total_operations > 0
                else 0.0
            )
            return {
                "failures": len(self._breakers),
                "states": len(self._breakers),
                "open_count": sum(
                    1
                    for rec in self._breakers.values()
                    if rec.state == c.CircuitBreakerState.OPEN
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
            return self._get_record(message_type).state

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
            rec = self._get_record(message_type)
            rec.failures += 1
            if rec.state == c.CircuitBreakerState.HALF_OPEN:
                rec.recovery_failures += 1
                self.transition_to_open(message_type)
            elif (
                rec.state == c.CircuitBreakerState.CLOSED
                and rec.failures >= self.threshold
            ):
                self.transition_to_open(message_type)

        def record_success(self, message_type: str) -> None:
            """Record successful operation and update state."""
            rec = self._get_record(message_type)
            rec.total_successes += 1
            if rec.state == c.CircuitBreakerState.HALF_OPEN:
                rec.success_count += 1
                if rec.success_count >= self.success_threshold:
                    rec.recovery_successes += 1
                    self.transition_to_closed(message_type)
            elif rec.state == c.CircuitBreakerState.CLOSED:
                rec.failures = 0

        def set_state(self, message_type: str, state: str) -> None:
            """Set state for message type."""
            self._get_record(message_type).state = state

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
            rec = self._get_record(message_type)
            rec.state = new_state
            if new_state == c.CircuitBreakerState.CLOSED:
                rec.failures = 0
                rec.success_count = 0
                rec.opened_at = 0.0
            elif new_state == c.CircuitBreakerState.OPEN:
                rec.opened_at = time.time()
                rec.success_count = 0
            elif new_state == c.CircuitBreakerState.HALF_OPEN:
                rec.success_count = 0

    class RateWindow(FlextModelFoundation.FlexibleInternalModel):
        """Per-message-type rate limit window state."""

        window_start: Annotated[
            t.NonNegativeFloat,
            Field(
                default=0.0,
                description="Epoch timestamp marking the start of the current rate window.",
            ),
        ] = 0.0
        count: Annotated[
            t.NonNegativeInt,
            Field(
                default=0,
                description="Number of requests recorded in the current rate window.",
            ),
        ] = 0

    class RateLimiterManager(FlextModelFoundation.ArbitraryTypesModel):
        """Enforce per-message rate limits with a sliding window algorithm."""

        max_requests: Annotated[
            t.PositiveInt,
            Field(description="Maximum requests allowed per window"),
        ]
        window_seconds: Annotated[
            t.PositiveFloat,
            Field(description="Time window in seconds for rate limiting"),
        ]
        jitter_factor: Annotated[
            t.DecimalFraction,
            Field(
                default=0.1,
                description="Jitter variance as fraction between 0.0 and 1.0",
            ),
        ] = 0.1
        _windows: dict[str, FlextModelsDispatcher.RateWindow] = PrivateAttr(
            default_factory=lambda: dict[str, FlextModelsDispatcher.RateWindow](),
        )

        @override
        def model_post_init(self, __context: t.ScalarMapping | None, /) -> None:
            self.jitter_factor = max(0.0, min(self.jitter_factor, 1.0))

        def check_rate_limit(self, message_type: str) -> r[bool]:
            """Return whether dispatch is allowed under the current rate window.

            Returns:
                Success result if rate limit allows dispatch, failure otherwise.

            """
            current_time = time.time()
            if message_type not in self._windows:
                self._windows[message_type] = FlextModelsDispatcher.RateWindow(
                    window_start=current_time,
                )
            window = self._windows[message_type]
            if current_time - window.window_start >= self.window_seconds:
                window.window_start = current_time
                window.count = 0
            if window.count >= self.max_requests:
                elapsed = current_time - window.window_start
                retry_after = max(0, int(self.window_seconds - elapsed))
                return r[bool].fail(
                    f"Rate limit exceeded for message type '{message_type}'",
                    error_code=c.OPERATION_ERROR,
                    error_data=t.ConfigMap(
                        root={
                            "message_type": message_type,
                            "limit": self.max_requests,
                            "window_seconds": self.window_seconds,
                            "current_count": window.count,
                            "retry_after": retry_after,
                        },
                    ),
                )
            window.count += 1
            return r[bool].ok(True)

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

    class RetryPolicy(FlextModelFoundation.ArbitraryTypesModel):
        """Coordinate retry attempts with configurable backoff for dispatcher steps."""

        max_attempts: Annotated[
            t.PositiveInt,
            Field(description="Maximum retry attempts allowed"),
        ]
        retry_delay: Annotated[
            t.PositiveFloat,
            Field(description="Base delay in seconds between retry attempts"),
        ]
        exponential_factor: Annotated[
            t.BackoffMultiplier,
            Field(
                default=2.0,
                description="Multiplier for exponential backoff between retries",
            ),
        ] = 2.0
        max_delay: Annotated[
            t.PositiveFloat,
            Field(
                default=c.DEFAULT_MAX_DELAY_SECONDS,
                description="Maximum delay cap in seconds",
            ),
        ] = c.DEFAULT_MAX_DELAY_SECONDS
        _attempts: dict[str, int] = PrivateAttr(
            default_factory=lambda: dict[str, int](),
        )

        @override
        def model_post_init(self, __context: t.ScalarMapping | None, /) -> None:
            self.max_attempts = max(self.max_attempts, c.DEFAULT_RETRY_DELAY_SECONDS)
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
                self.retry_delay * self.exponential_factor**attempt_number
            )
            return min(exponential_delay, self.max_delay)

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
