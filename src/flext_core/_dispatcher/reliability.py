"""Reliability helpers for ``FlextDispatcher``.

Provide reusable circuit breaking, rate limiting, and retry primitives used by
the dispatcher pipeline to protect CQRS handlers. Splitting these helpers into
their own module keeps orchestration readable while preserving the same
runtime behavior and typed surface exposed by the dispatcher.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import secrets
import time

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import t


class CircuitBreakerManager:
    """Manage per-message circuit breaker state for dispatcher executions.

    Handles state transitions (CLOSED → OPEN → HALF_OPEN → CLOSED) with
    configurable thresholds and recovery timeouts to protect downstream
    handlers from cascading failures.
    """

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
        super().__init__()
        self._failures: dict[str, int] = {}
        self._states: dict[str, str] = {}
        self._opened_at: dict[str, float] = {}
        self._success_counts: dict[str, int] = {}
        self._threshold = threshold
        self._recovery_timeout = recovery_timeout
        self._success_threshold = success_threshold
        self._recovery_successes: dict[str, int] = {}
        self._recovery_failures: dict[str, int] = {}
        self._total_successes: dict[str, int] = {}

    def get_state(self, message_type: str) -> str:
        """Get current state for message type."""
        return self._states.get(
            message_type,
            FlextConstants.Reliability.CircuitBreakerState.CLOSED,
        )

    def set_state(self, message_type: str, state: str) -> None:
        """Set state for message type."""
        self._states[message_type] = state

    def is_open(self, message_type: str) -> bool:
        """Check if circuit breaker is open for message type."""
        return (
            self.get_state(message_type)
            == FlextConstants.Reliability.CircuitBreakerState.OPEN
        )

    def record_success(self, message_type: str) -> None:
        """Record successful operation and update state."""
        current_state = self.get_state(message_type)
        self._total_successes[message_type] = (
            self._total_successes.get(message_type, 0) + 1
        )

        if current_state == FlextConstants.Reliability.CircuitBreakerState.HALF_OPEN:
            success_count = self._success_counts.get(message_type, 0) + 1
            self._success_counts[message_type] = success_count

            if success_count >= self._success_threshold:
                self._recovery_successes[message_type] = (
                    self._recovery_successes.get(message_type, 0) + 1
                )
                self.transition_to_closed(message_type)

        elif current_state == FlextConstants.Reliability.CircuitBreakerState.CLOSED:
            self._failures[message_type] = 0

    def record_failure(self, message_type: str) -> None:
        """Record failed operation and update state."""
        current_state = self.get_state(message_type)
        current_failures = self._failures.get(message_type, 0) + 1
        self._failures[message_type] = current_failures

        if current_state == FlextConstants.Reliability.CircuitBreakerState.HALF_OPEN:
            self._recovery_failures[message_type] = (
                self._recovery_failures.get(message_type, 0) + 1
            )
            self.transition_to_open(message_type)
        elif (
            current_state == FlextConstants.Reliability.CircuitBreakerState.CLOSED
            and current_failures >= self._threshold
        ):
            self.transition_to_open(message_type)

    def transition_to_state(self, message_type: str, new_state: str) -> None:
        """Transition to specified state."""
        self.set_state(message_type, new_state)
        if new_state == FlextConstants.Reliability.CircuitBreakerState.CLOSED:
            self._failures[message_type] = 0
            self._success_counts[message_type] = 0
            if message_type in self._opened_at:
                del self._opened_at[message_type]
        elif new_state == FlextConstants.Reliability.CircuitBreakerState.OPEN:
            self._opened_at[message_type] = time.time()
            self._success_counts[message_type] = 0
        elif new_state == FlextConstants.Reliability.CircuitBreakerState.HALF_OPEN:
            self._success_counts[message_type] = 0

    def transition_to_closed(self, message_type: str) -> None:
        """Transition to CLOSED state."""
        self.transition_to_state(
            message_type,
            FlextConstants.Reliability.CircuitBreakerState.CLOSED,
        )

    def transition_to_open(self, message_type: str) -> None:
        """Transition to OPEN state."""
        self.transition_to_state(
            message_type,
            FlextConstants.Reliability.CircuitBreakerState.OPEN,
        )

    def transition_to_half_open(self, message_type: str) -> None:
        """Transition to HALF_OPEN state."""
        self.transition_to_state(
            message_type,
            FlextConstants.Reliability.CircuitBreakerState.HALF_OPEN,
        )

    def attempt_reset(self, message_type: str) -> None:
        """Attempt recovery if circuit is open."""
        if self.is_open(message_type):
            opened_at = self._opened_at.get(message_type, 0.0)
            if (time.time() - opened_at) >= self._recovery_timeout:
                self.transition_to_half_open(message_type)

    def check_before_dispatch(self, message_type: str) -> FlextResult[bool]:
        """Return a result indicating whether dispatch can proceed.

        Returns:
            FlextResult[bool]: Success with ``True`` when the circuit is closed
                or half-open; failure with metadata when the circuit remains
                open.

        """
        self.attempt_reset(message_type)
        if self.is_open(message_type):
            return FlextResult[bool].fail(
                f"Circuit breaker is open for message type '{message_type}'",
                error_code=FlextConstants.Errors.OPERATION_ERROR,
                error_data={
                    "message_type": message_type,
                    "state": self.get_state(message_type),
                    "failure_count": self.get_failure_count(message_type),
                },
            )
        return FlextResult[bool].ok(True)

    def get_failure_count(self, message_type: str) -> int:
        """Get current failure count."""
        return self._failures.get(message_type, 0)

    def get_threshold(self) -> int:
        """Get circuit breaker threshold."""
        return self._threshold

    def cleanup(self) -> None:
        """Clear all state."""
        self._failures.clear()
        self._states.clear()
        self._opened_at.clear()
        self._success_counts.clear()
        self._recovery_successes.clear()
        self._recovery_failures.clear()
        self._total_successes.clear()

    def get_metrics(self) -> dict[str, t.GeneralValueType]:
        """Collect circuit breaker metrics, including recovery statistics."""
        total_recovery_attempts = sum(
            self._recovery_successes.get(mt, 0) + self._recovery_failures.get(mt, 0)
            for mt in self._states
        )
        total_recovery_successes = sum(
            self._recovery_successes.get(mt, 0) for mt in self._states
        )
        recovery_success_rate = (
            (total_recovery_successes / total_recovery_attempts * 100)
            if total_recovery_attempts > 0
            else 0.0
        )

        total_failures = sum(self._failures.values())
        total_successes = sum(self._total_successes.values())
        total_operations = total_failures + total_successes
        failure_rate = (
            (total_failures / total_operations * 100) if total_operations > 0 else 0.0
        )

        return {
            "failures": len(self._failures),
            "states": len(self._states),
            "open_count": sum(
                1
                for state in self._states.values()
                if state == FlextConstants.Reliability.CircuitBreakerState.OPEN
            ),
            "recovery_success_rate": recovery_success_rate,
            "failure_rate": failure_rate,
            "total_recovery_attempts": total_recovery_attempts,
            "total_recovery_successes": total_recovery_successes,
            "total_operations": total_operations,
        }


class RateLimiterManager:
    """Enforce per-message rate limits with a sliding window algorithm."""

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
        super().__init__()
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._jitter_factor = max(0.0, min(jitter_factor, 1.0))
        self._windows: dict[str, tuple[float, int]] = {}

    def _apply_jitter(self, base_delay: float) -> float:
        """Apply jitter variance to a delay value."""
        if base_delay <= 0.0 or self._jitter_factor == 0.0:
            return base_delay

        secure_random = secrets.SystemRandom()
        variance = (2.0 * secure_random.random() - 1.0) * self._jitter_factor
        jittered = base_delay * (1.0 + variance)
        return max(0.0, jittered)

    def check_rate_limit(self, message_type: str) -> FlextResult[bool]:
        """Return whether dispatch is allowed under the current rate window."""
        current_time = time.time()
        window_start, count = self._windows.get(message_type, (current_time, 0))

        if current_time - window_start >= self._window_seconds:
            window_start = current_time
            count = 0

        if count >= self._max_requests:
            elapsed = current_time - window_start
            retry_after = max(0, int(self._window_seconds - elapsed))
            return FlextResult[bool].fail(
                f"Rate limit exceeded for message type '{message_type}'",
                error_code=FlextConstants.Errors.OPERATION_ERROR,
                error_data={
                    "message_type": message_type,
                    "limit": self._max_requests,
                    "window_seconds": self._window_seconds,
                    "current_count": count,
                    "retry_after": retry_after,
                },
            )

        self._windows[message_type] = (window_start, count + 1)
        return FlextResult[bool].ok(True)

    def get_max_requests(self) -> int:
        """Get maximum requests per window."""
        return self._max_requests

    def get_window_seconds(self) -> float:
        """Get rate limit window duration in seconds."""
        return self._window_seconds

    def cleanup(self) -> None:
        """Clear all rate limit windows."""
        self._windows.clear()


class RetryPolicy:
    """Coordinate retry attempts with configurable backoff for dispatcher steps."""

    def __init__(self, max_attempts: int, retry_delay: float) -> None:
        """Initialize retry policy manager.

        Args:
            max_attempts: Maximum retry attempts allowed
            retry_delay: Base delay in seconds between retry attempts

        """
        super().__init__()
        self._max_attempts = max(max_attempts, 1)
        self._base_delay = max(retry_delay, 0.0)
        self._attempts: dict[str, int] = {}
        self._exponential_factor = 2.0
        self._max_delay = 300.0

    def should_retry(self, current_attempt: int) -> bool:
        """Check if we should retry the operation."""
        return current_attempt < self._max_attempts - 1

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
        return any(pattern.lower() in error.lower() for pattern in retriable_patterns)

    def get_exponential_delay(self, attempt_number: int) -> float:
        """Calculate exponential backoff delay for given attempt."""
        if self._base_delay == 0.0:
            return 0.0

        exponential_delay = self._base_delay * (
            self._exponential_factor**attempt_number
        )
        return min(exponential_delay, self._max_delay)

    def get_retry_delay(self) -> float:
        """Get base delay between retry attempts."""
        return self._base_delay

    def get_max_attempts(self) -> int:
        """Get maximum retry attempts."""
        return self._max_attempts

    def record_attempt(self, message_type: str) -> None:
        """Record an attempt for tracking purposes."""
        self._attempts[message_type] = self._attempts.get(message_type, 0) + 1

    def reset(self, message_type: str) -> None:
        """Reset attempt tracking for a message type."""
        _ = self._attempts.pop(message_type, None)

    def cleanup(self) -> None:
        """Clear all attempt tracking."""
        self._attempts.clear()


__all__ = ["CircuitBreakerManager", "RateLimiterManager", "RetryPolicy"]
