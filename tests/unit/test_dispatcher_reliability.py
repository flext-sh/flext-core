"""Real reliability tests for dispatcher helpers (no mocks)."""

from __future__ import annotations

import time

from flext_tests import tm

from flext_core import FlextModelsDispatcher
from tests import c

CircuitBreakerManager = FlextModelsDispatcher.CircuitBreakerManager
RateLimiterManager = FlextModelsDispatcher.RateLimiterManager
RetryPolicy = FlextModelsDispatcher.RetryPolicy


def test_circuit_breaker_transitions_and_metrics() -> None:
    """Exercise open → half-open → closed transitions with real timing."""
    message_type = "cmd"
    cb = CircuitBreakerManager(threshold=1, recovery_timeout=0.1, success_threshold=1)
    tm.ok(cb.check_before_dispatch(message_type))
    cb.record_failure(message_type)
    failure = cb.check_before_dispatch(message_type)
    tm.fail(failure)
    tm.that(failure.error_code, eq=c.OPERATION_ERROR)
    tm.that(cb.is_open(message_type), eq=True)
    time.sleep(0.12)
    half_open = cb.check_before_dispatch(message_type)
    tm.ok(half_open)
    tm.that(cb.get_state(message_type), eq=c.CircuitBreakerState.HALF_OPEN)
    cb.record_success(message_type)
    tm.that(cb.get_state(message_type), eq=c.CircuitBreakerState.CLOSED)
    metrics = cb.get_metrics()
    failures_val = metrics.get("failures")
    total_ops_val = metrics.get("total_operations")
    tm.that(isinstance(failures_val, int) and failures_val >= 1, eq=True)
    tm.that(isinstance(total_ops_val, int) and total_ops_val >= 1, eq=True)
    cb.cleanup()
    tm.that(cb.get_state(message_type), eq=c.CircuitBreakerState.CLOSED)


def test_rate_limiter_blocks_then_recovers() -> None:
    """Validate sliding window rate limiting without mocks."""
    limiter = RateLimiterManager(max_requests=2, window_seconds=0.2, jitter_factor=0.0)
    msg_type = "rate-limited"
    tm.ok(limiter.check_rate_limit(msg_type))
    tm.ok(limiter.check_rate_limit(msg_type))
    blocked = limiter.check_rate_limit(msg_type)
    tm.fail(blocked)
    tm.that(blocked.error_code, eq=c.OPERATION_ERROR)
    tm.that(blocked.error_data is not None, eq=True)
    assert blocked.error_data is not None
    retry_after_val = blocked.error_data.get("retry_after")
    tm.that(isinstance(retry_after_val, (int, float)) and retry_after_val >= 0, eq=True)
    time.sleep(0.22)
    tm.ok(limiter.check_rate_limit(msg_type))
    limiter.cleanup()
    tm.ok(limiter.check_rate_limit(msg_type))


def test_rate_limiter_jitter_application() -> None:
    """Ensure jitter calculation respects bounds and zero factor short-circuit."""
    limiter = RateLimiterManager(max_requests=1, window_seconds=1.0, jitter_factor=0.5)
    jittered = limiter._apply_jitter(2.0)
    tm.that(jittered >= 0.0, eq=True)
    limiter_zero = RateLimiterManager(
        max_requests=1,
        window_seconds=1.0,
        jitter_factor=0.0,
    )
    tm.that(abs(limiter_zero._apply_jitter(0.5) - 0.5) < 1e-9, eq=True)


def test_retry_policy_behavior() -> None:
    """Cover retry policy helpers and exponential backoff."""
    policy = RetryPolicy(max_attempts=3, retry_delay=0.1)
    tm.that(policy.should_retry(0), eq=True)
    tm.that(policy.should_retry(1), eq=True)
    tm.that(policy.should_retry(2), eq=False)
    tm.that(policy.is_retriable_error("Temporary failure - try again later"), eq=True)
    tm.that(policy.is_retriable_error(None), eq=False)
    tm.that(abs(policy.get_retry_delay() - 0.1) < 1e-9, eq=True)
    tm.that(policy.get_max_attempts(), eq=3)
    tm.that(abs(policy.get_exponential_delay(0) - 0.1) < 1e-9, eq=True)
    expected_delay = min(0.1 * 2.0**2, 300.0)
    tm.that(abs(policy.get_exponential_delay(2) - expected_delay) < 1e-9, eq=True)
    policy.record_attempt("cmd")
    policy.reset("cmd")
    policy.cleanup()
    tm.that(policy.should_retry(0), eq=True)
