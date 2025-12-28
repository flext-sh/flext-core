"""Real reliability tests for dispatcher helpers (no mocks)."""

from __future__ import annotations
from flext_core.typings import t

import time

from flext_core import c
from flext_core._dispatcher.reliability import (
    CircuitBreakerManager,
    RateLimiterManager,
    RetryPolicy,
)


def test_circuit_breaker_transitions_and_metrics() -> None:
    """Exercise open → half-open → closed transitions with real timing."""
    message_type = "cmd"
    cb = CircuitBreakerManager(threshold=1, recovery_timeout=0.1, success_threshold=1)

    # Start closed and allow dispatch
    assert cb.check_before_dispatch(message_type).is_success

    # Trip the breaker
    cb.record_failure(message_type)
    failure = cb.check_before_dispatch(message_type)
    assert failure.is_failure
    assert failure.error_code == c.Errors.OPERATION_ERROR
    assert cb.is_open(message_type)

    # After timeout, circuit moves to HALF_OPEN and can recover
    time.sleep(0.12)
    half_open = cb.check_before_dispatch(message_type)
    assert half_open.is_success
    assert cb.get_state(message_type) == c.Reliability.CircuitBreakerState.HALF_OPEN

    cb.record_success(message_type)
    assert cb.get_state(message_type) == c.Reliability.CircuitBreakerState.CLOSED

    metrics = cb.get_metrics()
    # Type narrowing: metrics values are t.GeneralValueType, need to check for int
    failures_val = metrics.get("failures")
    total_ops_val = metrics.get("total_operations")
    assert isinstance(failures_val, int) and failures_val >= 1
    assert isinstance(total_ops_val, int) and total_ops_val >= 1
    cb.cleanup()
    assert cb.get_state(message_type) == c.Reliability.CircuitBreakerState.CLOSED


def test_rate_limiter_blocks_then_recovers() -> None:
    """Validate sliding window rate limiting without mocks."""
    limiter = RateLimiterManager(max_requests=2, window_seconds=0.2, jitter_factor=0.0)
    msg_type = "rate-limited"

    assert limiter.check_rate_limit(msg_type).is_success
    assert limiter.check_rate_limit(msg_type).is_success

    blocked = limiter.check_rate_limit(msg_type)
    assert blocked.is_failure
    assert blocked.error_code == c.Errors.OPERATION_ERROR
    assert blocked.error_data is not None
    retry_after_val = blocked.error_data.get("retry_after")
    assert isinstance(retry_after_val, (int, float)) and retry_after_val >= 0

    time.sleep(0.22)
    assert limiter.check_rate_limit(msg_type).is_success
    limiter.cleanup()
    assert limiter.check_rate_limit(msg_type).is_success


def test_rate_limiter_jitter_application() -> None:
    """Ensure jitter calculation respects bounds and zero factor short-circuit."""
    limiter = RateLimiterManager(max_requests=1, window_seconds=1.0, jitter_factor=0.5)
    jittered = limiter._apply_jitter(2.0)
    assert jittered >= 0.0

    limiter_zero = RateLimiterManager(
        max_requests=1,
        window_seconds=1.0,
        jitter_factor=0.0,
    )
    assert limiter_zero._apply_jitter(0.5) == 0.5


def test_retry_policy_behavior() -> None:
    """Cover retry policy helpers and exponential backoff."""
    policy = RetryPolicy(max_attempts=3, retry_delay=0.1)

    assert policy.should_retry(0)
    assert policy.should_retry(1)
    assert not policy.should_retry(2)

    assert policy.is_retriable_error("Temporary failure - try again later")
    assert not policy.is_retriable_error(None)

    assert policy.get_retry_delay() == 0.1
    assert policy.get_max_attempts() == 3
    assert policy.get_exponential_delay(0) == 0.1
    assert policy.get_exponential_delay(2) == min(0.1 * (2.0**2), 300.0)

    policy.record_attempt("cmd")
    policy.reset("cmd")
    policy.cleanup()
    assert policy.should_retry(0)
