"""Real reliability tests for dispatcher helpers (no mocks)."""

from __future__ import annotations

import time

from flext_core import FlextModelsDispatcher
from flext_tests import tm
from tests import c, u


def test_circuit_breaker_transitions_and_metrics() -> None:
    """Exercise open → half-open → closed transitions with real timing."""
    message_type = "cmd"
    cb = FlextModelsDispatcher.CircuitBreakerManager(
        threshold=1, recovery_timeout=0.1, success_threshold=1
    )
    tm.ok(cb.check_before_dispatch(message_type))
    cb.record_failure(message_type)
    failure = cb.check_before_dispatch(message_type)
    tm.fail(failure)
    tm.that(failure.error_code, eq=c.ErrorCode.OPERATION_ERROR.value)
    tm.that(cb.resolve_open(message_type), eq=True)
    time.sleep(0.12)
    half_open = cb.check_before_dispatch(message_type)
    tm.ok(half_open)
    tm.that(cb.resolve_state(message_type), eq=c.CircuitBreakerState.HALF_OPEN)
    cb.record_success(message_type)
    tm.that(cb.resolve_state(message_type), eq=c.CircuitBreakerState.CLOSED)
    metrics = cb.metrics
    failures_val = metrics.get("failures")
    total_ops_val = metrics.get("total_operations")
    tm.that(isinstance(failures_val, int) and failures_val >= 1, eq=True)
    tm.that(isinstance(total_ops_val, int) and total_ops_val >= 1, eq=True)
    cb.cleanup()
    tm.that(cb.resolve_state(message_type), eq=c.CircuitBreakerState.CLOSED)


def test_rate_limiter_blocks_then_recovers() -> None:
    """Validate sliding window rate limiting without mocks."""
    limiter = FlextModelsDispatcher.RateLimiterManager(
        max_requests=2, window_seconds=0.2, jitter_factor=0.0
    )
    msg_type = "rate-limited"
    tm.ok(limiter.check_rate_limit(msg_type))
    tm.ok(limiter.check_rate_limit(msg_type))
    blocked = limiter.check_rate_limit(msg_type)
    tm.fail(blocked)
    tm.that(blocked.error_code, eq=c.ErrorCode.OPERATION_ERROR.value)
    tm.that(blocked.error_data, none=False)
    assert blocked.error_data is not None
    retry_after_val = blocked.error_data.get("retry_after")
    tm.that(isinstance(retry_after_val, (int, float)) and retry_after_val >= 0, eq=True)
    time.sleep(0.22)
    tm.ok(limiter.check_rate_limit(msg_type))
    limiter.cleanup()
    tm.ok(limiter.check_rate_limit(msg_type))


def test_rate_limiter_jitter_application() -> None:
    """Ensure the canonical jitter utility respects bounds and zero short-circuit."""
    jittered = u.apply_jitter(2.0, 0.5)
    tm.that(jittered, gte=0.0)
    tm.that(abs(u.apply_jitter(0.5, 0.0) - 0.5), lt=1e-9)


def test_circuit_breaker_half_open_and_rate_limiter_public_contract() -> None:
    """Test transition_to_half_open and direct/public rate-limiter state access."""
    cb = FlextModelsDispatcher.CircuitBreakerManager(
        threshold=3,
        recovery_timeout=1.0,
        success_threshold=2,
    )
    cb.transition_to_half_open("x")
    cb.record_failure("x")
    assert cb.resolve_state("x") == c.CircuitBreakerState.OPEN
    rl = FlextModelsDispatcher.RateLimiterManager(max_requests=1, window_seconds=1.5)
    assert rl.max_requests == 1
    assert abs(rl.window_seconds - 1.5) < 1e-9
