"""Tests for Dispatcher Reliability full coverage."""

from __future__ import annotations

from flext_core import c, m, r, u
from flext_core._dispatcher import reliability as disp_rel


def test_dispatcher_reliability_branch_paths() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(m.ConfigMap({"k": 1}), m.ConfigMap)
    assert u.to_str(1) == "1"
    cb = disp_rel.CircuitBreakerManager(
        threshold=3,
        recovery_timeout=1.0,
        success_threshold=2,
    )
    cb.transition_to_half_open("x")
    cb.record_failure("x")
    assert cb.get_state("x") == c.Reliability.CircuitBreakerState.OPEN
    rl = disp_rel.RateLimiterManager(max_requests=1, window_seconds=1.5)
    assert rl.get_max_requests() == 1
    assert abs(rl.get_window_seconds() - 1.5) < 1e-9
    rp = disp_rel.RetryPolicy(max_attempts=1, retry_delay=0.0)
    assert abs(rp.get_exponential_delay(1) - 0.0) < 1e-9
