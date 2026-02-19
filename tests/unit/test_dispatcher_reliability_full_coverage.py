from __future__ import annotations

from flext_core import c, m, r, t, u


disp_rel = __import__(
    "flext_core._dispatcher.reliability",
    fromlist=["CircuitBreakerManager", "RateLimiterManager", "RetryPolicy"],
)


def test_dispatcher_reliability_branch_paths() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"k": 1}), t.ConfigMap)
    assert u.Conversion.to_str(1) == "1"

    cb = disp_rel.CircuitBreakerManager(
        threshold=3, recovery_timeout=1.0, success_threshold=2
    )
    cb.transition_to_half_open("x")
    cb.record_failure("x")
    assert cb.get_state("x") == c.Reliability.CircuitBreakerState.OPEN

    rl = disp_rel.RateLimiterManager(max_requests=1, window_seconds=1.5)
    assert rl.get_max_requests() == 1
    assert rl.get_window_seconds() == 1.5

    rp = disp_rel.RetryPolicy(max_attempts=1, retry_delay=0.0)
    assert rp.get_exponential_delay(1) == 0.0
