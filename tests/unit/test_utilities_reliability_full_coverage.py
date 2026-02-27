"""Coverage tests for current utilities reliability APIs."""

from __future__ import annotations

from typing import Never

import flext_core._utilities.reliability as reliability_module
import pytest
from flext_core import c, m, r, t, u


def test_utilities_reliability_branches() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"k": 1}), t.ConfigMap)

    fail: r[Never] = u.Reliability.retry(
        lambda: r.fail("e"),
        max_attempts=1,
        delay_seconds=0.0,
    )
    assert fail.is_failure

    delay_default = u.Reliability.calculate_delay(0, None)
    assert isinstance(delay_default, float)

    with_cleanup = u.Reliability.with_retry(
        lambda: (_ for _ in ()).throw(ValueError("x")),
        max_attempts=2,
        cleanup_func=lambda: None,
    )
    assert with_cleanup.is_failure

    assert u.Reliability.pipe("x").is_success
    assert callable(u.Reliability.compose(lambda x: x, mode="pipe"))


def test_utilities_reliability_uncovered_retry_compose_and_sequence_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "flext_core._utilities.reliability.time.sleep",
        lambda seconds: sleep_calls.append(seconds),
    )

    def _raise_once() -> r[Never]:
        msg = "boom"
        raise ValueError(msg)

    failed: r[Never] = u.Reliability.retry(
        _raise_once,
        max_attempts=2,
        delay_seconds=0.01,
        retry_on=(ValueError,),
    )
    assert failed.is_failure
    assert len(sleep_calls) == 1

    exhausted = u.Reliability.with_retry(lambda: r[int].ok(1), max_attempts=0)
    assert exhausted.is_failure


def test_utilities_reliability_compose_returns_non_result_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    monkeypatch.setattr(
        reliability_module.FlextUtilitiesReliability,
        "pipe",
        staticmethod(lambda *_args, **_kwargs: r[int].ok(7)),
    )
    piped = reliability_module.FlextUtilitiesReliability.compose(
        lambda value: value,
        mode="pipe",
    )
    assert piped("x") == 7
