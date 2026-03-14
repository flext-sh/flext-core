"""Coverage tests for current utilities reliability APIs."""

from __future__ import annotations

from typing import Never

import pytest

import flext_core._utilities.reliability as reliability_module
from flext_core import c, m, r, t, u


def test_utilities_reliability_branches() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(m.ConfigMap({"k": 1}), m.ConfigMap)

    def _always_fail() -> r[t.Container]:
        return r[t.Container].fail("e")

    fail: r[t.Container] = u.retry(
        _always_fail,
        max_attempts=1,
        delay_seconds=0.0,
    )
    assert fail.is_failure
    delay_default = u.calculate_delay(0, None)
    assert isinstance(delay_default, float)
    assert u.with_retry(
        lambda: (_ for _ in ()).throw(ValueError("x")),
        max_attempts=2,
        cleanup_func=lambda: None,
    ).is_failure
    assert u.pipe("x").is_success
    assert callable(u.compose(lambda x: x, mode="pipe"))


def test_utilities_reliability_uncovered_retry_compose_and_sequence_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_calls: list[float] = []

    def _record_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(
        reliability_module.time,
        "sleep",
        _record_sleep,
    )

    def _raise_once() -> r[Never]:
        msg = "boom"
        raise ValueError(msg)

    failed: r[Never] = u.retry(
        _raise_once,
        max_attempts=2,
        delay_seconds=0.01,
        retry_on=(ValueError,),
    )
    assert failed.is_failure
    assert len(sleep_calls) == 1
    exhausted = u.with_retry(lambda: r[int].ok(1), max_attempts=0)
    assert exhausted.is_failure


def test_utilities_reliability_compose_returns_non_result_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    def _always_ok(*_args, **_kwargs: t.Scalar) -> r[int]:
        return r[int].ok(7)

    monkeypatch.setattr(
        reliability_module.FlextUtilitiesReliability,
        "pipe",
        staticmethod(_always_ok),
    )
    piped = reliability_module.FlextUtilitiesReliability.compose(
        lambda value: value,
        mode="pipe",
    )
    assert piped("x") == 7
