"""Dispatcher reliability smoke tests based on result semantics."""

from __future__ import annotations

from tests import p, r


class TestsFlextDispatcherReliability:
    def test_reliability_chain_stops_on_failure(self) -> None:
        def fail_step(_: int) -> p.Result[int]:
            return r[int].fail("stop")

        result = r[int].ok(1).flow_through(fail_step)
        assert result.failure
        assert result.error == "stop"

    def test_reliability_chain_keeps_success(self) -> None:
        def ok_step(value: int) -> p.Result[int]:
            return r[int].ok(value + 1)

        result = r[int].ok(1).flow_through(ok_step)
        assert result.success
        assert result.value == 2
