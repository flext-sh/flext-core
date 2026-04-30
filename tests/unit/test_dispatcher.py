"""Dispatcher tests consolidated around public result and DSL contracts."""

from __future__ import annotations

import pytest

from tests import m, p, r, t, u


class TestsFlextDispatcher:
    @pytest.mark.parametrize(
        ("value", "expected_success", "expected_error"),
        [
            ("dispatch", True, None),
            (None, False, "failed"),
        ],
    )
    def test_minimal_result_contract(
        self,
        value: str | None,
        expected_success: bool,
        expected_error: str | None,
    ) -> None:
        result = r[str].ok(value) if value is not None else r[str].fail("failed")
        assert result.success is expected_success
        assert result.error == expected_error

    @pytest.mark.parametrize(
        ("start", "step_kind", "expected_success", "expected_value", "expected_error"),
        [
            (1, "ok", True, 2, None),
            (1, "fail", False, None, "stop"),
        ],
    )
    def test_reliability_flow_through(
        self,
        start: int,
        step_kind: str,
        expected_success: bool,
        expected_value: int | None,
        expected_error: str | None,
    ) -> None:
        def ok_step(value: int) -> p.Result[int]:
            return r[int].ok(value + 1)

        def fail_step(_: int) -> p.Result[int]:
            return r[int].fail("stop")

        step: t.DispatchableHandler
        step = ok_step if step_kind == "ok" else fail_step
        result = r[int].ok(start).flow_through(step)

        assert result.success is expected_success
        if expected_success:
            assert result.value == expected_value
            assert result.error is None
        else:
            assert result.failure
            assert result.error == expected_error

    def test_dispatcher_builder_returns_protocol_aligned_dispatcher(self) -> None:
        dispatcher = u.build_dispatcher()
        assert isinstance(dispatcher, p.Dispatcher)

        def handle(_: m.Command) -> str:
            return "handled"

        setattr(handle, "message_type", m.Command)
        assert dispatcher.register_handler(handle).success


__all__: t.MutableSequenceOf[str] = ["TestsFlextDispatcher"]
