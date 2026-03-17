from __future__ import annotations

import time

from flext_tests import tb, tm, tt
from hypothesis import given, strategies as st
from pydantic import BaseModel

from flext_core import FlextHandlers, h
from tests import c, m, t


class TestAutomatedFlextHandlers:
    def test_create_from_callable(self) -> None:
        handler = FlextHandlers.create_from_callable(
            handler_callable=lambda value: str(value),
            handler_name="my_handler",
        )
        tm.that(handler.handler_name, eq="my_handler")

    def test_execute(self) -> None:
        handler = FlextHandlers.create_from_callable(
            handler_callable=lambda value: str(value),
            handler_name="exec_handler",
        )
        tm.ok(handler.execute("abc"), eq="abc")

    def test_validate_message(self) -> None:
        handler = FlextHandlers.create_from_callable(
            handler_callable=lambda value: str(value),
            handler_name="validator",
        )
        tm.ok(handler.validate_message("ok"), eq=True)

    def test_execute_failure_path(self) -> None:
        def _failing_handler(value: t.Scalar) -> t.Scalar:
            if value == "bad":
                msg = "bad payload"
                raise ValueError(msg)
            return str(value)

        handler = FlextHandlers.create_from_callable(
            handler_callable=_failing_handler,
            handler_name="failing",
        )
        tm.fail(handler.execute("bad"), has="bad payload")

    def test_can_handle(self) -> None:
        handler = FlextHandlers.create_from_callable(
            handler_callable=lambda value: str(value),
            handler_name="can_handle",
        )
        tm.that(handler.can_handle(str), is_=bool)
        tm.that(handler.can_handle(str), eq=True)

    def test_mode_property(self) -> None:
        handler = FlextHandlers.create_from_callable(
            handler_callable=lambda value: str(value),
            handler_name="mode",
            handler_type=c.Cqrs.HandlerType.COMMAND,
        )
        tm.that(handler.mode, eq=c.Cqrs.HandlerType.COMMAND)

    def test_discovery_has_handlers(self) -> None:
        def _decorated(message: BaseModel) -> BaseModel:
            return message

        class _Service:
            handle_command = staticmethod(h.handler(command=m.Command)(_decorated))

        tm.that(FlextHandlers.Discovery.has_handlers(_Service), eq=True)

    @given(st.text(min_size=1))
    def test_create_from_callable_hypothesis(self, handler_name: str) -> None:
        handler = FlextHandlers.create_from_callable(
            handler_callable=lambda value: str(value),
            handler_name=handler_name,
        )
        tm.that(handler.handler_name, eq=handler_name)
        tm.ok(handler.execute("x"), eq="x")

    def test_execute_benchmark(self) -> None:
        handler = FlextHandlers.create_from_callable(
            handler_callable=lambda value: str(value),
            handler_name="bench_handler",
        )
        tm.ok(tb.Tests.Result.ok("bench"), eq="bench")
        op = tt.op("simple")
        _ = op()
        start = time.perf_counter()
        last_success = True
        for _ in range(1000):
            result = handler.execute("payload")
            last_success = result.is_success
        elapsed = time.perf_counter() - start
        tm.that(last_success, eq=True)
        tm.that(elapsed, gte=0.0)
