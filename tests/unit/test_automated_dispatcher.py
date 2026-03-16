from __future__ import annotations

from time import perf_counter

from flext_tests import tb, tm, tt

from flext_core import FlextDispatcher, FlextHandlers, r
from tests import m, t


class TestAutomatedFlextDispatcher:
    def test_constructor(self) -> None:
        dispatcher = FlextDispatcher()
        tm.that(dispatcher.__class__.__name__, eq="FlextDispatcher")

    def test_register_and_dispatch(self) -> None:
        dispatcher = FlextDispatcher()
        handler = FlextHandlers.create_from_callable(
            handler_callable=lambda value: str(value),
            handler_name="stringer",
        )
        tm.ok(dispatcher.register_handler(handler), eq=True)
        command = m.Command(command_type="auto_route", command_id="cmd-1")
        tm.ok(dispatcher.dispatch(command), eq=str(command))

    def test_dispatch_unregistered_fails(self) -> None:
        dispatcher = FlextDispatcher()
        command = m.Command(command_type="missing", command_id="cmd-2")
        tm.fail(dispatcher.dispatch(command), has="No handler found")

    def test_dispatch_typed(self) -> None:
        dispatcher = FlextDispatcher()
        handler = FlextHandlers.create_from_callable(
            handler_callable=lambda value: str(value),
            handler_name="typed",
        )
        tm.ok(dispatcher.register_handler(handler), eq=True)
        command = m.Command(command_type="typed_route", command_id="cmd-3")
        tm.ok(dispatcher.dispatch_typed(command, str), has="typed_route")

    def test_publish_event(self) -> None:
        dispatcher = FlextDispatcher()
        event_handler = FlextHandlers.create_from_callable(
            handler_callable=lambda value: str(value),
            handler_name="event_handler",
        )
        tm.ok(dispatcher.register_handler(event_handler), eq=True)
        tm.ok(tb.Tests.Result.ok("payload"), eq="payload")
        event_data = t.Dict.model_validate({"item": "created"})
        event_metadata = t.Dict.model_validate({})
        event = m.Event(
            event_type="item_created",
            aggregate_id="agg-1",
            event_id="evt-1",
            data=event_data,
            metadata=event_metadata,
        )
        tm.ok(dispatcher.publish(event), eq=True)

    def test_dispatch_benchmark(self) -> None:
        dispatcher = FlextDispatcher()
        handler = FlextHandlers.create_from_callable(
            handler_callable=lambda value: str(value),
            handler_name="bench",
        )
        tm.ok(dispatcher.register_handler(handler), eq=True)
        command = m.Command(command_type="bench_route", command_id="cmd-4")
        op = tt.op("simple")
        _ = op()
        start = perf_counter()
        result = r[str].ok("init")
        for _ in range(1000):
            result = dispatcher.dispatch_typed(command, str)
        elapsed = perf_counter() - start
        tm.ok(result, has="bench_route")
        tm.that(elapsed, gte=0.0)
