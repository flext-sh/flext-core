"""Comprehensive coverage tests for strict FlextDispatcher implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest
from pydantic import Field

from flext_core import FlextDispatcher, FlextModelsCqrs, r
from tests import m, p, t


class TestDispatcherFullCoverage:
    class _SampleCommand(m.Command):
        """Refined sample command."""

        command_type: str = "sample_command"
        payload: str

    class _SampleQuery(m.Query):
        """Refined sample query."""

        pagination: FlextModelsCqrs.Pagination | t.Dict = Field(default_factory=t.Dict)
        query_type: str | None = "sample_query"

    class _SampleEvent(m.Event):
        """Refined sample event."""

        event_type: str = "sample_event"

    class _UnregisteredCommand(m.Command):
        """Command that won't be registered."""

        command_type: str = "unregistered"

    class _SampleHandler:
        """Strict handler implementation matching Handle."""

        message_type = "sample_command"

        def handle(self, message: p.Routable) -> r[str]:
            payload = getattr(message, "payload", "")
            return r[str].ok(f"handled:{payload}")

        def can_handle(self, message_type: type) -> bool:
            return message_type is TestDispatcherFullCoverage._SampleCommand

    class _QueryHandler:
        """Query handler matching Handle."""

        message_type = "sample_query"

        def handle(
            self,
            message: TestDispatcherFullCoverage._SampleQuery,
        ) -> r[t.ConfigMap]:
            query_id = getattr(message, "query_id", None)
            return r[t.ConfigMap].ok(
                t.ConfigMap(root={"result": "data", "id": str(query_id)}),
            )

        def can_handle(self, message_type: type) -> bool:
            return message_type is TestDispatcherFullCoverage._SampleQuery

    class _EventHandler:
        """Event handler matching Handle."""

        message_type = "sample_event"

        def handle(self, message: p.Routable) -> r[bool]:
            _ = message
            return r[bool].ok(True)

        def can_handle(
            self,
            message_type: type[TestDispatcherFullCoverage._SampleEvent],
        ) -> bool:
            return message_type is TestDispatcherFullCoverage._SampleEvent

    class _PredicateHandler:
        def can_handle(self, msg_type: type) -> bool:
            return msg_type is TestDispatcherFullCoverage._SampleCommand

        def handle(self, message: p.Routable) -> r[str]:
            _ = message
            return r[str].ok("ok")

    @staticmethod
    def _force_handler(
        obj: Callable[[m.Command], str] | str | t.MutableStrMapping,
    ) -> t.DispatchableHandler:
        _ = obj

        def _wrapper(
            *_args: t.Container,
            **_kwargs: t.Scalar,
        ) -> p.Result[t.Container] | t.Container | None:
            return "invalid"

        return cast("t.DispatchableHandler", _wrapper)

    @staticmethod
    def _force_routable(obj: str | None) -> p.Routable:
        _ = obj

        class _FakeRoutable:
            command_type: str | None = None
            query_type: str | None = None
            event_type: str | None = None

        return _FakeRoutable()

    @pytest.fixture
    def dispatcher(self) -> p.Dispatcher:
        return FlextDispatcher()

    def test_strict_registration_and_dispatch(
        self,
        dispatcher: p.Dispatcher,
    ) -> None:
        handler = self._SampleHandler()
        res = dispatcher.register_handler(handler)
        assert res.success
        cmd = self._SampleCommand(payload="hello", command_id="cmd-1")
        dispatch_res = dispatcher.dispatch(cmd)
        assert dispatch_res.success
        assert dispatch_res.value == "handled:hello"
        unreg_cmd = self._UnregisteredCommand(command_id="cmd-2")
        fail_res = dispatcher.dispatch(unreg_cmd)
        assert fail_res.failure
        assert (
            fail_res.error is not None and "no handler found" in fail_res.error.lower()
        )

    def test_invalid_registration_attempts(self, dispatcher: p.Dispatcher) -> None:
        assert dispatcher.register_handler(
            self._force_handler("not-a-handler"),
        ).failure
        assert dispatcher.register_handler(
            self._force_handler({"some": "dict"}),
        ).failure

        def nameless_handler(msg: m.Command) -> str:
            _ = msg
            return "ok"

        assert dispatcher.register_handler(
            self._force_handler(nameless_handler),
        ).failure

    def test_event_publishing_strict(self, dispatcher: p.Dispatcher) -> None:
        handler = self._EventHandler()
        registration = dispatcher.register_handler(handler, is_event=True)
        assert registration.success
        evt = self._SampleEvent(
            aggregate_id="agg-1",
            event_type="sample_event",
            event_id="evt-1",
            data=t.Dict(root={}),
            metadata=t.Dict(root={}),
        )
        res = dispatcher.publish(evt)
        assert res.success
        batch_res = dispatcher.publish([evt, evt])
        assert batch_res.success

    def test_handler_attribute_discovery(self, dispatcher: p.Dispatcher) -> None:
        res = dispatcher.register_handler(self._PredicateHandler())
        assert res.success
        assert dispatcher.dispatch(
            self._SampleCommand(payload="p", command_id="cmd-p"),
        ).success

    def test_query_handler_can_return_config_map(
        self,
        dispatcher: p.Dispatcher,
    ) -> None:
        """Dispatch should preserve BaseModel payloads allowed by the protocol."""
        registration = dispatcher.register_handler(self._QueryHandler())
        assert registration.success
        query = self._SampleQuery(query_id="query-1")
        result = dispatcher.dispatch(query)
        assert result.success
        assert isinstance(result.value, t.ConfigMap)
        assert result.value.root["result"] == "data"

    def test_callable_registration_with_attribute(
        self,
        dispatcher: p.Dispatcher,
    ) -> None:
        def func_handler(msg: TestDispatcherFullCoverage._SampleCommand) -> str:
            _ = msg
            return "func:ok"

        setattr(func_handler, "message_type", self._SampleCommand)
        res = dispatcher.register_handler(func_handler)
        assert res.success
        assert (
            dispatcher.dispatch(
                self._SampleCommand(payload="x", command_id="cmd-x"),
            ).value
            == "func:ok"
        )

    def test_dispatch_invalid_input_types(self, dispatcher: p.Dispatcher) -> None:
        assert dispatcher.dispatch(self._force_routable(None)).failure
        assert dispatcher.dispatch(self._force_routable("not-a-model")).failure

    def test_exception_handling_in_dispatch(self, dispatcher: p.Dispatcher) -> None:
        def breaking_handler(msg: TestDispatcherFullCoverage._SampleCommand) -> str:
            _ = msg
            error_msg = "broken"
            raise ValueError(error_msg)

        setattr(breaking_handler, "message_type", self._SampleCommand)
        dispatcher.register_handler(breaking_handler)
        res = dispatcher.dispatch(
            self._SampleCommand(payload="x", command_id="cmd-break"),
        )
        assert res.failure
        assert isinstance(res.error, str)
        assert "broken" in res.error
