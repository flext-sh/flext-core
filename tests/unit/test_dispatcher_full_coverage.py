"""Comprehensive coverage tests for the dispatcher DSL."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import Field

from tests import m, p, r, t, u


class TestDispatcherFullCoverage:
    class _SampleCommand(m.Command):
        """Refined sample command."""

        command_type: str = "sample_command"
        payload: str

    class _SampleQuery(m.Query):
        """Refined sample query."""

        pagination: m.Pagination | t.Dict = Field(default_factory=t.Dict)
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

        def handle(self, message: p.Routable) -> p.Result[str]:
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
        ) -> p.Result[t.RecursiveContainerMapping]:
            query_id = getattr(message, "query_id", None)
            return r[t.RecursiveContainerMapping].ok(
                {"result": "data", "id": str(query_id)},
            )

        def can_handle(self, message_type: type) -> bool:
            return message_type is TestDispatcherFullCoverage._SampleQuery

    class _EventHandler:
        """Event handler matching Handle."""

        message_type = "sample_event"

        def handle(self, message: p.Routable) -> p.Result[bool]:
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

        def handle(self, message: p.Routable) -> p.Result[str]:
            _ = message
            return r[str].ok("ok")

    class _UnroutedMessage:
        @property
        def command_type(self) -> str | None:
            return None

        @property
        def query_type(self) -> str | None:
            return None

        @property
        def event_type(self) -> str | None:
            return None

    @pytest.fixture
    def dispatcher(self) -> p.Dispatcher:
        return u.build_dispatcher()

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
        def nameless_handler(msg: m.Command) -> str:
            _ = msg
            return "ok"

        assert dispatcher.register_handler(nameless_handler).failure

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
        assert isinstance(result.value, Mapping)
        assert result.value["result"] == "data"

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
        assert dispatcher.dispatch(self._UnroutedMessage()).failure

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
