"""Comprehensive coverage tests for strict FlextDispatcher implementation."""

from __future__ import annotations

from typing import cast, override

import pytest
from flext_core import m, p, r, t
from flext_core.dispatcher import FlextDispatcher

# --- Test Models ---


class SampleCommand(m.Command):
    """Refined sample command."""

    command_type: str = "sample_command"
    payload: str


class SampleQuery(m.Query):
    """Refined sample query."""

    query_type: str | None = "sample_query"


class SampleEvent(m.Event):
    """Refined sample event."""

    event_type: str = "sample_event"


class UnregisteredCommand(m.Command):
    """Command that won't be registered."""

    command_type: str = "unregistered"


# --- Test Handlers ---


class SampleHandler(p.Handler[SampleCommand, str]):
    """Strict handler implementation."""

    message_type = SampleCommand

    @override
    def _protocol_name(self) -> str:
        return "sample-handler"

    @override
    def handle(self, message: SampleCommand) -> p.Result[str]:
        return cast("p.Result[str]", r[str].ok(f"handled:{message.payload}"))

    @override
    def can_handle(self, message_type: type) -> bool:
        return message_type is SampleCommand


class QueryHandler(p.Handler[SampleQuery, dict[str, t.GeneralValueType]]):
    """Query handler."""

    message_type = SampleQuery

    @override
    def _protocol_name(self) -> str:
        return "query-handler"

    @override
    def handle(self, message: SampleQuery) -> p.Result[dict[str, t.GeneralValueType]]:
        return cast(
            "p.Result[dict[str, t.GeneralValueType]]",
            r[dict[str, t.GeneralValueType]].ok({
                "result": "data",
                "id": message.query_id,
            }),
        )

    @override
    def can_handle(self, message_type: type) -> bool:
        return message_type is SampleQuery


class EventHandler(p.Handler[SampleEvent, bool]):
    """Event handler."""

    message_type = SampleEvent

    @override
    def _protocol_name(self) -> str:
        return "event-handler"

    @override
    def handle(self, message: SampleEvent) -> p.Result[bool]:
        return cast("p.Result[bool]", r[bool].ok(True))

    @override
    def can_handle(self, message_type: type) -> bool:
        return message_type is SampleEvent


# --- Fixtures ---


@pytest.fixture
def dispatcher() -> FlextDispatcher:
    """Fresh dispatcher instance."""
    return FlextDispatcher()


# --- Core Logic Tests ---


def test_strict_registration_and_dispatch(dispatcher: FlextDispatcher) -> None:
    """Verify that only strict model registrations and dispatches work."""
    handler = SampleHandler()

    # 1. Register with message_type attribute
    res = dispatcher.register_handler(handler)
    assert res.is_success

    # 2. Dispatch correct model
    cmd = SampleCommand(payload="hello")
    dispatch_res = dispatcher.dispatch(cmd)
    assert dispatch_res.is_success
    assert dispatch_res.value == "handled:hello"

    # 3. Dispatch unregistered model fails strictly
    unreg_cmd = UnregisteredCommand()
    fail_res = dispatcher.dispatch(unreg_cmd)
    assert fail_res.is_failure
    assert "no handler found" in fail_res.error.lower()


def test_invalid_registration_attempts(dispatcher: FlextDispatcher) -> None:
    """Verify that non-strict registrations are rejected."""
    # Attempting to register a string or dict should fail
    # We cast to t.HandlerType to simulate bypass of static types
    assert dispatcher.register_handler(
        cast("t.HandlerType", "not-a-handler")
    ).is_failure
    assert dispatcher.register_handler(
        cast("t.HandlerType", {"some": "dict"})
    ).is_failure

    # Attempting to register a function without message_type should fail
    def nameless_handler(msg: SampleCommand) -> str:
        return "ok"

    assert dispatcher.register_handler(
        cast("t.HandlerType", nameless_handler)
    ).is_failure


def test_event_publishing_strict(dispatcher: FlextDispatcher) -> None:
    """Verify that event publishing strictly uses event_type or can_handle."""
    handler = EventHandler()
    # Register as event handler
    dispatcher.register_handler(handler, is_event=True)

    evt = SampleEvent(aggregate_id="agg-1", event_type="sample_event")

    # Publish single event
    res = dispatcher.publish(evt)
    assert res.is_success

    # Publish batch
    batch_res = dispatcher.publish([evt, evt])
    assert batch_res.is_success


def test_handler_attribute_discovery(dispatcher: FlextDispatcher) -> None:
    """Verify that different self-describing attributes work."""

    # Using can_handle
    class PredicateHandler:
        def can_handle(self, msg_type: type) -> bool:
            return msg_type is SampleCommand

        def handle(self, msg: SampleCommand) -> str:
            return "ok"

    res = dispatcher.register_handler(cast("t.HandlerType", PredicateHandler()))
    assert res.is_success
    assert dispatcher.dispatch(SampleCommand(payload="p")).is_success


def test_callable_registration_with_attribute(dispatcher: FlextDispatcher) -> None:
    """Verify that self-describing functions with message_type attribute are accepted."""

    def func_handler(msg: SampleCommand) -> str:
        return "func:ok"

    # Self-describing handler via attribute (set before registration)
    func_handler.message_type = SampleCommand

    res = dispatcher.register_handler(cast("t.HandlerType", func_handler))
    assert res.is_success
    assert dispatcher.dispatch(SampleCommand(payload="x")).value == "func:ok"


def test_dispatch_invalid_input_types(dispatcher: FlextDispatcher) -> None:
    """Strictly reject non-model inputs to dispatch."""
    # The dispatcher.dispatch(message: m.Message) signature suggests m.Message
    # but runtime should be safe too.
    assert dispatcher.dispatch(cast("m.Message", None)).is_failure
    assert dispatcher.dispatch(cast("m.Message", "not-a-model")).is_failure


def test_exception_handling_in_dispatch(dispatcher: FlextDispatcher) -> None:
    """Verify that exceptions in handlers are caught and returned as failures."""

    def breaking_handler(msg: SampleCommand) -> str:
        msg = "broken"
        raise ValueError(msg)

    # Self-describing handler via attribute (set before registration)
    breaking_handler.message_type = SampleCommand
    dispatcher.register_handler(cast("t.HandlerType", breaking_handler))

    res = dispatcher.dispatch(SampleCommand(payload="x"))
    assert res.is_failure
    assert "broken" in res.error
