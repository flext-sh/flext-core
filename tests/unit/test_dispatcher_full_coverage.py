"""Comprehensive coverage tests for strict FlextDispatcher implementation."""

from __future__ import annotations

import pytest

from flext_core import FlextDispatcher, m, p, r, t
from flext_core.dispatcher import _DispatchableHandler


def _force_handler(obj) -> _DispatchableHandler:
    """Return a no-op _DispatchableHandler for error-path testing.

    For non-callable objects (str, dict), returns a wrapper with no route
    attributes so the dispatcher rejects it.
    """

    def _wrapper(
        *_args,
        **_kwargs: t.Scalar,
    ) -> p.ResultLike[t.Container] | t.Container | None:
        return "invalid"

    return _wrapper


def _force_routable(obj) -> p.Routable:
    """Create a minimal Routable-compatible object for error-path testing."""

    class _FakeRoutable:
        command_type: str | None = None
        query_type: str | None = None
        event_type: str | None = None

    return _FakeRoutable()


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


class SampleHandler:
    """Strict handler implementation matching Handle."""

    message_type = SampleCommand

    def handle(self, message: p.Routable) -> r[str]:
        """Handle the message."""
        payload = getattr(message, "payload", "")
        return r[str].ok(f"handled:{payload}")

    def can_handle(self, message_type: type) -> bool:
        return message_type is SampleCommand


class QueryHandler:
    """Query handler matching Handle."""

    message_type = SampleQuery

    def handle(self, message: SampleQuery) -> r[m.ConfigMap]:
        """Handle the query."""
        query_id = getattr(message, "query_id", None)
        return r[m.ConfigMap].ok(
            m.ConfigMap(root={"result": "data", "id": str(query_id)}),
        )

    def can_handle(self, message_type: type) -> bool:
        return message_type is SampleQuery


class EventHandler:
    """Event handler matching Handle."""

    message_type = SampleEvent

    def handle(self, message: p.Routable) -> r[bool]:
        """Handle the event."""
        _ = message
        return r[bool].ok(True)

    def can_handle(self, message_type: type[SampleEvent]) -> bool:
        return message_type is SampleEvent


@pytest.fixture
def dispatcher() -> FlextDispatcher:
    """Fresh dispatcher instance."""
    return FlextDispatcher()


def test_strict_registration_and_dispatch(dispatcher: FlextDispatcher) -> None:
    """Verify that only strict model registrations and dispatches work."""
    handler = SampleHandler()
    res = dispatcher.register_handler(handler)
    assert res.is_success
    cmd = SampleCommand(payload="hello", command_id="cmd-1")
    dispatch_res = dispatcher.dispatch(cmd)
    assert dispatch_res.is_success
    assert dispatch_res.value == "handled:hello"
    unreg_cmd = UnregisteredCommand(command_id="cmd-2")
    fail_res = dispatcher.dispatch(unreg_cmd)
    assert fail_res.is_failure
    assert fail_res.error is not None and "no handler found" in fail_res.error.lower()


def test_invalid_registration_attempts(dispatcher: FlextDispatcher) -> None:
    """Verify that non-strict registrations are rejected."""
    assert dispatcher.register_handler(_force_handler("not-a-handler")).is_failure
    assert dispatcher.register_handler(_force_handler({"some": "dict"})).is_failure

    def nameless_handler(msg: SampleCommand) -> str:
        return "ok"

    assert dispatcher.register_handler(_force_handler(nameless_handler)).is_failure


def test_event_publishing_strict(dispatcher: FlextDispatcher) -> None:
    """Verify that event publishing strictly uses event_type or can_handle."""
    handler = EventHandler()
    registration = dispatcher.register_handler(handler, is_event=True)
    assert registration.is_success
    evt = SampleEvent(
        aggregate_id="agg-1",
        event_type="sample_event",
        event_id="evt-1",
        data=m.Dict(root={}),
        metadata=m.Dict(root={}),
    )
    res = dispatcher.publish(evt)
    assert res.is_success
    batch_res = dispatcher.publish([evt, evt])
    assert batch_res.is_success


def test_handler_attribute_discovery(dispatcher: FlextDispatcher) -> None:
    """Verify that different self-describing attributes work."""

    class PredicateHandler:
        def can_handle(self, msg_type: type) -> bool:
            return msg_type is SampleCommand

        def handle(self, message: p.Routable) -> r[str]:
            _ = message
            return r[str].ok("ok")

    res = dispatcher.register_handler(PredicateHandler())
    assert res.is_success
    assert dispatcher.dispatch(
        SampleCommand(payload="p", command_id="cmd-p")
    ).is_success


def test_callable_registration_with_attribute(dispatcher: FlextDispatcher) -> None:
    """Verify that self-describing functions with message_type attribute are accepted."""

    def func_handler(msg: SampleCommand) -> str:
        return "func:ok"

    setattr(func_handler, "message_type", SampleCommand)
    res = dispatcher.register_handler(func_handler)
    assert res.is_success
    assert (
        dispatcher.dispatch(SampleCommand(payload="x", command_id="cmd-x")).value
        == "func:ok"
    )


def test_dispatch_invalid_input_types(dispatcher: FlextDispatcher) -> None:
    """Strictly reject non-model inputs to dispatch."""
    assert dispatcher.dispatch(_force_routable(None)).is_failure
    assert dispatcher.dispatch(_force_routable("not-a-model")).is_failure


def test_exception_handling_in_dispatch(dispatcher: FlextDispatcher) -> None:
    """Verify that exceptions in handlers are caught and returned as failures."""

    def breaking_handler(msg: SampleCommand) -> str:
        error_msg = "broken"
        raise ValueError(error_msg)

    setattr(breaking_handler, "message_type", SampleCommand)
    dispatcher.register_handler(breaking_handler)
    res = dispatcher.dispatch(SampleCommand(payload="x", command_id="cmd-break"))
    assert res.is_failure
    assert isinstance(res.error, str)
    assert "broken" in res.error
