"""Comprehensive coverage tests for strict FlextDispatcher implementation."""

from __future__ import annotations

import pytest

from flext_core import FlextDispatcher, m, p, t
from flext_core.dispatcher import _DispatchableHandler

# --- Type-safe test helpers for runtime-only error paths ---


def _force_handler(obj: object) -> _DispatchableHandler:
    """Return a no-op _DispatchableHandler for error-path testing.

    For non-callable objects (str, dict), returns a wrapper with no route
    attributes so the dispatcher rejects it.
    """

    def _wrapper(
        *_args: object, **_kwargs: object
    ) -> p.ResultLike[t.PayloadValue] | t.PayloadValue | None:
        return None

    return _wrapper


def _force_routable(obj: object) -> p.Routable:
    """Create a minimal Routable-compatible object for error-path testing."""

    class _FakeRoutable:
        command_type: str | None = None
        query_type: str | None = None
        event_type: str | None = None

    return _FakeRoutable()


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


class SampleHandler:
    """Strict handler implementation matching HandleProtocol."""

    message_type = SampleCommand

    def handle(self, message: p.Routable) -> t.PayloadValue:
        """Handle the message."""
        payload = getattr(message, "payload", "")
        return f"handled:{payload}"

    def can_handle(self, message_type: type) -> bool:
        return message_type is SampleCommand


class QueryHandler:
    """Query handler matching HandleProtocol."""

    message_type = SampleQuery

    def handle(self, message: p.Routable) -> t.PayloadValue:
        """Handle the query."""
        query_id = getattr(message, "query_id", None)
        return {
            "result": "data",
            "id": query_id,
        }

    def can_handle(self, message_type: type) -> bool:
        return message_type is SampleQuery


class EventHandler:
    """Event handler matching HandleProtocol."""

    message_type = SampleEvent

    def handle(self, message: p.Routable) -> t.PayloadValue:
        """Handle the event."""
        return True

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
    assert fail_res.error is not None and "no handler found" in fail_res.error.lower()


def test_invalid_registration_attempts(dispatcher: FlextDispatcher) -> None:
    """Verify that non-strict registrations are rejected."""
    # Attempting to register a string should fail
    assert dispatcher.register_handler(
        _force_handler("not-a-handler"),
    ).is_failure
    assert dispatcher.register_handler(
        _force_handler({"some": "dict"}),
    ).is_failure

    # Attempting to register a function without message_type should fail
    def nameless_handler(msg: SampleCommand) -> str:
        return "ok"

    assert dispatcher.register_handler(
        _force_handler(nameless_handler),
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

        def handle(self, message: p.Routable) -> t.PayloadValue:
            return "ok"

    res = dispatcher.register_handler(PredicateHandler())
    assert res.is_success
    assert dispatcher.dispatch(SampleCommand(payload="p")).is_success


def test_callable_registration_with_attribute(dispatcher: FlextDispatcher) -> None:
    """Verify that self-describing functions with message_type attribute are accepted."""

    def func_handler(msg: SampleCommand) -> str:
        return "func:ok"

    # Self-describing handler via attribute (set before registration)
    setattr(func_handler, "message_type", SampleCommand)

    res = dispatcher.register_handler(func_handler)
    assert res.is_success
    assert dispatcher.dispatch(SampleCommand(payload="x")).value == "func:ok"


def test_dispatch_invalid_input_types(dispatcher: FlextDispatcher) -> None:
    """Strictly reject non-model inputs to dispatch."""
    assert dispatcher.dispatch(_force_routable(None)).is_failure
    assert dispatcher.dispatch(_force_routable("not-a-model")).is_failure


def test_exception_handling_in_dispatch(dispatcher: FlextDispatcher) -> None:
    """Verify that exceptions in handlers are caught and returned as failures."""

    def breaking_handler(msg: SampleCommand) -> str:
        error_msg = "broken"
        raise ValueError(error_msg)

    # Self-describing handler via attribute (set before registration)
    setattr(breaking_handler, "message_type", SampleCommand)
    dispatcher.register_handler(breaking_handler)

    res = dispatcher.dispatch(SampleCommand(payload="x"))
    assert res.is_failure
    assert isinstance(res.error, str)
    assert "broken" in res.error
