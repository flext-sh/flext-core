"""Minimal dispatcher flow coverage with real handlers (no mocks).

Tests the strict FlextDispatcher API:
- register_handler(handler: t.HandlerType) — handler must expose
  message_type, event_type, or can_handle.
- dispatch(message: m.Message) — accepts only CQRS messages.
"""

from __future__ import annotations

from typing import cast

from flext_core import FlextDispatcher, c, t
from flext_core.models import m

# ---------------------------------------------------------------------------
# Test handlers
# ---------------------------------------------------------------------------


class EchoHandler:
    """Handler that echoes command_type back."""

    message_type = "EchoRoute"

    def handle(self, msg: m.Cqrs.Command) -> str:
        return f"handled:{msg.command_type}"


class ExplodingHandler:
    """Handler that raises on handle()."""

    message_type = "ExplodeRoute"

    def handle(self, _: m.Cqrs.Command) -> str:
        msg = "boom"
        raise RuntimeError(msg)


class AutoCommand(m.Cqrs.Command):
    """Auto-routed command fixture."""

    command_type: str = "AutoRoute"
    payload: str = "auto"


class AutoDiscoveryHandler:
    """Handler using can_handle for route resolution."""

    def can_handle(self, msg_type: type) -> bool:
        return msg_type is AutoCommand

    def handle(self, msg: AutoCommand) -> str:
        return f"auto:{msg.command_type}"


class EventSubscriber:
    """Event handler for publish() testing."""

    message_type = "OrderCreated"

    def __init__(self) -> None:
        """Initialize received events list."""
        self.received: list[m.Cqrs.Event] = []

    def handle(self, event: m.Cqrs.Event) -> None:
        self.received.append(event)


# ---------------------------------------------------------------------------
# Tests: register_handler
# ---------------------------------------------------------------------------


def test_register_handler_with_message_type() -> None:
    """Handler with message_type attribute registers successfully."""
    dispatcher = FlextDispatcher()
    res = dispatcher.register_handler(cast("t.HandlerType", EchoHandler()))
    assert res.is_success


def test_register_handler_with_can_handle() -> None:
    """Handler with can_handle registers as auto-discovery handler."""
    dispatcher = FlextDispatcher()
    res = dispatcher.register_handler(cast("t.HandlerType", AutoDiscoveryHandler()))
    assert res.is_success


def test_register_handler_without_route_fails() -> None:
    """Handler without message_type/event_type/can_handle must fail."""
    dispatcher = FlextDispatcher()

    class BareHandler:
        def handle(self, msg: object) -> str:
            return "bare"

    res = dispatcher.register_handler(cast("t.HandlerType", BareHandler()))
    assert res.is_failure
    assert "must expose" in (res.error or "")


def test_register_handler_as_event_subscriber() -> None:
    """Handler registered with is_event=True goes to event subscribers."""
    dispatcher = FlextDispatcher()
    subscriber = EventSubscriber()
    res = dispatcher.register_handler(cast("t.HandlerType", subscriber), is_event=True)
    assert res.is_success


# ---------------------------------------------------------------------------
# Tests: dispatch
# ---------------------------------------------------------------------------


def test_dispatch_command_success() -> None:
    """Dispatch a Command to its registered handler."""
    dispatcher = FlextDispatcher()
    dispatcher.register_handler(cast("t.HandlerType", EchoHandler()))

    cmd = m.Cqrs.Command(command_type="EchoRoute")
    result = dispatcher.dispatch(cast("m.Message", cmd))
    assert result.is_success
    assert result.value == "handled:EchoRoute"


def test_dispatch_no_handler_fails() -> None:
    """Dispatch with no matching handler returns failure."""
    dispatcher = FlextDispatcher()
    cmd = m.Cqrs.Command(command_type="UnknownRoute")
    result = dispatcher.dispatch(cast("m.Message", cmd))
    assert result.is_failure
    assert result.error_code == c.Errors.COMMAND_HANDLER_NOT_FOUND


def test_dispatch_handler_exception_returns_failure() -> None:
    """Handler that raises returns a failure result."""
    dispatcher = FlextDispatcher()
    dispatcher.register_handler(cast("t.HandlerType", ExplodingHandler()))

    cmd = m.Cqrs.Command(command_type="ExplodeRoute")
    result = dispatcher.dispatch(cast("m.Message", cmd))
    assert result.is_failure
    assert "boom" in (result.error or "")
    assert result.error_code == c.Errors.COMMAND_PROCESSING_FAILED


def test_dispatch_auto_discovery_handler() -> None:
    """Auto-discovery handler is found via can_handle fallback."""
    dispatcher = FlextDispatcher()
    dispatcher.register_handler(cast("t.HandlerType", AutoDiscoveryHandler()))

    cmd = AutoCommand()
    result = dispatcher.dispatch(cast("m.Message", cmd))
    assert result.is_success
    assert result.value == "auto:AutoRoute"


def test_dispatch_non_callable_handler_fails() -> None:
    """A non-callable entry in _handlers should yield failure."""
    dispatcher = FlextDispatcher()
    dispatcher._handlers["BadRoute"] = cast("t.HandlerType", "not_callable")

    cmd = m.Cqrs.Command(command_type="BadRoute")
    result = dispatcher.dispatch(cast("m.Message", cmd))
    assert result.is_failure
    assert "not callable" in (result.error or "")


# ---------------------------------------------------------------------------
# Tests: publish
# ---------------------------------------------------------------------------


def test_publish_event_to_subscriber() -> None:
    """Published event is delivered to registered subscriber."""
    dispatcher = FlextDispatcher()
    subscriber = EventSubscriber()
    dispatcher.register_handler(cast("t.HandlerType", subscriber), is_event=True)

    event = m.Cqrs.Event(
        event_type="OrderCreated",
        aggregate_id="order-1",
    )
    res = dispatcher.publish(cast("m.Message", event))
    assert res.is_success
    assert len(subscriber.received) == 1


def test_publish_no_subscribers_succeeds() -> None:
    """Publishing with no subscribers succeeds silently."""
    dispatcher = FlextDispatcher()
    event = m.Cqrs.Event(
        event_type="NobodyListening",
        aggregate_id="agg-1",
    )
    res = dispatcher.publish(cast("m.Message", event))
    assert res.is_success
