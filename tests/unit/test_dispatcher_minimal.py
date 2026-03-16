"""Minimal dispatcher flow coverage with real handlers (no mocks).

Tests the strict FlextDispatcher API:
- register_handler(handler: p.HandlerLike) — handler must expose
  message_type, event_type, or can_handle.
- dispatch(message: p.Routable) — accepts only CQRS messages.
"""

from __future__ import annotations

from flext_core import FlextDispatcher, r
from tests.constants import c
from tests.models import m


class EchoHandler:
    """Handler that echoes command_type back."""

    message_type = "EchoRoute"

    def handle(self, msg: m.Command) -> str:
        return f"handled:{msg.command_type}"

    def __call__(self, msg: m.Command) -> str:
        return self.handle(msg)


class ExplodingHandler:
    """Handler that raises on handle()."""

    message_type = "ExplodeRoute"

    def handle(self, _: m.Command) -> str:
        msg = "boom"
        raise RuntimeError(msg)

    def __call__(self, msg: m.Command) -> str:
        return self.handle(msg)


class AutoCommand(m.Command):
    """Auto-routed command fixture."""

    command_type: str = "AutoRoute"
    payload: str = "auto"


class AutoDiscoveryHandler:
    """Handler using can_handle for route resolution."""

    def can_handle(self, msg_type: type) -> bool:
        return msg_type is AutoCommand

    def handle(self, msg: AutoCommand) -> str:
        return f"auto:{msg.command_type}"

    def __call__(self, msg: AutoCommand) -> str:
        return self.handle(msg)


class EventSubscriber:
    """Event handler for publish() testing."""

    message_type = "OrderCreated"

    def __init__(self) -> None:
        """Initialize received events list."""
        self.received: list[m.Event] = []

    def handle(self, event: m.Event) -> None:
        self.received.append(event)

    def __call__(self, event: m.Event) -> None:
        self.handle(event)


def test_register_handler_with_message_type() -> None:
    """Handler with message_type attribute registers successfully."""
    dispatcher = FlextDispatcher()
    res = dispatcher.register_handler(EchoHandler())
    assert res.is_success


def test_register_handler_with_can_handle() -> None:
    """Handler with can_handle registers as auto-discovery handler."""
    dispatcher = FlextDispatcher()
    res = dispatcher.register_handler(AutoDiscoveryHandler())
    assert res.is_success


def test_register_handler_without_route_fails() -> None:
    """Handler without message_type/event_type/can_handle must fail."""
    dispatcher = FlextDispatcher()

    class BareHandler:
        """Callable handler lacking routing attributes — should fail registration."""

        def __call__(self, msg: m.Command) -> r[str]:
            _ = msg
            return r[str].ok("bare")

    res = dispatcher.register_handler(BareHandler())
    assert res.is_failure
    assert "must expose" in (res.error or "")


def test_register_handler_as_event_subscriber() -> None:
    """Handler registered with is_event=True goes to event subscribers."""
    dispatcher = FlextDispatcher()
    subscriber = EventSubscriber()
    res = dispatcher.register_handler(subscriber, is_event=True)
    assert res.is_success


def test_dispatch_command_success() -> None:
    """Dispatch a Command to its registered handler."""
    dispatcher = FlextDispatcher()
    dispatcher.register_handler(EchoHandler())
    cmd = m.Command(command_type="EchoRoute", command_id="cmd-echo")
    result = dispatcher.dispatch(cmd)
    assert result.is_success
    assert result.value == "handled:EchoRoute"


def test_dispatch_no_handler_fails() -> None:
    """Dispatch with no matching handler returns failure."""
    dispatcher = FlextDispatcher()
    cmd = m.Command(command_type="UnknownRoute", command_id="cmd-unknown")
    result = dispatcher.dispatch(cmd)
    assert result.is_failure
    assert result.error_code == c.Errors.COMMAND_HANDLER_NOT_FOUND


def test_dispatch_handler_exception_returns_failure() -> None:
    """Handler that raises returns a failure result."""
    dispatcher = FlextDispatcher()
    dispatcher.register_handler(ExplodingHandler())
    cmd = m.Command(command_type="ExplodeRoute", command_id="cmd-explode")
    result = dispatcher.dispatch(cmd)
    assert result.is_failure
    assert "boom" in (result.error or "")
    assert result.error_code == c.Errors.COMMAND_PROCESSING_FAILED


def test_dispatch_auto_discovery_handler() -> None:
    """Auto-discovery handler is found via can_handle fallback."""
    dispatcher = FlextDispatcher()
    dispatcher.register_handler(AutoDiscoveryHandler())
    cmd = AutoCommand(command_id="cmd-auto")
    result = dispatcher.dispatch(cmd)
    assert result.is_success
    assert result.value == "auto:AutoRoute"


def test_dispatch_after_handler_removal_fails() -> None:
    """Dispatching when handler route is cleared fails gracefully."""
    dispatcher = FlextDispatcher()
    dispatcher.register_handler(EchoHandler())
    dispatcher._handlers.clear()
    cmd = m.Command(command_type="EchoRoute", command_id="cmd-cleared")
    result = dispatcher.dispatch(cmd)
    assert result.is_failure
    assert result.error_code == c.Errors.COMMAND_HANDLER_NOT_FOUND


def test_publish_event_to_subscriber() -> None:
    """Published event is delivered to registered subscriber."""
    dispatcher = FlextDispatcher()
    subscriber = EventSubscriber()
    dispatcher.register_handler(subscriber, is_event=True)
    event = m.Event(
        event_type="OrderCreated",
        aggregate_id="order-1",
        event_id="evt-order",
        data=m.Dict({}),
        metadata=m.Dict({}),
    )
    res = dispatcher.publish(event)
    assert res.is_success
    assert len(subscriber.received) == 1


def test_publish_no_subscribers_succeeds() -> None:
    """Publishing with no subscribers succeeds silently."""
    dispatcher = FlextDispatcher()
    event = m.Event(
        event_type="NobodyListening",
        aggregate_id="agg-1",
        event_id="evt-none",
        data=m.Dict({}),
        metadata=m.Dict({}),
    )
    res = dispatcher.publish(event)
    assert res.is_success
