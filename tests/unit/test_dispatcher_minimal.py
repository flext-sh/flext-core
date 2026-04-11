"""Minimal dispatcher flow coverage with real handlers (no mocks).

Tests the strict FlextDispatcher API:
- register_handler(handler: p.HandlerLike) — handler must expose
  message_type, event_type, or can_handle.
- dispatch(message: p.Routable) — accepts only CQRS messages.
"""

from __future__ import annotations

from collections.abc import MutableSequence

from flext_core import FlextDispatcher, r
from tests import c, m, t


class TestDispatcherMinimal:
    class _EchoHandler:
        """Handler that echoes command_type back."""

        message_type = "EchoRoute"

        def handle(self, msg: m.Command) -> str:
            return f"handled:{msg.command_type}"

        def __call__(self, msg: m.Command) -> str:
            return self.handle(msg)

    class _ExplodingHandler:
        """Handler that raises on handle()."""

        message_type = "ExplodeRoute"

        def handle(self, _: m.Command) -> str:
            msg = "boom"
            raise RuntimeError(msg)

        def __call__(self, msg: m.Command) -> str:
            return self.handle(msg)

    class _AutoCommand(m.Command):
        """Auto-routed command fixture."""

        command_type: str = "AutoRoute"
        payload: str = "auto"

    class _AutoDiscoveryHandler:
        """Handler using can_handle for route resolution."""

        def can_handle(self, msg_type: type) -> bool:
            return msg_type is TestDispatcherMinimal._AutoCommand

        def handle(self, msg: TestDispatcherMinimal._AutoCommand) -> str:
            return f"auto:{msg.command_type}"

        def __call__(self, msg: TestDispatcherMinimal._AutoCommand) -> str:
            return self.handle(msg)

    class _EventSubscriber:
        """Event handler for publish() testing."""

        message_type = "OrderCreated"

        def __init__(self) -> None:
            """Initialize received events list."""
            self.received: MutableSequence[m.Event] = []

        def handle(self, event: m.Event) -> None:
            self.received.append(event)

        def __call__(self, event: m.Event) -> None:
            self.handle(event)

    class _BareHandler:
        """Callable handler lacking routing attributes — should fail registration."""

        def __call__(self, msg: m.Command) -> r[str]:
            _ = msg
            return r[str].ok("bare")

    def test_register_handler_with_message_type(self) -> None:
        """Handler with message_type attribute registers successfully."""
        dispatcher = FlextDispatcher()
        res = dispatcher.register_handler(self._EchoHandler())
        assert res.success

    def test_register_handler_with_can_handle(self) -> None:
        """Handler with can_handle registers as auto-discovery handler."""
        dispatcher = FlextDispatcher()
        res = dispatcher.register_handler(self._AutoDiscoveryHandler())
        assert res.success

    def test_register_handler_without_route_fails(self) -> None:
        """Handler without message_type/event_type/can_handle must fail."""
        dispatcher = FlextDispatcher()
        res = dispatcher.register_handler(self._BareHandler())
        assert res.failure
        assert "must expose" in (res.error or "")

    def test_register_handler_as_event_subscriber(self) -> None:
        """Handler registered with is_event=True goes to event subscribers."""
        dispatcher = FlextDispatcher()
        subscriber = self._EventSubscriber()
        res = dispatcher.register_handler(subscriber, is_event=True)
        assert res.success

    def test_dispatch_command_success(self) -> None:
        """Dispatch a Command to its registered handler."""
        dispatcher = FlextDispatcher()
        dispatcher.register_handler(self._EchoHandler())
        cmd = m.Command(command_type="EchoRoute", command_id="cmd-echo")
        result = dispatcher.dispatch(cmd)
        assert result.success
        assert result.value == "handled:EchoRoute"

    def test_dispatch_no_handler_fails(self) -> None:
        """Dispatch with no matching handler returns failure."""
        dispatcher = FlextDispatcher()
        cmd = m.Command(command_type="UnknownRoute", command_id="cmd-unknown")
        result = dispatcher.dispatch(cmd)
        assert result.failure
        assert result.error_code == c.ErrorCode.COMMAND_HANDLER_NOT_FOUND.value

    def test_dispatch_handler_exception_returns_failure(self) -> None:
        """Handler that raises returns a failure result."""
        dispatcher = FlextDispatcher()
        dispatcher.register_handler(self._ExplodingHandler())
        cmd = m.Command(command_type="ExplodeRoute", command_id="cmd-explode")
        result = dispatcher.dispatch(cmd)
        assert result.failure
        assert "boom" in (result.error or "")
        assert result.error_code == c.ErrorCode.COMMAND_PROCESSING_FAILED.value

    def test_dispatch_auto_discovery_handler(self) -> None:
        """Auto-discovery handler is found via can_handle fallback."""
        dispatcher = FlextDispatcher()
        dispatcher.register_handler(self._AutoDiscoveryHandler())
        cmd = self._AutoCommand(command_id="cmd-auto")
        result = dispatcher.dispatch(cmd)
        assert result.success
        assert result.value == "auto:AutoRoute"

    def test_dispatch_after_handler_removal_fails(self) -> None:
        """Dispatching when handler route is cleared fails gracefully."""
        dispatcher = FlextDispatcher()
        dispatcher.register_handler(self._EchoHandler())
        dispatcher._handlers.clear()
        cmd = m.Command(command_type="EchoRoute", command_id="cmd-cleared")
        result = dispatcher.dispatch(cmd)
        assert result.failure
        assert result.error_code == c.ErrorCode.COMMAND_HANDLER_NOT_FOUND.value

    def test_publish_event_to_subscriber(self) -> None:
        """Published event is delivered to registered subscriber."""
        dispatcher = FlextDispatcher()
        subscriber = self._EventSubscriber()
        dispatcher.register_handler(subscriber, is_event=True)
        event = m.Event(
            event_type="OrderCreated",
            aggregate_id="order-1",
            event_id="evt-order",
            data=t.Dict({}),
            metadata=t.Dict({}),
        )
        res = dispatcher.publish(event)
        assert res.success
        assert len(subscriber.received) == 1

    def test_publish_no_subscribers_succeeds(self) -> None:
        """Publishing with no subscribers succeeds silently."""
        dispatcher = FlextDispatcher()
        event = m.Event(
            event_type="NobodyListening",
            aggregate_id="agg-1",
            event_id="evt-none",
            data=t.Dict({}),
            metadata=t.Dict({}),
        )
        res = dispatcher.publish(event)
        assert res.success
