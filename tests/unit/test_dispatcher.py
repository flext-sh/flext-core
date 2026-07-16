"""Behavioral tests for FlextDispatcher public contract.

Exercises the observable command-bus surface exposed through the test
facades: handler registration, message dispatch, event publication and the
protocol-conformance guarantee of the builder. Every assertion targets the
public return contract (``r[T]`` outcome, payloads, error messages) — never
internal registries or logging side effects.
"""

from __future__ import annotations

import pytest
from flext_tests import r

from tests.models import m
from tests.protocols import p
from tests.typings import p, t
from tests.utilities import u


class RouteMessage(m.BaseModel):
    """Routable test message satisfying the ``p.Routable`` contract.

    Declares all three CQRS route discriminators so a single model can stand
    in for a command, a query or an event depending on which field is set.
    """

    command_type: str | None = None
    query_type: str | None = None
    event_type: str | None = None


class RecordingHandler:
    """Handler that records every message it receives and returns a payload."""

    def __init__(self, route: str) -> None:
        self.message_type = route
        self.received: list[p.Routable] = []

    def handle(self, message: p.Routable) -> p.Result[t.JsonPayload]:
        self.received.append(message)
        return r[t.JsonPayload].ok({"route": self.message_type})


class TestsFlextCoreDispatcher:
    """Public-behavior contract for the flext-core message dispatcher."""

    @pytest.fixture
    def dispatcher(self) -> p.Dispatcher:
        """Return a fresh dispatcher per test (isolation guarantee)."""
        return u.build_dispatcher()

    def test_build_dispatcher_returns_protocol_conformant_instance(self) -> None:
        # Arrange / Act
        dispatcher = u.build_dispatcher()

        # Assert: builder promises a Dispatcher-shaped object, not a concrete type.
        assert isinstance(dispatcher, p.Dispatcher)

    def test_register_callable_handler_succeeds(self, dispatcher: p.Dispatcher) -> None:
        # Arrange / Act
        result = dispatcher.register_handler(RecordingHandler("register_ok"))

        # Assert
        assert result.success
        assert result.value is True

    def test_dispatch_routes_message_to_registered_handler(
        self, dispatcher: p.Dispatcher
    ) -> None:
        # Arrange
        handler = RecordingHandler("routed_cmd")
        _ = dispatcher.register_handler(handler)
        command = RouteMessage(command_type="routed_cmd")

        # Act
        result = dispatcher.dispatch(command)

        # Assert: handler ran with the exact message and its payload is surfaced.
        assert result.success
        assert result.value == {"route": "routed_cmd"}
        assert result.error is None
        assert handler.received == [command]

    def test_dispatch_without_matching_handler_fails(
        self, dispatcher: p.Dispatcher
    ) -> None:
        # Arrange: a well-routed message but no handler registered for it.
        command = RouteMessage(command_type="never_registered")

        # Act
        result = dispatcher.dispatch(command)

        # Assert
        assert result.failure
        assert result.error is not None
        assert "No handler found" in result.error

    def test_dispatch_message_without_route_fails(
        self, dispatcher: p.Dispatcher
    ) -> None:
        # Arrange: no route discriminator set -> unroutable message.
        message = RouteMessage()

        # Act
        result = dispatcher.dispatch(message)

        # Assert: routing failure surfaces as a failed result, not an exception.
        assert result.failure
        assert result.error is not None
        assert "dispatch message" in result.error

    def test_registered_handler_is_not_invoked_for_other_routes(
        self, dispatcher: p.Dispatcher
    ) -> None:
        # Arrange
        handler = RecordingHandler("owns_this")
        _ = dispatcher.register_handler(handler)

        # Act: dispatch an unrelated, unregistered route.
        result = dispatcher.dispatch(RouteMessage(command_type="something_else"))

        # Assert: the handler must not observe messages outside its route.
        assert result.failure
        assert handler.received == []

    def test_register_callable_without_route_fails(
        self, dispatcher: p.Dispatcher
    ) -> None:
        # Arrange: callable exposing no message_type / event_type / can_handle.
        def orphan_handler(message: p.Routable) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok({})

        # Act
        result = dispatcher.register_handler(orphan_handler)

        # Assert
        assert result.failure
        assert result.error is not None
        assert "message_type" in result.error

    def test_publish_invokes_subscriber_and_reports_success(
        self, dispatcher: p.Dispatcher
    ) -> None:
        # Arrange
        subscriber = RecordingHandler("thing_happened")
        _ = dispatcher.register_handler(subscriber, is_event=True)
        event = RouteMessage(event_type="thing_happened")

        # Act
        result = dispatcher.publish(event)

        # Assert: publication succeeds and the subscriber observed the event.
        assert result.success
        assert result.value is True
        assert subscriber.received == [event]

    def test_publish_sequence_fans_out_to_each_event(
        self, dispatcher: p.Dispatcher
    ) -> None:
        # Arrange
        subscriber = RecordingHandler("batched")
        _ = dispatcher.register_handler(subscriber, is_event=True)
        event = RouteMessage(event_type="batched")

        # Act
        result = dispatcher.publish([event, event, event])

        # Assert
        assert result.success
        assert len(subscriber.received) == 3

    def test_publish_without_subscribers_is_successful_noop(
        self, dispatcher: p.Dispatcher
    ) -> None:
        # Arrange: no subscriber registered for this event route.
        event = RouteMessage(event_type="unheard")

        # Act
        result = dispatcher.publish(event)

        # Assert: publishing to nobody is a benign success, not an error.
        assert result.success
        assert result.value is True

    def test_dispatchers_do_not_share_handler_registrations(self) -> None:
        # Arrange
        registered = u.build_dispatcher()
        _ = registered.register_handler(RecordingHandler("isolated_cmd"))
        other = u.build_dispatcher()
        command = RouteMessage(command_type="isolated_cmd")

        # Act
        registered_result = registered.dispatch(command)
        other_result = other.dispatch(command)

        # Assert: registration on one instance never leaks into another.
        assert registered_result.success
        assert other_result.failure


__all__: list[str] = ["TestsFlextCoreDispatcher"]
