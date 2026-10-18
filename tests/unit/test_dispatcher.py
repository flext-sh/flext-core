"""Behavioral tests for FlextDispatcher public contract.

Exercises the observable command-bus surface exposed through the test
facades: handler registration, message dispatch, event publication and the
protocol-conformance guarantee of the builder. Every assertion targets the
public return contract (``r[T]`` outcome, payloads, error messages) — never
internal registries or logging side effects.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from typing import override

import pytest

from tests import m, p, r, t, tm, u


class TestsFlextCoreDispatcher:
    """Public-behavior contract for the flext-core message dispatcher."""

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
            """Initialize with a route discriminator and an empty received log."""
            self.message_type = route
            self.handler_id = route
            self.received: list[p.Routable] = []

        def handle(self, message: p.Routable) -> p.Result[t.JsonPayload]:
            """Provide ``handle``.

            Returns:
                The resulting ``p.Result[t.JsonPayload]``.

            """
            self.received.append(message)
            return r[t.JsonPayload].ok({"route": self.message_type})

    class FailingHandler(RecordingHandler):
        """Handler whose execution always raises, exercising the failure path."""

        def __init__(self, route: str, failure_detail: str) -> None:
            """Initialize with a route discriminator and the failure payload."""
            super().__init__(route)
            self.failure_detail = failure_detail

        @override
        def handle(self, message: p.Routable) -> p.Result[t.JsonPayload]:
            self.received.append(message)
            raise RuntimeError(self.failure_detail)

    class AutoSubscriber:
        """An existing auto-discovery capability with no declared route."""

        def __init__(self, calls: list[str]) -> None:
            self.calls = calls

        def can_handle(self, message_type: type) -> bool:
            """Accept every event type for observable ordering tests."""
            return isinstance(message_type, type)

        def handle(self, _message: p.Routable) -> p.Result[t.JsonPayload]:
            """Record execution without changing the subscriber collection."""
            self.calls.append("auto")
            return r[t.JsonPayload].ok(True)

    @staticmethod
    @pytest.fixture
    def dispatcher() -> p.Dispatcher:
        """Return a fresh dispatcher per test (isolation guarantee).

        Returns:
            A fresh dispatcher per test (isolation guarantee).

        """
        return u.build_dispatcher()

    @staticmethod
    def test_build_dispatcher_returns_protocol_conformant_instance() -> None:
        # Arrange / Act
        """Test build dispatcher returns protocol conformant instance."""
        dispatcher = u.build_dispatcher()

        # Assert: builder promises a Dispatcher-shaped object, not a concrete type.
        assert isinstance(dispatcher, p.Dispatcher)

    @staticmethod
    def test_register_callable_handler_succeeds(dispatcher: p.Dispatcher) -> None:
        # Arrange / Act
        """Test register callable handler succeeds."""
        result = dispatcher.register_handler(RecordingHandler("register_ok"))

        # Assert
        assert result.success
        assert result.value is True

    @staticmethod
    def test_dispatch_routes_message_to_registered_handler(
        dispatcher: p.Dispatcher,
    ) -> None:
        # Arrange
        """Test dispatch routes message to registered handler."""
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

    @staticmethod
    @pytest.mark.parametrize("wrapped", [False, True])
    def test_public_execution_preserves_model_payload(*, wrapped: bool) -> None:
        """The public helper accepts raw and result-wrapped Pydantic payloads."""
        message = RouteMessage(command_type="model_payload")
        payload = RouteMessage(query_type="model_response")

        def handle(_message: p.Routable) -> t.JsonPayload | p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok(payload) if wrapped else payload

        result = u.execute_dispatcher_handler(
            resolved_handler=handle,
            message=message,
            route_name="model_payload",
            logger=u.fetch_logger(__name__),
        )

        assert result.success
        assert result.value is payload

    @staticmethod
    def test_dispatch_without_matching_handler_fails(
        dispatcher: p.Dispatcher,
    ) -> None:
        # Arrange: a well-routed message but no handler registered for it.
        """Test dispatch without matching handler fails."""
        command = RouteMessage(command_type="never_registered")

        # Act
        result = dispatcher.dispatch(command)

        # Assert
        assert result.failure
        assert result.error is not None
        assert "No handler found" in result.error

    @staticmethod
    def test_dispatch_message_without_route_fails(
        dispatcher: p.Dispatcher,
    ) -> None:
        # Arrange: no route discriminator set -> unroutable message.
        """Test dispatch message without route fails."""
        message = RouteMessage()

        # Act
        result = dispatcher.dispatch(message)

        # Assert: routing failure surfaces as a failed result, not an exception.
        assert result.failure
        assert result.error is not None
        assert "dispatch message" in result.error

    @staticmethod
    def test_registered_handler_is_not_invoked_for_other_routes(
        dispatcher: p.Dispatcher,
    ) -> None:
        # Arrange
        """Test registered handler is not invoked for other routes."""
        handler = RecordingHandler("owns_this")
        _ = dispatcher.register_handler(handler)

        # Act: dispatch an unrelated, unregistered route.
        result = dispatcher.dispatch(RouteMessage(command_type="something_else"))

        # Assert: the handler must not observe messages outside its route.
        assert result.failure
        assert handler.received == []

    @staticmethod
    def test_register_callable_without_route_fails(
        dispatcher: p.Dispatcher,
    ) -> None:
        # Arrange: callable exposing no message_type / event_type / can_handle.
        """Test register callable without route fails."""

        def orphan_handler(_message: p.Routable) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok({})

        # Act
        result = dispatcher.register_handler(orphan_handler)

        # Assert
        assert result.failure
        assert result.error is not None
        assert "message_type" in result.error

    @staticmethod
    def test_publish_invokes_subscriber_and_reports_success(
        dispatcher: p.Dispatcher,
    ) -> None:
        # Arrange
        """Test publish invokes subscriber and reports success."""
        subscriber = RecordingHandler("thing_happened")
        _ = dispatcher.register_handler(subscriber, is_event=True)
        event = RouteMessage(event_type="thing_happened")

        # Act
        result = dispatcher.publish(event)

        # Assert: publication succeeds and the subscriber observed the event.
        assert result.success
        assert result.value is True
        assert subscriber.received == [event]

    @staticmethod
    def test_publish_sequence_fans_out_to_each_event(
        dispatcher: p.Dispatcher,
    ) -> None:
        # Arrange
        """Test publish sequence fans out to each event."""
        subscriber = RecordingHandler("batched")
        _ = dispatcher.register_handler(subscriber, is_event=True)
        event = RouteMessage(event_type="batched")

        # Act
        result = dispatcher.publish([event, event, event])

        # Assert
        assert result.success
        assert len(subscriber.received) == 3

    @staticmethod
    def test_publish_without_subscribers_is_successful_noop(
        dispatcher: p.Dispatcher,
    ) -> None:
        # Arrange: no subscriber registered for this event route.
        """Test publish without subscribers is successful noop."""
        event = RouteMessage(event_type="unheard")

        # Act
        result = dispatcher.publish(event)

        # Assert: publishing to nobody is a benign success, not an error.
        assert result.success
        assert result.value is True

    @staticmethod
    def test_dispatchers_do_not_share_handler_registrations() -> None:
        # Arrange
        """Test dispatchers do not share handler registrations."""
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

    @staticmethod
    def test_dispatch_logs_handler_exception_type_and_message() -> None:
        # Arrange: a handler whose execution raises a runtime failure.
        """Test dispatch logs handler exception type and message."""
        failure_detail = "upstream inventory unavailable"

        # Act: capture what an operator would actually see on the log stream.
        stream = io.StringIO()
        with redirect_stdout(stream):
            dispatcher = u.build_dispatcher()
            _ = dispatcher.register_handler(FailingHandler("explodes", failure_detail))
            result = dispatcher.dispatch(RouteMessage(command_type="explodes"))
            deadline = time.monotonic() + 0.25
            while time.monotonic() < deadline:
                if "RuntimeError" in stream.getvalue():
                    break
                time.sleep(0.01)
        emitted = stream.getvalue()

        # Assert: the failure is reported AND its cause is diagnosable in the log.
        assert result.failure
        assert result.error is not None
        assert failure_detail in result.error
        assert "RuntimeError" in emitted
        assert failure_detail in emitted

    @staticmethod
    def test_command_conflict_preserves_original_and_bound_method_identity(
        dispatcher: p.Dispatcher,
    ) -> None:
        """Repeated bound-method access is idempotent, not a new handler."""
        original = RecordingHandler("identity")
        replacement = RecordingHandler("identity")
        tm.ok(
            dispatcher.register_handlers((
                m.HandlerRegistration(
                    handler=original.handle,
                    message_type="identity",
                ),
            )),
            where=lambda report: len(report.registered) == 1 and not report.skipped,
        )
        tm.ok(
            dispatcher.register_handlers((
                m.HandlerRegistration(
                    handler=original.handle,
                    message_type="identity",
                ),
            )),
            where=lambda report: not report.registered and len(report.skipped) == 1,
        )
        tm.fail(
            dispatcher.register_handler(replacement.handle, message_type="identity")
        )
        message = RouteMessage(command_type="identity")
        tm.ok(dispatcher.dispatch(message))
        tm.that(original.received == [message])
        tm.that(not replacement.received)

    @staticmethod
    @pytest.mark.parametrize("invalid_last", [False, True])
    def test_batch_failure_never_publishes_earlier_entries(
        dispatcher: p.Dispatcher,
        *,
        invalid_last: bool,
    ) -> None:
        """Internal conflicts and noncallable tails preserve the old snapshot."""
        first = RecordingHandler("atomic")
        last: t.DispatchableHandler = (
            RouteMessage() if invalid_last else RecordingHandler("atomic")
        )
        tm.fail(
            dispatcher.register_handlers((
                m.HandlerRegistration(handler=first),
                m.HandlerRegistration(handler=last),
            ))
        )
        tm.fail(dispatcher.dispatch(RouteMessage(command_type="atomic")))
        tm.that(not first.received)

    @staticmethod
    def test_existing_command_conflict_does_not_publish_new_route(
        dispatcher: p.Dispatcher,
    ) -> None:
        """A conflict with active state rejects every binding in the batch."""
        original = RecordingHandler("occupied")
        tm.ok(dispatcher.register_handler(original))
        tm.fail(
            dispatcher.register_handlers((
                m.HandlerRegistration(handler=RecordingHandler("new")),
                m.HandlerRegistration(handler=RecordingHandler("occupied")),
            ))
        )
        tm.fail(dispatcher.dispatch(RouteMessage(command_type="new")))
        tm.ok(dispatcher.dispatch(RouteMessage(command_type="occupied")))
        tm.that(len(original.received) == 1)

    @staticmethod
    def test_explicit_bindings_route_original_callable_and_reject_contradiction(
        dispatcher: p.Dispatcher,
    ) -> None:
        """Explicit routes bind callables without changing their metadata."""
        first = RecordingHandler("first")
        second = RecordingHandler("second")
        tm.ok(
            dispatcher.register_handlers((
                m.HandlerRegistration(handler=first.handle, message_type="first"),
                m.HandlerRegistration(handler=second.handle, message_type="second"),
            )),
            where=lambda report: len(report.registered) == 2 and not report.skipped,
        )
        messages = (
            RouteMessage(command_type="first"),
            RouteMessage(command_type="second"),
        )
        for message in messages:
            tm.ok(dispatcher.dispatch(message))
        tm.that(first.received == [messages[0]])
        tm.that(second.received == [messages[1]])
        tm.fail(dispatcher.register_handler(first, message_type="contradiction"))
        tm.fail(dispatcher.dispatch(RouteMessage(command_type="contradiction")))

    @staticmethod
    def test_event_batch_fans_out_distinct_subscribers_once(
        dispatcher: p.Dispatcher,
    ) -> None:
        """Event route duplicates retain distinct subscribers and deduplicate identities."""
        first = RecordingHandler("fanout")
        second = RecordingHandler("fanout")
        tm.ok(
            dispatcher.register_handlers((
                m.HandlerRegistration(
                    handler=first.handle, message_type="fanout", is_event=True
                ),
                m.HandlerRegistration(
                    handler=second.handle, message_type="fanout", is_event=True
                ),
                m.HandlerRegistration(
                    handler=first.handle, message_type="fanout", is_event=True
                ),
            )),
            where=lambda report: (
                len(report.registered) == 2 and len(report.skipped) == 1
            ),
        )
        event = RouteMessage(event_type="fanout")
        tm.ok(dispatcher.publish(event))
        tm.that(first.received == [event])
        tm.that(second.received == [event])

    @staticmethod
    @pytest.mark.parametrize("raises", [False, True])
    def test_publication_stops_at_failure_and_preserves_cause(
        dispatcher: p.Dispatcher,
        *,
        raises: bool,
    ) -> None:
        """Failure halts remaining subscribers/events, retaining earlier effects."""
        cause = RuntimeError("subscriber unavailable")
        before = RecordingHandler("failure")
        after = RecordingHandler("failure")
        failures: list[p.Routable] = []

        def failing(message: p.Routable) -> p.Result[t.JsonPayload]:
            failures.append(message)
            if raises:
                raise cause
            return r[t.JsonPayload].fail("subscriber unavailable", exception=cause)

        tm.ok(
            dispatcher.register_handlers((
                m.HandlerRegistration(handler=before, is_event=True),
                m.HandlerRegistration(
                    handler=failing, message_type="failure", is_event=True
                ),
                m.HandlerRegistration(handler=after, is_event=True),
            )),
            where=lambda report: len(report.registered) == 3 and not report.skipped,
        )
        event = RouteMessage(event_type="failure")
        result = dispatcher.publish((event, event))
        tm.fail(result)
        tm.that(result.exception is cause)
        tm.that(before.received == [event])
        tm.that(failures == [event])
        tm.that(not after.received)

    @staticmethod
    def test_registration_during_publish_uses_next_operation_snapshot(
        dispatcher: p.Dispatcher,
    ) -> None:
        """Reentrant registration does not deadlock or change an active sequence."""
        late = RecordingHandler("snapshot")
        registrations: list[p.Result[bool]] = []

        def registering(_message: p.Routable) -> p.Result[t.JsonPayload]:
            registrations.append(dispatcher.register_handler(late, is_event=True))
            return r[t.JsonPayload].ok(True)

        tm.ok(
            dispatcher.register_handler(
                registering, message_type="snapshot", is_event=True
            )
        )
        event = RouteMessage(event_type="snapshot")
        tm.ok(dispatcher.publish((event, event)))
        tm.that(not late.received)
        tm.ok(dispatcher.publish(event))
        tm.that(late.received == [event])
        for registration in registrations:
            tm.ok(registration, eq=True)

    @staticmethod
    def test_concurrent_command_writers_cannot_overwrite_each_other(
        dispatcher: p.Dispatcher,
    ) -> None:
        """One writer owns a command route and every rejected batch stays absent."""
        first = RecordingHandler("concurrent")
        second = RecordingHandler("concurrent")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    dispatcher.register_handlers,
                    (
                        m.HandlerRegistration(handler=handler),
                        m.HandlerRegistration(handler=RecordingHandler(extra)),
                    ),
                )
                for handler, extra in (
                    (first, "winner_first"),
                    (second, "winner_second"),
                )
            ]
            results = [future.result() for future in futures]
        tm.that(sum(result.success for result in results) == 1)
        tm.ok(dispatcher.dispatch(RouteMessage(command_type="concurrent")))
        tm.that(len(first.received) + len(second.received) == 1)
        for result, route in zip(
            results, ("winner_first", "winner_second"), strict=True
        ):
            observed = dispatcher.dispatch(RouteMessage(command_type=route))
            tm.that(observed.success is result.success)

    @staticmethod
    def test_auto_and_explicit_subscribers_follow_registration_order(
        dispatcher: p.Dispatcher,
    ) -> None:
        """Explicit routing does not jump ahead of an older auto subscriber."""
        calls: list[str] = []
        auto = TestsFlextCoreDispatcher.AutoSubscriber(calls)

        def explicit(_message: p.Routable) -> p.Result[t.JsonPayload]:
            calls.append("explicit")
            return r[t.JsonPayload].ok(True)

        tm.ok(
            dispatcher.register_handlers((
                m.HandlerRegistration(handler=auto, is_event=True),
                m.HandlerRegistration(
                    handler=explicit, message_type="ordered", is_event=True
                ),
            ))
        )
        tm.ok(dispatcher.publish(RouteMessage(event_type="ordered")))
        tm.that(calls == ["auto", "explicit"])


RouteMessage = TestsFlextCoreDispatcher.RouteMessage
RecordingHandler = TestsFlextCoreDispatcher.RecordingHandler
FailingHandler = TestsFlextCoreDispatcher.FailingHandler
