"""Public dispatcher behavior with native messages, callbacks and shared capabilities.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from tests import c, h, m, p, r, t, tm, u


class TestsFlextCoreDispatcher:
    """Test one atomic route owner, not local DTOs or a second handler engine."""

    @staticmethod
    def test_inferred_route_preserves_broader_runtime_capability() -> None:
        """A request annotation must not narrow an explicit can_handle capability."""
        command, event = u.Tests.CapabilityHandler(), u.Tests.CapabilityHandler()
        dispatcher = u.build_dispatcher()
        tm.ok(dispatcher.register_handler(command))
        tm.ok(dispatcher.register_handler(event, is_event=True))
        message = m.Command(command_type="secondary-capability-route")
        tm.that(tm.ok(dispatcher.dispatch(message)), eq=message.command_type)
        tm.ok(dispatcher.publish(message))
        tm.that(command.handled, eq=[message.command_type])
        tm.that(event.handled, eq=[message.command_type])

    @staticmethod
    def test_builder_and_explicit_callback_registration() -> None:
        """The builder returns the public capability and binds the original callable."""
        dispatcher: p.Dispatcher = u.build_dispatcher()
        received: t.MutableSequenceOf[p.Routable] = []

        def handler(message: p.Routable) -> p.Result[t.JsonPayload]:
            received.append(message)
            return r[t.JsonPayload].ok("handled")

        tm.ok(dispatcher.register_handler(handler, message_type="explicit"))
        message = m.Command(command_type="explicit")
        tm.that(tm.ok(dispatcher.dispatch(message)), eq="handled")
        tm.that(len(received), eq=1)
        tm.that(received[0] is message)

    @staticmethod
    @pytest.mark.parametrize("wrapped", [False, True])
    def test_public_execution_preserves_model_payload(*, wrapped: bool) -> None:
        """Neither a native model nor a successful Result is converted to JSON text."""
        dispatcher = u.build_dispatcher()
        payload = m.ConfigMap(root={"native": "payload"})

        def handler(_message: p.Routable) -> p.Result[t.JsonPayload] | t.JsonPayload:
            return r[t.JsonPayload].ok(payload) if wrapped else payload

        tm.ok(dispatcher.register_handler(handler, message_type="model"))
        tm.that(tm.ok(dispatcher.dispatch(m.Command(command_type="model"))) is payload)

    @staticmethod
    @pytest.mark.parametrize("registered", [False, True])
    def test_unmatched_route_never_invokes_a_handler(*, registered: bool) -> None:
        """Missing routes fail without executing another route's handler."""
        dispatcher = u.build_dispatcher()
        received: t.MutableSequenceOf[p.Routable] = []

        def handler(message: p.Routable) -> p.Result[t.JsonPayload]:
            received.append(message)
            return r[t.JsonPayload].ok(value=True)

        if registered:
            tm.ok(dispatcher.register_handler(handler, message_type="known"))
        tm.that(dispatcher.dispatch(m.Command(command_type="unknown")).failure)
        tm.that(received, empty=True)

    @staticmethod
    def test_message_without_a_route_fails_with_its_cause() -> None:
        """The native route boundary rejects unroutable data without a fake default."""
        result = u.build_dispatcher().dispatch(m.ConfigMap(root={}))
        tm.that(result.failure)
        tm.that(isinstance(result.exception, (TypeError, ValueError)))

    @staticmethod
    def test_callable_without_a_route_fails_before_execution() -> None:
        """Opaque callbacks require explicit route metadata or a matching capability."""
        received: t.MutableSequenceOf[p.Routable] = []

        def handler(message: p.Routable) -> p.Result[t.JsonPayload]:
            received.append(message)
            return r[t.JsonPayload].ok(value=True)

        tm.that(u.build_dispatcher().register_handler(handler).failure)
        tm.that(received, empty=True)

    @staticmethod
    @pytest.mark.parametrize("sequence", [False, True])
    def test_publication_delivers_original_events_in_input_order(
        *, sequence: bool
    ) -> None:
        """Events reach the same subscriber without reconstruction."""
        dispatcher = u.build_dispatcher()
        received: t.MutableSequenceOf[p.Routable] = []

        def handler(message: p.Routable) -> p.Result[t.JsonPayload]:
            received.append(message)
            return r[t.JsonPayload].ok(value=True)

        tm.ok(
            dispatcher.register_handler(
                handler, message_type=c.HandlerType.EVENT, is_event=True
            )
        )
        first = m.Event(event_type=c.HandlerType.EVENT, aggregate_id="dispatcher")
        second = m.Event(event_type=c.HandlerType.EVENT, aggregate_id="dispatcher")
        tm.ok(dispatcher.publish((first, second) if sequence else first))
        tm.that(len(received), eq=2 if sequence else 1)
        tm.that(received[0] is first)
        if sequence:
            tm.that(received[1] is second)

    @staticmethod
    def test_publication_without_subscribers_is_a_successful_noop() -> None:
        """Publishing without subscribers succeeds without effects."""
        dispatcher = u.build_dispatcher()
        tm.that(
            tm.ok(
                dispatcher.publish(
                    m.Event(event_type="absent", aggregate_id="dispatcher")
                )
            ),
            eq=True,
        )
        tm.ok(tm.ok(dispatcher.publish(())))

    @staticmethod
    def test_composed_dispatchers_do_not_share_routes() -> None:
        """Each composed dispatcher owns independent route bindings."""
        first, second = u.build_dispatcher(), u.build_dispatcher()

        def handler(_message: p.Routable) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok("first")

        tm.ok(first.register_handler(handler, message_type="isolated"))
        message = m.Command(command_type="isolated")
        tm.that(tm.ok(first.dispatch(message)), eq="first")
        tm.that(second.dispatch(message).failure)

    @staticmethod
    def test_handler_exception_preserves_cause_and_emits_real_diagnostics(
        capfd: p.Tests.OutputCapture,
    ) -> None:
        """Handler failures retain their cause and emit native diagnostics."""
        dispatcher = u.build_dispatcher()
        cause = RuntimeError("native handler failure")

        def handler(_message: p.Routable) -> p.Result[t.JsonPayload]:
            raise cause

        tm.ok(dispatcher.register_handler(handler, message_type="explodes"))
        result = dispatcher.dispatch(m.Command(command_type="explodes"))
        captured = capfd.readouterr()
        tm.that(result.failure)
        tm.that(result.exception is cause)
        tm.that(captured.out + captured.err, contains=type(cause).__name__)
        tm.that(captured.out + captured.err, contains=str(cause))

    @staticmethod
    def test_bound_method_repeat_is_idempotent_and_another_receiver_conflicts() -> None:
        """Repeated bound methods are idempotent only for the same receiver."""
        dispatcher = u.build_dispatcher()
        original, other = u.Tests.AutoSubscriber(), u.Tests.AutoSubscriber()
        first = original.handle
        tm.ok(dispatcher.register_handler(first, message_type="bound"))
        repeat: p.HandlerRegistrationReport = tm.ok(
            dispatcher.register_handlers((
                m.HandlerRegistration(handler=original.handle, message_type="bound"),
            ))
        )
        tm.that(repeat.registered, empty=True)
        tm.that(len(repeat.skipped), eq=1)
        tm.that(dispatcher.register_handler(other.handle, message_type="bound").failure)
        message = m.Command(command_type="bound")
        tm.ok(dispatcher.dispatch(message), eq=True)
        tm.that(original.handled, eq=[message.command_type])
        tm.that(other.handled, empty=True)

    @staticmethod
    @pytest.mark.parametrize("existing", [False, True])
    def test_batch_conflict_never_publishes_an_earlier_binding(
        *, existing: bool
    ) -> None:
        """Batch conflicts preserve the previously published graph."""
        dispatcher = u.build_dispatcher()

        def original(_message: p.Routable) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok("original")

        def replacement(_message: p.Routable) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok("replacement")

        if existing:
            tm.ok(dispatcher.register_handler(original, message_type="conflict"))
        bindings = [m.HandlerRegistration(handler=replacement, message_type="new")]
        if not existing:
            bindings.append(
                m.HandlerRegistration(handler=original, message_type="conflict")
            )
        bindings.append(
            m.HandlerRegistration(handler=replacement, message_type="conflict")
        )
        tm.that(dispatcher.register_handlers(bindings).failure)
        tm.that(dispatcher.dispatch(m.Command(command_type="new")).failure)
        result = dispatcher.dispatch(m.Command(command_type="conflict"))
        if existing:
            tm.that(tm.ok(result), eq="original")
        else:
            tm.that(result.failure)

    @staticmethod
    def test_explicit_binding_preserves_callable_and_rejects_conflict() -> None:
        """Explicit bindings retain identity and reject contradictory metadata."""
        dispatcher = u.build_dispatcher()

        @h.handler(command=m.Command)
        def handler(_message: p.Routable) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok("original")

        report: p.HandlerRegistrationReport = tm.ok(
            dispatcher.register_handlers((
                m.HandlerRegistration(
                    handler=handler, message_type=c.DEFAULT_COMMAND_TYPE
                ),
            ))
        )
        tm.that(report.registered[0].handler is handler)
        tm.that(
            dispatcher.register_handler(handler, message_type="contradictory").failure
        )
        tm.that(tm.ok(dispatcher.dispatch(m.Command())), eq="original")

    @staticmethod
    def test_event_batch_fans_out_distinct_callbacks_once() -> None:
        """Each distinct subscriber receives one event from the batch."""
        dispatcher = u.build_dispatcher()
        calls: t.MutableSequenceOf[str] = []

        def first(_message: p.Routable) -> p.Result[t.JsonPayload]:
            calls.append("first")
            return r[t.JsonPayload].ok(value=True)

        def second(_message: p.Routable) -> p.Result[t.JsonPayload]:
            calls.append("second")
            return r[t.JsonPayload].ok(value=True)

        report: p.HandlerRegistrationReport = tm.ok(
            dispatcher.register_handlers(
                tuple(
                    m.HandlerRegistration(
                        handler=handler, message_type="fanout", is_event=True
                    )
                    for handler in (first, second, first)
                )
            )
        )
        tm.that(len(report.registered), eq=2)
        tm.that(len(report.skipped), eq=1)
        tm.ok(
            dispatcher.publish(m.Event(event_type="fanout", aggregate_id="dispatcher"))
        )
        tm.that(calls, eq=["first", "second"])

    @staticmethod
    def test_publication_stops_at_failure_without_undoing_prior_effects() -> None:
        """Publication retains prior effects and stops at the first failure."""
        dispatcher = u.build_dispatcher()
        calls: t.MutableSequenceOf[str] = []
        cause = RuntimeError("subscriber failure")
        failure = r[t.JsonPayload].fail(
            "subscriber rejected", exception=cause, error_code="event-rejected"
        )

        def first(_message: p.Routable) -> p.Result[t.JsonPayload]:
            calls.append("first")
            return r[t.JsonPayload].ok(value=True)

        def failed(_message: p.Routable) -> p.Result[t.JsonPayload]:
            calls.append("failed")
            return failure

        def later(_message: p.Routable) -> p.Result[t.JsonPayload]:
            calls.append("later")
            return r[t.JsonPayload].ok(value=True)

        tm.ok(
            dispatcher.register_handlers(
                tuple(
                    m.HandlerRegistration(
                        handler=handler, message_type="ordered", is_event=True
                    )
                    for handler in (first, failed, later)
                )
            )
        )
        result = dispatcher.publish((
            m.Event(event_type="ordered", aggregate_id="dispatcher"),
            m.Event(event_type="ordered", aggregate_id="dispatcher"),
        ))
        tm.that(result.failure)
        tm.that(result.exception is cause)
        tm.that(result.error_code, eq=failure.error_code)
        tm.that(calls, eq=["first", "failed"])

    @staticmethod
    def test_reentrant_registration_is_visible_only_to_the_next_snapshot() -> None:
        """Registration during publication affects only later snapshots."""
        dispatcher = u.build_dispatcher()
        calls: t.MutableSequenceOf[str] = []

        def later(_message: p.Routable) -> p.Result[t.JsonPayload]:
            calls.append("later")
            return r[t.JsonPayload].ok(value=True)

        def first(_message: p.Routable) -> p.Result[t.JsonPayload]:
            calls.append("first")
            tm.ok(
                dispatcher.register_handler(
                    later, message_type="snapshot", is_event=True
                )
            )
            return r[t.JsonPayload].ok(value=True)

        tm.ok(
            dispatcher.register_handler(first, message_type="snapshot", is_event=True)
        )
        event = m.Event(event_type="snapshot", aggregate_id="dispatcher")
        tm.ok(dispatcher.publish((event, event)))
        tm.that(calls, eq=["first", "first"])
        tm.ok(dispatcher.publish(event))
        tm.that(calls, eq=["first", "first", "first", "later"])

    @staticmethod
    def test_concurrent_command_writers_never_overwrite_the_winner() -> None:
        """Concurrent registrations elect exactly one command handler."""
        dispatcher = u.build_dispatcher()

        def left(_message: p.Routable) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok("left")

        def right(_message: p.Routable) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok("right")

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                dispatcher.register_handler, left, message_type="concurrent"
            )
            second = pool.submit(
                dispatcher.register_handler, right, message_type="concurrent"
            )
            outcomes = (first.result(), second.result())
        tm.that(sum(result.success for result in outcomes), eq=1)
        tm.that(
            tm.ok(dispatcher.dispatch(m.Command(command_type="concurrent"))),
            eq="left" if outcomes[0].success else "right",
        )

    @staticmethod
    @pytest.mark.parametrize("auto_first", [False, True])
    def test_auto_and_explicit_subscribers_follow_registration_order(
        *, auto_first: bool
    ) -> None:
        """Automatic and explicit subscribers retain registration order."""
        dispatcher = u.build_dispatcher()
        auto = u.Tests.AutoSubscriber()
        observed: t.MutableSequenceOf[t.StrSequence] = []

        def explicit(_message: p.Routable) -> p.Result[t.JsonPayload]:
            observed.append(tuple(auto.handled))
            return r[t.JsonPayload].ok(value=True)

        auto_binding = m.HandlerRegistration(handler=auto, is_event=True)
        explicit_binding = m.HandlerRegistration(
            handler=explicit, message_type="ordered", is_event=True
        )
        tm.ok(
            dispatcher.register_handlers(
                (auto_binding, explicit_binding)
                if auto_first
                else (explicit_binding, auto_binding),
            )
        )
        tm.ok(
            dispatcher.publish(m.Event(event_type="ordered", aggregate_id="dispatcher"))
        )
        tm.that(observed, eq=[("ordered",)] if auto_first else [()])
        tm.that(auto.handled, eq=["ordered"])
