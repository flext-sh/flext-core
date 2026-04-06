"""FlextDispatcher — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import override

from pydantic import BaseModel

from examples import Examples, m, p, t
from flext_core import FlextDispatcher, r


class Ex04FlextDispatcher(Examples):
    """Golden-file tests for ``FlextDispatcher`` public API."""

    class CreateUser(m.Command):
        """Command model for user creation."""

        username: str

    class GetUser(m.Query):
        """Query model for user retrieval."""

        username: str

    class DeleteUser(m.Command):
        """Command model for user deletion."""

        username: str

    class FailingDelete(m.Command):
        """Command model that intentionally fails."""

        username: str

    class AutoCommand(m.Command):
        """Command routed through can_handle auto-discovery."""

        payload: str

    class Ping(m.Command):
        """Command handled by a callable returning plain value."""

        value: str

    class UnknownQuery(m.Query):
        """Query model with no registered handler."""

        payload: str

    class UserCreated(m.Event):
        """Event model published to subscribers."""

        username: str
        event_type: str = "user_created"
        aggregate_id: str = "users"

    class NoSubscriberEvent(m.Event):
        """Event model without registered subscribers."""

        marker: str
        event_type: str = "no_subscribers"
        aggregate_id: str = "events"

    class CreateUserHandler:
        """Handle handler for CreateUser commands."""

        message_type: type[p.Routable]

        def __init__(self) -> None:
            """Bind handler to CreateUser message type."""
            self.message_type = Ex04FlextDispatcher.CreateUser

        def handle(self, message: p.Routable) -> t.Container | BaseModel:
            """Create a deterministic response for CreateUser."""
            typed_message = Ex04FlextDispatcher.CreateUser.model_validate(message)
            return f"created:{typed_message.username}"

    class GetUserDispatcher:
        """DispatchMessage handler for GetUser queries."""

        message_type: type[p.Routable]

        def __init__(self) -> None:
            """Bind dispatcher to GetUser query type."""
            self.message_type = Ex04FlextDispatcher.GetUser

        def dispatch_message(self, message: p.Routable) -> t.Container | BaseModel:
            """Return deterministic user payload for GetUser."""
            typed_message = Ex04FlextDispatcher.GetUser.model_validate(message)
            return t.ConfigMap(
                root={"state": "active", "username": typed_message.username},
            )

    class DeleteExecutor:
        """Execute handler for DeleteUser commands."""

        message_type: type[p.Routable]

        def __init__(self) -> None:
            """Bind executor to DeleteUser command type."""
            self.message_type = Ex04FlextDispatcher.DeleteUser

        def execute(self, message: p.Routable) -> t.Container | BaseModel:
            """Create deterministic deletion output."""
            typed_message = Ex04FlextDispatcher.DeleteUser.model_validate(message)
            return f"deleted:{typed_message.username}"

    class FailingDeleteCallable:
        """Callable handler that returns a failure result."""

        message_type: type[p.Routable]

        def __init__(self) -> None:
            """Bind callable to FailingDelete command type."""
            self.message_type = Ex04FlextDispatcher.FailingDelete

        def __call__(self, message: p.Routable) -> r[str]:
            """Reject deletion to exercise dispatcher failure handling."""
            typed_message = Ex04FlextDispatcher.FailingDelete.model_validate(message)
            return r[str].fail(f"deletion blocked for {typed_message.username}")

    class PingCallable:
        """Callable handler that returns a bare value."""

        message_type: type[p.Routable]

        def __init__(self) -> None:
            """Bind callable to Ping command type."""
            self.message_type = Ex04FlextDispatcher.Ping

        def __call__(self, message: p.Routable) -> str:
            """Return a bare pong value to test automatic wrapping."""
            typed_message = Ex04FlextDispatcher.Ping.model_validate(message)
            return f"pong:{typed_message.value}"

    class AutoHandler:
        """Auto-discovery handler selected through can_handle."""

        def can_handle(self, message: type | p.Routable) -> bool:
            """Report support for AutoCommand class or instance."""
            return bool(message)

        def handle(self, message: p.Routable) -> t.Container | BaseModel:
            """Handle discovered command and return a synthetic payload."""
            typed_message = Ex04FlextDispatcher.AutoCommand.model_validate(message)
            return f"auto:{typed_message.payload}"

    class UserCreatedSubscriber:
        """Event subscriber implementing Handle."""

        event_type: type[p.Routable]

        def __init__(self) -> None:
            """Create an in-memory event sink."""
            self.event_type = Ex04FlextDispatcher.UserCreated
            self.events: MutableSequence[str] = []

        def handle(self, message: p.Routable) -> t.Container | BaseModel:
            """Store event entries when receiving UserCreated."""
            typed_message = Ex04FlextDispatcher.UserCreated.model_validate(message)
            self.events.append(f"user:{typed_message.username}")
            return True

    class AuditSubscriber:
        """Event subscriber implementing DispatchMessage."""

        event_type: type[p.Routable]

        def __init__(self) -> None:
            """Create an in-memory audit sink."""
            self.event_type = Ex04FlextDispatcher.UserCreated
            self.events: MutableSequence[str] = []

        def dispatch_message(self, message: p.Routable) -> t.Container | BaseModel:
            """Store audit entries when receiving UserCreated."""
            typed_message = Ex04FlextDispatcher.UserCreated.model_validate(message)
            self.events.append(f"audit:{typed_message.username}")
            return True


class _Ex04Exercise(Ex04FlextDispatcher):
    """Exercise runner — separated to keep bindings above the exercise logic."""

    @override
    def exercise(self) -> None:
        """Run all scenarios and record deterministic golden output."""
        self._exercise_register_and_dispatch()
        self._exercise_auto_discovery()
        self._exercise_error_cases()
        self._exercise_event_publishing()

    def _exercise_register_and_dispatch(self) -> None:
        """Cover constructor, register_handler and dispatch happy paths."""
        self.section("register_and_dispatch")
        dispatcher = FlextDispatcher()
        self.check("constructor.type", type(dispatcher).__name__)
        reg_handle = dispatcher.register_handler(self.CreateUserHandler())
        self.check("register(Handle).is_success", reg_handle.is_success)
        reg_dispatch_msg = dispatcher.register_handler(
            self.GetUserDispatcher(),
            is_event=False,
        )
        self.check("register(DispatchMessage).is_success", reg_dispatch_msg.is_success)
        reg_execute = dispatcher.register_handler(self.DeleteExecutor())
        self.check("register(Execute).is_success", reg_execute.is_success)
        reg_callable = dispatcher.register_handler(self.PingCallable())
        self.check("register(callable).is_success", reg_callable.is_success)
        create_r = dispatcher.dispatch(self.CreateUser(username="alice"))
        self.check("dispatch(command).is_success", create_r.is_success)
        self.check("dispatch(command).value", create_r.value)
        get_r = dispatcher.dispatch(self.GetUser(username="alice"))
        self.check("dispatch(query).is_success", get_r.is_success)
        self.check("dispatch(query).value", get_r.value)
        delete_r = dispatcher.dispatch(self.DeleteUser(username="alice"))
        self.check("dispatch(execute).is_success", delete_r.is_success)
        self.check("dispatch(execute).value", delete_r.value)
        ping_r = dispatcher.dispatch(self.Ping(value="x"))
        self.check("dispatch(callable).is_success", ping_r.is_success)
        self.check("dispatch(callable).value", ping_r.value)

    def _exercise_auto_discovery(self) -> None:
        """Cover can_handle route discovery for dispatch fallback."""
        self.section("auto_discovery")
        dispatcher = FlextDispatcher()
        reg_auto = dispatcher.register_handler(self.AutoHandler())
        self.check("register(can_handle).is_success", reg_auto.is_success)
        auto_r = dispatcher.dispatch(self.AutoCommand(payload="fallback"))
        self.check("dispatch(auto_discovery).is_success", auto_r.is_success)
        self.check("dispatch(auto_discovery).value", auto_r.value)

    def _exercise_error_cases(self) -> None:
        """Cover registration and dispatch failure paths."""
        self.section("error_cases")
        dispatcher = FlextDispatcher()

        def _invalid_handler(message: p.Routable) -> str:
            """Return a value but lacks route metadata for registration."""
            return type(message).__name__

        reg_invalid = dispatcher.register_handler(_invalid_handler)
        self.check("register(no_route_attrs).is_failure", reg_invalid.is_failure)
        no_handler_r = dispatcher.dispatch(self.UnknownQuery(payload="none"))
        self.check("dispatch(no_handler).is_failure", no_handler_r.is_failure)
        reg_fail_handler = dispatcher.register_handler(self.FailingDeleteCallable())
        self.check("register(failing_callable).is_success", reg_fail_handler.is_success)
        failing_r = dispatcher.dispatch(self.FailingDelete(username="alice"))
        self.check("dispatch(handler_returns_fail).is_failure", failing_r.is_failure)

    def _exercise_event_publishing(self) -> None:
        """Cover event registration and publish paths."""
        self.section("event_publishing")
        dispatcher = FlextDispatcher()
        subscriber = self.UserCreatedSubscriber()
        audit_subscriber = self.AuditSubscriber()
        reg_user = dispatcher.register_handler(subscriber, is_event=True)
        self.check("register(event_subscriber).is_success", reg_user.is_success)
        reg_audit = dispatcher.register_handler(audit_subscriber, is_event=True)
        self.check("register(audit_subscriber).is_success", reg_audit.is_success)
        pub_one = dispatcher.publish(self.UserCreated(username="alice"))
        self.check("publish(single).is_success", pub_one.is_success)
        pub_many = dispatcher.publish([
            self.UserCreated(username="bruno"),
            self.UserCreated(username="carla"),
        ])
        self.check("publish(list).is_success", pub_many.is_success)
        self.check("subscriber.events", subscriber.events)
        self.check("audit_subscriber.events", audit_subscriber.events)
        pub_none = dispatcher.publish(self.NoSubscriberEvent(marker="ok"))
        self.check("publish(no_subscribers).is_success", pub_none.is_success)
        self.check("publish(no_subscribers).value", pub_none.value)


if __name__ == "__main__":
    _Ex04Exercise(__file__).run()
