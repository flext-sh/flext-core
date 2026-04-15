"""Dispatcher DSL example with golden-file validation."""

from __future__ import annotations

from collections.abc import MutableSequence
from logging import ERROR
from typing import cast, override

from examples import (
    Ex04AutoCommand,
    Ex04CreateUser,
    Ex04DeleteUser,
    Ex04FailingDelete,
    Ex04GetUser,
    Ex04NoSubscriberEvent,
    Ex04Ping,
    Ex04UnknownQuery,
    Ex04UserCreated,
    Examples,
    p,
    r,
    t,
    u,
)


class Ex04DispatchDsl(Examples):
    """Golden-file tests for the canonical dispatcher DSL."""

    class CreateUserHandler:
        """Handle handler for CreateUser commands."""

        message_type: type[p.Routable]

        def __init__(self) -> None:
            """Bind handler to CreateUser message type."""
            self.message_type = Ex04CreateUser

        def handle(self, message: p.Routable) -> t.ScalarOrModel:
            """Create a deterministic response for CreateUser."""
            typed_message = Ex04CreateUser.model_validate(message)
            return f"created:{typed_message.username}"

    class GetUserDispatcher:
        """DispatchMessage handler for GetUser queries."""

        message_type: type[p.Routable]

        def __init__(self) -> None:
            """Bind dispatcher to GetUser query type."""
            self.message_type = Ex04GetUser

        def dispatch_message(
            self,
            message: p.Routable,
            operation: str = "dispatch_message",
        ) -> str:
            """Return deterministic user payload for GetUser."""
            _ = operation
            typed_message = Ex04GetUser.model_validate(message)
            return f"active:{typed_message.username}"

    class DeleteExecutor:
        """Execute handler for DeleteUser commands."""

        message_type: type[p.Routable]

        def __init__(self) -> None:
            """Bind executor to DeleteUser command type."""
            self.message_type = Ex04DeleteUser

        def execute(self, message: p.Routable) -> t.ScalarOrModel:
            """Create deterministic deletion output."""
            typed_message = Ex04DeleteUser.model_validate(message)
            return f"deleted:{typed_message.username}"

    class FailingDeleteCallable:
        """Callable handler that returns a failure result."""

        message_type: type[p.Routable]

        def __init__(self) -> None:
            """Bind callable to FailingDelete command type."""
            self.message_type = Ex04FailingDelete

        def __call__(self, message: p.Routable) -> p.Result[str]:
            """Reject deletion to exercise dispatcher failure handling."""
            typed_message = Ex04FailingDelete.model_validate(message)
            return r[str].fail_op(
                "delete user",
                f"deletion blocked for {typed_message.username}",
            )

    class PingCallable:
        """Callable handler that returns a bare value."""

        message_type: type[p.Routable]

        def __init__(self) -> None:
            """Bind callable to Ping command type."""
            self.message_type = Ex04Ping

        def __call__(self, message: p.Routable) -> str:
            """Return a bare pong value to test automatic wrapping."""
            typed_message = Ex04Ping.model_validate(message)
            return f"pong:{typed_message.value}"

    class AutoHandler:
        """Auto-discovery handler selected through can_handle."""

        def can_handle(self, message: type | p.Routable) -> bool:
            """Report support for AutoCommand class or instance."""
            return bool(message)

        def handle(self, message: p.Routable) -> t.ScalarOrModel:
            """Handle discovered command and return a synthetic payload."""
            typed_message = Ex04AutoCommand.model_validate(message)
            return f"auto:{typed_message.payload}"

    class UserCreatedSubscriber:
        """Event subscriber implementing Handle."""

        message_type: str

        def __init__(self) -> None:
            """Create an in-memory event sink."""
            self.message_type = "user_created"
            self.events: MutableSequence[str] = []

        def handle(self, message: p.Routable) -> t.ScalarOrModel:
            """Store event entries when receiving UserCreated."""
            typed_message = Ex04UserCreated.model_validate(message)
            self.events.append(f"user:{typed_message.username}")
            return True

    class AuditSubscriber:
        """Event subscriber implementing DispatchMessage."""

        message_type: str

        def __init__(self) -> None:
            """Create an in-memory audit sink."""
            self.message_type = "user_created"
            self.events: MutableSequence[str] = []

        def dispatch_message(
            self,
            message: p.Routable,
            operation: str = "dispatch_message",
        ) -> t.ScalarOrModel:
            """Store audit entries when receiving UserCreated."""
            _ = operation
            typed_message = Ex04UserCreated.model_validate(message)
            self.events.append(f"audit:{typed_message.username}")
            return True


class _Ex04Exercise(Ex04DispatchDsl):
    """Exercise runner — separated to keep bindings above the exercise logic."""

    @override
    def exercise(self) -> None:
        """Run all scenarios and record deterministic golden output."""
        u.configure_structlog(log_level=ERROR)
        self._exercise_register_and_dispatch()
        self._exercise_auto_discovery()
        self._exercise_error_cases()
        self._exercise_event_publishing()

    def _exercise_register_and_dispatch(self) -> None:
        """Cover constructor, register_handler and dispatch happy paths."""
        self.section("register_and_dispatch")
        dispatcher = u.build_dispatcher()
        self.check("constructor.protocol", isinstance(dispatcher, p.Dispatcher))
        reg_handle = dispatcher.register_handler(
            cast("p.Handle", self.CreateUserHandler())
        )
        self.check("register(Handle).is_success", reg_handle.success)
        reg_dispatch_msg = dispatcher.register_handler(
            cast("p.DispatchMessage", self.GetUserDispatcher()),
            is_event=False,
        )
        self.check("register(DispatchMessage).is_success", reg_dispatch_msg.success)
        reg_execute = dispatcher.register_handler(
            cast("p.Execute", self.DeleteExecutor())
        )
        self.check("register(Execute).is_success", reg_execute.success)
        reg_callable = dispatcher.register_handler(self.PingCallable())
        self.check("register(callable).is_success", reg_callable.success)
        create_r = dispatcher.dispatch(Ex04CreateUser(username="alice"))
        self.check("dispatch(command).is_success", create_r.success)
        self.check("dispatch(command).value", create_r.value)
        get_r = dispatcher.dispatch(Ex04GetUser(username="alice"))
        self.check("dispatch(query).is_success", get_r.success)
        self.check("dispatch(query).value", get_r.value)
        delete_r = dispatcher.dispatch(Ex04DeleteUser(username="alice"))
        self.check("dispatch(execute).is_success", delete_r.success)
        self.check("dispatch(execute).value", delete_r.value)
        ping_r = dispatcher.dispatch(Ex04Ping(value="x"))
        self.check("dispatch(callable).is_success", ping_r.success)
        self.check("dispatch(callable).value", ping_r.value)

    def _exercise_auto_discovery(self) -> None:
        """Cover can_handle route discovery for dispatch fallback."""
        self.section("auto_discovery")
        dispatcher = u.build_dispatcher()
        reg_auto = dispatcher.register_handler(
            cast("p.AutoDiscoverableHandler", self.AutoHandler()),
        )
        self.check("register(can_handle).is_success", reg_auto.success)
        auto_r = dispatcher.dispatch(Ex04AutoCommand(payload="fallback"))
        self.check("dispatch(auto_discovery).is_success", auto_r.success)
        self.check("dispatch(auto_discovery).value", auto_r.value)

    def _exercise_error_cases(self) -> None:
        """Cover registration and dispatch failure paths."""
        self.section("error_cases")
        dispatcher = u.build_dispatcher()

        def _invalid_handler(message: p.Routable) -> str:
            """Return a value but lacks route metadata for registration."""
            return type(message).__name__

        reg_invalid = dispatcher.register_handler(_invalid_handler)
        self.check("register(no_route_attrs).is_failure", reg_invalid.failure)
        no_handler_r = dispatcher.dispatch(Ex04UnknownQuery(payload="none"))
        self.check("dispatch(no_handler).is_failure", no_handler_r.failure)
        reg_fail_handler = dispatcher.register_handler(self.FailingDeleteCallable())
        self.check("register(failing_callable).is_success", reg_fail_handler.success)
        failing_r = dispatcher.dispatch(Ex04FailingDelete(username="alice"))
        self.check("dispatch(handler_returns_fail).is_failure", failing_r.failure)

    def _exercise_event_publishing(self) -> None:
        """Cover event registration and publish paths."""
        self.section("event_publishing")
        dispatcher = u.build_dispatcher()
        subscriber = self.UserCreatedSubscriber()
        audit_subscriber = self.AuditSubscriber()
        reg_user = dispatcher.register_handler(
            cast("p.Handle", subscriber),
            is_event=True,
        )
        self.check("register(event_subscriber).is_success", reg_user.success)
        reg_audit = dispatcher.register_handler(
            cast("p.DispatchMessage", audit_subscriber),
            is_event=True,
        )
        self.check("register(audit_subscriber).is_success", reg_audit.success)
        pub_one = dispatcher.publish(Ex04UserCreated(username="alice"))
        self.check("publish(single).is_success", pub_one.success)
        pub_many = dispatcher.publish([
            Ex04UserCreated(username="bruno"),
            Ex04UserCreated(username="carla"),
        ])
        self.check("publish(list).is_success", pub_many.success)
        self.check("subscriber.events", subscriber.events)
        self.check("audit_subscriber.events", audit_subscriber.events)
        pub_none = dispatcher.publish(Ex04NoSubscriberEvent(marker="ok"))
        self.check("publish(no_subscribers).is_success", pub_none.success)
        self.check("publish(no_subscribers).value", pub_none.value)


if __name__ == "__main__":
    _Ex04Exercise(caller_file=__file__).run()
