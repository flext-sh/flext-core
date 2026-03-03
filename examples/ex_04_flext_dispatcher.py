"""FlextDispatcher — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from shared import Examples

from flext_core import FlextDispatcher, r, t


@dataclass(slots=True)
class CreateUser:
    """Command model for user creation."""

    command_type: ClassVar[str] = "create_user"
    username: str


@dataclass(slots=True)
class GetUser:
    """Query model for user retrieval."""

    query_type: ClassVar[str] = "get_user"
    username: str


@dataclass(slots=True)
class DeleteUser:
    """Command model for user deletion."""

    command_type: ClassVar[str] = "delete_user"
    username: str


@dataclass(slots=True)
class FailingDelete:
    """Command model that intentionally fails."""

    command_type: ClassVar[str] = "failing_delete"
    username: str


@dataclass(slots=True)
class AutoCommand:
    """Command routed through can_handle auto-discovery."""

    command_type: ClassVar[str] = "auto_command"
    payload: str


@dataclass(slots=True)
class Ping:
    """Command handled by a callable returning plain value."""

    command_type: ClassVar[str] = "ping"
    value: str


@dataclass(slots=True)
class UnknownQuery:
    """Query model with no registered handler."""

    query_type: ClassVar[str] = "unknown_query"
    payload: str


@dataclass(slots=True)
class UserCreated:
    """Event model published to subscribers."""

    event_type: ClassVar[str] = "user_created"
    username: str


@dataclass(slots=True)
class NoSubscriberEvent:
    """Event model without registered subscribers."""

    event_type: ClassVar[str] = "no_subscribers"
    marker: str


class CreateUserHandler:
    """HandleProtocol handler for CreateUser commands."""

    message_type = CreateUser

    def handle(self, message: t.ContainerValue) -> str:
        """Return confirmation for a supported create command."""
        if isinstance(message, CreateUser):
            return f"created:{message.username}"
        return "created:"


class GetUserDispatcher:
    """DispatchMessageProtocol handler for GetUser queries."""

    message_type = GetUser

    def dispatch_message(self, message: t.ContainerValue) -> dict[str, str]:
        """Return a synthetic user payload for supported query."""
        if isinstance(message, GetUser):
            return {"state": "active", "username": message.username}
        return {"state": "active", "username": ""}


class DeleteExecutor:
    """ExecuteProtocol handler for DeleteUser commands."""

    message_type = DeleteUser

    def execute(self, message: t.ContainerValue) -> str:
        """Return deletion status for supported command."""
        if isinstance(message, DeleteUser):
            return f"deleted:{message.username}"
        return "deleted:"


class FailingDeleteCallable:
    """Callable handler that returns a failure result."""

    message_type = FailingDelete

    def __call__(self, message: t.ContainerValue) -> r[str]:
        """Reject deletion to exercise dispatcher failure handling."""
        if isinstance(message, FailingDelete):
            return r[str].fail(f"deletion blocked for {message.username}")
        return r[str].fail("deletion blocked")


class PingCallable:
    """Callable handler that returns a bare value."""

    message_type = Ping

    def __call__(self, message: t.ContainerValue) -> str:
        """Return a bare pong value to test automatic wrapping."""
        if isinstance(message, Ping):
            return f"pong:{message.value}"
        return "pong:"


class AutoHandler:
    """Auto-discovery handler selected through can_handle."""

    def can_handle(self, message: t.ContainerValue) -> bool:
        """Report support for AutoCommand class or instance."""
        if isinstance(message, type):
            return issubclass(message, AutoCommand)
        return isinstance(message, AutoCommand)

    def handle(self, message: t.ContainerValue) -> str:
        """Handle discovered command and return a synthetic payload."""
        if isinstance(message, AutoCommand):
            return f"auto:{message.payload}"
        return "auto:"


class UserCreatedSubscriber:
    """Event subscriber implementing HandleProtocol."""

    event_type = UserCreated

    def __init__(self) -> None:
        """Create an in-memory event sink."""
        self.events: list[str] = []

    def handle(self, message: t.ContainerValue) -> bool:
        """Store event entries when receiving UserCreated."""
        if isinstance(message, UserCreated):
            self.events.append(f"user:{message.username}")
        return True


class AuditSubscriber:
    """Event subscriber implementing DispatchMessageProtocol."""

    event_type = UserCreated

    def __init__(self) -> None:
        """Create an in-memory audit sink."""
        self.events: list[str] = []

    def dispatch_message(self, message: t.ContainerValue) -> bool:
        """Store audit entries when receiving UserCreated."""
        if isinstance(message, UserCreated):
            self.events.append(f"audit:{message.username}")
        return True


def invalid_handler(message: t.ContainerValue) -> str:
    """Return a value but lacks route metadata for registration."""
    return type(message).__name__


class Ex04FlextDispatcher(Examples):
    """Exercise FlextDispatcher public API."""

    def exercise(self) -> None:
        """Run all scenarios and verify against golden file."""
        self._exercise_register_and_dispatch()
        self._exercise_auto_discovery()
        self._exercise_error_cases()
        self._exercise_event_publishing()

    def _exercise_register_and_dispatch(self) -> None:
        """Cover constructor, register_handler and dispatch happy paths."""
        self.section("register_and_dispatch")

        dispatcher = FlextDispatcher()
        self.check("constructor.type", type(dispatcher).__name__)

        reg_handle = dispatcher.register_handler(CreateUserHandler())
        self.check("register(HandleProtocol).is_success", reg_handle.is_success)

        reg_dispatch_msg = dispatcher.register_handler(
            GetUserDispatcher(), is_event=False
        )
        self.check(
            "register(DispatchMessageProtocol).is_success", reg_dispatch_msg.is_success
        )

        reg_execute = dispatcher.register_handler(DeleteExecutor())
        self.check("register(ExecuteProtocol).is_success", reg_execute.is_success)

        reg_callable = dispatcher.register_handler(PingCallable())
        self.check("register(callable).is_success", reg_callable.is_success)

        username = self.rand_str(6)
        ping_value = self.rand_str(5)

        create_r = dispatcher.dispatch(CreateUser(username=username))
        self.check("dispatch(command).is_success", create_r.is_success)
        self.check("dispatch(command).value", create_r.value)
        self.check(
            "dispatch(command).value_matches", create_r.value == f"created:{username}"
        )

        get_r = dispatcher.dispatch(GetUser(username=username))
        self.check("dispatch(query).is_success", get_r.is_success)
        self.check("dispatch(query).value", get_r.value)
        self.check(
            "dispatch(query).value_matches",
            get_r.value == {"state": "active", "username": username},
        )

        delete_r = dispatcher.dispatch(DeleteUser(username=username))
        self.check("dispatch(execute).is_success", delete_r.is_success)
        self.check("dispatch(execute).value", delete_r.value)
        self.check(
            "dispatch(execute).value_matches", delete_r.value == f"deleted:{username}"
        )

        ping_r = dispatcher.dispatch(Ping(value=ping_value))
        self.check("dispatch(callable).is_success", ping_r.is_success)
        self.check("dispatch(callable).value", ping_r.value)
        self.check(
            "dispatch(callable).value_matches", ping_r.value == f"pong:{ping_value}"
        )

    def _exercise_auto_discovery(self) -> None:
        """Cover can_handle route discovery for dispatch fallback."""
        self.section("auto_discovery")

        dispatcher = FlextDispatcher()

        reg_auto = dispatcher.register_handler(AutoHandler())
        self.check("register(can_handle).is_success", reg_auto.is_success)

        payload = self.rand_str(7)
        auto_r = dispatcher.dispatch(AutoCommand(payload=payload))
        self.check("dispatch(auto_discovery).is_success", auto_r.is_success)
        self.check("dispatch(auto_discovery).value", auto_r.value)
        self.check(
            "dispatch(auto_discovery).value_matches", auto_r.value == f"auto:{payload}"
        )

    def _exercise_error_cases(self) -> None:
        """Cover registration and dispatch failure paths."""
        self.section("error_cases")

        dispatcher = FlextDispatcher()

        reg_invalid = dispatcher.register_handler(invalid_handler)
        self.check("register(no_route_attrs).is_failure", reg_invalid.is_failure)

        unknown_payload = self.rand_str(6)
        no_handler_r = dispatcher.dispatch(UnknownQuery(payload=unknown_payload))
        self.check("dispatch(no_handler).is_failure", no_handler_r.is_failure)
        self.check("dispatch(no_handler).has_error", no_handler_r.error is not None)

        reg_fail_handler = dispatcher.register_handler(FailingDeleteCallable())
        self.check("register(failing_callable).is_success", reg_fail_handler.is_success)

        failing_username = self.rand_str(6)
        failing_r = dispatcher.dispatch(FailingDelete(username=failing_username))
        self.check("dispatch(handler_returns_fail).is_failure", failing_r.is_failure)
        self.check(
            "dispatch(handler_returns_fail).error_matches",
            failing_r.error == f"deletion blocked for {failing_username}",
        )

    def _exercise_event_publishing(self) -> None:
        """Cover event registration and publish paths."""
        self.section("event_publishing")

        dispatcher = FlextDispatcher()
        subscriber = UserCreatedSubscriber()
        audit_subscriber = AuditSubscriber()

        reg_user = dispatcher.register_handler(subscriber, is_event=True)
        self.check("register(event_subscriber).is_success", reg_user.is_success)

        reg_audit = dispatcher.register_handler(audit_subscriber, is_event=True)
        self.check("register(audit_subscriber).is_success", reg_audit.is_success)

        username_one = self.rand_str(6)
        username_two = self.rand_str(6)
        username_three = self.rand_str(6)
        pub_one = dispatcher.publish(UserCreated(username=username_one))
        self.check("publish(single).is_success", pub_one.is_success)
        self.check("publish(single).value", pub_one.value)

        pub_many = dispatcher.publish([
            UserCreated(username=username_two),
            UserCreated(username=username_three),
        ])
        self.check("publish(list).is_success", pub_many.is_success)
        self.check("publish(list).value", pub_many.value)

        self.check("subscriber.events", subscriber.events)
        self.check("audit_subscriber.events", audit_subscriber.events)
        self.check(
            "subscriber.events_matches",
            subscriber.events
            == [
                f"user:{username_one}",
                f"user:{username_two}",
                f"user:{username_three}",
            ],
        )
        self.check(
            "audit_subscriber.events_matches",
            audit_subscriber.events
            == [
                f"audit:{username_one}",
                f"audit:{username_two}",
                f"audit:{username_three}",
            ],
        )

        marker = self.rand_str(4)
        pub_none = dispatcher.publish(NoSubscriberEvent(marker=marker))
        self.check("publish(no_subscribers).is_success", pub_none.is_success)
        self.check("publish(no_subscribers).value", pub_none.value)


if __name__ == "__main__":
    Ex04FlextDispatcher(__file__).run()
