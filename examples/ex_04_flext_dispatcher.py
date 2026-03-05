"""FlextDispatcher — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from flext_core import FlextDispatcher, r, u

_RESULTS: list[str] = []


def _check(label: str, value: object) -> None:
    _RESULTS.append(f"{label}: {_ser(value)}")


def _section(name: str) -> None:
    if _RESULTS:
        _RESULTS.append("")
    _RESULTS.append(f"[{name}]")


def _ser(v: object) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return repr(v)
    if u.is_list(v):
        return "[" + ", ".join(_ser(x) for x in v) + "]"
    if u.is_dict_like(v):
        pairs = ", ".join(
            f"{_ser(k)}: {_ser(val)}"
            for k, val in sorted(v.items(), key=lambda kv: str(kv[0]))
        )
        return "{" + pairs + "}"
    if isinstance(v, type):
        return v.__name__
    return type(v).__name__


def _verify() -> None:
    actual = "\n".join(_RESULTS).strip() + "\n"
    me = Path(__file__)
    expected_path = me.with_suffix(".expected")
    n = sum(1 for line in _RESULTS if ": " in line and not line.startswith("["))
    if expected_path.exists():
        expected = expected_path.read_text(encoding="utf-8")
        if actual == expected:
            sys.stdout.write(f"PASS: {me.stem} ({n} checks)\n")
        else:
            actual_path = me.with_suffix(".actual")
            actual_path.write_text(actual, encoding="utf-8")
            sys.stdout.write(
                f"FAIL: {me.stem} — diff {expected_path.name} {actual_path.name}\n"
            )
            sys.exit(1)
    else:
        expected_path.write_text(actual, encoding="utf-8")
        sys.stdout.write(f"GENERATED: {expected_path.name} ({n} checks)\n")


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

    def handle(self, message: object) -> str:
        """Return confirmation for a supported create command."""
        if isinstance(message, CreateUser):
            return f"created:{message.username}"
        return "created:"


class GetUserDispatcher:
    """DispatchMessageProtocol handler for GetUser queries."""

    message_type = GetUser

    def dispatch_message(self, message: object) -> Mapping[str, str]:
        """Return a synthetic user payload for supported query."""
        if isinstance(message, GetUser):
            return {"state": "active", "username": message.username}
        return {"state": "active", "username": ""}


class DeleteExecutor:
    """ExecuteProtocol handler for DeleteUser commands."""

    message_type = DeleteUser

    def execute(self, message: object) -> str:
        """Return deletion status for supported command."""
        if isinstance(message, DeleteUser):
            return f"deleted:{message.username}"
        return "deleted:"


class FailingDeleteCallable:
    """Callable handler that returns a failure result."""

    message_type = FailingDelete

    def __call__(self, message: object) -> r[str]:
        """Reject deletion to exercise dispatcher failure handling."""
        if isinstance(message, FailingDelete):
            return r[str].fail(f"deletion blocked for {message.username}")
        return r[str].fail("deletion blocked")


class PingCallable:
    """Callable handler that returns a bare value."""

    message_type = Ping

    def __call__(self, message: object) -> str:
        """Return a bare pong value to test automatic wrapping."""
        if isinstance(message, Ping):
            return f"pong:{message.value}"
        return "pong:"


class AutoHandler:
    """Auto-discovery handler selected through can_handle."""

    def can_handle(self, message: object) -> bool:
        """Report support for AutoCommand class or instance."""
        if isinstance(message, type):
            return issubclass(message, AutoCommand)
        return isinstance(message, AutoCommand)

    def handle(self, message: object) -> str:
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

    def handle(self, message: object) -> bool:
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

    def dispatch_message(self, message: object) -> bool:
        """Store audit entries when receiving UserCreated."""
        if isinstance(message, UserCreated):
            self.events.append(f"audit:{message.username}")
        return True


def invalid_handler(message: object) -> str:
    """Return a value but lacks route metadata for registration."""
    return type(message).__name__


def demo_register_and_dispatch() -> None:
    """Cover constructor, register_handler and dispatch happy paths."""
    _section("register_and_dispatch")

    dispatcher = FlextDispatcher()
    _check("constructor.type", type(dispatcher))

    reg_handle = dispatcher.register_handler(CreateUserHandler())
    _check("register(HandleProtocol).is_success", reg_handle.is_success)

    reg_dispatch_msg = dispatcher.register_handler(GetUserDispatcher(), is_event=False)
    _check("register(DispatchMessageProtocol).is_success", reg_dispatch_msg.is_success)

    reg_execute = dispatcher.register_handler(DeleteExecutor())
    _check("register(ExecuteProtocol).is_success", reg_execute.is_success)

    reg_callable = dispatcher.register_handler(PingCallable())
    _check("register(callable).is_success", reg_callable.is_success)

    create_r = dispatcher.dispatch(CreateUser(username="alice"))
    _check("dispatch(command).is_success", create_r.is_success)
    _check("dispatch(command).value", create_r.value)

    get_r = dispatcher.dispatch(GetUser(username="alice"))
    _check("dispatch(query).is_success", get_r.is_success)
    _check("dispatch(query).value", get_r.value)

    delete_r = dispatcher.dispatch(DeleteUser(username="alice"))
    _check("dispatch(execute).is_success", delete_r.is_success)
    _check("dispatch(execute).value", delete_r.value)

    ping_r = dispatcher.dispatch(Ping(value="x"))
    _check("dispatch(callable).is_success", ping_r.is_success)
    _check("dispatch(callable).value", ping_r.value)


def demo_auto_discovery() -> None:
    """Cover can_handle route discovery for dispatch fallback."""
    _section("auto_discovery")

    dispatcher = FlextDispatcher()

    reg_auto = dispatcher.register_handler(AutoHandler())
    _check("register(can_handle).is_success", reg_auto.is_success)

    auto_r = dispatcher.dispatch(AutoCommand(payload="fallback"))
    _check("dispatch(auto_discovery).is_success", auto_r.is_success)
    _check("dispatch(auto_discovery).value", auto_r.value)


def demo_error_cases() -> None:
    """Cover registration and dispatch failure paths."""
    _section("error_cases")

    dispatcher = FlextDispatcher()

    reg_invalid = dispatcher.register_handler(invalid_handler)
    _check("register(no_route_attrs).is_failure", reg_invalid.is_failure)

    no_handler_r = dispatcher.dispatch(UnknownQuery(payload="none"))
    _check("dispatch(no_handler).is_failure", no_handler_r.is_failure)

    reg_fail_handler = dispatcher.register_handler(FailingDeleteCallable())
    _check("register(failing_callable).is_success", reg_fail_handler.is_success)

    failing_r = dispatcher.dispatch(FailingDelete(username="alice"))
    _check("dispatch(handler_returns_fail).is_failure", failing_r.is_failure)


def demo_event_publishing() -> None:
    """Cover event registration and publish paths."""
    _section("event_publishing")

    dispatcher = FlextDispatcher()
    subscriber = UserCreatedSubscriber()
    audit_subscriber = AuditSubscriber()

    reg_user = dispatcher.register_handler(subscriber, is_event=True)
    _check("register(event_subscriber).is_success", reg_user.is_success)

    reg_audit = dispatcher.register_handler(audit_subscriber, is_event=True)
    _check("register(audit_subscriber).is_success", reg_audit.is_success)

    pub_one = dispatcher.publish(UserCreated(username="alice"))
    _check("publish(single).is_success", pub_one.is_success)

    pub_many = dispatcher.publish(
        [
            UserCreated(username="bruno"),
            UserCreated(username="carla"),
        ]
    )
    _check("publish(list).is_success", pub_many.is_success)

    _check("subscriber.events", subscriber.events)
    _check("audit_subscriber.events", audit_subscriber.events)

    pub_none = dispatcher.publish(NoSubscriberEvent(marker="ok"))
    _check("publish(no_subscribers).is_success", pub_none.is_success)
    _check("publish(no_subscribers).value", pub_none.value)


def main() -> None:
    """Run all scenarios and verify against golden file."""
    demo_register_and_dispatch()
    demo_auto_discovery()
    demo_error_cases()
    demo_event_publishing()
    _verify()


if __name__ == "__main__":
    main()
