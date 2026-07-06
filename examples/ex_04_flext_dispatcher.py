"""Dispatcher example exercising the public routing APIs with real handlers."""

from __future__ import annotations

from pathlib import Path

from examples import ExamplesFlextShared, m, p, r, u


class _CreateUserHandler:
    message_type = m.Examples.CreateUser

    def handle(self, message: p.Routable) -> p.Result[str]:
        if not isinstance(message, m.Examples.CreateUser):
            return r[str].fail("unexpected_message")
        return r[str].ok(f"created:{message.username}")


class _GetUserHandler:
    message_type = m.Examples.GetUser

    def dispatch_message(
        self,
        message: p.Routable,
        operation: str = "dispatch",
    ) -> p.Result[str]:
        if not isinstance(message, m.Examples.GetUser):
            return r[str].fail(f"{operation}:unexpected_message")
        return r[str].ok(f"active:{message.username}")


class _DeleteUserHandler:
    message_type = m.Examples.DeleteUser

    def execute(self, message: p.Routable) -> p.Result[str]:
        if not isinstance(message, m.Examples.DeleteUser):
            return r[str].fail("unexpected_message")
        return r[str].ok(f"deleted:{message.username}")


class _AutoFallbackHandler:
    def can_handle(self, message_type: type) -> bool:
        return message_type is m.Examples.UnknownQuery

    def handle(self, message: p.Routable) -> p.Result[str]:
        if not isinstance(message, m.Examples.UnknownQuery):
            return r[str].fail("unexpected_message")
        return r[str].ok("auto:fallback")


class _EventSubscriber:
    message_type = m.Examples.UserCreated

    def __init__(self) -> None:
        self.events: list[str] = []

    def handle(self, message: p.Routable) -> p.Result[bool]:
        if not isinstance(message, m.Examples.UserCreated):
            return r[bool].fail("unexpected_message")
        self.events.append(message.username)
        return r[bool].ok(True)


class _AuditSubscriber:
    message_type = m.Examples.UserCreated

    def __init__(self) -> None:
        self.events: list[str] = []

    def handle(self, message: p.Routable) -> p.Result[bool]:
        if not isinstance(message, m.Examples.UserCreated):
            return r[bool].fail("unexpected_message")
        self.events.append(f"audit:{message.username}")
        return r[bool].ok(True)


class _PingHandler:
    message_type = m.Examples.Ping

    def __call__(self, message: p.Routable) -> p.Result[str]:
        if not isinstance(message, m.Examples.Ping):
            return r[str].fail("unexpected_message")
        return r[str].ok(f"pong:{message.value}")


class _FailingDeleteHandler:
    message_type = m.Examples.FailingDelete

    def __call__(self, message: p.Routable) -> p.Result[str]:
        if not isinstance(message, m.Examples.FailingDelete):
            return r[str].fail("unexpected_message")
        return r[str].fail("delete_failed")


def _no_route_handler(message: p.Routable) -> p.Result[str]:
    _ = message
    return r[str].ok("no-route")


class Ex04DispatchDsl:
    """Public dispatcher DSL example used by docs and README snippets."""

    @staticmethod
    def build_dispatcher() -> p.Dispatcher:
        """Create a dispatcher populated with the example handlers."""
        dispatcher = u.build_dispatcher()
        _ = dispatcher.register_handler(_CreateUserHandler())
        _ = dispatcher.register_handler(_GetUserHandler())
        _ = dispatcher.register_handler(_DeleteUserHandler())
        _ = dispatcher.register_handler(_PingHandler())
        return dispatcher

    @classmethod
    def run(cls) -> p.Result[str]:
        """Dispatch a real ping command through the public dispatcher."""
        dispatcher = cls.build_dispatcher()
        result = dispatcher.dispatch(m.Examples.Ping(value="dispatcher-example"))
        if result.failure:
            return r[str].fail(result.error or "dispatcher example failed")
        return r[str].ok(str(result.value))


class _Ex04DispatchGolden(ExamplesFlextShared):
    """Golden-file harness for the dispatcher example."""

    def exercise(self) -> None:
        """Exercise handler registration, dispatching, auto-discovery, and events."""
        dispatcher = Ex04DispatchDsl.build_dispatcher()

        self.section("register_and_dispatch")
        self.audit_check(
            "constructor.protocol",
            type(dispatcher).__name__ == "FlextDispatcher",
        )
        self.audit_check(
            "register(Handle).is_success",
            dispatcher.register_handler(_CreateUserHandler()).success,
        )
        self.audit_check(
            "register(DispatchMessage).is_success",
            dispatcher.register_handler(_GetUserHandler()).success,
        )
        self.audit_check(
            "register(Execute).is_success",
            dispatcher.register_handler(_DeleteUserHandler()).success,
        )
        self.audit_check(
            "register(callable).is_success",
            dispatcher.register_handler(_PingHandler()).success,
        )
        created = dispatcher.dispatch(m.Examples.CreateUser(username="alice"))
        fetched = dispatcher.dispatch(m.Examples.GetUser(username="alice"))
        deleted = dispatcher.dispatch(m.Examples.DeleteUser(username="alice"))
        pinged = dispatcher.dispatch(m.Examples.Ping(value="x"))
        self.audit_check("dispatch(command).is_success", created.success)
        self.audit_check("dispatch(command).value", created.unwrap_or(""))
        self.audit_check("dispatch(query).is_success", fetched.success)
        self.audit_check("dispatch(query).value", fetched.unwrap_or(""))
        self.audit_check("dispatch(execute).is_success", deleted.success)
        self.audit_check("dispatch(execute).value", deleted.unwrap_or(""))
        self.audit_check("dispatch(callable).is_success", pinged.success)
        self.audit_check("dispatch(callable).value", pinged.unwrap_or(""))

        self.section("auto_discovery")
        auto_discovery_registration = dispatcher.register_handler(
            _AutoFallbackHandler(),
        )
        auto_discovery = dispatcher.dispatch(m.Examples.UnknownQuery(payload="x"))
        self.audit_check(
            "register(can_handle).is_success",
            auto_discovery_registration.success,
        )
        self.audit_check("dispatch(auto_discovery).is_success", auto_discovery.success)
        self.audit_check("dispatch(auto_discovery).value", auto_discovery.unwrap_or(""))

        self.section("error_cases")
        no_route_registration = dispatcher.register_handler(_no_route_handler)
        no_handler = u.build_dispatcher().dispatch(
            m.Examples.GetUser(username="missing"),
        )
        failing_registration = dispatcher.register_handler(_FailingDeleteHandler())
        failing_dispatch = dispatcher.dispatch(
            m.Examples.FailingDelete(username="alice"),
        )
        self.audit_check(
            "register(no_route_attrs).is_failure",
            no_route_registration.failure,
        )
        self.audit_check("dispatch(no_handler).is_failure", no_handler.failure)
        self.audit_check(
            "register(failing_callable).is_success",
            failing_registration.success,
        )
        self.audit_check(
            "dispatch(handler_returns_fail).is_failure",
            failing_dispatch.failure,
        )

        self.section("event_publishing")
        subscriber = _EventSubscriber()
        audit_subscriber = _AuditSubscriber()
        register_subscriber = dispatcher.register_handler(subscriber, is_event=True)
        register_audit_subscriber = dispatcher.register_handler(
            audit_subscriber,
            is_event=True,
        )
        publish_single = dispatcher.publish(m.Examples.UserCreated(username="alice"))
        publish_list = dispatcher.publish([
            m.Examples.UserCreated(username="alice"),
            m.Examples.UserCreated(username="bob"),
        ])
        publish_without_subscribers = dispatcher.publish(
            m.Examples.NoSubscriberEvent(marker="none"),
        )
        self.audit_check(
            "register(event_subscriber).is_success",
            register_subscriber.success,
        )
        self.audit_check(
            "register(audit_subscriber).is_success",
            register_audit_subscriber.success,
        )
        self.audit_check("publish(single).is_success", publish_single.success)
        self.audit_check("publish(list).is_success", publish_list.success)
        self.audit_check("subscriber.events", subscriber.events)
        self.audit_check("audit_subscriber.events", audit_subscriber.events)
        self.audit_check(
            "publish(no_subscribers).is_success",
            publish_without_subscribers.success,
        )
        self.audit_check(
            "publish(no_subscribers).value",
            publish_without_subscribers.unwrap_or(False),
        )


if __name__ == "__main__":
    _Ex04DispatchGolden(caller_file=Path(__file__)).run()
