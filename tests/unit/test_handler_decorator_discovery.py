"""Comprehensive tests for h.handler() decorator and h.Discovery namespace."""

from __future__ import annotations

import types
from collections.abc import MutableSequence
from typing import override

from flext_tests import tm

from flext_core import h, r, s
from tests import c, m, p


class TestHandlerDecoratorDiscovery:
    def test_decorator_stores_metadata_on_method(self) -> None:
        class CreateCommand:
            pass

        class Service:
            @h.handler(command=CreateCommand, priority=10)
            def handle_user(self, cmd: CreateCommand) -> r[str]:
                _ = cmd
                return r[str].ok("handled")

        method = Service.handle_user
        tm.that(hasattr(method, c.HANDLER_ATTR), eq=True)

    def test_decorator_metadata_contains_command_type(self) -> None:
        class CreateCommand:
            pass

        class Service:
            @h.handler(command=CreateCommand)
            def handle_user(self, cmd: CreateCommand) -> r[str]:
                _ = cmd
                return r[str].ok("handled")

        config: m.DecoratorConfig = getattr(Service.handle_user, c.HANDLER_ATTR)
        tm.that(config.command is CreateCommand, eq=True)

    def test_decorator_priority_timeout_and_middleware(self) -> None:
        class CreateCommand:
            pass

        middleware_types: MutableSequence[type[p.Middleware]] = []

        class Service:
            @h.handler(
                command=CreateCommand,
                priority=42,
                timeout=5.0,
                middleware=middleware_types,
            )
            def handle_user(self, cmd: CreateCommand) -> r[str]:
                _ = cmd
                return r[str].ok("handled")

        config: m.DecoratorConfig = getattr(Service.handle_user, c.HANDLER_ATTR)
        tm.that(config.priority, eq=42)
        if config.timeout is not None:
            tm.that(abs(config.timeout - 5.0), lt=1e-9)
        tm.that(config.middleware, eq=middleware_types)

    def test_decorator_defaults(self) -> None:
        class CreateCommand:
            pass

        class Service:
            @h.handler(command=CreateCommand)
            def handle_user(self, cmd: CreateCommand) -> r[str]:
                _ = cmd
                return r[str].ok("handled")

        config: m.DecoratorConfig = getattr(Service.handle_user, c.HANDLER_ATTR)
        tm.that(config.priority, eq=c.DEFAULT_MAX_COMMAND_RETRIES)
        tm.that(config.timeout, eq=c.DEFAULT_TIMEOUT_SECONDS)
        tm.that(config.middleware, eq=[])

    def test_decorator_preserves_function_identity(self) -> None:
        class CreateCommand:
            pass

        def original_handler(self: s, cmd: CreateCommand) -> r[str]:
            _ = self
            _ = cmd
            return r[str].ok("handled")

        decorated = h.handler(command=CreateCommand)(original_handler)
        tm.that(decorated is original_handler, eq=True)

    def test_scan_class_discovers_and_sorts_handlers(self) -> None:
        class CreateCommand:
            pass

        class DeleteCommand:
            pass

        class QueryCommand:
            pass

        class Service:
            @h.handler(command=CreateCommand, priority=100)
            def handle_create(self, cmd: CreateCommand) -> r[str]:
                _ = cmd
                return r[str].ok("create")

            @h.handler(command=DeleteCommand, priority=50)
            def handle_delete(self, cmd: DeleteCommand) -> r[str]:
                _ = cmd
                return r[str].ok("delete")

            @h.handler(command=QueryCommand, priority=10)
            def handle_query(self, cmd: QueryCommand) -> r[str]:
                _ = cmd
                return r[str].ok("query")

            def non_handler_method(self) -> str:
                return "non_handler"

        handlers = h.Discovery.scan_class(Service)
        tm.that(len(handlers), eq=3)
        tm.that(handlers[0][0], eq="handle_create")
        tm.that(handlers[1][0], eq="handle_delete")
        tm.that(handlers[2][0], eq="handle_query")

    def test_scan_class_empty_and_has_handlers(self) -> None:
        class ServiceWithoutHandlers:
            def process(self) -> str:
                return "ok"

        class ServiceWithHandler:
            @h.handler(command=str)
            def handle(self, cmd: str) -> r[str]:
                return r[str].ok(cmd)

        tm.that(not h.Discovery.scan_class(ServiceWithoutHandlers), eq=True)
        tm.that(h.Discovery.has_handlers(ServiceWithoutHandlers) is False, eq=True)
        tm.that(h.Discovery.has_handlers(ServiceWithHandler) is True, eq=True)

    def test_scan_class_single_handler(self) -> None:
        class EventPublished:
            def __init__(self, event_id: str) -> None:
                self.event_id = event_id

        class OrderService:
            @h.handler(command=EventPublished, priority=25)
            def handle_event(self, event: EventPublished) -> r[str]:
                return r[str].ok(f"processed_{event.event_id}")

        handlers = h.Discovery.scan_class(OrderService)
        tm.that(len(handlers), eq=1)
        name, config = handlers[0]
        tm.that(name, eq="handle_event")
        tm.that(config.command is EventPublished, eq=True)
        tm.that(config.priority, eq=25)

    def test_scan_module_finds_decorated_functions(self) -> None:
        class CreateCommand:
            def __init__(self, name: str) -> None:
                self.name = name

        class DeleteCommand:
            def __init__(self, user_id: str) -> None:
                self.user_id = user_id

        module = types.ModuleType("decorated_module")

        @h.handler(command=CreateCommand, priority=100)
        def handle_user_create_globally(cmd: CreateCommand) -> r[str]:
            return r[str].ok(f"global_create_{cmd.name}")

        @h.handler(command=DeleteCommand, priority=50)
        def handle_user_delete_globally(cmd: DeleteCommand) -> r[str]:
            return r[str].ok(f"global_delete_{cmd.user_id}")

        setattr(module, "handle_user_create_globally", handle_user_create_globally)
        setattr(module, "handle_user_delete_globally", handle_user_delete_globally)
        setattr(module, "non_callable", 123)

        handlers = h.Discovery.scan_module(module)
        tm.that(len(handlers), eq=2)
        tm.that(handlers[0][0], eq="handle_user_create_globally")
        tm.that(handlers[1][0], eq="handle_user_delete_globally")

    def test_scan_module_ignores_private_functions(self) -> None:
        class CreateCommand:
            pass

        module = types.ModuleType("private_check_module")

        @h.handler(command=CreateCommand)
        def _private_handler(cmd: CreateCommand) -> r[str]:
            _ = cmd
            return r[str].ok("private")

        @h.handler(command=CreateCommand)
        def public_handler(cmd: CreateCommand) -> r[str]:
            _ = cmd
            return r[str].ok("public")

        setattr(module, "_private_handler", _private_handler)
        setattr(module, "public_handler", public_handler)
        handlers = h.Discovery.scan_module(module)
        names = [name for name, _, _ in handlers]
        tm.that("_private_handler" not in names, eq=True)
        tm.that(names, has="public_handler")

    def test_scan_class_with_inherited_handlers(self) -> None:
        class CreateCommand:
            pass

        class DeleteCommand:
            pass

        class BaseService:
            @h.handler(command=CreateCommand, priority=10)
            def handle_create(self, cmd: CreateCommand) -> r[str]:
                _ = cmd
                return r[str].ok("created")

        class DerivedService(BaseService):
            @h.handler(command=DeleteCommand, priority=5)
            def handle_delete(self, cmd: DeleteCommand) -> r[str]:
                _ = cmd
                return r[str].ok("deleted")

        handlers = h.Discovery.scan_class(DerivedService)
        names = [name for name, _ in handlers]
        tm.that(names, has="handle_create")
        tm.that(names, has="handle_delete")

    def test_handler_decorator_with_none_timeout(self) -> None:
        class CreateCommand:
            pass

        class Service:
            @h.handler(command=CreateCommand, timeout=None)
            def handle(self, cmd: CreateCommand) -> r[str]:
                _ = cmd
                return r[str].ok("ok")

        config: m.DecoratorConfig = getattr(Service.handle, c.HANDLER_ATTR)
        tm.that(config.timeout, none=True)

    def test_multiple_decorations_overwrites_previous(self) -> None:
        class CreateCommand:
            pass

        class DeleteCommand:
            pass

        class Service:
            @h.handler(command=CreateCommand, priority=10)
            @h.handler(command=DeleteCommand, priority=20)
            def handle(self, cmd: DeleteCommand) -> r[str]:
                _ = cmd
                return r[str].ok("ok")

        config: m.DecoratorConfig = getattr(Service.handle, c.HANDLER_ATTR)
        tm.that(config.command is DeleteCommand, eq=True)
        tm.that(config.priority, eq=20)

    def test_service_integration_with_flext_service(self) -> None:
        class CreateCommand:
            def __init__(self, name: str) -> None:
                self.name = name

        class Service(s[str]):
            @h.handler(command=CreateCommand, priority=10)
            def handle_user_create(self, cmd: CreateCommand) -> r[str]:
                return r[str].ok(f"created_{cmd.name}")

            @override
            def execute(self) -> r[str]:
                return r[str].ok("executed")

        handlers = h.Discovery.scan_class(Service)
        tm.that(len(handlers), gte=1)
        method_name, config = handlers[0]
        tm.that(method_name, eq="handle_user_create")
        tm.that(config.command is CreateCommand, eq=True)
        tm.that(config.priority, eq=10)


__all__ = ["TestHandlerDecoratorDiscovery"]
