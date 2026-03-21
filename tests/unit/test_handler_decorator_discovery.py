"""Comprehensive tests for h.handler() decorator and h.Discovery namespace."""

from __future__ import annotations

import types
from typing import override

from flext_tests import tm

from flext_core import FlextService, h, r
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

        middleware_types: list[type[p.Middleware]] = []

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
        tm.that(config.priority == 42, eq=True)
        if config.timeout is not None:
            tm.that(abs(config.timeout - 5.0) < 1e-9, eq=True)
        tm.that(config.middleware == middleware_types, eq=True)

    def test_decorator_defaults(self) -> None:
        class CreateCommand:
            pass

        class Service:
            @h.handler(command=CreateCommand)
            def handle_user(self, cmd: CreateCommand) -> r[str]:
                _ = cmd
                return r[str].ok("handled")

        config: m.DecoratorConfig = getattr(Service.handle_user, c.HANDLER_ATTR)
        tm.that(config.priority == c.DEFAULT_PRIORITY, eq=True)
        tm.that(config.timeout == c.DEFAULT_TIMEOUT, eq=True)
        tm.that(config.middleware == [], eq=True)

    def test_decorator_preserves_function_identity(self) -> None:
        class CreateCommand:
            pass

        def original_handler(self: FlextService, cmd: CreateCommand) -> r[str]:
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
        tm.that(len(handlers) == 3, eq=True)
        tm.that(handlers[0][0] == "handle_create", eq=True)
        tm.that(handlers[1][0] == "handle_delete", eq=True)
        tm.that(handlers[2][0] == "handle_query", eq=True)

    def test_scan_class_empty_and_has_handlers(self) -> None:
        class ServiceWithoutHandlers:
            def process(self) -> str:
                return "ok"

        class ServiceWithHandler:
            @h.handler(command=str)
            def handle(self, cmd: str) -> r[str]:
                return r[str].ok(cmd)

        tm.that(len(h.Discovery.scan_class(ServiceWithoutHandlers)) == 0, eq=True)
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
        tm.that(len(handlers) == 1, eq=True)
        name, config = handlers[0]
        tm.that(name == "handle_event", eq=True)
        tm.that(config.command is EventPublished, eq=True)
        tm.that(config.priority == 25, eq=True)

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
        tm.that(len(handlers) == 2, eq=True)
        tm.that(handlers[0][0] == "handle_user_create_globally", eq=True)
        tm.that(handlers[1][0] == "handle_user_delete_globally", eq=True)

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
        tm.that("public_handler" in names, eq=True)

    def test_has_handlers_module(self) -> None:
        module_with_handlers = types.ModuleType("module_with_handlers")
        module_without_handlers = types.ModuleType("module_without_handlers")

        @h.handler(command=str)
        def handler(cmd: str) -> r[str]:
            return r[str].ok(cmd)

        setattr(module_with_handlers, "handler", handler)
        tm.that(h.Discovery.has_handlers_module(module_with_handlers) is True, eq=True)
        tm.that(
            h.Discovery.has_handlers_module(module_without_handlers) is False, eq=True
        )

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
        tm.that("handle_create" in names, eq=True)
        tm.that("handle_delete" in names, eq=True)

    def test_handler_decorator_with_none_timeout(self) -> None:
        class CreateCommand:
            pass

        class Service:
            @h.handler(command=CreateCommand, timeout=None)
            def handle(self, cmd: CreateCommand) -> r[str]:
                _ = cmd
                return r[str].ok("ok")

        config: m.DecoratorConfig = getattr(Service.handle, c.HANDLER_ATTR)
        tm.that(config.timeout is None, eq=True)

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
        tm.that(config.priority == 20, eq=True)

    def test_service_integration_with_flext_service(self) -> None:
        class CreateCommand:
            def __init__(self, name: str) -> None:
                self.name = name

        class Service(FlextService[str]):
            @h.handler(command=CreateCommand, priority=10)
            def handle_user_create(self, cmd: CreateCommand) -> r[str]:
                return r[str].ok(f"created_{cmd.name}")

            @override
            def execute(self) -> r[str]:
                return r[str].ok("executed")

        handlers = h.Discovery.scan_class(Service)
        tm.that(len(handlers) >= 1, eq=True)
        method_name, config = handlers[0]
        tm.that(method_name == "handle_user_create", eq=True)
        tm.that(config.command is CreateCommand, eq=True)
        tm.that(config.priority == 10, eq=True)


__all__ = ["TestHandlerDecoratorDiscovery"]
