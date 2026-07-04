"""Handler class discovery tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_tests import h, r, tm

if TYPE_CHECKING:
    from tests.protocols import p


class TestsFlextHandlerDiscoveryClass:
    def test_scan_class_discovers_and_sorts_handlers(self) -> None:
        class CreateCommand:
            pass

        class DeleteCommand:
            pass

        class QueryCommand:
            pass

        class Service:
            @h.handler(command=CreateCommand, priority=100)
            def handle_create(self, cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("create")

            @h.handler(command=DeleteCommand, priority=50)
            def handle_delete(self, cmd: DeleteCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("delete")

            @h.handler(command=QueryCommand, priority=10)
            def handle_query(self, cmd: QueryCommand) -> p.Result[str]:
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
            def handle(self, cmd: str) -> p.Result[str]:
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
            def handle_event(self, event: EventPublished) -> p.Result[str]:
                return r[str].ok(f"processed_{event.event_id}")

        handlers = h.Discovery.scan_class(OrderService)
        tm.that(len(handlers), eq=1)
        name, settings = handlers[0]
        tm.that(name, eq="handle_event")
        tm.that(settings.command is EventPublished, eq=True)
        tm.that(settings.priority, eq=25)

    def test_scan_class_with_inherited_handlers(self) -> None:
        class CreateCommand:
            pass

        class DeleteCommand:
            pass

        class BaseService:
            @h.handler(command=CreateCommand, priority=10)
            def handle_create(self, cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("created")

        class DerivedService(BaseService):
            @h.handler(command=DeleteCommand, priority=5)
            def handle_delete(self, cmd: DeleteCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("deleted")

        handlers = h.Discovery.scan_class(DerivedService)
        names = [name for name, _ in handlers]
        tm.that(names, has="handle_create")
        tm.that(names, has="handle_delete")
