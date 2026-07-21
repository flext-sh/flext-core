"""Behavioral tests for FlextHandlers.Discovery class scanning.

Exercises only the public contract of ``h.Discovery``:
``scan_class`` (discover + sort decorated methods) and ``has_handlers``
(presence probe). No private attributes or internal collaborators are touched.
"""

from __future__ import annotations

import pytest

from flext_tests import h, r, tm
from tests import m, p


class TestsFlextCoreHandlerDiscoveryClass:
    """Public-contract tests for handler auto-discovery on classes."""

    def test_scan_class_discovers_every_decorated_method(self) -> None:
        # Arrange
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

        # Act
        handlers = h.Discovery.scan_class(Service)

        # Assert: only the three decorated methods are discovered.
        discovered = {name for name, _ in handlers}
        tm.that(len(handlers), eq=3)
        tm.that(discovered, eq={"handle_create", "handle_delete", "handle_query"})
        tm.that("non_handler_method" in discovered, eq=False)

    def test_scan_class_orders_handlers_by_priority_descending(self) -> None:
        # Arrange
        class LowCommand:
            pass

        class MidCommand:
            pass

        class HighCommand:
            pass

        class Service:
            @h.handler(command=LowCommand, priority=10)
            def handle_low(self, cmd: LowCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("low")

            @h.handler(command=HighCommand, priority=100)
            def handle_high(self, cmd: HighCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("high")

            @h.handler(command=MidCommand, priority=50)
            def handle_mid(self, cmd: MidCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("mid")

        # Act
        ordered_names = [name for name, _ in h.Discovery.scan_class(Service)]
        ordered_priorities = [
            settings.priority for _, settings in h.Discovery.scan_class(Service)
        ]

        # Assert: highest priority first regardless of declaration order.
        tm.that(ordered_names, eq=["handle_high", "handle_mid", "handle_low"])
        tm.that(ordered_priorities, eq=[100, 50, 10])

    def test_scan_class_binds_config_command_and_priority(self) -> None:
        # Arrange
        class EventPublished(m.Value):
            event_id: str

        class OrderService:
            @h.handler(command=EventPublished, priority=25)
            def handle_event(self, event: EventPublished) -> p.Result[str]:
                return r[str].ok(f"processed_{event.event_id}")

        # Act
        handlers = h.Discovery.scan_class(OrderService)
        name, settings = handlers[0]

        # Assert: the returned config reflects the decorator arguments.
        tm.that(len(handlers), eq=1)
        tm.that(name, eq="handle_event")
        tm.that(settings.command is EventPublished, eq=True)
        tm.that(settings.priority, eq=25)

    def test_scan_class_uses_default_priority_when_unspecified(self) -> None:
        # Arrange
        class PlainCommand:
            pass

        class Service:
            @h.handler(command=PlainCommand)
            def handle(self, cmd: PlainCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("done")

        # Act
        _, settings = h.Discovery.scan_class(Service)[0]

        # Assert: an undeclared priority falls back to the default (0).
        tm.that(settings.command is PlainCommand, eq=True)
        tm.that(settings.priority, eq=0)

    def test_scan_class_returns_empty_for_class_without_handlers(self) -> None:
        # Arrange
        class ServiceWithoutHandlers:
            def process(self) -> str:
                return "ok"

        # Act
        handlers = h.Discovery.scan_class(ServiceWithoutHandlers)

        # Assert
        tm.that(len(handlers), eq=0)
        tm.that(bool(handlers), eq=False)

    @pytest.mark.parametrize("expected_present", [True, False])
    def test_has_handlers_reflects_presence_of_decorated_methods(
        self, *, expected_present: bool
    ) -> None:
        # Arrange
        class Command:
            pass

        class ServiceWithHandler:
            @h.handler(command=Command)
            def handle(self, cmd: Command) -> p.Result[str]:
                _ = cmd
                return r[str].ok("ok")

        class ServiceWithoutHandler:
            def process(self) -> str:
                return "ok"

        target = ServiceWithHandler if expected_present else ServiceWithoutHandler

        # Act
        result = h.Discovery.has_handlers(target)

        # Assert
        tm.that(result, eq=expected_present)

    def test_scan_class_includes_inherited_handlers(self) -> None:
        # Arrange
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

        # Act
        handlers = h.Discovery.scan_class(DerivedService)
        names = [name for name, _ in handlers]

        # Assert: both inherited and own handlers are discovered, priority-sorted.
        tm.that(names, eq=["handle_create", "handle_delete"])
        tm.that(h.Discovery.has_handlers(DerivedService), eq=True)

    def test_scan_class_is_idempotent(self) -> None:
        # Arrange
        class CommandA:
            pass

        class CommandB:
            pass

        class Service:
            @h.handler(command=CommandA, priority=2)
            def handle_a(self, cmd: CommandA) -> p.Result[str]:
                _ = cmd
                return r[str].ok("a")

            @h.handler(command=CommandB, priority=1)
            def handle_b(self, cmd: CommandB) -> p.Result[str]:
                _ = cmd
                return r[str].ok("b")

        # Act: two independent scans of the same class.
        first = [(name, cfg.priority) for name, cfg in h.Discovery.scan_class(Service)]
        second = [(name, cfg.priority) for name, cfg in h.Discovery.scan_class(Service)]

        # Assert: discovery is stable across repeated calls.
        tm.that(first, eq=second)
        tm.that(first, eq=[("handle_a", 2), ("handle_b", 1)])
