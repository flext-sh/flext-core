"""Handler decorator edge case tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from flext_tests import h, r, tm

from tests.base import s
from tests.constants import c

if TYPE_CHECKING:
    from tests.models import m
    from tests.protocols import p


class TestsFlextHandlerDecoratorEdges:
    def test_handler_decorator_with_none_timeout(self) -> None:
        class CreateCommand:
            pass

        class Service:
            @h.handler(command=CreateCommand, timeout=None)
            def handle(self, cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("ok")

        settings: m.DecoratorConfig = getattr(Service.handle, c.HANDLER_ATTR)
        tm.that(settings.timeout, none=True)

    def test_multiple_decorations_overwrites_previous(self) -> None:
        class CreateCommand:
            pass

        class DeleteCommand:
            pass

        class Service:
            @h.handler(command=CreateCommand, priority=10)
            @h.handler(command=DeleteCommand, priority=20)
            def handle(self, cmd: DeleteCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("ok")

        settings: m.DecoratorConfig = getattr(Service.handle, c.HANDLER_ATTR)
        tm.that(settings.command is DeleteCommand, eq=True)
        tm.that(settings.priority, eq=20)

    def test_service_integration_with_flext_service(self) -> None:
        class CreateCommand:
            def __init__(self, name: str) -> None:
                self.name = name

        class Service(s[str]):
            @h.handler(command=CreateCommand, priority=10)
            def handle_user_create(self, cmd: CreateCommand) -> p.Result[str]:
                return r[str].ok(f"created_{cmd.name}")

            @override
            def execute(self) -> p.Result[str]:
                return r[str].ok("executed")

        handlers = h.Discovery.scan_class(Service)
        tm.that(len(handlers), gte=1)
        method_name, settings = handlers[0]
        tm.that(method_name, eq="handle_user_create")
        tm.that(settings.command is CreateCommand, eq=True)
        tm.that(settings.priority, eq=10)
