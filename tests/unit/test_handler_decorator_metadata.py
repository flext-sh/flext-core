"""Handler decorator metadata tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_tests import h, r, tm

from tests.constants import c

if TYPE_CHECKING:
    from collections.abc import MutableSequence

    from tests.base import s
    from tests.models import m
    from tests.protocols import p


class TestsFlextHandlerDecoratorMetadata:
    def test_decorator_stores_metadata_on_method(self) -> None:
        class CreateCommand:
            pass

        class Service:
            @h.handler(command=CreateCommand, priority=10)
            def handle_user(self, cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("handled")

        settings: m.DecoratorConfig = getattr(Service.handle_user, c.HANDLER_ATTR)
        tm.that(settings is not None, eq=True)
        tm.that(settings.command is CreateCommand, eq=True)
        tm.that(settings.priority, eq=10)

    def test_decorator_metadata_contains_command_type(self) -> None:
        class CreateCommand:
            pass

        class Service:
            @h.handler(command=CreateCommand)
            def handle_user(self, cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("handled")

        settings: m.DecoratorConfig = getattr(Service.handle_user, c.HANDLER_ATTR)
        tm.that(settings.command is CreateCommand, eq=True)

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
            def handle_user(self, cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("handled")

        settings: m.DecoratorConfig = getattr(Service.handle_user, c.HANDLER_ATTR)
        tm.that(settings.priority, eq=42)
        if settings.timeout is not None:
            tm.that(abs(settings.timeout - 5.0), lt=1e-9)
        tm.that(settings.middleware, eq=middleware_types)

    def test_decorator_defaults(self) -> None:
        class CreateCommand:
            pass

        class Service:
            @h.handler(command=CreateCommand)
            def handle_user(self, cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("handled")

        settings: m.DecoratorConfig = getattr(Service.handle_user, c.HANDLER_ATTR)
        tm.that(settings.priority, eq=c.DEFAULT_MAX_COMMAND_RETRIES)
        tm.that(settings.timeout, eq=c.DEFAULT_TIMEOUT_SECONDS)
        tm.that(settings.middleware, empty=True)

    def test_decorator_preserves_function_identity(self) -> None:
        class CreateCommand:
            pass

        def original_handler(self: s[str], cmd: CreateCommand) -> p.Result[str]:
            _ = self
            _ = cmd
            return r[str].ok("handled")

        decorated = h.handler(command=CreateCommand)(original_handler)
        tm.that(decorated is original_handler, eq=True)
