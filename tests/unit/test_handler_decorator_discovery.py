"""Behavioral contract for the ``h.handler`` decorator + ``h.Discovery`` pair.

Asserts the OBSERVABLE round-trip a caller depends on: a function or method
decorated with ``h.handler`` is found by ``h.Discovery`` and, when invoked, still
produces the correct ``r[T]`` outcome. Tests read only the public surface — the
returned ``(name, ...)`` tuples, the public ``DecoratorConfig`` fields, and the
``r[T]`` result of calling the discovered handler — never how metadata is stored.
"""

from __future__ import annotations

import types

import pytest

from flext_tests import h, r
from tests import m, p


class TestsFlextCoreHandlerDecoratorDiscovery:
    """Public contract: decorate -> discover -> invoke preserves the r[T] outcome."""

    @staticmethod
    def _module(name: str) -> types.ModuleType:
        return types.ModuleType(name)

    def test_discovered_module_handler_invokes_with_success_outcome(self) -> None:
        class CreateCommand(m.Value):
            name: str

        module = self._module("success_module")

        @h.handler(command=CreateCommand, priority=100)
        def handle_create(cmd: CreateCommand) -> p.Result[str]:
            return r[str].ok(f"created_{cmd.name}")

        module.__dict__["handle_create"] = handle_create

        ((name, _, config),) = h.Discovery.scan_module(module)
        outcome = handle_create(CreateCommand(name="alice"))

        assert name == "handle_create"
        assert config.command is CreateCommand
        assert outcome.success is True
        assert outcome.unwrap() == "created_alice"

    def test_discovered_module_handler_preserves_failure_outcome(self) -> None:
        class DeleteCommand(m.Value):
            user_id: str

        module = self._module("failure_module")

        @h.handler(command=DeleteCommand)
        def handle_delete(cmd: DeleteCommand) -> p.Result[str]:
            return r[str].fail(f"missing_{cmd.user_id}")

        module.__dict__["handle_delete"] = handle_delete

        ((name, _, _),) = h.Discovery.scan_module(module)
        outcome = handle_delete(DeleteCommand(user_id="u42"))

        assert name == "handle_delete"
        assert outcome.failure is True
        assert outcome.error == "missing_u42"

    def test_scan_module_returns_empty_when_no_handlers_decorated(self) -> None:
        module = self._module("plain_module")

        def plain(value: int) -> p.Result[int]:
            return r[int].ok(value)

        module.__dict__["plain"] = plain
        module.__dict__["constant"] = "not-a-handler"

        assert h.Discovery.scan_module(module) == []

    def test_scan_module_excludes_private_functions(self) -> None:
        class Command:
            pass

        module = self._module("privacy_module")

        @h.handler(command=Command)
        def _private_handler(cmd: Command) -> p.Result[str]:
            _ = cmd
            return r[str].ok("private")

        @h.handler(command=Command)
        def public_handler(cmd: Command) -> p.Result[str]:
            _ = cmd
            return r[str].ok("public")

        module.__dict__["_private_handler"] = _private_handler
        module.__dict__["public_handler"] = public_handler

        names = [name for name, _, _ in h.Discovery.scan_module(module)]

        assert names == ["public_handler"]

    @pytest.mark.parametrize("priority", [0, 25, 100])
    def test_scan_module_reports_decorated_priority_and_command(
        self, priority: int
    ) -> None:
        class Command:
            pass

        module = self._module("config_module")

        @h.handler(command=Command, priority=priority)
        def handle(cmd: Command) -> p.Result[str]:
            _ = cmd
            return r[str].ok("ok")

        module.__dict__["handle"] = handle

        ((_, _, config),) = h.Discovery.scan_module(module)

        assert config.command is Command
        assert config.priority == priority

    def test_discovered_class_handler_invokes_via_instance(self) -> None:
        class EventPublished(m.Value):
            event_id: str

        class OrderService:
            @h.handler(command=EventPublished, priority=25)
            def handle_event(self, event: EventPublished) -> p.Result[str]:
                return r[str].ok(f"processed_{event.event_id}")

        ((name, _),) = h.Discovery.scan_class(OrderService)
        outcome = getattr(OrderService(), name)(EventPublished(event_id="e7"))

        assert name == "handle_event"
        assert outcome.success is True
        assert outcome.unwrap() == "processed_e7"

    @pytest.mark.parametrize(
        ("priority", "expected_name"), [(10, "handle_low"), (90, "handle_high")]
    )
    def test_scan_class_reports_command_and_priority(
        self, priority: int, expected_name: str
    ) -> None:
        class Command:
            pass

        class Service:
            @h.handler(command=Command, priority=10)
            def handle_low(self, cmd: Command) -> p.Result[str]:
                _ = cmd
                return r[str].ok("low")

            @h.handler(command=Command, priority=90)
            def handle_high(self, cmd: Command) -> p.Result[str]:
                _ = cmd
                return r[str].ok("high")

        by_name = dict(h.Discovery.scan_class(Service))

        assert by_name[expected_name].priority == priority
        assert by_name[expected_name].command is Command

    def test_has_handlers_reflects_presence_of_decorated_methods(self) -> None:
        class Command:
            pass

        class WithoutHandlers:
            def process(self) -> str:
                return "ok"

        class WithHandler:
            @h.handler(command=Command)
            def handle(self, cmd: Command) -> p.Result[str]:
                _ = cmd
                return r[str].ok("done")

        assert h.Discovery.has_handlers(WithoutHandlers) is False
        assert h.Discovery.has_handlers(WithHandler) is True

    def test_scan_class_discovers_inherited_handlers(self) -> None:
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

        names = {name for name, _ in h.Discovery.scan_class(DerivedService)}

        assert names == {"handle_create", "handle_delete"}
