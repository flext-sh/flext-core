"""Behavioral tests for the ``@h.handler`` decorator and its discovery API.

These tests assert only the PUBLIC contract of the handler decorator:

* the configuration surfaced by ``h.Discovery.scan_class`` /
  ``h.Discovery.has_handlers`` (the caller-facing discovery API), and
* the public fields of ``m.DecoratorConfig`` (command / priority / timeout /
  middleware).

They never reach into the private marker attribute the decorator stores on the
method; that is an implementation detail of how discovery is wired.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import pytest
from dataclasses import dataclass

from flext_tests import h, r
from tests.base import s

if TYPE_CHECKING:
    from tests.protocols import p


class TestsFlextHandlerDecoratorEdges:
    """Public-contract behavior of the handler decorator and discovery."""

    def test_scan_class_exposes_declared_command_and_priority(self) -> None:
        # Arrange
        class CreateCommand:
            pass

        class Service:
            @h.handler(command=CreateCommand, priority=10)
            def handle(self, cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("ok")

        # Act
        handlers = h.Discovery.scan_class(Service)

        # Assert
        assert len(handlers) == 1
        method_name, config = handlers[0]
        assert method_name == "handle"
        assert config.command is CreateCommand
        assert config.priority == 10

    def test_defaults_are_applied_when_priority_and_timeout_omitted(self) -> None:
        # Arrange
        class CreateCommand:
            pass

        class Service:
            @h.handler(command=CreateCommand)
            def handle(self, cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("ok")

        # Act
        _, config = h.Discovery.scan_class(Service)[0]

        # Assert: defaults surfaced through the public config
        dumped = config.model_dump()
        assert config.priority == 0
        assert dumped["timeout"] == 30
        assert dumped["middleware"] == ()

    def test_none_timeout_is_preserved(self) -> None:
        # Arrange
        class CreateCommand:
            pass

        class Service:
            @h.handler(command=CreateCommand, timeout=None)
            def handle(self, cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("ok")

        # Act
        _, config = h.Discovery.scan_class(Service)[0]

        # Assert
        assert config.model_dump()["timeout"] is None

    @pytest.mark.parametrize("timeout", [0.5, 5.0, 120.0])
    def test_explicit_timeout_is_preserved(self, timeout: float) -> None:
        # Arrange
        class CreateCommand:
            pass

        class Service:
            @h.handler(command=CreateCommand, timeout=timeout)
            def handle(self, cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("ok")

        # Act
        _, config = h.Discovery.scan_class(Service)[0]

        # Assert
        assert config.model_dump()["timeout"] == timeout

    def test_stacked_decorators_innermost_wins(self) -> None:
        # Arrange: the innermost decorator runs first and takes precedence.
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

        # Act
        _, config = h.Discovery.scan_class(Service)[0]

        # Assert: the inner (DeleteCommand/20) configuration is observed.
        assert config.command is DeleteCommand
        assert config.priority == 20

    def test_scan_class_sorts_handlers_by_priority_descending(self) -> None:
        # Arrange
        class LowCommand:
            pass

        class MidCommand:
            pass

        class HighCommand:
            pass

        class Service:
            @h.handler(command=LowCommand, priority=1)
            def handle_low(self, cmd: LowCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("low")

            @h.handler(command=MidCommand, priority=5)
            def handle_mid(self, cmd: MidCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("mid")

            @h.handler(command=HighCommand, priority=9)
            def handle_high(self, cmd: HighCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("high")

        # Act
        handlers = h.Discovery.scan_class(Service)

        # Assert: ordered highest-priority first.
        assert [name for name, _ in handlers] == [
            "handle_high",
            "handle_mid",
            "handle_low",
        ]
        assert [config.priority for _, config in handlers] == [9, 5, 1]

    def test_has_handlers_reflects_presence_of_decorated_methods(self) -> None:
        # Arrange
        class CreateCommand:
            pass

        class Decorated:
            @h.handler(command=CreateCommand)
            def handle(self, cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("ok")

        class Plain:
            def handle(self, cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("ok")

        # Act / Assert
        assert h.Discovery.has_handlers(Decorated) is True
        assert h.Discovery.has_handlers(Plain) is False

    def test_scan_class_returns_empty_for_undecorated_class(self) -> None:
        # Arrange
        class Plain:
            def handle(self) -> p.Result[str]:
                return r[str].ok("ok")

        # Act / Assert
        assert h.Discovery.scan_class(Plain) == []

    def test_decorated_method_stays_callable_and_returns_success(self) -> None:
        # Arrange: decoration must not alter the method's runtime behavior.
        @dataclass
        class CreateCommand:
            name: str

        class Service:
            @h.handler(command=CreateCommand, priority=3)
            def handle(self, cmd: CreateCommand) -> p.Result[str]:
                return r[str].ok(f"created_{cmd.name}")

        # Act
        result = Service().handle(CreateCommand("alpha"))

        # Assert
        assert result.success
        assert result.unwrap() == "created_alpha"

    def test_service_integration_discovers_handler_via_scan_class(self) -> None:
        # Arrange: a real FlextService subclass with a decorated handler.
        @dataclass
        class CreateCommand:
            name: str

        class Service(s[str]):
            @h.handler(command=CreateCommand, priority=10)
            def handle_user_create(self, cmd: CreateCommand) -> p.Result[str]:
                return r[str].ok(f"created_{cmd.name}")

            @override
            def execute(self) -> p.Result[str]:
                return r[str].ok("executed")

        # Act
        handlers = h.Discovery.scan_class(Service)

        # Assert
        assert len(handlers) >= 1
        method_name, config = handlers[0]
        assert method_name == "handle_user_create"
        assert config.command is CreateCommand
        assert config.priority == 10
