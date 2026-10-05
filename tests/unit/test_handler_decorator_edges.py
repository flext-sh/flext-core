"""Behavioral tests for the ``@h.handler`` decorator and its discovery API.

These tests assert only the PUBLIC contract of the handler decorator:

* the configuration surfaced by ``h.Discovery.scan_class`` /
  ``h.Discovery.has_handlers`` (the caller-facing discovery API), and
* the public fields of ``m.DecoratorConfig`` (command / priority / timeout /
  middleware).

They never reach into the private marker attribute the decorator stores on the
method; that is an implementation detail of how discovery is wired.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import pytest
from flext_tests import h, r

from tests.base import s
from tests.models import m

if TYPE_CHECKING:
    from tests.protocols import p


class TestsFlextHandlerDecoratorEdges:
    """Public-contract behavior of the handler decorator and discovery."""

    @staticmethod
    def test_scan_class_exposes_declared_command_and_priority() -> None:
        # Arrange
        """Test scan class exposes declared command and priority."""

        class CreateCommand(m.BaseModel):
            pass

        class Service:
            @staticmethod
            @h.handler(command=CreateCommand, priority=10)
            def handle(cmd: CreateCommand) -> p.Result[str]:
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

    @staticmethod
    def test_defaults_are_applied_when_priority_and_timeout_omitted() -> None:
        # Arrange
        """Test defaults are applied when priority and timeout omitted."""

        class CreateCommand(m.BaseModel):
            pass

        class Service:
            @staticmethod
            @h.handler(command=CreateCommand)
            def handle(cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("ok")

        # Act
        _, config = h.Discovery.scan_class(Service)[0]

        # Assert: the model's own declared defaults surface through discovery
        declared = m.DecoratorConfig(command=CreateCommand)
        assert config.model_dump() == declared.model_dump()

    @staticmethod
    def test_none_timeout_is_preserved() -> None:
        # Arrange
        """Test none timeout is preserved."""

        class CreateCommand(m.BaseModel):
            pass

        class Service:
            @staticmethod
            @h.handler(command=CreateCommand, timeout=None)
            def handle(cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("ok")

        # Act
        _, config = h.Discovery.scan_class(Service)[0]

        # Assert
        assert config.model_dump()["timeout"] is None

    @pytest.mark.parametrize("timeout", [0.5, 5.0, 120.0])
    @staticmethod
    def test_explicit_timeout_is_preserved(timeout: float) -> None:
        # Arrange
        """Test explicit timeout is preserved."""

        class CreateCommand(m.BaseModel):
            pass

        class Service:
            @staticmethod
            @h.handler(command=CreateCommand, timeout=timeout)
            def handle(cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("ok")

        # Act
        _, config = h.Discovery.scan_class(Service)[0]

        # Assert
        assert config.model_dump()["timeout"] == timeout

    @staticmethod
    def test_stacked_decorators_innermost_wins() -> None:
        # Arrange: the innermost decorator runs first and takes precedence.
        """Test stacked decorators innermost wins."""

        class CreateCommand(m.BaseModel):
            pass

        class DeleteCommand(m.BaseModel):
            pass

        class Service:
            @staticmethod
            @h.handler(command=CreateCommand, priority=10)
            @h.handler(command=DeleteCommand, priority=20)
            def handle(cmd: DeleteCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("ok")

        # Act
        _, config = h.Discovery.scan_class(Service)[0]

        # Assert: the inner (DeleteCommand/20) configuration is observed.
        assert config.command is DeleteCommand
        assert config.priority == 20

    @staticmethod
    def test_scan_class_sorts_handlers_by_priority_descending() -> None:
        # Arrange
        """Test scan class sorts handlers by priority descending."""

        class LowCommand(m.BaseModel):
            pass

        class MidCommand(m.BaseModel):
            pass

        class HighCommand(m.BaseModel):
            pass

        class Service:
            @staticmethod
            @h.handler(command=LowCommand, priority=1)
            def handle_low(cmd: LowCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("low")

            @staticmethod
            @h.handler(command=MidCommand, priority=5)
            def handle_mid(cmd: MidCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("mid")

            @staticmethod
            @h.handler(command=HighCommand, priority=9)
            def handle_high(cmd: HighCommand) -> p.Result[str]:
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

    @staticmethod
    def test_has_handlers_reflects_presence_of_decorated_methods() -> None:
        # Arrange
        """Test has handlers reflects presence of decorated methods."""

        class CreateCommand(m.BaseModel):
            pass

        class Decorated:
            @staticmethod
            @h.handler(command=CreateCommand)
            def handle(cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("ok")

        class Plain:
            @staticmethod
            def handle(cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("ok")

        # Act / Assert
        assert h.Discovery.has_handlers(Decorated) is True
        assert h.Discovery.has_handlers(Plain) is False

    @staticmethod
    def test_scan_class_returns_empty_for_undecorated_class() -> None:
        # Arrange
        """Test scan class returns empty for undecorated class."""

        class Plain:
            @staticmethod
            def handle() -> p.Result[str]:
                return r[str].ok("ok")

        # Act / Assert
        assert h.Discovery.scan_class(Plain) == []

    @staticmethod
    def test_decorated_method_stays_callable_and_returns_success() -> None:
        # Arrange: decoration must not alter the method's runtime behavior.
        """Test decorated method stays callable and returns success."""

        class CreateCommand(m.BaseModel):
            name: str

        class Service:
            @staticmethod
            @h.handler(command=CreateCommand, priority=3)
            def handle(cmd: CreateCommand) -> p.Result[str]:
                return r[str].ok(f"created_{cmd.name}")

        # Act
        result = Service().handle(CreateCommand(name="alpha"))

        # Assert
        assert result.success
        assert result.unwrap() == "created_alpha"

    @staticmethod
    def test_service_integration_discovers_handler_via_scan_class() -> None:
        # Arrange: a real FlextService subclass with a decorated handler.
        """Test service integration discovers handler via scan class."""

        class CreateCommand(m.BaseModel):
            name: str

        class Service(s[str]):
            @staticmethod
            @h.handler(command=CreateCommand, priority=10)
            def handle_user_create(cmd: CreateCommand) -> p.Result[str]:
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
