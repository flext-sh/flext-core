"""Behavioral tests for the ``h.handler`` decorator public contract.

The decorator's documented contract is: the declared ``m.DecoratorConfig`` is
surfaced through the public discovery API (``h.Discovery``) so registries can
auto-discover handlers, while the original callable is returned unchanged.
These tests exercise that observable contract only, never the marker attribute
the decorator uses internally.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests import c, h, m, r, tm

if TYPE_CHECKING:
    from collections.abc import Callable, MutableSequence

    from tests import p, s


class TestsFlextHandlerDecoratorMetadata:
    """Tests for ``FlextHandlerDecoratorMetadata``."""

    @staticmethod
    def test_decorated_method_exposes_handler_config() -> None:
        """Test decorated method exposes handler config."""

        class CreateCommand:
            pass

        class Service:
            @staticmethod
            @h.handler(command=CreateCommand, priority=10)
            def handle_user(cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("handled")

        _, config = h.Discovery.scan_class(Service)[0]
        tm.that(config.command is CreateCommand, eq=True)
        tm.that(config.priority, eq=10)

    @staticmethod
    def test_undecorated_method_has_no_handler_config() -> None:
        """Test undecorated method has no handler config."""

        class Service:
            @staticmethod
            def handle_user() -> p.Result[str]:
                return r[str].ok("handled")

        tm.that(h.Discovery.has_handlers(Service), eq=False)

    @staticmethod
    @pytest.mark.parametrize(
        "priority",
        [0, 1, 42, 7],
    )
    def test_priority_is_recorded_without_implicit_timeout(
        priority: int,
    ) -> None:
        """Priority orders discovery without requesting an unsupported policy."""

        class CreateCommand:
            pass

        class Service:
            @staticmethod
            @h.handler(command=CreateCommand, priority=priority)
            def handle_user(cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("handled")

        _, config = h.Discovery.scan_class(Service)[0]
        tm.that(config.priority, eq=priority)
        tm.that(config.model_dump()["timeout"] is None, eq=True)

    @staticmethod
    @pytest.mark.parametrize("timeout", [0.5, 5.0, 30.0])
    def test_unsupported_timeout_is_rejected(timeout: float) -> None:
        """A requested timeout fails at the decorator, before registration."""
        with pytest.raises(ValueError, match="timeout execution is not supported"):
            h.handler(command=str, timeout=timeout)

    @staticmethod
    def test_negative_priority_is_rejected() -> None:
        """Test negative priority is rejected."""

        class CreateCommand:
            pass

        def define_invalid_service() -> None:
            class Service:
                @staticmethod
                @h.handler(command=CreateCommand, priority=-3)
                def handle_user(cmd: CreateCommand) -> p.Result[str]:
                    _ = cmd
                    return r[str].ok("handled")

            _ = Service

        with pytest.raises(c.ValidationError):
            define_invalid_service()

    @staticmethod
    def test_defaults_apply_when_only_command_given() -> None:
        """Test defaults apply when only command given."""

        class CreateCommand:
            pass

        class Service:
            @staticmethod
            @h.handler(command=CreateCommand)
            def handle_user(cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("handled")

        _, config = h.Discovery.scan_class(Service)[0]
        declared = m.DecoratorConfig(command=CreateCommand)
        tm.that(config.model_dump(), eq=declared.model_dump())

    @staticmethod
    def test_middleware_sequence_is_recorded() -> None:
        """Test middleware sequence is recorded."""

        class CreateCommand:
            pass

        middleware_types: MutableSequence[type[p.Middleware]] = []

        class Service:
            @staticmethod
            @h.handler(command=CreateCommand, middleware=middleware_types)
            def handle_user(cmd: CreateCommand) -> p.Result[str]:
                _ = cmd
                return r[str].ok("handled")

        _, config = h.Discovery.scan_class(Service)[0]
        tm.that(config.model_dump()["middleware"], eq=middleware_types)

    @staticmethod
    def test_unsupported_middleware_is_rejected() -> None:
        """Nonempty middleware is rejected rather than silently ignored."""

        class CreateCommand:
            pass

        class PassthroughMiddleware:
            @staticmethod
            def process[TResult](
                command: p.Model,
                next_handler: Callable[[p.Model], p.Result[TResult]],
            ) -> p.Result[TResult]:
                return next_handler(command)

        middleware_types: MutableSequence[type[p.Middleware]] = [PassthroughMiddleware]

        with pytest.raises(ValueError, match="middleware execution is not supported"):
            h.handler(command=CreateCommand, middleware=middleware_types)

    @staticmethod
    def test_decorator_returns_same_callable() -> None:
        """Test decorator returns same callable."""

        class CreateCommand:
            pass

        def original_handler(self: s[str], cmd: CreateCommand) -> p.Result[str]:
            _ = self
            _ = cmd
            return r[str].ok("handled")

        decorated = h.handler(command=CreateCommand)(original_handler)
        tm.that(decorated is original_handler, eq=True)

    @staticmethod
    def test_stacked_declarations_are_rejected() -> None:
        """A handler has exactly one schema declaration, never competing metadata."""

        class CreateCommand:
            pass

        class OtherCommand:
            pass

        def handle_user(cmd: p.Base) -> p.Result[str]:
            _ = cmd
            return r[str].ok("handled")

        decorated = h.handler(command=CreateCommand, priority=1)(handle_user)
        with pytest.raises(ValueError, match="metadata is already declared"):
            h.handler(command=OtherCommand, priority=99)(decorated)
        tm.that(decorated is handle_user, eq=True)
