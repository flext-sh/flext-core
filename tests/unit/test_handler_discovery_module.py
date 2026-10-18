"""Behavioral tests for FlextHandlers.Discovery.scan_module public contract.

These tests exercise only the observable contract of ``h.Discovery.scan_module``:
which module-level functions it discovers, the ordering guarantee, the metadata
it surfaces, and the behavior of the callable it hands back. No private members
of the unit under test are touched.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING

import pytest

from tests import h, m, r, tm

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests import p, t


class TestsFlextHandlerDiscoveryModule:
    """Public-contract tests for module handler discovery."""

    class _CreateCommand:
        """Sample command payload used to decorate discovered handlers."""

    class _DeleteCommand:
        """Second sample command payload for multi-handler modules."""

    @staticmethod
    def test_scan_module_discovers_every_decorated_public_function() -> None:
        """All decorated public functions are discovered."""
        # Arrange
        module = types.ModuleType("decorated_module")

        @h.handler(
            command=TestsFlextHandlerDiscoveryModule._CreateCommand,
            priority=100,
        )
        def handle_create(
            cmd: TestsFlextHandlerDiscoveryModule._CreateCommand,
        ) -> p.Result[str]:
            _ = cmd
            return r[str].ok("created")

        @h.handler(command=TestsFlextHandlerDiscoveryModule._DeleteCommand, priority=50)
        def handle_delete(
            cmd: TestsFlextHandlerDiscoveryModule._DeleteCommand,
        ) -> p.Result[str]:
            _ = cmd
            return r[str].ok("deleted")

        module.__dict__["handle_create"] = handle_create
        module.__dict__["handle_delete"] = handle_delete

        # Act
        handlers = h.Discovery.scan_module(module)
        names = [name for name, _, _ in handlers]

        # Assert
        tm.that(len(handlers), eq=2)
        tm.that(names, has="handle_create")
        tm.that(names, has="handle_delete")

    @staticmethod
    def test_scan_module_omits_underscore_prefixed_functions() -> None:
        """Private module functions are excluded from discovery."""
        # Arrange
        module = types.ModuleType("private_check_module")

        @h.handler(command=TestsFlextHandlerDiscoveryModule._CreateCommand)
        def _private_handler(
            cmd: TestsFlextHandlerDiscoveryModule._CreateCommand,
        ) -> p.Result[str]:
            _ = cmd
            return r[str].ok("private")

        @h.handler(command=TestsFlextHandlerDiscoveryModule._CreateCommand)
        def public_handler(
            cmd: TestsFlextHandlerDiscoveryModule._CreateCommand,
        ) -> p.Result[str]:
            _ = cmd
            return r[str].ok("public")

        module.__dict__["_private_handler"] = _private_handler
        module.__dict__["public_handler"] = public_handler

        # Act
        handlers = h.Discovery.scan_module(module)
        names = [name for name, _, _ in handlers]

        # Assert
        tm.that("_private_handler" not in names, eq=True)
        tm.that(names, has="public_handler")

    @staticmethod
    def test_scan_module_ignores_non_callable_and_undecorated_members() -> None:
        """Undecorated functions and non-callable values are ignored."""
        # Arrange
        module = types.ModuleType("mixed_members_module")

        @h.handler(command=TestsFlextHandlerDiscoveryModule._CreateCommand)
        def decorated(
            cmd: TestsFlextHandlerDiscoveryModule._CreateCommand,
        ) -> p.Result[str]:
            _ = cmd
            return r[str].ok("ok")

        def plain(cmd: TestsFlextHandlerDiscoveryModule._CreateCommand) -> str:
            _ = cmd
            return "plain"

        module.__dict__["decorated"] = decorated
        module.__dict__["plain"] = plain
        module.__dict__["constant"] = 123

        # Act
        handlers = h.Discovery.scan_module(module)
        names = [name for name, _, _ in handlers]

        # Assert
        tm.that(len(handlers), eq=1)
        tm.that(names, has="decorated")
        tm.that("plain" not in names, eq=True)
        tm.that("constant" not in names, eq=True)

    @staticmethod
    def _priority_module(
        module_name: str,
        handlers: tuple[tuple[str, int, str], ...],
    ) -> types.ModuleType:
        """Build one synthetic module holding priority-decorated handlers.

        Args:
            module_name: Synthetic module name.
            handlers: ``(attribute_name, priority, payload)`` triples.

        Returns:
            The resulting ``types.ModuleType``.

        """
        module = types.ModuleType(module_name)

        def build(
            attr_name: str,
            priority: int,
            payload: str,
        ) -> tuple[str, Callable[..., t.UnconstrainedValue]]:
            @h.handler(
                command=TestsFlextHandlerDiscoveryModule._CreateCommand,
                priority=priority,
            )
            def handler(
                cmd: TestsFlextHandlerDiscoveryModule._CreateCommand,
            ) -> p.Result[str]:
                _ = cmd
                return r[str].ok(payload)

            return attr_name, handler

        for attr_name, priority, payload in handlers:
            name, handler = build(attr_name, priority, payload)
            module.__dict__[name] = handler
        return module

    @staticmethod
    def test_scan_module_orders_by_priority_descending() -> None:
        """Discovered functions are ordered from highest to lowest priority."""
        # Arrange
        module = TestsFlextHandlerDiscoveryModule._priority_module(
            "priority_order_module",
            (("low", 10, "low"), ("high", 90, "high")),
        )

        # Act
        ordered = [name for name, _, _ in h.Discovery.scan_module(module)]

        # Assert
        tm.that(ordered, eq=["high", "low"])

    @staticmethod
    def test_scan_module_breaks_priority_ties_by_name() -> None:
        """Equal-priority functions use their names as a stable tie breaker."""
        # Arrange
        module = TestsFlextHandlerDiscoveryModule._priority_module(
            "tie_break_module",
            (("bravo", 5, "b"), ("alpha", 5, "a")),
        )

        # Act
        ordered = [name for name, _, _ in h.Discovery.scan_module(module)]

        # Assert
        tm.that(ordered, eq=["alpha", "bravo"])

    @staticmethod
    def test_scan_module_surfaces_decorator_metadata() -> None:
        """Discovery preserves the handler's public decorator metadata."""
        # Arrange
        module = types.ModuleType("metadata_module")

        @h.handler(command=TestsFlextHandlerDiscoveryModule._DeleteCommand, priority=7)
        def handle(
            cmd: TestsFlextHandlerDiscoveryModule._DeleteCommand,
        ) -> p.Result[str]:
            _ = cmd
            return r[str].ok("done")

        module.__dict__["handle"] = handle

        # Act
        _, _, config = h.Discovery.scan_module(module)[0]

        # Assert
        tm.that(
            config.command is TestsFlextHandlerDiscoveryModule._DeleteCommand,
            eq=True,
        )
        tm.that(config.priority, eq=7)

    @staticmethod
    def test_scan_module_returns_empty_for_module_without_handlers() -> None:
        """A module without decorated handlers yields an empty sequence."""
        # Arrange
        module = types.ModuleType("bare_module")
        module.__dict__["value"] = 42

        # Act
        handlers = h.Discovery.scan_module(module)

        # Assert
        tm.that(len(handlers), eq=0)

    @pytest.mark.parametrize(
        "returned",
        [
            42,
            "payload",
            None,
            [1, 2],
            m.Tests.DispatchRequest(
                command_name="dispatch",
                tenant="test",
                environment="test",
            ),
        ],
    )
    @staticmethod
    def test_discovered_callable_preserves_original_return_value(
        returned: t.JsonPayload,
    ) -> None:
        """Discovery preserves identity and never serializes the return value."""
        # Arrange
        module = types.ModuleType("coercion_module")

        @h.handler(command=TestsFlextHandlerDiscoveryModule._CreateCommand)
        def produce(
            cmd: TestsFlextHandlerDiscoveryModule._CreateCommand,
        ) -> t.JsonPayload:
            _ = cmd
            return returned

        module.__dict__["produce"] = produce
        _, discovered, _ = h.Discovery.scan_module(module)[0]
        tm.that(discovered is produce, eq=True)
        tm.that(discovered({}) is returned, eq=True)

    @staticmethod
    def test_rediscovery_preserves_identity_and_does_not_resolve_replacements() -> None:
        """A discovered function survives replacement/removal of its module name."""
        module = types.ModuleType("identity_module")
        outcome = r[str].ok("original")

        @h.handler(command=TestsFlextHandlerDiscoveryModule._CreateCommand)
        def original(cmd: p.Base) -> p.Result[str]:
            _ = cmd
            return outcome

        module.__dict__["handle"] = original
        _, first, _ = h.Discovery.scan_module(module)[0]
        _, second, _ = h.Discovery.scan_module(module)[0]
        tm.that(first is second is original, eq=True)
        module.__dict__["handle"] = lambda _cmd: "replacement"
        tm.that(first({}) is outcome, eq=True)
        del module.__dict__["handle"]
        tm.that(first({}) is outcome, eq=True)

    @staticmethod
    def test_discovered_callable_preserves_failure_cause() -> None:
        """Discovery does not repackage a failed Result or replace its cause."""
        module = types.ModuleType("failure_identity_module")
        cause = ValueError("request failed")
        outcome = r[str].fail("request failed", exception=cause)

        @h.handler(command=TestsFlextHandlerDiscoveryModule._CreateCommand)
        def original(cmd: p.Base) -> p.Result[str]:
            _ = cmd
            return outcome

        module.__dict__["handle"] = original
        _, discovered, _ = h.Discovery.scan_module(module)[0]
        tm.that(discovered({}) is outcome, eq=True)
        tm.that(outcome.exception is cause, eq=True)
