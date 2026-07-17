"""Behavioral tests for FlextHandlers.Discovery.scan_module public contract.

These tests exercise only the observable contract of ``h.Discovery.scan_module``:
which module-level functions it discovers, the ordering guarantee, the metadata
it surfaces, and the behavior of the callable it hands back. No private members
of the unit under test are touched.
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING

import pytest
from flext_tests import h, r, tm

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.protocols import p
    from tests.typings import t


class _CreateCommand:
    """Sample command payload used to decorate discovered handlers."""


class _DeleteCommand:
    """Second sample command payload for multi-handler modules."""


class TestsFlextHandlerDiscoveryModule:
    """Public-contract tests for module handler discovery."""

    def test_scan_module_discovers_every_decorated_public_function(self) -> None:
        """All decorated public functions are discovered."""
        # Arrange
        module = types.ModuleType("decorated_module")

        @h.handler(command=_CreateCommand, priority=100)
        def handle_create(cmd: _CreateCommand) -> p.Result[str]:
            _ = cmd
            return r[str].ok("created")

        @h.handler(command=_DeleteCommand, priority=50)
        def handle_delete(cmd: _DeleteCommand) -> p.Result[str]:
            _ = cmd
            return r[str].ok("deleted")

        setattr(module, "handle_create", handle_create)
        setattr(module, "handle_delete", handle_delete)

        # Act
        handlers = h.Discovery.scan_module(module)
        names = [name for name, _, _ in handlers]

        # Assert
        tm.that(len(handlers), eq=2)
        tm.that(names, has="handle_create")
        tm.that(names, has="handle_delete")

    def test_scan_module_omits_underscore_prefixed_functions(self) -> None:
        """Private module functions are excluded from discovery."""
        # Arrange
        module = types.ModuleType("private_check_module")

        @h.handler(command=_CreateCommand)
        def _private_handler(cmd: _CreateCommand) -> p.Result[str]:
            _ = cmd
            return r[str].ok("private")

        @h.handler(command=_CreateCommand)
        def public_handler(cmd: _CreateCommand) -> p.Result[str]:
            _ = cmd
            return r[str].ok("public")

        setattr(module, "_private_handler", _private_handler)
        setattr(module, "public_handler", public_handler)

        # Act
        handlers = h.Discovery.scan_module(module)
        names = [name for name, _, _ in handlers]

        # Assert
        tm.that("_private_handler" not in names, eq=True)
        tm.that(names, has="public_handler")

    def test_scan_module_ignores_non_callable_and_undecorated_members(self) -> None:
        """Undecorated functions and non-callable values are ignored."""
        # Arrange
        module = types.ModuleType("mixed_members_module")

        @h.handler(command=_CreateCommand)
        def decorated(cmd: _CreateCommand) -> p.Result[str]:
            _ = cmd
            return r[str].ok("ok")

        def plain(cmd: _CreateCommand) -> str:
            _ = cmd
            return "plain"

        setattr(module, "decorated", decorated)
        setattr(module, "plain", plain)
        setattr(module, "constant", 123)

        # Act
        handlers = h.Discovery.scan_module(module)
        names = [name for name, _, _ in handlers]

        # Assert
        tm.that(len(handlers), eq=1)
        tm.that(names, has="decorated")
        tm.that("plain" not in names, eq=True)
        tm.that("constant" not in names, eq=True)

    def test_scan_module_orders_by_priority_descending(self) -> None:
        """Discovered functions are ordered from highest to lowest priority."""
        # Arrange
        module = types.ModuleType("priority_order_module")

        @h.handler(command=_CreateCommand, priority=10)
        def low(cmd: _CreateCommand) -> p.Result[str]:
            _ = cmd
            return r[str].ok("low")

        @h.handler(command=_CreateCommand, priority=90)
        def high(cmd: _CreateCommand) -> p.Result[str]:
            _ = cmd
            return r[str].ok("high")

        setattr(module, "low", low)
        setattr(module, "high", high)

        # Act
        ordered = [name for name, _, _ in h.Discovery.scan_module(module)]

        # Assert
        tm.that(ordered, eq=["high", "low"])

    def test_scan_module_breaks_priority_ties_by_name(self) -> None:
        """Equal-priority functions use their names as a stable tie breaker."""
        # Arrange
        module = types.ModuleType("tie_break_module")

        @h.handler(command=_CreateCommand, priority=5)
        def bravo(cmd: _CreateCommand) -> p.Result[str]:
            _ = cmd
            return r[str].ok("b")

        @h.handler(command=_CreateCommand, priority=5)
        def alpha(cmd: _CreateCommand) -> p.Result[str]:
            _ = cmd
            return r[str].ok("a")

        setattr(module, "bravo", bravo)
        setattr(module, "alpha", alpha)

        # Act
        ordered = [name for name, _, _ in h.Discovery.scan_module(module)]

        # Assert
        tm.that(ordered, eq=["alpha", "bravo"])

    def test_scan_module_surfaces_decorator_metadata(self) -> None:
        """Discovery preserves the handler's public decorator metadata."""
        # Arrange
        module = types.ModuleType("metadata_module")

        @h.handler(command=_DeleteCommand, priority=7)
        def handle(cmd: _DeleteCommand) -> p.Result[str]:
            _ = cmd
            return r[str].ok("done")

        setattr(module, "handle", handle)

        # Act
        _, _, config = h.Discovery.scan_module(module)[0]

        # Assert
        tm.that(config.command is _DeleteCommand, eq=True)
        tm.that(config.priority, eq=7)

    def test_scan_module_returns_empty_for_module_without_handlers(self) -> None:
        """A module without decorated handlers yields an empty sequence."""
        # Arrange
        module = types.ModuleType("bare_module")
        setattr(module, "value", 42)

        # Act
        handlers = h.Discovery.scan_module(module)

        # Assert
        tm.that(len(handlers), eq=0)

    @pytest.mark.parametrize(
        ("returned", "expected"),
        [(42, 42), ("payload", "payload"), (None, None), ([1, 2], "[1, 2]")],
    )
    def test_discovered_callable_coerces_result_to_scalar_or_none(
        self, returned: t.JsonValue, expected: t.Scalar | None
    ) -> None:
        """The discovered callable exposes the documented scalar coercion."""
        # Arrange
        module = types.ModuleType("coercion_module")

        @h.handler(command=_CreateCommand)
        def produce(cmd: _CreateCommand) -> t.JsonValue:
            _ = cmd
            return returned

        setattr(module, "produce", produce)
        _, discovered, _ = h.Discovery.scan_module(module)[0]
        invoke: Callable[..., t.Scalar | None] = discovered

        # Act
        result = invoke({})

        # Assert
        tm.that(result, eq=expected)
