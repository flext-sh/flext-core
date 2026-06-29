"""Dispatcher test helpers for flext-core tests."""

from __future__ import annotations

from typing import override

from flext_tests import h, r

from tests.constants import c
from tests.protocols import p
from tests.typings import t


class TestsFlextUtilitiesDispatchMixin:
    """Dispatcher test helpers."""

    # --- from test_registry_full_coverage.py ---

    class Handler(h[t.JsonPayload, t.JsonPayload]):
        """Simple handler used by public registry scenarios."""

        @override
        def handle(self, message: t.JsonPayload) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok(message)

    class FalseyDispatcher(p.Dispatcher):
        """Dispatcher that is present but reports itself as unavailable."""

        def __bool__(self) -> bool:
            """Return False to indicate dispatcher is unavailable."""
            return False

        @override
        def publish(
            self,
            event: p.Routable | t.SequenceOf[p.Routable],
        ) -> p.Result[bool]:
            _ = event
            return r[bool].ok(True)

        @override
        def register_handler(
            self,
            handler: t.DispatchableHandler,
            *,
            is_event: bool = False,
        ) -> p.Result[bool]:
            _ = handler
            _ = is_event
            return r[bool].ok(True)

        @override
        def dispatch(self, message: p.Routable) -> p.Result[t.JsonPayload]:
            _ = message
            return r[t.JsonPayload].fail(c.Tests.DISPATCHER_UNCONFIGURED)

    class FailDispatcher(p.Dispatcher):
        """Dispatcher that rejects public handler registration."""

        @override
        def publish(
            self,
            event: p.Routable | t.SequenceOf[p.Routable],
        ) -> p.Result[bool]:
            _ = event
            return r[bool].ok(True)

        @override
        def register_handler(
            self,
            handler: t.DispatchableHandler,
            *,
            is_event: bool = False,
        ) -> p.Result[bool]:
            _ = handler
            _ = is_event
            return r[bool].fail(c.Tests.DISPATCHER_FAIL)

        @override
        def dispatch(self, message: p.Routable) -> p.Result[t.JsonPayload]:
            _ = message
            return r[t.JsonPayload].fail(c.Tests.DISPATCHER_FAIL)

    class OkDispatcher(p.Dispatcher):
        """Dispatcher that accepts public registry operations."""

        @override
        def publish(
            self,
            event: p.Routable | t.SequenceOf[p.Routable],
        ) -> p.Result[bool]:
            _ = event
            return r[bool].ok(True)

        @override
        def register_handler(
            self,
            handler: t.DispatchableHandler,
            *,
            is_event: bool = False,
        ) -> p.Result[bool]:
            _ = handler
            _ = is_event
            return r[bool].ok(True)

        @override
        def dispatch(self, message: p.Routable) -> p.Result[t.JsonPayload]:
            _ = message
            return r[t.JsonPayload].ok(True)


__all__: list[str] = ["TestsFlextUtilitiesDispatchMixin"]
