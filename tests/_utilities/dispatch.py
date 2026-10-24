"""Dispatcher test helpers for flext-core tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, override

from flext_tests import h, r, s, u

from tests import m, p, t

if TYPE_CHECKING:
    from collections.abc import Callable


class TestsFlextUtilitiesDispatchMixin:
    """Dispatcher test helpers."""

    class ForwardingMiddleware:
        """Delegate the original command to the supplied next handler."""

        def process[TResult](
            self,
            command: p.Model,
            next_handler: Callable[[p.Model], p.Result[TResult]],
        ) -> p.Result[TResult]:
            return next_handler(command)

    @staticmethod
    def handler_module(
        name: str,
        **members: p.RegisterableService | p.DispatchableHandler,
    ) -> t.RegistryModule:
        """Build an actual module for the public discovery API.

        Returns:
            A module exposing the supplied members without import patches.

        """
        module = ModuleType(name)
        for member_name, value in members.items():
            setattr(module, member_name, value)
        return module

    class HandlerService(s):
        """Real service methods consumed by the class-discovery contract."""

        @h.handler(command=m.Command, priority=100)
        def create(self, request: p.CommandRoutable) -> p.Result[str]:
            return r[str].ok(request.command_type)

        @h.handler(command=m.Event, priority=50)
        def delete(self, _request: p.Routable) -> p.Result[str]:
            return r[str].ok("deleted")

        @h.handler(command=m.Tests.RegistryQuery, priority=10)
        def query(self, _request: p.Routable) -> p.Result[str]:
            return r[str].ok("queried")

        def plain(self, _request: p.Routable) -> p.Result[str]:
            return r[str].ok("plain")

    class InheritedHandlerService(HandlerService):
        """An inherited API changes only the explicitly overridden operation."""

        @override
        @h.handler(command=m.Command, priority=200)
        def create(self, request: p.CommandRoutable) -> p.Result[str]:
            return r[str].ok("overridden")

    class PrivateHandlerService(s):
        """Class discovery preserves explicitly decorated private operations."""

        @h.handler(command=m.Command)
        def _private(self, _request: p.Routable) -> p.Result[str]:
            return r[str].ok("private")

    class DescriptorHandlerService(s):
        """Native static/class method descriptors keep their declaration metadata."""

        @staticmethod
        @h.handler(command=m.Command)
        def static(_request: p.Routable) -> p.Result[str]:
            return r[str].ok("static")

        @classmethod
        @h.handler(command=m.Event)
        def class_method(cls, _request: p.Routable) -> p.Result[str]:
            return r[str].ok("class")

    class CapabilityHandler(m.Tests.HandlerObservations):
        """Accept command models beyond the request's inferred first route."""

        def can_handle(self, message_type: type) -> bool:
            return issubclass(message_type, m.Command)

        def handle(self, message: p.Tests.InferredCommand) -> p.Result[t.JsonPayload]:
            self.handled.append(message.command_type)
            return r[t.JsonPayload].ok(message.command_type)

    class AutoSubscriber(m.Tests.HandlerObservations):
        """Observe any routable publication through the existing capability protocol."""

        def can_handle(self, _message_type: type) -> bool:
            return True

        def handle(self, message: p.Routable) -> p.Result[t.JsonPayload]:
            self.handled.append(u.resolve_message_route(message))
            return r[t.JsonPayload].ok(True)

    class Handler(h[t.JsonPayload, t.JsonPayload]):
        """Simple handler used by public registry scenarios."""

        @override
        def handle(self, message: t.JsonPayload) -> p.Result[t.JsonPayload]:
            return r[t.JsonPayload].ok(message)


__all__: list[str] = ["TestsFlextUtilitiesDispatchMixin"]
