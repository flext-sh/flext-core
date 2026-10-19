"""Atomic handler registration and snapshot-based message dispatch.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from threading import Lock
from types import MappingProxyType
from typing import cast

from flext_core import c, m, p, r, t, u


class FlextDispatcher:
    """Own routes and publish complete immutable snapshots before execution."""

    def __init__(self) -> None:
        """Initialize an isolated route store and its publication lock."""
        self.logger = u.fetch_logger(__name__)
        self._lock = Lock()
        self._routes: t.MappingKV[str | None, tuple[p.DispatcherRouteEntry, ...]] = (
            MappingProxyType({})
        )

    def dispatch(self, message: p.Routable) -> p.Result[t.JsonPayload]:
        """Execute a command against one snapshot without holding its lock.

        Returns:
            The selected handler's result or a routing failure.

        """
        routes = self._routes
        try:
            route = u.resolve_message_route(message)
            for key in (route, None):
                for handler, resolved, accepted, is_event, _ in routes.get(key, ()):
                    if not is_event and (
                        key is not None
                        or u.can_handle_message_type(accepted, type(message))
                        or (
                            isinstance(handler, p.AutoDiscoverableHandler)
                            and handler.can_handle(type(message))
                        )
                    ):
                        return u.execute_dispatcher_handler(
                            resolved_handler=resolved,
                            message=message,
                            route_name=route,
                            logger=self.logger,
                        )
        except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
            return r[t.JsonPayload].fail_op("dispatch message", exc)
        return r[t.JsonPayload].fail_op(
            "resolve message handler", f"No handler found for {route}"
        )

    def publish(self, event: p.Routable | t.SequenceOf[p.Routable]) -> p.Result[bool]:
        """Publish in input/registration order, stopping at the first failure.

        All events in a sequence use the same snapshot. Earlier external effects
        are retained when a later subscriber fails.

        Returns:
            Success after all subscribers, or the first failure.

        """
        routes = self._routes
        events = (
            cast("t.SequenceOf[p.Routable]", event)
            if isinstance(event, Sequence)
            else (event,)
        )
        try:
            for message in events:
                result = self._publish_message(message, routes)
                if result.failure:
                    return result
        except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
            return r[bool].fail_op("publish event", exc)
        return r[bool].ok(value=True)

    def _publish_message(
        self,
        message: p.Routable,
        routes: t.MappingKV[str | None, tuple[p.DispatcherRouteEntry, ...]],
    ) -> p.Result[bool]:
        """Publish one message against the caller's immutable route snapshot.

        Returns:
            Subscriber success or the first handler failure.

        """
        route = u.resolve_message_route(message)
        invoked: list[p.DispatchableHandler] = []
        subscribers = sorted(
            ((key, entry) for key in (route, None) for entry in routes.get(key, ())),
            key=lambda item: item[1][4],
        )
        for key, (handler, resolved, accepted, is_event, _) in subscribers:
            if not is_event or any(
                u.registry_same_value(handler, previous) for previous in invoked
            ):
                continue
            if key is None and not (
                u.can_handle_message_type(accepted, type(message))
                or (
                    isinstance(handler, p.AutoDiscoverableHandler)
                    and handler.can_handle(type(message))
                )
            ):
                continue
            invoked.append(handler)
            result = u.execute_dispatcher_handler(
                resolved_handler=resolved,
                message=message,
                route_name=route,
                logger=self.logger,
            )
            if result.failure:
                return r[bool].from_failure(result)
        return r[bool].ok(value=True)

    def register_handler(
        self,
        handler: p.DispatchableHandler,
        *,
        message_type: t.RegistryBindingKey | None = None,
        is_event: bool = False,
    ) -> p.Result[bool]:
        """Register one binding through the canonical atomic batch operation.

        Returns:
            Registration success or the batch validation failure.

        """
        return self.register_handlers((
            m.HandlerRegistration(
                handler=handler,
                message_type=message_type,
                is_event=is_event,
            ),
        )).map(lambda _report: True)

    def register_handlers(
        self, bindings: t.SequenceOf[p.HandlerRegistration]
    ) -> p.Result[p.HandlerRegistrationReport]:
        """Validate the entire batch, then atomically publish new bindings.

        The report distinguishes identical repeats. Conflicts never replace a
        command handler or expose a partially registered batch to readers.

        Returns:
            The atomic registration report or validation failure.

        """
        prepared: list[
            tuple[str | None, p.DispatcherRouteEntry, p.HandlerRegistration]
        ] = []
        try:
            for binding in bindings:
                item = self._prepare_binding(binding, len(prepared))
                if item.failure:
                    return r[p.HandlerRegistrationReport].from_failure(item)
                prepared.append(item.value)
        except c.CATCHABLE_RUNTIME_EXCEPTIONS as exc:
            return r[p.HandlerRegistrationReport].fail_op("register handlers", exc)
        with self._lock:
            candidate = dict(self._routes)
            offset = sum(len(entries) for entries in candidate.values())
            registered: list[p.HandlerRegistration] = []
            skipped: list[p.HandlerRegistration] = []
            for route, entry, binding in prepared:
                existing = candidate.get(route, ())
                if any(
                    item[3] == entry[3] and u.registry_same_value(item[0], entry[0])
                    for item in existing
                ):
                    skipped.append(binding)
                    continue
                if (
                    route is not None
                    and not entry[3]
                    and any(not item[3] for item in existing)
                ):
                    return r[p.HandlerRegistrationReport].fail_op(
                        "register handlers", f"Handler already registered for {route}"
                    )
                candidate[route] = (*existing, (*entry[:4], offset + len(registered)))
                registered.append(binding)
            report = m.HandlerRegistrationReport(
                registered=tuple(registered),
                skipped=tuple(skipped),
            )
            self._routes = MappingProxyType(candidate)
        return r[p.HandlerRegistrationReport].ok(report)

    @staticmethod
    def _normalize_binding(
        binding: p.HandlerRegistration,
        accepted: tuple[type, ...],
    ) -> p.Result[p.HandlerRegistration]:
        """Resolve declarations without publishing an incomplete route.

        Returns:
            The normalized binding or a contradictory-route failure.

        """
        handler = binding.handler
        metadata = u.handler_metadata(handler, name=type(handler).__qualname__)
        route = (
            u.resolve_message_route(binding.message_type)
            if binding.message_type is not None
            else None
        )
        for declaration in (
            getattr(handler, "message_type", None),
            metadata.command if metadata is not None else None,
        ):
            if declaration is not None:
                declared_route = u.resolve_message_route(declaration)
                if route is not None and route != declared_route:
                    return r[p.HandlerRegistration].fail_op(
                        "bind handler route",
                        "Explicit route contradicts handler metadata",
                    )
                route = declared_route
        autodiscoverable = isinstance(handler, p.AutoDiscoverableHandler)
        if route is None and accepted and not autodiscoverable:
            route = u.resolve_message_route(accepted[0])
        if not route and (route is not None or not autodiscoverable):
            return r[p.HandlerRegistration].fail_op(
                "discover handler route", c.ERR_HANDLER_ROUTE_DISCOVERY_REQUIRED
            )
        return r[p.HandlerRegistration].ok(
            m.HandlerRegistration(
                handler=handler,
                message_type=route,
                is_event=binding.is_event
                or (
                    isinstance(handler, p.Handler)
                    and handler.mode == c.HandlerType.EVENT
                ),
            )
        )

    @staticmethod
    def _prepare_binding(
        binding: p.HandlerRegistration,
        index: int,
    ) -> p.Result[tuple[str | None, p.DispatcherRouteEntry, p.HandlerRegistration]]:
        """Prepare one validated binding before acquiring the publication lock.

        Returns:
            The prepared route entry or its validation failure.

        """
        if not isinstance(binding, p.HandlerRegistration):
            return r[
                tuple[str | None, p.DispatcherRouteEntry, p.HandlerRegistration]
            ].fail_op("register handlers", "Invalid handler registration")
        handler = binding.handler
        resolved: p.RoutedHandlerCallable
        match handler:
            case p.DispatchMessage():
                resolved = handler.dispatch_message
            case p.Handle():
                resolved = handler.handle
            case p.Execute():
                resolved = handler.execute
            case callable_handler if callable(callable_handler):
                resolved = callable_handler
            case _:
                return r[
                    tuple[str | None, p.DispatcherRouteEntry, p.HandlerRegistration]
                ].fail_op("register handler", c.ERR_HANDLER_MUST_BE_CALLABLE)
        accepted = tuple(u.compute_accepted_message_types(type(handler)))
        normalized = FlextDispatcher._normalize_binding(binding, accepted)
        if normalized.failure:
            return r[
                tuple[str | None, p.DispatcherRouteEntry, p.HandlerRegistration]
            ].from_failure(normalized)
        result = normalized.value
        route = (
            u.resolve_message_route(result.message_type)
            if result.message_type is not None
            else None
        )
        return r[tuple[str | None, p.DispatcherRouteEntry, p.HandlerRegistration]].ok((
            route,
            (handler, resolved, accepted, result.is_event, index),
            result,
        ))


__all__: list[str] = ["FlextDispatcher"]
