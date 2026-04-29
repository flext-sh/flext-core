"""Message dispatch orchestration with reliability features.

Coordinates command and query execution with routing, reliability policies,
and observability features for handler registration and execution.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from collections.abc import (
    MutableSequence,
    Sequence,
)

from flext_core import (
    FlextConstants as c,
    FlextProtocols as p,
    FlextResult as r,
    FlextTypes as t,
    FlextUtilities as u,
)
from flext_core._utilities.dispatcher_execute import execute_dispatcher_handler


class FlextDispatcher:
    """Application-level dispatcher that satisfies the command bus protocol.

    The dispatcher exposes CQRS routing, handler registration, and execution
    with layered reliability controls for message dispatching and handler execution.
    """

    def __init__(self) -> None:
        """Initialize dispatcher."""
        super().__init__()
        self.logger = u.fetch_logger(__name__)
        self._handlers: t.RegistryDict[
            tuple[t.DispatchableHandler, t.RoutedHandlerCallable]
        ] = {}
        self._auto_handlers: MutableSequence[
            tuple[
                t.DispatchableHandler,
                t.RoutedHandlerCallable,
                tuple[t.TypeHintSpecifier, ...],
            ]
        ] = []
        self._event_subscribers: t.RegistryDict[
            MutableSequence[tuple[t.DispatchableHandler, t.RoutedHandlerCallable]]
        ] = {}

    def dispatch(self, message: p.Routable) -> p.Result[t.JsonPayload]:
        """Dispatch a CQRS message to its registered handler.

        Args:
            message: The Pydantic model representing command or query.

        Returns:
            Result containing the handler result or failure message.

        """
        try:
            route_name = u.resolve_message_route(message)
        except (TypeError, ValueError) as exc:
            return r[t.JsonPayload].fail_op("dispatch message", exc)
        handler_entry = self._handlers.get(route_name)
        if not handler_entry:
            msg_type = message.__class__
            for auto_h, resolved_handler, accepted in self._auto_handlers:
                if u.can_handle_message_type(accepted, msg_type) or (
                    isinstance(auto_h, p.AutoDiscoverableHandler)
                    and auto_h.can_handle(msg_type)
                ):
                    handler_entry = (auto_h, resolved_handler)
                    break
        if not handler_entry:
            return r[t.JsonPayload].fail_op(
                "resolve message handler",
                f"No handler found for {route_name}",
            )
        _, resolved_handler = handler_entry
        return self._execute_handler(resolved_handler, message, route_name)

    def publish(self, event: p.Routable | Sequence[p.Routable]) -> p.Result[bool]:
        """Publish events to all registered subscribers.

        Args:
            event: Single event model or list of event models.

        Returns:
            Result indicating if publication started successfully.

        """
        if isinstance(event, Sequence):
            for evt in event:
                _ = self.publish(evt)
            return r[bool].ok(True)
        route_name = u.resolve_message_route(event)
        handlers = self._event_subscribers.get(route_name, [])
        evt_type = event.__class__
        for auto_h, resolved_handler, accepted in self._auto_handlers:
            if (
                u.can_handle_message_type(accepted, evt_type)
                or (
                    isinstance(auto_h, p.AutoDiscoverableHandler)
                    and auto_h.can_handle(evt_type)
                )
            ) and all(existing_handler != auto_h for existing_handler, _ in handlers):
                handlers.append((auto_h, resolved_handler))
        if not handlers:
            return r[bool].ok(True)
        for _, resolved_handler in handlers:
            _ = self._execute_handler(resolved_handler, event, route_name)
        return r[bool].ok(True)

    def register_handler(
        self,
        handler: t.DispatchableHandler,
        *,
        is_event: bool = False,
    ) -> p.Result[bool]:
        """Register a handler for a specific message type.

        Args:
            handler: A callable or canonical handler value with handle/can_handle methods.
                     Must expose message_type, event_type, or can_handle
                     for route discovery.
            is_event: If True, register as event subscriber.

        Returns:
            r[bool]: Success or failure with error message.

        """
        route_name: str | None = None
        accepted_message_types: tuple[t.TypeHintSpecifier, ...] = tuple(
            u.compute_accepted_message_types(handler.__class__),
        )
        resolved_handler: t.RoutedHandlerCallable
        is_auto_discoverable = isinstance(handler, p.AutoDiscoverableHandler)
        match handler:
            case p.DispatchMessage():
                resolved_handler = handler.dispatch_message
            case p.Handle():
                resolved_handler = handler.handle
            case p.Execute():
                resolved_handler = handler.execute
            case callable_handler if callable(callable_handler):
                resolved_handler = callable_handler
            case _:
                return r[bool].fail_op(
                    "register handler",
                    c.ERR_HANDLER_MUST_BE_CALLABLE,
                )
        handler_message_type = getattr(handler, "message_type", None)
        route_candidates: tuple[t.TypeHintSpecifier | str | None, ...] = (
            handler_message_type,
            accepted_message_types[0] if accepted_message_types else None,
        )
        for candidate in route_candidates:
            match candidate:
                case None:
                    continue
                case str() as route_text:
                    route_name = route_text
                case type() as route_type:
                    try:
                        route_name = u.resolve_message_route(route_type)
                    except (TypeError, ValueError):
                        continue
                case _:
                    continue
            break
        if route_name is None:
            if is_auto_discoverable:
                self._auto_handlers.append((
                    handler,
                    resolved_handler,
                    accepted_message_types,
                ))
                self.logger.info(
                    c.LOG_REGISTERED_AUTO_DISCOVERY_HANDLER,
                    handler=str(handler),
                )
                return r[bool].ok(True)
            return r[bool].fail_op(
                "discover handler route",
                c.ERR_HANDLER_ROUTE_DISCOVERY_REQUIRED,
            )
        if is_event:
            self._event_subscribers.setdefault(route_name, []).append((
                handler,
                resolved_handler,
            ))
            self.logger.info(c.LOG_REGISTERED_EVENT_SUBSCRIBER, route=route_name)
        else:
            self._handlers[route_name] = (handler, resolved_handler)
            self.logger.info(c.LOG_REGISTERED_HANDLER, route=route_name)
        return r[bool].ok(True)

    def _execute_handler(
        self,
        resolved_handler: t.RoutedHandlerCallable,
        message: p.Routable,
        route_name: str,
    ) -> p.Result[t.JsonPayload]:
        """Execute a handler against a message via the extracted helper."""
        return execute_dispatcher_handler(
            resolved_handler=resolved_handler,
            message=message,
            route_name=route_name,
            logger=self.logger,
        )
