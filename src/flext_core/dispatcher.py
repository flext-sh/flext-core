"""Message dispatch orchestration with reliability features.

Coordinates command and query execution with routing, reliability policies,
and observability features for handler registration and execution.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, MutableSequence, Sequence

from pydantic import BaseModel

from flext_core import FlextLogger, c, p, r, t, u

type DispatcherResolvedCallable = Callable[
    [p.Routable],
    t.RuntimeAtomic | p.ResultLike[t.RuntimeAtomic] | None,
]


class FlextDispatcher:
    """Application-level dispatcher that satisfies the command bus protocol.

    The dispatcher exposes CQRS routing, handler registration, and execution
    with layered reliability controls for message dispatching and handler execution.
    """

    def __init__(self) -> None:
        """Initialize dispatcher."""
        super().__init__()
        self._logger = FlextLogger.create_module_logger(__name__)
        self._handlers: t.RegistryDict[
            tuple[t.HandlerProtocolVariant, DispatcherResolvedCallable]
        ] = {}
        self._auto_handlers: MutableSequence[
            tuple[
                t.HandlerProtocolVariant,
                DispatcherResolvedCallable,
                tuple[t.MessageTypeSpecifier, ...],
            ]
        ] = []
        self._event_subscribers: t.RegistryDict[
            MutableSequence[tuple[t.HandlerProtocolVariant, DispatcherResolvedCallable]]
        ] = {}

    def dispatch(self, message: p.Routable) -> r[t.RuntimeAtomic]:
        """Dispatch a CQRS message to its registered handler.

        Args:
            message: The Pydantic model representing command or query.

        Returns:
            Result containing the handler result or failure message.

        """
        try:
            route_name = u.get_message_route(message)
        except (TypeError, ValueError) as e:
            return r[t.RuntimeAtomic].fail(
                f"Dispatch failed: {e!s}",
                error_code=c.COMMAND_PROCESSING_FAILED,
            )
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
            return r[t.RuntimeAtomic].fail(
                f"No handler found for {route_name}",
                error_code=c.COMMAND_HANDLER_NOT_FOUND,
            )
        _, resolved_handler = handler_entry
        return self._execute_handler(resolved_handler, message, route_name)

    def dispatch_typed[DispatchValueT](
        self,
        message: p.Routable,
        expected_type: type[DispatchValueT],
    ) -> r[DispatchValueT]:
        """Dispatch a message and return a strongly typed payload.

        This is the canonical ergonomic API for examples and application code that
        expects a concrete payload type and should avoid ad-hoc narrowing logic.
        """

        def _coerce(value: t.RuntimeAtomic) -> r[DispatchValueT]:
            if isinstance(value, expected_type):
                return r[DispatchValueT].ok(value)
            if isinstance(value, BaseModel):
                return r[DispatchValueT].fail(
                    f"Expected {expected_type.__name__}, got {value.__class__.__name__}",
                )
            return u.parse(value, expected_type)

        return self.dispatch(message).flat_map(_coerce)

    def publish(self, event: p.Routable | Sequence[p.Routable]) -> r[bool]:
        """Publish events to all registered subscribers.

        Args:
            event: Single event model or list of event models.

        Returns:
            Result indicating if publication started successfully.

        """
        if isinstance(event, Sequence):
            for e in event:
                _ = self.publish(e)
            return r[bool].ok(value=True)
        route_name = u.get_message_route(event)
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
            return r[bool].ok(value=True)
        for _, resolved_handler in handlers:
            _ = self._execute_handler(resolved_handler, event, route_name)
        return r[bool].ok(value=True)

    def register_handler(
        self,
        handler: t.HandlerProtocolVariant,
        *,
        is_event: bool = False,
    ) -> r[bool]:
        """Register a handler for a specific message type.

        Args:
            handler: A callable or t.NormalizedValue with handle/can_handle methods.
                     Must expose message_type, event_type, or can_handle
                     for route discovery.
            is_event: If True, register as event subscriber.

        Returns:
            r[bool]: Success or failure with error message.

        """
        route_name: str | None = None
        accepted_message_types: tuple[t.MessageTypeSpecifier, ...] = tuple(
            u.compute_accepted_message_types(handler.__class__),
        )
        resolved_handler: DispatcherResolvedCallable
        if isinstance(handler, p.DispatchMessage):
            resolved_handler = handler.dispatch_message
        elif isinstance(handler, p.Handle):
            resolved_handler = handler.handle
        elif isinstance(handler, p.Execute):
            resolved_handler = handler.execute
        else:
            if not callable(handler):
                return r[bool].fail("Handler must be callable")
            resolved_handler = handler
        handler_message_type = getattr(handler, "message_type", None)
        if isinstance(handler_message_type, str):
            route_name = handler_message_type
        elif handler_message_type is not None:
            with contextlib.suppress(TypeError, ValueError):
                route_name = u.get_message_route(handler_message_type)
        if route_name is None and accepted_message_types:
            with contextlib.suppress(TypeError, ValueError):
                route_name = u.get_message_route(accepted_message_types[0])
        if route_name is None:
            if isinstance(handler, p.AutoDiscoverableHandler):
                self._auto_handlers.append((
                    handler,
                    resolved_handler,
                    accepted_message_types,
                ))
                self._logger.info(
                    "Registered auto-discovery handler",
                    handler=str(handler),
                )
                return r[bool].ok(value=True)
            return r[bool].fail(
                "Handler must expose message_type, event_type, or can_handle",
            )
        if is_event:
            if route_name not in self._event_subscribers:
                self._event_subscribers[route_name] = []
            self._event_subscribers[route_name].append((handler, resolved_handler))
            self._logger.info("Registered event subscriber", route=route_name)
        else:
            self._handlers[route_name] = (handler, resolved_handler)
            self._logger.info("Registered handler", route=route_name)
        return r[bool].ok(value=True)

    def _execute_handler(
        self,
        resolved_handler: DispatcherResolvedCallable,
        message: p.Routable,
        route_name: str,
    ) -> r[t.RuntimeAtomic]:
        """Execute a handler against a message."""
        dispatch_result = r[t.RuntimeAtomic]
        try:
            raw_output = resolved_handler(message)
            if u.is_result_like(raw_output):
                if raw_output.is_failure:
                    error_data_value = raw_output.error_data
                    return dispatch_result.fail(
                        raw_output.error or "Handler failed",
                        error_code=raw_output.error_code,
                        error_data=error_data_value
                        if isinstance(error_data_value, BaseModel)
                        else None,
                    )
                value: t.RuntimeAtomic | None = raw_output.value
                if not u.is_container(value):
                    return dispatch_result.fail(
                        "Handler returned non-container value in success result",
                    )
                return dispatch_result.ok(value)
            if raw_output is None:
                return dispatch_result.fail("Handler returned None")
            if not u.is_container(raw_output):
                return dispatch_result.fail("Handler returned non-container value")
            return dispatch_result.ok(raw_output)
        except (
            TypeError,
            ValueError,
            RuntimeError,
            KeyError,
            AttributeError,
            OSError,
            LookupError,
            ArithmeticError,
        ) as exc:
            self._logger.exception("Handler execution failed", route=route_name)
            return dispatch_result.fail(
                f"Handler execution failed: {exc}",
                error_code=c.COMMAND_PROCESSING_FAILED,
            )
