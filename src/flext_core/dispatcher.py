"""Message dispatch orchestration with reliability features.

Coordinates command and query execution with routing, reliability policies,
and observability features for handler registration and execution.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence

from flext_core import c, p, r, t, u


class FlextDispatcher:
    """Application-level dispatcher that satisfies the command bus protocol.

    The dispatcher exposes CQRS routing, handler registration, and execution
    with layered reliability controls for message dispatching and handler execution.
    """

    def __init__(self) -> None:
        """Initialize dispatcher."""
        super().__init__()
        self._logger = u.fetch_logger(__name__)
        self._handlers: t.RegistryDict[
            tuple[t.HandlerProtocolVariant, t.RoutedHandlerCallable]
        ] = {}
        self._auto_handlers: MutableSequence[
            tuple[
                t.HandlerProtocolVariant,
                t.RoutedHandlerCallable,
                tuple[t.TypeHintSpecifier, ...],
            ]
        ] = []
        self._event_subscribers: t.RegistryDict[
            MutableSequence[tuple[t.HandlerProtocolVariant, t.RoutedHandlerCallable]]
        ] = {}

    def dispatch(self, message: p.Routable) -> p.Result[t.RuntimeAtomic]:
        """Dispatch a CQRS message to its registered handler.

        Args:
            message: The Pydantic model representing command or query.

        Returns:
            Result containing the handler result or failure message.

        """
        try:
            route_name = u.resolve_message_route(message)
        except (TypeError, ValueError) as exc:
            return r[t.RuntimeAtomic].fail_op(
                "dispatch message",
                exc,
                error_code=c.ErrorCode.COMMAND_PROCESSING_FAILED.value,
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
            return r[t.RuntimeAtomic].fail_op(
                "resolve message handler",
                f"No handler found for {route_name}",
                error_code=c.ErrorCode.COMMAND_HANDLER_NOT_FOUND.value,
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
        handler: t.HandlerProtocolVariant,
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
        match handler:
            case p.DispatchMessage():
                resolved_handler = handler.dispatch_message
            case p.Handle():
                resolved_handler = handler.handle
            case p.Execute():
                resolved_handler = handler.execute
            case _:
                if not callable(handler):
                    return r[bool].fail_op(
                        "register handler",
                        c.ERR_HANDLER_MUST_BE_CALLABLE,
                    )
                resolved_handler = handler
        handler_message_type = getattr(handler, "message_type", None)
        if isinstance(handler_message_type, str):
            route_name = handler_message_type
        elif handler_message_type is not None:
            try:
                route_name = u.resolve_message_route(handler_message_type)
            except (TypeError, ValueError):
                route_name = None
        if route_name is None and accepted_message_types:
            first_accepted = accepted_message_types[0]
            if isinstance(first_accepted, (str, type)):
                try:
                    route_name = u.resolve_message_route(first_accepted)
                except (TypeError, ValueError):
                    route_name = None
        if route_name is None:
            if isinstance(handler, p.AutoDiscoverableHandler):
                self._auto_handlers.append((
                    handler,
                    resolved_handler,
                    accepted_message_types,
                ))
                self._logger.info(
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
            self._logger.info(c.LOG_REGISTERED_EVENT_SUBSCRIBER, route=route_name)
        else:
            self._handlers[route_name] = (handler, resolved_handler)
            self._logger.info(c.LOG_REGISTERED_HANDLER, route=route_name)
        return r[bool].ok(True)

    def _execute_handler(
        self,
        resolved_handler: t.RoutedHandlerCallable,
        message: p.Routable,
        route_name: str,
    ) -> p.Result[t.RuntimeAtomic]:
        """Execute a handler against a message."""
        dispatch_result = r[t.RuntimeAtomic]
        try:
            raw_output = resolved_handler(message)
            if u.result_like(raw_output):
                if raw_output.failure:
                    error_data_value = raw_output.error_data
                    return dispatch_result.fail(
                        raw_output.error or c.ERR_HANDLER_FAILED,
                        error_code=raw_output.error_code,
                        error_data=error_data_value
                        if u.pydantic_model(error_data_value)
                        else None,
                    )
                value: t.RuntimeAtomic | None = raw_output.value
                if not u.container(value) and not u.pydantic_model(value):
                    return dispatch_result.fail_op(
                        "validate handler success payload",
                        c.ERR_HANDLER_RETURNED_NON_CONTAINER_SUCCESS_RESULT,
                    )
                return dispatch_result.ok(value)
            if raw_output is None:
                return dispatch_result.fail_op(
                    "execute resolved handler",
                    c.ERR_HANDLER_RETURNED_NONE,
                )
            if not u.container(raw_output) and not u.pydantic_model(raw_output):
                return dispatch_result.fail_op(
                    "validate handler return payload",
                    c.ERR_HANDLER_RETURNED_NON_CONTAINER_VALUE,
                )
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
            self._logger.exception(c.LOG_HANDLER_EXECUTION_FAILED, route=route_name)
            return dispatch_result.fail_op(
                "execute resolved handler",
                exc,
                error_code=c.ErrorCode.COMMAND_PROCESSING_FAILED.value,
            )
