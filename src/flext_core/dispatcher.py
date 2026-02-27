"""Message dispatch orchestration with reliability features.

Coordinates command and query execution with routing, reliability policies,
and observability features for handler registration and execution.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from typing import cast, override

from flext_core.constants import c
from flext_core.models import m
from flext_core.protocols import p
from flext_core.result import r
from flext_core.service import s
from flext_core.typings import t


class FlextDispatcher(s[bool]):
    """Application-level dispatcher that satisfies the command bus protocol.

    The dispatcher exposes CQRS routing, handler registration, and execution
    with layered reliability controls for message dispatching and handler execution.
    """

    @override
    def __init__(
        self,
        **data: t.ScalarValue | m.ConfigMap | Sequence[t.ScalarValue],
    ) -> None:
        """Initialize dispatcher."""
        super().__init__(**data)

        self._handlers: MutableMapping[str, t.HandlerType] = {}
        self._auto_handlers: list[t.HandlerType] = []
        self._event_subscribers: MutableMapping[str, list[t.HandlerType]] = {}

    @override
    def execute(self) -> r[bool]:
        """Execute service - satisfies s abstract method."""
        return r[bool].ok(value=True)

    def register_handler(
        self,
        handler: t.HandlerType,
        *,
        is_event: bool = False,
    ) -> r[bool]:
        """Register a handler for a specific message type.

        Args:
            handler: A callable or object with handle/can_handle methods.
                     Must expose message_type, event_type, or can_handle
                     for route discovery.
            is_event: If True, register as event subscriber.

        Returns:
            r[bool]: Success or failure with error message.

        """
        route_name: str | None = None

        # Extract route from handler metadata
        has_message_type = getattr(handler, "message_type", None)
        has_event_type = getattr(handler, "event_type", None)
        has_can_handle = getattr(handler, "can_handle", None)

        if has_message_type:
            route_name = self._resolve_route(has_message_type)
        elif has_event_type:
            route_name = self._resolve_route(has_event_type)
        elif callable(has_can_handle):
            self._auto_handlers.append(handler)
            self.logger.info(f"Registered auto-discovery handler: {handler}")
            return r[bool].ok(value=True)
        else:
            return r[bool].fail(
                "Handler must expose message_type, event_type, or can_handle"
            )

        if is_event:
            if route_name not in self._event_subscribers:
                self._event_subscribers[route_name] = []
            self._event_subscribers[route_name].append(handler)
            self.logger.info(f"Registered event subscriber for {route_name}")
        else:
            self._handlers[route_name] = handler
            self.logger.info(f"Registered handler for {route_name}")

        return r[bool].ok(value=True)

    def dispatch(
        self,
        message: p.Routable,
    ) -> r[t.PayloadValue]:
        """Dispatch a CQRS message to its registered handler.

        Args:
            message: The Pydantic model representing command or query.

        Returns:
            Result containing the handler result or failure message.

        """
        try:
            route_name = self._resolve_route(message)
        except (TypeError, ValueError) as e:
            return r[t.PayloadValue].fail(
                f"Dispatch failed: {e!s}",
                error_code=c.Errors.COMMAND_PROCESSING_FAILED,
            )
        handler = self._handlers.get(route_name)

        # Auto-discovery fallback
        if not handler:
            msg_type = message.__class__
            for auto_h in self._auto_handlers:
                has_can_handle = getattr(auto_h, "can_handle", None)
                if callable(has_can_handle) and has_can_handle(msg_type):
                    handler = auto_h
                    break

        if not handler:
            return r[t.PayloadValue].fail(
                f"No handler found for {route_name}",
                error_code=c.Errors.COMMAND_HANDLER_NOT_FOUND,
            )

        return self._execute_handler(handler, message, route_name)

    def publish(
        self,
        event: p.Routable | list[p.Routable],
    ) -> r[bool]:
        """Publish events to all registered subscribers.

        Args:
            event: Single event model or list of event models.

        Returns:
            Result indicating if publication started successfully.

        """
        if isinstance(event, list):
            for e in event:
                self.publish(e)
            return r[bool].ok(value=True)

        route_name = self._resolve_route(event)
        handlers = self._event_subscribers.get(route_name, [])

        # Auto-discovery fallback for events
        evt_type = event.__class__
        for auto_h in self._auto_handlers:
            has_can_handle = getattr(auto_h, "can_handle", None)
            if (
                callable(has_can_handle)
                and has_can_handle(evt_type)
                and auto_h not in handlers
            ):
                handlers.append(auto_h)

        if not handlers:
            return r[bool].ok(value=True)

        for handler in handlers:
            self._execute_handler(handler, event, route_name)

        return r[bool].ok(value=True)

    def _execute_handler(
        self,
        handler: t.HandlerType,
        message: p.Routable,
        route_name: str,
    ) -> r[t.PayloadValue]:
        """Execute a handler against a message.

        Supports handlers with dispatch_message, handle, execute methods,
        or plain callables.
        """
        try:
            if hasattr(handler, "dispatch_message"):
                result_raw = handler.dispatch_message(message)
            elif hasattr(handler, "handle"):
                result_raw = handler.handle(message)
            elif hasattr(handler, "execute"):
                result_raw = handler.execute(message)
            elif callable(handler):
                result_raw = cast("Callable[[p.Routable], t.PayloadValue]", handler)(
                    message
                )
            else:
                return r[t.PayloadValue].fail(
                    f"Handler for {route_name} is not callable"
                )

            # Handle ResultLike returns natively
            if hasattr(result_raw, "is_failure") and hasattr(result_raw, "is_success"):
                result_like = cast("p.ResultLike[t.PayloadValue]", result_raw)
                if result_like.is_failure:
                    return r[t.PayloadValue].fail(
                        result_like.error or "Handler failed",
                        error_code=result_like.error_code,
                        error_data=result_like.error_data,
                    )
                return r[t.PayloadValue].ok(result_like.value)

            # Bare value return
            return r[t.PayloadValue].ok(cast("t.PayloadValue", result_raw))

        except Exception as exc:
            self.logger.exception(f"Handler execution failed for {route_name}")
            return r[t.PayloadValue].fail(
                f"Handler execution failed: {exc}",
                error_code=c.Errors.COMMAND_PROCESSING_FAILED,
            )

    def _resolve_route(self, msg: p.Routable | type[p.Routable] | str) -> str:
        """Resolve route name strictly from Routable attributes or string."""
        if isinstance(msg, str):
            return msg

        # 1. Try instance/class attributes - only accept strings
        # Properties return the property object when accessed on the class,
        # so we filter for str.
        for attr in ["command_type", "query_type", "event_type"]:
            val = getattr(msg, attr, None)
            if isinstance(val, str) and val:
                return val

        # 2. Try Pydantic class model_fields defaults
        if isinstance(msg, type) and hasattr(msg, "model_fields"):
            for attr in ["command_type", "query_type", "event_type"]:
                if attr in msg.model_fields:
                    field = msg.model_fields[attr]
                    d = getattr(field, "default", None)
                    # Check for pydantic_core.PydanticUndefined without import
                    if (
                        d is not None
                        and str(d) != "PydanticUndefined"
                        and isinstance(d, str)
                    ):
                        return d

        msg_type_error = (
            f"Message {msg} does not provide a valid route via "
            "command_type, query_type, or event_type"
        )
        raise TypeError(msg_type_error)
