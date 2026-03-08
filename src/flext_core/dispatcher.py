"""Message dispatch orchestration with reliability features.

Coordinates command and query execution with routing, reliability policies,
and observability features for handler registration and execution.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

from pydantic import BaseModel

from flext_core import c, p, r, t
from flext_core.loggings import FlextLogger


@runtime_checkable
class DispatchMessageProtocol(Protocol):
    """Protocol for objects that can dispatch messages."""

    def dispatch_message(self, message: p.Routable) -> t.ContainerValue:
        """Dispatch a message."""
        ...


@runtime_checkable
class HandleProtocol(Protocol):
    """Protocol for objects that can handle messages."""

    def handle(self, message: p.Routable) -> t.ContainerValue:
        """Handle a message."""
        ...


@runtime_checkable
class ExecuteProtocol(Protocol):
    """Protocol for objects that can execute messages."""

    def execute(self, message: p.Routable) -> t.ContainerValue:
        """Execute a message."""
        ...


# Union of all handler types the dispatcher accepts
# Callables returning ResultLike (FlextResult) or ContainerValue, plus protocol objects
_DispatchableHandler: TypeAlias = (
    Callable[..., p.ResultLike[t.ContainerValue] | t.ContainerValue | None]
    | DispatchMessageProtocol
    | HandleProtocol
    | ExecuteProtocol
)

_DispatchValueT = TypeVar("_DispatchValueT")


class FlextDispatcher:
    """Application-level dispatcher that satisfies the command bus protocol.

    The dispatcher exposes CQRS routing, handler registration, and execution
    with layered reliability controls for message dispatching and handler execution.
    """

    def __init__(self) -> None:
        """Initialize dispatcher."""
        super().__init__()
        self._logger = FlextLogger.create_module_logger(__name__)

        self._handlers: MutableMapping[str, _DispatchableHandler] = {}
        self._auto_handlers: list[_DispatchableHandler] = []
        self._event_subscribers: MutableMapping[str, list[_DispatchableHandler]] = {}

    def dispatch(
        self,
        message: p.Routable,
    ) -> r[t.ContainerValue]:
        """Dispatch a CQRS message to its registered handler.

        Args:
            message: The Pydantic model representing command or query.

        Returns:
            Result containing the handler result or failure message.

        """
        try:
            route_name = self._resolve_route(message)
        except (TypeError, ValueError) as e:
            return r[t.ContainerValue].fail(
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
            return r[t.ContainerValue].fail(
                f"No handler found for {route_name}",
                error_code=c.Errors.COMMAND_HANDLER_NOT_FOUND,
            )

        return self._execute_handler(handler, message, route_name)

    def dispatch_typed(
        self,
        message: p.Routable,
        expected_type: type[_DispatchValueT],
    ) -> r[_DispatchValueT]:
        """Dispatch a message and return a strongly typed payload.

        This is the canonical ergonomic API for examples and application code that
        expects a concrete payload type and should avoid ad-hoc narrowing logic.
        """
        return self.dispatch(message).flat_map(
            lambda value: self._coerce_dispatch_value(value, expected_type)
        )

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
                _ = self.publish(e)
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
            _ = self._execute_handler(handler, event, route_name)

        return r[bool].ok(value=True)

    def register_handler(
        self,
        handler: _DispatchableHandler,
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
            self._logger.info("Registered auto-discovery handler", handler=str(handler))
            return r[bool].ok(value=True)
        else:
            return r[bool].fail(
                "Handler must expose message_type, event_type, or can_handle",
            )

        if is_event:
            if route_name not in self._event_subscribers:
                self._event_subscribers[route_name] = []
            self._event_subscribers[route_name].append(handler)
            self._logger.info("Registered event subscriber", route=route_name)
        else:
            self._handlers[route_name] = handler
            self._logger.info("Registered handler", route=route_name)

        return r[bool].ok(value=True)

    def _execute_handler(
        self,
        handler: _DispatchableHandler,
        message: p.Routable,
        route_name: str,
    ) -> r[t.ContainerValue]:
        """Execute a handler against a message.

        Supports handlers with dispatch_message, handle, execute methods,
        or plain callables.
        """
        result_raw: p.ResultLike[t.ContainerValue] | t.ContainerValue | None = None
        try:
            if isinstance(handler, DispatchMessageProtocol):
                result_raw = handler.dispatch_message(message)
            elif isinstance(handler, HandleProtocol):
                result_raw = handler.handle(message)
            elif isinstance(handler, ExecuteProtocol):
                result_raw = handler.execute(message)
            elif callable(handler):
                result_raw = handler(
                    message,
                )

            # Handle ResultLike returns natively
            if isinstance(result_raw, p.ResultLike):
                result_like = result_raw
                if result_like.is_failure:
                    error_data_value = result_like.error_data
                    return r[t.ContainerValue].fail(
                        result_like.error or "Handler failed",
                        error_code=result_like.error_code,
                        error_data=error_data_value
                        if isinstance(error_data_value, BaseModel)
                        else None,
                    )
                value = result_like.value
                if value is None:
                    return r[t.ContainerValue].fail(
                        "Handler returned None in success result"
                    )
                return r[t.ContainerValue].ok(value)

            # Bare value return
            if result_raw is None:
                return r[t.ContainerValue].fail("Handler returned None")
            return r[t.ContainerValue].ok(result_raw)

        except Exception as exc:
            self._logger.exception("Handler execution failed", route=route_name)
            return r[t.ContainerValue].fail(
                f"Handler execution failed: {exc}",
                error_code=c.Errors.COMMAND_PROCESSING_FAILED,
            )

    @staticmethod
    def _coerce_dispatch_value(
        value: t.ContainerValue,
        expected_type: type[_DispatchValueT],
    ) -> r[_DispatchValueT]:
        """Coerce dispatcher payload to expected type with typed result."""
        if isinstance(value, expected_type):
            return r[_DispatchValueT].ok(value)
        return r[_DispatchValueT].fail(
            f"Dispatch returned {type(value).__name__}; expected {expected_type.__name__}",
            error_code=c.Errors.VALIDATION_ERROR,
        )

    def _protocol_name(self) -> str:
        """Return the protocol name for introspection."""
        return "core-command-bus"

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
            model_fields = getattr(msg, "model_fields", None)
            if model_fields:
                for attr in ["command_type", "query_type", "event_type"]:
                    if attr in model_fields:
                        field = model_fields[attr]
                        default_val = getattr(field, "default", None)
                        if (
                            isinstance(default_val, str)
                            and default_val
                            and default_val != "PydanticUndefined"
                        ):
                            return default_val

        msg_type_error = (
            f"Message {msg} does not provide a valid route via "
            "command_type, query_type, or event_type"
        )
        raise TypeError(msg_type_error)
