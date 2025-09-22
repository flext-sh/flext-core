"""Dispatcher facade delivering the Phase 1 unified dispatcher charter.

The façade wraps ``FlextBus`` so handler registration, context propagation, and
metadata-aware dispatch all match the expectations documented in ``README.md``
and ``docs/architecture.md`` for the 1.0.0 modernization programme.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import Token
from typing import Literal, cast

from flext_core.bus import FlextBus
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.context import FlextContext
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class FlextDispatcher:
    """Orchestrates CQRS execution while enforcing context-first observability.

    The dispatcher is the front door promoted across the ecosystem: all handler
    registration flows, context scoping, and dispatch telemetry align with the
    modernization plan so downstream packages can adopt a consistent runtime
    contract without bespoke buses.
    """

    def __init__(
        self,
        *,
        config: dict[str, object] | None = None,
        bus: FlextBus | None = None,
    ) -> None:
        """Initialize dispatcher with Pydantic configuration model.

        Args:
            config: Optional dispatcher configuration model
            bus: Optional bus instance (created if not provided)

        """
        # Use provided config or create from global configuration
        if config is None:
            global_config = FlextConfig.get_global_instance()
            config = {
                "auto_context": getattr(global_config, "dispatcher_auto_context", True),
                "timeout_seconds": getattr(
                    global_config,
                    "dispatcher_timeout_seconds",
                    FlextConstants.Defaults.TIMEOUT,
                ),
                "enable_metrics": getattr(
                    global_config, "dispatcher_enable_metrics", True
                ),
                "enable_logging": getattr(
                    global_config, "dispatcher_enable_logging", True
                ),
                "bus_config": None,
            }

        self._config = config
        bus_config = config.get("bus_config")
        self._bus = bus or FlextBus.create_command_bus(
            bus_config=cast("dict[str, object] | None", bus_config)
            if isinstance(bus_config, (dict, type(None)))
            else None
        )
        self._logger = FlextLogger(self.__class__.__name__)

    @property
    def config(self) -> dict[str, object]:
        """Access the dispatcher configuration."""
        return self._config

    @property
    def bus(self) -> FlextBus:
        """Access the underlying bus implementation."""
        return self._bus

    # ------------------------------------------------------------------
    # Registration methods using structured models
    # ------------------------------------------------------------------
    def register_handler_with_request(
        self,
        request: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Register handler using structured request model.

        Args:
            request: Pydantic model containing registration details

        Returns:
            FlextResult with registration details or error

        """
        # Validate handler mode using constants
        if (
            request.get("handler_mode")
            not in FlextConstants.Dispatcher.VALID_HANDLER_MODES
        ):
            return FlextResult[dict[str, object]].fail(
                FlextConstants.Dispatcher.ERROR_INVALID_HANDLER_MODE
            )

        # Validate handler is provided
        if request.get("handler") is None:
            return FlextResult[dict[str, object]].fail(
                FlextConstants.Dispatcher.ERROR_HANDLER_REQUIRED
            )

        # Register with bus
        bus_result = (
            self._bus.register_handler(
                request.get("message_type"), request.get("handler")
            )
            if request.get("message_type")
            else self._bus.register_handler(request.get("handler"))
        )

        if bus_result.is_failure:
            return FlextResult[dict[str, object]].fail(
                f"Bus registration failed: {bus_result.error}"
            )

        # Create registration details
        details = {
            "registration_id": request.get("registration_id"),
            "message_type_name": getattr(request.get("message_type"), "__name__", None)
            if request.get("message_type")
            else None,
            "handler_mode": request.get("handler_mode"),
            "timestamp": FlextUtilities.Generators.generate_timestamp(),
            "status": FlextConstants.Dispatcher.REGISTRATION_STATUS_ACTIVE,
        }

        if self._config.get("enable_logging"):
            self._logger.info(
                "handler_registered",
                registration_id=details.get("registration_id"),
                handler_mode=details.get("handler_mode"),
                message_type=details.get("message_type_name"),
            )

        return FlextResult[dict[str, object]].ok(details)

    def register_handler(
        self,
        handler: FlextHandlers[object, object],
        *,
        handler_mode: str = FlextConstants.Dispatcher.DEFAULT_HANDLER_MODE,
        handler_config: dict[str, object] | None = None,
    ) -> FlextResult[dict[str, object]]:
        """Register handler with minimal parameters using structured model internally.

        Args:
            handler: Handler instance to register
            handler_mode: Handler operation mode (command/query)
            handler_config: Optional handler configuration

        Returns:
            FlextResult with registration details or error

        """
        # Create structured request
        request = dict[str, object](
            handler=handler,
            message_type=None,
            handler_mode=handler_mode,
            handler_config=handler_config,
        )

        return self.register_handler_with_request(request)

    def register_command(
        self,
        command_type: type[object],
        handler: FlextHandlers[object, object],
        *,
        handler_config: dict[str, object] | None = None,
    ) -> FlextResult[dict[str, object]]:
        """Register command handler using structured model internally.

        Args:
            command_type: Command message type
            handler: Handler instance
            handler_config: Optional handler configuration

        Returns:
            FlextResult with registration details or error

        """
        request = dict[str, object](
            handler=handler,
            message_type=command_type,
            handler_mode=FlextConstants.Dispatcher.HANDLER_MODE_COMMAND,
            handler_config=handler_config,
        )

        return self.register_handler_with_request(request)

    def register_query(
        self,
        query_type: type[object],
        handler: FlextHandlers[object, object],
        *,
        handler_config: dict[str, object] | None = None,
    ) -> FlextResult[dict[str, object]]:
        """Register query handler using structured model internally.

        Args:
            query_type: Query message type
            handler: Handler instance
            handler_config: Optional handler configuration

        Returns:
            FlextResult with registration details or error

        """
        request = dict[str, object](
            handler=handler,
            message_type=query_type,
            handler_mode=FlextConstants.Dispatcher.HANDLER_MODE_QUERY,
            handler_config=handler_config,
        )

        return self.register_handler_with_request(request)

    def register_function(
        self,
        message_type: type[object],
        handler_func: Callable[[object], object | FlextResult[object]],
        *,
        handler_config: dict[str, object] | None = None,
        mode: Literal["command", "query"] = "command",
    ) -> FlextResult[dict[str, object]]:
        """Register function as handler using factory pattern.

        Args:
            message_type: Message type to handle
            handler_func: Function to wrap as handler
            handler_config: Optional handler configuration
            mode: Handler mode (command/query)

        Returns:
            FlextResult with registration details or error

        """
        # Validate mode
        if mode not in FlextConstants.Dispatcher.VALID_HANDLER_MODES:
            return FlextResult[dict[str, object]].fail(
                FlextConstants.Dispatcher.ERROR_INVALID_HANDLER_MODE
            )

        # Create handler from function
        handler_result = self._create_handler_from_function(
            handler_func, handler_config, mode
        )

        if handler_result.is_failure:
            return FlextResult[dict[str, object]].fail(
                f"Handler creation failed: {handler_result.error}"
            )

        # Register the created handler
        request = dict[str, object](
            handler=handler_result.value,
            message_type=message_type,
            handler_mode=mode,
            handler_config=handler_config,
        )

        return self.register_handler_with_request(request)

    def _create_handler_from_function(
        self,
        handler_func: Callable[[object], object | FlextResult[object]],
        handler_config: dict[str, object] | None,
        mode: Literal["command", "query"],
    ) -> FlextResult[FlextHandlers[object, object]]:
        """Create handler from function using FlextHandlers constructor.

        Args:
            handler_func: Function to wrap
            handler_config: Optional configuration
            mode: Handler mode

        Returns:
            FlextResult with handler instance or error

        """
        try:
            # Create a simple handler class that wraps the function
            class FunctionHandler(FlextHandlers[object, object]):
                def __init__(self) -> None:
                    super().__init__(
                        handler_config=handler_config,
                        handler_mode=mode,
                    )
                    self._handler_func = handler_func

                def execute(self, message: object) -> FlextResult[object]:
                    """Route execution directly through the wrapped callable."""

                    return self.handle(message)

                def handle(self, message: object) -> FlextResult[object]:
                    """Handle message using wrapped function.

                    Args:
                        message: Message to process

                    Returns:
                        FlextResult with processing result or error

                    """
                    try:
                        result = self._handler_func(message)
                        if isinstance(result, FlextResult):
                            return cast("FlextResult[object]", result)
                        return FlextResult[object].ok(result)
                    except Exception as error:
                        return FlextResult[object].fail(
                            f"Function handler failed: {error}"
                        )

            handler = FunctionHandler()
            return FlextResult[FlextHandlers[object, object]].ok(handler)

        except Exception as error:
            return FlextResult[FlextHandlers[object, object]].fail(
                f"Handler creation failed: {error}"
            )

    # ------------------------------------------------------------------
    # Dispatch execution using structured models
    # ------------------------------------------------------------------
    def dispatch_with_request(
        self,
        request: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Dispatch using structured request model.

        Args:
            request: Pydantic model containing dispatch details

        Returns:
            FlextResult with structured dispatch result

        """
        start_time = time.time()

        # Validate request
        if request.get("message") is None:
            return FlextResult[dict[str, object]].fail(
                FlextConstants.Dispatcher.ERROR_MESSAGE_REQUIRED
            )

        # Execute dispatch with context management
        context_metadata = request.get("context_metadata")
        metadata_dict = self._normalize_context_metadata(context_metadata)
        correlation_id = request.get("correlation_id")
        correlation_id_str = str(correlation_id) if correlation_id is not None else None
        with self._context_scope(metadata_dict, correlation_id_str):
            result = self._bus.execute(request.get("message"))

            execution_time_ms = int((time.time() - start_time) * 1000)

            if result.is_success:
                dispatch_result = dict[str, object](
                    success=True,
                    result=result.value,
                    error_message=None,
                    request_id=request.get("request_id"),
                    execution_time_ms=execution_time_ms,
                    correlation_id=request.get("correlation_id"),
                )

                if self._config.get("enable_logging"):
                    self._logger.debug(
                        "dispatch_succeeded",
                        request_id=request.get("request_id"),
                        message_type=type(request.get("message")).__name__,
                        execution_time_ms=execution_time_ms,
                    )

                return FlextResult[dict[str, object]].ok(dispatch_result)
            dispatch_result = dict[str, object](
                success=False,
                result=None,
                error_message=result.error or "Unknown error",
                request_id=request.get("request_id"),
                execution_time_ms=execution_time_ms,
                correlation_id=request.get("correlation_id"),
            )

            if self._config.get("enable_logging"):
                self._logger.error(
                    "dispatch_failed",
                    request_id=request.get("request_id"),
                    message_type=type(request.get("message")).__name__,
                    error=dispatch_result.get("error_message"),
                    execution_time_ms=execution_time_ms,
                )

            return FlextResult[dict[str, object]].ok(dispatch_result)

    def dispatch(
        self,
        message: object,
        *,
        metadata: FlextTypes.Core.Dict | None = None,
        correlation_id: str | None = None,
        timeout_override: int | None = None,
    ) -> FlextResult[object]:
        """Dispatch message with optional metadata using structured model internally.

        Args:
            message: Message to dispatch
            metadata: Optional execution context metadata
            correlation_id: Optional correlation ID for tracing
            timeout_override: Optional timeout override

        Returns:
            FlextResult with execution result or error

        """
        # Create structured request
        metadata_obj = None
        if metadata:
            # Convert dict[str, object] to dict[str, object] for Metadata model
            string_metadata: dict[str, object] = {
                k: str(v) for k, v in metadata.items()
            }
            metadata_obj = FlextModels.Metadata(attributes=string_metadata)
        request = dict[str, object](
            message=message,
            context_metadata=metadata_obj,
            correlation_id=correlation_id,
            timeout_override=timeout_override,
        )

        # Execute structured dispatch
        structured_result = self.dispatch_with_request(request)

        if structured_result.is_failure:
            return FlextResult[object].fail(
                structured_result.error or "Dispatch failed"
            )

        dispatch_result = structured_result.value

        if dispatch_result.get("success"):
            return FlextResult[object].ok(dispatch_result.get("result"))
        return FlextResult[object].fail(
            str(dispatch_result.get("error_message")) or "Unknown error"
        )

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_context_metadata(
        metadata: object | None,
    ) -> FlextTypes.Core.Dict | None:
        """Convert metadata payloads into context-friendly dictionaries.

        Args:
            metadata: Metadata payload supplied via dispatch request.

        Returns:
            Normalized dictionary suitable for context propagation or ``None``.
        """

        if metadata is None:
            return None

        normalized: FlextTypes.Core.Dict | None = None

        if isinstance(metadata, dict):
            normalized = dict(metadata)
        elif isinstance(metadata, FlextModels.Metadata):
            normalized = {
                str(key): value for key, value in metadata.attributes.items()
            }
            metadata_dump = metadata.model_dump()
            if isinstance(metadata_dump, dict):
                for key, value in metadata_dump.items():
                    if key == "attributes" or value is None:
                        continue
                    normalized.setdefault(str(key), value)
        else:
            attributes = getattr(metadata, "attributes", None)
            if isinstance(attributes, dict):
                normalized = {
                    str(key): value for key, value in attributes.items()
                }
            else:
                model_dump_fn = getattr(metadata, "model_dump", None)
                if callable(model_dump_fn):
                    try:
                        dumped = model_dump_fn()
                    except TypeError:
                        dumped = None
                    if isinstance(dumped, dict):
                        normalized = {
                            str(key): value
                            for key, value in dumped.items()
                            if value is not None
                        }

        if normalized is None and hasattr(metadata, "items"):
            try:
                normalized = {
                    str(key): value for key, value in dict(metadata).items()
                }
            except Exception:  # pragma: no cover - defensive fallback
                normalized = None

        if normalized is None:
            return None

        return {str(key): value for key, value in normalized.items()}

    @contextmanager
    def _context_scope(
        self,
        metadata: FlextTypes.Core.Dict | None = None,
        correlation_id: str | None = None,
    ) -> Generator[None]:
        """Manage execution context with optional metadata and correlation ID.

        Args:
            metadata: Optional metadata to include in context
            correlation_id: Optional correlation ID for tracing

        """
        if not self._config.get("auto_context"):
            yield
            return

        metadata_token: Token[FlextTypes.Core.Dict | None] | None = None
        metadata_var = FlextContext.Variables.Performance.OPERATION_METADATA

        with FlextContext.Correlation.inherit_correlation() as active_correlation_id:
            # Use provided correlation ID or the inherited one
            effective_correlation_id = correlation_id or active_correlation_id

            if metadata:
                metadata_token = metadata_var.set(metadata)

            if self._config.get("enable_logging"):
                self._logger.debug(
                    "dispatch_context_entered",
                    correlation_id=effective_correlation_id,
                )
            try:
                yield
            finally:
                if metadata_token is not None:
                    metadata_var.reset(metadata_token)

                if self._config.get("enable_logging"):
                    self._logger.debug(
                        "dispatch_context_exited",
                        correlation_id=effective_correlation_id,
                    )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @classmethod
    def create_with_config(
        cls,
        config: dict[str, object] | None = None,
    ) -> FlextResult[FlextDispatcher]:
        """Create dispatcher with explicit configuration model.

        Args:
            config: Dispatcher configuration model

        Returns:
            FlextResult with dispatcher instance or error

        """
        try:
            instance = cls(config=config)
            return FlextResult[FlextDispatcher].ok(instance)
        except Exception as error:
            return FlextResult[FlextDispatcher].fail(
                f"Dispatcher creation failed: {error}"
            )

    @classmethod
    def create_from_global_config(cls) -> FlextResult[FlextDispatcher]:
        """Create dispatcher using global FlextConfig instance.

        Returns:
            FlextResult with dispatcher instance or error

        """
        try:
            instance = cls()
            return FlextResult[FlextDispatcher].ok(instance)
        except Exception as error:
            return FlextResult[FlextDispatcher].fail(
                f"Dispatcher creation failed: {error}"
            )


__all__ = ["FlextDispatcher"]
