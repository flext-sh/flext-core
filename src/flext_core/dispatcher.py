"""Dispatcher facade delivering the Phase 1 unified dispatcher charter.

The façade wraps ``FlextBus`` so handler registration, context propagation, and
metadata-aware dispatch all match the expectations documented in ``README.md``
and ``docs/architecture.md`` for the 1.0.0 modernization programme.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator, Mapping
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
        super().__init__()
        # Use provided config or create from global configuration
        if config is None:
            global_config = FlextConfig.get_global_instance()
            bus_config = dict(global_config.get_cqrs_bus_config())
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
                "bus_config": bus_config,
                "execution_timeout": bus_config.get("execution_timeout"),
            }
        else:
            config = dict(config)
            bus_config_raw = config.get("bus_config")

            if not isinstance(bus_config_raw, dict):
                global_config = FlextConfig.get_global_instance()
                default_bus_config = dict(global_config.get_cqrs_bus_config())

                if "execution_timeout" in config:
                    default_bus_config["execution_timeout"] = config[
                        "execution_timeout"
                    ]
                elif "timeout_seconds" in config:
                    default_bus_config["execution_timeout"] = config["timeout_seconds"]

                bus_config_raw = default_bus_config
                config["bus_config"] = bus_config_raw

            # Type-narrow bus_config_raw to dict[str, object] - Python 3.13+ type narrowing
            # At this point, bus_config_raw is guaranteed to be a dict due to the isinstance check above
            typed_bus_config: dict[str, object] = cast(
                "dict[str, object]", bus_config_raw
            )

            # typed_bus_config is already typed as dict[str, object]
            config.setdefault(
                "execution_timeout", typed_bus_config.get("execution_timeout")
            )

        self._config = config
        bus_config_raw = config.get("bus_config")

        # Handle both dict and FlextModels.CqrsConfig.Bus for bus_config
        final_bus_config_dict: dict[str, object] | None
        if isinstance(bus_config_raw, FlextModels.CqrsConfig.Bus):
            # Convert typed model to dict for FlextBus.create_command_bus
            final_bus_config_dict = bus_config_raw.model_dump()
        elif isinstance(bus_config_raw, dict):
            final_bus_config_dict = cast("dict[str, object]", bus_config_raw)
        else:
            final_bus_config_dict = None

        self._bus = bus or FlextBus.create_command_bus(bus_config=final_bus_config_dict)
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
            handler = FlextHandlers.from_callable(
                handler_func,
                mode=mode,
                handler_config=handler_config,
            )
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

        # Get timeout from request override or config
        timeout_override = request.get("timeout_override")
        config_timeout = self._config.get("timeout_seconds")
        timeout_seconds = (
            timeout_override if timeout_override is not None else config_timeout
        )

        # Execute dispatch with context management and timeout enforcement
        context_metadata = request.get("context_metadata")
        metadata_dict = self._normalize_context_metadata(context_metadata)
        correlation_id = request.get("correlation_id")
        correlation_id_str = str(correlation_id) if correlation_id is not None else None

        with self._context_scope(metadata_dict, correlation_id_str):
            # Execute with timeout if configured
            if (
                timeout_seconds
                and isinstance(timeout_seconds, (int, float))
                and timeout_seconds > 0
            ):
                # Use FlextUtilities.Reliability.with_timeout for timeout enforcement
                result = FlextUtilities.Reliability.with_timeout(
                    lambda: self._bus.execute(request.get("message")),
                    float(timeout_seconds),
                )
            else:
                # No timeout configured, execute directly
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
                    timeout_seconds=timeout_seconds,
                )

                if self._config.get("enable_logging"):
                    self._logger.debug(
                        "dispatch_succeeded",
                        request_id=request.get("request_id"),
                        message_type=type(request.get("message")).__name__,
                        execution_time_ms=execution_time_ms,
                        timeout_seconds=timeout_seconds,
                    )

                return FlextResult[dict[str, object]].ok(dispatch_result)

            dispatch_result = dict[str, object](
                success=False,
                result=None,
                error_message=result.error or "Unknown error",
                request_id=request.get("request_id"),
                execution_time_ms=execution_time_ms,
                correlation_id=request.get("correlation_id"),
                timeout_seconds=timeout_seconds,
            )

            if self._config.get("enable_logging"):
                self._logger.error(
                    "dispatch_failed",
                    request_id=request.get("request_id"),
                    message_type=type(request.get("message")).__name__,
                    error=dispatch_result.get("error_message"),
                    execution_time_ms=execution_time_ms,
                    timeout_seconds=timeout_seconds,
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
    def _normalize_context_metadata(
        self, metadata: object | None
    ) -> FlextTypes.Core.Dict | None:
        """Normalize metadata payloads to plain dictionaries."""
        if metadata is None:
            return None

        raw_metadata: Mapping[str, object] | None = None

        if isinstance(metadata, FlextModels.Metadata):
            attributes = metadata.attributes
            # Python 3.13+ type narrowing: attributes is already Mapping[str, object]
            if attributes and len(attributes) > 0:
                raw_metadata = attributes
            else:
                try:
                    dumped = metadata.model_dump()
                except Exception:
                    dumped = None
                if isinstance(dumped, Mapping):
                    attributes_section = dumped.get("attributes")
                    if isinstance(attributes_section, Mapping) and attributes_section:
                        raw_metadata = cast("Mapping[str, object]", attributes_section)
                    else:
                        raw_metadata = cast("Mapping[str, object]", dumped)
        elif isinstance(metadata, Mapping):
            # Python 3.13+ type narrowing: metadata is already Mapping[str, object]
            raw_metadata = cast("Mapping[str, object]", metadata)
        else:
            attributes_value = getattr(metadata, "attributes", None)
            if isinstance(attributes_value, Mapping) and attributes_value:
                # Python 3.13+ type narrowing: attributes_value is already Mapping[str, object]
                raw_metadata = cast("Mapping[str, object]", attributes_value)
            else:
                model_dump = getattr(metadata, "model_dump", None)
                if callable(model_dump):
                    try:
                        dumped = model_dump()
                    except Exception:
                        dumped = None
                    if isinstance(dumped, Mapping):
                        raw_metadata = cast("Mapping[str, object]", dumped)

        if raw_metadata is None:
            return None

        normalized: FlextTypes.Core.Dict = {
            str(key): value for key, value in raw_metadata.items()
        }

        return dict(normalized)

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
        correlation_token: Token[str | None] | None = None
        parent_token: Token[str | None] | None = None
        metadata_var = FlextContext.Variables.Performance.OPERATION_METADATA
        correlation_var = FlextContext.Variables.Correlation.CORRELATION_ID
        parent_var = FlextContext.Variables.Correlation.PARENT_CORRELATION_ID

        # Store current context values for restoration
        current_correlation = correlation_var.get()
        _current_parent = parent_var.get()

        # Set new correlation ID if provided
        if correlation_id is not None:
            correlation_token = correlation_var.set(correlation_id)
            # Set parent correlation ID if there was a previous one
            if (
                current_correlation is not None
                and current_correlation != correlation_id
            ):
                parent_token = parent_var.set(current_correlation)

        try:
            # Set metadata if provided
            if metadata:
                metadata_token = metadata_var.set(metadata)

            # Use provided correlation ID or generate one if needed
            effective_correlation_id = correlation_id
            if effective_correlation_id is None:
                effective_correlation_id = (
                    FlextContext.Correlation.generate_correlation_id()
                )

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
        finally:
            # Restore context in reverse order
            if parent_token is not None:
                parent_var.reset(parent_token)
            if correlation_token is not None:
                correlation_var.reset(correlation_token)

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
