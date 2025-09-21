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

    This implementation uses FlextModels for structured validation, FlextConfig
    for centralized configuration, and FlextConstants for default values.
    """

    def __init__(
        self,
        *,
        config: FlextModels.DispatcherConfiguration | None = None,
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
            config = FlextModels.DispatcherConfiguration(
                auto_context=global_config.dispatcher_auto_context,
                timeout_seconds=global_config.dispatcher_timeout_seconds,
                enable_metrics=global_config.dispatcher_enable_metrics,
                enable_logging=global_config.dispatcher_enable_logging,
                bus_config=None,
            )

        self._config = config
        self._bus = bus or FlextBus.create_command_bus(bus_config=config.bus_config)
        self._logger = FlextLogger(self.__class__.__name__)

    @property
    def config(self) -> FlextModels.DispatcherConfiguration:
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
        request: FlextModels.HandlerRegistrationRequest,
    ) -> FlextResult[FlextModels.RegistrationDetails]:
        """Register handler using structured request model.

        Args:
            request: Pydantic model containing registration details

        Returns:
            FlextResult with registration details or error

        """
        # Validate handler mode using constants
        if request.handler_mode not in FlextConstants.Dispatcher.VALID_HANDLER_MODES:
            return FlextResult[FlextModels.RegistrationDetails].fail(
                FlextConstants.Dispatcher.ERROR_INVALID_HANDLER_MODE
            )

        # Validate handler is provided
        if request.handler is None:
            return FlextResult[FlextModels.RegistrationDetails].fail(
                FlextConstants.Dispatcher.ERROR_HANDLER_REQUIRED
            )

        # Register with bus
        bus_result = (
            self._bus.register_handler(request.message_type, request.handler)
            if request.message_type
            else self._bus.register_handler(request.handler)
        )

        if bus_result.is_failure:
            return FlextResult[FlextModels.RegistrationDetails].fail(
                f"Bus registration failed: {bus_result.error}"
            )

        # Create registration details
        details = FlextModels.RegistrationDetails(
            registration_id=request.registration_id,
            message_type_name=request.message_type.__name__
            if request.message_type
            else None,
            handler_mode=request.handler_mode,
            timestamp=FlextUtilities.Generators.generate_timestamp(),
            status="active",  # Use literal string for type safety
        )

        if self._config.enable_logging:
            self._logger.info(
                "handler_registered",
                registration_id=details.registration_id,
                handler_mode=details.handler_mode,
                message_type=details.message_type_name,
            )

        return FlextResult[FlextModels.RegistrationDetails].ok(details)

    def register_handler(
        self,
        handler: FlextHandlers[object, object],
        *,
        handler_mode: str = FlextConstants.Dispatcher.DEFAULT_HANDLER_MODE,
        handler_config: FlextModels.CqrsConfig.Handler | None = None,
    ) -> FlextResult[FlextModels.RegistrationDetails]:
        """Register handler with minimal parameters using structured model internally.

        Args:
            handler: Handler instance to register
            handler_mode: Handler operation mode (command/query)
            handler_config: Optional handler configuration

        Returns:
            FlextResult with registration details or error

        """
        # Create structured request
        request = FlextModels.HandlerRegistrationRequest(
            handler=handler,
            message_type=None,
            handler_mode=handler_mode,  # type: ignore[arg-type]
            handler_config=handler_config,
        )

        return self.register_handler_with_request(request)

    def register_command(
        self,
        command_type: type[object],
        handler: FlextHandlers[object, object],
        *,
        handler_config: FlextModels.CqrsConfig.Handler | None = None,
    ) -> FlextResult[FlextModels.RegistrationDetails]:
        """Register command handler using structured model internally.

        Args:
            command_type: Command message type
            handler: Handler instance
            handler_config: Optional handler configuration

        Returns:
            FlextResult with registration details or error

        """
        request = FlextModels.HandlerRegistrationRequest(
            handler=handler,
            message_type=command_type,
            handler_mode=FlextConstants.Dispatcher.HANDLER_MODE_COMMAND,  # type: ignore[arg-type]
            handler_config=handler_config,
        )

        return self.register_handler_with_request(request)

    def register_query(
        self,
        query_type: type[object],
        handler: FlextHandlers[object, object],
        *,
        handler_config: FlextModels.CqrsConfig.Handler | None = None,
    ) -> FlextResult[FlextModels.RegistrationDetails]:
        """Register query handler using structured model internally.

        Args:
            query_type: Query message type
            handler: Handler instance
            handler_config: Optional handler configuration

        Returns:
            FlextResult with registration details or error

        """
        request = FlextModels.HandlerRegistrationRequest(
            handler=handler,
            message_type=query_type,
            handler_mode=FlextConstants.Dispatcher.HANDLER_MODE_QUERY,  # type: ignore[arg-type]
            handler_config=handler_config,
        )

        return self.register_handler_with_request(request)

    def register_function(
        self,
        message_type: type[object],
        handler_func: Callable[[object], object | FlextResult[object]],
        *,
        handler_config: FlextModels.CqrsConfig.Handler | None = None,
        mode: str = FlextConstants.Dispatcher.DEFAULT_HANDLER_MODE,
    ) -> FlextResult[FlextModels.RegistrationDetails]:
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
            return FlextResult[FlextModels.RegistrationDetails].fail(
                FlextConstants.Dispatcher.ERROR_INVALID_HANDLER_MODE
            )

        # Create handler from function
        handler_result = self._create_handler_from_function(
            handler_func, handler_config, mode
        )

        if handler_result.is_failure:
            return FlextResult[FlextModels.RegistrationDetails].fail(
                f"Handler creation failed: {handler_result.error}"
            )

        # Register the created handler
        request = FlextModels.HandlerRegistrationRequest(
            handler=handler_result.value,
            message_type=message_type,
            handler_mode=mode,  # type: ignore[arg-type]
            handler_config=handler_config,
        )

        return self.register_handler_with_request(request)

    def _create_handler_from_function(
        self,
        handler_func: Callable[[object], object | FlextResult[object]],
        handler_config: FlextModels.CqrsConfig.Handler | None,
        mode: str,
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
                        config=handler_config,
                        handler_mode=mode,  # type: ignore[arg-type]
                    )
                    self._handler_func = handler_func

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
                            return result
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
        request: FlextModels.DispatchRequest,
    ) -> FlextResult[FlextModels.DispatchResult]:
        """Dispatch using structured request model.

        Args:
            request: Pydantic model containing dispatch details

        Returns:
            FlextResult with structured dispatch result

        """
        start_time = time.time()

        # Validate request
        if request.message is None:
            return FlextResult[FlextModels.DispatchResult].fail(
                FlextConstants.Dispatcher.ERROR_MESSAGE_REQUIRED
            )

        # Execute dispatch with context management
        metadata_dict: dict[str, object] | None = None
        if request.context_metadata:
            # Convert dict[str, str] to dict[str, object] for context scope
            metadata_dict = dict(request.context_metadata.value.items())
        with self._context_scope(metadata_dict, request.correlation_id):
            result = self._bus.execute(request.message)

            execution_time_ms = int((time.time() - start_time) * 1000)

            if result.is_success:
                dispatch_result = FlextModels.DispatchResult(
                    success=True,
                    result=result.value,
                    error_message=None,
                    request_id=request.request_id,
                    execution_time_ms=execution_time_ms,
                    correlation_id=request.correlation_id,
                )

                if self._config.enable_logging:
                    self._logger.debug(
                        "dispatch_succeeded",
                        request_id=request.request_id,
                        message_type=type(request.message).__name__,
                        execution_time_ms=execution_time_ms,
                    )

                return FlextResult[FlextModels.DispatchResult].ok(dispatch_result)
            dispatch_result = FlextModels.DispatchResult(
                success=False,
                result=None,
                error_message=result.error or "Unknown error",
                request_id=request.request_id,
                execution_time_ms=execution_time_ms,
                correlation_id=request.correlation_id,
            )

            if self._config.enable_logging:
                self._logger.error(
                    "dispatch_failed",
                    request_id=request.request_id,
                    message_type=type(request.message).__name__,
                    error=dispatch_result.error_message,
                    execution_time_ms=execution_time_ms,
                )

            return FlextResult[FlextModels.DispatchResult].ok(dispatch_result)

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
            # Convert dict[str, object] to dict[str, str] for Metadata model
            string_metadata = {k: str(v) for k, v in metadata.items()}
            metadata_obj = FlextModels.Metadata(value=string_metadata)
        request = FlextModels.DispatchRequest(
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

        if dispatch_result.success:
            return FlextResult[object].ok(dispatch_result.result)
        return FlextResult[object].fail(
            dispatch_result.error_message or "Unknown error"
        )

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------
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
        if not self._config.auto_context:
            yield
            return

        metadata_token: Token[FlextTypes.Core.Dict | None] | None = None
        metadata_var = FlextContext.Variables.Performance.OPERATION_METADATA

        with FlextContext.Correlation.inherit_correlation() as active_correlation_id:
            # Use provided correlation ID or the inherited one
            effective_correlation_id = correlation_id or active_correlation_id

            if metadata:
                metadata_token = metadata_var.set(metadata)

            if self._config.enable_logging:
                self._logger.debug(
                    "dispatch_context_entered",
                    correlation_id=effective_correlation_id,
                )
            try:
                yield
            finally:
                if metadata_token is not None:
                    metadata_var.reset(metadata_token)

                if self._config.enable_logging:
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
        config: FlextModels.DispatcherConfiguration,
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
