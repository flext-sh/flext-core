"""Dispatcher facade delivering the Phase 1 unified dispatcher charter.

The façade wraps ``FlextBus`` so handler registration, context propagation, and
metadata-aware dispatch all match the expectations documented in ``README.md``
and ``docs/architecture.md`` for the 1.0.0 modernization programme.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import Token
from dataclasses import dataclass
from queue import Queue
from typing import Any, Literal, cast, override

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

    @override
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
            bus_config: dict[str, object] = dict(global_config.get_cqrs_bus_config())
            # Map timeout_seconds to execution_timeout for bus compatibility
            bus_config["execution_timeout"] = bus_config.get(
                "timeout_seconds", FlextConstants.Defaults.TIMEOUT
            )
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
                "max_retries": getattr(global_config, "dispatcher_max_retries", 3),
                "retry_delay": getattr(global_config, "dispatcher_retry_delay", 0.1),
                "bus_config": bus_config,
                "execution_timeout": bus_config.get("timeout_seconds"),
            }
        else:
            config = dict(config)
            bus_config_raw = config.get("bus_config")

            if not isinstance(bus_config_raw, dict):
                global_config = FlextConfig.get_global_instance()
                default_bus_config: dict[str, object] = dict(
                    global_config.get_cqrs_bus_config()
                )

                if "execution_timeout" in config:
                    default_bus_config["execution_timeout"] = config[
                        "execution_timeout"
                    ]
                elif "timeout_seconds" in config:
                    default_bus_config["execution_timeout"] = config["timeout_seconds"]

                bus_config_raw = default_bus_config
                config["bus_config"] = bus_config_raw

            # At this point, bus_config_raw is guaranteed to be a dict due to the isinstance check above
            bus_config_dict = cast("dict[str, object]", bus_config_raw)
            execution_timeout_value: object | None = bus_config_dict.get(
                "execution_timeout"
            )
            config.setdefault("execution_timeout", execution_timeout_value)

        self._config: dict[str, object] = config

        # Circuit breaker state
        self._circuit_breaker_failures: dict[str, int] = {}
        circuit_breaker_threshold = config.get("circuit_breaker_threshold", 5)
        if isinstance(circuit_breaker_threshold, (int, str)):
            self._circuit_breaker_threshold = int(circuit_breaker_threshold)
        else:
            self._circuit_breaker_threshold = 5

        # Rate limiting state
        self._rate_limit_requests: dict[str, list[float]] = {}
        rate_limit = config.get("rate_limit", 100)
        if isinstance(rate_limit, (int, str)):
            self._rate_limit = int(rate_limit)
        else:
            self._rate_limit = 100

        rate_limit_window = config.get("rate_limit_window", 60)
        if isinstance(rate_limit_window, (int, str)):
            self._rate_limit_window = int(rate_limit_window)
        else:
            self._rate_limit_window = 60

        # Audit log
        self._audit_log: list[dict[str, object]] = []

        # Performance metrics
        self._performance_metrics: dict[str, dict[str, float]] = {}

        bus_config_raw = config.get("bus_config")

        # Handle both dict and FlextModels.CqrsConfig.Bus for bus_config
        bus_config_dict_final: dict[str, object] | None
        if isinstance(bus_config_raw, FlextModels.CqrsConfig.Bus):
            # Convert typed model to dict for FlextBus.create_command_bus
            bus_config_dict_final = bus_config_raw.model_dump()
        elif isinstance(bus_config_raw, dict):
            bus_config_dict_final = cast("dict[str, object]", bus_config_raw)
        else:
            bus_config_dict_final = None

        self._bus = bus or FlextBus.create_command_bus(bus_config=bus_config_dict_final)
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
        message_type_or_handler: str
        | FlextHandlers[object, object]
        | Callable[[object], object | FlextResult[object]],
        handler: FlextHandlers[object, object]
        | Callable[[object], object | FlextResult[object]]
        | None = None,
        *,
        handler_mode: Literal["command", "query"] = "command",
        handler_config: dict[str, object] | None = None,
    ) -> FlextResult[dict[str, object]]:
        """Register handler with support for both old and new API.

        Args:
            message_type_or_handler: Message type (str) or handler instance
            handler: Handler instance (when message_type is provided)
            handler_mode: Handler operation mode (command/query)
            handler_config: Optional handler configuration

        Returns:
            FlextResult with registration details or error

        """
        # Support both old API (message_type, handler) and new API (handler only)
        if isinstance(message_type_or_handler, str) and handler is not None:
            # Old API: register_handler(message_type, handler)
            # Convert callable to FlextHandlers if needed
            resolved_handler = handler
            if callable(handler) and not isinstance(handler, FlextHandlers):
                handler_result = self.create_handler_from_function(
                    handler_func=handler,
                    handler_config=handler_config,
                    mode=handler_mode,
                )
                if handler_result.is_failure:
                    return FlextResult[dict[str, object]].fail(
                        handler_result.error or "Handler creation failed"
                    )
                resolved_handler = handler_result.data

            # Create structured request with message type
            request = dict[str, object](
                handler=resolved_handler,
                message_type=message_type_or_handler,
                handler_mode=handler_mode,
                handler_config=handler_config,
            )
        else:
            # New API: register_handler(handler)
            # Convert callable to FlextHandlers if needed
            resolved_handler = message_type_or_handler
            if callable(message_type_or_handler) and not isinstance(
                message_type_or_handler, FlextHandlers
            ):
                handler_result = self.create_handler_from_function(
                    handler_func=message_type_or_handler,
                    handler_config=handler_config,
                    mode=handler_mode,
                )
                if handler_result.is_failure:
                    return FlextResult[dict[str, object]].fail(
                        handler_result.error or "Handler creation failed"
                    )
                resolved_handler = handler_result.data

            # Create structured request
            request = dict[str, object](
                handler=resolved_handler,
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
        handler_result = self.create_handler_from_function(
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

    def create_handler_from_function(
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
                callable_func=handler_func,
                handler_name=getattr(handler_func, "__name__", "FunctionHandler"),
                handler_type=mode,
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
        normalized_metadata = self._normalize_context_metadata(context_metadata)
        metadata_dict = normalized_metadata if normalized_metadata is not None else {}
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
                execution_result = FlextUtilities.Reliability.with_timeout(
                    lambda: self._bus.execute(request.get("message")),
                    float(timeout_seconds),
                )
            else:
                # No timeout configured, execute directly
                execution_result = self._bus.execute(request.get("message"))

            execution_time_ms = int((time.time() - start_time) * 1000)

            # Get message type for circuit breaker and audit
            message = request.get("message")
            message_type = type(message).__name__ if message else "unknown"

            # Update circuit breaker state
            if not execution_result.is_success:
                self._circuit_breaker_failures[message_type] = (
                    self._circuit_breaker_failures.get(message_type, 0) + 1
                )
            else:
                # Reset failures on success
                self._circuit_breaker_failures[message_type] = 0

            # Add to audit log
            audit_entry = {
                "timestamp": time.time(),
                "message_type": message_type,
                "success": execution_result.is_success,
                "execution_time_ms": execution_time_ms,
                "correlation_id": request.get("correlation_id"),
                "request_id": request.get("request_id"),
            }
            self._audit_log.append(audit_entry)

            # Update performance metrics
            if message_type not in self._performance_metrics:
                self._performance_metrics[message_type] = {
                    "total_executions": 0,
                    "total_execution_time": 0.0,
                    "avg_execution_time": 0.0,
                    "success_count": 0,
                    "failure_count": 0,
                }

            metrics = self._performance_metrics[message_type]
            metrics["total_executions"] += 1
            metrics["total_execution_time"] += execution_time_ms / 1000.0
            metrics["avg_execution_time"] = (
                metrics["total_execution_time"] / metrics["total_executions"]
            )

            if execution_result.is_success:
                metrics["success_count"] += 1
            else:
                metrics["failure_count"] += 1

            if execution_result.is_success:
                dispatch_result = dict[str, object](
                    success=True,
                    result=execution_result.value,
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
                error_message=execution_result.error or "Unknown error",
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
        message_or_type: object | str,
        data: object | None = None,
        *,
        metadata: FlextTypes.Core.Dict | None = None,
        correlation_id: str | None = None,
        timeout_override: int | None = None,
    ) -> FlextResult[object]:
        """Dispatch message with support for both old and new API.

        Args:
            message_or_type: Message object or message type string
            data: Data to dispatch (when message_or_type is string)
            metadata: Optional execution context metadata
            correlation_id: Optional correlation ID for tracing
            timeout_override: Optional timeout override

        Returns:
            FlextResult with execution result or error

        """
        # Support both old API (message_type, data) and new API (message)
        message_type = None
        if isinstance(message_or_type, str) and data is not None:
            # Old API: dispatch(message_type, data)
            # Validate data is not empty
            if not data or data is None:
                return FlextResult[object].fail("Message is required")

            message_type = message_or_type
        else:
            # New API: dispatch(message)
            message = message_or_type
            message_type = type(message).__name__ if message else "unknown"

        # Check circuit breaker
        if self.is_circuit_breaker_open(message_type):
            return FlextResult[object].fail("Circuit breaker is open")

        # Check rate limiting
        current_time = time.time()
        if message_type in self._rate_limit_requests:
            # Clean old requests outside the window
            self._rate_limit_requests[message_type] = [
                req_time
                for req_time in self._rate_limit_requests[message_type]
                if current_time - req_time < self._rate_limit_window
            ]

            # Check if we're over the limit
            if len(self._rate_limit_requests[message_type]) >= self._rate_limit:
                return FlextResult[object].fail("Rate limit exceeded")

        # Record this request
        if message_type not in self._rate_limit_requests:
            self._rate_limit_requests[message_type] = []
        self._rate_limit_requests[message_type].append(current_time)

        if isinstance(message_or_type, str) and data is not None:
            # Create a message object with the correct type name
            @dataclass
            class MessageWrapper:
                data: object
                message_type: str

                def __post_init__(self) -> None:
                    self.__class__.__name__ = self.message_type

                def __str__(self) -> str:
                    """Return string representation of the data for handler compatibility."""
                    return str(self.data)

            message = MessageWrapper(data, message_or_type)
            message_type = message_or_type
        else:
            # New API: dispatch(message)
            message = message_or_type
        # Create structured request
        metadata_obj = None
        if metadata:
            # Convert dict[str, object] to dict[str, object] for Metadata model
            string_metadata: dict[str, object] = {
                k: str(v) for k, v in metadata.items()
            }
            metadata_model: FlextModels.Metadata = FlextModels.Metadata(
                attributes=string_metadata
            )
            metadata_obj = metadata_model
        request = dict[str, object](
            message=message,
            message_type=message_type,
            context_metadata=metadata_obj,
            correlation_id=correlation_id,
            timeout_override=timeout_override,
        )

        # Execute structured dispatch with retry logic
        max_retries_raw = self._config.get("max_retries", 3)
        max_retries: int = (
            int(max_retries_raw) if isinstance(max_retries_raw, (int, float)) else 3
        )
        retry_delay_raw = self._config.get("retry_delay", 0.1)
        retry_delay: float = (
            float(retry_delay_raw) if isinstance(retry_delay_raw, (int, float)) else 0.1
        )

        for attempt in range(max_retries):
            # Use threading timeout for the dispatch call
            timeout_raw = self._config.get("timeout", 30.0)
            timeout_seconds: float = (
                float(timeout_raw) if isinstance(timeout_raw, (int, float)) else 30.0
            )
            result_queue: Queue[tuple[bool, FlextResult[dict[str, object]]]] = Queue()

            def dispatch_target(
                queue: Queue[tuple[bool, FlextResult[dict[str, object]]]],
            ) -> None:
                try:
                    result = self.dispatch_with_request(request)
                    queue.put((True, result))
                except Exception as e:
                    queue.put((False, FlextResult[dict[str, object]].fail(str(e))))

            thread = threading.Thread(target=dispatch_target, args=(result_queue,))
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)

            if thread.is_alive():
                # Timeout occurred
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return FlextResult[object].fail("Dispatch timeout")

            try:
                success, structured_result = result_queue.get_nowait()
                if not success:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    return FlextResult[object].fail(
                        structured_result.error or "Dispatch failed"
                    )
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return FlextResult[object].fail(f"Dispatch error: {e}")

            if structured_result.is_failure:
                if attempt < max_retries - 1:
                    # Retry on failure (except for the last attempt)
                    time.sleep(retry_delay)
                    continue
                return FlextResult[object].fail(
                    structured_result.error or "Dispatch failed"
                )

            dispatch_result = structured_result.value

            if dispatch_result and dispatch_result.get("success"):
                return FlextResult[object].ok(dispatch_result.get("result"))

            # Check if this is a temporary failure that should be retried
            error_msg = (
                str(dispatch_result.get("error_message"))
                if dispatch_result
                else "Unknown error"
            )
            if attempt < max_retries - 1 and "Temporary failure" in error_msg:
                time.sleep(retry_delay)
                continue

            return FlextResult[object].fail(error_msg)

        return FlextResult[object].fail("Max retries exceeded")

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
                metadata_var.set(metadata)

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

    # =============================================================================
    # Missing Methods for Test Compatibility
    # =============================================================================

    def cleanup(self) -> None:
        """Clean up dispatcher resources."""
        try:
            # Clear all handlers using the public API
            if hasattr(self, "_bus") and self._bus:
                # Use the public clear_handlers method
                if hasattr(self._bus, "clear_handlers"):
                    self._bus.clear_handlers()

                if hasattr(self._bus, "cleanup"):
                    self._bus.cleanup()

            # Clear internal state
            self._circuit_breaker_failures.clear()
            self._rate_limit_requests.clear()
            self._audit_log.clear()
            self._performance_metrics.clear()

        except Exception as e:
            self._logger.warning("Cleanup failed", error=str(e))

    def get_handlers(self, message_type: str) -> list[object]:
        """Get handlers for specific message type.

        Args:
            message_type: Type of message

        Returns:
            List of handlers for the message type

        """
        if hasattr(self, "_bus") and self._bus:
            registered_handlers = self._bus.get_registered_handlers()
            handlers = registered_handlers.get(message_type, [])
            if not isinstance(handlers, list):
                handlers = [handlers]

            # Extract original callables from CallableHandler objects for compatibility
            result: list[object] = []
            for handler in handlers:
                # Type-safe access to handler - cast to Any to handle unknown types
                handler_any: Any = handler
                if hasattr(handler_any, "original_callable"):
                    # Type-safe access to original_callable attribute
                    original_callable = getattr(handler_any, "original_callable", None)
                    if original_callable is not None:
                        result.append(original_callable)
                    else:
                        result.append(handler_any)
                else:
                    result.append(handler_any)
            return result
        return []

    def clear_handlers(self) -> None:
        """Clear all registered handlers."""
        try:
            if (
                hasattr(self, "_bus")
                and self._bus
                and hasattr(self._bus, "clear_handlers")
            ):
                self._bus.clear_handlers()
        except Exception as e:
            self._logger.warning("Clear handlers failed", error=str(e))

    def get_statistics(self) -> dict[str, object]:
        """Get dispatcher statistics.

        Returns:
            Dictionary of statistics

        """
        stats: dict[str, object] = {
            "dispatcher_initialized": True,
            "bus_available": hasattr(self, "_bus") and bool(self._bus),
            "config_loaded": hasattr(self, "_config") and bool(self._config),
        }

        # Add bus statistics if available
        if hasattr(self, "_bus") and self._bus and hasattr(self._bus, "get_statistics"):
            try:
                bus_stats = self._bus.get_statistics()
                stats["bus_statistics"] = bus_stats
            except Exception:
                stats["bus_statistics"] = "unavailable"

        return stats

    def validate(self) -> FlextResult[None]:
        """Validate dispatcher configuration and state.

        Returns:
            FlextResult with validation result

        """
        try:
            # Validate configuration
            if not hasattr(self, "_config") or not self._config:
                return FlextResult[None].fail("Dispatcher not properly configured")

            # Validate bus
            if not hasattr(self, "_bus") or not self._bus:
                return FlextResult[None].fail("Dispatcher bus not available")

            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Dispatcher validation failed: {e}")

    def export_config(self) -> dict[str, object]:
        """Export dispatcher configuration.

        Returns:
            Dictionary of configuration

        """
        config = {}

        if hasattr(self, "_config"):
            config.update(self._config)

        if hasattr(self, "_bus") and self._bus and hasattr(self._bus, "export_config"):
            try:
                bus_config = self._bus.export_config()
                config["bus_config"] = bus_config
            except Exception:
                config["bus_config"] = "unavailable"

        # Include handler information
        if hasattr(self, "_bus") and self._bus:
            handlers: dict[str, Any] = {}
            try:
                registered_handlers = self._bus.get_registered_handlers()
                # registered_handlers is already a dict from get_registered_handlers()
                # Cast to handle the unknown return type
                registered_handlers_any: dict[str, Any] = cast(
                    "dict[str, Any]", registered_handlers
                )
                handlers.update(registered_handlers_any)
                config["handlers"] = handlers
            except Exception:
                config["handlers"] = {}

        return config

    def get_metrics(self) -> dict[str, object]:
        """Get dispatcher metrics."""
        # Convert performance metrics to the expected format
        result: dict[str, object] = {}
        for message_type, metrics in self._performance_metrics.items():
            result[message_type] = {
                "dispatches": metrics["total_executions"],
                "successes": metrics["success_count"],
                "failures": metrics["failure_count"],
                "avg_execution_time": metrics["avg_execution_time"],
            }
        return result

    def import_config(self, config: dict[str, object]) -> FlextResult[None]:
        """Import dispatcher configuration."""
        try:
            # Import handlers
            if "handlers" in config:
                handlers = config["handlers"]
                if isinstance(handlers, dict):
                    # Cast handlers to proper type to handle unknown dict types
                    handlers_dict: dict[str, Any] = cast("dict[str, Any]", handlers)
                    for message_type, handler_list in handlers_dict.items():
                        # Ensure message_type is a string and handler_list is properly typed
                        message_type_str = str(message_type)
                        if not isinstance(handler_list, list):
                            handlers_for_type: list[Any] = [handler_list]
                        else:
                            handlers_for_type = cast("list[Any]", handler_list)
                        for handler in handlers_for_type:
                            self.register_handler(message_type_str, handler)

            # Import other state
            if "circuit_breaker_failures" in config:
                failures = config["circuit_breaker_failures"]
                if isinstance(failures, dict):
                    self._circuit_breaker_failures.update(failures)
            if "rate_limit_requests" in config:
                requests = config["rate_limit_requests"]
                if isinstance(requests, dict):
                    self._rate_limit_requests.update(requests)
            if "performance_metrics" in config:
                metrics = config["performance_metrics"]
                if isinstance(metrics, dict):
                    self._performance_metrics.update(metrics)

            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Config import failed: {e}")

    def dispatch_batch(
        self, message_type: str, messages: list[object]
    ) -> list[FlextResult[object]]:
        """Dispatch multiple messages in batch."""
        results: list[FlextResult[object]] = []
        for message in messages:
            result = self.dispatch(message_type, message)
            results.append(result)
        return results

    def is_circuit_breaker_open(self, message_type: str) -> bool:
        """Check if circuit breaker is open for message type."""
        failures = self._circuit_breaker_failures.get(message_type, 0)
        return failures >= self._circuit_breaker_threshold

    def get_audit_log(self) -> list[dict[str, object]]:
        """Get audit log entries."""
        return self._audit_log.copy()

    def get_performance_metrics(self) -> dict[str, object]:
        """Get performance metrics."""
        return {k: v.copy() for k, v in self._performance_metrics.items()}


__all__ = ["FlextDispatcher"]
