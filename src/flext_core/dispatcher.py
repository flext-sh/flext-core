"""Dispatcher facade delivering the Phase 1 unified dispatcher charter.

The façade wraps ``FlextBus`` so handler registration, context propagation, and
metadata-aware dispatch all match the expectations documented in ``README.md``
and ``docs/architecture.md`` for the 1.0.0 modernization programme.

Refactored to eliminate SOLID violations by delegating to specialized components:
- Circuit breaker, rate limiting, audit, metrics → FlextProcessors
- Timeout, retry, batch processing → FlextService patterns
- Handler registration → FlextRegistry patterns
- Context management → FlextContext
- Logging → FlextLogger

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""
# ruff: disable=E402
# pyright: basic

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from typing import cast, override

from flext_core.bus import FlextBus
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.context import FlextContext
from flext_core.handlers import FlextHandlers
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class FlextDispatcher(FlextMixins.Service):
    """Orchestrates CQRS execution while enforcing context observability.

    The dispatcher is the front door promoted across the ecosystem: all
    handler registration flows, context scoping, and dispatch telemetry
    align with the modernization plan so downstream packages can adopt
    a consistent runtime contract without bespoke buses.

    **Inherited Infrastructure** (from FlextMixins.Service):
        - container: FlextContainer (via FlextMixins.Container)
        - context: object (via FlextMixins.Context)
        - logger: FlextLogger (via FlextMixins.Logging) - per-dispatcher logger instance
        - config: object (via FlextMixins.Configurable) - global config access
        - _track_operation: context manager (via FlextMixins.Metrics)
        - _enrich_context, _with_correlation_id, etc. (via FlextMixins.Service)

    **Function**: High-level message dispatch orchestration
        - Handler registration for command and query patterns
        - Message dispatch with context propagation and tracing
        - Circuit breaker pattern for fault tolerance
        - Rate limiting for request throttling
        - Retry logic with exponential backoff
        - Timeout enforcement for operation boundaries
        - Audit logging for compliance and debugging
        - Performance metrics collection and reporting
        - Batch processing for multiple messages
        - Configuration import/export for persistence

    **Uses**: Core FLEXT infrastructure for dispatch
        - FlextBus for low-level command/query execution
        - FlextProcessors for circuit breaker and rate limiting
        - FlextContext for execution context management
        - FlextLogger for structured logging
        - FlextConfig for global configuration
        - FlextHandlers for handler base class patterns
        - FlextResult[T] for all operation results
        - FlextUtilities for ID generation and validation
        - FlextConstants for error codes and defaults
        - FlextModels for domain models and metadata
        - concurrent.futures for timeout enforcement
        - contextvars for context propagation

    **How to use**: Message dispatch and handler registration
        ```python
        from flext_core import FlextDispatcher, FlextResult

        # Example 1: Create dispatcher with default config
        dispatcher_result = FlextDispatcher.create_from_global_config()
        if dispatcher_result.is_success:
            dispatcher = dispatcher_result.unwrap()


        # Example 2: Register handler with message type
        class CreateUserCommand:
            email: str


        def create_user_handler(cmd: CreateUserCommand) -> str:
            return f"User created: {cmd.email}"


        reg_result = dispatcher.register_handler(
            CreateUserCommand, create_user_handler, handler_mode="command"
        )

        # Example 3: Dispatch message with context metadata
        command = CreateUserCommand(email="user@example.com")
        result = dispatcher.dispatch(
            command,
            metadata={"user_id": "123", "request_id": "abc"},
            correlation_id="req-123",
        )

        # Example 4: Dispatch with timeout override
        result = dispatcher.dispatch(
            command,
            timeout_override=FlextConstants.Dispatcher.DEFAULT_TIMEOUT_SECONDS,  # Use standard timeout
        )

        # Example 5: Create dispatcher from global configuration
        dispatcher_result = FlextDispatcher.create_from_global_config()
        if dispatcher_result.is_success:
            dispatcher = dispatcher_result.unwrap()
            dispatcher.dispatch(command)
        ```

        - [ ] Add distributed tracing integration (OpenTelemetry)
        - [ ] Implement priority queue for message ordering
        - [ ] Support dispatch patterns with io
        - [ ] Add circuit breaker auto-recovery with backoff
        - [ ] Implement adaptive rate limiting based on metrics
        - [ ] Support message routing and transformation
        - [ ] Add saga pattern coordinator integration
        - [ ] Implement dead letter queue for failed messages
        - [ ] Support message replay for debugging
        - [ ] Add health check endpoints for monitoring

    Attributes:
        config: Dispatcher configuration dictionary.
        bus: Underlying FlextBus instance for execution.
        processors: FlextProcessors for specialized processing.

    Note:
        All dispatch methods return FlextResult for consistency.
        Circuit breaker and rate limiting are automatic.
        Context metadata propagates to all handlers. Correlation
        IDs enable distributed tracing across services.

    Warning:
        Circuit breaker threshold from FlextConstants.Reliability.DEFAULT_CIRCUIT_BREAKER_THRESHOLD.
        Rate limit from FlextConstants.Reliability.DEFAULT_RATE_LIMIT_MAX_REQUESTS per
        FlextConstants.Reliability.DEFAULT_RATE_LIMIT_WINDOW_SECONDS.
        Timeout from FlextConstants.Defaults.TIMEOUT_SECONDS, override per message.
        Batch processing may impact rate limiting calculations.

    Example:
        Complete dispatcher workflow with error handling:

        >>> dispatcher = FlextDispatcher.create_from_global_config()
        >>> dispatcher = dispatcher.unwrap()
        >>> reg_result = dispatcher.register_handler(CreateUserCommand, handler)
        >>> result = dispatcher.dispatch(command)
        >>> print(result.is_success)
        True

    See Also:
        FlextBus: For low-level command/query execution.
        FlextProcessors: For circuit breaker and rate limiting.
        FlextContext: For context management patterns.
        FlextHandlers: For handler base class patterns.

    """

    @override
    def __init__(
        self,
        *,
        config: FlextTypes.Dict | None = None,
        bus: FlextBus | None = None,
    ) -> None:
        """Initialize dispatcher with Pydantic configuration model.

        Refactored to eliminate SOLID violations by delegating to specialized components.

        Args:
            config: Optional dispatcher configuration model
            bus: Optional bus instance (created if not provided)

        """
        super().__init__()

        # Initialize service infrastructure (DI, Context, Logging, Metrics)
        self._init_service("flext_dispatcher")

        # Use global config instance for consistency across the system
        global_config = FlextConfig.get_global_instance()
        self._global_config = global_config

        if config is None:
            busglobal_config: FlextTypes.Dict = {
                "auto_context": global_config.dispatcher_auto_context,
                "timeout_seconds": global_config.dispatcher_timeout_seconds,
                "enable_metrics": global_config.dispatcher_enable_metrics,
                "enable_logging": global_config.dispatcher_enable_logging,
            }
            busglobal_config["execution_timeout"] = busglobal_config.get(
                "timeout_seconds",
                FlextConstants.Defaults.TIMEOUT,
            )

            config_dict: FlextTypes.Dict = {
                "auto_context": global_config.dispatcher_auto_context,
                "timeout_seconds": global_config.dispatcher_timeout_seconds,
                "enable_metrics": global_config.dispatcher_enable_metrics,
                "enable_logging": global_config.dispatcher_enable_logging,
                "max_retries": getattr(
                    global_config,
                    "dispatcher_max_retries",
                    getattr(global_config, "max_retry_attempts", 3),
                ),
                "retry_delay": getattr(
                    global_config,
                    "dispatcher_retry_delay",
                    getattr(global_config, "retry_delay_seconds", 0),
                ),
                "busglobal_config": busglobal_config,
                "execution_timeout": busglobal_config.get("timeout_seconds"),
            }
        else:
            config_dict = dict(config)
            busglobal_config_raw = config_dict.get("busglobal_config")

            if not isinstance(busglobal_config_raw, dict):
                busglobal_config = {
                    "auto_context": global_config.dispatcher_auto_context,
                    "timeout_seconds": global_config.dispatcher_timeout_seconds,
                    "enable_metrics": global_config.dispatcher_enable_metrics,
                    "enable_logging": global_config.dispatcher_enable_logging,
                }

                if "execution_timeout" in config_dict:
                    busglobal_config["execution_timeout"] = config_dict[
                        "execution_timeout"
                    ]
                elif "timeout_seconds" in config_dict:
                    busglobal_config["execution_timeout"] = config_dict[
                        "timeout_seconds"
                    ]

                config_dict["busglobal_config"] = busglobal_config
            else:
                busglobal_config = cast("FlextTypes.Dict", busglobal_config_raw)

            execution_timeout_value: object | None = busglobal_config.get(
                "execution_timeout",
            )
            if execution_timeout_value is not None:
                config_dict.setdefault("execution_timeout", execution_timeout_value)

        self.global_config: FlextTypes.Dict = config_dict
        config = config_dict

        # Initialize bus
        busglobal_config_raw = config.get("busglobal_config")
        busglobal_config_dict_final: FlextTypes.Dict | None
        if isinstance(busglobal_config_raw, FlextModels.Cqrs.Bus):
            busglobal_config_dict_final = busglobal_config_raw.model_dump()
        elif isinstance(busglobal_config_raw, dict):
            busglobal_config_dict_final = cast("FlextTypes.Dict", busglobal_config_raw)
        else:
            busglobal_config_dict_final = None

        self._bus = bus or FlextBus(bus_config=busglobal_config_dict_final)

        # Circuit breaker state - failure counts per message type
        self._circuit_breaker_failures: dict[str, int] = {}
        circuit_breaker_threshold_raw = config.get(
            "circuit_breaker_threshold",
            getattr(
                global_config,
                "circuit_breaker_threshold",
                FlextConstants.Reliability.DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
            ),
        )
        self._circuit_breaker_threshold = (
            int(circuit_breaker_threshold_raw)
            if isinstance(circuit_breaker_threshold_raw, (int, str))
            else FlextConstants.Reliability.DEFAULT_CIRCUIT_BREAKER_THRESHOLD
        )

        # Rate limiting state - sliding window with count, window_start, block_until
        self._rate_limit_requests: dict[str, FlextTypes.FloatList] = {}
        self._rate_limit_state: dict[
            str, FlextTypes.Reliability.DispatcherRateLimiterState
        ] = {}
        rate_limit_raw = config.get(
            "rate_limit",
            getattr(
                global_config,
                "rate_limit_max_requests",
                FlextConstants.Reliability.DEFAULT_RATE_LIMIT_MAX_REQUESTS,
            ),
        )
        self._rate_limit = (
            int(rate_limit_raw)
            if isinstance(rate_limit_raw, (int, str))
            else FlextConstants.Reliability.DEFAULT_RATE_LIMIT_MAX_REQUESTS
        )
        rate_limit_window_raw = config.get(
            "rate_limit_window",
            getattr(
                global_config,
                "rate_limit_window_seconds",
                FlextConstants.Reliability.DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
            ),
        )
        self._rate_limit_window = (
            float(rate_limit_window_raw)
            if isinstance(rate_limit_window_raw, (int, float, str))
            else float(FlextConstants.Reliability.DEFAULT_RATE_LIMIT_WINDOW_SECONDS)
        )
        rate_limit_grace_raw = config.get(
            "rate_limit_block_grace",
            max(1.0, 0.5 * self._rate_limit_window),
        )
        self._rate_limit_block_grace = float(
            rate_limit_grace_raw
            if isinstance(rate_limit_grace_raw, (int, float, str))
            else max(1.0, 0.5 * self._rate_limit_window),
        )

        # Timeout handling configuration
        self._use_timeout_executor = bool(
            config.get("enable_timeout_executor", False)
            or ("timeout" in config)
            or ("execution_timeout" in config),
        )

        # Thread pool for timeout handling (lazy initialization)
        self._executor_workers = self._resolve_executor_workers()
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

    @property
    def dispatcher_config(self) -> FlextTypes.Dict:
        """Access the dispatcher configuration."""
        return self.global_config

    @property
    def bus(self) -> FlextBus:
        """Access the underlying bus implementation."""
        return self._bus

    # ------------------------------------------------------------------
    # Registration methods using structured models
    # ------------------------------------------------------------------
    def register_handler_with_request(
        self,
        request: FlextTypes.Dict,
    ) -> FlextResult[FlextTypes.Dict]:
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
            return FlextResult[FlextTypes.Dict].fail(
                FlextConstants.Dispatcher.ERROR_INVALID_HANDLER_MODE,
            )

        # Validate handler is provided
        if request.get("handler") is None:
            return FlextResult[FlextTypes.Dict].fail(
                FlextConstants.Dispatcher.ERROR_HANDLER_REQUIRED,
            )

        # Register with bus
        bus_result = (
            self._bus.register_handler(
                request.get("message_type"),
                request.get("handler"),
            )
            if request.get("message_type")
            else self._bus.register_handler(request.get("handler"))
        )

        if bus_result.is_failure:
            return FlextResult[FlextTypes.Dict].fail(
                f"Bus registration failed: {bus_result.error}",
            )

        # Create registration details
        details: FlextTypes.Dict = {
            "registration_id": request.get("registration_id"),
            "message_type_name": getattr(request.get("message_type"), "__name__", None)
            if request.get("message_type")
            else None,
            "handler_mode": request.get("handler_mode"),
            "timestamp": FlextUtilities.Generators.generate_timestamp(),
            "status": FlextConstants.Dispatcher.REGISTRATION_STATUS_ACTIVE,
        }

        if self.global_config.get("enable_logging"):
            self._log_with_context(
                "info",
                "handler_registered",
                registration_id=details.get("registration_id"),
                handler_mode=details.get("handler_mode"),
                message_type=details.get("message_type_name"),
            )

        return FlextResult[FlextTypes.Dict].ok(details)

    def register_handler(
        self,
        message_type_or_handler: str
        | FlextHandlers[object, object]
        | Callable[[object], object | FlextResult[object]],
        handler: FlextHandlers[object, object]
        | Callable[[object], object | FlextResult[object]]
        | None = None,
        *,
        handler_mode: FlextConstants.HandlerModeSimple = "command",
        handler_config: FlextTypes.Dict | None = None,
    ) -> FlextResult[FlextTypes.Dict]:
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
                    return FlextResult[FlextTypes.Dict].fail(
                        handler_result.error or "Handler creation failed",
                    )
                resolved_handler = handler_result.data

            # Create structured request with message type
            request: FlextTypes.Dict = {
                "handler": resolved_handler,
                "message_type": message_type_or_handler,
                "handler_mode": handler_mode,
                "handler_config": handler_config,
            }
        else:
            # New API: register_handler(handler)
            # Ensure we have a handler, not a string
            if isinstance(message_type_or_handler, str):
                return FlextResult[FlextTypes.Dict].fail(
                    "Cannot register handler: message type string provided without handler",
                )

            # Convert callable to FlextHandlers if needed
            resolved_handler = message_type_or_handler
            if callable(message_type_or_handler) and not isinstance(
                message_type_or_handler,
                FlextHandlers,
            ):
                handler_result = self.create_handler_from_function(
                    handler_func=message_type_or_handler,
                    handler_config=handler_config,
                    mode=handler_mode,
                )
                if handler_result.is_failure:
                    return FlextResult[FlextTypes.Dict].fail(
                        handler_result.error or "Handler creation failed",
                    )
                resolved_handler = handler_result.data

            # Create structured request
            request = {
                "handler": resolved_handler,
                "message_type": None,
                "handler_mode": handler_mode,
                "handler_config": handler_config,
            }

        return self.register_handler_with_request(request)

    def register_command(
        self,
        command_type: type[object],
        handler: FlextHandlers[object, object],
        *,
        handler_config: FlextTypes.Dict | None = None,
    ) -> FlextResult[FlextTypes.Dict]:
        """Register command handler using structured model internally.

        Args:
            command_type: Command message type
            handler: Handler instance
            handler_config: Optional handler configuration

        Returns:
            FlextResult with registration details or error

        """
        request: FlextTypes.Dict = {
            "handler": handler,
            "message_type": command_type,
            "handler_mode": FlextConstants.Dispatcher.HANDLER_MODE_COMMAND,
            "handler_config": handler_config,
        }

        return self.register_handler_with_request(request)

    def register_query(
        self,
        query_type: type[object],
        handler: FlextHandlers[object, object],
        *,
        handler_config: FlextTypes.Dict | None = None,
    ) -> FlextResult[FlextTypes.Dict]:
        """Register query handler using structured model internally.

        Args:
            query_type: Query message type
            handler: Handler instance
            handler_config: Optional handler configuration

        Returns:
            FlextResult with registration details or error

        """
        request: FlextTypes.Dict = {
            "handler": handler,
            "message_type": query_type,
            "handler_mode": FlextConstants.Dispatcher.HANDLER_MODE_QUERY,
            "handler_config": handler_config,
        }

        return self.register_handler_with_request(request)

    def register_function(
        self,
        message_type: type[object],
        handler_func: Callable[[object], object | FlextResult[object]],
        *,
        handler_config: FlextTypes.Dict | None = None,
        mode: FlextConstants.HandlerModeSimple = "command",
    ) -> FlextResult[FlextTypes.Dict]:
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
            return FlextResult[FlextTypes.Dict].fail(
                FlextConstants.Dispatcher.ERROR_INVALID_HANDLER_MODE,
            )

        # Create handler from function
        handler_result = self.create_handler_from_function(
            handler_func,
            handler_config,
            mode,
        )

        if handler_result.is_failure:
            return FlextResult[FlextTypes.Dict].fail(
                f"Handler creation failed: {handler_result.error}",
            )

        # Register the created handler
        request: FlextTypes.Dict = {
            "handler": handler_result.value,
            "message_type": message_type,
            "handler_mode": mode,
            "handler_config": handler_config,
        }

        return self.register_handler_with_request(request)

    def create_handler_from_function(
        self,
        handler_func: Callable[[object], object | FlextResult[object]],
        handler_config: FlextTypes.Dict | None,
        mode: FlextConstants.HandlerModeSimple,
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
                f"Handler creation failed: {error}",
            )

    # ------------------------------------------------------------------
    # Dispatch execution using structured models
    # ------------------------------------------------------------------
    def dispatch_with_request(
        self,
        request: FlextTypes.Dict,
    ) -> FlextResult[FlextTypes.Dict]:
        """Dispatch using structured request model.

        Args:
            request: Pydantic model containing dispatch details

        Returns:
            FlextResult with structured dispatch result

        """
        # Propagate context for distributed tracing
        message = request.get("message")
        message_type = type(message).__name__ if message else "unknown"
        self._propagate_context(f"dispatch_with_request_{message_type}")

        start_time = time.time()

        # Validate request
        if request.get("message") is None:
            return FlextResult[FlextTypes.Dict].fail(
                FlextConstants.Dispatcher.ERROR_MESSAGE_REQUIRED,
            )

        # Get timeout from request override or config
        timeout_override = request.get("timeout_override")
        config_timeout = self.global_config.get("timeout_seconds")
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

            if execution_result.is_success:
                dispatch_result: FlextTypes.Dict = {
                    "success": True,
                    "result": execution_result.value,
                    "error_message": None,
                    "request_id": request.get("request_id"),
                    "execution_time_ms": execution_time_ms,
                    "correlation_id": request.get("correlation_id"),
                    "timeout_seconds": timeout_seconds,
                }

                if self.global_config.get("enable_logging"):
                    self._log_with_context(
                        "debug",
                        "dispatch_succeeded",
                        request_id=request.get("request_id"),
                        message_type=type(request.get("message")).__name__,
                        execution_time_ms=execution_time_ms,
                        timeout_seconds=timeout_seconds,
                    )

                return FlextResult[FlextTypes.Dict].ok(dispatch_result)

            dispatch_result = {
                "success": False,
                "result": None,
                "error_message": execution_result.error or "Unknown error",
                "request_id": request.get("request_id"),
                "execution_time_ms": execution_time_ms,
                "correlation_id": request.get("correlation_id"),
                "timeout_seconds": timeout_seconds,
            }

            if self.global_config.get("enable_logging"):
                self._log_with_context(
                    "error",
                    "dispatch_failed",
                    request_id=request.get("request_id"),
                    message_type=type(request.get("message")).__name__,
                    error=dispatch_result.get("error_message"),
                    execution_time_ms=execution_time_ms,
                    timeout_seconds=timeout_seconds,
                )

            return FlextResult[FlextTypes.Dict].ok(dispatch_result)

    def dispatch(
        self,
        message_or_type: object | str,
        data: object | None = None,
        *,
        metadata: FlextTypes.Dict | None = None,
        correlation_id: str | None = None,
        timeout_override: int | None = None,
    ) -> FlextResult[object]:
        """Dispatch message with support for both old and new API.

        Refactored to use specialized processors for SOLID compliance:
        - Circuit breaker, rate limiting, caching → FlextProcessors
        - Timeout, retry → Uses threading with processors

        Args:
            message_or_type: Message object or message type string
            data: Data to dispatch (when message_or_type is string)
            metadata: Optional execution context metadata
            correlation_id: Optional correlation ID for tracing (reserved for future use)
            timeout_override: Optional timeout override (reserved for future use)

        Returns:
            FlextResult with execution result or error

        """
        # Propagate context for distributed tracing
        dispatch_type = (
            type(message_or_type).__name__
            if not isinstance(message_or_type, str)
            else str(message_or_type)
        )
        self._propagate_context(f"dispatch_{dispatch_type}")

        # Support both old API (message_type, data) and new API (message)
        if isinstance(message_or_type, str):
            if data is not None:
                # Old API: dispatch(message_type, data)
                if not data or data is None:
                    return FlextResult[object].fail("Message is required")
                message_type = message_or_type
                message = data
            else:
                # Old API: dispatch(message_type) - no data provided
                message_type = message_or_type
                message = None
        else:
            # New API: dispatch(message)
            message = message_or_type
            message_type = type(message).__name__ if message else "unknown"

        # Check circuit breaker
        failures = self._circuit_breaker_failures.get(message_type, 0)
        if failures >= self._circuit_breaker_threshold:
            return FlextResult[object].fail(
                f"Circuit breaker is open for message type '{message_type}'",
                error_code=FlextConstants.Errors.OPERATION_ERROR,
                error_data={
                    "message_type": message_type,
                    "failure_count": failures,
                    "threshold": self._circuit_breaker_threshold,
                    "reason": "circuit_breaker_open",
                },
            )

        # Check rate limiting
        current_time = time.time()
        state = self._rate_limit_state.get(message_type)
        if state is None:
            new_state: FlextTypes.Reliability.DispatcherRateLimiterState = {
                "count": 0,
                "window_start": current_time,
                "block_until": 0.0,
            }
            self._rate_limit_state[message_type] = new_state
            state = new_state

        if current_time < state.get("block_until", 0.0):
            retry_after = int(state.get("block_until", 0.0) - current_time)
            return FlextResult[object].fail(
                f"Rate limit exceeded for message type '{message_type}' - blocked until recovery",
                error_code=FlextConstants.Errors.OPERATION_ERROR,
                error_data={
                    "message_type": message_type,
                    "limit": self._rate_limit,
                    "window_seconds": self._rate_limit_window,
                    "retry_after": retry_after,
                    "reason": "rate_limit_blocked",
                },
            )

        if (
            current_time - state.get("window_start", current_time)
            >= self._rate_limit_window
        ):
            state["count"] = 0
            state["window_start"] = current_time
            state["block_until"] = 0.0

        if state["count"] >= self._rate_limit:
            state["block_until"] = (
                current_time + self._rate_limit_window + self._rate_limit_block_grace
            )
            retry_after = int(self._rate_limit_window + self._rate_limit_block_grace)
            return FlextResult[object].fail(
                f"Rate limit exceeded for message type '{message_type}' - too many requests",
                error_code=FlextConstants.Errors.OPERATION_ERROR,
                error_data={
                    "message_type": message_type,
                    "limit": self._rate_limit,
                    "window_seconds": self._rate_limit_window,
                    "current_count": state["count"],
                    "retry_after": retry_after,
                    "reason": "rate_limit_exceeded",
                },
            )

        state["count"] += 1
        if state["count"] >= self._rate_limit:
            state["block_until"] = (
                current_time + self._rate_limit_window + self._rate_limit_block_grace
            )
        else:
            state["block_until"] = 0.0

        requests_list = self._rate_limit_requests.setdefault(message_type, [])
        requests_list.append(current_time)
        while len(requests_list) > self._rate_limit * 2:
            requests_list.pop(0)

        # Create message object
        if isinstance(message_or_type, str) and data is not None:

            class MessageWrapper(FlextModels.Value):
                """Temporary message wrapper using FlextModels.Value."""

                data: object
                message_type: str

                def model_post_init(self, /, __context: object) -> None:
                    """Post-initialization to set class name."""
                    super().model_post_init(__context)
                    self.__class__.__name__ = self.message_type

                def __str__(self) -> str:
                    """String representation."""
                    return str(self.data)

            message = MessageWrapper(data=data, message_type=message_or_type)
            message_type = message_or_type
        else:
            message = message_or_type

        # Create structured request
        if metadata:
            string_metadata: FlextTypes.Dict = {k: str(v) for k, v in metadata.items()}
            FlextModels.Metadata(attributes=string_metadata)

        # Execute dispatch with retry logic using bus directly
        max_retries_raw = self.global_config.get(
            "max_retries",
            FlextConstants.Reliability.DEFAULT_MAX_RETRIES,
        )
        max_retries: int = (
            int(max_retries_raw)
            if isinstance(max_retries_raw, (int, float))
            else FlextConstants.Reliability.DEFAULT_MAX_RETRIES
        )
        retry_delay_raw = self.global_config.get("retry_delay", 0.1)
        retry_delay: float = (
            float(retry_delay_raw) if isinstance(retry_delay_raw, (int, float)) else 0.1
        )

        # start_time = time.time()  # Unused for now

        for attempt in range(max_retries):
            # attempt_start_time = time.time()  # Unused for now
            try:
                # Get timeout from config
                timeout_seconds = float(
                    cast(
                        "int | float",
                        self.global_config.get(
                            "timeout",
                            FlextConstants.Defaults.TIMEOUT_SECONDS,
                        ),
                    ),
                )
                if timeout_override:
                    timeout_seconds = float(timeout_override)

                # Execute with timeout using shared ThreadPoolExecutor when enabled
                def execute_with_context() -> FlextResult[object]:
                    if correlation_id is not None or timeout_override is not None:
                        context_metadata: FlextTypes.Dict = {}
                        if timeout_override is not None:
                            context_metadata["timeout_override"] = timeout_override

                        with self._context_scope(context_metadata, correlation_id):
                            return self._bus.execute(message)
                    else:
                        return self._bus.execute(message)

                use_executor = (
                    self._use_timeout_executor or timeout_override is not None
                )

                if use_executor:
                    executor = self._ensure_executor()
                    future: concurrent.futures.Future[FlextResult[object]] | None = None
                    try:
                        future = executor.submit(execute_with_context)
                        bus_result = future.result(timeout=timeout_seconds)
                    except concurrent.futures.TimeoutError:
                        # Cancel the future and return timeout error
                        if future is not None:
                            future.cancel()
                        return FlextResult[object].fail(
                            f"Operation timeout after {timeout_seconds} seconds",
                        )
                    except RuntimeError:
                        # Executor was shut down; reinitialize and retry immediately
                        self._executor = None
                        continue
                else:
                    bus_result = execute_with_context()
                # attempt_end_time = time.time()  # Unused for now
                # attempt_duration = attempt_end_time - attempt_start_time  # Unused for now

                if bus_result.is_success:
                    # Reset circuit breaker failures on success
                    self._circuit_breaker_failures[message_type] = 0
                    return FlextResult[object].ok(bus_result.value)

                # Track circuit breaker failure
                self._circuit_breaker_failures[message_type] = (
                    self._circuit_breaker_failures.get(message_type, 0) + 1
                )

                # Check if this is a temporary failure that should be retried
                if attempt < max_retries - 1 and "Temporary failure" in str(
                    bus_result.error,
                ):
                    time.sleep(retry_delay)
                    continue

                return FlextResult[object].fail(bus_result.error or "Dispatch failed")
            except Exception as e:
                # attempt_end_time = time.time()  # Unused for now
                # attempt_duration = attempt_end_time - attempt_start_time  # Unused for now

                # Track circuit breaker failure for exceptions
                self._circuit_breaker_failures[message_type] = (
                    self._circuit_breaker_failures.get(message_type, 0) + 1
                )

                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return FlextResult[object].fail(f"Dispatch error: {e}")

        # Record final failure
        # end_time = time.time()  # Unused for now
        # total_duration = end_time - start_time  # Unused for now
        return FlextResult[object].fail("Max retries exceeded")

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------
    def _resolve_executor_workers(self) -> int:
        """Determine worker count for the shared dispatcher executor."""
        workers_candidate = (
            self.global_config.get("executor_workers")
            or self.global_config.get("max_workers")
            or FlextConstants.Container.DEFAULT_WORKERS
        )
        try:
            if isinstance(workers_candidate, (int, str)):
                workers = int(workers_candidate)
            else:
                workers = FlextConstants.Container.DEFAULT_WORKERS
        except (TypeError, ValueError):
            workers = FlextConstants.Container.DEFAULT_WORKERS
        return max(workers, 1)

    def _ensure_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Create the shared executor on demand."""
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._executor_workers,
                thread_name_prefix="flext-dispatcher",
            )
        return self._executor

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------
    def _normalize_context_metadata(
        self,
        metadata: object | None,
    ) -> FlextTypes.Dict | None:
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

        normalized: FlextTypes.Dict = {
            str(key): value for key, value in raw_metadata.items()
        }

        return dict(normalized)

    @contextmanager
    def _context_scope(
        self,
        metadata: FlextTypes.Dict | None = None,
        correlation_id: str | None = None,
    ) -> Generator[None]:
        """Manage execution context with optional metadata and correlation ID.

        Args:
            metadata: Optional metadata to include in context
            correlation_id: Optional correlation ID for tracing

        """
        if not self.global_config.get("auto_context"):
            yield
            return

        metadata_var = FlextContext.Variables.Performance.OPERATION_METADATA
        correlation_var = FlextContext.Variables.Correlation.CORRELATION_ID
        parent_var = FlextContext.Variables.Correlation.PARENT_CORRELATION_ID

        # Store current context values for restoration
        current_parent = parent_var.get()

        # Set new correlation ID if provided
        if correlation_id is not None:
            correlation_var.set(correlation_id)
            # Set parent correlation ID if there was a previous one
            if current_parent is not None and current_parent != correlation_id:
                parent_var.set(current_parent)

        # Set metadata if provided
        if metadata:
            metadata_var.set(metadata)

            # Use provided correlation ID or generate one if needed
            effective_correlation_id = correlation_id
            if effective_correlation_id is None:
                effective_correlation_id = (
                    FlextContext.Correlation.generate_correlation_id()
                )

            if self.global_config.get("enable_logging"):
                self._log_with_context(
                    "debug",
                    "dispatch_context_entered",
                    correlation_id=effective_correlation_id,
                )

            yield

            if self.global_config.get("enable_logging"):
                self._log_with_context(
                    "debug",
                    "dispatch_context_exited",
                    correlation_id=effective_correlation_id,
                )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
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
                f"Dispatcher creation failed: {error}",
            )

    # =============================================================================
    # Missing Methods for Test Compatibility
    # =============================================================================

    def dispatch_batch(
        self,
        message_type: str,
        messages: list[object],
    ) -> list[FlextResult[object]]:
        """Dispatch multiple messages in batch.

        Args:
            message_type: Type of messages to dispatch
            messages: List of message data to dispatch

        Returns:
            List of FlextResult objects for each dispatched message

        """
        return [self.dispatch(message_type, msg) for msg in messages]

    def get_performance_metrics(self) -> FlextTypes.Dict:
        """Get performance metrics for the dispatcher.

        Returns:
            Dictionary containing performance metrics

        """
        # Basic metrics - can be extended with actual performance data
        return {
            "total_dispatches": 0,  # Track actual dispatches (future enhancement)
            "circuit_breaker_failures": len(self._circuit_breaker_failures),
            "rate_limit_states": len(self._rate_limit_state),
            "executor_workers": self._executor_workers if self._executor else 0,
        }

    def cleanup(self) -> None:
        """Clean up dispatcher resources using processors."""
        try:
            # Clear all handlers using the public API
            if (
                hasattr(self, "_bus")
                and self._bus
                and hasattr(self._bus, "clear_handlers")
            ):
                self._bus.clear_handlers()

            # Clear internal state
            self._circuit_breaker_failures.clear()
            self._rate_limit_requests.clear()
            self._rate_limit_state.clear()
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None

        except Exception as e:
            self._log_with_context("warning", "Cleanup failed", error=str(e))


__all__ = ["FlextDispatcher"]
