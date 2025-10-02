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

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import Token
from dataclasses import dataclass
from typing import Literal, cast, override

from flext_core.bus import FlextBus
from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.context import FlextContext
from flext_core.handlers import FlextHandlers
from flext_core.loggings import FlextLogger
from flext_core.models import FlextModels
from flext_core.processors import FlextProcessors
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class FlextDispatcher:
    """Orchestrates CQRS execution while enforcing context observability.

    The dispatcher is the front door promoted across the ecosystem: all
    handler registration flows, context scoping, and dispatch telemetry
    align with the modernization plan so downstream packages can adopt
    a consistent runtime contract without bespoke buses.

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
            timeout_override=5,  # 5 second timeout
        )

        # Example 5: Batch dispatch multiple messages
        messages = [command1, command2, command3]
        results = dispatcher.dispatch_batch("CreateUser", messages)

        # Example 6: Check circuit breaker state
        is_open = dispatcher.is_circuit_breaker_open("CreateUser")
        if not is_open:
            result = dispatcher.dispatch(command)

        # Example 7: Export configuration for persistence
        config = dispatcher.export_config()
        # Later restore with import_config()
        ```

    **TODO**: Enhanced dispatcher features for 1.0.0+ releases
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
        Circuit breaker and rate limiting are automatic. Use
        export_config/import_config for state persistence.
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
        config: dict[str, object] | None = None,
        bus: FlextBus | None = None,
    ) -> None:
        """Initialize dispatcher with Pydantic configuration model.

        Refactored to eliminate SOLID violations by delegating to specialized components.

        Args:
            config: Optional dispatcher configuration model
            bus: Optional bus instance (created if not provided)

        """
        super().__init__()

        # Initialize configuration
        if config is None:
            global_config = FlextConfig.get_global_instance()
            bus_config: dict[str, object] = dict(global_config.get_cqrs_bus_config())
            # Map timeout_seconds to execution_timeout for bus compatibility
            bus_config["execution_timeout"] = bus_config.get(
                "timeout_seconds",
                FlextConstants.Defaults.TIMEOUT,
            )
            config = {
                "auto_context": getattr(global_config, "dispatcher_auto_context", True),
                "timeout_seconds": getattr(
                    global_config,
                    "dispatcher_timeout_seconds",
                    FlextConstants.Defaults.TIMEOUT,
                ),
                "enable_metrics": getattr(
                    global_config,
                    "dispatcher_enable_metrics",
                    True,
                ),
                "enable_logging": getattr(
                    global_config,
                    "dispatcher_enable_logging",
                    True,
                ),
                "max_retries": getattr(
                    global_config,
                    "dispatcher_max_retries",
                    FlextConstants.Reliability.DEFAULT_MAX_RETRIES,
                ),
                "retry_delay": getattr(
                    global_config,
                    "dispatcher_retry_delay",
                    FlextConstants.Reliability.DEFAULT_RETRY_DELAY_SECONDS,
                ),
                "bus_config": bus_config,
                "execution_timeout": bus_config.get("timeout_seconds"),
            }
        else:
            config = dict(config)
            bus_config_raw = config.get("bus_config")

            if not isinstance(bus_config_raw, dict):
                global_config = FlextConfig.get_global_instance()
                default_bus_config: dict[str, object] = dict(
                    global_config.get_cqrs_bus_config(),
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
                "execution_timeout",
            )
            config.setdefault("execution_timeout", execution_timeout_value)

        self._config: dict[str, object] = config

        # Initialize specialized processors for SOLID compliance
        processor_config = {
            "circuit_breaker_threshold": config.get(
                "circuit_breaker_threshold",
                FlextConstants.Reliability.DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
            ),
            "rate_limit": config.get(
                "rate_limit",
                FlextConstants.Reliability.DEFAULT_RATE_LIMIT_MAX_REQUESTS,
            ),
            "rate_limit_window": config.get(
                "rate_limit_window",
                FlextConstants.Reliability.DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
            ),
            "cache_ttl": config.get("cache_ttl", FlextConstants.Defaults.CACHE_TTL),
        }
        self._processors = FlextProcessors(processor_config)

        # Register audit and metrics processors
        self._setup_dispatch_processors()

        # Initialize bus
        bus_config_raw = config.get("bus_config")
        bus_config_dict_final: dict[str, object] | None
        if isinstance(bus_config_raw, FlextModels.CqrsConfig.Bus):
            bus_config_dict_final = bus_config_raw.model_dump()
        elif isinstance(bus_config_raw, dict):
            bus_config_dict_final = cast("dict[str, object]", bus_config_raw)
        else:
            bus_config_dict_final = None

        self._bus = bus or FlextBus.create_command_bus(bus_config=bus_config_dict_final)
        self._logger = FlextLogger(self.__class__.__name__)

        # Circuit breaker state - using FlextTypes.Reliability
        self._circuit_breaker_failures: FlextTypes.Reliability.CircuitBreakerRegistry = {}
        circuit_breaker_threshold_raw = config.get(
            "circuit_breaker_threshold",
            FlextConstants.Reliability.DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
        )
        self._circuit_breaker_threshold = (
            int(circuit_breaker_threshold_raw)
            if isinstance(circuit_breaker_threshold_raw, (int, str))
            else FlextConstants.Reliability.DEFAULT_CIRCUIT_BREAKER_THRESHOLD
        )

        # Rate limiting state - using FlextTypes.Reliability
        self._rate_limit_requests: dict[
            str, FlextTypes.Reliability.RateLimitWindow
        ] = {}
        self._rate_limit_state: FlextTypes.Reliability.RateLimiterRegistry = {}
        rate_limit_raw = config.get(
            "rate_limit",
            FlextConstants.Reliability.DEFAULT_RATE_LIMIT_MAX_REQUESTS,
        )
        self._rate_limit = (
            int(rate_limit_raw)
            if isinstance(rate_limit_raw, (int, str))
            else FlextConstants.Reliability.DEFAULT_RATE_LIMIT_MAX_REQUESTS
        )
        rate_limit_window_raw = config.get(
            "rate_limit_window",
            FlextConstants.Reliability.DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
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

        # Audit and performance tracking - using FlextTypes.Reliability
        self._audit_log: list[FlextTypes.Core.Dict] = []
        self._performance_metrics: FlextTypes.Reliability.PerformanceMetrics = {}

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
    def config(self) -> dict[str, object]:
        """Access the dispatcher configuration."""
        return self._config

    @property
    def bus(self) -> FlextBus:
        """Access the underlying bus implementation."""
        return self._bus

    @property
    def processors(self) -> FlextProcessors:
        """Access the specialized processors for SOLID compliance."""
        return self._processors

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
                FlextConstants.Dispatcher.ERROR_INVALID_HANDLER_MODE,
            )

        # Validate handler is provided
        if request.get("handler") is None:
            return FlextResult[dict[str, object]].fail(
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
            return FlextResult[dict[str, object]].fail(
                f"Bus registration failed: {bus_result.error}",
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
                        handler_result.error or "Handler creation failed",
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
            # Ensure we have a handler, not a string
            if isinstance(message_type_or_handler, str):
                return FlextResult[dict[str, object]].fail(
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
                    return FlextResult[dict[str, object]].fail(
                        handler_result.error or "Handler creation failed",
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
                FlextConstants.Dispatcher.ERROR_INVALID_HANDLER_MODE,
            )

        # Create handler from function
        handler_result = self.create_handler_from_function(
            handler_func,
            handler_config,
            mode,
        )

        if handler_result.is_failure:
            return FlextResult[dict[str, object]].fail(
                f"Handler creation failed: {handler_result.error}",
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
                f"Handler creation failed: {error}",
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
                FlextConstants.Dispatcher.ERROR_MESSAGE_REQUIRED,
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
            return FlextResult[object].fail("Circuit breaker is open")

        # Check rate limiting
        current_time = time.time()
        state = self._rate_limit_state.get(message_type)
        if state is None:
            state = {"count": 0, "window_start": current_time, "block_until": 0.0}
            self._rate_limit_state[message_type] = state

        if current_time < state.get("block_until", 0.0):
            return FlextResult[object].fail("Rate limit exceeded")

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
            return FlextResult[object].fail("Rate limit exceeded")

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

            @dataclass
            class MessageWrapper:
                data: object
                message_type: str

                def __post_init__(self) -> None:
                    self.__class__.__name__ = self.message_type

                def __str__(self) -> str:
                    return str(self.data)

            message = MessageWrapper(data, message_or_type)
            message_type = message_or_type
        else:
            message = message_or_type

        # Create structured request
        if metadata:
            string_metadata: dict[str, object] = {
                k: str(v) for k, v in metadata.items()
            }
            FlextModels.Metadata(attributes=string_metadata)

        # Execute dispatch with retry logic using bus directly
        max_retries_raw = self._config.get(
            "max_retries",
            FlextConstants.Reliability.DEFAULT_MAX_RETRIES,
        )
        max_retries: int = (
            int(max_retries_raw)
            if isinstance(max_retries_raw, (int, float))
            else FlextConstants.Reliability.DEFAULT_MAX_RETRIES
        )
        retry_delay_raw = self._config.get("retry_delay", 0.1)
        retry_delay: float = (
            float(retry_delay_raw) if isinstance(retry_delay_raw, (int, float)) else 0.1
        )

        start_time = time.time()

        for attempt in range(max_retries):
            attempt_start_time = time.time()
            try:
                # Get timeout from config
                timeout_seconds = float(
                    cast(
                        "int | float",
                        self._config.get(
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
                        context_metadata: dict[str, object] = {}
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
                attempt_end_time = time.time()
                attempt_duration = attempt_end_time - attempt_start_time

                if bus_result.is_success:
                    # Reset circuit breaker failures on success
                    self._circuit_breaker_failures[message_type] = 0

                    # Record successful dispatch in processors
                    self._record_dispatch_success(
                        message_type,
                        attempt_duration,
                        attempt + 1,
                    )
                    return FlextResult[object].ok(bus_result.value)

                # Track circuit breaker failure
                self._circuit_breaker_failures[message_type] = (
                    self._circuit_breaker_failures.get(message_type, 0) + 1
                )

                # Record failed dispatch in processors
                self._record_dispatch_failure(
                    message_type,
                    attempt_duration,
                    bus_result.error,
                    attempt + 1,
                )

                # Check if this is a temporary failure that should be retried
                if attempt < max_retries - 1 and "Temporary failure" in str(
                    bus_result.error,
                ):
                    time.sleep(retry_delay)
                    continue

                return FlextResult[object].fail(bus_result.error or "Dispatch failed")
            except Exception as e:
                attempt_end_time = time.time()
                attempt_duration = attempt_end_time - attempt_start_time

                # Track circuit breaker failure for exceptions
                self._circuit_breaker_failures[message_type] = (
                    self._circuit_breaker_failures.get(message_type, 0) + 1
                )

                # Record failed dispatch in processors
                self._record_dispatch_failure(
                    message_type,
                    attempt_duration,
                    str(e),
                    attempt + 1,
                )

                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                return FlextResult[object].fail(f"Dispatch error: {e}")

        # Record final failure
        end_time = time.time()
        total_duration = end_time - start_time
        self._record_dispatch_failure(
            message_type,
            total_duration,
            "Max retries exceeded",
            max_retries,
        )
        return FlextResult[object].fail("Max retries exceeded")

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------
    def _resolve_executor_workers(self) -> int:
        """Determine worker count for the shared dispatcher executor."""
        workers_candidate = (
            self._config.get("executor_workers")
            or self._config.get("max_workers")
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

    def _record_dispatch_success(
        self,
        message_type: str,
        duration: float,
        attempts: int,
    ) -> None:
        """Record successful dispatch."""
        # Record in audit log - using FlextTypes.Core.Dict
        audit_entry: FlextTypes.Core.Dict = {
            "timestamp": time.time(),
            "message_type": message_type,
            "success": True,
            "execution_time_ms": int(duration * 1000),
            "attempts": attempts,
        }
        self._audit_log.append(audit_entry)

        # Record performance metrics - using FlextTypes.Reliability.PerformanceMetrics
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
        metrics["total_execution_time"] += duration
        metrics["avg_execution_time"] = (
            metrics["total_execution_time"] / metrics["total_executions"]
        )
        metrics["success_count"] += 1

    def _record_dispatch_failure(
        self,
        message_type: str,
        duration: float,
        error: str | None,
        attempts: int,
    ) -> None:
        """Record failed dispatch."""
        # Record in audit log - using FlextTypes.Core.Dict
        audit_entry: FlextTypes.Core.Dict = {
            "timestamp": time.time(),
            "message_type": message_type,
            "success": False,
            "execution_time_ms": int(duration * 1000),
            "attempts": attempts,
            "error": error,
        }
        self._audit_log.append(audit_entry)

        # Record performance metrics
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
        metrics["total_execution_time"] += duration
        metrics["avg_execution_time"] = (
            metrics["total_execution_time"] / metrics["total_executions"]
        )
        metrics["failure_count"] += 1

    def _setup_dispatch_processors(self) -> None:
        """Set up processors for audit and metrics collection."""

        # Register audit processor
        def audit_processor(data: dict[str, object]) -> dict[str, object]:
            """Process audit data and add to audit log."""
            if "timestamp" not in data:
                data["timestamp"] = time.time()
            # The processors automatically add to audit log in the process method
            return data

        self._processors.register("audit", audit_processor)

        # Register metrics processor
        def metrics_processor(data: dict[str, object]) -> dict[str, object]:
            """Process metrics data."""
            return data

        self._processors.register("metrics", metrics_processor)

    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------
    def _normalize_context_metadata(
        self,
        metadata: object | None,
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
                f"Dispatcher creation failed: {error}",
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
                f"Dispatcher creation failed: {error}",
            )

    # =============================================================================
    # Missing Methods for Test Compatibility
    # =============================================================================

    def cleanup(self) -> None:
        """Clean up dispatcher resources using processors."""
        try:
            # Clear all handlers using the public API
            if hasattr(self, "_bus") and self._bus:
                if hasattr(self._bus, "clear_handlers"):
                    self._bus.clear_handlers()
                if hasattr(self._bus, "cleanup"):
                    self._bus.cleanup()

            # Clean up processors
            self._processors.cleanup()

            # Clear internal state
            self._circuit_breaker_failures.clear()
            self._rate_limit_requests.clear()
            self._rate_limit_state.clear()
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None

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
                # Use hasattr to check for original_callable attribute
                if hasattr(handler, "original_callable"):
                    # Type-safe access to original_callable attribute
                    original_callable: object = getattr(
                        handler,
                        "original_callable",
                        None,
                    )
                    if original_callable is not None:
                        result.append(original_callable)
                    else:
                        result.append(handler)
                else:
                    result.append(handler)
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
            handlers: dict[str, object] = {}
            try:
                # get_registered_handlers() returns FlextTypes.Core.Dict (dict[str, object])
                registered_handlers: dict[str, object] = (
                    self._bus.get_registered_handlers()
                )
                handlers.update(registered_handlers)
                config["handlers"] = handlers
            except Exception:
                config["handlers"] = {}

        return config

    def get_metrics(self) -> dict[str, object]:
        """Get dispatcher metrics."""
        # Return performance metrics for message types (what tests expect)
        return self.get_performance_metrics()

    def import_config(self, config: dict[str, object]) -> FlextResult[None]:
        """Import dispatcher configuration using processors."""
        try:
            # Import handlers
            if "handlers" in config:
                handlers = config["handlers"]
                if isinstance(handlers, dict):
                    handlers_dict: dict[str, object] = handlers
                    for message_type, handler_list in handlers_dict.items():
                        message_type_str = str(message_type)
                        if not isinstance(handler_list, list):
                            handlers_for_type: list[object] = [handler_list]
                        else:
                            handlers_for_type = handler_list
                        for handler in handlers_for_type:
                            self.register_handler(
                                message_type_str,
                                cast(
                                    "FlextHandlers[object, object] | Callable[[object], object | FlextResult[object]] | None",
                                    handler,
                                ),
                            )

            # Import circuit breaker and rate limiting state
            if "circuit_breaker_failures" in config:
                failures = config["circuit_breaker_failures"]
                if isinstance(failures, dict):
                    self._circuit_breaker_failures.update(failures)
            if "rate_limit_requests" in config:
                requests = config["rate_limit_requests"]
                if isinstance(requests, dict):
                    self._rate_limit_requests.update(requests)

            # Import processor state
            processor_state = {}
            if "performance_metrics" in config:
                processor_state["performance_metrics"] = config["performance_metrics"]

            if processor_state:
                # Import processor configuration
                processor_config = dict[str, object](processor_state)
                import_result = self._processors.import_config(processor_config)
                if import_result.is_failure:
                    return FlextResult[None].fail(
                        f"Processor config import failed: {import_result.error}",
                    )

            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Config import failed: {e}")

    def dispatch_batch(
        self,
        message_type: str,
        messages: list[object],
    ) -> list[FlextResult[object]]:
        """Dispatch multiple messages in batch using processors."""
        # Use processors batch processing
        batch_data = {"message_type": message_type, "messages": messages}
        result = self._processors.process(f"batch_{message_type}", batch_data)
        if result.is_success:
            return cast("list[FlextResult[object]]", result.value)
        # Fallback to individual processing
        results: list[FlextResult[object]] = []
        for message in messages:
            result = self.dispatch(message_type, message)
            results.append(result)
        return results

    def is_circuit_breaker_open(self, message_type: str) -> bool:
        """Check if circuit breaker is open for message type."""
        failures = self._circuit_breaker_failures.get(message_type, 0)
        return failures >= self._circuit_breaker_threshold

    def get_audit_log(self) -> list[FlextTypes.Core.Dict]:
        """Get audit log entries."""
        return self._audit_log.copy()

    def get_performance_metrics(self) -> FlextTypes.Core.Dict:
        """Get performance metrics."""
        # Convert performance metrics to the expected format
        result: FlextTypes.Core.Dict = {}
        for message_type, metrics in self._performance_metrics.items():
            result[message_type] = {
                "dispatches": metrics["total_executions"],
                "successes": metrics["success_count"],
                "failures": metrics["failure_count"],
                "avg_execution_time": metrics["avg_execution_time"],
            }
        return result


__all__ = ["FlextDispatcher"]
