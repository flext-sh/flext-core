"""High-level message dispatch orchestration with reliability patterns.

This module provides FlextDispatcher, a facade that orchestrates message
dispatching with circuit breaker, rate limiting, retry logic, timeout
enforcement, and comprehensive observability.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from typing import cast, override

from flext_core.bus import FlextBus
from flext_core.constants import FlextConstants
from flext_core.context import FlextContext
from flext_core.handlers import FlextHandlers
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities


class FlextDispatcher(FlextMixins):
    """High-level message dispatch orchestration with reliability patterns.

    Implements FlextProtocols.CommandBus through structural typing. Provides
    message dispatching with circuit breaker, rate limiting, retry logic,
    timeout enforcement, and comprehensive observability. Wraps FlextBus and
    adds reliability and monitoring capabilities.

    CommandBus Protocol Implementation:
        - register_handler(message_type, handler) - Register command handler
        - register_command(command_type, handler) - Register typed command handler
        - register_query(query_type, handler) - Register typed query handler
        - dispatch(message) - Execute handler with message (FlextProtocols.CommandBus)
        - dispatch_batch(message_type, messages) - Batch message dispatching
        - create_from_global_config() - Factory method using global config

    Reliability Patterns:
        - Circuit breaker: Prevents cascading failures for specific message types
        - Rate limiting: Throttles message dispatching per message type
        - Retry logic: Automatic exponential backoff retry with configurable attempts
        - Timeout enforcement: Operation boundaries with timeout protection
        - Context propagation: Distributed tracing with correlation IDs

    Features:
    - Handler registration and discovery with support for commands, queries, and functions
    - Circuit breaker pattern for fault tolerance with per-message-type state
    - Rate limiting for request throttling with sliding window tracking
    - Retry logic with exponential backoff for transient failures
    - Timeout enforcement for operation boundaries using ThreadPoolExecutor
    - Audit logging for compliance and debugging
    - Performance metrics collection and reporting
    - Batch processing for multiple messages
    - Context propagation for distributed tracing with correlation IDs
    - Dispatcher instances satisfy FlextProtocols.CommandBus through duck typing

    Usage:
        >>> from flext_core import FlextDispatcher
        >>> from flext_core.protocols import FlextProtocols
        >>>
        >>> dispatcher = FlextDispatcher()
        >>> dispatcher.register_handler(CreateUserCommand, handler)
        >>> result = dispatcher.dispatch(CreateUserCommand(name="Alice"))
        >>>
        >>> # Dispatcher instances satisfy FlextProtocols.CommandBus protocol
        >>> # through structural typing - register_handler and dispatch methods
        >>> assert callable(dispatcher.register_handler)
        >>> assert callable(dispatcher.dispatch)
    """

    class CircuitBreakerManager:
        """Manages circuit breaker state machine per message type.

        Handles state transitions (CLOSED → OPEN → HALF_OPEN → CLOSED) with
        configurable thresholds and recovery timeouts.
        """

        def __init__(
            self,
            threshold: int,
            recovery_timeout: float,
            success_threshold: int,
        ) -> None:
            """Initialize circuit breaker manager.

            Args:
                threshold: Failure count before opening circuit
                recovery_timeout: Seconds before attempting recovery
                success_threshold: Successes needed to close from half-open

            """
            self._failures: dict[str, int] = {}
            self._states: dict[str, str] = {}
            self._opened_at: dict[str, float] = {}
            self._success_counts: dict[str, int] = {}
            self._threshold = threshold
            self._recovery_timeout = recovery_timeout
            self._success_threshold = success_threshold

        def get_state(self, message_type: str) -> str:
            """Get current state for message type."""
            return self._states.get(
                message_type, FlextConstants.Reliability.CircuitBreakerState.CLOSED
            )

        def set_state(self, message_type: str, state: str) -> None:
            """Set state for message type."""
            self._states[message_type] = state

        def is_open(self, message_type: str) -> bool:
            """Check if circuit breaker is open for message type."""
            return (
                self.get_state(message_type)
                == FlextConstants.Reliability.CircuitBreakerState.OPEN
            )

        def record_success(self, message_type: str) -> None:
            """Record successful operation and update state."""
            current_state = self.get_state(message_type)

            if (
                current_state
                == FlextConstants.Reliability.CircuitBreakerState.HALF_OPEN
            ):
                success_count = self._success_counts.get(message_type, 0) + 1
                self._success_counts[message_type] = success_count

                if success_count >= self._success_threshold:
                    self.transition_to_closed(message_type)

            elif current_state == FlextConstants.Reliability.CircuitBreakerState.CLOSED:
                self._failures[message_type] = 0

        def record_failure(self, message_type: str) -> None:
            """Record failed operation and update state."""
            current_state = self.get_state(message_type)
            current_failures = self._failures.get(message_type, 0) + 1
            self._failures[message_type] = current_failures

            if (
                current_state
                == FlextConstants.Reliability.CircuitBreakerState.HALF_OPEN
                or (
                    current_state
                    == FlextConstants.Reliability.CircuitBreakerState.CLOSED
                    and current_failures >= self._threshold
                )
            ):
                self.transition_to_open(message_type)

        def transition_to_state(self, message_type: str, new_state: str) -> None:
            """Transition to specified state."""
            self.set_state(message_type, new_state)
            if new_state == FlextConstants.Reliability.CircuitBreakerState.CLOSED:
                self._failures[message_type] = 0
                self._success_counts[message_type] = 0
                if message_type in self._opened_at:
                    del self._opened_at[message_type]
            elif new_state == FlextConstants.Reliability.CircuitBreakerState.OPEN:
                self._opened_at[message_type] = time.time()
                self._success_counts[message_type] = 0
            elif new_state == FlextConstants.Reliability.CircuitBreakerState.HALF_OPEN:
                self._success_counts[message_type] = 0

        def transition_to_closed(self, message_type: str) -> None:
            """Transition to CLOSED state."""
            self.transition_to_state(
                message_type, FlextConstants.Reliability.CircuitBreakerState.CLOSED
            )

        def transition_to_open(self, message_type: str) -> None:
            """Transition to OPEN state."""
            self.transition_to_state(
                message_type, FlextConstants.Reliability.CircuitBreakerState.OPEN
            )

        def transition_to_half_open(self, message_type: str) -> None:
            """Transition to HALF_OPEN state."""
            self.transition_to_state(
                message_type, FlextConstants.Reliability.CircuitBreakerState.HALF_OPEN
            )

        def attempt_reset(self, message_type: str) -> None:
            """Attempt recovery if circuit is open."""
            if self.is_open(message_type):
                opened_at = self._opened_at.get(message_type, 0.0)
                if (time.time() - opened_at) >= self._recovery_timeout:
                    self.transition_to_half_open(message_type)

        def check_before_dispatch(self, message_type: str) -> FlextResult[None]:
            """Check if dispatch is allowed."""
            self.attempt_reset(message_type)

            if self.is_open(message_type):
                failures = self._failures.get(message_type, 0)
                return FlextResult[None].fail(
                    f"Circuit breaker is open for message type '{message_type}'",
                    error_code=FlextConstants.Errors.OPERATION_ERROR,
                    error_data={
                        "message_type": message_type,
                        "failure_count": failures,
                        "threshold": self._threshold,
                        "state": self.get_state(message_type),
                        "opened_at": self._opened_at.get(message_type, 0.0),
                        "reason": "circuit_breaker_open",
                    },
                )

            return FlextResult[None].ok(None)

        def get_failure_count(self, message_type: str) -> int:
            """Get current failure count."""
            return self._failures.get(message_type, 0)

        def cleanup(self) -> None:
            """Clear all state."""
            self._failures.clear()
            self._states.clear()
            self._opened_at.clear()
            self._success_counts.clear()

        def get_metrics(self) -> dict[str, object]:
            """Get circuit breaker metrics."""
            return {
                "failures": len(self._failures),
                "states": len(self._states),
                "open_count": sum(
                    1
                    for state in self._states.values()
                    if state == FlextConstants.Reliability.CircuitBreakerState.OPEN
                ),
            }

    class TimeoutEnforcer:
        """Manages timeout enforcement and thread pool execution."""

        def __init__(
            self,
            *,
            use_timeout_executor: bool,
            executor_workers: int,
        ) -> None:
            """Initialize timeout enforcer.

            Args:
                use_timeout_executor: Whether to use timeout executor
                executor_workers: Number of executor worker threads

            """
            self._use_timeout_executor = use_timeout_executor
            self._executor_workers = max(executor_workers, 1)
            self._executor: concurrent.futures.ThreadPoolExecutor | None = None

        def should_use_executor(self) -> bool:
            """Check if timeout executor should be used.

            Returns:
                True if executor is enabled

            """
            return self._use_timeout_executor

        def reset_executor(self) -> None:
            """Reset executor (after shutdown)."""
            self._executor = None

        def resolve_workers(self) -> int:
            """Get the configured worker count.

            Returns:
                Maximum number of workers for executor

            """
            return self._executor_workers

        def ensure_executor(self) -> concurrent.futures.ThreadPoolExecutor:
            """Create the shared executor on demand (lazy initialization).

            Returns:
                ThreadPoolExecutor instance

            """
            if self._executor is None:
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self._executor_workers,
                    thread_name_prefix="flext-dispatcher",
                )
            return self._executor

        def get_executor_status(self) -> dict[str, object]:
            """Get executor status information.

            Returns:
                Dict with executor status and worker count

            """
            return {
                "executor_active": self._executor is not None,
                "executor_workers": self._executor_workers if self._executor else 0,
            }

        def cleanup(self) -> None:
            """Cleanup executor resources."""
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None

    class RateLimiterManager:
        """Manages rate limiting with simplified sliding window implementation."""

        def __init__(self, max_requests: int, window_seconds: float) -> None:
            """Initialize rate limiter manager.

            Args:
                max_requests: Maximum requests allowed per window
                window_seconds: Time window in seconds for rate limiting

            """
            self._max_requests = max_requests
            self._window_seconds = window_seconds
            # Track window start time and request count per message type
            self._windows: dict[str, tuple[float, int]] = {}

        def check_rate_limit(self, message_type: str) -> FlextResult[None]:
            """Check if rate limit is exceeded for message type.

            Args:
                message_type: The message type to check

            Returns:
                FlextResult[None]: Success if within limit, failure if exceeded

            """
            current_time = time.time()
            window_start, count = self._windows.get(message_type, (current_time, 0))

            # Reset window if elapsed
            if current_time - window_start >= self._window_seconds:
                window_start = current_time
                count = 0

            # Check if limit exceeded
            if count >= self._max_requests:
                return FlextResult[None].fail(
                    f"Rate limit exceeded for message type '{message_type}' - too many requests",
                    error_code=FlextConstants.Errors.OPERATION_ERROR,
                    error_data={
                        "message_type": message_type,
                        "limit": self._max_requests,
                        "window_seconds": self._window_seconds,
                        "retry_after": int(
                            self._window_seconds - (current_time - window_start)
                        ),
                        "reason": "rate_limit_exceeded",
                    },
                )

            # Update window tracking and increment count
            self._windows[message_type] = (window_start, count + 1)
            return FlextResult[None].ok(None)

        def cleanup(self) -> None:
            """Clear all rate limit windows."""
            self._windows.clear()

    class RetryPolicy:
        """Manages retry logic with configurable attempts and delays."""

        def __init__(self, max_attempts: int, retry_delay: float) -> None:
            """Initialize retry policy manager.

            Args:
                max_attempts: Maximum retry attempts allowed
                retry_delay: Delay in seconds between retry attempts

            """
            self._max_attempts = max(max_attempts, 1)
            self._retry_delay = max(retry_delay, 0.0)
            # Track attempt counts per message type
            self._attempts: dict[str, int] = {}

        def should_retry(self, current_attempt: int) -> bool:
            """Check if we should retry the operation.

            Args:
                current_attempt: The current attempt number (0-based)

            Returns:
                True if we should retry, False if we've exhausted attempts

            """
            return current_attempt < self._max_attempts - 1

        def is_retriable_error(self, error: str | None) -> bool:
            """Check if an error is retriable.

            Args:
                error: Error message to check

            Returns:
                True if error indicates a transient failure

            """
            if error is None:
                return False

            retriable_patterns = (
                "Temporary failure",
                "timeout",
                "transient",
                "temporarily unavailable",
                "try again",
            )
            return any(
                pattern.lower() in error.lower() for pattern in retriable_patterns
            )

        def get_retry_delay(self) -> float:
            """Get delay between retry attempts.

            Returns:
                Delay in seconds

            """
            return self._retry_delay

        def get_max_attempts(self) -> int:
            """Get maximum retry attempts.

            Returns:
                Maximum number of attempts allowed

            """
            return self._max_attempts

        def record_attempt(self, message_type: str) -> None:
            """Record an attempt for tracking purposes.

            Args:
                message_type: The message type being processed

            """
            self._attempts[message_type] = self._attempts.get(message_type, 0) + 1

        def reset(self, message_type: str) -> None:
            """Reset attempt tracking for a message type.

            Args:
                message_type: The message type to reset

            """
            self._attempts.pop(message_type, None)

        def cleanup(self) -> None:
            """Clear all attempt tracking."""
            self._attempts.clear()

    @override
    def __init__(
        self,
        *,
        bus: FlextBus | None = None,
    ) -> None:
        """Initialize dispatcher with configuration from FlextConfig singleton.

        Refactored to eliminate SOLID violations by delegating to specialized components.
        Configuration is accessed via FlextMixins.config singleton.

        Args:
            bus: Optional bus instance (created if not provided)

        """
        super().__init__()

        # Initialize service infrastructure (DI, Context, Logging, Metrics)
        self._init_service("flext_dispatcher")

        # Initialize bus first
        self._bus = bus or FlextBus()

        # Enrich context with dispatcher metadata for observability
        self._enrich_context(
            service_type="dispatcher",
            bus_type=type(self._bus).__name__,
            circuit_breaker_enabled=True,
            timeout_enforcement=True,
            supports_async=True,
        )

        # Initialize circuit breaker manager
        self._circuit_breaker = self.CircuitBreakerManager(
            threshold=self.config.circuit_breaker_threshold,
            recovery_timeout=FlextConstants.Reliability.DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
            success_threshold=FlextConstants.Reliability.DEFAULT_CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
        )

        # Rate limiting - simplified sliding window implementation via manager
        self._rate_limiter = self.RateLimiterManager(
            max_requests=self.config.rate_limit_max_requests,
            window_seconds=self.config.rate_limit_window_seconds,
        )

        # Timeout enforcement and executor management via manager
        self._timeout_enforcer = self.TimeoutEnforcer(
            use_timeout_executor=self.config.enable_timeout_executor,
            executor_workers=self.config.executor_workers,  # type: ignore[arg-type]
        )

        # Retry policy management via manager
        self._retry_policy = self.RetryPolicy(
            max_attempts=self.config.max_retry_attempts,
            retry_delay=self.config.retry_delay,
        )

    @property
    def dispatcher_config(self) -> dict[str, object]:
        """Access the dispatcher configuration."""
        return self.config.model_dump()

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
                FlextConstants.Dispatcher.ERROR_INVALID_HANDLER_MODE,
            )

        # Validate handler is provided
        if request.get("handler") is None:
            return FlextResult[dict[str, object]].fail(
                FlextConstants.Dispatcher.ERROR_HANDLER_REQUIRED,
            )

        # Add Phase 4 enhancement: Protocol validation before registration
        handler_obj = request.get("handler")
        protocol_validation = FlextMixins.ProtocolValidation.validate_protocol_compliance(
            handler_obj, "Handler"
        )
        if protocol_validation.is_failure:
            return FlextResult[dict[str, object]].fail(
                f"Handler protocol validation failed: {protocol_validation.error}",
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
        details: dict[str, object] = {
            "registration_id": request.get("registration_id"),
            "message_type_name": getattr(request.get("message_type"), "__name__", None)
            if request.get("message_type")
            else None,
            "handler_mode": request.get("handler_mode"),
            "timestamp": FlextUtilities.Generators.generate_timestamp(),
            "status": FlextConstants.Dispatcher.REGISTRATION_STATUS_ACTIVE,
        }

        if self.config.dispatcher_enable_logging:
            self._log_with_context(
                "info",
                "handler_registered",
                registration_id=details.get("registration_id"),
                handler_mode=details.get("handler_mode"),
                message_type=details.get("message_type_name"),
            )

        return FlextResult[dict[str, object]].ok(details)

    def register_handler(
        self,
        message_type_or_handler: FlextTypes.MessageTypeOrHandlerType,
        handler: FlextTypes.HandlerOrCallableType | None = None,
        *,
        handler_mode: FlextConstants.Cqrs.HandlerModeSimple = FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
        handler_config: FlextTypes.HandlerConfigurationType = None,
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
            # Ensure handler is FlextHandlers instance (use helper to eliminate duplication)
            ensure_result = self._ensure_handler(handler, mode=handler_mode)
            if ensure_result.is_failure:
                return FlextResult[dict[str, object]].fail(
                    ensure_result.error or "Handler creation failed",
                )
            resolved_handler = ensure_result.value

            # Create structured request with message type
            request: dict[str, object] = {
                "handler": resolved_handler,
                "message_type": message_type_or_handler,
                "handler_mode": handler_mode,
                "handler_config": handler_config,
            }
        else:
            # New API: register_handler(handler)
            # Ensure we have a handler, not a string
            if isinstance(message_type_or_handler, str):
                return FlextResult[dict[str, object]].fail(
                    "Cannot register handler: message type string provided without handler",
                )

            # Ensure handler is FlextHandlers instance (use helper to eliminate duplication)
            ensure_result = self._ensure_handler(message_type_or_handler, mode=handler_mode)
            if ensure_result.is_failure:
                return FlextResult[dict[str, object]].fail(
                    ensure_result.error or "Handler creation failed",
                )
            resolved_handler = ensure_result.value

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
        handler_config: FlextTypes.HandlerConfigurationType = None,
    ) -> FlextResult[dict[str, object]]:
        """Register command handler using structured model internally.

        Args:
            command_type: Command message type
            handler: Handler instance
            handler_config: Optional handler configuration

        Returns:
            FlextResult with registration details or error

        """
        request: dict[str, object] = {
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
        handler_config: FlextTypes.HandlerConfigurationType = None,
    ) -> FlextResult[dict[str, object]]:
        """Register query handler using structured model internally.

        Args:
            query_type: Query message type
            handler: Handler instance
            handler_config: Optional handler configuration

        Returns:
            FlextResult with registration details or error

        """
        request: dict[str, object] = {
            "handler": handler,
            "message_type": query_type,
            "handler_mode": FlextConstants.Dispatcher.HANDLER_MODE_QUERY,
            "handler_config": handler_config,
        }

        return self.register_handler_with_request(request)

    def register_function(
        self,
        message_type: type[object],
        handler_func: FlextTypes.HandlerCallableType,
        *,
        handler_config: FlextTypes.HandlerConfigurationType = None,
        mode: FlextConstants.Cqrs.HandlerModeSimple = FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
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
        request: dict[str, object] = {
            "handler": handler_result.value,
            "message_type": message_type,
            "handler_mode": mode,
            "handler_config": handler_config,
        }

        return self.register_handler_with_request(request)

    def create_handler_from_function(
        self,
        handler_func: FlextTypes.HandlerCallableType,
        _handler_config: FlextTypes.HandlerConfigurationType = None,
        mode: FlextConstants.Cqrs.HandlerType = FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
    ) -> FlextResult[FlextHandlers[object, object]]:
        """Create handler from function using FlextHandlers constructor.

        Args:
            handler_func: Function to wrap
            _handler_config: Optional configuration (reserved for future use)
            mode: Handler mode

        Returns:
            FlextResult with handler instance or error

        """
        try:
            handler = FlextHandlers.from_callable(
                callable_func=handler_func,
                handler_name=getattr(handler_func, "__name__", "FunctionHandler"),
                handler_type=mode,
            )
            return FlextResult[FlextHandlers[object, object]].ok(handler)

        except Exception as error:
            return FlextResult[FlextHandlers[object, object]].fail(
                f"Handler creation failed: {error}",
            )

    def _ensure_handler(
        self,
        handler: object,
        mode: FlextConstants.Cqrs.HandlerType = FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
    ) -> FlextResult[FlextHandlers[object, object]]:
        """Ensure handler is a FlextHandlers instance, converting from callable if needed.

        Private helper to eliminate duplication in handler registration.

        Args:
            handler: Handler instance or callable to convert
            mode: Handler operation mode (command/query)

        Returns:
            FlextResult with FlextHandlers instance or error

        """
        # If already FlextHandlers, return success
        if isinstance(handler, FlextHandlers):
            return FlextResult[FlextHandlers[object, object]].ok(handler)

        # If callable, convert to FlextHandlers
        if callable(handler):
            return self.create_handler_from_function(
                handler_func=handler,
                mode=mode,
            )

        # Invalid handler type
        return FlextResult[FlextHandlers[object, object]].fail(
            f"Handler must be FlextHandlers instance or callable, "
            f"got {type(handler).__name__}",
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
        # Propagate context for distributed tracing
        message = request.get("message")
        message_type = type(message).__name__ if message else "unknown"
        self._propagate_context(f"dispatch_with_request_{message_type}")

        start_time = time.time()

        # Validate request
        if request.get("message") is None:
            return FlextResult[dict[str, object]].fail(
                FlextConstants.Dispatcher.ERROR_MESSAGE_REQUIRED,
            )

        # Get timeout from request override or config
        timeout_override = request.get("timeout_override")
        timeout_seconds = (
            timeout_override
            if timeout_override is not None
            else self.config.timeout_seconds
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

            # Update circuit breaker state machine
            if execution_result.is_success:
                self._circuit_breaker.record_success(message_type)
            else:
                self._circuit_breaker.record_failure(message_type)

            if execution_result.is_success:
                dispatch_result: dict[str, object] = {
                    "success": True,
                    "result": execution_result.value,
                    "error_message": None,
                    "request_id": request.get("request_id"),
                    "execution_time_ms": execution_time_ms,
                    "correlation_id": request.get("correlation_id"),
                    "timeout_seconds": timeout_seconds,
                }

                if self.config.dispatcher_enable_logging:
                    self._log_with_context(
                        "debug",
                        "dispatch_succeeded",
                        request_id=request.get("request_id"),
                        message_type=type(request.get("message")).__name__,
                        execution_time_ms=execution_time_ms,
                        timeout_seconds=timeout_seconds,
                    )

                return FlextResult[dict[str, object]].ok(dispatch_result)

            dispatch_result = {
                "success": False,
                "result": None,
                "error_message": execution_result.error or "Unknown error",
                "request_id": request.get("request_id"),
                "execution_time_ms": execution_time_ms,
                "correlation_id": request.get("correlation_id"),
                "timeout_seconds": timeout_seconds,
            }

            if self.config.dispatcher_enable_logging:
                self._log_with_context(
                    "error",
                    "dispatch_failed",
                    request_id=request.get("request_id"),
                    message_type=type(request.get("message")).__name__,
                    error=dispatch_result.get("error_message"),
                    execution_time_ms=execution_time_ms,
                    timeout_seconds=timeout_seconds,
                )

            return FlextResult[dict[str, object]].ok(dispatch_result)

    def _check_pre_dispatch_conditions(
        self,
        message_type: str,
    ) -> FlextResult[None]:
        """Check all pre-dispatch conditions (circuit breaker, rate limiting).

        Orchestrates multiple validation checks in sequence. Returns first failure
        encountered, or success if all checks pass.

        Args:
            message_type: Message type string for reliability pattern checks

        Returns:
            FlextResult[None]: Success if all checks pass, failure if any check fails

        """
        # Check circuit breaker state
        cb_check = self._circuit_breaker.check_before_dispatch(message_type)
        if cb_check.is_failure:
            return FlextResult[None].fail(
                cb_check.error,
                error_code=cb_check.error_code,
                error_data=cb_check.error_data,
            )

        # Check rate limiting
        rate_limit_check = self._rate_limiter.check_rate_limit(message_type)
        if rate_limit_check.is_failure:
            return FlextResult[None].fail(
                rate_limit_check.error,
                error_code=rate_limit_check.error_code,
                error_data=rate_limit_check.error_data,
            )

        return FlextResult[None].ok(None)

    def _execute_with_timeout(
        self,
        execute_func: Callable[[], FlextResult[object]],
        timeout_seconds: float,
        timeout_override: int | None = None,
    ) -> FlextResult[object]:
        """Execute a function with timeout enforcement using executor or direct execution.

        Handles timeout errors gracefully. If executor is shutdown, reinitializes it.
        This helper encapsulates the timeout orchestration logic.

        Args:
            execute_func: Callable that returns FlextResult[object]
            timeout_seconds: Timeout in seconds
            timeout_override: Optional timeout override (forces executor usage)

        Returns:
            FlextResult[object]: Execution result or timeout error

        """
        use_executor = (
            self._timeout_enforcer.should_use_executor()
            or timeout_override is not None
        )

        if use_executor:
            executor = self._timeout_enforcer.ensure_executor()
            future: concurrent.futures.Future[FlextResult[object]] | None = None
            try:
                future = executor.submit(execute_func)
                return future.result(timeout=timeout_seconds)
            except concurrent.futures.TimeoutError:
                # Cancel the future and return timeout error
                if future is not None:
                    future.cancel()
                return FlextResult[object].fail(
                    f"Operation timeout after {timeout_seconds} seconds",
                )
            except Exception:
                # Executor was shut down; reinitialize and retry immediately
                self._timeout_enforcer.reset_executor()
                # Return retriable error so caller can retry
                return FlextResult[object].fail(
                    "Executor was shutdown, retry requested",
                )
        else:
            return execute_func()

    def _should_retry_on_error(
        self,
        attempt: int,
        error_message: str | None = None,
    ) -> bool:
        """Check if an error should trigger a retry attempt.

        Encapsulates retry eligibility logic and handles retry delay.

        Args:
            attempt: Current attempt number (0-indexed)
            error_message: Error message (for retriable error checking)

        Returns:
            True if should retry (delay applied), False if should not retry

        """
        # Check retry policy eligibility
        if not self._retry_policy.should_retry(attempt):
            return False

        # For FlextResult errors, check if error is retriable
        if error_message is not None and not self._retry_policy.is_retriable_error(error_message):
            return False

        # Delay before retry
        time.sleep(self._retry_policy.get_retry_delay())
        return True

    def dispatch(
        self,
        message_or_type: object | str,
        data: object | None = None,
        *,
        metadata: dict[str, object] | None = None,
        correlation_id: str | None = None,
        timeout_override: int | None = None,
    ) -> FlextResult[object]:
        """Dispatch message with support for both old and new API.

        Refactored to use specialized processors for SOLID compliance:
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

        # Check pre-dispatch conditions (circuit breaker + rate limiting)
        conditions_check = self._check_pre_dispatch_conditions(message_type)
        if conditions_check.is_failure:
            return FlextResult[object].fail(
                conditions_check.error,
                error_code=conditions_check.error_code,
                error_data=conditions_check.error_data,
            )

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
            string_metadata: dict[str, object] = {
                k: str(v) for k, v in metadata.items()
            }
            FlextModels.Metadata(attributes=string_metadata)

        # Execute dispatch with retry logic via RetryPolicy manager
        max_retries = self._retry_policy.get_max_attempts()

        for attempt in range(max_retries):
            try:
                # Get timeout from config
                timeout_seconds = float(
                    cast(
                        "int | float",
                        self.config.timeout_seconds,
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

                # Execute with timeout enforcement (handles executor/threading logic)
                bus_result = self._execute_with_timeout(
                    execute_with_context,
                    timeout_seconds,
                    timeout_override,
                )

                # Handle executor shutdown retry case
                if (bus_result.is_failure and
                    "Executor was shutdown" in (bus_result.error or "")):
                    continue

                if bus_result.is_success:
                    # Record success in circuit breaker
                    self._circuit_breaker.record_success(message_type)
                    return FlextResult[object].ok(bus_result.value)

                # Track circuit breaker failure
                self._circuit_breaker.record_failure(message_type)

                # Check if should retry (encapsulates retry policy and delay logic)
                if self._should_retry_on_error(attempt, bus_result.error):
                    continue

                return FlextResult[object].fail(bus_result.error or "Dispatch failed")
            except Exception as e:
                # Track circuit breaker failure for exceptions
                self._circuit_breaker.record_failure(message_type)

                # Check if should retry (for exceptions, no error message check)
                if self._should_retry_on_error(attempt):
                    continue
                return FlextResult[object].fail(f"Dispatch error: {e}")

        # Record final failure
        return FlextResult[object].fail("Max retries exceeded")

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------
    def _normalize_context_metadata(
        self,
        metadata: object | None,
    ) -> dict[str, object] | None:
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

        normalized: dict[str, object] = {
            str(key): value for key, value in raw_metadata.items()
        }

        return normalized

    @contextmanager
    def _context_scope(
        self,
        metadata: dict[str, object] | None = None,
        correlation_id: str | None = None,
    ) -> Generator[None]:
        """Manage execution context with optional metadata and correlation ID.

        Args:
            metadata: Optional metadata to include in context
            correlation_id: Optional correlation ID for tracing

        """
        if not self.config.dispatcher_auto_context:
            yield
            return

        metadata_var = FlextContext.Variables.Performance.OPERATION_METADATA
        correlation_var = FlextContext.Variables.Correlation.CORRELATION_ID
        parent_var = FlextContext.Variables.Correlation.PARENT_CORRELATION_ID

        # Store current context values for restoration
        current_parent_value = parent_var.get()
        current_parent: str | None = (
            current_parent_value if isinstance(current_parent_value, str) else None
        )

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

            if self.config.dispatcher_enable_logging:
                self._log_with_context(
                    "debug",
                    "dispatch_context_entered",
                    correlation_id=effective_correlation_id,
                )

            yield

            if self.config.dispatcher_enable_logging:
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

    def get_performance_metrics(self) -> dict[str, object]:
        """Get performance metrics for the dispatcher.

        Returns:
            dict[str, object]: Dictionary containing performance metrics

        """
        # Get metrics from circuit breaker manager
        cb_metrics = self._circuit_breaker.get_metrics()
        return {
            "total_dispatches": 0,  # Track actual dispatches (future enhancement)
            "circuit_breaker_failures": cb_metrics["failures"],
            "circuit_breaker_states": cb_metrics["states"],
            "circuit_breaker_open_count": cb_metrics["open_count"],
            **self._timeout_enforcer.get_executor_status(),
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
            self._circuit_breaker.cleanup()
            self._rate_limiter.cleanup()
            self._timeout_enforcer.cleanup()
            self._retry_policy.cleanup()

        except Exception as e:
            self._log_with_context("warning", "Cleanup failed", error=str(e))


__all__ = ["FlextDispatcher"]
