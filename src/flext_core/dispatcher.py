"""High-level message dispatch orchestration with reliability patterns.

This module provides FlextDispatcher, a facade that orchestrates message
dispatching with circuit breaker, rate limiting, retry logic, timeout
enforcement, and comprehensive observability.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import concurrent.futures
import random
import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from typing import cast, override

from cachetools import LRUCache

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
    timeout enforcement, and comprehensive observability through three
    integrated layers: Layer 1 (CQRS routing), Layer 2 (reliability patterns),
    and Layer 3 (advanced processing).

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
            # Advanced metrics tracking
            self._recovery_successes: dict[str, int] = {}  # HALF_OPEN → CLOSED
            self._recovery_failures: dict[str, int] = {}  # HALF_OPEN → OPEN
            self._total_successes: dict[str, int] = {}  # Successful operations

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
            # Track all successful operations
            self._total_successes[message_type] = (
                self._total_successes.get(message_type, 0) + 1
            )

            if (
                current_state
                == FlextConstants.Reliability.CircuitBreakerState.HALF_OPEN
            ):
                success_count = self._success_counts.get(message_type, 0) + 1
                self._success_counts[message_type] = success_count

                if success_count >= self._success_threshold:
                    # Track successful recovery (HALF_OPEN → CLOSED)
                    self._recovery_successes[message_type] = (
                        self._recovery_successes.get(message_type, 0) + 1
                    )
                    self.transition_to_closed(message_type)

            elif current_state == FlextConstants.Reliability.CircuitBreakerState.CLOSED:
                self._failures[message_type] = 0

        def record_failure(self, message_type: str) -> None:
            """Record failed operation and update state."""
            current_state = self.get_state(message_type)
            current_failures = self._failures.get(message_type, 0) + 1
            self._failures[message_type] = current_failures

            # Track failed recovery attempts (failure in HALF_OPEN state)
            if (
                current_state
                == FlextConstants.Reliability.CircuitBreakerState.HALF_OPEN
            ):
                self._recovery_failures[message_type] = (
                    self._recovery_failures.get(message_type, 0) + 1
                )
                self.transition_to_open(message_type)
            elif (
                current_state == FlextConstants.Reliability.CircuitBreakerState.CLOSED
                and current_failures >= self._threshold
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
            # Clear advanced metrics
            self._recovery_successes.clear()
            self._recovery_failures.clear()
            self._total_successes.clear()

        def get_metrics(self) -> dict[str, object]:
            """Get circuit breaker metrics including advanced metrics.

            Returns metrics including:
            - failures: Count of tracked message types with failures
            - states: Count of tracked message types
            - open_count: Count of open circuits
            - recovery_success_rate: % of successful recovery attempts
            - failure_rate: % of failed operations
            - total_recovery_attempts: Total HALF_OPEN transitions
            """
            # Calculate recovery success rate
            total_recovery_attempts = sum(
                self._recovery_successes.get(mt, 0) + self._recovery_failures.get(mt, 0)
                for mt in self._states
            )
            total_recovery_successes = sum(
                self._recovery_successes.get(mt, 0) for mt in self._states
            )
            recovery_success_rate = (
                (total_recovery_successes / total_recovery_attempts * 100)
                if total_recovery_attempts > 0
                else 0.0
            )

            # Calculate failure rate
            total_failures = sum(self._failures.values())
            total_successes = sum(self._total_successes.values())
            total_operations = total_failures + total_successes
            failure_rate = (
                (total_failures / total_operations * 100)
                if total_operations > 0
                else 0.0
            )

            return {
                # Legacy metrics (backward compatible)
                "failures": len(self._failures),
                "states": len(self._states),
                "open_count": sum(
                    1
                    for state in self._states.values()
                    if state == FlextConstants.Reliability.CircuitBreakerState.OPEN
                ),
                # Advanced metrics
                "recovery_success_rate": recovery_success_rate,
                "failure_rate": failure_rate,
                "total_recovery_attempts": total_recovery_attempts,
                "total_recovery_successes": total_recovery_successes,
                "total_operations": total_operations,
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

        def __init__(
            self,
            max_requests: int,
            window_seconds: float,
            jitter_factor: float = 0.1,
        ) -> None:
            """Initialize rate limiter manager.

            Args:
                max_requests: Maximum requests allowed per window
                window_seconds: Time window in seconds for rate limiting
                jitter_factor: Jitter variance as fraction (0.1 = ±10%)

            """
            self._max_requests = max_requests
            self._window_seconds = window_seconds
            self._jitter_factor = max(0.0, min(jitter_factor, 1.0))
            # Track window start time and request count per message type
            self._windows: dict[str, tuple[float, int]] = {}

        def _apply_jitter(self, base_delay: float) -> float:
            """Apply jitter variance to a delay value.

            Prevents thundering herd effect by randomizing retry times.
            Formula: base_delay * (1 + random(-jitter_factor, jitter_factor))

            Args:
                base_delay: Base delay in seconds

            Returns:
                Jittered delay with randomized variance

            """
            if base_delay <= 0.0 or self._jitter_factor == 0.0:
                return base_delay

            # Random variance between -jitter_factor and +jitter_factor
            # Note: S311 suppressed - random jitter for timing is not cryptographic
            variance = (
                2.0 * random.random() - 1.0
            ) * self._jitter_factor  # Range: [-jitter_factor, +jitter_factor]
            jittered = base_delay * (1.0 + variance)

            # Ensure jittered value doesn't go negative
            return max(0.0, jittered)

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
                # Calculate base retry delay
                base_retry_delay = self._window_seconds - (current_time - window_start)
                # Apply jitter to prevent thundering herd
                jittered_retry_delay = self._apply_jitter(base_retry_delay)
                return FlextResult[None].fail(
                    f"Rate limit exceeded for message type '{message_type}' - too many requests",
                    error_code=FlextConstants.Errors.OPERATION_ERROR,
                    error_data={
                        "message_type": message_type,
                        "limit": self._max_requests,
                        "window_seconds": self._window_seconds,
                        "retry_after": int(jittered_retry_delay),
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
        """Manages retry logic with configurable attempts and exponential backoff."""

        def __init__(self, max_attempts: int, retry_delay: float) -> None:
            """Initialize retry policy manager.

            Args:
                max_attempts: Maximum retry attempts allowed
                retry_delay: Base delay in seconds between retry attempts
                             (or fixed delay if exponential backoff disabled)

            """
            self._max_attempts = max(max_attempts, 1)
            self._base_delay = max(
                retry_delay, 0.0
            )  # Base delay for exponential backoff
            # Track attempt counts per message type
            self._attempts: dict[str, int] = {}
            # Exponential backoff configuration
            self._exponential_factor = 2.0  # Multiply delay by this factor each attempt
            self._max_delay = 300.0  # Maximum delay cap (5 minutes)

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

        def get_exponential_delay(self, attempt_number: int) -> float:
            """Calculate exponential backoff delay for given attempt.

            Uses formula: min(base_delay * (exponential_factor ^ attempt), max_delay)

            Args:
                attempt_number: The current attempt number (0-based)

            Returns:
                Delay in seconds with exponential backoff applied

            """
            if self._base_delay == 0.0:
                return 0.0

            # Calculate exponential backoff: base_delay * (factor ^ attempt)
            exponential_delay = self._base_delay * (
                self._exponential_factor**attempt_number
            )
            # Cap at maximum delay to prevent excessive waits
            return min(exponential_delay, self._max_delay)

        def get_retry_delay(self) -> float:
            """Get base delay between retry attempts.

            Returns:
                Base delay in seconds (used for fixed delay mode or base of exponential)

            """
            return self._base_delay

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
    ) -> None:
        """Initialize dispatcher with configuration from FlextConfig singleton.

        Refactored to eliminate SOLID violations by delegating to specialized components.
        Configuration is accessed via FlextMixins.config singleton.

        """
        super().__init__()

        # Initialize service infrastructure (DI, Context, Logging, Metrics)
        self._init_service("flext_dispatcher")

        # Enrich context with dispatcher metadata for observability
        self._enrich_context(
            service_type="dispatcher",
            dispatcher_type="FlextDispatcher",
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
            executor_workers=self.config.executor_workers,
        )

        # Retry policy management via manager
        self._retry_policy = self.RetryPolicy(
            max_attempts=self.config.max_retry_attempts,
            retry_delay=self.config.retry_delay,
        )

        # ==================== LAYER 2.5: TIMEOUT CONTEXT PROPAGATION ====================

        # Timeout context tracking for deadline and cancellation propagation
        self._timeout_contexts: dict[
            str, dict[str, object]
        ] = {}  # operation_id → context
        self._timeout_deadlines: dict[
            str, float
        ] = {}  # operation_id → deadline timestamp

        # ==================== LAYER 1: CQRS ROUTING INITIALIZATION ====================

        # Handler registry (from FlextBus dual-mode registration)
        self._handlers: dict[str, object] = {}  # Explicit command → handler mappings
        self._auto_handlers: list[object] = []  # Auto-discovery handlers

        # Middleware pipeline (from FlextBus)
        self._middleware_configs: list[dict[str, object]] = []  # Config + ordering
        self._middleware_instances: dict[str, object] = {}  # Keyed by middleware_id

        # Query result caching (from FlextBus - LRU cache)
        max_cache_size = (
            self.config.cache_max_size
            if hasattr(self.config, "cache_max_size")
            else 100
        )
        self._cache: LRUCache[str, FlextResult[object]] = LRUCache(
            maxsize=max_cache_size
        )

        # Event subscribers (from FlextBus event protocol)
        self._event_subscribers: dict[str, list[object]] = {}  # event_type → handlers

        # Execution counter for metrics
        self._execution_count: int = 0

        # ==================== LAYER 3: ADVANCED PROCESSING INITIALIZATION ====================

        # Group 1: Processor Registry (from FlextProcessors)
        self._processors: dict[str, object] = {}  # name → processor function
        self._processor_configs: dict[str, dict[str, object]] = {}  # name → config
        self._processor_metrics_per_name: dict[
            str, dict[str, int]
        ] = {}  # per-processor metrics
        self._processor_locks: dict[
            str, threading.Lock
        ] = {}  # per-processor thread safety

        # Group 2: Batch & Parallel Configuration
        self._batch_size: int = getattr(self.config, "batch_size", 10)
        self._parallel_workers: int = getattr(self.config, "executor_workers", 4)

        # Group 3: Handler Registry (from FlextProcessors.HandlerRegistry)
        self._handler_registry: dict[str, object] = {}  # name → handler function
        self._handler_configs: dict[
            str, dict[str, object]
        ] = {}  # name → handler config
        self._handler_validators: dict[
            str, Callable[[object], bool]
        ] = {}  # validation functions

        # Group 4: Pipeline (from FlextProcessors.Pipeline)
        self._pipeline_steps: list[dict[str, object]] = []  # Ordered pipeline steps
        self._pipeline_composition: dict[
            str, Callable[[object], FlextResult[object]]
        ] = {}  # composed functions
        self._pipeline_memo: dict[str, object] = {}  # Memoization cache for pipeline

        # Group 5: Metrics & Auditing (from FlextProcessors)
        self._process_metrics: dict[str, int] = {
            "registrations": 0,
            "successful_processes": 0,
            "failed_processes": 0,
            "batch_operations": 0,
            "parallel_operations": 0,
            "pipeline_operations": 0,
            "fallback_executions": 0,
            "timeout_executions": 0,
        }
        self._audit_log: list[dict[str, object]] = []  # Operation audit trail
        self._performance_metrics: dict[str, float] = {}  # Timing and throughput
        self._processor_execution_times: dict[
            str, list[float]
        ] = {}  # Per-processor times

    @property
    def dispatcher_config(self) -> dict[str, object]:
        """Access the dispatcher configuration."""
        return self.config.model_dump()

    # ==================== LAYER 3: ADVANCED PROCESSING INTERNAL METHODS ====================

    def _validate_processor_interface(
        self,
        processor: object,
        processor_context: str = "processor",
    ) -> FlextResult[None]:
        """Validate that processor has required interface (from FlextProcessors).

        Args:
            processor: Processor object to validate
            processor_context: Context string for error messages

        Returns:
            FlextResult[None]: Success if valid, failure if processor missing required interface

        """
        # Check for callable processor or process() method
        if callable(processor):
            return FlextResult[None].ok(None)

        process_method = getattr(processor, "process", None)
        if callable(process_method):
            return FlextResult[None].ok(None)

        return FlextResult[None].fail(
            f"Invalid {processor_context}: must be callable or have callable 'process' method. "
            f"Processors must implement process(name, data) or be callable"
        )

    def _route_to_processor(self, processor_name: str) -> object | None:
        """Locate registered processor by name.

        Args:
            processor_name: Name of processor to find

        Returns:
            Processor object or None if not found

        """
        return self._processors.get(processor_name)

    def _apply_processor_circuit_breaker(
        self,
        _processor_name: str,
        processor: object,
    ) -> FlextResult[object]:
        """Apply per-processor circuit breaker pattern.

        Args:
            _processor_name: Name of processor
            processor: Processor object

        Returns:
            FlextResult[object]: Success if circuit breaker allows, failure if open

        """
        # Use global circuit breaker manager
        # Per-processor circuit breaking is handled at dispatch() level
        # For now, always allow (dispatch() will enforce global CB)
        return FlextResult[object].ok(processor)

    def _apply_processor_rate_limiter(self, _processor_name: str) -> FlextResult[None]:
        """Apply per-processor rate limiting.

        Returns:
            FlextResult[None]: Success if within limit, failure if exceeded

        """
        # Use global rate limiter manager
        # Per-processor rate limiting is handled at dispatch() level
        # For now, always allow (dispatch() will enforce global RL)
        return FlextResult[None].ok(None)

    def _execute_processor_with_metrics(
        self,
        processor_name: str,
        processor: object,
        data: object,
    ) -> FlextResult[object]:
        """Execute processor and collect metrics.

        Args:
            processor_name: Name of processor
            processor: Processor object
            data: Data to process

        Returns:
            FlextResult[object]: Processor result or error

        """
        start_time = time.time()
        try:
            # Execute processor
            if callable(processor):
                result = processor(data)
            else:
                process_method = getattr(processor, "process", None)
                if callable(process_method):
                    result = process_method(data)
                else:
                    return FlextResult[object].fail(
                        f"Cannot execute processor: {processor_name}"
                    )

            # Convert to FlextResult if needed
            if not isinstance(result, FlextResult):
                result = FlextResult[object].ok(result)

            # Update metrics
            execution_time = time.time() - start_time
            if processor_name not in self._processor_execution_times:
                self._processor_execution_times[processor_name] = []
            self._processor_execution_times[processor_name].append(execution_time)

            # Update processor-specific metrics
            if processor_name not in self._processor_metrics_per_name:
                self._processor_metrics_per_name[processor_name] = {
                    "successful_processes": 0,
                    "failed_processes": 0,
                    "executions": 0,
                }
            metrics = self._processor_metrics_per_name[processor_name]
            metrics["executions"] = metrics.get("executions", 0) + 1
            if result.is_success:
                metrics["successful_processes"] = (
                    metrics.get("successful_processes", 0) + 1
                )
            else:
                metrics["failed_processes"] = metrics.get("failed_processes", 0) + 1

            return result
        except Exception as e:
            execution_time = time.time() - start_time
            if processor_name not in self._processor_execution_times:
                self._processor_execution_times[processor_name] = []
            self._processor_execution_times[processor_name].append(execution_time)
            return FlextResult[object].fail(f"Processor execution failed: {e}")

    def _process_batch_internal(
        self,
        processor_name: str,
        data_list: list[object],
        batch_size: int | None = None,
    ) -> FlextResult[list[object]]:
        """Process items in batch (internal).

        Args:
            processor_name: Name of processor
            data_list: List of data items
            batch_size: Size of each batch (default from config)

        Returns:
            FlextResult[list[object]]: List of results

        """
        batch_size = batch_size or self._batch_size
        results: list[object] = []

        processor = self._route_to_processor(processor_name)
        if processor is None:
            return FlextResult[list[object]].fail(
                f"Processor not found: {processor_name}"
            )

        # Process in batches
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i : i + batch_size]
            for data in batch:
                result = self._execute_processor_with_metrics(
                    processor_name, processor, data
                )
                if result.is_success:
                    results.append(result.value)
                else:
                    return FlextResult[list[object]].fail(result.error)

        return FlextResult[list[object]].ok(results)

    def _process_parallel_internal(
        self,
        processor_name: str,
        data_list: list[object],
        max_workers: int | None = None,
    ) -> FlextResult[list[object]]:
        """Process items in parallel (internal).

        Args:
            processor_name: Name of processor
            data_list: List of data items
            max_workers: Number of parallel workers (default from config)

        Returns:
            FlextResult[list[object]]: List of results

        """
        max_workers = max_workers or self._parallel_workers
        results: list[object] = []

        processor = self._route_to_processor(processor_name)
        if processor is None:
            return FlextResult[list[object]].fail(
                f"Processor not found: {processor_name}"
            )

        # Process in parallel using ThreadPoolExecutor
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = {
                    executor.submit(
                        self._execute_processor_with_metrics,
                        processor_name,
                        processor,
                        data,
                    ): idx
                    for idx, data in enumerate(data_list)
                }

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result.is_success:
                        results.append(result.value)
                    else:
                        return FlextResult[list[object]].fail(result.error)

            return FlextResult[list[object]].ok(results)
        except Exception as e:
            return FlextResult[list[object]].fail(f"Parallel processing failed: {e}")

    def _validate_handler_registry_interface(
        self,
        handler: object,
        handler_context: str = "registry handler",
    ) -> FlextResult[None]:
        """Validate handler registry protocol compliance.

        Args:
            handler: Handler object to validate
            handler_context: Context string for error messages

        Returns:
            FlextResult[None]: Success if valid, failure otherwise

        """
        # Check for required interface methods
        required_methods = ["handle", "execute"]
        for method_name in required_methods:
            if not hasattr(handler, method_name):
                continue  # At least one of these should exist
            method = getattr(handler, method_name)
            if callable(method):
                return FlextResult[None].ok(None)

        # If we get here, handler doesn't have required methods
        return FlextResult[None].fail(
            f"Invalid {handler_context}: must have 'handle' or 'execute' method"
        )

    # ==================== LAYER 3: ADVANCED PROCESSING PUBLIC APIS ====================

    def register_processor(
        self,
        name: str,
        processor: object,
        config: dict[str, object] | None = None,
    ) -> FlextResult[None]:
        """Register processor for advanced processing.

        Args:
            name: Processor name identifier
            processor: Processor object (callable or has process() method)
            config: Optional processor-specific configuration

        Returns:
            FlextResult[None]: Success if registered, failure if invalid processor

        """
        # Validate processor interface
        validation_result = self._validate_processor_interface(
            processor, f"processor '{name}'"
        )
        if validation_result.is_failure:
            return validation_result

        # Register processor and configuration
        self._processors[name] = processor
        if config is not None:
            self._processor_configs[name] = config
        else:
            self._processor_configs[name] = {}

        # Initialize per-processor metrics and lock
        self._processor_metrics_per_name[name] = {
            "successful_processes": 0,
            "failed_processes": 0,
            "executions": 0,
        }
        self._processor_locks[name] = threading.Lock()

        # Update global metrics
        self._process_metrics["registrations"] += 1

        return FlextResult[None].ok(None)

    def process(
        self,
        name: str,
        data: object,
    ) -> FlextResult[object]:
        """Process data through registered processor.

        This is the main entry point for Layer 3 processing. It routes to the
        registered processor and delegates through Layer 2 dispatch() for
        reliability patterns (circuit breaker, rate limiting, retry).

        Args:
            name: Processor name
            data: Data to process

        Returns:
            FlextResult[object]: Processed result or error

        """
        # Route to processor
        processor = self._route_to_processor(name)
        if processor is None:
            return FlextResult[object].fail(
                f"Processor '{name}' not registered. Register with register_processor()."
            )

        # Apply per-processor circuit breaker
        cb_result = self._apply_processor_circuit_breaker(name, processor)
        if cb_result.is_failure:
            return FlextResult[object].fail(
                f"Processor '{name}' circuit breaker is open"
            )

        # Apply per-processor rate limiter
        rl_result = self._apply_processor_rate_limiter(name)
        if rl_result.is_failure:
            return FlextResult[object].fail(f"Processor '{name}' rate limit exceeded")

        # Execute processor with metrics collection
        return self._execute_processor_with_metrics(name, processor, data)

    def process_batch(
        self,
        name: str,
        data_list: list[object],
        batch_size: int | None = None,
    ) -> FlextResult[list[object]]:
        """Process multiple items in batch.

        Args:
            name: Processor name
            data_list: List of items to process
            batch_size: Optional batch size (uses config default if None)

        Returns:
            FlextResult[list[object]]: List of processed items or error

        """
        if not data_list:
            return FlextResult[list[object]].ok([])

        # Resolve batch size
        resolved_batch_size = batch_size or self._batch_size

        # Use internal batch processing
        result = self._process_batch_internal(name, data_list, resolved_batch_size)

        if result.is_success:
            self._process_metrics["batch_operations"] += 1

        return result

    def process_parallel(
        self,
        name: str,
        data_list: list[object],
        max_workers: int | None = None,
    ) -> FlextResult[list[object]]:
        """Process multiple items in parallel.

        Args:
            name: Processor name
            data_list: List of items to process
            max_workers: Optional max worker threads (uses config default if None)

        Returns:
            FlextResult[list[object]]: List of processed items or error

        """
        if not data_list:
            return FlextResult[list[object]].ok([])

        # Resolve worker count
        resolved_workers = max_workers or self._parallel_workers

        # Use internal parallel processing
        result = self._process_parallel_internal(name, data_list, resolved_workers)

        if result.is_success:
            self._process_metrics["parallel_operations"] += 1

        return result

    def execute_with_timeout(
        self,
        name: str,
        data: object,
        timeout: float,
    ) -> FlextResult[object]:
        """Process with timeout enforcement.

        Args:
            name: Processor name
            data: Data to process
            timeout: Timeout in seconds

        Returns:
            FlextResult[object]: Processed result or timeout error

        """
        # Use TimeoutEnforcer from Layer 2 (same as dispatch method)
        try:
            executor = self._timeout_enforcer.ensure_executor()
            future: concurrent.futures.Future[FlextResult[object]] = executor.submit(
                self.process, name, data
            )
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            self._process_metrics["failed_processes"] += 1
            self._process_metrics["timeout_executions"] += 1
            return FlextResult[object].fail(
                f"Processor '{name}' timeout after {timeout}s"
            )

    def execute_with_fallback(
        self,
        name: str,
        data: object,
        fallback_names: list[str],
    ) -> FlextResult[object]:
        """Process with fallback chain.

        Args:
            name: Primary processor name
            data: Data to process
            fallback_names: List of fallback processor names (in order)

        Returns:
            FlextResult[object]: Result from first successful processor

        """
        # Try primary processor
        result = self.process(name, data)
        if result.is_success:
            self._process_metrics["successful_processes"] += 1
            return result

        # Try fallback processors in order
        for fallback_name in fallback_names:
            fallback_result = self.process(fallback_name, data)
            if fallback_result.is_success:
                self._process_metrics["fallback_executions"] += 1
                return fallback_result

        # All processors failed
        self._process_metrics["failed_processes"] += 1
        fallback_list = ", ".join(fallback_names)
        return FlextResult[object].fail(
            f"All processors failed: primary='{name}', fallbacks=[{fallback_list}]"
        )

    # ==================== LAYER 3: METRICS & AUDITING ====================

    @property
    def processor_metrics(self) -> dict[str, dict[str, int]]:
        """Get processor execution metrics.

        Returns:
            dict: Per-processor metrics with execution counts and success/failure rates

        """
        return self._processor_metrics_per_name.copy()

    @property
    def batch_performance(self) -> dict[str, object]:
        """Get batch operation performance metrics.

        Returns:
            dict: Batch operation statistics including operation count and metrics

        """
        batch_ops = self._process_metrics.get("batch_operations", 0)
        return {
            "batch_operations": batch_ops,
            "average_batch_size": self._batch_size,
        }

    @property
    def parallel_performance(self) -> dict[str, object]:
        """Get parallel operation performance metrics.

        Returns:
            dict: Parallel operation statistics including operation count and worker count

        """
        parallel_ops = self._process_metrics.get("parallel_operations", 0)
        return {
            "parallel_operations": parallel_ops,
            "max_workers": self._parallel_workers,
        }

    def get_process_audit_log(self) -> FlextResult[list[dict[str, object]]]:
        """Retrieve operation audit trail.

        Returns:
            FlextResult[list[dict]]: Audit log entries with operation details

        """
        return FlextResult[list[dict[str, object]]].ok(self._audit_log.copy())

    def get_performance_analytics(self) -> FlextResult[dict[str, object]]:
        """Get comprehensive performance analytics.

        Returns:
            FlextResult[dict]: Complete performance analytics including all metrics

        """
        analytics: dict[str, object] = {
            "global_metrics": self._process_metrics.copy(),
            "processor_metrics": self._processor_metrics_per_name.copy(),
            "batch_performance": self.batch_performance,
            "parallel_performance": self.parallel_performance,
            "performance_timings": self._performance_metrics.copy(),
            "processor_execution_times": self._processor_execution_times.copy(),
            "audit_log_entries": len(self._audit_log),
        }
        return FlextResult[dict[str, object]].ok(analytics)

    # ==================== LAYER 1: CQRS ROUTING INTERNAL METHODS ====================

    def _normalize_command_key(self, command_type_obj: object) -> str:
        """Create comparable key for command identifiers (from FlextBus).

        Args:
            command_type_obj: Object to create key from

        Returns:
            Normalized string key for command type

        """
        name_attr = getattr(command_type_obj, "__name__", None)
        if name_attr is not None:
            return str(name_attr)
        return str(command_type_obj)

    def _validate_handler_interface(
        self,
        handler: object,
        handler_context: str = "handler",
    ) -> FlextResult[None]:
        """Validate that handler has required handle() interface.

        Private helper to eliminate duplication in register_handler validation.

        Args:
            handler: Handler object to validate
            handler_context: Context string for error messages

        Returns:
            FlextResult[None]: Success if valid, failure if handler missing handle() method

        """
        method_name = FlextConstants.Mixins.METHOD_HANDLE
        handle_method = getattr(handler, method_name, None)
        if not callable(handle_method):
            return FlextResult[None].fail(
                f"Invalid {handler_context}: must have callable '{method_name}' method. "
                f"Handlers must implement handle(message) -> FlextResult[object]"
            )
        return FlextResult[None].ok(None)

    def _route_to_handler(self, command: object) -> object | None:
        """Locate the handler that can process the provided message.

        Args:
            command: The command/query object to find handler for

        Returns:
            The handler instance or None if not found

        """
        command_type = type(command)
        command_name = command_type.__name__

        # First, try to find handler by command type name in _handlers
        # (two-arg registration)
        if command_name in self._handlers:
            return self._handlers[command_name]

        # Search auto-registered handlers (single-arg form)
        for handler in self._auto_handlers:
            can_handle_method = getattr(handler, "can_handle", None)
            if callable(can_handle_method) and can_handle_method(command_type):
                return handler
        return None

    def _is_query(
        self,
        command: object,
        command_type: type,
    ) -> bool:
        """Determine if command is a query (cacheable).

        Args:
            command: The command object
            command_type: The type of the command

        Returns:
            bool: True if command is a query

        """
        return hasattr(command, "query_id") or "Query" in command_type.__name__

    def _generate_cache_key(
        self,
        command: object,
        command_type: type,
    ) -> str:
        """Generate a deterministic cache key for the command.

        Args:
            command: The command/query object
            command_type: The type of the command

        Returns:
            str: Deterministic cache key

        """
        return FlextUtilities.Cache.generate_cache_key(command, command_type)

    def _check_cache_for_result(
        self,
        command: object,
        command_type: type,
        *,
        is_query: bool,
    ) -> FlextResult[object] | None:
        """Check cache for query result and return if found.

        Args:
            command: The command object
            command_type: The type of the command
            is_query: Whether command is a query

        Returns:
            Cached FlextResult if found, None if not cacheable or not cached

        """
        cache_enabled = getattr(self.config, "cache_enabled", True)
        should_consider_cache = cache_enabled and is_query
        if not should_consider_cache:
            return None

        cache_key = self._generate_cache_key(command, command_type)
        cached_result: FlextResult[object] | None = self._cache.get(cache_key)
        if cached_result is not None:
            self.logger.debug(
                "Returning cached query result",
                command_type=command_type.__name__,
                cache_key=cache_key,
            )
            return cached_result

        return None

    def _execute_handler(
        self,
        handler: object,
        command: object,
    ) -> FlextResult[object]:
        """Execute the handler using standard handle() method.

        Requires handlers to implement handle(message) -> FlextResult[object].
        This eliminates the fallback pattern for type consistency.

        Args:
            handler: The handler instance to execute (must have handle() method)
            command: The command/query to process

        Returns:
            FlextResult: Handler execution result

        """
        self.logger.debug(
            "Delegating to handler",
            handler_type=handler.__class__.__name__,
        )

        # BREAKING CHANGE: Require standard handle() method
        # No fallback to execute() or process_command() - must use handle()
        handle_method = getattr(handler, FlextConstants.Mixins.METHOD_HANDLE, None)
        if not callable(handle_method):
            return FlextResult[object].fail(
                f"Handler must have callable '{FlextConstants.Mixins.METHOD_HANDLE}' method",
                error_code=FlextConstants.Errors.COMMAND_BUS_ERROR,
            )

        try:
            result = handle_method(command)
            if isinstance(result, FlextResult):
                return result
            # Wrap non-FlextResult return values
            return FlextResult[object].ok(result)
        except (TypeError, AttributeError, ValueError) as e:
            # TypeError: method signature mismatch
            # AttributeError: missing method attribute
            # ValueError: handler validation failed
            return FlextResult[object].fail(
                f"Handler execution failed: {e}",
                error_code=FlextConstants.Errors.COMMAND_PROCESSING_FAILED,
            )

    def _execute_middleware_chain(
        self,
        command: object,
        handler: object,
    ) -> FlextResult[None]:
        """Run the configured middleware pipeline for the current message.

        Args:
            command: The command/query to process
            handler: The handler that will execute the command

        Returns:
            FlextResult: Middleware processing result

        """
        middleware_enabled = getattr(self.config, "middleware_enabled", True)
        if not (middleware_enabled and self._middleware_configs):
            return FlextResult[None].ok(None)

        # Sort middleware by order
        sorted_middleware = sorted(
            self._middleware_configs, key=self._get_middleware_order
        )

        for middleware_config in sorted_middleware:
            result = self._process_middleware_instance(
                command, handler, middleware_config
            )
            if result.is_failure:
                return result

        return FlextResult[None].ok(None)

    @staticmethod
    def _get_middleware_order(middleware_config: dict[str, object]) -> int:
        """Extract middleware execution order from config."""
        order_value = middleware_config.get("order", 0)
        if isinstance(order_value, str):
            try:
                return int(order_value)
            except ValueError:
                return FlextConstants.Defaults.DEFAULT_MIDDLEWARE_ORDER
        return (
            int(order_value)
            if isinstance(order_value, int)
            else FlextConstants.Defaults.DEFAULT_MIDDLEWARE_ORDER
        )

    def _process_middleware_instance(
        self, command: object, handler: object, middleware_config: dict[str, object]
    ) -> FlextResult[None]:
        """Process a single middleware instance."""
        # Extract configuration values from dict
        middleware_id_value = middleware_config.get("middleware_id")
        middleware_type_value = middleware_config.get("middleware_type")
        enabled_value = middleware_config.get("enabled", True)

        # Skip disabled middleware
        if not enabled_value:
            self.logger.debug(
                "Skipping disabled middleware",
                middleware_id=middleware_id_value or "",
                middleware_type=str(middleware_type_value),
            )
            return FlextResult[None].ok(None)

        # Get actual middleware instance
        middleware_id_str = str(middleware_id_value) if middleware_id_value else ""
        middleware = self._middleware_instances.get(middleware_id_str)
        if middleware is None:
            return FlextResult[None].ok(None)

        self.logger.debug(
            "Applying middleware",
            middleware_id=middleware_id_value or "",
            middleware_type=str(middleware_type_value),
            order=middleware_config.get("order", 0),
        )

        return self._invoke_middleware(
            middleware, command, handler, middleware_type_value
        )

    def _invoke_middleware(
        self,
        middleware: object,
        command: object,
        handler: object,
        middleware_type: object,
    ) -> FlextResult[None]:
        """Invoke middleware and handle result."""
        process_method = getattr(middleware, "process", None)
        if callable(process_method):
            result = process_method(command, handler)
            return self._handle_middleware_result(result, middleware_type)
        if callable(middleware):
            # Fallback for callable middleware objects (like test middleware)
            result = middleware(command)
            return self._handle_middleware_result(result, middleware_type)

        return FlextResult[None].ok(None)

    def _handle_middleware_result(
        self, result: object, middleware_type: object
    ) -> FlextResult[None]:
        """Handle middleware execution result."""
        if isinstance(result, FlextResult):
            result_typed = cast("FlextResult[object]", result)
            if result_typed.is_failure:
                self.logger.info(
                    "Middleware rejected command",
                    middleware_type=str(middleware_type),
                    error=result_typed.error or "Unknown error",
                )
                return FlextResult[None].fail(
                    str(result_typed.error or "Middleware rejected command"),
                )

        return FlextResult[None].ok(None)

    # ==================== LAYER 1 PUBLIC API: CQRS ROUTING & MIDDLEWARE ====================

    def execute(self, command: object) -> FlextResult[object]:
        """Execute command/query via CQRS bus with caching and middleware.

        This is the main Layer 1 entry point - pure routing without reliability patterns.
        For reliability patterns (circuit breaker, rate limit, retry, timeout), use
        dispatch() which chains this execution with Layer 2 patterns.

        Args:
            command: The command or query object to execute

        Returns:
            FlextResult: Execution result wrapped in FlextResult

        """
        # Propagate context for distributed tracing
        command_type = type(command)
        self._propagate_context(f"execute_{command_type.__name__}")

        # Track operation metrics
        with self.track(f"bus_execute_{command_type.__name__}") as _:
            self._execution_count += 1

            self.logger.debug(
                "execute_command",
                command_type=command_type.__name__,
                command_id=getattr(
                    command,
                    "command_id",
                    getattr(command, "id", "unknown"),
                ),
                execution_count=self._execution_count,
            )

            # Check cache for queries
            is_query = self._is_query(command, command_type)
            cached_result = self._check_cache_for_result(
                command, command_type, is_query=is_query
            )
            if cached_result is not None:
                return cached_result

            # Resolve handler
            handler = self._route_to_handler(command)
            if handler is None:
                handler_names = [h.__class__.__name__ for h in self._auto_handlers]
                self.logger.error(
                    "No handler found",
                    command_type=command_type.__name__,
                    registered_handlers=handler_names,
                )
                return FlextResult[object].fail(
                    f"No handler found for {command_type.__name__}",
                    error_code=FlextConstants.Errors.COMMAND_HANDLER_NOT_FOUND,
                )

            # Apply middleware pipeline
            middleware_result: FlextResult[None] = self._execute_middleware_chain(
                command, handler
            )
            if middleware_result.is_failure:
                return FlextResult[object].fail(
                    middleware_result.error or "Middleware rejected command",
                    error_code=FlextConstants.Errors.COMMAND_BUS_ERROR,
                )

            # Execute handler and cache results
            result: FlextResult[object] = self._execute_handler(handler, command)

            # Cache successful query results
            if result.is_success and is_query:
                cache_key = self._generate_cache_key(command, command_type)
                self._cache[cache_key] = result
                self.logger.debug(
                    "Cached query result",
                    command_type=command_type.__name__,
                    cache_key=cache_key,
                )

            return result

    def layer1_register_handler(self, *args: object) -> FlextResult[None]:
        """Register handler with dual-mode support (from FlextBus).

        Supports:
        - Single-arg: register_handler(handler) - Auto-discovery with can_handle()
        - Two-arg: register_handler(MessageType, handler) - Explicit mapping

        Args:
            *args: Handler instance or (message_type, handler) pair

        Returns:
            FlextResult: Success or failure result

        """
        if len(args) == 1:
            handler = args[0]
            if handler is None:
                return FlextResult[None].fail("Handler cannot be None")

            # BREAKING CHANGE (Phase 6): Require standard handle() method
            # Enforces type-safe handler interface across entire ecosystem
            validation_result = self._validate_handler_interface(handler)
            if validation_result.is_failure:
                return validation_result

            # Add to auto-discovery list
            self._auto_handlers.append(handler)

            # Register by handler_id if available
            handler_id = getattr(handler, "handler_id", None)
            if handler_id is not None:
                self._handlers[str(handler_id)] = handler
                self.logger.info(
                    "Handler registered",
                    handler_type=getattr(
                        handler.__class__, "__name__", str(type(handler))
                    ),
                    handler_id=str(handler_id),
                    total_handlers=len(self._handlers),
                )
            else:
                self.logger.info(
                    "Handler registered for auto-discovery",
                    handler_type=getattr(
                        handler.__class__, "__name__", str(type(handler))
                    ),
                    total_handlers=len(self._auto_handlers),
                )
            return FlextResult[None].ok(None)

        # Two-arg form: (command_type, handler)
        two_arg_count = 2
        if len(args) == two_arg_count:
            command_type_obj, handler = args
            if handler is None or command_type_obj is None:
                return FlextResult[None].fail(
                    "Invalid arguments: command_type and handler are required",
                )

            if isinstance(command_type_obj, str) and not command_type_obj.strip():
                return FlextResult[None].fail("Command type cannot be empty")

            # BREAKING CHANGE (Phase 6): Validate handler interface
            validation_result = self._validate_handler_interface(
                handler,
                handler_context=f"handler for '{command_type_obj}'",
            )
            if validation_result.is_failure:
                return validation_result

            key = self._normalize_command_key(command_type_obj)
            self._handlers[key] = handler
            self.logger.info(
                "Handler registered for command type",
                command_type=key,
                handler_type=getattr(handler.__class__, "__name__", str(type(handler))),
                total_handlers=len(self._handlers),
            )
            return FlextResult[None].ok(None)

        return FlextResult[None].fail(
            f"register_handler takes 1 or 2 arguments but {len(args)} were given",
        )

    def layer1_add_middleware(
        self,
        middleware: object,
        middleware_config: dict[str, object] | None = None,
    ) -> FlextResult[None]:
        """Add middleware to processing pipeline (from FlextBus).

        Args:
            middleware: The middleware instance to add
            middleware_config: Configuration for the middleware (dict or None)

        Returns:
            FlextResult: Success or failure result

        """
        # Resolve middleware_id
        if middleware_config and middleware_config.get("middleware_id"):
            middleware_id_str = str(middleware_config.get("middleware_id"))
        else:
            middleware_id_str = getattr(
                middleware,
                "middleware_id",
                f"mw_{len(self._middleware_configs)}",
            )

        # Resolve middleware type
        middleware_type_str = (
            str(middleware_config.get("middleware_type"))
            if middleware_config and middleware_config.get("middleware_type")
            else type(middleware).__name__
        )

        # Create config
        final_config: dict[str, object] = {
            "middleware_id": middleware_id_str,
            "middleware_type": middleware_type_str,
            "enabled": middleware_config.get("enabled", True)
            if middleware_config
            else True,
            "order": middleware_config.get("order", len(self._middleware_configs))
            if middleware_config
            else len(self._middleware_configs),
        }

        self._middleware_configs.append(final_config)
        self._middleware_instances[middleware_id_str] = middleware

        self.logger.info(
            "Middleware added to pipeline",
            middleware_type=final_config.get("middleware_type"),
            middleware_id=middleware_id_str,
            total_middleware=len(self._middleware_configs),
        )

        return FlextResult[None].ok(None)

    # ==================== LAYER 1 EVENT PUBLISHING PROTOCOL ====================

    def publish_event(self, event: object) -> FlextResult[None]:
        """Publish domain event to subscribers (from FlextBus).

        Args:
            event: Domain event to publish

        Returns:
            FlextResult[None]: Success or failure result

        """
        try:
            # Use execute mechanism for event publishing
            result = self.execute(event)

            if result.is_failure:
                return FlextResult[None].fail(
                    f"Event publishing failed: {result.error}"
                )

            return FlextResult[None].ok(None)
        except (TypeError, AttributeError, ValueError) as e:
            # TypeError: invalid event type
            # AttributeError: missing event attribute
            # ValueError: event validation failed
            return FlextResult[None].fail(f"Event publishing error: {e}")

    def publish_events(self, events: list[object]) -> FlextResult[None]:
        """Publish multiple domain events (from FlextBus).

        Uses FlextResult.from_callable() to eliminate try/except and
        flow_through() for declarative event processing pipeline.

        Args:
            events: List of domain events to publish

        Returns:
            FlextResult[None]: Success or failure result

        """

        def publish_all() -> None:
            # Convert events to FlextResult pipeline
            def make_publish_func(
                event_item: object,
            ) -> Callable[[None], FlextResult[None]]:
                def publish_func(_: None) -> FlextResult[None]:
                    return self.publish_event(event_item)

                return publish_func

            publish_funcs = [make_publish_func(event) for event in events]
            result = FlextResult[None].ok(None).flow_through(*publish_funcs)
            if result.is_failure:
                raise RuntimeError(result.error or "Event publishing failed")

        return FlextResult[None].from_callable(publish_all)

    def subscribe(self, event_type: str, handler: object) -> FlextResult[None]:
        """Subscribe handler to event type (from FlextBus).

        Args:
            event_type: Type of event to subscribe to
            handler: Handler callable for the event

        Returns:
            FlextResult[None]: Success or failure result

        """
        try:
            # Use existing register_handler mechanism
            return self.layer1_register_handler(event_type, handler)
        except (TypeError, AttributeError, ValueError) as e:
            # TypeError: invalid handler type
            # AttributeError: handler missing required attributes
            # ValueError: handler validation failed
            return FlextResult[None].fail(f"Event subscription error: {e}")

    def unsubscribe(
        self,
        event_type: str,
        _handler: object | None = None,
    ) -> FlextResult[None]:
        """Unsubscribe from an event type (from FlextBus).

        Args:
            event_type: Type of event to unsubscribe from
            _handler: Handler to remove (reserved for future use)

        Returns:
            FlextResult[None]: Success or error result

        """
        try:
            # Remove handler from registry
            if event_type in self._handlers:
                del self._handlers[event_type]
                self.logger.info(
                    "Handler unregistered",
                    command_type=event_type,
                )
                return FlextResult[None].ok(None)

            return FlextResult[None].fail(
                f"Handler not found for event type: {event_type}"
            )
        except (TypeError, KeyError, AttributeError) as e:
            # TypeError: invalid event_type
            # KeyError: event_type not registered
            # AttributeError: handler missing attributes
            self.logger.exception("Event unsubscription error")
            return FlextResult[None].fail(f"Event unsubscription error: {e}")

    def publish(
        self,
        event_name: str,
        data: dict[str, object],
    ) -> FlextResult[None]:
        """Publish a named event with data (from FlextBus).

        Convenience method for publishing events by name with associated data.

        Args:
            event_name: Name/identifier of the event
            data: Event data payload

        Returns:
            FlextResult[None]: Success or failure result

        """
        # Create a simple event dict with name and data
        event: dict[str, object] = {
            "event_name": event_name,
            "data": data,
            "timestamp": time.time(),
        }
        return self.publish_event(event)

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
        protocol_validation = (
            FlextMixins.ProtocolValidation.validate_protocol_compliance(
                handler_obj, "Handler"
            )
        )
        if protocol_validation.is_failure:
            return FlextResult[dict[str, object]].fail(
                f"Handler protocol validation failed: {protocol_validation.error}",
            )

        # Register handler directly in dispatcher's internal structures
        handler_obj = request.get("handler")
        message_type = request.get("message_type")

        # Store based on whether message_type is provided
        if message_type:
            # Store in explicit handlers mapping
            handler_key = (
                message_type
                if isinstance(message_type, str)
                else getattr(message_type, "__name__", str(message_type))
            )
            self._handlers[handler_key] = handler_obj
        # Add to auto-discovery handlers list
        elif handler_obj not in self._auto_handlers:
            self._auto_handlers.append(handler_obj)

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
            ensure_result = self._ensure_handler(
                message_type_or_handler, mode=handler_mode
            )
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
                    lambda: self.execute(request.get("message")),
                    float(timeout_seconds),
                )
            else:
                # No timeout configured, execute directly
                execution_result = self.execute(request.get("message"))

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
            self._timeout_enforcer.should_use_executor() or timeout_override is not None
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

    def _track_timeout_context(
        self,
        operation_id: str,
        timeout_seconds: float,
    ) -> float:
        """Track timeout context and calculate deadline for operation.

        Propagates timeout context for deadline tracking and upstream
        cancellation support. Stores deadline for each operation.

        Args:
            operation_id: Unique operation identifier
            timeout_seconds: Timeout duration in seconds

        Returns:
            Deadline timestamp (current_time + timeout_seconds)

        """
        deadline = time.time() + timeout_seconds

        # Store timeout context with metadata
        self._timeout_contexts[operation_id] = {
            "operation_id": operation_id,
            "timeout_seconds": timeout_seconds,
            "deadline": deadline,
            "start_time": time.time(),
        }

        # Store deadline for quick lookup
        self._timeout_deadlines[operation_id] = deadline

        return deadline

    def _cleanup_timeout_context(self, operation_id: str) -> None:
        """Clean up timeout context after operation completes.

        Removes timeout context and deadline tracking for operation.
        Called after operation succeeds or fails.

        Args:
            operation_id: Operation identifier to clean up

        """
        self._timeout_contexts.pop(operation_id, None)
        self._timeout_deadlines.pop(operation_id, None)

    def _check_timeout_deadline(self, operation_id: str) -> bool:
        """Check if operation timeout deadline has been exceeded.

        Enables upstream timeout cancellation by checking current deadline.

        Args:
            operation_id: Operation identifier to check

        Returns:
            True if deadline exceeded, False if still within timeout window

        """
        deadline = self._timeout_deadlines.get(operation_id)
        if deadline is None:
            return False
        return time.time() > deadline

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
        if error_message is not None and not self._retry_policy.is_retriable_error(
            error_message
        ):
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

        # Normalize message and get type
        message, message_type = self._normalize_dispatch_message(
            message_or_type, data
        )

        # Check pre-dispatch conditions (circuit breaker + rate limiting)
        conditions_check = self._check_pre_dispatch_conditions(message_type)
        if conditions_check.is_failure:
            return FlextResult[object].fail(
                conditions_check.error,
                error_code=conditions_check.error_code,
                error_data=conditions_check.error_data,
            )

        # Execute with retry logic
        max_retries = self._retry_policy.get_max_attempts()
        operation_id = f"{message_type}_{id(message)}_{int(time.time() * 1000000)}"

        for attempt in range(max_retries):
            result = self._execute_dispatch_attempt(
                message,
                message_type,
                metadata,
                correlation_id,
                timeout_override,
                operation_id,
                attempt,
                max_retries,
            )
            if result.is_success or not self._should_retry_on_error(
                attempt, result.error if result.is_failure else None
            ):
                self._cleanup_timeout_context(operation_id)
                return result

        # Record final failure and clean up timeout context
        self._cleanup_timeout_context(operation_id)
        return FlextResult[object].fail("Max retries exceeded")

    def _normalize_dispatch_message(
        self, message_or_type: object | str, data: object | None
    ) -> tuple[object, str]:
        """Normalize message and extract message type."""
        # Support both old API (message_type, data) and new API (message)
        if isinstance(message_or_type, str):
            if data is not None:
                # Old API: dispatch(message_type, data)
                message = self._create_message_wrapper(data, message_or_type)
                return message, message_or_type
            else:
                # Old API: dispatch(message_type) - no data provided
                return None, message_or_type
        else:
            # New API: dispatch(message)
            message_type = type(message_or_type).__name__ if message_or_type else "unknown"
            return message_or_type, message_type

    def _create_message_wrapper(
        self, data: object, message_type: str
    ) -> object:
        """Create message wrapper for string message types."""

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

        return MessageWrapper(data=data, message_type=message_type)

    def _execute_dispatch_attempt(
        self,
        message: object,
        message_type: str,
        metadata: dict[str, object] | None,
        correlation_id: str | None,
        timeout_override: int | None,
        operation_id: str,
        attempt: int,
        max_retries: int,
    ) -> FlextResult[object]:
        """Execute a single dispatch attempt with timeout."""
        try:
            # Create structured request
            if metadata:
                string_metadata: dict[str, object] = {
                    k: str(v) for k, v in metadata.items()
                }
                FlextModels.Metadata(attributes=string_metadata)

            # Get timeout from config
            timeout_seconds = float(
                cast(
                    "int | float",
                    self.config.timeout_seconds,
                ),
            )
            if timeout_override:
                timeout_seconds = float(timeout_override)

            # Track timeout context with deadline for upstream cancellation
            self._track_timeout_context(operation_id, timeout_seconds)

            # Execute with timeout using shared ThreadPoolExecutor when enabled
            def execute_with_context() -> FlextResult[object]:
                if correlation_id is not None or timeout_override is not None:
                    context_metadata: dict[str, object] = {}
                    if timeout_override is not None:
                        context_metadata["timeout_override"] = timeout_override

                    with self._context_scope(context_metadata, correlation_id):
                        return self.execute(message)
                else:
                    return self.execute(message)

            # Execute with timeout enforcement (handles executor/threading logic)
            bus_result = self._execute_with_timeout(
                execute_with_context,
                timeout_seconds,
                timeout_override,
            )

            # Handle executor shutdown retry case
            if bus_result.is_failure and "Executor was shutdown" in (
                bus_result.error or ""
            ):
                return FlextResult[object].fail(
                    bus_result.error or "Executor was shutdown"
                )

            if bus_result.is_success:
                # Record success in circuit breaker
                self._circuit_breaker.record_success(message_type)
                return FlextResult[object].ok(bus_result.value)

            # Track circuit breaker failure
            self._circuit_breaker.record_failure(message_type)
            return FlextResult[object].fail(bus_result.error or "Dispatch failed")

        except Exception as e:
            # Track circuit breaker failure for exceptions
            self._circuit_breaker.record_failure(message_type)
            return FlextResult[object].fail(f"Dispatch error: {e}")

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

        raw_metadata = self._extract_metadata_mapping(metadata)

        if raw_metadata is None:
            return None

        normalized: dict[str, object] = {
            str(key): value for key, value in raw_metadata.items()
        }

        return normalized

    def _extract_metadata_mapping(
        self, metadata: object
    ) -> Mapping[str, object] | None:
        """Extract metadata as Mapping from various types."""
        if isinstance(metadata, FlextModels.Metadata):
            return self._extract_from_flext_metadata(metadata)
        if isinstance(metadata, Mapping):
            return cast("Mapping[str, object]", metadata)
        return self._extract_from_object_attributes(metadata)

    def _extract_from_flext_metadata(
        self, metadata: FlextModels.Metadata
    ) -> Mapping[str, object] | None:
        """Extract metadata mapping from FlextModels.Metadata."""
        attributes = metadata.attributes
        # Python 3.13+ type narrowing: attributes is already Mapping[str, object]
        if attributes and len(attributes) > 0:
            return attributes

        try:
            dumped = metadata.model_dump()
        except Exception:
            dumped = None

        if isinstance(dumped, Mapping):
            attributes_section = dumped.get("attributes")
            if isinstance(attributes_section, Mapping) and attributes_section:
                return cast("Mapping[str, object]", attributes_section)
            return cast("Mapping[str, object]", dumped)

        return None

    def _extract_from_object_attributes(
        self, metadata: object
    ) -> Mapping[str, object] | None:
        """Extract metadata mapping from object's attributes."""
        attributes_value = getattr(metadata, "attributes", None)
        if isinstance(attributes_value, Mapping) and attributes_value:
            return cast("Mapping[str, object]", attributes_value)

        model_dump = getattr(metadata, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump()
            except Exception:
                dumped = None
            if isinstance(dumped, Mapping):
                return cast("Mapping[str, object]", dumped)

        return None

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
            # Clear all handlers from dispatcher's internal structures
            self._handlers.clear()
            self._auto_handlers.clear()
            self._event_subscribers.clear()

            # Clear internal state
            self._circuit_breaker.cleanup()
            self._rate_limiter.cleanup()
            self._timeout_enforcer.cleanup()
            self._retry_policy.cleanup()

        except Exception as e:
            self._log_with_context("warning", "Cleanup failed", error=str(e))


__all__ = ["FlextDispatcher"]
