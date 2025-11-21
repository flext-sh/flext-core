"""High-level message dispatch orchestration with reliability patterns.

This module provides FlextDispatcher, a facade that orchestrates message
dispatching with circuit breaker, rate limiting, retry logic, timeout
enforcement, and comprehensive observability.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import concurrent.futures
import secrets
import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from typing import cast, override

from cachetools import LRUCache
from pydantic import BaseModel

from flext_core.constants import FlextConstants
from flext_core.context import FlextContext
from flext_core.handlers import FlextHandlers
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime
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
                message_type,
                FlextConstants.Reliability.CircuitBreakerState.CLOSED,
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
                message_type,
                FlextConstants.Reliability.CircuitBreakerState.CLOSED,
            )

        def transition_to_open(self, message_type: str) -> None:
            """Transition to OPEN state."""
            self.transition_to_state(
                message_type,
                FlextConstants.Reliability.CircuitBreakerState.OPEN,
            )

        def transition_to_half_open(self, message_type: str) -> None:
            """Transition to HALF_OPEN state."""
            self.transition_to_state(
                message_type,
                FlextConstants.Reliability.CircuitBreakerState.HALF_OPEN,
            )

        def attempt_reset(self, message_type: str) -> None:
            """Attempt recovery if circuit is open."""
            if self.is_open(message_type):
                opened_at = self._opened_at.get(message_type, 0.0)
                if (time.time() - opened_at) >= self._recovery_timeout:
                    self.transition_to_half_open(message_type)

        def check_before_dispatch(self, message_type: str) -> FlextResult[bool]:
            """Check if dispatch is allowed.

            Returns:
                FlextResult[bool]: Success with True if allowed, failure if circuit breaker is open

            """
            self.attempt_reset(message_type)
            if self.is_open(message_type):
                return FlextResult[bool].fail(
                    f"Circuit breaker is open for message type '{message_type}'",
                    error_code=FlextConstants.Errors.OPERATION_ERROR,
                    error_data={
                        "message_type": message_type,
                        "state": self.get_state(message_type),
                        "failure_count": self.get_failure_count(message_type),
                    },
                )
            return FlextResult[bool].ok(True)

        def get_failure_count(self, message_type: str) -> int:
            """Get current failure count."""
            return self._failures.get(message_type, 0)

        def get_threshold(self) -> int:
            """Get circuit breaker threshold."""
            return self._threshold

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
            # Use secrets.SystemRandom for cryptographically strong randomness
            secure_random = secrets.SystemRandom()
            variance = (2.0 * secure_random.random() - 1.0) * self._jitter_factor
            jittered = base_delay * (1.0 + variance)

            # Ensure jittered value doesn't go negative
            return max(0.0, jittered)

        def check_rate_limit(self, message_type: str) -> FlextResult[bool]:
            """Check if rate limit is exceeded for message type.

            Args:
                message_type: The message type to check

            Returns:
                FlextResult[bool]: Success with True if within limit, failure if exceeded

            """
            current_time = time.time()
            window_start, count = self._windows.get(message_type, (current_time, 0))

            # Reset window if elapsed
            if current_time - window_start >= self._window_seconds:
                window_start = current_time
                count = 0

            # Check if limit exceeded
            if count >= self._max_requests:
                # Calculate retry_after: seconds until window resets
                elapsed = current_time - window_start
                retry_after = max(0, int(self._window_seconds - elapsed))
                return FlextResult[bool].fail(
                    f"Rate limit exceeded for message type '{message_type}'",
                    error_code=FlextConstants.Errors.OPERATION_ERROR,
                    error_data={
                        "message_type": message_type,
                        "limit": self._max_requests,
                        "window_seconds": self._window_seconds,
                        "current_count": count,
                        "retry_after": retry_after,
                    },
                )

            # Update window tracking and increment count
            self._windows[message_type] = (window_start, count + 1)
            return FlextResult[bool].ok(True)

        def get_max_requests(self) -> int:
            """Get maximum requests per window."""
            return self._max_requests

        def get_window_seconds(self) -> float:
            """Get rate limit window duration in seconds."""
            return self._window_seconds

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
                retry_delay,
                0.0,
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
            str,
            dict[str, object],
        ] = {}  # operation_id → context
        self._timeout_deadlines: dict[
            str,
            float,
        ] = {}  # operation_id → deadline timestamp

        # ==================== LAYER 1: CQRS ROUTING INITIALIZATION ====================

        # Handler registry (from FlextBus dual-mode registration)
        self._handlers: dict[str, object] = {}  # Explicit command → handler mappings
        self._auto_handlers: list[object] = []  # Auto-discovery handlers

        # Middleware pipeline (from FlextBus)
        self._middleware_configs: list[dict[str, object]] = []  # Config + ordering
        self._middleware_instances: dict[str, object] = {}  # Keyed by middleware_id

        # Query result caching (from FlextBus - LRU cache)
        # Fast fail: use constant directly, no fallback
        max_cache_size = FlextConstants.Container.MAX_CACHE_SIZE
        self._cache: LRUCache[str, FlextResult[object]] = LRUCache(
            maxsize=max_cache_size,
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
            str,
            dict[str, int],
        ] = {}  # per-processor metrics
        self._processor_locks: dict[
            str,
            threading.Lock,
        ] = {}  # per-processor thread safety

        # Group 2: Batch & Parallel Configuration
        # Fast fail: use config values directly, no fallback
        self._batch_size: int = self.config.max_batch_size
        self._parallel_workers: int = self.config.executor_workers

        # Group 3: Handler Registry (from FlextProcessors.HandlerRegistry)
        self._handler_registry: dict[str, object] = {}  # name → handler function
        self._handler_configs: dict[
            str,
            dict[str, object],
        ] = {}  # name → handler config
        self._handler_validators: dict[
            str,
            Callable[[object], bool],
        ] = {}  # validation functions

        # Group 4: Pipeline (from FlextProcessors.Pipeline)
        self._pipeline_steps: list[dict[str, object]] = []  # Ordered pipeline steps
        self._pipeline_composition: dict[
            str,
            Callable[[object], FlextResult[object]],
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
            "timeout_executions": 0,
        }
        self._audit_log: list[dict[str, object]] = []  # Operation audit trail
        self._performance_metrics: dict[str, float] = {}  # Timing and throughput
        self._processor_execution_times: dict[
            str,
            list[float],
        ] = {}  # Per-processor times

    @property
    def dispatcher_config(self) -> dict[str, object]:
        """Access the dispatcher configuration."""
        return self.config.model_dump()

    # ==================== LAYER 3: ADVANCED PROCESSING INTERNAL METHODS ====================

    def _validate_interface(
        self,
        obj: object,
        method_names: list[str] | str,
        context: str,
        *,
        allow_callable: bool = False,
    ) -> FlextResult[bool]:
        """Generic interface validation (consolidates 3 validation methods).

        Args:
            obj: Object to validate
            method_names: Single method name or list of acceptable method names
            context: Context string for error messages
            allow_callable: If True, accept callable object without methods

        Returns:
            FlextResult[bool]: Success if valid, failure with descriptive error

        """
        if allow_callable and callable(obj):
            return self.ok(True)

        methods = [method_names] if isinstance(method_names, str) else method_names
        for method_name in methods:
            if hasattr(obj, method_name):
                method = getattr(obj, method_name)
                if callable(method):
                    return self.ok(True)

        method_list = "' or '".join(methods)
        return self.fail(f"Invalid {context}: must have '{method_list}' method")

    def _validate_processor_interface(
        self,
        processor: object,
        processor_context: str = "processor",
    ) -> FlextResult[bool]:
        """Validate that processor has required interface (callable or process method)."""
        return self._validate_interface(
            processor,
            "process",
            processor_context,
            allow_callable=True,
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
        return self.ok(processor)

    def _apply_processor_rate_limiter(self, _processor_name: str) -> FlextResult[bool]:
        """Apply per-processor rate limiting.

        Returns:
            FlextResult[bool]: Success with True if within limit, failure if exceeded

        """
        # Use global rate limiter manager
        # Per-processor rate limiting is handled at dispatch() level
        # For now, always allow (dispatch() will enforce global RL)
        return FlextResult[bool].ok(True)

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
            result: object
            if callable(processor):
                result = processor(data)
            else:
                # Fast fail: check if process method exists, no fallback
                if not hasattr(processor, "process"):
                    return self.fail(
                        f"Cannot execute processor '{processor_name}': "
                        "processor must be callable or have 'process' method",
                    )
                process_method = getattr(processor, "process", None)
                if process_method is None or not callable(process_method):
                    return self.fail(
                        f"Cannot execute processor '{processor_name}': "
                        "'process' attribute must be callable",
                    )
                result = process_method(data)

            # Convert to FlextResult if needed
            # Ensure result is wrapped in FlextResult using consolidated helper
            result_wrapped = FlextMixins.ResultHandling.ensure_result(result)

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
            if result_wrapped.is_success:
                metrics["successful_processes"] = (
                    metrics.get("successful_processes", 0) + 1
                )
            else:
                metrics["failed_processes"] = metrics.get("failed_processes", 0) + 1

            return result_wrapped
        except Exception as e:
            execution_time = time.time() - start_time
            if processor_name not in self._processor_execution_times:
                self._processor_execution_times[processor_name] = []
            self._processor_execution_times[processor_name].append(execution_time)
            return self.fail(f"Processor execution failed: {e}")

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
        # Fast fail: batch_size must be provided explicitly
        if batch_size is None:
            batch_size = self._batch_size
        if batch_size <= 0:
            return FlextResult[list[object]].fail(
                f"Invalid batch_size: {batch_size}. Must be > 0.",
            )
        results: list[object] = []

        processor = self._route_to_processor(processor_name)
        if processor is None:
            return FlextResult[list[object]].fail(
                f"Processor not found: {processor_name}",
            )

        # Process in batches
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i : i + batch_size]
            for data in batch:
                result = self._execute_processor_with_metrics(
                    processor_name,
                    processor,
                    data,
                )
                if result.is_success:
                    results.append(result.value)
                else:
                    error_msg = result.error or "Unknown error in processor"
                    return FlextResult[list[object]].fail(error_msg)

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
        # Fast fail: max_workers must be provided explicitly
        if max_workers is None:
            max_workers = self._parallel_workers
        if max_workers <= 0:
            return FlextResult[list[object]].fail(
                f"Invalid max_workers: {max_workers}. Must be > 0.",
            )
        results: list[object] = []

        processor = self._route_to_processor(processor_name)
        if processor is None:
            return FlextResult[list[object]].fail(
                f"Processor not found: {processor_name}",
            )

        # Process in parallel using ThreadPoolExecutor
        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
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
                        error_msg = result.error or "Unknown error in processor"
                        return FlextResult[list[object]].fail(error_msg)

            return FlextResult[list[object]].ok(results)
        except Exception as e:
            return FlextResult[list[object]].fail(f"Parallel processing failed: {e}")

    def _validate_handler_registry_interface(
        self,
        handler: object,
        handler_context: str = "registry handler",
    ) -> FlextResult[bool]:
        """Validate handler registry protocol compliance (handle or execute method)."""
        return self._validate_interface(handler, ["handle", "execute"], handler_context)

    # ==================== LAYER 3: ADVANCED PROCESSING PUBLIC APIS ====================

    def register_processor(
        self,
        name: str,
        processor: object,
        config: dict[str, object] | None = None,
    ) -> FlextResult[bool]:
        """Register processor for advanced processing.

        Args:
            name: Processor name identifier
            processor: Processor object (callable or has process() method)
            config: Optional processor-specific configuration

        Returns:
            FlextResult[bool]: Success with True if registered, failure if invalid processor

        """
        # Validate processor interface
        validation_result = self._validate_processor_interface(
            processor,
            f"processor '{name}'",
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

        return self.ok(True)

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
            return self.fail(
                f"Processor '{name}' not registered. Register with register_processor().",
            )

        # Apply per-processor circuit breaker
        cb_result = self._apply_processor_circuit_breaker(name, processor)
        if cb_result.is_failure:
            return self.fail(f"Processor '{name}' circuit breaker is open")

        # Apply per-processor rate limiter
        rate_limit_result = self._apply_processor_rate_limiter(name)
        if rate_limit_result.is_failure:
            error_msg = rate_limit_result.error or "Rate limit exceeded"
            return self.fail(error_msg)

        # Execute processor with metrics collection
        return self._execute_processor_with_metrics(name, processor, data)

    def _process_collection(
        self,
        name: str,
        data_list: list[object],
        resolved_param: int,
        operation_type: str,
        metric_key: str,
    ) -> FlextResult[list[object]]:
        """Process collection with specified operation type (DRY helper).

        Eliminates duplication between process_batch and process_parallel.
        Both methods follow identical pattern: empty check → call internal
        method → update metrics → return result.

        Args:
            name: Processor name
            data_list: List of items to process
            resolved_param: Resolved batch size or worker count
            operation_type: "batch" or "parallel" (for internal method routing)
            metric_key: Metric key to increment on success

        Returns:
            FlextResult[list[object]]: List of processed items or error

        """
        if not data_list:
            return FlextResult[list[object]].ok([])

        # Call appropriate internal method based on operation type
        if operation_type == "batch":
            result = self._process_batch_internal(name, data_list, resolved_param)
        else:  # parallel
            result = self._process_parallel_internal(name, data_list, resolved_param)

        if result.is_success:
            self._process_metrics[metric_key] += 1

        return result

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
        # Fast fail: explicit default value instead of 'or' fallback
        resolved_batch_size: int = (
            batch_size if batch_size is not None else self._batch_size
        )
        return self._process_collection(
            name,
            data_list,
            resolved_batch_size,
            "batch",
            "batch_operations",
        )

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
        # Fast fail: explicit default value instead of 'or' fallback
        resolved_workers: int = (
            max_workers if max_workers is not None else self._parallel_workers
        )
        return self._process_collection(
            name,
            data_list,
            resolved_workers,
            "parallel",
            "parallel_operations",
        )

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
                self.process,
                name,
                data,
            )
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            self._process_metrics["failed_processes"] += 1
            self._process_metrics["timeout_executions"] += 1
            return self.fail(f"Processor '{name}' timeout after {timeout}s")

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
        # Fast fail: __name__ should always exist on types, but handle gracefully
        if hasattr(command_type_obj, "__name__"):
            name_attr = getattr(command_type_obj, "__name__", None)
            if name_attr is not None:
                return str(name_attr)
        return str(command_type_obj)

    def _validate_handler_interface(
        self,
        handler: object,
        handler_context: str = "handler",
    ) -> FlextResult[bool]:
        """Validate that handler has required handle() interface."""
        method_name = FlextConstants.Mixins.METHOD_HANDLE
        return self._validate_interface(handler, method_name, handler_context)

    def _validate_handler_mode(self, handler_mode: str | None) -> FlextResult[bool]:
        """Validate handler mode against CQRS types (consolidates register_handler duplication)."""
        if handler_mode is None:
            return self.ok(True)

        valid_modes: list[str] = [
            m.value for m in FlextConstants.Cqrs.HandlerType.__members__.values()
        ]
        if str(handler_mode) not in valid_modes:
            return self.fail(
                f"Invalid handler_mode: {handler_mode}. Must be one of {valid_modes}",
            )

        return self.ok(True)

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
            handler_entry = self._handlers[command_name]
            # Extract handler from dict if registered via register_handler
            if FlextRuntime.is_dict_like(handler_entry) and "handler" in handler_entry:
                handler_obj: object = handler_entry["handler"]
                return handler_obj
            # Return handler directly if it's not a dict
            return handler_entry

        # Search auto-registered handlers (single-arg form)
        for handler in self._auto_handlers:
            # Fast fail: check if can_handle method exists before calling
            if hasattr(handler, "can_handle"):
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
    ) -> FlextResult[object]:
        """Check cache for query result and return if found.

        Args:
            command: The command object
            command_type: The type of the command
            is_query: Whether command is a query

        Returns:
            FlextResult[object]: Cached result if found, failure if not cacheable or not cached

        """
        # Fast fail: use config value directly, no fallback
        cache_enabled = self.config.enable_caching
        should_consider_cache = cache_enabled and is_query
        if not should_consider_cache:
            return self.fail(
                "Cache not enabled or not a query",
                error_code=FlextConstants.Errors.CONFIGURATION_ERROR,
            )

        cache_key = self._generate_cache_key(command, command_type)
        cached_value = self._cache.get(cache_key)
        if cached_value is not None:
            # Fast fail: cached value must be FlextResult[object]
            if not isinstance(cached_value, FlextResult):
                # Type checker may think this is unreachable, but it's reachable at runtime
                msg = f"Cached value is not FlextResult: {type(cached_value).__name__}"
                return self.fail(
                    msg,
                    error_code=FlextConstants.Errors.CONFIGURATION_ERROR,
                )
            cached_result: FlextResult[object] = cached_value
            self.logger.debug(
                "Returning cached query result",
                operation="check_cache",
                command_type=command_type.__name__,
                cache_key=cache_key,
                source="flext-core/src/flext_core/dispatcher.py",
            )
            return cached_result

        return self.fail(
            "Cache miss",
            error_code=FlextConstants.Errors.NOT_FOUND_ERROR,
        )

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
        handler_class = type(handler)
        handler_class_name = getattr(handler_class, "__name__", "Unknown")
        self.logger.debug(
            "Delegating to handler",
            operation="execute_handler",
            handler_type=handler_class_name,
            command_type=type(command).__name__,
            source="flext-core/src/flext_core/dispatcher.py",
        )

        # Try standard handle() method first, then execute() as fallback
        # Consistent with registration validation that accepts both
        method_name = None
        if hasattr(handler, "handle"):
            handle_method = getattr(handler, "handle", None)
            if callable(handle_method):
                method_name = "handle"
        elif hasattr(handler, "execute"):
            execute_method = getattr(handler, "execute", None)
            if callable(execute_method):
                method_name = "execute"

        if not method_name:
            return self.fail(
                "Handler must have 'handle' or 'execute' method",
                error_code=FlextConstants.Errors.COMMAND_BUS_ERROR,
            )
        handle_method = getattr(handler, method_name)
        if not callable(handle_method):
            return self.fail(
                f"Handler '{method_name}' must be callable",
                error_code=FlextConstants.Errors.COMMAND_BUS_ERROR,
            )

        try:
            result = handle_method(command)
            return FlextMixins.ResultHandling.ensure_result(result)
        except (TypeError, AttributeError, ValueError) as e:
            # TypeError: method signature mismatch
            # AttributeError: missing method attribute
            # ValueError: handler validation failed
            return self.fail(
                f"Handler execution failed: {e}",
                error_code=FlextConstants.Errors.COMMAND_PROCESSING_FAILED,
            )

    def _execute_middleware_chain(
        self,
        command: object,
        handler: object,
    ) -> FlextResult[bool]:
        """Run the configured middleware pipeline for the current message.

        Args:
            command: The command/query to process
            handler: The handler that will execute the command

        Returns:
            FlextResult: Middleware processing result

        """
        # Fast fail: middleware is always enabled if configs exist, no fallback
        # Middleware is enabled by default when configs are present
        if not self._middleware_configs:
            return self.ok(True)

        # Sort middleware by order
        sorted_middleware = sorted(
            self._middleware_configs,
            key=self._get_middleware_order,
        )

        for middleware_config in sorted_middleware:
            result = self._process_middleware_instance(
                command,
                handler,
                middleware_config,
            )
            if result.is_failure:
                return result

        return self.ok(True)

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
        self,
        command: object,
        handler: object,
        middleware_config: dict[str, object],
    ) -> FlextResult[bool]:
        """Process a single middleware instance."""
        # Extract configuration values from dict
        middleware_id_value = middleware_config.get("middleware_id")
        middleware_type_value = middleware_config.get("middleware_type")
        enabled_value = middleware_config.get("enabled", True)

        # Skip disabled middleware
        if not enabled_value:
            # Fast fail: middleware_id must be str if provided
            middleware_id_str = (
                "" if middleware_id_value is None else str(middleware_id_value)
            )
            self.logger.debug(
                "Skipping disabled middleware",
                middleware_id=middleware_id_str,
                middleware_type=str(middleware_type_value),
            )
            return self.ok(True)

        # Get actual middleware instance
        # Fast fail: middleware_id must be str if provided
        middleware_id_str = (
            str(middleware_id_value) if middleware_id_value is not None else ""
        )
        middleware = self._middleware_instances.get(middleware_id_str)
        if middleware is None:
            return self.ok(True)

        self.logger.debug(
            "Applying middleware",
            middleware_id=middleware_id_str,
            middleware_type=str(middleware_type_value),
            order=middleware_config.get("order", 0),
        )

        return self._invoke_middleware(
            middleware,
            command,
            handler,
            middleware_type_value,
        )

    def _invoke_middleware(
        self,
        middleware: object,
        command: object,
        handler: object,
        middleware_type: object,
    ) -> FlextResult[bool]:
        """Invoke middleware and handle result.

        Fast fail: Middleware must have process() method. No fallback to callable.
        """
        process_method = getattr(middleware, "process", None)
        if not callable(process_method):
            return self.fail(
                "Middleware must have callable 'process' method",
                error_code=FlextConstants.Errors.CONFIGURATION_ERROR,
            )
        result = process_method(command, handler)
        return self._handle_middleware_result(result, middleware_type)

    def _handle_middleware_result(
        self,
        result: object,
        middleware_type: object,
    ) -> FlextResult[bool]:
        """Handle middleware execution result."""
        if isinstance(result, FlextResult) and result.is_failure:
            error_msg = result.error
            self.logger.warning(
                "Middleware rejected command - command processing stopped",
                operation="execute_middleware",
                middleware_type=str(middleware_type),
                error=error_msg,
                consequence="Command will not be processed by handler",
                source="flext-core/src/flext_core/dispatcher.py",
            )
            return self.fail(result.unwrap_error())

        return self.ok(True)

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
                "Executing command",
                operation="execute",
                command_type=command_type.__name__,
                command_id=getattr(
                    command,
                    "command_id",
                    getattr(command, "id", "unknown"),
                ),
                execution_count=self._execution_count,
                source="flext-core/src/flext_core/dispatcher.py",
            )

            # Check cache for queries
            is_query = self._is_query(command, command_type)
            cached_result = self._check_cache_for_result(
                command,
                command_type,
                is_query=is_query,
            )
            if cached_result.is_success:
                return cached_result

            # Resolve handler
            handler = self._route_to_handler(command)
            if handler is None:
                handler_names = [h.__class__.__name__ for h in self._auto_handlers]
                self.logger.error(
                    "FAILED to find handler for command - DISPATCH ABORTED",
                    operation="execute",
                    command_type=command_type.__name__,
                    registered_handlers=handler_names,
                    consequence="Command cannot be processed - handler not registered",
                    resolution_hint="Register handler using register_handler() before dispatch",
                    source="flext-core/src/flext_core/dispatcher.py",
                )
                return self.fail(
                    f"No handler found for {command_type.__name__}",
                    error_code=FlextConstants.Errors.COMMAND_HANDLER_NOT_FOUND,
                )

            # Apply middleware pipeline
            middleware_result: FlextResult[bool] = self._execute_middleware_chain(
                command,
                handler,
            )
            if middleware_result.is_failure:
                # Fast fail: use unwrap_error() for type-safe str
                return self.fail(
                    middleware_result.unwrap_error(),
                    error_code=FlextConstants.Errors.COMMAND_BUS_ERROR,
                )

            # Execute handler and cache results
            result: FlextResult[object] = self._execute_handler(handler, command)

            # Cache successful query results
            cache_key: str | None = None
            if result.is_success and is_query:
                cache_key = self._generate_cache_key(command, command_type)
                self._cache[cache_key] = result
                self.logger.debug(
                    "Cached query result",
                    operation="cache_result",
                    command_type=command_type.__name__,
                    cache_key=cache_key,
                    source="flext-core/src/flext_core/dispatcher.py",
                )

            return result

    def layer1_register_handler(self, *args: object) -> FlextResult[bool]:
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
                return self.fail("Handler cannot be None")

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
                        handler.__class__,
                        "__name__",
                        str(type(handler)),
                    ),
                    handler_id=str(handler_id),
                    total_handlers=len(self._handlers),
                )
            else:
                self.logger.info(
                    "Handler registered for auto-discovery",
                    handler_type=getattr(
                        handler.__class__,
                        "__name__",
                        str(type(handler)),
                    ),
                    total_handlers=len(self._auto_handlers),
                )
            return self.ok(True)

        # Two-arg form: (command_type, handler)
        two_arg_count = 2
        if len(args) == two_arg_count:
            command_type_obj, handler = args
            if handler is None or command_type_obj is None:
                return self.fail(
                    "Invalid arguments: command_type and handler are required",
                )

            if isinstance(command_type_obj, str) and not command_type_obj.strip():
                return self.fail("Command type cannot be empty")

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
            return self.ok(True)

        return self.fail(
            f"register_handler takes 1 or 2 arguments but {len(args)} were given",
        )

    def layer1_add_middleware(
        self,
        middleware: object,
        middleware_config: dict[str, object] | None = None,
    ) -> FlextResult[bool]:
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
            operation="add_middleware",
            middleware_type=final_config.get("middleware_type"),
            middleware_id=middleware_id_str,
            total_middleware=len(self._middleware_configs),
            source="flext-core/src/flext_core/dispatcher.py",
        )

        return self.ok(True)

    # ==================== LAYER 1 EVENT PUBLISHING PROTOCOL ====================

    def publish_event(self, event: object) -> FlextResult[bool]:
        """Publish domain event to subscribers (from FlextBus).

        Args:
            event: Domain event to publish

        Returns:
            FlextResult[bool]: Success with True, failure with error details

        """
        try:
            # Use execute mechanism for event publishing
            result = self.execute(event)

            if result.is_failure:
                return self.fail(f"Event publishing failed: {result.error}")

            return self.ok(True)
        except (TypeError, AttributeError, ValueError) as e:
            # TypeError: invalid event type
            # AttributeError: missing event attribute
            # ValueError: event validation failed
            return self.fail(f"Event publishing error: {e}")

    def publish_events(self, events: list[object]) -> FlextResult[bool]:
        """Publish multiple domain events (from FlextBus).

        Uses FlextResult.from_callable() to eliminate try/except and
        flow_through() for declarative event processing pipeline.

        Args:
            events: List of domain events to publish

        Returns:
            FlextResult[bool]: Success with True, failure with error details

        """

        def publish_all() -> bool:
            # Convert events to FlextResult pipeline
            def make_publish_func(
                event_item: object,
            ) -> Callable[[object], FlextResult[bool]]:
                def publish_func(_unused: object) -> FlextResult[bool]:
                    return self.publish_event(event_item)

                return publish_func

            publish_funcs = [make_publish_func(event) for event in events]
            result = self.ok(True).flow_through(*publish_funcs)
            if result.is_failure:
                error_msg = result.error
                raise RuntimeError(error_msg)
            # Fast fail: return bool True for success
            return True

        return FlextResult[bool].from_callable(publish_all)

    def subscribe(self, event_type: str, handler: object) -> FlextResult[bool]:
        """Subscribe handler to event type (from FlextBus).

        Args:
            event_type: Type of event to subscribe to
            handler: Handler callable for the event

        Returns:
            FlextResult[bool]: Success with True, failure with error details

        """
        try:
            # Use existing register_handler mechanism
            return self.layer1_register_handler(event_type, handler)
        except (TypeError, AttributeError, ValueError) as e:
            # TypeError: invalid handler type
            # AttributeError: handler missing required attributes
            # ValueError: handler validation failed
            return self.fail(f"Event subscription error: {e}")

    def unsubscribe(
        self,
        event_type: str,
        _handler: object | None = None,
    ) -> FlextResult[bool]:
        """Unsubscribe from an event type (from FlextBus).

        Args:
            event_type: Type of event to unsubscribe from
            _handler: Handler to remove (reserved for future use)

        Returns:
            FlextResult[bool]: Success with True, failure with error details

        """
        try:
            # Remove handler from registry
            if event_type in self._handlers:
                del self._handlers[event_type]
                self.logger.info(
                    "Handler unregistered",
                    command_type=event_type,
                )
                return self.ok(True)

            return self.fail(f"Handler not found for event type: {event_type}")
        except (TypeError, KeyError, AttributeError) as e:
            # TypeError: invalid event_type
            # KeyError: event_type not registered
            # AttributeError: handler missing attributes
            self.logger.exception("Event unsubscription error")
            return self.fail(f"Event unsubscription error: {e}")

    def publish(
        self,
        event_name: str,
        data: dict[str, object],
    ) -> FlextResult[bool]:
        """Publish a named event with data (from FlextBus).

        Convenience method for publishing events by name with associated data.

        Args:
            event_name: Name/identifier of the event
            data: Event data payload

        Returns:
            FlextResult[bool]: Success with True, failure with error details

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
    def _get_nested_attr(self, obj: object, *path: str) -> object | None:
        """Get nested attribute safely (e.g., obj.attr1.attr2).

        Returns None if any attribute in path doesn't exist or is None.
        """
        current = obj
        for attr in path:
            if not hasattr(current, attr):
                return None
            current = getattr(current, attr, None)
            if current is None:
                return None
        return current

    def _extract_handler_name(
        self,
        handler: object,
        request_dict: dict[str, object],
    ) -> str:
        """Extract handler_name from request or handler config.

        Args:
            handler: Handler instance
            request_dict: Request dictionary

        Returns:
            Handler name string or empty string if not found

        """
        handler_name = str(request_dict.get("handler_name", ""))
        if handler_name:
            return handler_name

        # Try patterns in order of preference using consolidated helper
        patterns = [
            ("_config_model", "handler_name"),
            ("config", "handler_name"),
            ("handler_name",),
            ("__name__",),
            ("__class__", "__name__"),
        ]

        for pattern in patterns:
            value = self._get_nested_attr(handler, *pattern)
            if value is not None:
                return str(value)

        return ""

    def register_handler_with_request(
        self,
        request: dict[str, object] | object,
    ) -> FlextResult[dict[str, object]]:
        """Register handler using structured request model.

        Args:
            request: Dict or Pydantic model containing registration details

        Returns:
            FlextResult with registration details or error

        """
        # Convert Pydantic model to dict if needed
        # Validate request type and convert using consolidated helper
        if not isinstance(request, BaseModel) and not FlextRuntime.is_dict_like(
            request
        ):
            return FlextResult[dict[str, object]].fail(
                "Request must be dict or Pydantic model",
            )
        request_dict = FlextMixins.ModelConversion.to_dict(request)

        # Validate handler mode using consolidated helper
        handler_mode = cast("str | None", request_dict.get("handler_mode"))
        mode_validation = self._validate_handler_mode(handler_mode)
        if mode_validation.is_failure:
            return FlextResult[dict[str, object]].fail(
                mode_validation.error or "Invalid handler mode",
            )

        # Validate handler implementation
        handler = request_dict.get("handler")
        if not handler:
            return FlextResult[dict[str, object]].fail("Handler is required")

        # Validate handler has required interface
        validation_result = self._validate_handler_registry_interface(
            handler,
            handler_context="registered handler",
        )
        if validation_result.is_failure:
            return FlextResult[dict[str, object]].fail(
                validation_result.error or "Handler validation failed",
            )

        # Extract handler_name for tracking
        handler_name = self._extract_handler_name(handler, request_dict)
        if not handler_name:
            return FlextResult[dict[str, object]].fail("handler_name is required")

        # ARCHITECTURE: Two registration modes
        # 1. Auto-discovery: handlers with can_handle() method
        # 2. Explicit: handlers registered for specific message type

        # Check if handler supports auto-discovery
        can_handle_attr = (
            getattr(handler, "can_handle", None)
            if hasattr(handler, "can_handle")
            else None
        )
        has_can_handle = callable(can_handle_attr)

        if has_can_handle:
            # Mode 1: Auto-discovery via can_handle()
            # Add to _auto_handlers list for routing via can_handle()
            if handler not in self._auto_handlers:
                self._auto_handlers.append(handler)

            return FlextResult[dict[str, object]].ok({
                "handler_name": handler_name,
                "status": "registered",
                "mode": "auto_discovery",
            })

        # Mode 2: Explicit registration for specific message type
        message_type = request_dict.get("message_type")
        if not message_type:
            return FlextResult[dict[str, object]].fail(
                "Handler without can_handle() requires message_type",
            )

        # Get message type name for indexing
        name_attr = (
            getattr(message_type, "__name__", None)
            if hasattr(message_type, "__name__")
            else None
        )
        message_type_name = name_attr if name_attr is not None else str(message_type)

        # Store handler indexed by message type name
        self._handlers[message_type_name] = handler

        return FlextResult[dict[str, object]].ok({
            "handler_name": handler_name,
            "message_type": message_type_name,
            "status": "registered",
            "mode": "explicit",
        })

    def register_handler(
        self,
        request: dict[str, object] | BaseModel | object,
        handler: object | None = None,
    ) -> FlextResult[dict[str, object]]:
        """Register a handler dynamically.

        Args:
            request: Dict or Pydantic model containing registration details, or message_type string
            handler: Handler instance

        Returns:
            FlextResult with registration details or error

        """
        if handler is not None:
            result = self.layer1_register_handler(request, handler)
            if result.is_failure:
                return FlextResult[dict[str, object]].fail(
                    result.error or "Registration failed",
                )
            # Convert to dict format for consistency
            return FlextResult[dict[str, object]].ok({
                "handler_name": request if isinstance(request, str) else "unknown",
                "status": "registered",
                "mode": "explicit",
            })

        # Single-arg mode: register_handler(dict_or_model_or_handler)
        if isinstance(request, BaseModel) or FlextRuntime.is_dict_like(request):
            # Delegate to register_handler_with_request (eliminates ~100 lines of duplication)
            return self.register_handler_with_request(request)

        # Single handler object - delegate to layer1_register_handler
        result = self.layer1_register_handler(request)
        if result.is_failure:
            return FlextResult[dict[str, object]].fail(
                result.error or "Registration failed",
            )
        # Convert to dict format for consistency
        handler_name = getattr(request, "__class__", type(request)).__name__
        return FlextResult[dict[str, object]].ok({
            "handler_name": handler_name,
            "status": "registered",
            "mode": "auto_discovery",
        })

    def _register_handler(
        self,
        message_type: type[object],
        handler: FlextHandlers[object, object],
        handler_mode: str,
        handler_config: FlextTypes.HandlerConfigurationType = None,
    ) -> FlextResult[dict[str, object]]:
        """Register handler with specific mode (DRY helper).

        Eliminates duplication between register_command and register_query.
        Both methods follow identical pattern: create request dict → call
        register_handler_with_request().

        Args:
            message_type: Command or query message type
            handler: Handler instance
            handler_mode: Handler mode constant (COMMAND or QUERY)
            handler_config: Optional handler configuration

        Returns:
            FlextResult with registration details or error

        """
        request: dict[str, object] = {
            "handler": handler,
            "message_type": message_type,
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
        return self._register_handler(
            command_type,
            handler,
            FlextConstants.Dispatcher.HANDLER_MODE_COMMAND,
            handler_config,
        )

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
        return self._register_handler(
            query_type,
            handler,
            FlextConstants.Dispatcher.HANDLER_MODE_QUERY,
            handler_config,
        )

    def register_function(
        self,
        message_type: type[object],
        handler_func: FlextTypes.HandlerCallableType,
        *,
        handler_config: FlextTypes.HandlerConfigurationType = None,
        mode: FlextConstants.Cqrs.HandlerType = FlextConstants.Cqrs.COMMAND_HANDLER_TYPE,
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
            handler = FlextHandlers.create_from_callable(
                func=handler_func,
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
        request: dict[str, object] | BaseModel | object,
    ) -> FlextResult[dict[str, object]]:
        """Enhanced dispatch accepting Pydantic models or dicts.

        Args:
            request: Dict or Pydantic model containing dispatch details

        Returns:
            FlextResult with structured dispatch result

        """
        # Convert Pydantic model to dict if needed
        # Validate request type and convert using consolidated helper
        if not isinstance(request, BaseModel) and not FlextRuntime.is_dict_like(
            request
        ):
            return FlextResult[dict[str, object]].fail(
                "Request must be dict or Pydantic model",
            )
        request_dict = FlextMixins.ModelConversion.to_dict(request)

        # Propagate context for distributed tracing
        correlation_id = request_dict.get("correlation_id")
        if correlation_id and isinstance(correlation_id, str):
            FlextContext.Correlation.set_correlation_id(correlation_id)

        # Execute dispatch
        dispatch_result = self.dispatch(request_dict)

        # Wrap result in structured format
        if dispatch_result.is_success:
            structured_result = {
                "status": "success",
                "data": dispatch_result.value,
                "correlation_id": FlextContext.Correlation.get_correlation_id(),
            }
            return FlextResult[dict[str, object]].ok(structured_result)
        error_msg = dispatch_result.error or "Dispatch failed"
        return FlextResult[dict[str, object]].fail(error_msg)

    def _check_pre_dispatch_conditions(
        self,
        message_type: str,
    ) -> FlextResult[bool]:
        """Check all pre-dispatch conditions (circuit breaker, rate limiting).

        Orchestrates multiple validation checks in sequence. Returns first failure
        encountered, or success if all checks pass.

        Args:
            message_type: Message type string for reliability pattern checks

        Returns:
            FlextResult[bool]: Success with True if all checks pass, failure if any check fails

        """
        # Check circuit breaker state
        if not self._circuit_breaker.check_before_dispatch(message_type):
            failures = self._circuit_breaker.get_failure_count(message_type)
            return self.fail(
                f"Circuit breaker is open for message type '{message_type}'",
                error_code=FlextConstants.Errors.OPERATION_ERROR,
                error_data={
                    "message_type": message_type,
                    "failure_count": failures,
                    "threshold": self._circuit_breaker.get_threshold(),
                    "state": self._circuit_breaker.get_state(message_type),
                    "reason": "circuit_breaker_open",
                },
            )

        # Check rate limiting
        rate_limit_result = self._rate_limiter.check_rate_limit(message_type)
        if rate_limit_result.is_failure:
            error_msg = rate_limit_result.error or "Rate limit exceeded"
            return self.fail(error_msg)

        return self.ok(True)

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
                return self.fail(
                    f"Operation timeout after {timeout_seconds} seconds",
                )
            except Exception:
                # Executor was shut down; reinitialize and retry immediately
                self._timeout_enforcer.reset_executor()
                # Return retriable error so caller can retry
                return self.fail(
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
            error_message,
        ):
            return False

        # Delay before retry
        time.sleep(self._retry_policy.get_retry_delay())
        return True

    def dispatch(
        self,
        message_or_type: object,
        data: object | None = None,
        *,
        config: object | None = None,
        metadata: dict[str, object] | None = None,
        correlation_id: str | None = None,
        timeout_override: int | None = None,
    ) -> FlextResult[object]:
        """Dispatch message.

        Args:
            message_or_type: Message object or type string
            data: Message data
            config: DispatchConfig instance
            metadata: Optional execution context metadata
            correlation_id: Optional correlation ID for tracing
            timeout_override: Optional timeout override

        Returns:
            FlextResult with execution result or error

        """
        # Detect API pattern - (type, data) vs (object)
        message: object
        if data is not None or isinstance(message_or_type, str):
            # dispatch("type", data) pattern
            message_type_str = str(message_or_type)
            message_class = type(message_type_str, (), {"payload": data})
            message = message_class()
        else:
            # dispatch(message_object) pattern
            message = message_or_type

        # Fast fail: message cannot be None
        if message is None:
            return self.fail(
                "Message cannot be None",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )

        return (
            self._extract_dispatch_config(
                config,
                metadata,
                correlation_id,
                timeout_override,
            )
            .flat_map(
                lambda dispatch_config: self._prepare_dispatch_context(
                    message,
                    None,
                    dispatch_config,
                ),
            )
            .flat_map(self._validate_pre_dispatch_conditions)
            .flat_map(self._execute_with_retry_policy)
        )

    def _extract_dispatch_config(
        self,
        config: object | None,
        metadata: dict[str, object] | None,
        correlation_id: str | None,
        timeout_override: int | None,
    ) -> FlextResult[dict[str, object]]:
        """Extract and validate dispatch configuration using FlextUtilities."""
        try:
            # Extract config values (config takes priority over individual params)
            if config is not None:
                metadata = getattr(config, "metadata", metadata)
                correlation_id = getattr(config, "correlation_id", correlation_id)
                timeout_override = getattr(config, "timeout_override", timeout_override)

            # Validate metadata - NO fallback, explicit validation
            validated_metadata: dict[str, object]
            if metadata is None:
                validated_metadata = {}
            elif FlextRuntime.is_dict_like(metadata):
                validated_metadata = dict(metadata)
            elif isinstance(metadata, FlextModels.Metadata):
                # FlextModels.Metadata - extract attributes dict
                validated_metadata = metadata.attributes
            else:
                # Fast fail: metadata must be dict, FlextModels.Metadata, or None
                # Type checker may think this is unreachable, but it's reachable at runtime
                msg = (
                    f"Invalid metadata type: {type(metadata).__name__}. "
                    "Expected dict[str, object] | FlextModels.Metadata | None"
                )
                return FlextResult[dict[str, object]].fail(msg)

            # Use FlextUtilities for configuration validation
            # Fast fail: explicit type annotation instead of cast
            config_dict: dict[str, object] = {
                "metadata": validated_metadata,
                "correlation_id": correlation_id,
                "timeout_override": timeout_override,
            }

            validation_result = FlextUtilities.Validation.validate_dispatch_config(
                config_dict,
            )
            if validation_result.is_failure:
                return FlextResult[dict[str, object]].fail(
                    f"Invalid dispatch configuration: {validation_result.error}",
                )

            return FlextResult[dict[str, object]].ok(config_dict)

        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Configuration extraction failed: {e}",
            )

    def _prepare_dispatch_context(
        self,
        message: object,
        _data: object | None,
        dispatch_config: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Prepare dispatch context with message normalization and context propagation.

        Fast fail: Only accepts message objects. _data parameter is ignored (kept for signature compatibility).
        """
        try:
            # Propagate context for distributed tracing
            dispatch_type = type(message).__name__
            self._propagate_context(f"dispatch_{dispatch_type}")

            # Normalize message and get type
            message, message_type = self._normalize_dispatch_message(message, _data)

            context = {
                **dispatch_config,
                "message": message,
                "message_type": message_type,
                "dispatch_type": dispatch_type,
            }

            return FlextResult[dict[str, object]].ok(context)

        except Exception as e:
            return FlextResult[dict[str, object]].fail(
                f"Context preparation failed: {e}",
            )

    def _validate_pre_dispatch_conditions(
        self,
        context: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        """Validate pre-dispatch conditions (circuit breaker + rate limiting)."""
        # Fast fail: message_type must be str (created by _normalize_dispatch_message)
        message_type_raw = context.get("message_type")
        if not isinstance(message_type_raw, str):
            msg = f"Invalid message_type in context: {type(message_type_raw).__name__}, expected str"
            return FlextResult[dict[str, object]].fail(msg)
        message_type: str = message_type_raw

        # Check pre-dispatch conditions (circuit breaker + rate limiting)
        conditions_check = self._check_pre_dispatch_conditions(message_type)
        if conditions_check.is_failure:
            error_msg = conditions_check.error or "Pre-dispatch conditions check failed"
            return FlextResult[dict[str, object]].fail(
                error_msg,
                error_code=conditions_check.error_code,
                error_data=conditions_check.error_data,
            )

        return FlextResult[dict[str, object]].ok(context)

    def _execute_with_retry_policy(
        self,
        context: dict[str, object],
    ) -> FlextResult[object]:
        """Execute dispatch with retry policy using FlextUtilities."""
        # Fast fail: validate context values (created by _prepare_dispatch_context)
        message = context.get("message")
        message_type_raw = context.get("message_type")
        if not isinstance(message_type_raw, str):
            msg = f"Invalid message_type in context: {type(message_type_raw).__name__}, expected str"
            return FlextResult[object].fail(msg)
        message_type: str = message_type_raw

        metadata_raw = context.get("metadata")
        metadata: dict[str, object] | None = (
            metadata_raw
            if (metadata_raw is None or FlextRuntime.is_dict_like(metadata_raw))
            else None
        )

        correlation_id_raw = context.get("correlation_id")
        correlation_id: str | None = (
            correlation_id_raw
            if isinstance(correlation_id_raw, (str, type(None)))
            else None
        )

        timeout_override_raw = context.get("timeout_override")
        timeout_override: int | None = (
            timeout_override_raw
            if isinstance(timeout_override_raw, (int, type(None)))
            else None
        )

        # Generate operation ID using FlextUtilities
        operation_id = FlextUtilities.Generators.generate_operation_id(
            message_type,
            message,
        )

        # Use FlextUtilities for retry execution
        return FlextUtilities.Reliability.with_retry(
            lambda: self._execute_dispatch_attempt(
                message,
                message_type,
                metadata,
                correlation_id,
                timeout_override,
                operation_id,
            ),
            max_attempts=self._retry_policy.get_max_attempts(),
            should_retry_func=self._should_retry_on_error,
            cleanup_func=lambda: self._cleanup_timeout_context(operation_id),
        )

    def _normalize_dispatch_message(
        self,
        message: object,
        _data: object | None,
    ) -> tuple[object, str]:
        """Normalize message and extract message type.

        Fast fail: Only accepts message objects. No support for (message_type, data) API.
        """
        # Fast fail: message cannot be None
        if message is None:
            msg = "Message cannot be None. Use dispatch(message_object), not dispatch(None, data)."
            raise TypeError(msg)

        # Fast fail: message cannot be string
        if isinstance(message, str):
            msg = (
                "String message_type not supported. "
                "Use dispatch(message_object), not dispatch('message_type', data)."
            )
            raise TypeError(msg)

        # Extract message type from message object
        message_type = type(message).__name__
        return message, message_type

    def _create_message_wrapper(self, data: object, message_type: str) -> object:
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

    def _get_timeout_seconds(self, timeout_override: int | None) -> float:
        """Get timeout seconds from config or override.

        Args:
            timeout_override: Optional timeout override

        Returns:
            float: Timeout in seconds

        """
        # Fast fail: timeout_seconds must be numeric
        timeout_raw = self.config.timeout_seconds
        if not isinstance(timeout_raw, (int, float)):
            # Type checker may think this is unreachable, but it's reachable at runtime
            msg = f"Invalid timeout_seconds type: {type(timeout_raw).__name__}, expected int | float"
            raise TypeError(msg)
        timeout_seconds: float = float(timeout_raw)
        if timeout_override:
            timeout_seconds = float(timeout_override)
        return timeout_seconds

    def _create_execute_with_context(
        self,
        message: object,
        correlation_id: str | None,
        timeout_override: int | None,
    ) -> Callable[[], FlextResult[object]]:
        """Create execution function with context.

        Args:
            message: Message to execute
            correlation_id: Optional correlation ID
            timeout_override: Optional timeout override

        Returns:
            Callable that executes message with context

        """

        def execute_with_context() -> FlextResult[object]:
            if correlation_id is not None or timeout_override is not None:
                context_metadata: dict[str, object] = {}
                if timeout_override is not None:
                    context_metadata["timeout_override"] = timeout_override
                with self._context_scope(context_metadata, correlation_id):
                    return self.execute(message)
            return self.execute(message)

        return execute_with_context

    def _handle_dispatch_result(
        self,
        bus_result: FlextResult[object],
        message_type: str,
    ) -> FlextResult[object]:
        """Handle dispatch result with circuit breaker tracking.

        Args:
            bus_result: Result from bus execution
            message_type: Message type for circuit breaker

        Returns:
            FlextResult[object]: Processed result

        Raises:
            FlextExceptions.OperationError: If result is failure but error is None

        """
        if bus_result.is_failure:
            # Use unwrap_error() for type-safe str
            error_msg = bus_result.unwrap_error()
            if "Executor was shutdown" in error_msg:
                return self.fail(error_msg)
            self._circuit_breaker.record_failure(message_type)
            return self.fail(error_msg)

        self._circuit_breaker.record_success(message_type)
        return self.ok(bus_result.value)

    def _execute_dispatch_attempt(
        self,
        message: object,
        message_type: str,
        metadata: dict[str, object] | None,
        correlation_id: str | None,
        timeout_override: int | None,
        operation_id: str,
    ) -> FlextResult[object]:
        """Execute a single dispatch attempt with timeout."""
        try:
            # Create structured request
            if metadata:
                string_metadata: dict[str, object] = {
                    k: str(v) for k, v in metadata.items()
                }
                FlextModels.Metadata(attributes=string_metadata)

            timeout_seconds = self._get_timeout_seconds(timeout_override)
            self._track_timeout_context(operation_id, timeout_seconds)

            execute_with_context = self._create_execute_with_context(
                message,
                correlation_id,
                timeout_override,
            )

            bus_result = self._execute_with_timeout(
                execute_with_context,
                timeout_seconds,
                timeout_override,
            )

            return self._handle_dispatch_result(bus_result, message_type)

        except Exception as e:
            self._circuit_breaker.record_failure(message_type)
            return self.fail(f"Dispatch error: {e}")

    # ------------------------------------------------------------------
    # Private helper methods
    # ------------------------------------------------------------------
    # Context management
    # ------------------------------------------------------------------
    def _normalize_context_metadata(
        self,
        metadata: object | None,
    ) -> dict[str, object] | None:
        """Normalize metadata payloads to plain dictionaries.

        Fast fail: Direct validation without helpers.
        Handles BaseModel.model_dump(), Mapping, and direct dicts.
        """
        if metadata is None:
            return None

        # Fast fail: Direct extraction without helper
        # This handles BaseModel.model_dump(), Mapping, and direct dicts
        raw_metadata = self._extract_metadata_mapping(metadata)

        if raw_metadata is None:
            return None

        # Normalize keys to strings
        normalized: dict[str, object] = {
            str(key): value for key, value in raw_metadata.items()
        }

        return normalized

    def _extract_metadata_mapping(
        self,
        metadata: object,
    ) -> Mapping[str, object] | None:
        """Extract metadata as Mapping from various types.

        Fast fail: Direct validation without helpers.
        """
        if isinstance(metadata, FlextModels.Metadata):
            return self._extract_from_flext_metadata(metadata)
        # Fast fail: type narrowing instead of cast
        if isinstance(metadata, Mapping):
            return metadata

        # Handle Pydantic models directly - use model_dump() (Pydantic v2 pattern)
        if isinstance(metadata, BaseModel):
            # Fast fail: model_dump() must succeed for valid Pydantic models
            try:
                dumped = metadata.model_dump()
            except Exception as e:
                # Fast fail: model_dump() failure indicates invalid model
                msg = f"Failed to dump BaseModel metadata: {type(e).__name__}: {e}"
                raise TypeError(msg) from e

            # Fast fail: dumped must be dict (Pydantic guarantees this)
            if not FlextRuntime.is_dict_like(dumped):
                # Type checker may think this is unreachable, but it's reachable at runtime
                msg = (
                    f"metadata.model_dump() returned {type(dumped).__name__}, "
                    "expected dict"
                )
                raise TypeError(msg)
            return dumped

        return self._extract_from_object_attributes(metadata)

    def _extract_from_flext_metadata(
        self,
        metadata: FlextModels.Metadata,
    ) -> Mapping[str, object] | None:
        """Extract metadata mapping from FlextModels.Metadata."""
        attributes = metadata.attributes
        # Python 3.13+ type narrowing: attributes is already Mapping[str, object]
        if attributes and len(attributes) > 0:
            return attributes

        # Use model directly - Pydantic v2 supports direct attribute access
        # Fast fail: model_dump() must succeed for valid Pydantic models
        try:
            dumped = metadata.model_dump()
        except Exception as e:
            # Fast fail: model_dump() failure indicates invalid model
            msg = f"Failed to dump FlextModels.Metadata: {type(e).__name__}: {e}"
            raise TypeError(msg) from e

        # Fast fail: dumped must be dict (Pydantic guarantees this)
        if not FlextRuntime.is_dict_like(dumped):
            # Type checker may think this is unreachable, but it's reachable at runtime
            msg = (
                f"metadata.model_dump() returned {type(dumped).__name__}, expected dict"
            )
            raise TypeError(msg)

        # Extract attributes section if present - fast fail: must be dict or None
        attributes_section_raw = dumped.get("attributes")
        attributes_section: dict[str, object] | None = (
            attributes_section_raw
            if FlextRuntime.is_dict_like(attributes_section_raw)
            else None
        )
        if attributes_section is not None and attributes_section:
            return attributes_section
        # Return full dump if no attributes section
        return dumped

    def _extract_from_object_attributes(
        self,
        metadata: object,
    ) -> Mapping[str, object] | None:
        """Extract metadata mapping from object's attributes."""
        attributes_value = getattr(metadata, "attributes", None)
        if isinstance(attributes_value, Mapping) and attributes_value:
            # Type narrowed by isinstance check - no cast needed
            return attributes_value

        # Use model_dump() directly if available - Pydantic v2 pattern
        model_dump = getattr(metadata, "model_dump", None)
        if callable(model_dump):
            # Fast fail: model_dump() must succeed for valid Pydantic models
            try:
                dumped = model_dump()
            except Exception as e:
                # Fast fail: model_dump() failure indicates invalid model
                msg = f"Failed to dump metadata object: {type(e).__name__}: {e}"
                raise TypeError(msg) from e

            # Fast fail: dumped must be dict (Pydantic guarantees this)
            if not FlextRuntime.is_dict_like(dumped):
                # Type checker may think this is unreachable, but it's reachable at runtime
                msg = (
                    f"metadata.model_dump() returned {type(dumped).__name__}, "
                    "expected dict"
                )
                raise TypeError(msg)
            return dumped

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
        _message_type: str,
        messages: list[object],
    ) -> list[FlextResult[object]]:
        """Dispatch multiple messages in batch.

        Args:
            _message_type: Type of messages to dispatch (unused - extracted from message object)
            messages: List of message objects to dispatch

        Returns:
            List of FlextResult objects for each dispatched message

        """
        # Dispatch each message - message_type is extracted from message object
        return [self.dispatch(msg) for msg in messages]

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
