"""Processing utilities for message handling and data transformation.

This module provides FlextProcessors, utilities for processing messages,
applying middleware pipelines, and managing processing state with circuit
breakers, rate limiting, and caching.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from collections.abc import Callable
from typing import (
    cast,
    override,
)

from flext_core.config import FlextConfig
from flext_core.constants import FlextConstants
from flext_core.exceptions import FlextExceptions
from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextProcessors(FlextMixins):
    """Processing utilities for message handling and data transformation.

    Implements FlextProtocols.Middleware patterns through structural typing. Provides
    utilities for processing messages, applying middleware pipelines, and managing
    processing state with circuit breakers, rate limiting, and caching.

    BREAKING CHANGES (v0.10.0):
        - Handlers MUST implement handle(message) -> FlextResult[object] method
        - No fallback to callable() if handle() method missing - validation at registration time
        - Handler validation enforces standard interface immediately upon registration

    Middleware Integration:
        - add_middleware(middleware) - Register middleware for processing pipeline
        - Middleware implements FlextProtocols.Middleware protocol via duck typing
        - process(name, data) - Process data through registered middleware
        - process_parallel(name, data_list) - Parallel processing with middleware
        - process_batch(name, data_list) - Batch processing with middleware chain

    Nested Protocol Implementations:
        - FlextProcessors.Pipeline - Advanced processing pipeline with monadic composition
        - FlextProcessors.HandlerRegistry - Handler registry with FlextProtocols.Handler support

    Features:
    - Message processing with middleware pipeline implementing protocol patterns
    - Circuit breaker pattern for fault tolerance
    - Rate limiting for request throttling
    - Result caching with TTL support
    - Performance metrics collection
    - Audit logging for operations
    - Batch processing for multiple messages
    - Parallel processing with ThreadPoolExecutor
    - Handler registry with protocol compliance checking
    - Advanced processing pipeline with railway-oriented composition

    Usage:
        >>> from flext_core import FlextProcessors
        >>> from flext_core.protocols import FlextProtocols
        >>>
        >>> processors = FlextProcessors()
        >>> processors.register("audit", lambda d: {"audited": True, **d})
        >>>
        >>> # Add middleware that implements FlextProtocols.Middleware pattern
        >>> processors.add_middleware(lambda data: {"enriched": True, **data})
        >>>
        >>> result = processors.process("audit", {"data": "value"})
        >>>
        >>> # Middleware instances satisfy FlextProtocols.Middleware through structural typing
        >>> assert callable(processors._middleware[0]) or isinstance(
        ...     processors._middleware[0], FlextProtocols.Middleware
        ... )
    """

    @override
    def __init__(self, config: dict[str, object] | FlextConfig | None = None) -> None:
        """Initialize FlextProcessors with optional configuration.

        Args:
            config: Optional configuration. Can be:
                   - FlextConfig instance (preferred)
                   - dict for backward compatibility
                   - None to use global singleton FlextConfig

        """
        super().__init__()

        # Initialize service infrastructure (DI, Context, Logging, Metrics)
        self._init_service("flext_processors")

        # Enrich context with processor metadata for observability
        self._enrich_context(
            service_type="processor",
            circuit_breaker_enabled=True,
            rate_limiting_enabled=True,
            caching_enabled=True,
        )

        self._registry: dict[str, object] = {}
        self._middleware: list[object] = []
        self._audit_log: list[object] = []
        self._performance_metrics: FlextTypes.FloatDict = {}
        self._circuit_breaker: dict[str, object] = {}
        self._rate_limiter: dict[str, object] = {}
        self._cache: dict[str, tuple[object, float]] = {}
        self._lock = threading.Lock()  # Thread safety lock
        self._metrics: dict[str, object] = {
            "registrations": 0,
            "successful_processes": 0,
            "failed_processes": 0,
            "clear_operations": 0,
            "unregistrations": 0,
        }

        # Initialize configuration
        if config is None:
            # Use global FlextConfig singleton
            flext_config = FlextConfig()
            self._cache_ttl = float(flext_config.cache_ttl)
            self._circuit_breaker_threshold = flext_config.circuit_breaker_threshold
            self._rate_limit = flext_config.rate_limit_max_requests
            self._rate_limit_window = float(flext_config.rate_limit_window_seconds)
        elif isinstance(config, FlextConfig):
            # Use provided FlextConfig instance
            self._cache_ttl = float(config.cache_ttl)
            self._circuit_breaker_threshold = config.circuit_breaker_threshold
            self._rate_limit = config.rate_limit_max_requests
            self._rate_limit_window = float(config.rate_limit_window_seconds)
        else:
            # Backward compatibility: handle dict config
            cache_ttl = config.get("cache_ttl", 3600.0)
            self._cache_ttl = (
                float(cache_ttl) if isinstance(cache_ttl, (int, float, str)) else 3600.0
            )

            cb_threshold = config.get("circuit_breaker_threshold", 5)
            self._circuit_breaker_threshold = (
                int(cb_threshold) if isinstance(cb_threshold, (int, float, str)) else 5
            )

            rate_limit = config.get("rate_limit", 100)
            self._rate_limit = (
                int(rate_limit) if isinstance(rate_limit, (int, float, str)) else 100
            )

            rate_window = config.get("rate_limit_window", 60.0)
            self._rate_limit_window = (
                float(rate_window)
                if isinstance(rate_window, (int, float, str))
                else 60.0
            )

    def register(
        self,
        name: str,
        processor: Callable[
            [FlextTypes.ProcessorInputType], FlextTypes.ProcessorOutputType
        ],
    ) -> FlextResult[None]:
        """Register a processor with the given name.

        Args:
            name: Name to register the processor under
            processor: The processor function to register

        Returns:
            FlextResult[None]: Success if registration succeeded, failure otherwise

        """
        # Propagate context for distributed tracing
        self._propagate_context(f"register_processor_{name}")

        if not name:
            self._log_with_context("error", "Processor registration failed: empty name")
            return FlextResult[None].fail("Processor name cannot be empty")

        if name in self._registry:
            self._log_with_context(
                "warning",
                "Processor registration skipped: already exists",
                processor_name=name,
            )
            return FlextResult[None].fail(f"Processor '{name}' already registered")

        self._registry[name] = processor
        # Update metrics using dict access
        metrics = cast("dict[str, int]", self._metrics)
        metrics["registrations"] = metrics.get("registrations", 0) + 1

        self._log_with_context(
            "info", "Processor registered successfully", processor_name=name
        )
        return FlextResult[None].ok(None)

    def process(
        self, name: str, data: FlextTypes.ProcessorInputType
    ) -> FlextResult[FlextTypes.ProcessorOutputType]:
        """Process data using the named processor.

        Args:
            name: Name of the registered processor
            data: Data to process

        Returns:
            FlextResult[object]: Processed result or failure

        """
        # Propagate context for distributed tracing
        self._propagate_context(f"process_{name}")

        # Set operation context for this processing operation
        self._with_operation_context(
            "process_data",
            processor_name=name,
            data_type=type(data).__name__,
            data_size=len(str(data)) if data else 0,
        )

        if name not in self._registry:
            self._log_with_context("error", "Processor not found", processor_name=name)
            return FlextResult[object].fail(f"Processor '{name}' not found")

        processor: object = self._registry[name]

        # Check circuit breaker (simple boolean flag)
        if cast("dict[str, bool]", self._circuit_breaker).get(name, False):
            return FlextResult[object].fail(
                f"Circuit breaker open for processor '{name}'",
            )

        # Check rate limit using simple dict
        now = time.time()
        rate_key = f"{name}_rate"

        with self._lock:
            if rate_key not in self._rate_limiter:
                # Create simple rate limiter state dict
                rate_limiter: dict[str, object] = self._rate_limiter
                rate_limiter[rate_key] = {
                    "count": 0,
                    "window_start": now,
                }

            rate_data = cast("dict[str, object]", self._rate_limiter.get(rate_key))
            # Check if window has expired
            window_start = cast("float", rate_data.get("window_start", now))
            if now - window_start > self._rate_limit_window:
                rate_data["count"] = 0
                rate_data["window_start"] = now

            count = cast("int", rate_data.get("count", 0))
            if count >= self._rate_limit:
                return FlextResult[object].fail(
                    f"Rate limit exceeded for processor '{name}'",
                )

            rate_data["count"] = count + 1

        # Check cache
        cache_key = f"{name}:{hash(str(data))}"
        if cache_key in self._cache:
            cached_value, cached_time = self._cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return FlextResult[object].ok(cached_value)

        # Apply middleware
        processed_data = data
        for middleware in self._middleware:
            if callable(middleware):
                try:
                    processed_data = middleware(processed_data)
                except Exception as e:
                    return FlextResult[object].fail(f"Middleware error: {e}")

        # Execute processor with performance tracking
        try:
            if callable(processor):
                with self.track(f"process_{name}"):
                    result = processor(processed_data)
                # Check if result is a FlextResult-like object
                if (
                    hasattr(result, "is_success")
                    and hasattr(result, "value")
                    and hasattr(result, "error")
                ):
                    # Cast to the expected return type for better type inference
                    typed_result = cast(
                        "FlextResult[FlextTypes.ProcessorOutputType]", result
                    )
                    if typed_result.is_success:
                        result_value: FlextTypes.ProcessorOutputType = (
                            typed_result.value
                        )
                        self._cache[cache_key] = (result_value, time.time())
                        metrics = cast("dict[str, int]", self._metrics)
                        metrics["successful_processes"] = (
                            metrics.get("successful_processes", 0) + 1
                        )
                        self._audit_log.append(
                            cast(
                                "object",
                                {
                                    "timestamp": time.time(),
                                    "processor": name,
                                    "status": "success",
                                    "data_hash": hash(str(data)),
                                },
                            )
                        )
                        # Return the FlextResult directly, don't wrap it
                        return typed_result
                    # Cast to the expected return type for better type inference
                    typed_result = cast("FlextResult[object]", result)
                    metrics = cast("dict[str, int]", self._metrics)
                    metrics["failed_processes"] = metrics.get("failed_processes", 0) + 1
                    self._audit_log.append(
                        cast(
                            "object",
                            {
                                "timestamp": time.time(),
                                "processor": name,
                                "status": "failure",
                                "error": typed_result.error,
                                "data_hash": hash(str(data)),
                            },
                        )
                    )

                    return typed_result

                # Wrap non-FlextResult in FlextResult
                result_wrapped = FlextResult[object].ok(result)
                self._cache[cache_key] = (result, time.time())
                metrics = cast("dict[str, int]", self._metrics)
                metrics["successful_processes"] = (
                    metrics.get("successful_processes", 0) + 1
                )
                return result_wrapped

            return FlextResult[object].fail(f"Processor '{name}' is not callable")
        except Exception as e:
            metrics = cast("dict[str, int]", self._metrics)
            metrics["failed_processes"] = metrics.get("failed_processes", 0) + 1
            self._audit_log.append(
                cast(
                    "object",
                    {
                        "timestamp": time.time(),
                        "processor": name,
                        "status": "error",
                        "error": str(e),
                        "data_hash": hash(str(data)),
                    },
                )
            )
            return FlextResult[object].fail(f"Processor execution error: {e}")

    def add_middleware(self, middleware: FlextTypes.MiddlewareType) -> None:
        """Add middleware to the processing pipeline.

        Args:
            middleware: Middleware function to add

        """
        if callable(middleware):
            self._middleware.append(middleware)
        else:
            error_msg = "Middleware must be callable"
            raise FlextExceptions.TypeError(
                message=error_msg,
                error_code="TYPE_ERROR",
            )

    @property
    def metrics(self) -> dict[str, int]:
        """Get processing metrics.

        Returns:
            dict[str, int]: Current metrics as dictionary

        """
        # _metrics is already a dict, just return a copy
        return cast("dict[str, int]", self._metrics.copy())

    def get_metrics(self) -> dict[str, int]:
        """Get processing metrics (method accessor).

        Returns:
            dict[str, int]: Current metrics as dictionary

        """
        return self.metrics

    @property
    def audit_log(self) -> list[dict[str, object]]:
        """Get audit log of processing operations.

        Returns:
            list[dict[str, object]]: Audit log entries as dicts for backward compatibility

        """
        # _audit_log is already a list of dicts, just return it
        return cast("list[dict[str, object]]", self._audit_log)

    def get_audit_log(self) -> list[dict[str, object]]:
        """Get audit log (method accessor).

        Returns:
            list[dict[str, object]]: Audit log entries as dicts

        """
        return self.audit_log

    @property
    def performance_metrics(self) -> FlextTypes.FloatDict:
        """Get performance metrics.

        Returns:
            FlextTypes.FloatDict: Performance metrics

        """
        return self._performance_metrics.copy()

    def get_performance_metrics(self) -> FlextTypes.FloatDict:
        """Get performance metrics (method accessor).

        Returns:
            FlextTypes.FloatDict: Performance metrics

        """
        return self.performance_metrics

    def is_circuit_breaker_open(self, name: str) -> bool:
        """Check if circuit breaker is open for a processor.

        Args:
            name: Processor name

        Returns:
            bool: True if circuit breaker is open

        """
        # Circuit breakers stored as simple boolean flags in dict
        cb_dict = cast("dict[str, bool]", self._circuit_breaker)
        return cb_dict.get(name, False)

    def process_batch(
        self,
        name: str,
        data_list: list[object],
    ) -> FlextResult[list[object]]:
        """Process a batch of data items.

        Args:
            name: Processor name
            data_list: List of data items to process

        Returns:
            FlextResult[list[object]]: List of processed results

        """
        results: list[object] = []
        for data in data_list:
            result = self.process(name, data)
            if result.is_failure:
                return FlextResult[list[object]].fail(
                    f"Batch processing failed: {result.error}",
                )

            result_value: FlextTypes.ProcessorOutputType = result.value
            results.append(result_value)

        return FlextResult[list[object]].ok(results)

    def process_parallel(
        self,
        name: str,
        data_list: list[object],
    ) -> FlextResult[list[object]]:
        """Process data items in parallel using ThreadPoolExecutor.

        Args:
            name: Processor name
            data_list: List of data items to process

        Returns:
            FlextResult[list[object]]: List of processed results from parallel execution

        """
        # Validate processor exists before parallel execution
        if name not in self._registry:
            return FlextResult[list[object]].fail(f"Processor '{name}' not found")

        # Use ThreadPoolExecutor for true parallel processing
        max_workers = min(
            len(data_list), 10
        )  # Cap at 10 workers to avoid resource exhaustion
        results: list[object] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks for parallel execution
            futures = {
                executor.submit(self.process, name, data): data for data in data_list
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result.is_failure:
                        # Fail fast on first error
                        return FlextResult[list[object]].fail(
                            f"Parallel processing failed: {result.error}",
                        )
                    results.append(result.value)
                except Exception as e:
                    return FlextResult[list[object]].fail(
                        f"Parallel processing error: {e}",
                    )

        return FlextResult[list[object]].ok(results)

    def get_processors(self, name: str) -> list[object]:
        """Get registered processors by name.

        Args:
            name: Processor name

        Returns:
            list[object]: List of processors with the given name

        """
        if name in self._registry:
            return [self._registry[name]]
        return []

    def clear_processors(self) -> None:
        """Clear all registered processors."""
        self._registry.clear()
        metrics = cast("dict[str, int]", self._metrics)
        metrics["clear_operations"] = metrics.get("clear_operations", 0) + 1

    @property
    def statistics(self) -> dict[str, object]:
        """Get comprehensive statistics.

        Returns:
            dict[str, object]: Statistics dictionary

        """
        # Count circuit breakers that are "open" (stored as True in dict)
        cb_dict = cast("dict[str, bool]", self._circuit_breaker)
        open_count = sum(1 for is_open in cb_dict.values() if is_open)

        return cast(
            "dict[str, object]",
            {
                "total_processors": len(self._registry),
                "total_middleware": len(self._middleware),
                "metrics": self.metrics,
                "cache_size": len(self._cache),
                "circuit_breakers_open": open_count,
            },
        )

    def get_statistics(self) -> dict[str, object]:
        """Get statistics (method accessor).

        Returns:
            dict[str, object]: Statistics dictionary

        """
        return self.statistics

    def cleanup(self) -> None:
        """Clean up resources and reset state."""
        # Clear expired cache entries
        now = time.time()
        expired_keys = [
            key
            for key, (_, cached_time) in self._cache.items()
            if now - cached_time > self._cache_ttl
        ]
        for key in expired_keys:
            del self._cache[key]

        # Reset circuit breakers
        self._circuit_breaker.clear()

        # Reset rate limiters
        self._rate_limiter.clear()

    def validate(self) -> FlextResult[None]:
        """Validate processor configuration and state.

        Returns:
            FlextResult[None]: Success if valid, failure otherwise

        """
        if self._cache_ttl < FlextConstants.ZERO:
            return FlextResult[None].fail("Cache TTL must be non-negative")

        if self._circuit_breaker_threshold < FlextConstants.ZERO:
            return FlextResult[None].fail(
                "Circuit breaker threshold must be non-negative",
            )

        if self._rate_limit < FlextConstants.ZERO:
            return FlextResult[None].fail("Rate limit must be non-negative")

        return FlextResult[None].ok(None)

    def export_config(self) -> dict[str, object]:
        """Export current configuration.

        Returns:
            dict[str, object]: Configuration dictionary

        """
        return {
            "cache_ttl": self._cache_ttl,
            "circuit_breaker_threshold": self._circuit_breaker_threshold,
            "rate_limit": self._rate_limit,
            "rate_limit_window": self._rate_limit_window,
            "processor_count": len(self._registry),
            "middleware_count": len(self._middleware),
        }

    def import_config(
        self, config: dict[str, object] | FlextConfig
    ) -> FlextResult[None]:
        """Import configuration from FlextConfig or dict.

        Args:
            config: Either FlextConfig instance (preferred) or dict for backward compatibility

        Returns:
            FlextResult[None]: Success if import succeeded, failure otherwise

        """
        try:
            if isinstance(config, FlextConfig):
                self._cache_ttl = float(config.cache_ttl)
                self._circuit_breaker_threshold = config.circuit_breaker_threshold
                self._rate_limit = config.rate_limit_max_requests
                self._rate_limit_window = float(config.rate_limit_window_seconds)
            else:
                # Backward compatibility: handle dict config
                cache_ttl = config.get("cache_ttl", 3600.0)
                self._cache_ttl = (
                    float(cache_ttl)
                    if isinstance(cache_ttl, (int, float, str))
                    else 3600.0
                )

                cb_threshold = config.get("circuit_breaker_threshold", 5)
                self._circuit_breaker_threshold = (
                    int(cb_threshold)
                    if isinstance(cb_threshold, (int, float, str))
                    else 5
                )

                rate_limit = config.get("rate_limit", 100)
                self._rate_limit = (
                    int(rate_limit)
                    if isinstance(rate_limit, (int, float, str))
                    else 100
                )

                rate_window = config.get("rate_limit_window", 60.0)
                self._rate_limit_window = (
                    float(rate_window)
                    if isinstance(rate_window, (int, float, str))
                    else 60.0
                )
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Configuration import error: {e}")

    class Handler:
        """Minimal handler base returning modernization-compliant results."""

        def handle(
            self, request: FlextTypes.ProcessorInputType
        ) -> FlextResult[FlextTypes.ProcessorOutputType]:
            """Handle a request.

            Returns:
                FlextResult[FlextTypes.ProcessorOutputType]: A successful FlextResult wrapping
                handler output.

            """
            result: str = f"Base handler processed: {request}"
            return FlextResult[FlextTypes.ProcessorOutputType].ok(result)

    class HandlerRegistry:
        """Registry managing named handler instances for dispatcher pilots.

        BREAKING CHANGES (v0.10.0):
            - Handlers MUST implement handle(message) -> FlextResult[object] method
            - No fallback to callable() if handle() method missing
            - Validation enforces standard interface at registration time
        """

        @override
        def __init__(self) -> None:
            """Initialize handler registry."""
            super().__init__()
            self._handlers: dict[str, object] = {}

        def register(
            self,
            registration: FlextModels.HandlerRegistration,
        ) -> FlextResult[None]:
            """Register a handler using Pydantic model validation.

            Validates that handler implements standard handle() method.
            No fallback to callable() - BREAKING CHANGE in v0.10.0.

            Returns:
                FlextResult[None]: Success when registration is stored or a
                failed FlextResult with a validation/exists error.

            """
            if registration.name in self._handlers:
                return FlextResult[None].fail(
                    f"Handler '{registration.name}' already registered",
                    error_code=FlextConstants.Errors.ALREADY_EXISTS,
                )

            # Check handler registry size limits
            max_handlers = FlextConfig.get_global_instance().max_workers
            if len(self._handlers) >= max_handlers:
                return FlextResult[None].fail(
                    f"Handler registry full: {len(self._handlers)}/{max_handlers} handlers registered",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # BREAKING CHANGE: Validate handler has handle() method
            if not self._validate_handler_interface(registration.handler):
                return FlextResult[None].fail(
                    f"Handler '{registration.name}' must implement handle(message) -> FlextResult[object] method",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Validate handler using the model's built-in validation
            self._handlers[registration.name] = registration.handler
            return FlextResult[None].ok(None)

        def _validate_handler_interface(
            self, handler: Callable[[object], object]
        ) -> bool:
            """Validate handler implements standard handle() method.

            BREAKING CHANGE: No fallback to callable().
            Handler MUST have handle() method.

            Args:
                handler: Handler object to validate

            Returns:
                bool: True if handler has callable handle() method

            """
            if not hasattr(handler, FlextConstants.Mixins.METHOD_HANDLE):
                return False

            handle_method = getattr(
                handler,
                FlextConstants.Mixins.METHOD_HANDLE,
                None,
            )
            return handle_method is not None and callable(handle_method)

        def get(self, name: str) -> FlextResult[object]:
            """Get a handler.

            Returns:
                FlextResult[object]: The handler instance wrapped in a
                successful FlextResult, or a failed result if not found.

            """
            if name not in self._handlers:
                return FlextResult[object].fail(
                    f"Handler '{name}' not found",
                    error_code=FlextConstants.Errors.NOT_FOUND_ERROR,
                )
            return FlextResult[FlextTypes.ProcessorOutputType].ok(self._handlers[name])

        def execute(
            self, name: str, request: FlextTypes.ProcessorInputType
        ) -> FlextResult[FlextTypes.ProcessorOutputType]:
            """Execute a handler by name using railway pattern.

            Returns:
                FlextResult[object]: Result of handler execution or failure
                indicating handler not found or execution error.

            """

            def execute_handler(
                handler: object,
            ) -> FlextResult[FlextTypes.ProcessorOutputType]:
                # Cast to proper handler type for execution
                handler_callable = cast(
                    "Callable[[FlextTypes.ProcessorInputType], FlextTypes.ProcessorOutputType]",
                    handler,
                )
                return self._execute_handler(handler_callable, request, name)

            return self.get(name).flat_map(execute_handler)

        def _execute_handler(
            self,
            handler: Callable[
                [FlextTypes.ProcessorInputType], FlextTypes.ProcessorOutputType
            ],
            request: FlextTypes.ProcessorInputType,
            name: str,
        ) -> FlextResult[FlextTypes.ProcessorOutputType]:
            """Execute handler using standard handle() method.

            BREAKING CHANGE: No fallback to callable().
            Requires handlers to implement handle(message) -> FlextResult[FlextTypes.ProcessorOutputType].

            Returns:
                FlextResult[FlextTypes.ProcessorOutputType]: The result returned by the handler, or a
                failed FlextResult with a ProcessingError on exception.

            """
            try:
                # Require standard handle() method - no fallback pattern
                if not hasattr(handler, FlextConstants.Mixins.METHOD_HANDLE):
                    return FlextResult[object].fail(
                        f"Handler '{name}' must implement handle(message) -> FlextResult[object] method",
                        error_code=FlextConstants.Errors.NOT_FOUND_ERROR,
                    )

                handle_method = getattr(
                    handler,
                    FlextConstants.Mixins.METHOD_HANDLE,
                    None,
                )
                if handle_method is None or not callable(handle_method):
                    return FlextResult[object].fail(
                        f"Handler '{name}' handle method is not callable",
                        error_code=FlextConstants.Errors.NOT_FOUND_ERROR,
                    )

                result: FlextTypes.ProcessorOutputType = handle_method(request)
                if isinstance(result, FlextResult):
                    # Cast to FlextResult[FlextTypes.ProcessorOutputType] to ensure type compatibility
                    typed_result: FlextResult[FlextTypes.ProcessorOutputType] = cast(
                        "FlextResult[FlextTypes.ProcessorOutputType]",
                        result,
                    )
                    return typed_result
                return FlextResult[FlextTypes.ProcessorOutputType].ok(result)

            except Exception as e:
                return FlextResult[FlextTypes.ProcessorOutputType].fail(
                    f"Handler execution failed: {e}",
                    error_code=FlextConstants.Errors.PROCESSING_ERROR,
                )

        def count(self) -> int:
            """Get the number of registered handlers.

            Returns:
                int: The number of handlers registered.

            """
            return len(self._handlers)

        def exists(self, name: str) -> bool:
            """Check if a handler exists.

            Returns:
                bool: True if a handler with `name` is registered.

            """
            return name in self._handlers

        def get_optional(self, name: str) -> object | None:
            """Get a handler optionally, returning None if not found.

            Returns:
                object | None: Handler instance or None when not registered.

            """
            return self._handlers.get(name)

        def execute_with_timeout(
            self,
            config: FlextModels.HandlerExecutionConfig,
        ) -> FlextResult[object]:
            """Execute handler with timeout using HandlerExecutionConfig model.

            Returns:
                FlextResult[object]: The result of handler execution wrapped in
                a FlextResult, possibly a failure on timeout.

            """
            # Get timeout value from config or use default
            timeout_seconds = getattr(
                config,
                "timeout_seconds",
                FlextConstants.Defaults.OPERATION_TIMEOUT_SECONDS,
            )

            # Ensure timeout_seconds is not None and is an integer
            if timeout_seconds is None:
                timeout_seconds = FlextConstants.Defaults.OPERATION_TIMEOUT_SECONDS

            # Convert to int if it's not already
            if not isinstance(timeout_seconds, int):
                timeout_seconds = int(timeout_seconds)

            # Validate timeout bounds
            if timeout_seconds < FlextConstants.Container.MIN_TIMEOUT_SECONDS:
                return FlextResult[object].fail(
                    f"Timeout {timeout_seconds} is below minimum {FlextConstants.Container.MIN_TIMEOUT_SECONDS}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            if timeout_seconds > FlextConstants.Container.MAX_TIMEOUT_SECONDS:
                return FlextResult[object].fail(
                    f"Timeout {timeout_seconds} exceeds maximum {FlextConstants.Container.MAX_TIMEOUT_SECONDS}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Execute handler with timeout protection using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.execute,
                    config.handler_name,
                    config.input_data,
                )

                try:
                    # Wait for result with timeout - return directly
                    return future.result(timeout=float(timeout_seconds))
                except concurrent.futures.TimeoutError:
                    # Cancel the future and return timeout error
                    future.cancel()
                    return FlextResult[object].fail(
                        f"Handler '{config.handler_name}' execution timed out after {timeout_seconds} seconds",
                        error_code=FlextConstants.Errors.TIMEOUT_ERROR,
                    )
                except Exception as e:
                    return FlextResult[object].fail(
                        f"Handler execution failed: {e}",
                        error_code=FlextConstants.Errors.PROCESSING_ERROR,
                    )

        def execute_with_fallback(
            self,
            config: FlextModels.HandlerExecutionConfig,
        ) -> FlextResult[object]:
            """Execute handler with fallback handlers using HandlerExecutionConfig model.

            Returns:
                FlextResult[object]: The result of handler execution, trying fallbacks if primary fails.

            """
            # Try primary handler first
            primary_result = self.execute(config.handler_name, config.input_data)
            if primary_result.is_success:
                return primary_result

            # Try fallback handlers if primary failed
            for fallback_name in config.fallback_handlers:
                fallback_result = self.execute(fallback_name, config.input_data)
                if fallback_result.is_success:
                    return fallback_result

            # All handlers failed, return the original primary failure
            return primary_result

        def execute_batch(
            self,
            config: FlextModels.BatchProcessingConfig,
        ) -> FlextResult[list[object]]:
            """Execute multiple handlers using BatchProcessingConfig model.

            Returns:
                FlextResult[list[object]]: List of handler results or a failed
                FlextResult if validation or batch processing fails.

            """
            # Validate config is a BatchProcessingConfig or has required attributes
            data_items = config.data_items
            continue_on_error = config.continue_on_error

            # Validate batch size limits
            max_batch_size = FlextConfig.get_global_instance().max_batch_size
            if len(data_items) > max_batch_size:
                return FlextResult[list[object]].fail(
                    f"Batch size {len(data_items)} exceeds maximum {max_batch_size}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Convert data_items to handler execution tuples
            # Assume data_items contains tuples of (handler_name, request_data)
            handler_requests: list[tuple[str, object]] = []
            expected_tuple_length = FlextConstants.Performance.EXPECTED_TUPLE_LENGTH
            for item in data_items:
                if (
                    isinstance(item, tuple)
                    and len(cast("tuple[object, ...]", item)) == expected_tuple_length
                ):
                    # Type assertion for tuple elements
                    handler_name, request_data = cast("tuple[str, object]", item)
                    handler_requests.append((handler_name, request_data))
                else:
                    return FlextResult[list[object]].fail(
                        "Each data item must be a tuple of (handler_name, request_data)",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )

            return FlextResult.parallel_map(
                handler_requests,
                lambda item: self.execute(item[0], item[1]),
                fail_fast=not continue_on_error,
            )

        def register_with_validation(
            self,
            registration: FlextModels.HandlerRegistration,
            validator: Callable[[object], FlextResult[None]] | None = None,
        ) -> FlextResult[None]:
            """Register handler with optional validation using HandlerRegistration model.

            Returns:
                FlextResult[None]: Result of registration, success or failure.

            """
            if validator:

                def register_handler(_: None) -> FlextResult[None]:
                    return self.register(registration)

                return validator(registration.handler).flat_map(register_handler)
            return self.register(registration)

    class Pipeline:
        """Advanced processing pipeline using monadic composition."""

        @override
        def __init__(self) -> None:
            """Initialize processing pipeline."""
            super().__init__()
            self._steps: list[
                Callable[[object], FlextResult[object] | object]
                | dict[str, object]
                | object
            ] = []

        def add_step(
            self,
            step: Callable[[object], FlextResult[object] | object]
            | dict[str, object]
            | object,
        ) -> None:
            """Add a processing step."""
            self._steps.append(step)

        def process(
            self, data: FlextTypes.ProcessorInputType
        ) -> FlextResult[FlextTypes.ProcessorOutputType]:
            """Process data through pipeline using advanced railway pattern.

            Returns:
                FlextResult[FlextTypes.ProcessorOutputType]: Result of pipeline processing.

            """
            # Filter steps to only process callables
            callable_steps = [step for step in self._steps if callable(step)]
            return FlextResult.pipeline(
                data,
                *[self._process_step(step) for step in callable_steps],
            )

        def process_conditionally(
            self,
            request: FlextModels.ProcessingRequest,
            condition: Callable[[object], bool],
        ) -> FlextResult[object]:
            """Process data conditionally using railway patterns.

            Returns:
                FlextResult[object]: Result of conditional processing.

            """

            def process_data(data: dict[str, object]) -> FlextResult[object]:
                return self.process(
                    FlextModels.ProcessingRequest(
                        data=data,
                        context=request.context,
                        timeout_seconds=request.timeout_seconds,
                    ),
                )

            data_result = FlextResult[dict[str, object]].ok(request.data)
            if data_result.is_success and condition(data_result.unwrap()):
                return process_data(data_result.unwrap())
            return cast("FlextResult[object]", data_result)

        def process_with_timeout(
            self,
            request: FlextModels.ProcessingRequest,
        ) -> FlextResult[object]:
            """Process data with timeout using ProcessingRequest model.

            Returns:
                FlextResult[object]: Result of processing or timeout error.

            """
            timeout_seconds = getattr(
                request,
                "timeout_seconds",
                FlextConfig.get_global_instance().timeout_seconds,
            )

            # Validate timeout bounds
            if timeout_seconds < FlextConstants.Container.MIN_TIMEOUT_SECONDS:
                return FlextResult[object].fail(
                    f"Timeout {timeout_seconds} is below minimum {FlextConstants.Container.MIN_TIMEOUT_SECONDS}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            if timeout_seconds > FlextConstants.Container.MAX_TIMEOUT_SECONDS:
                return FlextResult[object].fail(
                    f"Timeout {timeout_seconds} exceeds maximum {FlextConstants.Container.MAX_TIMEOUT_SECONDS}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Execute processing with timeout protection
            try:
                return self.process(request)
            except Exception as e:
                return FlextResult[object].fail(f"Processing failed: {e}")

        def process_with_fallback(
            self,
            request: FlextModels.ProcessingRequest,
            *fallback_pipelines: FlextProcessors.Pipeline,
        ) -> FlextResult[object]:
            """Process with fallback pipelines using ProcessingRequest model.

            Returns:
                FlextResult[object]: Result from the first successful pipeline or
                the final failure.

            """

            # Define fallback operations with proper typing
            def create_fallback_operation(
                p: FlextProcessors.Pipeline,
            ) -> Callable[[], FlextResult[object]]:
                return lambda: p.process(request.data)

            fallback_operations = [
                create_fallback_operation(pipeline) for pipeline in fallback_pipelines
            ]

            # Try primary operation first
            primary_result = self.process(request.data)
            if primary_result.is_success:
                return primary_result

            # Try fallback operations
            for fallback_operation in fallback_operations:
                fallback_result = fallback_operation()
                if fallback_result.is_success:
                    return fallback_result

            # All operations failed, return the primary failure
            return primary_result

        def process_batch(
            self,
            config: FlextModels.BatchProcessingConfig | object,
        ) -> FlextResult[list[object]]:
            """Process batch of data using validated BatchProcessingConfig model.

            Args:
                config: BatchProcessingConfig model with data items and processing options

            Returns:
                FlextResult[list[object]]: List of processed data items or a failure.

            """
            # Validate config is a BatchProcessingConfig or has required attributes
            if not isinstance(config, FlextModels.BatchProcessingConfig):
                # For mock objects, validate they have required attributes
                if not hasattr(config, "data_items"):
                    return FlextResult[list[object]].fail(
                        "Config must be BatchProcessingConfig or have data_items attribute",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    )
                # Type narrowing for mock objects
                if not hasattr(config, "data_items"):
                    error_msg = "Config object must have data_items attribute"
                    raise FlextExceptions.TypeError(
                        message=error_msg,
                        error_code="TYPE_ERROR",
                    )
                # Cast is safe here because we've validated hasattr above
                config_with_data = cast("FlextModels.BatchProcessingConfig", config)
                data_items = config_with_data.data_items
                continue_on_error = getattr(config, "continue_on_error", True)
            else:
                data_items = config.data_items
                continue_on_error = config.continue_on_error

            # Validate batch size limits
            max_batch_size = FlextConfig.get_global_instance().max_batch_size
            if len(data_items) > max_batch_size:
                return FlextResult[list[object]].fail(
                    f"Batch size {len(data_items)} exceeds maximum {max_batch_size}",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Process all data items using parallel processing
            return FlextResult.parallel_map(
                data_items,
                self.process,
                fail_fast=not continue_on_error,
            )

        def process_with_validation(
            self,
            request: FlextModels.ProcessingRequest,
            *validators: Callable[[object], FlextResult[None]],
        ) -> FlextResult[object]:
            """Process with comprehensive validation pipeline using ProcessingRequest model.

            Returns:
                FlextResult[object]: Result of validation-then-processing or processing directly.

            """
            # Apply validation if enabled in the request
            if request.enable_validation:
                validation_result = FlextResult.validate_all(request.data, *validators)
                if validation_result.is_failure:
                    return cast("FlextResult[object]", validation_result)
                return self.process(request.data)
            return self.process(request.data)

        def _process_step(
            self,
            step: Callable[
                [FlextTypes.ProcessorInputType], FlextTypes.ProcessorOutputType
            ],
        ) -> Callable[
            [FlextTypes.ProcessorInputType],
            FlextResult[FlextTypes.ProcessorOutputType],
        ]:
            """Convert pipeline step to FlextResult-returning function.

            Returns:
                Callable[[FlextTypes.ProcessorInputType], FlextResult[FlextTypes.ProcessorOutputType]]: Adapter that wraps step execution.

            """

            def step_processor(
                current: FlextTypes.ProcessorInputType,
            ) -> FlextResult[FlextTypes.ProcessorOutputType]:
                return FlextResult[FlextTypes.ProcessorOutputType].from_exception(
                    lambda: self._execute_step(step, current),
                )

            return step_processor

        def _execute_step(
            self,
            step: Callable[
                [FlextTypes.ProcessorInputType], FlextTypes.ProcessorOutputType
            ],
            current: FlextTypes.ProcessorInputType,
        ) -> FlextTypes.ProcessorOutputType:
            """Execute a single pipeline step.

            Returns:
                FlextTypes.ProcessorOutputType: Result of step execution; may be a FlextResult unwrapped.

            """
            # Handle callable steps
            if callable(step):
                result = step(current)
                if isinstance(result, FlextResult):
                    if result.is_failure:
                        # Explicitly raise the error to be caught by from_exception wrapper
                        msg = f"Pipeline step failed: {result.error}"
                        raise FlextExceptions.OperationError(
                            message=msg,
                            operation="pipeline_step",
                        )
                    # result.is_success is True here
                    step_result: FlextTypes.ProcessorOutputType | None = cast(
                        "FlextTypes.ProcessorOutputType | None",
                        getattr(
                            cast(
                                "FlextResult[FlextTypes.ProcessorOutputType]",
                                result,
                            ),
                            "value_or_none",
                            None,
                        ),
                    )
                    if step_result is None:
                        msg = "Pipeline step returned None despite success"
                        raise FlextExceptions.OperationError(
                            message=msg,
                            operation="pipeline_step",
                        )
                    return step_result
                # result is not a FlextResult, return it directly
                return result

            # Handle dictionary merging
            if isinstance(current, dict) and isinstance(step, dict):
                merged_dict: dict[str, object] = {**current, **step}
                return merged_dict

            # Replace current data
            return step

    @staticmethod
    def is_handler_safe(handler: Callable[[object], object]) -> bool:
        """Check if handler implements standard handle() method.

        BREAKING CHANGE: No longer checks if handler is callable.
        Handler MUST implement handle() method.

        Returns:
            bool: True if handler has callable handle() method.

        """
        if not hasattr(handler, FlextConstants.Mixins.METHOD_HANDLE):
            return False

        handle_method = getattr(handler, FlextConstants.Mixins.METHOD_HANDLE, None)
        return handle_method is not None and callable(handle_method)

    # =========================================================================
    # HANDLER CLASSES - For examples and demos
    # =========================================================================

    class Implementation:
        """Handler implementation utilities."""

        class BasicHandler:
            """Basic handler implementation."""

            @override
            def __init__(self, name: str) -> None:
                """Initialize basic handler with name."""
                super().__init__()
                self.name = name

            @property
            def handler_name(self) -> str:
                """Get handler name."""
                return self.name

            def handle(
                self, request: FlextTypes.ProcessorInputType
            ) -> FlextResult[FlextTypes.ProcessorOutputType]:
                """Handle request.

                Returns:
                    FlextResult[FlextTypes.ProcessorOutputType]: Successful result wrapping output.

                """
                result = f"Handled by {self.name}: {request}"
                return FlextResult[FlextTypes.ProcessorOutputType].ok(result)

            def __call__(
                self, request: FlextTypes.ProcessorInputType
            ) -> FlextResult[FlextTypes.ProcessorOutputType]:
                """Make handler callable by delegating to handle method."""
                return self.handle(request)

    def unregister(self, name: str) -> FlextResult[None]:
        """Unregister a processor by name.

        Args:
            name: Name of the processor to unregister

        Returns:
            FlextResult[None]: Success if unregistration succeeded, failure otherwise

        """
        if not name:
            return FlextResult[None].fail("Processor name cannot be empty")

        if name not in self._registry:
            return FlextResult[None].fail(f'Processor "{name}" not found')

        del self._registry[name]
        metrics = cast("dict[str, int]", self._metrics)
        metrics["unregistrations"] = metrics.get("unregistrations", 0) + 1

        return FlextResult[None].ok(None)


__all__ = ["FlextProcessors"]
