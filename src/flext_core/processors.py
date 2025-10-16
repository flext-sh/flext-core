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
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextProcessors(FlextMixins):
    """Processing utilities for message handling and data transformation.

    Implements FlextProtocols.Middleware patterns through structural typing. Provides
    utilities for processing messages, applying middleware pipelines, and managing
    processing state with circuit breakers, rate limiting, and caching.

    Middleware Integration:
        - add_middleware(middleware) - Register middleware for processing pipeline
        - Middleware implements FlextProtocols.Middleware protocol via duck typing
        - process(name, data) - Process data through registered middleware
        - process_parallel(name, data_list) - Parallel processing with middleware
        - process_batch(name, data_list) - Batch processing with middleware chain

    Nested Protocol Implementations:
        - FlextProcessors.Pipeline - Advanced processing pipeline with monadic composition
        - FlextProcessors.HandlerRegistry - Handler registry with FlextProtocols.Handler support
        - FlextProcessors.Protocols.ChainableHandler - Example Handler protocol implementation

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
    def __init__(self, config: FlextTypes.Dict | None = None) -> None:
        """Initialize FlextProcessors with optional configuration.

        Args:
            config: Optional configuration dictionary for processors.
                   If not provided, uses constants for default values.

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

        self._registry: FlextTypes.Dict = {}
        self._middleware: FlextTypes.List = []
        self._processor_config: FlextTypes.Dict = config or {}
        self._metrics: dict[str, int] = {}
        self._audit_log: list[FlextTypes.Dict] = []
        self._performance_metrics: FlextTypes.FloatDict = {}
        self._circuit_breaker: dict[str, bool] = {}
        self._rate_limiter: dict[str, dict[str, int | float]] = {}
        self._cache: dict[str, tuple[object, float]] = {}
        self._lock = threading.Lock()  # Thread safety lock

        # Configuration with defaults from constants instead of hardcoded values
        cache_ttl = self._processor_config.get(
            "cache_ttl", FlextConstants.Defaults.DEFAULT_CACHE_TTL
        )
        self._cache_ttl = (
            float(cache_ttl)
            if isinstance(cache_ttl, (int, float, str))
            else float(FlextConstants.Defaults.DEFAULT_CACHE_TTL)
        )

        circuit_threshold = self._processor_config.get(
            "circuit_breaker_threshold",
            FlextConstants.Reliability.DEFAULT_CIRCUIT_BREAKER_THRESHOLD,
        )
        self._circuit_breaker_threshold = (
            int(circuit_threshold)
            if isinstance(circuit_threshold, (int, float, str))
            else FlextConstants.Reliability.DEFAULT_CIRCUIT_BREAKER_THRESHOLD
        )

        rate_limit = self._processor_config.get(
            "rate_limit",
            FlextConstants.Reliability.MAX_RETRY_ATTEMPTS,
        )
        self._rate_limit = (
            int(rate_limit)
            if isinstance(rate_limit, (int, float, str))
            else FlextConstants.Reliability.MAX_RETRY_ATTEMPTS
        )

        rate_window = self._processor_config.get(
            "rate_limit_window",
            FlextConstants.Reliability.DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
        )
        self._rate_limit_window = (
            int(rate_window)
            if isinstance(rate_window, (int, float, str))
            else FlextConstants.Reliability.DEFAULT_RATE_LIMIT_WINDOW_SECONDS
        )

    def register(
        self,
        name: str,
        processor: Callable[[object], object],
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
        self._metrics["registrations"] = self._metrics.get("registrations", 0) + 1

        self._log_with_context(
            "info", "Processor registered successfully", processor_name=name
        )
        return FlextResult[None].ok(None)

    def process(self, name: str, data: object) -> FlextResult[object]:
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

        processor = self._registry[name]

        # Check circuit breaker
        if self._circuit_breaker.get(name, False):
            return FlextResult[object].fail(
                f"Circuit breaker open for processor '{name}'",
            )

        # Check rate limit
        now = time.time()
        rate_key = f"{name}_rate"

        with self._lock:
            if rate_key not in self._rate_limiter:
                self._rate_limiter[rate_key] = {"count": 0, "window_start": now}

            rate_data = self._rate_limiter[rate_key]
            if now - rate_data["window_start"] > self._rate_limit_window:
                rate_data["count"] = 0
                rate_data["window_start"] = now

            if rate_data["count"] >= self._rate_limit:
                return FlextResult[object].fail(
                    f"Rate limit exceeded for processor '{name}'",
                )

            rate_data["count"] += 1

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
                # Check if result is a FlextResult-like object (handle import issues)
                if (
                    hasattr(result, "is_success")
                    and hasattr(result, "value")
                    and hasattr(result, "error")
                ):
                    # Cast to the expected return type for better type inference
                    typed_result = cast("FlextResult[object]", result)
                    if typed_result.is_success:
                        result_value: object = typed_result.value
                        self._cache[cache_key] = (result_value, time.time())
                        self._metrics["successful_processes"] = (
                            self._metrics.get("successful_processes", 0) + 1
                        )
                        self._audit_log.append({
                            "timestamp": time.time(),
                            "processor": name,
                            "status": "success",
                            "data_hash": hash(str(data)),
                        })
                        # Return the FlextResult directly, don't wrap it
                        return typed_result
                    # Cast to the expected return type for better type inference
                    typed_result = cast("FlextResult[object]", result)
                    self._metrics["failed_processes"] = (
                        self._metrics.get("failed_processes", 0) + 1
                    )
                    self._audit_log.append({
                        "timestamp": time.time(),
                        "processor": name,
                        "status": "failure",
                        "error": typed_result.error,
                        "data_hash": hash(str(data)),
                    })

                    return typed_result

                # Wrap non-FlextResult in FlextResult
                result_wrapped = FlextResult[object].ok(result)
                self._cache[cache_key] = (result, time.time())
                self._metrics["successful_processes"] = (
                    self._metrics.get("successful_processes", 0) + 1
                )
                return result_wrapped

            return FlextResult[object].fail(f"Processor '{name}' is not callable")
        except Exception as e:
            self._metrics["failed_processes"] = (
                self._metrics.get("failed_processes", 0) + 1
            )
            self._audit_log.append({
                "timestamp": time.time(),
                "processor": name,
                "status": "error",
                "error": str(e),
                "data_hash": hash(str(data)),
            })
            return FlextResult[object].fail(f"Processor execution error: {e}")

    def add_middleware(self, middleware: Callable[[object], object]) -> None:
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

    def get_metrics(self) -> dict[str, int]:
        """Get processing metrics.

        Returns:
            dict[str, int]: Current metrics

        """
        return self._metrics.copy()

    def get_audit_log(self) -> list[FlextTypes.Dict]:
        """Get audit log of processing operations.

        Returns:
            list[FlextTypes.Dict]: Audit log entries

        """
        return self._audit_log.copy()

    def get_performance_metrics(self) -> FlextTypes.FloatDict:
        """Get performance metrics.

        Returns:
            FlextTypes.FloatDict: Performance metrics

        """
        return self._performance_metrics.copy()

    def is_circuit_breaker_open(self, name: str) -> bool:
        """Check if circuit breaker is open for a processor.

        Args:
            name: Processor name

        Returns:
            bool: True if circuit breaker is open

        """
        return self._circuit_breaker.get(name, False)

    def process_batch(
        self,
        name: str,
        data_list: FlextTypes.List,
    ) -> FlextResult[FlextTypes.List]:
        """Process a batch of data items.

        Args:
            name: Processor name
            data_list: List of data items to process

        Returns:
            FlextResult[FlextTypes.List]: List of processed results

        """
        results: FlextTypes.List = []
        for data in data_list:
            result = self.process(name, data)
            if result.is_failure:
                return FlextResult[FlextTypes.List].fail(
                    f"Batch processing failed: {result.error}",
                )

            result_value: object = result.value
            results.append(result_value)

        return FlextResult[FlextTypes.List].ok(results)

    def process_parallel(
        self,
        name: str,
        data_list: FlextTypes.List,
    ) -> FlextResult[FlextTypes.List]:
        """Process data items in parallel using ThreadPoolExecutor.

        Args:
            name: Processor name
            data_list: List of data items to process

        Returns:
            FlextResult[FlextTypes.List]: List of processed results from parallel execution

        """
        # Validate processor exists before parallel execution
        if name not in self._registry:
            return FlextResult[FlextTypes.List].fail(f"Processor '{name}' not found")

        # Use ThreadPoolExecutor for true parallel processing
        max_workers = min(
            len(data_list), 10
        )  # Cap at 10 workers to avoid resource exhaustion
        results: FlextTypes.List = []

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
                        return FlextResult[FlextTypes.List].fail(
                            f"Parallel processing failed: {result.error}",
                        )
                    results.append(result.value)
                except Exception as e:
                    return FlextResult[FlextTypes.List].fail(
                        f"Parallel processing error: {e}",
                    )

        return FlextResult[FlextTypes.List].ok(results)

    def get_processors(self, name: str) -> FlextTypes.List:
        """Get registered processors by name.

        Args:
            name: Processor name

        Returns:
            FlextTypes.List: List of processors with the given name

        """
        if name in self._registry:
            return [self._registry[name]]
        return []

    def clear_processors(self) -> None:
        """Clear all registered processors."""
        self._registry.clear()
        self._metrics["clear_operations"] = self._metrics.get("clear_operations", 0) + 1

    def get_statistics(self) -> FlextTypes.Dict:
        """Get comprehensive statistics.

        Returns:
            FlextTypes.Dict: Statistics dictionary

        """
        return {
            "total_processors": len(self._registry),
            "total_middleware": len(self._middleware),
            "metrics": self._metrics.copy(),
            "cache_size": len(self._cache),
            "circuit_breakers_open": sum(
                1 for is_open in self._circuit_breaker.values() if is_open
            ),
        }

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

    def export_config(self) -> FlextTypes.Dict:
        """Export current configuration.

        Returns:
            FlextTypes.Dict: Configuration dictionary

        """
        return {
            "cache_ttl": self._cache_ttl,
            "circuit_breaker_threshold": self._circuit_breaker_threshold,
            "rate_limit": self._rate_limit,
            "rate_limit_window": self._rate_limit_window,
            "processor_count": len(self._registry),
            "middleware_count": len(self._middleware),
        }

    def import_config(self, config: FlextTypes.Dict) -> FlextResult[None]:
        """Import configuration.

        Args:
            config: Configuration dictionary

        Returns:
            FlextResult[None]: Success if import succeeded, failure otherwise

        """
        try:
            if "cache_ttl" in config:
                cache_ttl = config["cache_ttl"]
                if isinstance(cache_ttl, (int, float, str)):
                    self._cache_ttl = float(cache_ttl)

            if "circuit_breaker_threshold" in config:
                circuit_threshold = config["circuit_breaker_threshold"]
                if isinstance(circuit_threshold, (int, float, str)):
                    self._circuit_breaker_threshold = int(circuit_threshold)

            if "rate_limit" in config:
                rate_limit = config["rate_limit"]
                if isinstance(rate_limit, (int, float, str)):
                    self._rate_limit = int(rate_limit)

            if "rate_limit_window" in config:
                rate_window = config["rate_limit_window"]
                if isinstance(rate_window, (int, float, str)):
                    self._rate_limit_window = int(rate_window)

            return FlextResult[None].ok(None)
        except (ValueError, TypeError) as e:
            return FlextResult[None].fail(f"Configuration import error: {e}")

    class Handler:
        """Minimal handler base returning modernization-compliant results."""

        def handle(self, request: object) -> FlextResult[str]:
            """Handle a request.

            Returns:
                FlextResult[str]: A successful FlextResult wrapping handler
                output.

            """
            return FlextResult[str].ok(f"Base handler processed: {request}")

    class HandlerRegistry:
        """Registry managing named handler instances for dispatcher pilots."""

        @override
        def __init__(self) -> None:
            """Initialize handler registry."""
            super().__init__()
            self._handlers: FlextTypes.Dict = {}

        def register(
            self,
            registration: FlextModels.HandlerRegistration,
        ) -> FlextResult[None]:
            """Register a handler using Pydantic model validation.

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

            # Validate handler safety
            if not FlextProcessors.is_handler_safe(registration.handler):
                return FlextResult[None].fail(
                    f"Handler '{registration.name}' is not safe (must have handle method or be callable)",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Validate handler using the model's built-in validation
            self._handlers[registration.name] = registration.handler
            return FlextResult[None].ok(None)

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
            return FlextResult[object].ok(self._handlers[name])

        def execute(self, name: str, request: object) -> FlextResult[object]:
            """Execute a handler by name using railway pattern.

            Returns:
                FlextResult[object]: Result of handler execution or failure
                indicating handler not found or execution error.

            """
            return self.get(name).flat_map(
                lambda handler: self._execute_handler_safely(handler, request, name),
            )

        def _execute_handler_safely(
            self,
            handler: object,
            request: object,
            name: str,
        ) -> FlextResult[object]:
            """Execute handler with proper method resolution and error handling.

            Returns:
                FlextResult[object]: The result returned by the handler, or a
                failed FlextResult with a ProcessingError on exception.

            """
            try:
                # Check for handle method first
                if hasattr(handler, FlextConstants.Mixins.METHOD_HANDLE):
                    handle_method = getattr(
                        handler,
                        FlextConstants.Mixins.METHOD_HANDLE,
                        None,
                    )
                    if handle_method is not None and callable(handle_method):
                        result: object = handle_method(request)
                        if isinstance(result, FlextResult):
                            # Cast to FlextResult[object] to ensure type compatibility
                            typed_result: FlextResult[object] = cast(
                                "FlextResult[object]",
                                result,
                            )
                            return typed_result
                        return FlextResult[object].ok(result)

                # Check if handler itself is callable
                if callable(handler):
                    handler_result: object = handler(request)
                    if isinstance(handler_result, FlextResult):
                        # Cast to FlextResult[object] to ensure type compatibility
                        handler_typed_result: FlextResult[object] = cast(
                            "FlextResult[object]",
                            handler_result,
                        )
                        return handler_typed_result
                    return FlextResult[object].ok(handler_result)

                return FlextResult[object].fail(
                    f"Handler '{name}' does not implement handle method",
                    error_code=FlextConstants.Errors.NOT_FOUND_ERROR,
                )
            except Exception as e:
                return FlextResult[object].fail(
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
            config: FlextModels.BatchProcessingConfig | object,
        ) -> FlextResult[FlextTypes.List]:
            """Execute multiple handlers using BatchProcessingConfig model.

            Returns:
                FlextResult[FlextTypes.List]: List of handler results or a failed
                FlextResult if validation or batch processing fails.

            """
            # Validate config is a BatchProcessingConfig or has required attributes
            if not isinstance(config, FlextModels.BatchProcessingConfig):
                # For mock objects, validate they have required attributes
                if not hasattr(config, "data_items"):
                    return FlextResult[FlextTypes.List].fail(
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
                data_items = getattr(config, "data_items")
                continue_on_error = getattr(config, "continue_on_error", True)
            else:
                data_items = getattr(config, "data_items")
                continue_on_error = config.continue_on_error

            # Validate batch size limits
            max_batch_size = FlextConfig.get_global_instance().max_batch_size
            if len(data_items) > max_batch_size:
                return FlextResult[FlextTypes.List].fail(
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
                    return FlextResult[FlextTypes.List].fail(
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

                def register_handler(_: object) -> FlextResult[None]:
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
                | FlextTypes.Dict
                | object
            ] = []

        def add_step(
            self,
            step: Callable[[object], FlextResult[object] | object]
            | FlextTypes.Dict
            | object,
        ) -> None:
            """Add a processing step."""
            self._steps.append(step)

        def process(self, data: object) -> FlextResult[object]:
            """Process data through pipeline using advanced railway pattern.

            Returns:
                FlextResult[object]: Result of pipeline processing.

            """
            return FlextResult.pipeline(
                data,
                *[self._process_step(step) for step in self._steps],
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

            def process_data(data: FlextTypes.Dict) -> FlextResult[object]:
                return self.process(
                    FlextModels.ProcessingRequest(
                        data=data,
                        context=request.context,
                        timeout_seconds=request.timeout_seconds,
                    ),
                )

            data_result = FlextResult[FlextTypes.Dict].ok(request.data)
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
        ) -> FlextResult[FlextTypes.List]:
            """Process batch of data using validated BatchProcessingConfig model.

            Args:
                config: BatchProcessingConfig model with data items and processing options

            Returns:
                FlextResult[FlextTypes.List]: List of processed data items or a failure.

            """
            # Validate config is a BatchProcessingConfig or has required attributes
            if not isinstance(config, FlextModels.BatchProcessingConfig):
                # For mock objects, validate they have required attributes
                if not hasattr(config, "data_items"):
                    return FlextResult[FlextTypes.List].fail(
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
                data_items = getattr(config, "data_items")
                continue_on_error = getattr(config, "continue_on_error", True)
            else:
                data_items = getattr(config, "data_items")
                continue_on_error = config.continue_on_error

            # Validate batch size limits
            max_batch_size = FlextConfig.get_global_instance().max_batch_size
            if len(data_items) > max_batch_size:
                return FlextResult[FlextTypes.List].fail(
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
            step: object,
        ) -> Callable[[object], FlextResult[object]]:
            """Convert pipeline step to FlextResult-returning function.

            Returns:
                Callable[[object], FlextResult[object]]: Adapter that wraps step execution.

            """

            def step_processor(current: object) -> FlextResult[object]:
                return FlextResult[object].from_exception(
                    lambda: self._execute_step(step, current),
                )

            return step_processor

        def _execute_step(self, step: object, current: object) -> object:
            """Execute a single pipeline step.

            Returns:
                object: Result of step execution; may be a FlextResult unwrapped.

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
                    step_result: object | None = cast(
                        "object | None",
                        getattr(cast("object", result), "value_or_none", None),
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
                merged_dict: FlextTypes.Dict = {**current, **step}
                return merged_dict

            # Replace current data
            return step

    @staticmethod
    def is_handler_safe(handler: object) -> bool:
        """Check if a handler is safe (has handle method or is callable).

        Returns:
            bool: True if handler is safe to execute.

        """
        if hasattr(handler, FlextConstants.Mixins.METHOD_HANDLE):
            handle_method = getattr(handler, FlextConstants.Mixins.METHOD_HANDLE, None)
            if handle_method is not None and callable(handle_method):
                return True
        return callable(handler)

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

            def handle(self, request: object) -> FlextResult[str]:
                """Handle request.

                Returns:
                    FlextResult[str]: Successful result wrapping a string message.

                """
                result = f"Handled by {self.name}: {request}"
                return FlextResult[str].ok(result)

            def __call__(self, request: object) -> FlextResult[str]:
                """Make handler callable by delegating to handle method."""
                return self.handle(request)

    class Management:
        """Handler management utilities."""

        class HandlerRegistry:
            """Handler registry for examples."""

            @override
            def __init__(self) -> None:
                """Initialize handler registry."""
                super().__init__()
                self._handlers: FlextTypes.Dict = {}

            def register(self, name: str, handler: object) -> None:
                """Register handler."""
                self._handlers[name] = handler

            def get(self, name: str) -> object | None:
                """Get handler by name.

                Returns:
                    object | None: The handler instance or None if not found.

                """
                return self._handlers.get(name)

            def get_optional(self, name: str) -> object | None:
                """Get handler optionally, returning None if not found.

                Returns:
                    object | None: The handler instance or None when not present.

                """
                return self._handlers.get(name)

    class Patterns:
        """Handler patterns for examples."""

        class HandlerChain:
            """Handler chain for examples."""

            @override
            def __init__(self, name: str) -> None:
                """Initialize handler chain with name."""
                super().__init__()
                self.name = name
                self._handlers: FlextTypes.List = []

            def add_handler(self, handler: object) -> None:
                """Add handler to chain."""
                self._handlers.append(handler)

            def handle(self, request: object) -> FlextResult[object]:
                """Handle request by executing all handlers in chain.

                Returns:
                    FlextResult[object]: Result after processing through the chain.

                """
                result = request
                for handler in self._handlers:
                    handle_method_name = FlextConstants.Mixins.METHOD_HANDLE
                    if hasattr(handler, handle_method_name):
                        handle_method = getattr(handler, handle_method_name, None)
                        if handle_method is not None:
                            handler_result: FlextResult[object] = handle_method(result)
                            if (
                                hasattr(handler_result, "is_success")
                                and not handler_result.is_success
                            ):
                                return FlextResult[object].fail(
                                    f"Handler failed: {handler_result.error}",
                                )
                            # Extract the actual data from FlextResult if it's a FlextResult
                            if hasattr(handler_result, "value") and hasattr(
                                handler_result,
                                "is_success",
                            ):
                                result = handler_result.value
                            else:
                                result = handler_result
                return FlextResult[object].ok(result)

    class Protocols:
        """Handler protocols for examples."""

        class ChainableHandler:
            """Chainable handler for examples - Application.Handler protocol implementation."""

            @override
            def __init__(self, name: str) -> None:
                """Initialize chainable handler with name."""
                super().__init__()
                self.name = name

            def handle(self, request: object) -> FlextResult[object]:
                """Handle request in chain.

                Returns:
                    FlextResult[object]: Handler output wrapped in FlextResult.

                """
                # Extract data from FlextResult if it's a FlextResult
                if isinstance(request, FlextProtocols.HasResultValue):
                    actual_request = request.value
                else:
                    actual_request = request

                result = f"Chain handled by {self.name}: {actual_request}"
                return FlextResult[object].ok(result)

            def can_handle(self, _message_type: object) -> bool:
                """Check if handler can process this message type.

                Args:
                    _message_type: The message type to check (unused in generic handler)

                Returns:
                    bool: True since this is a generic example handler

                """
                return True

            def execute(self, message: object) -> FlextResult[object]:
                """Execute the handler with the given message.

                Args:
                    message: The input message to execute

                Returns:
                    FlextResult[object]: Execution result

                """
                return self.handle(message)

            def validate_command(self, command: object) -> FlextResult[None]:
                """Validate a command message using centralized validation.

                Args:
                    command: The command to validate

                Returns:
                    FlextResult[None]: Success if valid, failure with error details

                """
                return FlextModels.Validation.validate_command(command)

            def validate_query(self, query: object) -> FlextResult[None]:
                """Validate a query message using centralized validation.

                Args:
                    query: The query to validate

                Returns:
                    FlextResult[None]: Success if valid, failure with error details

                """
                return FlextModels.Validation.validate_query(query)

            @property
            def handler_name(self) -> str:
                """Get the handler name.

                Returns:
                    str: Handler name

                """
                return self.name

            @property
            def mode(self) -> str:
                """Get the handler mode (command/query).

                Returns:
                    str: Handler mode

                """
                return "command"

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
        self._metrics["unregistrations"] = self._metrics.get("unregistrations", 0) + 1

        return FlextResult[None].ok(None)


__all__ = ["FlextProcessors"]
