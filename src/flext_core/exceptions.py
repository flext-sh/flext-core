"""Layer 2: Exception hierarchy aligned with the FLEXT 1.0.0 modernization charter.

This module provides structured exception classes with error codes and correlation
tracking for the entire FLEXT ecosystem. Use FlextException hierarchy for all
error handling throughout FLEXT applications.

Dependency Layer: 2 (Early Foundation)
Dependencies: FlextConstants, FlextTypes
Used by: All FlextCore modules requiring structured error handling

Error codes, correlation tracking, and structured payloads match the guidance
in ``README.md`` and ``docs/architecture.md`` so diagnostics remain uniform
across packages.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from collections.abc import Mapping
from typing import ClassVar, cast, override

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextExceptions:
    """Hierarchical exception system with modernization-ready diagnostics.

    Factory helpers create structured errors and record metrics so dispatcher
    flows, domain services, and configuration loaders surface consistent
    failures throughout the 1.0.0 rollout.
    """

    def __init__(self, config: dict[str, object] | None = None) -> None:
        """Initialize FlextExceptions with configuration.

        Args:
            config: Optional configuration dictionary

        """
        self._config = config or {}
        self._handlers: dict[type[Exception], list[object]] = {}
        self._middleware: list[object] = []
        self._audit_log: list[dict[str, object]] = []
        self._performance_metrics: dict[str, dict[str, int]] = {}
        self._circuit_breakers: dict[str, bool] = {}
        self.logger = logging.getLogger(__name__)
        self._rate_limiters: dict[str, dict[str, object]] = {}
        self._cache: dict[str, object] = {}

    def __call__(
        self,
        message: str,
        *,
        operation: str | None = None,  # Operation context for error tracking
        field: str | None = None,  # Field name for validation errors
        config_key: str | None = None,  # Configuration key for config errors
        error_code: str | None = None,  # Error code for categorization
        **kwargs: object,
    ) -> FlextExceptions.BaseError:
        """Allow FlextExceptions() to be called directly."""
        # Custom __call__ method to make exceptions callable
        return self.create(
            message,
            operation=operation,
            field=field,
            config_key=config_key,
            error_code=error_code,
            **kwargs,
        )

    # =============================================================================
    # Metrics Domain: Exception metrics and monitoring functionality
    # =============================================================================

    _metrics: ClassVar[FlextTypes.Core.CounterDict] = {}

    # Type conversion mapping for cleaner type resolution
    _TYPE_MAP: ClassVar[dict[str, type]] = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
    }

    @classmethod
    def record_exception(cls, exception_type: str) -> None:
        """Record exception occurrence."""
        cls._metrics[exception_type] = cls._metrics.get(exception_type, 0) + 1

    @classmethod
    def get_metrics(cls) -> FlextTypes.Core.CounterDict:
        """Get exception counts."""
        return dict(cls._metrics)

    @classmethod
    def clear_metrics(cls) -> None:
        """Clear all exception metrics."""
        cls._metrics.clear()

    # =============================================================================
    # Handler Registry Methods
    # =============================================================================

    def register_handler(
        self, exception_type: type[Exception], handler: object
    ) -> FlextResult[None]:
        """Register exception handler for specific exception type.

        Args:
            exception_type: Type of exception to handle
            handler: Handler function or object

        Returns:
            FlextResult with success or failure

        """
        try:
            # Validate that exception_type is actually a type that inherits from Exception
            if not isinstance(exception_type, type) or not issubclass(
                exception_type, Exception
            ):
                return FlextResult[None].fail(
                    f"Invalid exception type: {exception_type}"
                )

            if exception_type not in self._handlers:
                self._handlers[exception_type] = []
            self._handlers[exception_type].append(handler)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Failed to register handler: {e}")

    def unregister_handler(
        self, exception_type: type[Exception], handler: object
    ) -> FlextResult[None]:
        """Unregister exception handler.

        Args:
            exception_type: Type of exception
            handler: Handler to remove

        Returns:
            FlextResult with success or failure

        """
        try:
            if (
                exception_type in self._handlers
                and handler in self._handlers[exception_type]
            ):
                self._handlers[exception_type].remove(handler)
                if not self._handlers[exception_type]:
                    del self._handlers[exception_type]
                return FlextResult[None].ok(None)
            return FlextResult[None].fail("Handler not found")
        except Exception as e:
            return FlextResult[None].fail(f"Failed to unregister handler: {e}")

    def handle_exception(
        self, exception: Exception, correlation_id: str | None = None
    ) -> FlextResult[object]:
        """Handle exception using registered handlers.

        Args:
            exception: Exception to handle
            correlation_id: Optional correlation ID

        Returns:
            FlextResult with handling result

        """
        try:
            start_time = time.time()
            exception_type = type(exception)
            handlers = self._handlers.get(exception_type, [])

            if not handlers:
                return FlextResult[object].fail("No handler found")

            # Check circuit breaker
            exception_type_name = exception_type.__name__
            if self.is_circuit_breaker_open(exception_type_name):
                return FlextResult[object].fail("Circuit breaker is open")

            # Check rate limiting
            rate_limit = self._config.get("rate_limit", 10)  # Default 10 requests
            rate_limit_window = self._config.get(
                "rate_limit_window", 60
            )  # Default 60 seconds

            current_time = time.time()
            rate_limit_key = f"{exception_type_name}_rate_limit"

            if rate_limit_key not in self._rate_limiters:
                self._rate_limiters[rate_limit_key] = {
                    "requests": [],
                    "window_start": current_time,
                }

            rate_limiter = self._rate_limiters[rate_limit_key]

            # Clean old requests outside the window
            rate_limiter["requests"] = [
                req_time
                for req_time in rate_limiter["requests"]
                if current_time - req_time < rate_limit_window
            ]

            # Check if rate limit exceeded
            if len(rate_limiter["requests"]) >= rate_limit:
                return FlextResult[object].fail("Rate limit exceeded")

            # Add current request
            rate_limiter["requests"].append(current_time)

            # Apply middleware
            processed_exception = exception
            for middleware in self._middleware:
                if callable(middleware):
                    try:
                        processed_exception = middleware(processed_exception)
                    except Exception as middleware_error:
                        self.logger.warning(f"Middleware error: {middleware_error}")

            # Execute handlers with timeout
            timeout_seconds = self._config.get("timeout", 30)  # Default 30 seconds
            timeout_value = (
                float(timeout_seconds) if isinstance(timeout_seconds, (int, float, str)) else 30.0
            )

            results = []
            handler_exceptions = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(handlers)) as executor:
                # Submit all handlers
                future_to_handler = {
                    executor.submit(handler, processed_exception): handler
                    for handler in handlers
                    if callable(handler)
                }

                # Wait for results with timeout
                try:
                    for future in concurrent.futures.as_completed(future_to_handler, timeout=timeout_value):
                        try:
                            handler_result = future.result()
                            results.append(handler_result)
                        except Exception as handler_error:
                            handler_exceptions.append(str(handler_error))
                            results.append(f"Handler error: {handler_error}")
                except concurrent.futures.TimeoutError:
                    return FlextResult[object].fail("Handler timeout")

                # Add non-callable handlers
                non_callable_handlers = [handler for handler in handlers if not callable(handler)]
                results.extend(non_callable_handlers)

            # If any handler raised an exception, return failure
            if handler_exceptions:
                return FlextResult[object].fail(
                    f"Handler exception: {handler_exceptions[0]}"
                )

            # Record in audit log
            self._audit_log.append({
                "exception_type": exception_type.__name__,
                "exception_message": str(exception),
                "handlers_executed": len(handlers),
                "correlation_id": correlation_id,
                "timestamp": time.time(),
            })

            # Update performance metrics
            exception_name = exception_type.__name__
            execution_time = time.time() - start_time

            if exception_name not in self._performance_metrics:
                self._performance_metrics[exception_name] = {
                    "handled": 0,
                    "errors": 0,
                    "total_execution_time": 0.0,
                    "execution_count": 0,
                    "avg_execution_time": 0.0
                }

            # Update execution time metrics
            self._performance_metrics[exception_name]["total_execution_time"] += execution_time
            self._performance_metrics[exception_name]["execution_count"] += 1
            self._performance_metrics[exception_name]["avg_execution_time"] = (
                self._performance_metrics[exception_name]["total_execution_time"] /
                self._performance_metrics[exception_name]["execution_count"]
            )

            # Check circuit breaker threshold
            threshold_value = self._config.get("circuit_breaker_threshold", 5)
            circuit_breaker_threshold = (
                int(threshold_value) if isinstance(threshold_value, (int, str)) else 5
            )

            # Count failures in results
            failure_count = sum(
                1
                for result in results
                if isinstance(result, FlextResult) and result.is_failure
            )
            if failure_count > 0:
                self._performance_metrics[exception_name]["errors"] += failure_count
            else:
                self._performance_metrics[exception_name]["handled"] += 1

            # Check circuit breaker threshold after updating metrics
            if (
                self._performance_metrics[exception_name]["errors"]
                >= circuit_breaker_threshold
            ):
                self._circuit_breakers[exception_name] = True
                return FlextResult[object].fail(
                    "Circuit breaker opened due to repeated failures"
                )

            # Check if any handler failed
            if any(isinstance(result, FlextResult) and result.is_failure for result in results):
                # Return the first failure
                for result in results:
                    if isinstance(result, FlextResult) and result.is_failure:
                        return result

            # Return single result if only one handler, otherwise return list
            if len(results) == 1:
                return results[0]
            return FlextResult[object].ok(results)
        except Exception as e:
            return FlextResult[object].fail(f"Exception handling failed: {e}")

    def handle_batch(self, exceptions: list[Exception]) -> list[FlextResult[object]]:
        """Handle multiple exceptions in batch.

        Args:
            exceptions: List of exceptions to handle

        Returns:
            List of FlextResults with handling results

        """
        return [self.handle_exception(exc) for exc in exceptions]

    def handle_parallel(self, exceptions: list[Exception]) -> list[FlextResult[object]]:
        """Handle multiple exceptions in parallel.

        Args:
            exceptions: List of exceptions to handle

        Returns:
            List of FlextResults with handling results

        """
        results = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(exceptions)
        ) as executor:
            futures = [
                executor.submit(self.handle_exception, exc) for exc in exceptions
            ]
            results.extend(
                future.result() for future in concurrent.futures.as_completed(futures)
            )

        return results

    def add_middleware(self, middleware: object) -> None:
        """Add middleware for exception processing.

        Args:
            middleware: Middleware function or object

        """
        self._middleware.append(middleware)

    def is_circuit_breaker_open(self, exception_type: str) -> bool:
        """Check if circuit breaker is open for exception type.

        Args:
            exception_type: Name of exception type

        Returns:
            True if circuit breaker is open

        """
        return self._circuit_breakers.get(exception_type, False)

    def get_audit_log(self) -> list[dict[str, object]]:
        """Get audit log of exception handling.

        Returns:
            List of audit log entries

        """
        return self._audit_log.copy()

    def get_performance_metrics(self) -> dict[str, dict[str, int]]:
        """Get performance metrics.

        Returns:
            Dictionary of performance metrics

        """
        return self._performance_metrics.copy()

    def get_handlers(self, exception_type: type[Exception]) -> list[object]:
        """Get handlers for specific exception type.

        Args:
            exception_type: Type of exception

        Returns:
            List of handlers

        """
        return self._handlers.get(exception_type, []).copy()

    def clear_handlers(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()

    def get_statistics(self) -> dict[str, object]:
        """Get statistics about exception handling.

        Returns:
            Dictionary of statistics

        """
        stats = {
            "total_handlers": sum(
                len(handlers) for handlers in self._handlers.values()
            ),
            "exception_types": len(self._handlers),
            "middleware_count": len(self._middleware),
            "audit_log_entries": len(self._audit_log),
            "performance_metrics": self._performance_metrics.copy(),
        }

        # Add individual exception type statistics
        stats.update(self._performance_metrics)

        return stats

    def validate(self) -> FlextResult[None]:
        """Validate configuration and handlers.

        Returns:
            FlextResult with validation result

        """
        try:
            # Validate handlers
            for exception_type, handlers in self._handlers.items():
                if not isinstance(exception_type, type):
                    return FlextResult[None].fail(
                        f"Invalid exception type: {exception_type}"
                    )
                if not isinstance(handlers, list):
                    return FlextResult[None].fail(
                        f"Invalid handlers list for {exception_type}"
                    )

            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Validation failed: {e}")

    def export_config(self) -> dict[str, object]:
        """Export configuration.

        Returns:
            Dictionary of configuration

        """
        return {
            "config": self._config.copy(),
            "handler_count": len(self._handlers),
            "middleware_count": len(self._middleware),
            "audit_log_size": len(self._audit_log),
        }

    def import_config(self, config: dict[str, object]) -> FlextResult[None]:
        """Import configuration.

        Args:
            config: Configuration dictionary

        Returns:
            FlextResult with import result

        """
        try:
            if "config" in config and isinstance(config["config"], dict):
                config_dict = cast("dict[str, object]", config["config"])
                self._config.update(config_dict)
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Config import failed: {e}")

    def cleanup(self) -> None:
        """Clean up resources."""
        self._handlers.clear()
        self._middleware.clear()
        self._audit_log.clear()
        self._performance_metrics.clear()
        self._circuit_breakers.clear()
        self._rate_limiters.clear()
        self._cache.clear()

    # =============================================================================
    # BASE EXCEPTION CLASS - Clean hierarchical approach
    # =============================================================================

    class BaseError(Exception):
        """Base exception with structured error handling."""

        def __init__(
            self,
            message: str,
            *,
            code: str | None = None,
            context: Mapping[str, object] | None = None,
            correlation_id: str | None = None,
        ) -> None:
            """Initialize structured exception."""
            super().__init__(message)
            self.message = message
            self.code = code or FlextConstants.Errors.GENERIC_ERROR
            self.context: dict[str, object] = dict(context or {})
            self.correlation_id = correlation_id or f"flext_{int(time.time() * 1000)}"
            self.timestamp = time.time()
            FlextExceptions.record_exception(self.__class__.__name__)

        @override
        def __str__(self) -> str:
            """Return string representation with error code and message."""
            return f"[{self.code}] {self.message}"

        @property
        def error_code(self) -> str:
            """Get error code as string."""
            return str(self.code)

        @staticmethod
        def _build_context(
            base_context: Mapping[str, object] | None,
            **additional_context: object,
        ) -> dict[str, object]:
            """Build context dictionary from base context and additional items."""
            context_dict: dict[str, object] = (
                dict(base_context) if isinstance(base_context, dict) else {}
            )
            context_dict.update(additional_context)
            return context_dict

        @staticmethod
        def _extract_common_kwargs(
            kwargs: dict[str, object],
        ) -> tuple[dict[str, object] | None, str | None, str | None]:
            """Extract common kwargs (context, correlation_id, error_code) from kwargs."""
            context_raw = kwargs.get("context")
            context = (
                cast("dict[str, object]", context_raw)
                if isinstance(context_raw, dict)
                else None
            )

            correlation_id_raw = kwargs.get("correlation_id")
            correlation_id = (
                str(correlation_id_raw) if correlation_id_raw is not None else None
            )

            error_code_raw = kwargs.get("error_code")
            error_code = str(error_code_raw) if error_code_raw is not None else None

            return context, correlation_id, error_code

        @staticmethod
        def _convert_type_name(type_name: str | None) -> type | str:
            """Convert type name string to actual type object."""
            if not type_name:
                return ""
            return FlextExceptions._TYPE_MAP.get(type_name, type_name)

    @staticmethod
    def _extract_common_kwargs(
        kwargs: dict[str, object],
    ) -> tuple[dict[str, object] | None, str | None, str | None]:
        """Extract common kwargs (context, correlation_id, error_code) from kwargs."""
        context_raw = kwargs.get("context")
        context = (
            cast("dict[str, object]", context_raw)
            if isinstance(context_raw, dict)
            else None
        )

        correlation_id_raw = kwargs.get("correlation_id")
        correlation_id = (
            str(correlation_id_raw) if correlation_id_raw is not None else None
        )

        error_code_raw = kwargs.get("error_code")
        error_code = str(error_code_raw) if error_code_raw is not None else None

        return context, correlation_id, error_code

    # =============================================================================
    # SPECIFIC EXCEPTION CLASSES - Clean subclass hierarchy
    # =============================================================================

    class _AttributeError(BaseError, AttributeError):
        """Attribute access failure with attribute context."""

        def __init__(
            self,
            message: str,
            *,
            attribute_name: str | None = None,
            attribute_context: Mapping[str, object] | None = None,
            **kwargs: object,
        ) -> None:
            self.attribute_name = attribute_name

            # Extract common parameters using helper
            base_context, correlation_id, _ = self._extract_common_kwargs(kwargs)

            # Build context with specific fields
            context_data: dict[str, object] = {"attribute_name": attribute_name}
            if attribute_context:
                context_data["attribute_context"] = dict(attribute_context)

            context = self._build_context(base_context, **context_data)

            super().__init__(
                message,
                code=FlextConstants.Errors.OPERATION_ERROR,
                context=context,
                correlation_id=correlation_id,
            )

    class _OperationError(BaseError, RuntimeError):
        """Generic operation failure."""

        def __init__(
            self,
            message: str,
            *,
            operation: str | None = None,
            **kwargs: object,
        ) -> None:
            self.operation = operation

            # Extract common parameters using helper
            base_context, correlation_id, _ = self._extract_common_kwargs(kwargs)

            # Build context with specific fields
            context = self._build_context(base_context, operation=operation)

            super().__init__(
                message,
                code=FlextConstants.Errors.OPERATION_ERROR,
                context=context,
                correlation_id=correlation_id,
            )

    class _ValidationError(BaseError, ValueError):
        """Data validation failure."""

        def __init__(
            self,
            message: str,
            *,
            field: str | None = None,
            value: object = None,
            validation_details: object = None,
            **kwargs: object,
        ) -> None:
            self.field = field
            self.value = value
            self.validation_details = validation_details

            # Extract common parameters using helper
            base_context, correlation_id, error_code = self._extract_common_kwargs(
                kwargs
            )

            # Build context with specific fields
            context = self._build_context(
                base_context,
                field=field,
                value=value,
                validation_details=validation_details,
            )

            super().__init__(
                message,
                code=error_code or FlextConstants.Errors.VALIDATION_ERROR,
                context=context,
                correlation_id=correlation_id,
            )

    class _ConfigurationError(BaseError, ValueError):
        """System configuration error."""

        def __init__(
            self,
            message: str,
            *,
            config_key: str | None = None,
            config_file: str | None = None,
            **kwargs: object,
        ) -> None:
            self.config_key = config_key
            self.config_file = config_file
            context_raw = kwargs.get("context", {})
            if isinstance(context_raw, dict):
                context_dict: dict[str, object] = cast("dict[str, object]", context_raw)
            else:
                context_dict = {}
            context_dict.update({"config_key": config_key, "config_file": config_file})
            super().__init__(
                message,
                code=FlextConstants.Errors.CONFIGURATION_ERROR,
                context=context_dict,
                correlation_id=str(kwargs.get("correlation_id"))
                if kwargs.get("correlation_id") is not None
                else None,
            )

    class _ConnectionError(BaseError, ConnectionError):
        """Network or service connection failure."""

        def __init__(
            self,
            message: str,
            *,
            service: str | None = None,
            endpoint: str | None = None,
            **kwargs: object,
        ) -> None:
            self.service = service
            self.endpoint = endpoint
            context_raw = kwargs.get("context", {})
            if isinstance(context_raw, dict):
                context_dict: dict[str, object] = cast("dict[str, object]", context_raw)
            else:
                context_dict = {}
            context_dict.update({"service": service, "endpoint": endpoint})
            super().__init__(
                message,
                code=FlextConstants.Errors.CONNECTION_ERROR,
                context=context_dict,
                correlation_id=str(kwargs.get("correlation_id"))
                if kwargs.get("correlation_id") is not None
                else None,
            )

    class _ProcessingError(BaseError, RuntimeError):
        """Business logic or data processing failure."""

        def __init__(
            self,
            message: str,
            *,
            business_rule: str | None = None,
            operation: str | None = None,
            **kwargs: object,
        ) -> None:
            self.business_rule = business_rule
            self.operation = operation

            # Extract common parameters using helper
            base_context, correlation_id, _ = self._extract_common_kwargs(kwargs)

            # Build context with specific fields
            context = self._build_context(
                base_context,
                business_rule=business_rule,
                operation=operation,
            )

            super().__init__(
                message,
                code=FlextConstants.Errors.PROCESSING_ERROR,
                context=context,
                correlation_id=correlation_id,
            )

    class _TimeoutError(BaseError, TimeoutError):
        """Operation timeout with timing context."""

        def __init__(
            self,
            message: str,
            *,
            timeout_seconds: float | None = None,
            **kwargs: object,
        ) -> None:
            self.timeout_seconds = timeout_seconds

            # Extract common parameters using helper
            base_context, correlation_id, _ = self._extract_common_kwargs(kwargs)

            # Build context with specific fields
            context = self._build_context(
                base_context,
                timeout_seconds=timeout_seconds,
            )

            super().__init__(
                message,
                code=FlextConstants.Errors.TIMEOUT_ERROR,
                context=context,
                correlation_id=correlation_id,
            )

    class _NotFoundError(BaseError, FileNotFoundError):
        """Resource not found."""

        def __init__(
            self,
            message: str,
            *,
            resource_id: str | None = None,
            resource_type: str | None = None,
            **kwargs: object,
        ) -> None:
            self.resource_id = resource_id
            self.resource_type = resource_type

            # Extract common parameters using helper
            base_context, correlation_id, _ = self._extract_common_kwargs(kwargs)

            # Build context with specific fields
            context = self._build_context(
                base_context,
                resource_id=resource_id,
                resource_type=resource_type,
            )

            super().__init__(
                message,
                code=FlextConstants.Errors.NOT_FOUND,
                context=context,
                correlation_id=correlation_id,
            )

    class _AlreadyExistsError(BaseError, FileExistsError):
        """Resource already exists."""

        def __init__(
            self,
            message: str,
            *,
            resource_id: str | None = None,
            resource_type: str | None = None,
            **kwargs: object,
        ) -> None:
            self.resource_id = resource_id
            self.resource_type = resource_type

            # Extract common parameters using helper
            base_context, correlation_id, _ = self._extract_common_kwargs(kwargs)

            # Build context with specific fields
            context = self._build_context(
                base_context,
                resource_id=resource_id,
                resource_type=resource_type,
            )

            super().__init__(
                message,
                code=FlextConstants.Errors.ALREADY_EXISTS,
                context=context,
                correlation_id=correlation_id,
            )

    class _PermissionError(BaseError, PermissionError):
        """Insufficient permissions."""

        def __init__(
            self,
            message: str,
            *,
            required_permission: str | None = None,
            **kwargs: object,
        ) -> None:
            self.required_permission = required_permission

            # Extract common parameters using helper
            base_context, correlation_id, _ = self._extract_common_kwargs(kwargs)

            # Build context with specific fields
            context = self._build_context(
                base_context,
                required_permission=required_permission,
            )

            super().__init__(
                message,
                code=FlextConstants.Errors.PERMISSION_ERROR,
                context=context,
                correlation_id=correlation_id,
            )

    class _AuthenticationError(BaseError, PermissionError):
        """Authentication failure."""

        def __init__(
            self,
            message: str,
            *,
            auth_method: str | None = None,
            **kwargs: object,
        ) -> None:
            self.auth_method = auth_method

            # Extract common parameters using helper
            base_context, correlation_id, _ = self._extract_common_kwargs(kwargs)

            # Build context with specific fields
            context = self._build_context(
                base_context,
                auth_method=auth_method,
            )

            super().__init__(
                message,
                code=FlextConstants.Errors.AUTHENTICATION_ERROR,
                context=context,
                correlation_id=correlation_id,
            )

    class _TypeError(BaseError, TypeError):
        """Type validation failure."""

        def __init__(
            self,
            message: str,
            *,
            expected_type: str | None = None,
            actual_type: str | None = None,
            **kwargs: object,
        ) -> None:
            self.expected_type = expected_type
            self.actual_type = actual_type

            # Extract common parameters using helper
            base_context, correlation_id, _ = self._extract_common_kwargs(kwargs)

            # Convert type names to actual types using helper
            expected_type_obj = self._convert_type_name(expected_type)
            actual_type_obj = self._convert_type_name(actual_type)

            # Build context with specific fields
            context = self._build_context(
                base_context,
                expected_type=expected_type_obj,
                actual_type=actual_type_obj,
            )

            super().__init__(
                message,
                code=FlextConstants.Errors.TYPE_ERROR,
                context=context,
                correlation_id=correlation_id,
            )

    class _CriticalError(BaseError, SystemError):
        """Critical system error."""

        def __init__(self, message: str, **kwargs: object) -> None:
            # Extract special parameters
            context_raw = kwargs.pop("context", None)
            if isinstance(context_raw, dict):
                base_context: dict[str, object] | None = cast(
                    "dict[str, object]", context_raw
                )
            else:
                base_context = None
            correlation_id_raw = kwargs.pop("correlation_id", None)
            correlation_id = (
                str(correlation_id_raw) if correlation_id_raw is not None else None
            )

            # Add remaining kwargs to context for full functionality
            context_dict: dict[str, object]
            if base_context is not None:
                context_dict = dict(base_context)
                context_dict.update(kwargs)
            elif kwargs:
                context_dict = dict(kwargs)
            else:
                context_dict = {}

            super().__init__(
                message,
                code=FlextConstants.Errors.CRITICAL_ERROR,
                context=context_dict,
                correlation_id=correlation_id,
            )

    class _Error(BaseError, RuntimeError):
        """Generic FLEXT error."""

        def __init__(self, message: str, **kwargs: object) -> None:
            # Extract special parameters
            context_raw = kwargs.pop("context", None)
            if isinstance(context_raw, dict):
                base_context: dict[str, object] | None = cast(
                    "dict[str, object]", context_raw
                )
            else:
                base_context = None
            correlation_id_raw = kwargs.pop("correlation_id", None)
            correlation_id = (
                str(correlation_id_raw) if correlation_id_raw is not None else None
            )
            error_code_raw = kwargs.pop("error_code", None)
            error_code = str(error_code_raw) if error_code_raw is not None else None

            # Add remaining kwargs to context for full functionality
            context_dict: dict[str, object]
            if base_context is not None:
                context_dict = dict(base_context)
                context_dict.update(kwargs)
            elif kwargs:
                context_dict = dict(kwargs)
            else:
                context_dict = {}

            super().__init__(
                message,
                code=error_code or FlextConstants.Errors.GENERIC_ERROR,
                context=context_dict,
                correlation_id=correlation_id,
            )

    class _UserError(BaseError, TypeError):
        """User input or API usage error."""

        def __init__(self, message: str, **kwargs: object) -> None:
            # Extract common parameters using helper
            base_context, correlation_id, _ = self._extract_common_kwargs(kwargs)

            super().__init__(
                message,
                code=FlextConstants.Errors.TYPE_ERROR,
                context=base_context or {},
                correlation_id=correlation_id,
            )

    # =============================================================================
    # PUBLIC API ALIASES - Real exception classes with clean names
    # =============================================================================

    AttributeError = _AttributeError
    OperationError = _OperationError
    ValidationError = _ValidationError
    ConfigurationError = _ConfigurationError
    ConnectionError = _ConnectionError
    ProcessingError = _ProcessingError
    TimeoutError = _TimeoutError
    NotFoundError = _NotFoundError
    AlreadyExistsError = _AlreadyExistsError
    PermissionError = _PermissionError
    AuthenticationError = _AuthenticationError
    TypeError = _TypeError
    CriticalError = _CriticalError
    Error = _Error
    UserError = _UserError

    # =============================================================================
    # DIRECT CALLABLE INTERFACE - For general usage
    # =============================================================================

    @classmethod
    def create(
        cls,
        message: str,
        *,
        operation: str | None = None,
        field: str | None = None,
        config_key: str | None = None,
        error_code: str | None = None,
        **kwargs: object,
    ) -> BaseError:
        """Create exception with automatic type selection."""
        # Extract common parameters using helper
        base_context, correlation_id, extracted_error_code = cls._extract_common_kwargs(
            kwargs
        )
        final_error_code = error_code or extracted_error_code

        # Exception type mapping for cleaner selection
        if operation is not None:
            return cls._OperationError(
                message,
                operation=operation,
                context=base_context,
                correlation_id=correlation_id,
                error_code=final_error_code,
            )

        if field is not None:
            return cls._ValidationError(
                message,
                field=field,
                value=kwargs.get("value"),
                validation_details=kwargs.get("validation_details"),
                context=base_context,
                correlation_id=correlation_id,
                error_code=final_error_code,
            )

        if config_key is not None:
            config_file = (
                str(kwargs.get("config_file"))
                if isinstance(kwargs.get("config_file"), str)
                else None
            )
            return cls._ConfigurationError(
                message,
                config_key=config_key,
                config_file=config_file,
                context=base_context,
                correlation_id=correlation_id,
                error_code=final_error_code,
            )

        # Default to general error
        return cls._Error(
            message,
            context=base_context,
            correlation_id=correlation_id,
            error_code=final_error_code,
        )

    # =============================================================================
    # UTILITY METHODS
    # =============================================================================

    @staticmethod
    def create_module_exception_classes(
        module_name: str,
    ) -> dict[str, type[FlextExceptions.BaseError]]:
        """Create module-specific exception classes.

        Creates a dictionary of exception classes tailored for a specific module,
        following the FLEXT ecosystem naming conventions.

        Args:
            module_name: Name of the module (e.g., "flext_grpc")

        Returns:
            Dictionary mapping exception names to exception classes

        """
        # Normalize module name for class naming
        normalized_name = module_name.upper().replace("-", "_").replace(".", "_")

        # Create base exception class for the module
        class ModuleBaseError(FlextExceptions.BaseError):
            """Base exception for module-specific errors."""

        # Create configuration error class
        class ModuleConfigurationError(ModuleBaseError):
            """Configuration-related errors for the module."""

        # Create connection error class
        class ModuleConnectionError(ModuleBaseError):
            """Connection-related errors for the module."""

        # Create validation error class
        class ModuleValidationError(ModuleBaseError):
            """Validation-related errors for the module."""

        # Create authentication error class
        class ModuleAuthenticationError(ModuleBaseError):
            """Authentication-related errors for the module."""

        # Create processing error class
        class ModuleProcessingError(ModuleBaseError):
            """Processing-related errors for the module."""

        # Create timeout error class
        class ModuleTimeoutError(ModuleBaseError):
            """Timeout-related errors for the module."""

        # Return dictionary with module-specific naming
        return {
            f"{normalized_name}BaseError": ModuleBaseError,
            f"{normalized_name}Error": ModuleBaseError,  # General error alias
            f"{normalized_name}ConfigurationError": ModuleConfigurationError,
            f"{normalized_name}ConnectionError": ModuleConnectionError,
            f"{normalized_name}ValidationError": ModuleValidationError,
            f"{normalized_name}AuthenticationError": ModuleAuthenticationError,
            f"{normalized_name}ProcessingError": ModuleProcessingError,
            f"{normalized_name}TimeoutError": ModuleTimeoutError,
        }


__all__: FlextTypes.Core.StringList = [
    "FlextExceptions",
]
