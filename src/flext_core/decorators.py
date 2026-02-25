"""Automation decorators for infrastructure concerns.

This module provides FlextDecorators, a collection of decorators that
automatically handle common infrastructure concerns to reduce boilerplate
code in services, handlers, and other components.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Mapping
from contextlib import suppress
from functools import wraps
from typing import Literal, overload

from flext_core.constants import c
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.exceptions import FlextExceptions as e
from flext_core.loggings import FlextLogger
from flext_core.models import m
from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import P, R, T, t


class FlextDecorators(FlextRuntime):
    """Automation decorators for infrastructure concerns.

    Architecture: Layer 3 (Application Layer - Cross-Cutting Concerns)
    ==================================================================
    Provides decorators that automatically handle common infrastructure
    concerns to reduce boilerplate code in services, handlers, and other
    components. All decorators are designed to integrate seamlessly with
    FlextResult, FlextContext, FlextLogger, and FlextContainer.

    **Architecture and Integration**:
    This class provides decorator utilities that integrate seamlessly with
    FlextResult, FlextContext, FlextLogger, and FlextContainer through
    structural typing. All decorators follow consistent patterns:
    - 8 static methods providing cross-cutting concern automation
    - Automatic context propagation and cleanup (defensive programming)
    - Integration with Foundation layer (FlextResult, FlextContext)
    - Integration with Infrastructure layer (FlextLogger, FlextContainer)
    - Composable decorator patterns with correct wrapper ordering

    **Core Decorators** (8 patterns):
    1. **@inject**: Automatic dependency injection from FlextContainer
       - Eliminates manual container.resolve() calls
       - Keyword-only parameter injection
       - Thread-safe via global container singleton

    2. **@log_operation**: Automatic operation logging with structured context
        return func(*args, **kwargs)
       - Structured logging with operation context binding
       - Defensive cleanup via context.suppress() to ensure unbinding
       - Integration with FlextLogger for context propagation

    3. **@track_performance**: Automatic performance tracking and metrics
       - Tracks operation duration (milliseconds and seconds)
       - Logs performance metrics with success/failure status
       - Error-aware timing (tracks duration even on exceptions)
       - Integration with FlextLogger for metrics collection

    4. **@railway**: Automatic railway pattern wrapping with FlextResult
       - Converts exceptions to FlextResult.fail()
       - Converts successful returns to FlextResult.ok()
       - Idempotent for functions already returning FlextResult
       - Enables functional error handling without try/except

    5. **@retry**: Automatic retry logic with exponential/linear backoff
       - Uses FlextConstants.Reliability for defaults
       - Supports exponential and linear backoff strategies
       - Logs retry attempts and exhaustion
       - Integration with FlextLogger for retry tracking

    6. **@timeout**: Automatic operation timeout enforcement
       - Uses FlextConstants.Reliability.DEFAULT_TIMEOUT_SECONDS
       - Checks timeout after operation completion
       - Raises e.TimeoutError on violation
       - Tracks duration even on exceptions for accurate timeout detection

    7. **@with_correlation**: Correlation ID management for distributed tracing
       - Ensures correlation ID exists in FlextContext
       - Uses FlextContext.Utilities.ensure_correlation_id()
       - Essential for request tracing across services
       - No-op if correlation ID already set

        return func(*args, **kwargs)
       - Combines @inject, @log_operation, @track_performance, @railway
       - Single-line configuration for maximum automation
       - Correct wrapper ordering for proper exception propagation
       - Balances automation with code clarity

    **Additional Decorators**:
    - **@with_context**: Context variable binding for operation duration
    - **@track_operation**: Combined correlation ID + logging tracking

    **Integration Points**:
    - **FlextContainer** (Layer 1): Service resolution for @inject
    - **FlextResult** (Layer 1): Result wrapping for @railway
    - **FlextContext** (Layer 4): Correlation ID and context management
    - **FlextLogger** (Layer 4): Structured logging and context binding
    - **e** (Layer 1): TimeoutError for @timeout
    - **FlextConstants** (Layer 0): Default values for retry/timeout

    **Defensive Programming Patterns**:
    1. Context cleanup uses `with suppress(Exception)` to ensure unbinding
    2. Retry decorator handles both success and exception cases
    3. Timeout checks both normal and exceptional code paths
    4. Dependency injection gracefully falls back on resolution failure
    5. Logger resolution tries self.logger before creating new logger

    **Decorator Composition Ordering** (Correct for Exception Propagation):
    ```
    @railway (outermost - converts exceptions to FlextResult)
    @inject (provides dependencies)
    @track_performance (tracks execution time)
    @log_operation (logs operations)
    (function)
    ```

        return func(*args, **kwargs)
    - All decorators use thread-safe FlextContainer.get_global()
    - Context binding is thread-safe via FlextContext.contextvars
    - FlextLogger context binding is thread-safe
    - No decorator maintains mutable state across calls

    **Performance Characteristics**:
    - O(1) decorator wrapping at function definition time
    - O(1) context binding/unbinding at runtime
    - O(n) for dependency injection where n = dependency count (typically 1-3)
    - O(1) timing via time.perf_counter()
    - No reflection or introspection at runtime

    **Usage Patterns**:

    1. Simple dependency injection:
        >>> from flext_core import FlextDecorators, r
        >>>
        >>> class UserService:
        ...     @FlextDecorators.inject(repo=UserRepository)
        ...     def get_user(self, user_id: str, *, repo) -> r[User]:
        ...         return repo.find_by_id(user_id)

    2. Automatic operation logging:
        >>> class OrderService:
        ...     @FlextDecorators.log_operation("create_order")
        ...     def create_order(self, order_data: dict) -> r[Order]:
        ...         # Start/completion/failure automatically logged
        ...         return self._process_order(order_data)

    3. Performance tracking:
        >>> class ReportService:
        ...     @FlextDecorators.track_performance("generate_report")
        ...     def generate_report(self, params: dict) -> r[Report]:
        ...         # Duration and metrics automatically tracked
        ...         return self._generate(params)

    4. Railway pattern wrapping:
        >>> from pydantic import EmailStr
        >>> @FlextDecorators.railway(error_code="VALIDATION_ERROR")
        ... def process_user_email(email: EmailStr) -> str:
        ...     # Pydantic v2 EmailStr validates email format natively
        ...     return email.lower()
        >>>
        >>> result = process_user_email("user@example.com")
        >>> assert result.is_success
        return func(*args, **kwargs)
    5. Automatic retry logic:
        >>> class ApiClient:
        ...     @FlextDecorators.retry(
        ...         max_attempts=5,
        ...         delay_seconds=1.0,
        ...         backoff_strategy="exponential",
        ...     )
        ...     def call_external_api(
        ...         self, endpoint: str
         ...     ) -> m.ConfigMap:
        ...         # Automatically retries on failure with backoff
        ...         return requests.get(endpoint).json()

    6. Operation timeout enforcement:
        >>> class LongRunningService:
        ...     @FlextDecorators.timeout(timeout_seconds=30.0)
        ...     def expensive_operation(self, data: list) -> r[float]:
        ...         # Raises TimeoutError if exceeds 30 seconds
        ...         return self._expensive_computation(data)

    7. Correlation ID management:
        >>> class PaymentService:
        ...     @FlextDecorators.with_correlation()
        ...     def process_payment(self, payment_id: str) -> r[str]:
        ...         # Correlation ID automatically ensured
        ...         # All logs include correlation_id
        ...         return self._charge(payment_id)

    8. Maximum automation with @combined:
        >>> class OrderService:
        ...     @FlextDecorators.combined(
        ...         inject_deps={"repo": OrderRepository, "validator": OrderValidator},
        ...         operation_name="create_order",
        ...         track_perf=True,
        ...         use_railway=True,
        ...     )
        ...     def create_order(
        ...         self, order_data: dict, *, repo, validator
        ...     ) -> r[Order]:
        ...         # All infrastructure automatic:
        ...         # - DI injection
        return func(*args, **kwargs)
        ...         # - Performance tracking
        ...         # - Railway pattern
        ...         return validator.validate(order_data).flat_map(repo.create)

    9. Context variable binding:
        >>> class MultiTenantService:
        ...     @FlextDecorators.with_context(
        ...         tenant_id="tenant-123", user_id="user-456"
        ...     )
        ...     def process_tenant_data(self) -> r[dict]:
        ...         # tenant_id and user_id bound to context
        ...         # All logs include these values
        ...         return self._process()

    10. Comprehensive operation tracking:
        >>> class CriticalService:
        ...     @FlextDecorators.track_operation(
        ...         operation_name="critical_process", track_correlation=True
        ...     )
        ...     def critical_process(self) -> r[str]:
        ...         # Automatic correlation ID + logging + performance
        ...         return self._critical_work()

    **Error Handling Patterns**:

    1. Retry with exponential backoff:
        >>> @FlextDecorators.retry(max_attempts=3, backoff_strategy="exponential")
        ... def unreliable_operation() -> str:
        ...     # Fails, retries with delays: 1s, 2s, 4s
        ...     return api_call()

    2. Timeout protection:
        >>> @FlextDecorators.timeout(timeout_seconds=30)
        ... def long_operation() -> str:
        ...     # Raises TimeoutError if exceeds 30 seconds
        ...     return expensive_computation()

    3. Railway pattern error handling:
        >>> @FlextDecorators.railway(error_code="BUSINESS_ERROR")
         ... def business_operation() -> m.ConfigMap:
        ...     # All exceptions become FlextResult.fail()
        ...     return process_business_logic()
        >>>
        >>> result = business_operation()
        >>> if result.is_failure:
        ...     handle_error(result.error)

    **Complete Integration Example**:
        >>> from flext_core import (
        ...     FlextDecorators,
        ...     FlextResult,
        ...     FlextLogger,
        ...     FlextContainer,
        return func(*args, **kwargs)
        >>>
        >>> class ProductService:
        ...     def __init__(self):
        ...         self.logger = FlextLogger(__name__)
        ...
        ...     @FlextDecorators.combined(
        ...         inject_deps={"repo": ProductRepository},
        ...         operation_name="create_product",
        ...         track_perf=True,
        ...         use_railway=True,
        ...     )
        ...     def create_product(self, product_data: dict, *, repo) -> r[Product]:
        ...         # Automatic infrastructure:
        ...         # - Dependency injection
        ...         # - Structured logging
        ...         # - Performance tracking
        ...         # - Railway pattern
        ...         # - Exception handling
        ...         # - Context propagation
        ...
        ...         return (
        ...             r[dict]
        ...             .ok(product_data)
        ...             .flat_map(self._validate)
        ...             .flat_map(repo.save)
        ...         )
        ...
        ...     def _validate(self, data: dict) -> r[dict]:
        ...         if "name" not in data:
        ...             return r[dict].fail("Name required")
        ...         return r[dict].ok(data)
    """

    @staticmethod
    def deprecated(
        message: str,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to mark functions/variables as deprecated.

        Emits DeprecationWarning when decorated function is called.
        Used during v0.10 → v0.11 refactoring for constants migration.

        Architecture: Cross-Cutting Concern (Tier 3)
        ============================================
        Provides simple deprecation warnings for functions, methods, and constants
        being replaced during the constants refactoring phase. Warnings guide users
        to use new APIs before deprecated code is removed.

        Args:
            message: Deprecation message explaining what to use instead

        Returns:
            Decorator function that wraps the target callable

        Example:
            >>> @FlextDecorators.deprecated("Use new_constant instead")
            ... def old_function():
            ...     return "old"

        Note:
            This decorator is intended for v0.10 → v0.11 transition period.
            After deprecation cycle completes, remove decorator and aliases.

        """

        def decorator(
            func: Callable[P, R],
        ) -> Callable[P, R]:
            """Apply deprecation warning to callable."""

            @wraps(func)
            def wrapper(
                *args: P.args,
                **kwargs: P.kwargs,
            ) -> R:
                """Wrapper that emits warning before execution."""
                warnings.warn(
                    message,
                    DeprecationWarning,
                    stacklevel=2,
                )
                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def inject(
        **dependencies: str,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to automatically inject dependencies from FlextContainer.

        Automatically resolves and injects dependencies from the global
        container, eliminating manual container.resolve() calls in every
        method.

        Args:
            **dependencies: Mapping of parameter names to service types to
                inject

        Returns:
            Decorated function with automatic dependency injection

        Example:
            ```python
            from flext_core import FlextDecorators, FlextResult


            class MyService:
                @FlextDecorators.inject(repo=MyRepository, logger=FlextLogger)
                def process_data(self, data: dict, *, repo, logger) -> r[dict]:
                    # repo and logger are automatically injected!
                    logger.info("processing_data", data_keys=list(data.keys()))
                    return repo.save(data)
            ```

        Note:
            Injected parameters must be keyword-only (after * in signature)
            to avoid conflicts with positional arguments.

        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Get container from self if available, otherwise use global singleton
                container = FlextContainer.create()

                # Inject dependencies that aren't already provided
                for name, service_key in dependencies.items():
                    if name not in kwargs:
                        # Get from container using the service key
                        result = container.get(service_key)
                        if result.is_success:
                            # Use .value directly - FlextResult never returns None on success
                            kwargs[name] = result.value
                        # If resolution fails, let the function handle missing parameter

                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def log_operation(
        operation_name: str | None = None,
        *,
        track_perf: bool = False,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to automatically log operation execution with structured logging.

        Automatically logs operation start, completion, and failures with
        structured context, eliminating manual logging boilerplate.

        Args:
            operation_name: Name for the operation (defaults to function name)
            track_perf: If True, also tracks performance metrics (duration_ms,
                duration_seconds). Default False.

        Returns:
            Decorated function with automatic operation logging

        Example:
            ```python
            from flext_core import FlextDecorators, FlextResult


            class MyService:
                @FlextDecorators.log_operation("process_user_data")
                def process(self, user_id: str) -> r[dict]:
                    # Automatic logging of start/complete/failure
                    # Automatic context propagation
                    return self._do_processing(user_id)

                @FlextDecorators.log_operation("heavy_task", track_perf=True)
                def heavy(self) -> r[dict]:
                    # Also includes duration_ms and duration_seconds
                    return self._compute()
            ```

        Note:
            Works best with classes that have logger attribute. Falls back to
            FlextLogger.get_logger() otherwise.

        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Fast fail: explicit default value instead of 'or' fallback
                op_name: str = (
                    operation_name if operation_name is not None else func.__name__
                )

                # Get logger from self if available, otherwise create one
                logger = FlextDecorators._resolve_logger(args, func)
                # with_result() is specific to FlextLogger, not in protocol
                # Type narrowing: _resolve_logger always returns FlextLogger
                result_logger = logger.with_result()

                correlation_id = FlextDecorators._bind_operation_context(
                    operation=op_name,
                    logger=logger,
                    function_name=func.__name__,
                    ensure_correlation=True,
                )

                # Track timing if requested
                start_time = time.perf_counter() if track_perf else 0.0

                try:
                    start_extra: dict[str, t.MetadataScalarValue] = {
                        "function": func.__name__,
                        "func_module": func.__module__,
                    }
                    if correlation_id is not None:
                        start_extra["correlation_id"] = correlation_id

                    if correlation_id is not None:
                        _ = result_logger.debug(
                            "%s_started",
                            op_name,
                            extra={
                                "function": func.__name__,
                                "func_module": func.__module__,
                                "correlation_id": correlation_id,
                            },
                        )
                    else:
                        _ = result_logger.debug(
                            "%s_started",
                            op_name,
                            extra={
                                "function": func.__name__,
                                "func_module": func.__module__,
                            },
                        )

                    result = func(*args, **kwargs)

                    completion_extra: dict[str, t.MetadataScalarValue] = {
                        "function": func.__name__,
                        "success": True,
                    }
                    if correlation_id is not None:
                        completion_extra["correlation_id"] = correlation_id

                    # Add timing metrics if tracking performance
                    if track_perf:
                        duration = time.perf_counter() - start_time
                        completion_extra["duration_ms"] = (
                            duration * c.MILLISECONDS_MULTIPLIER
                        )
                        completion_extra["duration_seconds"] = duration

                    _ = result_logger.debug(
                        "%s_completed",
                        op_name,
                        extra=dict(completion_extra),
                    )
                    return result
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as exc:
                    failure_extra: dict[str, t.MetadataAttributeValue] = {
                        "function": func.__name__,
                        "success": False,
                        "error": str(exc),
                        "error_type": exc.__class__.__name__,
                        "operation": op_name,
                    }
                    if correlation_id is not None:
                        failure_extra["correlation_id"] = correlation_id

                    tracked_duration = (
                        time.perf_counter() - start_time if track_perf else 0.0
                    )
                    # Add timing metrics if tracking performance
                    if track_perf:
                        failure_extra["duration_ms"] = (
                            tracked_duration * c.MILLISECONDS_MULTIPLIER
                        )
                        failure_extra["duration_seconds"] = tracked_duration

                    exc_info_value = True
                    if correlation_id is not None and track_perf:
                        _ = result_logger.exception(
                            op_name,
                            exception=exc,
                            exc_info=exc_info_value,
                            function=func.__name__,
                            success=False,
                            error=str(exc),
                            error_type=exc.__class__.__name__,
                            operation=op_name,
                            correlation_id=correlation_id,
                            duration_ms=tracked_duration * c.MILLISECONDS_MULTIPLIER,
                            duration_seconds=tracked_duration,
                        )
                    elif correlation_id is not None:
                        _ = result_logger.exception(
                            op_name,
                            exception=exc,
                            exc_info=exc_info_value,
                            function=func.__name__,
                            success=False,
                            error=str(exc),
                            error_type=exc.__class__.__name__,
                            operation=op_name,
                            correlation_id=correlation_id,
                        )
                    elif track_perf:
                        _ = result_logger.exception(
                            op_name,
                            exception=exc,
                            exc_info=exc_info_value,
                            function=func.__name__,
                            success=False,
                            error=str(exc),
                            error_type=exc.__class__.__name__,
                            operation=op_name,
                            duration_ms=tracked_duration * c.MILLISECONDS_MULTIPLIER,
                            duration_seconds=tracked_duration,
                        )
                    else:
                        _ = result_logger.exception(
                            op_name,
                            exception=exc,
                            exc_info=exc_info_value,
                            function=func.__name__,
                            success=False,
                            error=str(exc),
                            error_type=exc.__class__.__name__,
                            operation=op_name,
                        )
                    raise
                finally:
                    # CRITICAL: Clear operation context (defensive cleanup)
                    # Use suppress to ensure cleanup never fails
                    with suppress(Exception):
                        FlextDecorators._clear_operation_scope(
                            logger=logger,
                            function_name=func.__name__,
                            operation=op_name,
                        )

            return wrapper

        return decorator

    @staticmethod
    def track_performance(
        operation_name: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to automatically track operation performance metrics.

        Tracks operation duration and logs performance metrics with structured
        logging, enabling performance monitoring without manual timing code.

        Args:
            operation_name: Name for the operation (defaults to function name)

        Returns:
            Decorated function with automatic performance tracking

        Example:
            ```python
            from flext_core import FlextDecorators, FlextResult


            class MyService:
                @FlextDecorators.track_performance("heavy_computation")
                def compute(self, data: list) -> r[float]:
                    # Automatic timing and performance logging
                    return self._expensive_calculation(data)
            ```

        Note:
            Performance metrics are logged to structured logging context and
            can be used with track() for metrics collection.

        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Fast fail: explicit default value instead of 'or' fallback
                op_name: str = (
                    operation_name if operation_name is not None else func.__name__
                )

                # Get logger from self if available, otherwise create one
                logger = FlextDecorators._resolve_logger(args, func)

                start_time = time.perf_counter()

                correlation_id = FlextDecorators._bind_operation_context(
                    operation=op_name,
                    logger=logger,
                    function_name=func.__name__,
                    ensure_correlation=True,
                )

                try:
                    result = func(*args, **kwargs)
                    duration = time.perf_counter() - start_time

                    success_extra: dict[str, t.MetadataAttributeValue] = {
                        "operation": op_name,
                        "duration_ms": duration * c.MILLISECONDS_MULTIPLIER,
                        "duration_seconds": duration,
                        "success": True,
                    }
                    if correlation_id is not None:
                        success_extra["correlation_id"] = correlation_id
                    logger.info("operation_completed", **success_extra)
                    return result
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as e:
                    duration = time.perf_counter() - start_time

                    failure_extra: dict[str, t.MetadataAttributeValue] = {
                        "operation": op_name,
                        "duration_ms": duration * c.MILLISECONDS_MULTIPLIER,
                        "duration_seconds": duration,
                        "success": False,
                        "error": str(e),
                        "error_type": e.__class__.__name__,
                    }
                    if correlation_id is not None:
                        failure_extra["correlation_id"] = correlation_id
                    if correlation_id is not None:
                        logger.exception(
                            "operation_failed",
                            operation=op_name,
                            duration_ms=duration * c.MILLISECONDS_MULTIPLIER,
                            duration_seconds=duration,
                            success=False,
                            error=str(e),
                            error_type=e.__class__.__name__,
                            correlation_id=correlation_id,
                        )
                    else:
                        logger.exception(
                            "operation_failed",
                            operation=op_name,
                            duration_ms=duration * c.MILLISECONDS_MULTIPLIER,
                            duration_seconds=duration,
                            success=False,
                            error=str(e),
                            error_type=e.__class__.__name__,
                        )
                    raise
                finally:
                    # CRITICAL: Clear operation context (defensive cleanup)
                    # Use suppress to ensure cleanup never fails
                    with suppress(Exception):
                        FlextDecorators._clear_operation_scope(
                            logger=logger,
                            function_name=func.__name__,
                            operation=op_name,
                        )

            return wrapper

        return decorator

    @staticmethod
    def railway(
        error_code: str | None = None,
    ) -> Callable[[Callable[P, T]], Callable[P, r[T]]]:
        """Decorator to automatically wrap function in railway pattern.

        Automatically converts exceptions to FlextResult failures and
        successful returns to FlextResult successes, eliminating manual
        try/except boilerplate.

        Args:
            error_code: Optional error code for failures

        Returns:
            Decorated function that returns r[T]

        Example:
            ```python
            from flext_core import FlextDecorators, FlextResult


            from pydantic import EmailStr


            @FlextDecorators.railway(error_code="VALIDATION_ERROR")
            def process_user_email(email: EmailStr) -> str:
                # Pydantic v2 EmailStr validates format natively
                # Exception automatically becomes FlextResult.fail()
                # Success automatically becomes FlextResult.ok()
                return email.lower()


            # Returns r[str] automatically
            result = process_user_email("user@example.com")
            assert result.is_success
            ```

        Note:
            If the function already returns a FlextResult, it's returned as-is.
            Only bare values or exceptions are wrapped.

        """

        def decorator(func: Callable[P, T]) -> Callable[P, r[T]]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> r[T]:
                try:
                    result = func(*args, **kwargs)

                    # Wrap successful result
                    return r[T].ok(result)

                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as e:
                    # Convert exception to FlextResult failure
                    effective_error_code: str = (
                        error_code if error_code is not None else "OPERATION_ERROR"
                    )
                    return r[T].fail(
                        str(e),
                        error_code=effective_error_code,
                    )

            return wrapper

        return decorator

    @staticmethod
    def retry(
        max_attempts: int | None = None,
        delay_seconds: float | None = None,
        backoff_strategy: str | None = None,
        error_code: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to automatically retry failed operations with exponential backoff.

        Uses FlextConstants.Reliability for default values and
        e for structured error handling, integrating foundation
        modules.

        Args:
            max_attempts: Maximum retry attempts (default:
                FlextConstants.Reliability.DEFAULT_MAX_RETRIES)
            delay_seconds: Initial delay between retries (default:
                FlextConstants.Reliability.DEFAULT_RETRY_DELAY_SECONDS)
            backoff_strategy: Backoff strategy ('exponential' or 'linear',
                default: FlextConstants.Reliability.DEFAULT_BACKOFF_STRATEGY)
            error_code: Optional error code for failures

        Returns:
            Decorated function with automatic retry logic

        Example:
            ```python
            from flext_core import FlextDecorators


            class MyService:
                @FlextDecorators.retry(
                    max_attempts=5,
                    delay_seconds=2.0,
                    backoff_strategy=c.Reliability.BACKOFF_STRATEGY_EXPONENTIAL,
                )
                def unreliable_operation(self) -> m.ConfigMap:
                    # Automatically retries on failure with exponential backoff
                    return self._make_api_call()
            ```

        Note:
            Uses FlextConstants.Reliability for defaults, ensuring consistency
            across the entire ecosystem. Logs retry attempts automatically.

        """
        # Fast fail: explicit default values instead of 'or' fallback
        attempts: int = (
            max_attempts
            if max_attempts is not None
            else c.Reliability.DEFAULT_MAX_RETRIES
        )
        delay: float = (
            delay_seconds
            if delay_seconds is not None
            else float(c.Reliability.DEFAULT_RETRY_DELAY_SECONDS)
        )
        strategy: str = (
            backoff_strategy
            if backoff_strategy is not None
            else c.Reliability.DEFAULT_BACKOFF_STRATEGY
        )

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                logger = FlextDecorators._resolve_logger(args, func)
                # Create retry config from parameters
                retry_config = m.RetryConfiguration(
                    max_attempts=attempts,
                    initial_delay_seconds=delay,
                    exponential_backoff=(
                        strategy == c.Reliability.BACKOFF_STRATEGY_EXPONENTIAL
                    ),
                )
                try:
                    retry_result = FlextDecorators._execute_retry_loop(
                        func,
                        args,
                        kwargs,
                        logger,
                        retry_config=retry_config,
                    )
                    if isinstance(retry_result, Exception):
                        FlextDecorators._handle_retry_exhaustion(
                            retry_result,
                            func,
                            attempts,
                            error_code,
                            logger,
                        )
                        retry_error_code = (
                            error_code
                            if error_code is not None
                            else "OPERATION_TIMEOUT"
                        )
                        timeout_message = f"Operation {func.__name__} failed after {attempts} attempts"
                        raise e.TimeoutError(
                            timeout_message,
                            error_code=retry_error_code,
                            operation=func.__name__,
                            attempts=attempts,
                            original_error=str(retry_result),
                        )
                    return retry_result
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as exc:
                    FlextDecorators._handle_retry_exhaustion(
                        exc,
                        func,
                        attempts,
                        error_code,
                        logger,
                    )
                    raise

            return wrapper

        return decorator

    @staticmethod
    def _resolve_logger(
        args: tuple[object, ...],
        func: Callable[..., R],
    ) -> FlextLogger:
        """Resolve logger from first argument or create module logger.

        Returns:
            FlextLogger instance (concrete type, not protocol)

        """
        first_arg = args[0] if args else None
        potential_logger: FlextLogger | None = (
            getattr(first_arg, "logger", None) if first_arg is not None else None
        )
        if potential_logger is not None:
            return potential_logger
        # FlextLogger constructor returns FlextLogger
        return FlextLogger(func.__module__)

    @staticmethod
    def _execute_retry_loop[R](
        func: Callable[..., R],  # Variadic: called with pre-split args/kwargs
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
        logger: FlextLogger,
        *,
        retry_config: m.RetryConfiguration | None = None,
    ) -> R | Exception:
        """Execute retry loop and return last exception.

        Uses RetryConfiguration model to reduce parameter count from 8 to 5.

        Args:
            func: Function to execute
            args: Function positional arguments
            kwargs: Function keyword arguments
            logger: Logger instance
            retry_config: RetryConfiguration instance (Pydantic v2)

        Returns:
            Function result on success

        """
        # Use default config if none provided
        if retry_config is None:
            retry_config = m.RetryConfiguration()

        attempts = retry_config.max_attempts
        delay = retry_config.initial_delay_seconds
        # Map exponential_backoff bool to strategy string
        strategy = (
            c.Reliability.BACKOFF_STRATEGY_EXPONENTIAL
            if retry_config.exponential_backoff
            else c.Reliability.BACKOFF_STRATEGY_LINEAR
        )

        last_exception: Exception | None = None
        current_delay = delay

        for attempt in range(1, attempts + 1):
            try:
                if attempt > 1:
                    logger.info(
                        "retry_attempt",
                        extra={
                            "function": func.__name__,
                            "attempt": attempt,
                            "max_attempts": attempts,
                            "delay_seconds": current_delay,
                        },
                    )
                    time.sleep(current_delay)

                return func(*args, **kwargs)

            except (
                AttributeError,
                TypeError,
                ValueError,
                RuntimeError,
                KeyError,
            ) as e:
                last_exception = e

                logger.warning(
                    "operation_failed_retrying",
                    extra={
                        "function": func.__name__,
                        "attempt": attempt,
                        "max_attempts": attempts,
                        "error": str(e),
                        "error_type": e.__class__.__name__,
                    },
                )

                # Calculate next delay based on strategy
                if strategy == c.Reliability.BACKOFF_STRATEGY_EXPONENTIAL:
                    current_delay *= 2
                elif strategy == c.Reliability.BACKOFF_STRATEGY_LINEAR:
                    current_delay += delay
                # For unknown strategies, keep constant delay

                # If this was the last attempt, we'll raise after loop
                if attempt == attempts:
                    break

        # Should never be None (loop always catches exceptions), but handle defensively
        if last_exception is None:
            msg = "Retry loop completed without success or exception"
            return RuntimeError(msg)
        return last_exception

    @staticmethod
    def _handle_retry_exhaustion(
        last_exception: Exception,
        func: Callable[..., R],
        attempts: int,
        error_code: str | None,
        logger: FlextLogger,
    ) -> None:
        """Handle retry exhaustion and raise appropriate exception."""
        # All retries exhausted
        logger.error(
            "operation_failed_all_retries_exhausted",
            extra={
                "function": func.__name__,
                "attempts": attempts,
                "error": str(last_exception),
                "error_type": last_exception.__class__.__name__,
            },
        )
        if last_exception:
            raise last_exception
        retry_error_code = error_code if error_code is not None else "OPERATION_TIMEOUT"
        timeout_message = f"Operation {func.__name__} failed after {attempts} attempts"
        raise e.TimeoutError(
            timeout_message,
            error_code=retry_error_code,
            operation=func.__name__,
            attempts=attempts,
            original_error=str(last_exception),
        )

    @staticmethod
    def _bind_operation_context(
        *,
        operation: str,
        logger: FlextLogger,
        function_name: str,
        ensure_correlation: bool,
    ) -> str | None:
        """Ensure correlation, bind operation context, and report failures."""
        correlation_id: str | None = None
        if ensure_correlation:
            correlation_id = FlextContext.Utilities.ensure_correlation_id()
        else:
            current_id = FlextContext.Variables.CorrelationId.get()
            if current_id:
                correlation_id = current_id

        FlextContext.Request.set_operation_name(operation)

        # Use bind_context with SCOPE_OPERATION (replaces bind_operation_context)
        binding_result = FlextLogger.bind_context(
            c.Context.SCOPE_OPERATION,
            operation=operation,
        )
        if binding_result.is_failure:
            logger.warning(
                "operation_context_binding_failed",
                extra={
                    "function": function_name,
                    "operation": operation,
                    "error": binding_result.error,
                    "error_code": binding_result.error_code,
                    "correlation_id": correlation_id,
                },
            )
        return correlation_id

    @staticmethod
    def _clear_operation_scope(
        *,
        logger: FlextLogger,
        function_name: str,
        operation: str,
    ) -> None:
        """Clear operation scope and log if cleanup fails."""
        clear_result = FlextLogger.clear_scope("operation")
        # clear_scope() returns r[bool] (never None), so is not None check is redundant
        if clear_result.is_failure:
            FlextDecorators._handle_log_result(
                result=clear_result,
                logger=logger,
                fallback_message="operation_context_clear_failed",
                kwargs=m.ConfigMap(
                    root={
                        "extra": {
                            "function": function_name,
                            "operation": operation,
                        },
                    }
                ),
            )

    @staticmethod
    def _handle_log_result(
        *,
        result: p.Result[bool] | FlextRuntime.RuntimeResult[bool],
        logger: FlextLogger,
        fallback_message: str,
        kwargs: m.ConfigMap,
    ) -> None:
        """Ensure FlextLogger call results are handled for diagnostics."""
        if result.is_failure:
            fallback_logger = getattr(logger, "logger", None)
            if fallback_logger is None or not hasattr(fallback_logger, "warning"):
                return
            fallback_kwargs = m.ConfigMap(root=dict(kwargs.root))
            _ = fallback_kwargs.setdefault("extra", {})
            extra_payload_raw = fallback_kwargs["extra"]
            extra_payload = (
                m.ConfigMap(root=dict(extra_payload_raw))
                if FlextRuntime.is_dict_like(extra_payload_raw)
                else m.ConfigMap(root={})
            )
            if FlextRuntime.is_dict_like(extra_payload):
                extra_payload["log_error"] = result.error
                extra_payload["log_error_code"] = result.error_code
                fallback_kwargs["extra"] = extra_payload
            else:
                fallback_kwargs["log_error"] = result.error
                fallback_kwargs["log_error_code"] = result.error_code
            _ = fallback_logger.warning(fallback_message, **fallback_kwargs.root)

    @staticmethod
    def timeout(
        timeout_seconds: float | None = None,
        error_code: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to enforce operation timeout.

        Uses FlextConstants.Reliability.DEFAULT_TIMEOUT_SECONDS for default
        timeout and e.TimeoutError for structured error
        handling.

        Args:
            timeout_seconds: Timeout in seconds (default:
                c.Reliability.DEFAULT_TIMEOUT_SECONDS)
            error_code: Optional error code for timeout

        Returns:
            Decorated function with timeout enforcement

        Example:
            ```python
            from flext_core import FlextDecorators


            class MyService:
                @FlextDecorators.timeout(timeout_seconds=30.0)
                def long_running_operation(self) -> m.ConfigMap:
                    # Automatically raises TimeoutError if exceeds 30 seconds
                    return self._process_data()
            ```

        Note:
            This is a simple timeout based on elapsed time checking. For true
            thread-based timeouts, use threading.Timer or asyncio.

        """
        # Use c.Reliability for default
        max_duration = (
            timeout_seconds
            if timeout_seconds is not None
            else c.Reliability.DEFAULT_TIMEOUT_SECONDS
        )

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                start_time = time.perf_counter()

                try:
                    result = func(*args, **kwargs)

                    # Check if operation exceeded timeout
                    duration = time.perf_counter() - start_time
                    if duration > max_duration:
                        msg = f"Operation {func.__name__} exceeded timeout of {max_duration}s (took {duration:.2f}s)"
                        effective_error_code: str = (
                            error_code
                            if error_code is not None
                            else "OPERATION_TIMEOUT"
                        )
                        raise e.TimeoutError(
                            msg,
                            error_code=effective_error_code,
                            timeout_seconds=max_duration,
                            operation=func.__name__,
                            duration_seconds=duration,
                        )

                    return result

                except e.TimeoutError:
                    # Re-raise timeout errors
                    raise
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as exc:
                    # Check duration even on exception
                    duration = time.perf_counter() - start_time
                    if duration > max_duration:
                        msg = f"Operation {func.__name__} exceeded timeout of {max_duration}s (took {duration:.2f}s) and raised {exc.__class__.__name__}"
                        raise e.TimeoutError(
                            msg,
                            error_code=error_code or "OPERATION_TIMEOUT",
                            timeout_seconds=max_duration,
                            operation=func.__name__,
                            duration_seconds=duration,
                            original_error=str(exc),
                        ) from exc
                    # Re-raise original exception if not timeout
                    raise

            return wrapper

        return decorator

    @overload
    @staticmethod
    def combined(
        *,
        inject_deps: Mapping[str, str] | None = None,
        operation_name: str | None = None,
        track_perf: bool = True,
        use_railway: Literal[False] = False,
        error_code: str | None = None,
    ) -> Callable[[Callable[..., R]], Callable[..., R]]: ...

    @overload
    @staticmethod
    def combined(
        *,
        inject_deps: Mapping[str, str] | None = None,
        operation_name: str | None = None,
        track_perf: bool = True,
        use_railway: Literal[True],
        error_code: str | None = None,
    ) -> Callable[[Callable[..., R]], Callable[..., r[R]]]: ...

    @staticmethod
    def combined(
        *,
        inject_deps: Mapping[str, str] | None = None,
        operation_name: str | None = None,
        track_perf: bool = True,
        use_railway: bool = False,
        error_code: str | None = None,
    ) -> Callable[[Callable[..., R]], Callable[..., R] | Callable[..., r[R]]]:
        """Combined decorator applying multiple automation patterns at once.

        Combines @inject, @log_operation (with optional track_perf), and optionally
        @railway into a single decorator for maximum code reduction.

        Args:
            inject_deps: Dependencies to inject (name -> type mapping)
            operation_name: Name for logging (defaults to function name)
            track_perf: Whether to track performance (default: True)
            use_railway: Whether to apply railway pattern (default: False)
            error_code: Error code for railway pattern failures

        Returns:
            Decorated function with all requested automations.
            When use_railway=True, returns Callable[..., r[R]].
            When use_railway=False, returns Callable[..., R].

        Example:
            ```python
            from flext_core import FlextDecorators, FlextResult


            class OrderService:
                @FlextDecorators.combined(
                    inject_deps={
                        "repo": OrderRepository,
                        "validator": OrderValidator,
                    },
                    operation_name="create_order",
                    track_perf=True,
                    use_railway=True,
                )
                def create_order(
                    self, order_data: dict, *, repo, validator
                ) -> r[Order]:
                    # All infrastructure automatic:
                    # - DI injection
                    # - Logging
                    # - Performance tracking
                    # - Railway pattern
                    return validator.validate(order_data).flat_map(repo.create)
            ```

        Note:
            This decorator provides maximum automation but use judiciously to
            maintain code clarity.

        """
        # Return different decorators based on use_railway flag
        # This separation ensures proper type inference by pyrefly
        if use_railway:
            # Railway path decorator
            def railway_decorator(func: Callable[..., R]) -> Callable[..., r[R]]:
                # Railway decorator changes return type from R to r[R]
                result = FlextDecorators.railway(error_code=error_code)(func)
                # Apply dependency injection to railway-wrapped function
                if inject_deps:
                    result = FlextDecorators.inject(**inject_deps)(result)
                # Apply unified log_operation with optional performance tracking
                return FlextDecorators.log_operation(
                    operation_name=operation_name,
                    track_perf=track_perf,
                )(result)

            return railway_decorator

        # Non-railway path decorator
        def standard_decorator(func: Callable[..., R]) -> Callable[..., R]:
            result = func
            if inject_deps:
                result = FlextDecorators.inject(**inject_deps)(result)
            # Apply unified log_operation with optional performance tracking
            return FlextDecorators.log_operation(
                operation_name=operation_name,
                track_perf=track_perf,
            )(result)

        return standard_decorator

    @staticmethod
    def with_correlation() -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to ensure correlation ID exists for operation tracking.

        Automatically ensures a correlation ID is present in the context,
        generating one if needed. Essential for distributed tracing and
        request correlation across services.

        Returns:
            Decorated function with correlation ID management

        Example:
            ```python
            from flext_core import FlextDecorators, FlextResult


            class OrderService:
                @FlextDecorators.with_correlation()
                def process_order(self, order_id: str) -> r[dict]:
                    # Correlation ID automatically ensured in context
                    # All logs will include correlation_id
                    return self._process(order_id)
            ```

        Note:
            Uses FlextRuntime.Integration for context management via
            structlog.contextvars (single source of truth).

        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Ensure correlation ID exists
                _ = FlextContext.Utilities.ensure_correlation_id()

                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def with_context(
        **context_vars: str | int | bool | None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to manage context lifecycle for an operation.

        Automatically binds context variables for the operation duration and
        unbinds them after completion. Enables automatic context propagation
        without manual bind/unbind calls.

        Args:
            **context_vars: Context variables to bind for operation duration

        Returns:
            Decorated function with automatic context management

        Example:
            ```python
            from flext_core import FlextDecorators, FlextResult


            class PaymentService:
                @FlextDecorators.with_context(
                    service_name="payment_service", service_version="1.0.0"
                )
                def process_payment(self, amount: float) -> r[str]:
                    # service_name and service_version automatically in context
                    # All logs will include these values
                    return self._charge(amount)
            ```

        Note:
            Context is automatically cleaned up after operation completes,
            even if exception occurs.

        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                logger = FlextDecorators._resolve_logger(args, func)

                try:
                    # Bind context variables to global logging context
                    if context_vars:
                        bind_result = FlextLogger.bind_global_context(**context_vars)
                        if bind_result.is_failure:
                            logger.warning(
                                "global_context_binding_failed",
                                function=func.__name__,
                                error=bind_result.error,
                                error_code=bind_result.error_code,
                                bound_keys=list(context_vars.keys()),
                            )

                    return func(*args, **kwargs)

                finally:
                    # Unbind context variables
                    if context_vars:
                        unbind_result = FlextLogger.unbind_global_context(
                            *tuple(context_vars.keys()),
                        )
                        if unbind_result.is_failure:
                            logger.warning(
                                "global_context_unbind_failed",
                                function=func.__name__,
                                error=unbind_result.error,
                                error_code=unbind_result.error_code,
                                bound_keys=list(context_vars.keys()),
                            )

            return wrapper

        return decorator

    @staticmethod
    def track_operation(
        operation_name: str | None = None,
        *,
        track_correlation: bool = True,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to track operation execution with FlextRuntime.Integration.

        Combines correlation ID management and structured logging using
        FlextRuntime.Integration pattern (Layer 0.5). Performance tracking
        happens automatically via FlextRuntime.Integration.
        No circular imports - uses structlog directly.

        Args:
            operation_name: Name for the operation (defaults to function name)
            track_correlation: Ensure correlation ID exists (default: True)

        Returns:
            Decorated function with comprehensive operation tracking

        Example:
            ```python
            from flext_core import FlextDecorators, FlextResult


            class UserService:
                @FlextDecorators.track_operation("create_user")
                def create_user(self, user_data: dict) -> r[User]:
                    # Automatic tracking:
                    # - Correlation ID ensured
                    # - Operation name bound to context
                    # - Performance metrics via FlextRuntime.Integration
                    # - All via structlog directly (no circular imports)
                    return self._create(user_data)
            ```

        Note:
            This decorator uses FlextRuntime.Integration which accesses
            structlog directly (Layer 0.5), avoiding circular imports between
            Foundation and Infrastructure layers.

        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                # Fast fail: explicit default value instead of 'or' fallback
                op_name: str = (
                    operation_name if operation_name is not None else func.__name__
                )

                logger = FlextDecorators._resolve_logger(args, func)

                correlation_id = FlextDecorators._bind_operation_context(
                    operation=op_name,
                    logger=logger,
                    function_name=func.__name__,
                    ensure_correlation=track_correlation,
                )
                if track_correlation and correlation_id is None:
                    logger.warning(
                        "correlation_id_missing",
                        extra={
                            "function": func.__name__,
                            "operation": op_name,
                        },
                    )

                try:
                    # Call the actual function
                    # Performance tracking via FlextRuntime.Integration
                    # happens automatically
                    return func(*args, **kwargs)

                finally:
                    # Clear operation context
                    with suppress(Exception):
                        FlextDecorators._clear_operation_scope(
                            logger=logger,
                            function_name=func.__name__,
                            operation=op_name,
                        )

            return wrapper

        return decorator

    @staticmethod
    def factory(
        name: str,
        *,
        singleton: bool = False,
        lazy: bool = True,
    ) -> t.DecoratorType:
        """Decorator to mark functions as factories for DI container.

        Stores factory configuration as metadata on the decorated function,
        enabling auto-discovery by FlextContainer and factory registries.

        Args:
            name: Name to register the factory under in the container
            singleton: Whether factory creates singleton instances. Default: False
            lazy: Whether to defer factory invocation until first use. Default: True

        Returns:
            Decorator function for marking factory functions

        Example:
            >>> @FlextDecorators.factory(name="user_service", singleton=True)
            ... def create_user_service(config: FlextSettings) -> UserService:
            ...     return UserService(config)

        """

        def decorator(func: t.HandlerCallable) -> t.HandlerCallable:
            """Apply factory configuration metadata to function."""
            config = m.HandlerFactoryDecoratorConfig(
                name=name,
                singleton=singleton,
                lazy=lazy,
            )
            setattr(func, c.Discovery.FACTORY_ATTR, config)
            return func

        return decorator

    # FactoryDecoratorsDiscovery lives in _decorators/; use it or d from facade.


d = FlextDecorators

__all__ = ["FlextDecorators", "d"]
