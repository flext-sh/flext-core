"""Automation decorators for infrastructure concerns.

This module provides FlextDecorators, a collection of decorators that
automatically handle common infrastructure concerns to reduce boilerplate
code in services, handlers, and other components.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import suppress
from functools import wraps
from typing import cast

from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.exceptions import FlextExceptions
from flext_core.loggings import FlextLogger
from flext_core.result import FlextResult
from flext_core.typings import P, R, T


class FlextDecorators:
    """Automation decorators for infrastructure concerns.

    Architecture: Layer 3 (Application Layer - Cross-Cutting Concerns)
    ==================================================================
    Provides decorators that automatically handle common infrastructure
    concerns to reduce boilerplate code in services, handlers, and other
    components. All decorators are designed to integrate seamlessly with
    FlextResult, FlextContext, FlextLogger, and FlextContainer.

    **Structural Typing and Protocol Compliance**:
    This class satisfies FlextProtocols.Decorator through structural typing
    (duck typing) via the following protocol-compliant interface:
    - 8 static methods providing cross-cutting concern automation
    - Automatic context propagation and cleanup (defensive programming)
    - Integration with Foundation layer (FlextResult, FlextContext)
    - Integration with Infrastructure layer (FlextLogger, FlextContainer)
    - Composable decorator patterns with correct wrapper ordering

    **Core Decorators** (8 patterns):
    1. **@inject**: Automatic dependency injection from FlextContainer
       - Eliminates manual container.resolve() calls
       - Keyword-only parameter injection
       - Graceful fallback on resolution failure
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
       - Raises FlextExceptions.TimeoutError on violation
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
    - **FlextExceptions** (Layer 1): TimeoutError for @timeout
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
        >>> from flext_core import FlextDecorators, FlextResult
        >>>
        >>> class UserService:
        ...     @FlextDecorators.inject(repo=UserRepository)
        ...     def get_user(self, user_id: str, *, repo) -> FlextResult[User]:
        ...         return repo.find_by_id(user_id)

    2. Automatic operation logging:
        >>> class OrderService:
        ...     @FlextDecorators.log_operation("create_order")
        ...     def create_order(self, order_data: dict) -> FlextResult[Order]:
        ...         # Start/completion/failure automatically logged
        ...         return self._process_order(order_data)

    3. Performance tracking:
        >>> class ReportService:
        ...     @FlextDecorators.track_performance("generate_report")
        ...     def generate_report(self, params: dict) -> FlextResult[Report]:
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
        ...     def call_external_api(self, endpoint: str) -> dict[str, object]:
        ...         # Automatically retries on failure with backoff
        ...         return requests.get(endpoint).json()

    6. Operation timeout enforcement:
        >>> class LongRunningService:
        ...     @FlextDecorators.timeout(timeout_seconds=30.0)
        ...     def expensive_operation(self, data: list) -> FlextResult[float]:
        ...         # Raises TimeoutError if exceeds 30 seconds
        ...         return self._expensive_computation(data)

    7. Correlation ID management:
        >>> class PaymentService:
        ...     @FlextDecorators.with_correlation()
        ...     def process_payment(self, payment_id: str) -> FlextResult[str]:
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
        ...     ) -> FlextResult[Order]:
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
        ...     def process_tenant_data(self) -> FlextResult[dict]:
        ...         # tenant_id and user_id bound to context
        ...         # All logs include these values
        ...         return self._process()

    10. Comprehensive operation tracking:
        >>> class CriticalService:
        ...     @FlextDecorators.track_operation(
        ...         operation_name="critical_process", track_correlation=True
        ...     )
        ...     def critical_process(self) -> FlextResult[str]:
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
        ... def business_operation() -> dict[str, object]:
        ...     # Any exception becomes FlextResult.fail()
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
        ...     def create_product(
        ...         self, product_data: dict, *, repo
        ...     ) -> FlextResult[Product]:
        ...         # Automatic infrastructure:
        ...         # - Dependency injection
        ...         # - Structured logging
        ...         # - Performance tracking
        ...         # - Railway pattern
        ...         # - Exception handling
        ...         # - Context propagation
        ...
        ...         return (
        ...             FlextResult[dict]
        ...             .ok(product_data)
        ...             .flat_map(self._validate)
        ...             .flat_map(repo.save)
        ...         )
        ...
        ...     def _validate(self, data: dict) -> FlextResult[dict]:
        ...         if "name" not in data:
        ...             return FlextResult[dict].fail("Name required")
        ...         return FlextResult[dict].ok(data)
    """

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
                def process_data(
                    self, data: dict, *, repo, logger
                ) -> FlextResult[dict]:
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
                # Get container from self if available, otherwise use global
                container = FlextContainer.get_global()

                # Inject dependencies that aren't already provided
                for name, service_key in dependencies.items():
                    if name not in kwargs:
                        # Get from container using the service key
                        result = container.get(service_key)
                        if result.is_success:
                            kwargs[name] = result.unwrap()
                        else:
                            # If resolution fails, let the function handle it
                            # or fail with missing parameter
                            pass

                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def log_operation(
        operation_name: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to automatically log operation execution with structured logging.

        Automatically logs operation start, completion, and failures with
        structured context, eliminating manual logging boilerplate.

        Args:
            operation_name: Name for the operation (defaults to function name)

        Returns:
            Decorated function with automatic operation logging

        Example:
            ```python
            from flext_core import FlextDecorators, FlextResult


            class MyService:
                @FlextDecorators.log_operation("process_user_data")
                def process(self, user_id: str) -> FlextResult[dict]:
                    # Automatic logging of start/complete/failure
                    # Automatic context propagation
                    return self._do_processing(user_id)
            ```

        Note:
            Works best with classes that have logger attribute. Falls back to
            FlextLogger.get_logger() otherwise.

        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                op_name = operation_name or func.__name__

                # Get logger from self if available, otherwise create one
                args_tuple = cast("tuple[object, ...]", args)
                logger = FlextDecorators._resolve_logger(args_tuple, func)

                correlation_id = FlextDecorators._bind_operation_context(
                    operation=op_name,
                    logger=logger,
                    function_name=func.__name__,
                    ensure_correlation=True,
                )

                try:
                    start_extra: dict[str, object] = {
                        "function": func.__name__,
                        "func_module": func.__module__,
                    }
                    if correlation_id is not None:
                        start_extra["correlation_id"] = correlation_id

                    log_start_result = logger.info(
                        f"{op_name}_started",
                        extra=start_extra,
                    )
                    FlextDecorators._handle_log_result(
                        result=log_start_result,
                        logger=logger,
                        fallback_message="operation_log_emit_failed",
                        kwargs={
                            "extra": {**start_extra, "log_state": "start"},
                        },
                    )

                    result = func(*args, **kwargs)
                    completion_extra: dict[str, object] = {
                        "function": func.__name__,
                        "success": True,
                    }
                    if correlation_id is not None:
                        completion_extra["correlation_id"] = correlation_id
                    log_completion_result = logger.info(
                        f"{op_name}_completed",
                        extra=completion_extra,
                    )
                    FlextDecorators._handle_log_result(
                        result=log_completion_result,
                        logger=logger,
                        fallback_message="operation_log_emit_failed",
                        kwargs={
                            "extra": {**completion_extra, "log_state": "completed"},
                        },
                    )
                    return result
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as e:
                    failure_extra: dict[str, object] = {
                        "function": func.__name__,
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    if correlation_id is not None:
                        failure_extra["correlation_id"] = correlation_id
                    failure_result = logger.exception(
                        f"{op_name}_failed",
                        exception=e,
                        extra=failure_extra,
                    )
                    FlextDecorators._handle_log_result(
                        result=failure_result,
                        logger=logger,
                        fallback_message="operation_log_emit_failed",
                        kwargs={
                            "extra": {
                                **failure_extra,
                                "log_state": "failed",
                                "exception_repr": repr(e),
                            },
                        },
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
                def compute(self, data: list) -> FlextResult[float]:
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
                op_name = operation_name or func.__name__

                # Get logger from self if available, otherwise create one
                args_tuple = cast("tuple[object, ...]", args)
                logger = FlextDecorators._resolve_logger(args_tuple, func)

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

                    success_extra: dict[str, object] = {
                        "operation": op_name,
                        "duration_ms": duration * 1000,
                        "duration_seconds": duration,
                        "success": True,
                    }
                    if correlation_id is not None:
                        success_extra["correlation_id"] = correlation_id
                    logger.info(
                        "operation_completed",
                        extra=success_extra,
                    )
                    return result
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as e:
                    duration = time.perf_counter() - start_time

                    failure_extra: dict[str, object] = {
                        "operation": op_name,
                        "duration_ms": duration * 1000,
                        "duration_seconds": duration,
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    if correlation_id is not None:
                        failure_extra["correlation_id"] = correlation_id
                    logger.exception(
                        "operation_failed",
                        extra=failure_extra,
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
    ) -> Callable[[Callable[P, T]], Callable[P, FlextResult[T]]]:
        """Decorator to automatically wrap function in railway pattern.

        Automatically converts exceptions to FlextResult failures and
        successful returns to FlextResult successes, eliminating manual
        try/except boilerplate.

        Args:
            error_code: Optional error code for failures

        Returns:
            Decorated function that returns FlextResult[T]

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


            # Returns FlextResult[str] automatically
            result = process_user_email("user@example.com")
            assert result.is_success
            ```

        Note:
            If the function already returns a FlextResult, it's returned as-is.
            Only bare values or exceptions are wrapped.

        """

        def decorator(func: Callable[P, T]) -> Callable[P, FlextResult[T]]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> FlextResult[T]:
                try:
                    result = func(*args, **kwargs)

                    # If already a FlextResult, return as-is
                    if isinstance(result, FlextResult):
                        return cast("FlextResult[T]", result)

                    # Wrap successful result
                    return FlextResult[T].ok(result)

                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as e:
                    # Convert exception to FlextResult failure
                    return FlextResult[T].fail(
                        str(e),
                        error_code=error_code or "OPERATION_ERROR",
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
        FlextExceptions for structured error handling, integrating foundation
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
                    backoff_strategy="exponential",
                )
                def unreliable_operation(self) -> dict[str, object]:
                    # Automatically retries on failure with exponential backoff
                    return self._make_api_call()
            ```

        Note:
            Uses FlextConstants.Reliability for defaults, ensuring consistency
            across the entire ecosystem. Logs retry attempts automatically.

        """
        # Use FlextConstants.Reliability for defaults
        attempts = max_attempts or FlextConstants.Reliability.DEFAULT_MAX_RETRIES
        delay = delay_seconds or float(
            FlextConstants.Reliability.DEFAULT_RETRY_DELAY_SECONDS
        )
        strategy = (
            backoff_strategy or FlextConstants.Reliability.DEFAULT_BACKOFF_STRATEGY
        )

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                logger = FlextDecorators._get_logger_for_retry(args, func)
                result_or_exception = FlextDecorators._execute_retry_loop(
                    func, args, kwargs, logger, attempts, delay, strategy
                )
                # Check if we got a successful result or an exception
                if isinstance(result_or_exception, Exception):
                    # All retries failed - handle exhaustion and raise
                    FlextDecorators._handle_retry_exhaustion(
                        result_or_exception, func, attempts, error_code, logger
                    )
                    # Unreachable, but needed for type checking
                    msg = f"Operation {func.__name__} failed"
                    raise FlextExceptions.TimeoutError(
                        msg, error_code=error_code or "RETRY_EXHAUSTED"
                    )
                # Success - return the result
                return result_or_exception

            return wrapper

        return decorator

    @staticmethod
    def _resolve_logger(
        args: tuple[object, ...],
        func: Callable[..., object],
    ) -> FlextLogger:
        """Resolve logger from first argument or create module logger."""
        first_arg = args[0] if args else None
        potential_logger: object | None = (
            getattr(first_arg, "logger", None) if first_arg is not None else None
        )
        if potential_logger is not None and isinstance(potential_logger, FlextLogger):
            return potential_logger
        return FlextLogger(func.__module__)

    @staticmethod
    def _get_logger_for_retry(
        args: tuple[object, ...], func: Callable[..., object]
    ) -> FlextLogger:
        """Get logger from self if available, otherwise create new logger."""
        return FlextDecorators._resolve_logger(args, func)

    @staticmethod
    def _execute_retry_loop[R](
        func: Callable[..., R],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        logger: FlextLogger,
        attempts: int,
        delay: float,
        strategy: str,
    ) -> R | Exception:
        """Execute retry loop and return last exception."""
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
                        "error_type": type(e).__name__,
                    },
                )

                # Calculate next delay based on strategy
                if strategy == "exponential":
                    current_delay *= 2
                elif strategy == "linear":
                    current_delay += delay
                # For unknown strategies, keep constant delay

                # If this was the last attempt, we'll raise after loop
                if attempt == attempts:
                    break

        # Should never be None (loop always catches exceptions), but handle defensively
        if last_exception is None:
            return RuntimeError("Retry loop completed without success or exception")
        return last_exception

    @staticmethod
    def _handle_retry_exhaustion(
        last_exception: Exception,
        func: Callable[..., object],
        attempts: int,
        error_code: str | None,
        logger: FlextLogger,
    ) -> None:
        """Handle retry exhaustion and raise appropriate exception."""
        # All retries exhausted
        if last_exception:
            logger.error(
                "operation_failed_all_retries_exhausted",
                extra={
                    "function": func.__name__,
                    "attempts": attempts,
                    "error": str(last_exception),
                    "error_type": type(last_exception).__name__,
                },
            )
        else:
            logger.error(
                "operation_failed_all_retries_exhausted",
                extra={
                    "function": func.__name__,
                    "attempts": attempts,
                    "error": "Unknown error",
                },
            )

        # Raise the last exception
        if last_exception:
            raise last_exception

        # Should never reach here, but type safety
        msg = f"Operation {func.__name__} failed after {attempts} attempts"
        raise FlextExceptions.TimeoutError(
            msg,
            error_code=error_code or "RETRY_EXHAUSTED",
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
            current_id = FlextContext.Variables.Correlation.CORRELATION_ID.get()
            if isinstance(current_id, str) and current_id:
                correlation_id = current_id

        FlextContext.Request.set_operation_name(operation)

        binding_result = FlextLogger.bind_operation_context(operation=operation)
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
        if clear_result.is_failure:
            FlextDecorators._handle_log_result(
                result=clear_result,
                logger=logger,
                fallback_message="operation_context_clear_failed",
                kwargs={
                    "extra": {
                        "function": function_name,
                        "operation": operation,
                    }
                },
            )

    @staticmethod
    def _handle_log_result(
        *,
        result: FlextResult[None],
        logger: FlextLogger,
        fallback_message: str,
        kwargs: dict[str, object],
    ) -> None:
        """Ensure FlextLogger call results are handled for diagnostics."""
        if result.is_failure:
            fallback_logger = getattr(logger, "logger", None)
            if fallback_logger is None or not hasattr(fallback_logger, "warning"):
                return
            fallback_kwargs = dict(kwargs)
            fallback_kwargs.setdefault("extra", {})
            extra_payload = fallback_kwargs["extra"]
            if isinstance(extra_payload, dict):
                extra_payload = dict(extra_payload)
                extra_payload["log_error"] = result.error
                extra_payload["log_error_code"] = result.error_code
                fallback_kwargs["extra"] = extra_payload
            else:
                fallback_kwargs["log_error"] = result.error
                fallback_kwargs["log_error_code"] = result.error_code
            fallback_logger.warning(fallback_message, **fallback_kwargs)

    @staticmethod
    def timeout(
        timeout_seconds: float | None = None,
        error_code: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Decorator to enforce operation timeout.

        Uses FlextConstants.Reliability.DEFAULT_TIMEOUT_SECONDS for default
        timeout and FlextExceptions.TimeoutError for structured error
        handling.

        Args:
            timeout_seconds: Timeout in seconds (default:
                FlextConstants.Reliability.DEFAULT_TIMEOUT_SECONDS)
            error_code: Optional error code for timeout

        Returns:
            Decorated function with timeout enforcement

        Example:
            ```python
            from flext_core import FlextDecorators


            class MyService:
                @FlextDecorators.timeout(timeout_seconds=30.0)
                def long_running_operation(self) -> dict[str, object]:
                    # Automatically raises TimeoutError if exceeds 30 seconds
                    return self._process_data()
            ```

        Note:
            This is a simple timeout based on elapsed time checking. For true
            thread-based timeouts, use threading.Timer or asyncio.

        """
        # Use FlextConstants.Defaults for default
        max_duration = (
            timeout_seconds
            if timeout_seconds is not None
            else FlextConstants.Reliability.DEFAULT_TIMEOUT_SECONDS
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
                        raise FlextExceptions.TimeoutError(
                            msg,
                            error_code=error_code or "OPERATION_TIMEOUT",
                            timeout_seconds=max_duration,
                            operation=func.__name__,
                            metadata={"duration_seconds": duration},
                        )

                    return result

                except FlextExceptions.TimeoutError:
                    # Re-raise timeout errors
                    raise
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    KeyError,
                ) as e:
                    # Check duration even on exception
                    duration = time.perf_counter() - start_time
                    if duration > max_duration:
                        msg = f"Operation {func.__name__} exceeded timeout of {max_duration}s (took {duration:.2f}s) and raised {type(e).__name__}"
                        raise FlextExceptions.TimeoutError(
                            msg,
                            error_code=error_code or "OPERATION_TIMEOUT",
                            timeout_seconds=max_duration,
                            operation=func.__name__,
                            metadata={
                                "duration_seconds": duration,
                                "original_error": str(e),
                            },
                        ) from e
                    # Re-raise original exception if not timeout
                    raise

            return wrapper

        return decorator

    @staticmethod
    def combined(
        *,
        inject_deps: dict[str, str] | None = None,
        operation_name: str | None = None,
        track_perf: bool = True,
        use_railway: bool = False,
        error_code: str | None = None,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Combined decorator applying multiple automation patterns at once.

        Combines @inject, @log_operation, @track_performance, and optionally
        @railway into a single decorator for maximum code reduction.

        Args:
            inject_deps: Dependencies to inject (name -> type mapping)
            operation_name: Name for logging (defaults to function name)
            track_perf: Whether to track performance (default: True)
            use_railway: Whether to apply railway pattern (default: False)
            error_code: Error code for railway pattern failures

        Returns:
            Decorated function with all requested automations

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
                ) -> FlextResult[Order]:
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

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            # Start with the base function
            decorated: Callable[..., R] = func

            # Apply railway pattern first if requested (outermost wrapper)
            if use_railway:
                railway_result = FlextDecorators.railway(error_code=error_code)(
                    decorated
                )
                decorated = cast("Callable[..., R]", railway_result)

            # Apply dependency injection
            if inject_deps:
                inject_result = FlextDecorators.inject(**inject_deps)(decorated)
                decorated = inject_result

            # Apply performance tracking
            if track_perf:
                perf_result = FlextDecorators.track_performance(
                    operation_name=operation_name
                )(decorated)
                decorated = perf_result

            # Apply operation logging (innermost wrapper)
            log_result = FlextDecorators.log_operation(operation_name=operation_name)(
                decorated
            )
            return cast("Callable[P, R]", log_result)

        return decorator

    # Backward compatibility class methods (deprecated, use static methods directly)

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
                def process_order(self, order_id: str) -> FlextResult[dict]:
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
                FlextContext.Utilities.ensure_correlation_id()

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
                def process_payment(self, amount: float) -> FlextResult[str]:
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
                args_tuple = cast("tuple[object, ...]", args)
                logger = FlextDecorators._resolve_logger(args_tuple, func)

                try:
                    # Bind context variables to global logging context
                    if context_vars:
                        bind_result = FlextLogger.bind_global_context(**context_vars)
                        if bind_result.is_failure:
                            logger.warning(
                                "global_context_binding_failed",
                                extra={
                                    "function": func.__name__,
                                    "error": bind_result.error,
                                    "error_code": bind_result.error_code,
                                    "bound_keys": tuple(context_vars.keys()),
                                },
                            )

                    return func(*args, **kwargs)

                finally:
                    # Unbind context variables
                    if context_vars:
                        unbind_result = FlextLogger.unbind_global_context(
                            *tuple(context_vars.keys())
                        )
                        if unbind_result.is_failure:
                            logger.warning(
                                "global_context_unbind_failed",
                                extra={
                                    "function": func.__name__,
                                    "error": unbind_result.error,
                                    "error_code": unbind_result.error_code,
                                    "bound_keys": tuple(context_vars.keys()),
                                },
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
                def create_user(self, user_data: dict) -> FlextResult[User]:
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
                op_name = operation_name or func.__name__

                args_tuple = cast("tuple[object, ...]", args)
                logger = FlextDecorators._resolve_logger(args_tuple, func)

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
                    # Performance tracking via FlextRuntime.Integration happens automatically
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


__all__ = [
    "FlextDecorators",
]
