"""Automation decorators for reducing code bloat across FLEXT ecosystem.

This module provides decorators that automatically handle:
- Dependency injection (@inject)
- Operation logging with context (@log_operation)
- Performance tracking (@track_performance)
- Railway pattern wrapping (@railway)
- Retry logic with exponential backoff (@retry)
- Operation timeout enforcement (@timeout)

These decorators significantly reduce boilerplate code in services, handlers,
and other components by automating infrastructure concerns. All decorators
integrate foundation modules (FlextLogger, FlextConstants, FlextExceptions)
for consistency across the ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar, cast

import structlog

from flext_core.constants import FlextConstants
from flext_core.container import FlextContainer
from flext_core.exceptions import FlextExceptions
from flext_core.loggings import FlextLogger
from flext_core.result import FlextResult

# Type variables for decorator signatures
P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")


class FlextDecorators:
    """Unified class for FLEXT automation decorators.

    Provides decorators that automatically handle infrastructure concerns:
    - Dependency injection
    - Operation logging with structured context
    - Performance tracking and metrics
    - Railway pattern wrapping
    - Retry logic with exponential backoff
    - Operation timeout enforcement

    All decorators integrate FlextLogger, FlextConstants, and FlextExceptions
    for consistency across the FLEXT ecosystem.

    Example:
        ```python
        from flext_core import FlextDecorators, FlextResult


        class MyService:
            @FlextDecorators.inject(repo=MyRepository)
            @FlextDecorators.log_operation("process_data")
            @FlextDecorators.track_performance()
            def process_data(self, data: dict, *, repo) -> FlextResult[dict]:
                return repo.save(data)
        ```

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
                for name in dependencies:
                    if name not in kwargs:
                        # Get from container
                        result = container.get(str(name))
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
            structlog.get_logger() otherwise.

        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                op_name = operation_name or func.__name__

                # Get logger from self if available, otherwise create one
                args_tuple = cast("tuple[object, ...]", args)
                first_arg = args_tuple[0] if args_tuple else None
                if first_arg is not None and hasattr(first_arg, "logger"):
                    potential_logger = getattr(first_arg, "logger")
                    logger = (
                        potential_logger
                        if isinstance(potential_logger, FlextLogger)
                        else FlextLogger(func.__module__)
                    )
                else:
                    logger = FlextLogger(func.__module__)

                # Bind operation context for structured logging
                structlog.contextvars.bind_contextvars(operation=op_name)

                try:
                    logger.info(
                        f"{op_name}_started",
                        extra={
                            "function": func.__name__,
                            "func_module": func.__module__,
                        },
                    )

                    result = func(*args, **kwargs)
                    logger.info(
                        f"{op_name}_completed",
                        extra={
                            "function": func.__name__,
                            "success": True,
                        },
                    )
                    return result
                except Exception as e:
                    logger.exception(
                        f"{op_name}_failed",
                        extra={
                            "function": func.__name__,
                            "success": False,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    raise
                finally:
                    # Unbind operation context
                    structlog.contextvars.unbind_contextvars("operation")

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
            can be used with _track_operation() for metrics collection.

        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                op_name = operation_name or func.__name__

                # Get logger from self if available, otherwise create one
                args_tuple = cast("tuple[object, ...]", args)
                first_arg = args_tuple[0] if args_tuple else None
                if first_arg is not None and hasattr(first_arg, "logger"):
                    potential_logger = getattr(first_arg, "logger")
                    logger = (
                        potential_logger
                        if isinstance(potential_logger, FlextLogger)
                        else FlextLogger(func.__module__)
                    )
                else:
                    logger = FlextLogger(func.__module__)

                start_time = time.perf_counter()

                # Bind operation context for structured logging
                structlog.contextvars.bind_contextvars(operation=op_name)

                try:
                    result = func(*args, **kwargs)
                    duration = time.perf_counter() - start_time

                    logger.info(
                        "operation_completed",
                        extra={
                            "operation": op_name,
                            "duration_ms": duration * 1000,
                            "duration_seconds": duration,
                            "success": True,
                        },
                    )
                    return result
                except Exception as e:
                    duration = time.perf_counter() - start_time

                    logger.exception(
                        "operation_failed",
                        extra={
                            "operation": op_name,
                            "duration_ms": duration * 1000,
                            "duration_seconds": duration,
                            "success": False,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    raise
                finally:
                    # Unbind operation context
                    structlog.contextvars.unbind_contextvars("operation")

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


            @FlextDecorators.railway(error_code="VALIDATION_ERROR")
            def validate_email(email: str) -> str:
                # object exception automatically becomes FlextResult.fail()
                # object success automatically becomes FlextResult.ok()
                if "@" not in email:
                    raise ValueError("Invalid email format")
                return email.lower()


            # Returns FlextResult[str] automatically
            result = validate_email("user@example.com")
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

                except Exception as e:
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
                    max_attempts=5, delay_seconds=2.0, backoff_strategy="exponential"
                )
                def unreliable_operation(self) -> dict:
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
                # Get logger from self if available
                args_tuple = cast("tuple[object, ...]", args)
                first_arg = args_tuple[0] if args_tuple else None
                if first_arg is not None and hasattr(first_arg, "logger"):
                    potential_logger = getattr(first_arg, "logger")
                    logger = (
                        potential_logger
                        if isinstance(potential_logger, FlextLogger)
                        else FlextLogger(func.__module__)
                    )
                else:
                    logger = FlextLogger(func.__module__)

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

                    except Exception as e:
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

            return wrapper

        return decorator

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
                def long_running_operation(self) -> dict:
                    # Automatically raises TimeoutError if exceeds 30 seconds
                    return self._process_data()
            ```

        Note:
            This is a simple timeout based on elapsed time checking. For true
            thread-based timeouts, use threading.Timer or asyncio.

        """
        # Use FlextConstants.Reliability for default
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
                except Exception as e:
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
                    inject_deps={"repo": OrderRepository, "validator": OrderValidator},
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
            decorated: object = func

            # Apply railway pattern first if requested (outermost wrapper)
            if use_railway:
                decorated = FlextDecorators.railway(error_code=error_code)(decorated)

            # Apply dependency injection
            if inject_deps:
                decorated = FlextDecorators.inject(**inject_deps)(decorated)

            # Apply performance tracking
            if track_perf:
                decorated = FlextDecorators.track_performance(
                    operation_name=operation_name
                )(decorated)

            # Apply operation logging (innermost wrapper)
            decorated = FlextDecorators.log_operation(operation_name=operation_name)(
                decorated
            )

            return cast("Callable[P, R]", decorated)

        return decorator


# Backward compatibility aliases (deprecated, use FlextDecorators.* instead)
inject = FlextDecorators.inject
log_operation = FlextDecorators.log_operation
track_performance = FlextDecorators.track_performance
railway = FlextDecorators.railway
retry = FlextDecorators.retry
timeout = FlextDecorators.timeout
combined = FlextDecorators.combined

__all__ = [
    "FlextDecorators",
    # Backward compatibility (deprecated)
    "combined",
    "inject",
    "log_operation",
    "railway",
    "retry",
    "timeout",
    "track_performance",
]
