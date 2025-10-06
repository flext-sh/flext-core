"""Automation decorators for reducing code bloat across FLEXT ecosystem.

This module provides decorators that automatically handle:
- Dependency injection (@inject)
- Operation logging with context (@log_operation)
- Performance tracking (@track_performance)
- Railway pattern wrapping (@railway)

These decorators significantly reduce boilerplate code in services, handlers,
and other components by automating infrastructure concerns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

import structlog

from flext_core.container import FlextContainer
from flext_core.result import FlextResult

# Type variables for decorator signatures
P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")


def inject(**dependencies: type) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to automatically inject dependencies from FlextContainer.

    Automatically resolves and injects dependencies from the global container,
    eliminating manual container.resolve() calls in every method.

    Args:
        **dependencies: Mapping of parameter names to service types to inject

    Returns:
        Decorated function with automatic dependency injection

    Example:
        ```python
        from flext_core.decorators import inject
        from flext_core import FlextMixins


        class MyService(FlextMixins.Service):
            @inject(repo=MyRepository, logger=FlextLogger)
            def process_data(self, data: dict, *, repo, logger) -> FlextResult[dict]:
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


def log_operation(
    operation_name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to automatically log operation execution with structured logging.

    Automatically logs operation start, completion, and failures with structured
    context, eliminating manual logging boilerplate.

    Args:
        operation_name: Name for the operation (defaults to function name)

    Returns:
        Decorated function with automatic operation logging

    Example:
        ```python
        from flext_core.decorators import log_operation
        from flext_core import FlextMixins


        class MyService(FlextMixins.Service):
            @log_operation("process_user_data")
            def process(self, user_id: str) -> FlextResult[dict]:
                # Automatic logging of start/complete/failure
                # Automatic context propagation
                return self._do_processing(user_id)
        ```

    Note:
        Works best with classes that have logger attribute (FlextMixins.Service).
        Falls back to structlog.get_logger() otherwise.

    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            op_name = operation_name or func.__name__

            # Get logger from self if available
            if args and hasattr(args[0], "logger"):
                logger = getattr(args[0], "logger", structlog.get_logger())
            else:
                logger = structlog.get_logger()

            # Bind operation context
            structlog.contextvars.bind_contextvars(operation=op_name)

            logger.info(
                f"{op_name}_started",
                function=func.__name__,
                module=func.__module__,
            )

            try:
                result = func(*args, **kwargs)
                logger.info(
                    f"{op_name}_completed",
                    function=func.__name__,
                    success=True,
                )
                return result
            except Exception:
                logger.exception(
                    f"{op_name}_failed",
                    function=func.__name__,
                    success=False,
                )
                raise
            finally:
                # Unbind operation context
                structlog.contextvars.unbind_contextvars("operation")

        return wrapper

    return decorator


def track_performance(
    operation_name: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to automatically track operation performance metrics.

    Tracks operation duration and logs performance metrics with structured logging,
    enabling performance monitoring without manual timing code.

    Args:
        operation_name: Name for the operation (defaults to function name)

    Returns:
        Decorated function with automatic performance tracking

    Example:
        ```python
        from flext_core.decorators import track_performance
        from flext_core import FlextMixins


        class MyService(FlextMixins.Service):
            @track_performance("heavy_computation")
            def compute(self, data: list) -> FlextResult[float]:
                # Automatic timing and performance logging
                return self._expensive_calculation(data)
        ```

    Note:
        Performance metrics are logged to structured logging context
        and can be used with _track_operation() for metrics collection.

    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            op_name = operation_name or func.__name__

            # Get logger from self if available
            if args and hasattr(args[0], "logger"):
                logger = getattr(args[0], "logger", structlog.get_logger())
            else:
                logger = structlog.get_logger()

            start_time = time.perf_counter()
            structlog.contextvars.bind_contextvars(operation=op_name)

            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time

                logger.info(
                    "operation_completed",
                    operation=op_name,
                    duration_ms=duration * 1000,
                    duration_seconds=duration,
                    success=True,
                )
                return result
            except Exception:
                duration = time.perf_counter() - start_time

                logger.exception(
                    "operation_failed",
                    operation=op_name,
                    duration_ms=duration * 1000,
                    duration_seconds=duration,
                    success=False,
                )
                raise
            finally:
                structlog.contextvars.unbind_contextvars("operation")

        return wrapper

    return decorator


def railway(
    error_code: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, FlextResult[T]]]:
    """Decorator to automatically wrap function in railway pattern.

    Automatically converts exceptions to FlextResult failures and successful
    returns to FlextResult successes, eliminating manual try/except boilerplate.

    Args:
        error_code: Optional error code for failures

    Returns:
        Decorated function that returns FlextResult[T]

    Example:
        ```python
        from flext_core.decorators import railway
        from flext_core import FlextResult


        @railway(error_code="VALIDATION_ERROR")
        def validate_email(email: str) -> str:
            # Any exception automatically becomes FlextResult.fail()
            # Any success automatically becomes FlextResult.ok()
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


def combined(
    *,
    inject_deps: dict[str, type] | None = None,
    operation_name: str | None = None,
    track_perf: bool = True,
    use_railway: bool = False,
    error_code: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Combined decorator applying multiple automation patterns at once.

    Combines @inject, @log_operation, @track_performance, and optionally @railway
    into a single decorator for maximum code reduction.

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
        from flext_core.decorators import combined
        from flext_core import FlextMixins, FlextResult


        class OrderService(FlextMixins.Service):
            @combined(
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
        This decorator provides maximum automation but use judiciously
        to maintain code clarity.

    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        # Start with the base function
        decorated: Any = func

        # Apply railway pattern first if requested (outermost wrapper)
        if use_railway:
            decorated = railway(error_code=error_code)(decorated)

        # Apply dependency injection
        if inject_deps:
            decorated = inject(**inject_deps)(decorated)

        # Apply performance tracking
        if track_perf:
            decorated = track_performance(operation_name=operation_name)(decorated)

        # Apply operation logging (innermost wrapper)
        decorated = log_operation(operation_name=operation_name)(decorated)

        return cast("Callable[P, R]", decorated)

    return decorator


__all__ = [
    "combined",
    "inject",
    "log_operation",
    "railway",
    "track_performance",
]
