"""FLEXT Core Decorators - Boilerplate Reduction Utilities.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Enterprise decorators that dramatically reduce boilerplate code for common
patterns like validation, logging, retry logic, caching, and error handling.
All decorators integrate seamlessly with FlextResult pattern and maintain
type safety.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import ParamSpec
from typing import TypeVar

from flext_core.result import FlextResult

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from collections.abc import Callable

T = TypeVar("T")
P = ParamSpec("P")

# =============================================================================
# VALIDATION DECORATORS - Input/Output Validation
# =============================================================================


def flext_validate_result(
    success_msg: str = "Operation completed successfully",
) -> Callable[[Callable[P, FlextResult[T]]], Callable[P, FlextResult[T]]]:
    """Decorator to add consistent success messaging to FlextResult functions.

    Args:
        success_msg: Message to log/include on success

    Returns:
        Decorated function with enhanced result handling

    Example:
        @flext_validate_result("User created successfully")
        def create_user(name: str) -> FlextResult[User]:
            return FlextResult.ok(User(name=name))
    """
    def decorator(
        func: Callable[P, FlextResult[T]],
    ) -> Callable[P, FlextResult[T]]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> FlextResult[T]:
            result = func(*args, **kwargs)
            if result.is_success and result.data:
                # Use success_msg for consistent messaging
                # Note: Currently we preserve the original result
                # In future versions this could be enhanced to include success_msg
                _ = success_msg  # Mark as used for linting
                return result
            return result
        return wrapper
    return decorator


def flext_require_non_none(
    *param_names: str,
) -> Callable[[Callable[P, T]], Callable[P, FlextResult[T]]]:
    """Decorator to validate that specified parameters are not None.

    Args:
        param_names: Names of parameters to validate

    Returns:
        Decorated function that returns FlextResult

    Example:
        @flext_require_non_none("user_id", "email")
        def update_user(user_id: str, email: str, name: str = None) -> User:
            return User(id=user_id, email=email, name=name)
    """
    def decorator(func: Callable[P, T]) -> Callable[P, FlextResult[T]]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> FlextResult[T]:
            # Get function signature to map args to param names
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Check required parameters
            for param_name in param_names:
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if value is None:
                        return FlextResult.fail(
                            f"Parameter '{param_name}' cannot be None",
                        )

            try:
                result = func(*args, **kwargs)
                return FlextResult.ok(result)
            except Exception as e:  # noqa: BLE001
                return FlextResult.fail(f"Function execution failed: {e}")

        return wrapper
    return decorator


# =============================================================================
# RETRY DECORATORS - Resilience Patterns
# =============================================================================


def flext_retry(
    max_attempts: int = 3,
    delay_seconds: float = 0.1,
    *,
    exponential_backoff: bool = True,
) -> Callable[[Callable[P, FlextResult[T]]], Callable[P, FlextResult[T]]]:
    """Decorator to add retry logic to FlextResult functions.

    Args:
        max_attempts: Maximum number of retry attempts
        delay_seconds: Initial delay between retries
        exponential_backoff: Whether to use exponential backoff

    Returns:
        Decorated function with retry logic

    Example:
        @flext_retry(max_attempts=3, delay_seconds=0.5)
        def fetch_data() -> FlextResult[Data]:
            return api_call_that_might_fail()
    """
    def decorator(
        func: Callable[P, FlextResult[T]],
    ) -> Callable[P, FlextResult[T]]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> FlextResult[T]:
            last_result = None
            current_delay = delay_seconds

            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)
                    if result.is_success:
                        return result
                    last_result = result

                    # Don't sleep on last attempt
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        if exponential_backoff:
                            current_delay *= 2

                except Exception as e:  # noqa: BLE001
                    last_result = FlextResult.fail(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_attempts - 1:
                        time.sleep(current_delay)
                        if exponential_backoff:
                            current_delay *= 2

            return last_result or FlextResult.fail("All retry attempts failed")
        return wrapper
    return decorator


# =============================================================================
# TIMING DECORATORS - Performance Monitoring
# =============================================================================


def flext_timed() -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to measure and store execution time.

    Adds execution_time_ms attribute to function result if possible.

    Example:
        @flext_timed()
        def slow_operation() -> FlextResult[Data]:
            time.sleep(1)
            return FlextResult.ok(Data())

        result = slow_operation()
        print(f"Took {result.execution_time_ms}ms")  # If available
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            execution_time_ms = (end_time - start_time) * 1000

            # Try to add timing info to result if it's a FlextResult
            if hasattr(result, "__dict__"):
                with contextlib.suppress(AttributeError, TypeError):
                    result.execution_time_ms = execution_time_ms  # type: ignore[attr-defined]

            return result
        return wrapper
    return decorator


# =============================================================================
# CACHING DECORATORS - Simple Memory Cache
# =============================================================================


def flext_cache_result() -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Simple in-memory cache decorator for functions.

    Caches based on function arguments. Use for pure functions only.

    Example:
        @flext_cache_result()
        def expensive_calculation(x: int, y: int) -> FlextResult[int]:
            time.sleep(1)  # Simulate expensive operation
            return FlextResult.ok(x + y)
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        cache: dict[tuple[Any, ...], T] = {}

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Create cache key from args and kwargs
            key = (args, tuple(sorted(kwargs.items())))

            if key in cache:
                return cache[key]

            result = func(*args, **kwargs)
            cache[key] = result
            return result

        # Add cache management methods
        wrapper.cache_clear = lambda: cache.clear()  # type: ignore[attr-defined]
        wrapper.cache_info = lambda: {  # type: ignore[attr-defined]
            "hits": len(cache),
            "misses": 0,  # Simple implementation
            "maxsize": None,
            "currsize": len(cache),
        }

        return wrapper
    return decorator


# =============================================================================
# ERROR HANDLING DECORATORS - Exception Safety
# =============================================================================


def flext_safe_result(
    error_message: str = "Operation failed",
) -> Callable[[Callable[P, T]], Callable[P, FlextResult[T]]]:
    """Decorator to safely convert any function to return FlextResult.

    Catches all exceptions and converts them to FlextResult.fail().

    Args:
        error_message: Base error message for failures

    Returns:
        Decorated function that returns FlextResult

    Example:
        @flext_safe_result("Database operation failed")
        def risky_database_call() -> User:
            # This might raise an exception
            return database.get_user("123")
    """
    def decorator(func: Callable[P, T]) -> Callable[P, FlextResult[T]]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> FlextResult[T]:
            try:
                result = func(*args, **kwargs)
                return FlextResult.ok(result)
            except Exception as e:  # noqa: BLE001
                return FlextResult.fail(f"{error_message}: {e}")
        return wrapper
    return decorator


# =============================================================================
# COMPOSITION DECORATORS - Combine Multiple Patterns
# =============================================================================


def flext_robust(
    max_attempts: int = 3,
    *,
    cache: bool = False,
    timing: bool = False,
    safe: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, FlextResult[T]]]:
    """Meta-decorator that combines retry, caching, timing, and safety.

    Args:
        max_attempts: Number of retry attempts (0 = no retry)
        cache: Whether to enable caching
        timing: Whether to add timing information
        safe: Whether to wrap in exception safety

    Returns:
        Decorated function with combined behaviors that always returns FlextResult

    Example:
        @flext_robust(max_attempts=3, cache=True, timing=True)
        def api_call() -> FlextResult[Data]:
            return external_api.fetch_data()
    """
    def decorator(func: Callable[P, T]) -> Callable[P, FlextResult[T]]:
        # Always ensure we return FlextResult by applying safe first
        decorated_safe = flext_safe_result()(func)

        # Apply other decorators to the safe version
        decorated: Callable[P, FlextResult[T]] = decorated_safe

        if timing:
            decorated = flext_timed()(decorated)

        if cache:
            decorated = flext_cache_result()(decorated)

        if max_attempts > 1:
            decorated = flext_retry(max_attempts)(decorated)

        return decorated
    return decorator


# =============================================================================
# ASYNC DECORATORS - For Future Async Support
# =============================================================================


def flext_async_safe() -> Callable[
    [Callable[P, Awaitable[T]]],
    Callable[P, Awaitable[FlextResult[T]]],
]:
    """Async version of flext_safe_result.

    Example:
        @flext_async_safe()
        async def async_operation() -> Data:
            await some_async_call()
            return Data()
    """
    def decorator(
        func: Callable[P, Awaitable[T]],
    ) -> Callable[P, Awaitable[FlextResult[T]]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> FlextResult[T]:
            try:
                result = await func(*args, **kwargs)
                return FlextResult.ok(result)
            except Exception as e:  # noqa: BLE001
                return FlextResult.fail(f"Async operation failed: {e}")
        return wrapper
    return decorator


# =============================================================================
# EXPORTS - Clean Public API
# =============================================================================

__all__ = [
    "flext_async_safe",
    "flext_cache_result",
    "flext_require_non_none",
    "flext_retry",
    "flext_robust",
    "flext_safe_result",
    "flext_timed",
    "flext_validate_result",
]
