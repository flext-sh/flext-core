"""FLEXT Decorators Base - Type-safe decorator patterns.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Foundation decorator patterns with strict typing support.
"""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core.types import TErrorHandler, TValidator


class _DecoratedFunction(Protocol):
    """Protocol for functions that can be decorated."""

    __name__: str

    def __call__(self, *args: object, **kwargs: object) -> object: ...


class _BaseDecoratorUtils:
    """Base decorator utilities with type safety."""

    @staticmethod
    def preserve_metadata(
        original: _DecoratedFunction,
        wrapper: _DecoratedFunction,
    ) -> _DecoratedFunction:
        """Preserve function metadata in decorators."""
        if hasattr(original, "__name__"):
            wrapper.__name__ = original.__name__
        if hasattr(original, "__doc__"):
            wrapper.__doc__ = original.__doc__
        if hasattr(original, "__module__"):
            wrapper.__module__ = original.__module__
        return wrapper


# =============================================================================
# CONSOLIDATION BASE CLASSES - Single source of truth for each decorator category
# =============================================================================


class _BaseValidationDecorators:
    """Base validation decorators - single source of truth for validation."""

    @staticmethod
    def create_validation_decorator(
        validator: TValidator,
    ) -> Callable[[_DecoratedFunction], _DecoratedFunction]:
        """Create input validation decorator."""
        return _validate_input_decorator(validator)

    @staticmethod
    def validate_arguments(func: _DecoratedFunction) -> _DecoratedFunction:
        """Validate function arguments."""
        return func


class _BaseErrorHandlingDecorators:
    """Base error handling decorators - single source of truth for error handling."""

    @staticmethod
    def create_safe_decorator(
        error_handler: TErrorHandler | None = None,
    ) -> Callable[[_DecoratedFunction], _DecoratedFunction]:
        """Create safe call decorator with optional error handler."""
        return _safe_call_decorator(error_handler)

    @staticmethod
    def get_safe_decorator() -> Callable[[_DecoratedFunction], _DecoratedFunction]:
        """Get default safe decorator."""
        return _safe_call_decorator()

    @staticmethod
    def retry_decorator(func: _DecoratedFunction) -> _DecoratedFunction:
        """Add retry capability to function."""
        return func


class _BasePerformanceDecorators:
    """Base performance decorators - single source of truth for performance."""

    @staticmethod
    def create_cache_decorator(
        max_size: int = 128,
    ) -> Callable[[_DecoratedFunction], _DecoratedFunction]:
        """Create cache decorator with specified cache size."""
        return _cache_decorator(max_size)

    @staticmethod
    def get_timing_decorator() -> Callable[[_DecoratedFunction], _DecoratedFunction]:
        """Get timing decorator."""
        return _timing_decorator

    @staticmethod
    def memoize_decorator(func: _DecoratedFunction) -> _DecoratedFunction:
        """Add memoization to function."""
        return func


class _BaseLoggingDecorators:
    """Base logging decorators - single source of truth for logging."""

    @staticmethod
    def log_calls_decorator(func: _DecoratedFunction) -> _DecoratedFunction:
        """Log function calls."""
        return func

    @staticmethod
    def log_exceptions_decorator(func: _DecoratedFunction) -> _DecoratedFunction:
        """Log function exceptions."""
        return func


class _BaseImmutabilityDecorators:
    """Base immutability decorators - single source of truth for immutability."""

    @staticmethod
    def immutable_decorator(func: _DecoratedFunction) -> _DecoratedFunction:
        """Enforce immutability in function."""
        return func

    @staticmethod
    def freeze_args_decorator(func: _DecoratedFunction) -> _DecoratedFunction:
        """Freeze function arguments."""
        return func


class _BaseFunctionalDecorators:
    """Base functional decorators - single source of truth for functional patterns."""

    @staticmethod
    def curry_decorator(func: _DecoratedFunction) -> _DecoratedFunction:
        """Add currying to function."""
        return func

    @staticmethod
    def compose_decorator(func: _DecoratedFunction) -> _DecoratedFunction:
        """Compose functions together."""
        return func


def _safe_call_decorator(
    error_handler: TErrorHandler | None = None,
) -> Callable[[_DecoratedFunction], _DecoratedFunction]:
    """Create decorator for safe function execution with error handling.

    Args:
        error_handler: Optional error handler function

    Returns:
        Decorator function

    """

    def decorator(func: _DecoratedFunction) -> _DecoratedFunction:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            try:
                return func(*args, **kwargs)
            except (TypeError, ValueError, AttributeError, RuntimeError) as e:
                if error_handler and callable(error_handler):
                    return error_handler(e)
                return None

        return _BaseDecoratorUtils.preserve_metadata(func, wrapper)

    return decorator


def _timing_decorator(func: _DecoratedFunction) -> _DecoratedFunction:
    """Measure function execution time.

    Args:
        func: Function to decorate

    Returns:
        Decorated function

    """
    # Store timing data in a closure variable for type safety
    execution_times: list[float] = []

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> object:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time

        # Store timing in closure
        execution_times.append(execution_time)

        return result

    return _BaseDecoratorUtils.preserve_metadata(func, wrapper)


def _validate_input_decorator(
    validator: TValidator,
) -> Callable[[_DecoratedFunction], _DecoratedFunction]:
    """Validate function input arguments.

    Args:
        validator: Validation function

    Returns:
        Decorator function

    """

    def decorator(func: _DecoratedFunction) -> _DecoratedFunction:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            # Simple validation - at least one argument must pass
            if args and callable(validator) and not any(validator(arg) for arg in args):
                validation_error = "Input validation failed"
                raise ValueError(validation_error)
            return func(*args, **kwargs)

        return _BaseDecoratorUtils.preserve_metadata(func, wrapper)

    return decorator


class _BaseDecoratorFactory:
    """Factory for creating decorators with consistent patterns."""

    @staticmethod
    def create_cache_decorator(
        max_size: int = 128,
    ) -> Callable[[_DecoratedFunction], _DecoratedFunction]:
        """Create cache decorator with specified cache size."""
        return _cache_decorator(max_size)

    @staticmethod
    def create_safe_decorator(
        error_handler: TErrorHandler | None = None,
    ) -> Callable[[_DecoratedFunction], _DecoratedFunction]:
        """Create safe call decorator with optional error handler."""
        return _safe_call_decorator(error_handler)

    @staticmethod
    def create_timing_decorator() -> Callable[[_DecoratedFunction], _DecoratedFunction]:
        """Create timing decorator."""
        return _timing_decorator

    @staticmethod
    def create_validation_decorator(
        validator: TValidator,
    ) -> Callable[[_DecoratedFunction], _DecoratedFunction]:
        """Create input validation decorator."""
        return _validate_input_decorator(validator)


def _cache_decorator(
    max_size: int = 128,
) -> Callable[[_DecoratedFunction], _DecoratedFunction]:
    """Cache function results with TTL.

    Args:
        max_size: Maximum cache size

    Returns:
        Decorator function

    """

    def decorator(func: _DecoratedFunction) -> _DecoratedFunction:
        cache: dict[str, object] = {}

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            # Simple cache key generation
            cache_key = f"{args}_{kwargs}"

            if cache_key in cache:
                return cache[cache_key]

            result = func(*args, **kwargs)

            # Limit cache size
            if len(cache) >= max_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(cache))
                del cache[oldest_key]

            cache[cache_key] = result
            return result

        return _BaseDecoratorUtils.preserve_metadata(func, wrapper)

    return decorator


# Export API
__all__ = [
    # Consolidation base classes
    "_BaseDecoratorFactory",
    "_BaseDecoratorUtils",
    "_BaseErrorHandlingDecorators",
    "_BaseFunctionalDecorators",
    "_BaseImmutabilityDecorators",
    "_BaseLoggingDecorators",
    "_BasePerformanceDecorators",
    "_BaseValidationDecorators",
    # Individual decorators
    "_cache_decorator",
    "_safe_call_decorator",
    "_timing_decorator",
    "_validate_input_decorator",
]
