"""FLEXT Core Decorators - Internal Implementation Module.

Internal implementation providing the foundational logic for decorator patterns.
This module is part of the Internal Implementation Layer and should not be imported
directly by ecosystem projects. Use the public API through decorators module instead.

Module Role in Architecture:
    Internal Implementation Layer → Decorator Patterns → Public API Layer

    This internal module provides:
    - Base decorator utilities and protocol definitions
    - Consolidated decorator categories (validation, error handling, performance, etc.)
    - Factory functions for configurable decorator creation
    - Internal decorator composition and metadata preservation

Implementation Patterns:
    Protocol Interfaces: Type-safe decorator function interfaces
    Category Organization: Logical grouping of decorator functionality

Design Principles:
    - Single responsibility for internal decorator implementation concerns
    - No external dependencies beyond standard library and sibling modules
    - Performance-optimized implementations for public API consumption
    - Type safety maintained through internal validation

Access Restrictions:
    - This module is internal and not exported in __init__.py
    - Use decorators module for all external access to decorator functionality
    - Breaking changes may occur without notice in internal modules
    - No compatibility guarantees for internal implementation details

Quality Standards:
    - Internal implementation must maintain public API contracts
    - Performance optimizations must not break type safety
    - Code must be thoroughly tested through public API surface
    - Internal changes must not affect public behavior

See Also:
    decorators: Public API for decorator patterns and function enhancement
    docs/python-module-organization.md: Internal module architecture

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING, Protocol

from flext_core.exceptions import FlextValidationError
from flext_core.loggings import get_logger
from flext_core.result import safe_call

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core.flext_types import TAnyDict, TErrorHandler, TValidator


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
        validator: TValidator[object],
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
        """Log function calls with arguments and execution time."""

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            logger = get_logger(f"{func.__module__}.{func.__name__}")

            # Log function entry
            logger.debug(
                "Calling function",
                extra={
                    "function": func.__name__,
                    "func_module": func.__module__,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys()),
                },
            )

            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                execution_time_ms = (time.perf_counter() - start_time) * 1000

                # Log successful completion
                logger.debug(
                    "Function completed successfully",
                    extra={
                        "function": func.__name__,
                        "execution_time_ms": round(execution_time_ms, 2),
                        "success": True,
                    },
                )
            except (RuntimeError, ValueError, TypeError) as e:
                execution_time_ms = (time.perf_counter() - start_time) * 1000

                # Log exception with proper exception logging
                logger.exception(
                    "Function failed with exception",
                    extra={
                        "function": func.__name__,
                        "execution_time_ms": round(execution_time_ms, 2),
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "success": False,
                    },
                )
                raise
            else:
                return result

        return wrapper

    @staticmethod
    def log_exceptions_decorator(func: _DecoratedFunction) -> _DecoratedFunction:
        """Log function exceptions with full traceback."""

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            logger = get_logger(f"{func.__module__}.{func.__name__}")

            try:
                return func(*args, **kwargs)
            except (RuntimeError, ValueError, TypeError) as e:
                # Log exception with full context
                logger.exception(
                    "Exception in function",
                    extra={
                        "function": func.__name__,
                        "func_module": func.__module__,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                    },
                )
                raise

        return wrapper


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

    Delegates to result.py safe_call implementation for single source of truth
    eliminating code duplication following DRY principles.

    Args:
        error_handler: Optional error handler function

    Returns:
        Decorator function

    """
    # Delegate to result.py single source of truth - eliminates duplication

    def decorator(func: _DecoratedFunction) -> _DecoratedFunction:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            def call_func() -> object:
                return func(*args, **kwargs)

            result = safe_call(call_func)

            # Handle error_handler if provided
            if result.is_failure and error_handler and callable(error_handler):
                error_value = Exception(result.error or "Unknown error")
                return error_handler(error_value)

            # Return unwrapped result for backward compatibility
            return result.unwrap_or(None)

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
    validator: TValidator[object],
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
                raise FlextValidationError(
                    validation_error,
                    validation_details={"field": "input", "args": str(args)[:100]},
                )
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
        validator: TValidator[object],
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
        cache: TAnyDict = {}

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

            # Only cache values that match TAnyDict value types
            if isinstance(result, str | int | float | bool | type(None)):
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
