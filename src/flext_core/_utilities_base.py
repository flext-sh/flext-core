"""FLEXT Base Utilities - Foundation utilities.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Foundation utility functions with performance tracking.
Only imports from immediate base modules.
"""

from __future__ import annotations

import hashlib
import random
import re
import string
import time
from datetime import UTC, datetime
from functools import wraps
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Callable

# Performance metrics storage - private
_performance_metrics: dict[str, dict[str, object]] = {}

# Time constants for formatting
_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600
_BYTES_PER_KB = 1024


class _DecoratedFunction(Protocol):
    """Protocol for functions that can be decorated with performance tracking."""

    __name__: str

    def __call__(self, *args: object, **kwargs: object) -> object: ...


def _track_performance(
    category: str,
) -> Callable[[_DecoratedFunction], _DecoratedFunction]:
    """Track function performance as decorator.

    Args:
        category: Category of the function to track

    Returns:
        Decorator function

    """

    def _decorator(func: _DecoratedFunction) -> _DecoratedFunction:
        @wraps(func)
        def _wrapper(*args: object, **kwargs: object) -> object:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                _record_performance(
                    category,
                    func.__name__,
                    execution_time,
                    success=True,
                )
            except (TypeError, ValueError, AttributeError, RuntimeError):
                execution_time = time.perf_counter() - start_time
                _record_performance(
                    category,
                    func.__name__,
                    execution_time,
                    success=False,
                )
                raise
            else:
                return result

        return _wrapper

    return _decorator


def _update_category_metrics(
    category_metrics: dict[str, object],
    execution_time: float,
    *,
    success: bool,
) -> None:
    """Update category-level metrics."""
    # Type-safe updates for metrics
    total_calls = category_metrics["total_calls"]
    if isinstance(total_calls, int):
        category_metrics["total_calls"] = total_calls + 1

    total_time = category_metrics["total_time"]
    if isinstance(total_time, (int, float)):
        category_metrics["total_time"] = total_time + execution_time

    if success:
        successful_calls = category_metrics["successful_calls"]
        if isinstance(successful_calls, int):
            category_metrics["successful_calls"] = successful_calls + 1
    else:
        failed_calls = category_metrics["failed_calls"]
        if isinstance(failed_calls, int):
            category_metrics["failed_calls"] = failed_calls + 1


def _update_function_metrics(
    functions: dict[str, object],
    function_name: str,
    execution_time: float,
) -> None:
    """Update function-level metrics."""
    if function_name not in functions:
        functions[function_name] = {
            "calls": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "min_time": float("inf"),
            "max_time": 0.0,
        }

    func_metrics = functions[function_name]
    if isinstance(func_metrics, dict):
        calls = func_metrics["calls"]
        if isinstance(calls, int):
            new_calls = calls + 1
            func_metrics["calls"] = new_calls

            total_time = func_metrics["total_time"]
            if isinstance(total_time, (int, float)):
                new_total_time = total_time + execution_time
                func_metrics["total_time"] = new_total_time
                func_metrics["avg_time"] = new_total_time / new_calls

            min_time = func_metrics["min_time"]
            if isinstance(min_time, (int, float)):
                func_metrics["min_time"] = min(min_time, execution_time)

            max_time = func_metrics["max_time"]
            if isinstance(max_time, (int, float)):
                func_metrics["max_time"] = max(max_time, execution_time)


def _record_performance(
    category: str,
    function_name: str,
    execution_time: float,
    *,
    success: bool,
) -> None:
    """Record performance metrics.

    Args:
        category: Function category
        function_name: Name of the function
        execution_time: Execution time in seconds
        success: Whether the function succeeded

    """
    if category not in _performance_metrics:
        _performance_metrics[category] = {
            "total_calls": 0,
            "total_time": 0.0,
            "successful_calls": 0,
            "failed_calls": 0,
            "functions": {},
        }

    category_metrics = _performance_metrics[category]
    _update_category_metrics(category_metrics, execution_time, success=success)

    # Track per-function metrics
    functions = category_metrics["functions"]
    if isinstance(functions, dict):
        _update_function_metrics(functions, function_name, execution_time)


class _BaseTypeGuards:
    """Base type guard utilities."""

    @staticmethod
    def has_attribute(obj: object, attr: str) -> bool:
        """Check if object has attribute.

        Args:
            obj: Object to check
            attr: Attribute name

        Returns:
            True if object has attribute, False otherwise

        """
        return hasattr(obj, attr)

    @staticmethod
    def is_instance_of(obj: object, target_type: type) -> bool:
        """Check if object is instance of target type.

        Args:
            obj: Object to check
            target_type: Target type

        Returns:
            True if instance of type, False otherwise

        """
        return isinstance(obj, target_type)

    @staticmethod
    def is_list_of(obj: object, item_type: type) -> bool:
        """Check if object is a list of specific type.

        Args:
            obj: Object to check
            item_type: Expected type of list items

        Returns:
            True if list of specified type, False otherwise

        """
        if not isinstance(obj, list):
            return False
        return all(isinstance(item, item_type) for item in obj)

    @staticmethod
    def is_callable_with_return(obj: object, return_type: type) -> bool:
        """Check if object is callable with specific return type.

        Args:
            obj: Object to check
            return_type: Expected return type

        Returns:
            True if callable, False otherwise

        Note:
            This is a basic check - actual return type validation
            requires runtime execution which is not feasible here.

        """
        # Note: return_type is used for documentation but not runtime validation
        _ = return_type  # Acknowledge parameter for linting
        return callable(obj)


class _BaseGenerators:
    """Base generation utilities."""

    @staticmethod
    def _generate_uuid() -> str:
        """Generate UUID string.

        Returns:
            UUID string

        """
        return str(uuid4())

    @staticmethod
    def _generate_short_id(length: int = 8) -> str:
        """Generate short alphanumeric ID.

        Args:
            length: Length of ID (default: 8)

        Returns:
            Short ID string

        """
        chars = string.ascii_letters + string.digits
        return "".join(random.choices(chars, k=length))

    @staticmethod
    def _generate_timestamp() -> float:
        """Generate current timestamp.

        Returns:
            Current timestamp

        """
        return time.time()

    @staticmethod
    def _generate_correlation_id() -> str:
        """Generate correlation ID for tracing.

        Returns:
            Correlation ID string

        """
        timestamp = int(time.time() * 1000)
        short_id = _BaseGenerators._generate_short_id(6)
        return f"{timestamp}-{short_id}"

    @staticmethod
    def _generate_prefixed_id(prefix: str, length: int = 8) -> str:
        """Generate prefixed ID.

        Args:
            prefix: Prefix for the ID
            length: Length of random part (default: 8)

        Returns:
            Prefixed ID string

        """
        short_id = _BaseGenerators._generate_short_id(length)
        return f"{prefix}_{short_id}"

    # Aliases for backwards compatibility
    @staticmethod
    def generate_short_id(length: int = 8) -> str:
        """Generate short ID (public alias)."""
        return _BaseGenerators._generate_short_id(length)

    @staticmethod
    def generate_prefixed_id(prefix: str, length: int = 8) -> str:
        """Generate prefixed ID (public alias)."""
        return _BaseGenerators._generate_prefixed_id(prefix, length)

    @staticmethod
    def generate_uuid() -> str:
        """Generate UUID (public alias)."""
        return _BaseGenerators._generate_uuid()

    @staticmethod
    def generate_entity_id() -> str:
        """Generate entity ID with FLEXT prefix."""
        return _BaseGenerators._generate_prefixed_id("FLEXT", 12)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique ID (public alias for UUID)."""
        return _BaseGenerators._generate_uuid()

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate correlation ID (public alias)."""
        return _BaseGenerators._generate_correlation_id()

    @staticmethod
    def generate_timestamp() -> float:
        """Generate current timestamp."""
        return time.time()

    @staticmethod
    def generate_iso_timestamp() -> str:
        """Generate ISO format timestamp."""
        return datetime.now(tz=UTC).isoformat()

    @staticmethod
    def generate_session_id() -> str:
        """Generate a session ID."""
        return f"sess_{_BaseGenerators._generate_short_id(16)}"

    @staticmethod
    def generate_hash_id(data: str) -> str:
        """Generate hash ID from data using SHA-256."""
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class _BaseFormatters:
    """Base formatting utilities."""

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable format.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted duration string

        """
        if seconds < 1:
            return f"{seconds * 1000:.2f}ms"
        if seconds < _SECONDS_PER_MINUTE:
            return f"{seconds:.2f}s"
        if seconds < _SECONDS_PER_HOUR:
            minutes = int(seconds // _SECONDS_PER_MINUTE)
            remaining_seconds = seconds % _SECONDS_PER_MINUTE
            return f"{minutes}m {remaining_seconds:.2f}s"
        hours = int(seconds // _SECONDS_PER_HOUR)
        remaining_minutes = int((seconds % _SECONDS_PER_HOUR) // _SECONDS_PER_MINUTE)
        return f"{hours}h {remaining_minutes}m"

    @staticmethod
    def _format_size(bytes_count: int) -> str:
        """Format size in human-readable format.

        Args:
            bytes_count: Size in bytes

        Returns:
            Formatted size string

        """
        current_size = float(bytes_count)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if current_size < _BYTES_PER_KB:
                return f"{current_size:.2f}{unit}"
            current_size /= _BYTES_PER_KB
        return f"{current_size:.2f}PB"

    @staticmethod
    def _sanitize_string(text: str, max_length: int = 100) -> str:
        """Sanitize string for safe logging.

        Args:
            text: Text to sanitize
            max_length: Maximum length

        Returns:
            Sanitized string

        """
        # Remove potentially sensitive patterns
        text = re.sub(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CARD]", text)
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)
        text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "[EMAIL]",
            text,
        )

        # Truncate if too long
        if len(text) > max_length:
            text = text[: max_length - 3] + "..."

        return text

    @staticmethod
    def format_dict(data: dict[str, object]) -> str:
        """Format dictionary for display.

        Args:
            data: Dictionary to format

        Returns:
            Formatted string representation

        """
        items = []
        for key, val in data.items():
            if isinstance(val, str):
                # Truncate long strings
                max_display_length = 50
                display_val = val[:47] + "..." if len(val) > max_display_length else val
                items.append(f"{key}='{display_val}'")
            else:
                items.append(f"{key}={val}")
        return ", ".join(items)

    @staticmethod
    def truncate(text: str, max_length: int = 100) -> str:
        """Truncate text to maximum length.

        Args:
            text: Text to truncate
            max_length: Maximum length

        Returns:
            Truncated text

        """
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    @staticmethod
    def snake_to_camel(snake_str: str) -> str:
        """Convert snake_case to camelCase.

        Args:
            snake_str: String in snake_case

        Returns:
            String in camelCase

        """
        components = snake_str.split("_")
        return components[0] + "".join(word.capitalize() for word in components[1:])

    @staticmethod
    def camel_to_snake(camel_str: str) -> str:
        """Convert camelCase to snake_case.

        Args:
            camel_str: String in camelCase

        Returns:
            String in snake_case

        """
        # Insert underscore before uppercase letters
        snake_str = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", camel_str)
        return snake_str.lower()

    @staticmethod
    def format_error_message(
        message: str,
        context: dict[str, object] | None = None,
    ) -> str:
        """Format error message with context.

        Args:
            message: Base error message
            context: Optional context dictionary

        Returns:
            Formatted error message

        """
        if not context:
            return message

        context_parts = []
        for key, value in context.items():
            context_parts.append(f"{key}={value}")

        context_str = ", ".join(context_parts)
        return f"{message} (Context: {context_str})"

    @staticmethod
    def format_entity_reference(entity_type: str, entity_id: str) -> str:
        """Format entity reference string.

        Args:
            entity_type: Type of the entity
            entity_id: ID of the entity

        Returns:
            Formatted entity reference

        """
        return f"{entity_type}({entity_id})"


def _get_performance_metrics() -> dict[str, dict[str, object]]:
    """Get performance metrics for observability.

    Returns:
        Performance metrics dictionary

    """
    return _performance_metrics.copy()


def _clear_performance_metrics() -> None:
    """Clear performance metrics (for testing)."""
    _performance_metrics.clear()


# =============================================================================
# DELEGATION OPTIMIZATION MIXINS - DRY pattern for delegation
# =============================================================================


class _DelegationMixin:
    """Mixin for optimizing delegation patterns with decorators.

    Provides common delegation utilities to reduce code duplication.
    "Entregar mais como muito menos" through delegation optimization.
    """

    @classmethod
    def _create_delegated_method(
        cls,
        base_method: Callable[..., object],
        method_name: str | None = None,
    ) -> staticmethod[object]:
        """Create a delegated method with automatic naming."""
        if method_name is None and hasattr(base_method, "__name__"):
            method_name = base_method.__name__

        def delegated_method(*args: object, **kwargs: object) -> object:
            return base_method(*args, **kwargs)

        if method_name:
            delegated_method.__name__ = method_name

        return staticmethod(delegated_method)

    @classmethod
    def _delegate_all_static_methods(cls, base_class: type) -> dict[str, object]:
        """Auto-delegate all static methods from a base class."""
        delegated = {}
        for attr_name in dir(base_class):
            if not attr_name.startswith("_"):
                attr = getattr(base_class, attr_name)
                if callable(attr):
                    delegated[attr_name] = cls._create_delegated_method(attr, attr_name)
        return delegated


def _delegate_with_tracking(base_method: Callable[..., object]) -> staticmethod[object]:
    """Decorate using tracked methods."""

    def decorator(*args: object, **kwargs: object) -> object:
        # Add performance tracking if needed
        return base_method(*args, **kwargs)

    if hasattr(base_method, "__name__"):
        decorator.__name__ = base_method.__name__

    return staticmethod(decorator)


__all__ = [
    "_BaseFormatters",
    "_BaseGenerators",
    "_BaseTypeGuards",
    "_DelegationMixin",
    "_clear_performance_metrics",
    "_delegate_with_tracking",
    "_get_performance_metrics",
    "_track_performance",
]
