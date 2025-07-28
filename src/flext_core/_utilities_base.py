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
from collections.abc import Callable
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


class _CallableProtocol(Protocol):
    """Protocol for callable objects without explicit Any."""

    def __call__(self, *args: object, **kwargs: object) -> object: ...


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
    """Consolidated generation utilities with DRY principles.

    Eliminates duplication by consolidating core generation logic.
    Follows 'entregar mais como muito menos' principle.
    """

    # =======================================================================
    # CORE IMPLEMENTATIONS - Single source of truth
    # =======================================================================

    @staticmethod
    def generate_uuid() -> str:
        """Generate UUID string - primary implementation."""
        return str(uuid4())

    @staticmethod
    def generate_short_id(length: int = 8) -> str:
        """Generate short alphanumeric ID - primary implementation."""
        chars = string.ascii_letters + string.digits
        return "".join(random.choices(chars, k=length))  # noqa: S311

    @staticmethod
    def generate_timestamp() -> float:
        """Generate current timestamp - primary implementation."""
        return time.time()

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate correlation ID for tracing - primary implementation."""
        timestamp = int(time.time() * 1000)
        short_id = _BaseGenerators.generate_short_id(6)
        return f"{timestamp}-{short_id}"

    @staticmethod
    def generate_prefixed_id(prefix: str, length: int = 8) -> str:
        """Generate prefixed ID - primary implementation."""
        short_id = _BaseGenerators.generate_short_id(length)
        return f"{prefix}_{short_id}"

    # =======================================================================
    # SPECIALIZED GENERATORS - Built on core implementations
    # =======================================================================

    @staticmethod
    def generate_entity_id() -> str:
        """Generate entity ID with FLEXT prefix."""
        return _BaseGenerators.generate_prefixed_id("FLEXT", 12)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique ID (delegates to UUID)."""
        return _BaseGenerators.generate_uuid()

    @staticmethod
    def generate_iso_timestamp() -> str:
        """Generate ISO format timestamp."""
        return datetime.now(tz=UTC).isoformat()

    @staticmethod
    def generate_session_id() -> str:
        """Generate a session ID."""
        return f"sess_{_BaseGenerators.generate_short_id(16)}"

    @staticmethod
    def generate_hash_id(data: str) -> str:
        """Generate hash ID from data using SHA-256."""
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class _BaseFormatters:
    """Consolidated formatting utilities with DRY principles.

    Eliminates duplication by consolidating common formatting patterns.
    """

    # =======================================================================
    # CORE FORMATTING UTILITIES - Single source of truth
    # =======================================================================

    @staticmethod
    def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text to maximum length - primary implementation."""
        if len(text) <= max_length:
            return text
        return text[: max_length - len(suffix)] + suffix

    @staticmethod
    def format_key_value_pairs(
        data: dict[str, object],
        separator: str = ", ",
        *,
        quote_strings: bool = True,
        max_value_length: int = 50,
    ) -> str:
        """Format key-value pairs - primary implementation."""
        items = []
        for key, val in data.items():
            if isinstance(val, str):
                display_val = _BaseFormatters.truncate(val, max_value_length)
                formatted_val = f"'{display_val}'" if quote_strings else display_val
            else:
                formatted_val = str(val)
            items.append(f"{key}={formatted_val}")
        return separator.join(items)

    # =======================================================================
    # SPECIALIZED FORMATTERS - Built on core implementations
    # =======================================================================

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
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
    def format_size(bytes_count: int) -> str:
        """Format size in human-readable format."""
        current_size = float(bytes_count)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if current_size < _BYTES_PER_KB:
                return f"{current_size:.2f}{unit}"
            current_size /= _BYTES_PER_KB
        return f"{current_size:.2f}PB"

    @staticmethod
    def sanitize_string(text: str, max_length: int = 100) -> str:
        """Sanitize string for safe logging."""
        # Remove potentially sensitive patterns
        card_pattern = r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
        sanitized = re.sub(card_pattern, "[CARD]", text)
        sanitized = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", sanitized)
        sanitized = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "[EMAIL]",
            sanitized,
        )
        return _BaseFormatters.truncate(sanitized, max_length)

    @staticmethod
    def format_dict(data: dict[str, object]) -> str:
        """Format dictionary for display."""
        return _BaseFormatters.format_key_value_pairs(data, quote_strings=True)

    @staticmethod
    def snake_to_camel(snake_str: str) -> str:
        """Convert snake_case to camelCase."""
        components = snake_str.split("_")
        return components[0] + "".join(word.capitalize() for word in components[1:])

    @staticmethod
    def camel_to_snake(camel_str: str) -> str:
        """Convert camelCase to snake_case."""
        snake_str = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", camel_str)
        return snake_str.lower()

    @staticmethod
    def format_error_message(
        message: str,
        context: dict[str, object] | None = None,
    ) -> str:
        """Format error message with context."""
        if not context:
            return message
        context_str = _BaseFormatters.format_key_value_pairs(
            context,
            quote_strings=False,
        )
        return f"{message} (Context: {context_str})"

    @staticmethod
    def format_entity_reference(entity_type: str, entity_id: str) -> str:
        """Format entity reference string."""
        return f"{entity_type}({entity_id})"


# =============================================================================
# CONSOLIDATED PERFORMANCE TRACKING - DRY principles with single source of truth
# =============================================================================


class _PerformanceTracker:
    """Consolidated performance tracking with DRY principles.

    Eliminates duplication by consolidating all performance tracking logic
    in a single class with unified patterns.
    """

    def __init__(self) -> None:
        """Initialize performance tracker."""
        self._metrics: dict[str, dict[str, object]] = {}

    def _safe_increment(
        self,
        container: dict[str, object],
        key: str,
        amount: float = 1,
    ) -> None:
        """Safely increment a numeric value in container."""
        current = container.get(key, 0)
        if isinstance(current, int | float):
            container[key] = current + amount

    def _ensure_category_exists(self, category: str) -> dict[str, object]:
        """Ensure category exists with proper structure."""
        if category not in self._metrics:
            self._metrics[category] = {
                "total_calls": 0,
                "total_time": 0.0,
                "successful_calls": 0,
                "failed_calls": 0,
                "functions": {},
            }
        return self._metrics[category]

    def _ensure_function_exists(
        self,
        functions: dict[str, object],
        function_name: str,
    ) -> dict[str, object]:
        """Ensure function metrics exist with proper structure."""
        if function_name not in functions:
            functions[function_name] = {
                "calls": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
            }
        result = functions[function_name]
        if isinstance(result, dict):
            return result
        # Fallback: create new dict if somehow corrupted
        functions[function_name] = {
            "calls": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            "min_time": float("inf"),
            "max_time": 0.0,
        }
        # Type-safe return ensuring dict[str, object]
        result = functions[function_name]
        if isinstance(result, dict):
            return result
        # Fallback should never be reached but ensures type safety
        return {}

    def record_performance(
        self,
        category: str,
        function_name: str,
        execution_time: float,
        *,
        success: bool,
    ) -> None:
        """Record performance metrics with consolidated logic."""
        # Ensure category exists
        category_metrics = self._ensure_category_exists(category)

        # Update category metrics using DRY pattern
        self._safe_increment(category_metrics, "total_calls")
        self._safe_increment(category_metrics, "total_time", execution_time)

        if success:
            self._safe_increment(category_metrics, "successful_calls")
        else:
            self._safe_increment(category_metrics, "failed_calls")

        # Update function metrics
        functions = category_metrics["functions"]
        if isinstance(functions, dict):
            func_metrics = self._ensure_function_exists(functions, function_name)
            if isinstance(func_metrics, dict):
                # Update call count and total time
                self._safe_increment(func_metrics, "calls")
                self._safe_increment(func_metrics, "total_time", execution_time)

                # Update average time
                calls = func_metrics.get("calls", 1)
                total_time = func_metrics.get("total_time", 0.0)
                if (
                    isinstance(calls, int | float)
                    and isinstance(total_time, int | float)
                    and calls > 0
                ):
                    func_metrics["avg_time"] = total_time / calls

                # Update min/max times
                min_time = func_metrics.get("min_time", float("inf"))
                max_time = func_metrics.get("max_time", 0.0)
                if isinstance(min_time, int | float):
                    func_metrics["min_time"] = min(min_time, execution_time)
                if isinstance(max_time, int | float):
                    func_metrics["max_time"] = max(max_time, execution_time)

    def track_performance(
        self,
        category: str,
    ) -> Callable[[_DecoratedFunction], _DecoratedFunction]:
        """Create performance tracking decorator with consolidated logic."""

        def decorator(func: _DecoratedFunction) -> _DecoratedFunction:
            @wraps(func)
            def wrapper(*args: object, **kwargs: object) -> object:
                start_time = time.perf_counter()
                success = False
                try:
                    result = func(*args, **kwargs)
                    success = True
                    return result
                finally:
                    execution_time = time.perf_counter() - start_time
                    self.record_performance(
                        category,
                        func.__name__,
                        execution_time,
                        success=success,
                    )

            return wrapper

        return decorator

    def get_metrics(self) -> dict[str, dict[str, object]]:
        """Get copy of all performance metrics."""
        return dict(self._metrics)

    def clear_metrics(self) -> None:
        """Clear all performance metrics."""
        self._metrics.clear()


# Global instance for backward compatibility
_global_tracker = _PerformanceTracker()


# Backward compatibility functions
def _track_performance(
    category: str,
) -> Callable[[_DecoratedFunction], _DecoratedFunction]:
    """Legacy function for backward compatibility."""
    return _global_tracker.track_performance(category)


def _record_performance(
    category: str,
    function_name: str,
    execution_time: float,
    *,
    success: bool,
) -> None:
    """Legacy function for backward compatibility."""
    _global_tracker.record_performance(
        category,
        function_name,
        execution_time,
        success=success,
    )


def _get_performance_metrics() -> dict[str, dict[str, object]]:
    """Get performance metrics for observability - delegates to consolidated tracker."""
    return _global_tracker.get_metrics()


def _clear_performance_metrics() -> None:
    """Clear performance metrics - delegates to consolidated tracker."""
    _global_tracker.clear_metrics()


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
        base_method: _CallableProtocol,
        method_name: str | None = None,
    ) -> object:
        """Create a delegated method with automatic naming."""
        # base_method is guaranteed to be callable by type annotation
        if method_name is None and hasattr(base_method, "__name__"):
            method_name = base_method.__name__

        def delegated_method(*args: object, **kwargs: object) -> object:
            # base_method is guaranteed to be callable by type annotation
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


def _delegate_with_tracking(
    base_method: _CallableProtocol,
) -> object:
    """Decorate using tracked methods."""

    def decorator(*args: object, **kwargs: object) -> object:
        # Add performance tracking if needed - base_method guaranteed callable by type
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
