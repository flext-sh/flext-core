"""FLEXT Core Utilities Module.

Utility functions for ID generation, formatting, type checking, and performance.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Protocol, TypeGuard

from flext_core.result import FlextResult, safe_call
from flext_core.validation import FlextValidators

if TYPE_CHECKING:
    from flext_core.types import T, TAnyDict, TFactory, TTransformer

# =============================================================================
# CONSTANTS
# =============================================================================

SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
BYTES_PER_KB = 1024

# Performance metrics dictionary
PERFORMANCE_METRICS: dict[str, float] = {}


class DecoratedFunction(Protocol):
    """Protocol for functions that can be decorated with performance tracking."""

    __name__: str

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Call decorated function with provided arguments."""
        ...


# =============================================================================
# FLEXT UTILITIES - Single class consolidating all functionality
# =============================================================================


class FlextUtilities:
    """Utility functions for common operations."""

    @classmethod
    def generate_uuid(cls) -> str:
        """Generate UUID."""
        return str(uuid.uuid4())

    @classmethod
    def generate_id(cls) -> str:
        """Generate unique ID."""
        return f"id_{uuid.uuid4().hex[:8]}"

    @classmethod
    def generate_timestamp(cls) -> float:
        """Generate timestamp."""
        return time.time()

    @classmethod
    def generate_iso_timestamp(cls) -> str:
        """Generate ISO format timestamp."""
        import datetime

        return datetime.datetime.now(datetime.UTC).isoformat()

    @classmethod
    def generate_correlation_id(cls) -> str:
        """Generate correlation ID."""
        return f"corr_{uuid.uuid4().hex[:12]}"

    @classmethod
    def generate_entity_id(cls) -> str:
        """Generate entity ID."""
        return f"entity_{uuid.uuid4().hex[:10]}"

    @classmethod
    def generate_session_id(cls) -> str:
        """Generate session ID."""
        return f"session_{uuid.uuid4().hex[:12]}"

    @classmethod
    def truncate(cls, text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text."""
        if len(text) <= max_length:
            return text
        return text[: max_length - len(suffix)] + suffix

    @classmethod
    def format_duration(cls, seconds: float) -> str:
        """Format duration."""
        if seconds < 1:
            return f"{seconds * 1000:.1f}ms"
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}m"
        return f"{seconds / 3600:.1f}h"

    @classmethod
    def has_attribute(cls, obj: object, attr: str) -> bool:
        """Check if object has attribute."""
        return hasattr(obj, attr)

    @classmethod
    def is_instance_of(cls, obj: object, target_type: type) -> bool:
        """Check if object is instance of type."""
        return isinstance(obj, target_type)

    @classmethod
    def safe_call(cls, func: TFactory[T]) -> FlextResult[T]:
        """Safely call function."""
        return safe_call(func)

    @classmethod
    def is_not_none_guard(cls, value: T | None) -> TypeGuard[T]:
        """Type guard for not None values."""
        return FlextValidators.is_not_none(value)


# =============================================================================
# PUBLIC API FUNCTIONS - Direct delegation to FlextUtilities
# =============================================================================


def flext_track_performance(
    category: str,
) -> TTransformer[DecoratedFunction, DecoratedFunction]:
    """Track function performance as decorator."""

    def decorator(func: DecoratedFunction) -> DecoratedFunction:
        def wrapper(*args: object, **kwargs: object) -> object:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                flext_record_performance(
                    category,
                    func.__name__,
                    execution_time,
                    success=True,
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                flext_record_performance(
                    category,
                    func.__name__,
                    execution_time,
                    success=False,
                )
                raise e

        return wrapper

    return decorator


def flext_get_performance_metrics() -> dict[str, TAnyDict]:
    """Get performance metrics for observability."""
    return {"metrics": dict(PERFORMANCE_METRICS)}


def flext_clear_performance_metrics() -> None:
    """Clear performance metrics (for testing)."""
    PERFORMANCE_METRICS.clear()


def flext_record_performance(
    category: str,
    function_name: str,
    execution_time: float,
    *,
    success: bool,
) -> None:
    """Record performance metrics for observability."""
    key = f"{category}.{function_name}"
    PERFORMANCE_METRICS[key] = execution_time


def flext_safe_call(func: TFactory[T]) -> FlextResult[T]:
    """Safely call function with FlextResult error handling."""
    return FlextUtilities.safe_call(func)


def flext_is_not_none(value: T | None) -> TypeGuard[T]:
    """Type guard to check if value is not None."""
    return FlextUtilities.is_not_none_guard(value)


def flext_generate_id() -> str:
    """Generate unique ID."""
    return FlextUtilities.generate_id()


def flext_generate_correlation_id() -> str:
    """Generate correlation ID."""
    return FlextUtilities.generate_correlation_id()


def flext_truncate(text: str, max_length: int = 100) -> str:
    """Truncate text."""
    return FlextUtilities.truncate(text, max_length)


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES - Essential for existing tests
# =============================================================================


def truncate(text: str, max_length: int = 100) -> str:
    """Truncate text to maximum length (backward compatibility)."""
    return FlextUtilities.truncate(text, max_length)


def generate_id() -> str:
    """Generate unique ID (backward compatibility)."""
    return FlextUtilities.generate_id()


def generate_correlation_id() -> str:
    """Generate correlation ID (backward compatibility)."""
    return FlextUtilities.generate_correlation_id()


def generate_uuid() -> str:
    """Generate UUID (backward compatibility)."""
    return FlextUtilities.generate_uuid()


def generate_iso_timestamp() -> str:
    """Generate ISO timestamp (backward compatibility)."""
    return FlextUtilities.generate_iso_timestamp()


def is_not_none(value: object) -> bool:
    """Check if value is not None (backward compatibility)."""
    return FlextValidators.is_not_none(value)


# safe_call is imported from result.py (single source of truth)
# and delegated through FlextUtilities.safe_call for consistency


# =============================================================================
# ALIASES FOR SPECIALIZED CLASSES - Direct access to base functionality
# =============================================================================


# Direct aliases to avoid duplication
class FlextTypeGuards:
    """Type guard utilities."""

    @staticmethod
    def has_attribute(obj: object, attr: str) -> bool:
        """Check if object has attribute."""
        return hasattr(obj, attr)

    @staticmethod
    def is_instance_of(obj: object, target_type: type) -> bool:
        """Check if object is instance of type."""
        return isinstance(obj, target_type)

    @staticmethod
    def is_list_of(obj: object, item_type: type) -> bool:
        """Check if object is a list of specific type."""
        if not isinstance(obj, list):
            return False
        return all(isinstance(item, item_type) for item in obj)


class FlextGenerators:
    """ID and timestamp generation utilities."""

    @staticmethod
    def generate_uuid() -> str:
        """Generate UUID."""
        return str(uuid.uuid4())

    @staticmethod
    def generate_id() -> str:
        """Generate unique ID."""
        return f"id_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def generate_timestamp() -> float:
        """Generate timestamp."""
        return time.time()

    @staticmethod
    def generate_iso_timestamp() -> str:
        """Generate ISO format timestamp."""
        import datetime

        return datetime.datetime.now(datetime.UTC).isoformat()

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate correlation ID."""
        return f"corr_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def generate_entity_id() -> str:
        """Generate entity ID."""
        return f"entity_{uuid.uuid4().hex[:10]}"

    @staticmethod
    def generate_session_id() -> str:
        """Generate session ID."""
        return f"session_{uuid.uuid4().hex[:12]}"


class FlextFormatters:
    """Text formatting utilities."""

    @staticmethod
    def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text."""
        if len(text) <= max_length:
            return text
        return text[: max_length - len(suffix)] + suffix

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration."""
        if seconds < 1:
            return f"{seconds * 1000:.1f}ms"
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}m"
        return f"{seconds / 3600:.1f}h"


# =============================================================================
# EXPORTS - Clean public API seguindo diretrizes
# =============================================================================

__all__ = [
    # Constants
    "BYTES_PER_KB",
    "PERFORMANCE_METRICS",
    "SECONDS_PER_HOUR",
    "SECONDS_PER_MINUTE",
    "DecoratedFunction",
    # Direct access to specialized classes (aliases to base classes)
    "FlextFormatters",
    "FlextGenerators",
    "FlextTypeGuards",
    # Main consolidated class
    "FlextUtilities",
    # Functions with flext_ prefix
    "flext_clear_performance_metrics",
    "flext_generate_correlation_id",
    "flext_generate_id",
    "flext_get_performance_metrics",
    "flext_is_not_none",
    "flext_record_performance",
    "flext_safe_call",
    "flext_track_performance",
    "flext_truncate",
    # Backward compatibility functions
    "generate_correlation_id",
    "generate_id",
    "generate_iso_timestamp",
    "generate_uuid",
    "is_not_none",
    "safe_call",
    "truncate",
]
