"""FLEXT Core Utilities - Core Pattern Layer Common Operations.

Essential utility functions consolidating ID generation, formatting, type checking,
and performance tracking across the 32-project FLEXT ecosystem. Eliminates code
duplication while providing consistent operational patterns for data integration
pipelines.

Module Role in Architecture:
    Core Pattern Layer → Utility Operations → Foundation Helpers

    This module provides common utility patterns used throughout FLEXT projects:
    - ID generation for entities, sessions, and correlations
    - Performance tracking for enterprise monitoring
    - Type safety guards for Python/Go bridge integration
    - Text formatting and safe data conversion utilities

Utility Operation Patterns:
    DRY Consolidation: Single source of truth for common operations
    Performance Tracking: Built-in observability for enterprise monitoring
    Type Safety: Guards and conversions for multi-language ecosystem
    CLI Error Handling: Standardized error management for command-line interfaces

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: ID generation, formatters, type guards, CLI handling
    🚧 Active Development: Performance optimization (Enhancement 1 - Priority High)
    📋 TODO Integration: Cross-language bridge utilities (Priority 4)

DRY Refactoring Achievements:
    safe_int_conversion(): Eliminates 18+ lines across client-a-oud-mig, taps, targets
    FlextGenerators: Single source for UUID, timestamps, correlation IDs
    CLI error handling: Standardized pattern for all FLEXT CLI applications
    Performance tracking: Consistent metrics across 32 projects

Ecosystem Usage Patterns:
    # Singer Taps/Targets
    correlation_id = flext_generate_correlation_id()
    entity_id = flext_generate_entity_id()

    # client-a Oracle Migration
    port = flext_safe_int_conversion(port_str, 1521)

    # CLI Applications
    FlextUtilities.handle_cli_main_errors(main_function, debug_mode=True)

    # Performance Monitoring
    @flext_track_performance("data_processing")
    def process_oracle_data(data): ...

Enterprise Utility Patterns:
    - Correlation ID propagation for distributed tracing
    - Performance metrics collection for SLA monitoring
    - Safe type conversions preventing pipeline failures
    - Standardized CLI error handling across ecosystem

Quality Standards:
    - All utility functions must be deterministic and side-effect free
    - Performance tracking overhead must be < 1ms per operation
    - Type conversions must handle edge cases gracefully
    - CLI error handling must provide actionable user feedback

See Also:
    docs/TODO.md: Enhancement 1 - Performance optimization
    result.py: FlextResult integration for safe operations
    validation.py: Type validation and guard functions

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import datetime
import sys
import time
import traceback
import uuid
from typing import TYPE_CHECKING, Protocol, TypeGuard

from flext_core.result import FlextResult, safe_call
from flext_core.validation import FlextValidators

try:
    from rich.console import Console
except ImportError:
    # Fallback if Rich is not available
    class _FallbackConsole:
        def print(self, message: str) -> None:
            sys.stdout.write(f"{message}\n")
            sys.stdout.flush()

    Console = _FallbackConsole  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core.flext_types import T, TAnyDict, TFactory, TTransformer

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
    """Utility functions for common operations - DRY REFACTORED."""

    # Time constants for duration formatting
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = 3600

    # DRY REFACTORING: Delegates to FlextGenerators (single source)
    @classmethod
    def generate_uuid(cls) -> str:
        """Generate UUID (delegates to FlextGenerators)."""
        return FlextGenerators.generate_uuid()

    @classmethod
    def generate_id(cls) -> str:
        """Generate unique ID (delegates to FlextGenerators)."""
        return FlextGenerators.generate_id()

    @classmethod
    def generate_timestamp(cls) -> float:
        """Generate timestamp (delegates to FlextGenerators)."""
        return FlextGenerators.generate_timestamp()

    @classmethod
    def generate_iso_timestamp(cls) -> str:
        """Generate ISO format timestamp (delegates to FlextGenerators)."""
        return FlextGenerators.generate_iso_timestamp()

    @classmethod
    def generate_correlation_id(cls) -> str:
        """Generate correlation ID (delegates to FlextGenerators)."""
        return FlextGenerators.generate_correlation_id()

    @classmethod
    def generate_entity_id(cls) -> str:
        """Generate entity ID (delegates to FlextGenerators)."""
        return FlextGenerators.generate_entity_id()

    @classmethod
    def generate_session_id(cls) -> str:
        """Generate session ID (delegates to FlextGenerators)."""
        return FlextGenerators.generate_session_id()

    @classmethod
    def truncate(cls, text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text."""
        if len(text) <= max_length:
            return text
        return text[: max_length - len(suffix)] + suffix

    @classmethod
    def handle_cli_main_errors(
        cls,
        cli_function: Callable[[], None],
        *,
        debug_mode: bool = False,
    ) -> None:
        """Handle CLI main function errors with consistent error handling.

        REFACTORED: DRY principle - eliminates duplicate error handling code.
        Used by all CLI applications to standardize error handling behavior.

        Args:
            cli_function: The main CLI function to execute
            debug_mode: Whether to show full tracebacks in error mode.

        """
        console = Console()

        try:
            cli_function()
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled by user[/yellow]")
            sys.exit(1)
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            ConnectionError,
            TimeoutError,
        ) as e:
            console.print(f"[red]Error: {e}[/red]")

            # Show full traceback in debug mode or when explicitly requested
            if debug_mode:
                console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
            sys.exit(1)

    @classmethod
    def format_duration(cls, seconds: float) -> str:
        """Format duration (delegates to FlextGenerators)."""
        return FlextGenerators.format_duration(seconds)

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

    # =================================================================
    # INTEGER CONVERSION UTILITIES - DRY REFACTORING
    # Eliminates duplication across FLEXT ecosystem projects
    # =================================================================

    @classmethod
    def safe_int_conversion(
        cls,
        value: object,
        default: int | None = None,
    ) -> int | None:
        """Safely convert value to integer with optional default.

        SOLID REFACTORING: Eliminates 18+ lines of duplicated integer conversion
        logic across multiple FLEXT projects (client-a-oud-mig, flext-tap-oracle-wms).

        Supports comprehensive type conversion patterns:
        - int: Direct return
        - str (digits): Parse to int
        - float: Truncate to int
        - Other types: Return default or None

        Args:
            value: Value to convert to integer
            default: Default value if conversion fails (None for no default)

        Returns:
            Converted integer, default value, or None if conversion fails

        Usage:
            # With default
            result = FlextUtilities.safe_int_conversion("123", 0)  # -> 123
            result = FlextUtilities.safe_int_conversion("invalid", 0)  # -> 0

            # Without default (returns None on failure)
            result = FlextUtilities.safe_int_conversion("123")  # -> 123
            result = FlextUtilities.safe_int_conversion("invalid")  # -> None

        """
        # Direct conversion strategies
        conversion_result = cls._try_direct_int_conversion(value)
        if conversion_result is not None:
            return conversion_result

        # Fallback string conversion
        conversion_result = cls._try_string_int_conversion(value)
        return conversion_result if conversion_result is not None else default

    @classmethod
    def _try_direct_int_conversion(cls, value: object) -> int | None:
        """Try direct integer conversion strategies."""
        # Fast path for already integers
        if isinstance(value, int):
            return value

        # String digit conversion
        if isinstance(value, str) and value.isdigit():
            try:
                return int(value)
            except ValueError:
                return None

        # Float truncation
        if isinstance(value, float):
            try:
                return int(value)
            except (ValueError, OverflowError):
                return None

        return None

    @classmethod
    def _try_string_int_conversion(cls, value: object) -> int | None:
        """Try string-based conversion as fallback."""
        try:
            return int(str(value))
        except (ValueError, TypeError, OverflowError):
            return None

    @classmethod
    def safe_int_conversion_with_default(cls, value: object, default: int) -> int:
        """Safely convert value to integer with guaranteed default return.

        SOLID REFACTORING: Template Method pattern for guaranteed integer return.
        Never returns None - always returns either converted value or default.

        Args:
            value: Value to convert to integer
            default: Default value if conversion fails (guaranteed return)

        Returns:
            Converted integer or default (never None)

        Usage:
            result = FlextUtilities.safe_int_conversion_with_default("123", 0)
            result = FlextUtilities.safe_int_conversion_with_default("invalid", 0)

        """
        converted = cls.safe_int_conversion(value, default)
        return converted if converted is not None else default


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
            except (RuntimeError, ValueError, TypeError):
                execution_time = time.time() - start_time
                flext_record_performance(
                    category,
                    func.__name__,
                    execution_time,
                    success=False,
                )
                raise
            else:
                execution_time = time.time() - start_time
                flext_record_performance(
                    category,
                    func.__name__,
                    execution_time,
                    success=True,
                )
                return result

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
    success: bool,  # noqa: ARG001
) -> None:
    """Record performance metrics for observability."""
    # Note: success parameter reserved for future observability features
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


def flext_safe_int_conversion(value: object, default: int | None = None) -> int | None:
    """Safely convert value to integer with optional default (public API).

    SOLID REFACTORING: Eliminates duplication across FLEXT ecosystem.

    Args:
        value: Value to convert to integer
        default: Default value if conversion fails

    Returns:
        Converted integer, default, or None if conversion fails

    """
    return FlextUtilities.safe_int_conversion(value, default)


def flext_safe_int_conversion_with_default(value: object, default: int) -> int:
    """Safely convert value to integer with guaranteed default (public API).

    SOLID REFACTORING: Template Method pattern - never returns None.

    Args:
        value: Value to convert to integer
        default: Default value if conversion fails (guaranteed return)

    Returns:
        Converted integer or default (never None)

    """
    return FlextUtilities.safe_int_conversion_with_default(value, default)


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


def safe_int_conversion(value: object, default: int | None = None) -> int | None:
    """Safely convert value to integer (backward compatibility)."""
    return FlextUtilities.safe_int_conversion(value, default)


def safe_int_conversion_with_default(value: object, default: int) -> int:
    """Safely convert value to integer with default (backward compatibility)."""
    return FlextUtilities.safe_int_conversion_with_default(value, default)


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
    """ID and timestamp generation utilities - SINGLE SOURCE OF TRUTH."""

    # Time constants for duration formatting
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = 3600

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
    def format_duration(cls, seconds: float) -> str:
        """Format duration."""
        if seconds < 1:
            return f"{seconds * 1000:.1f}ms"
        if seconds < cls.SECONDS_PER_MINUTE:
            return f"{seconds:.1f}s"
        if seconds < cls.SECONDS_PER_HOUR:
            return f"{seconds / cls.SECONDS_PER_MINUTE:.1f}m"
        return f"{seconds / cls.SECONDS_PER_HOUR:.1f}h"


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
        """Format duration using FlextGenerators implementation."""
        return FlextGenerators.format_duration(seconds)


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
    "flext_safe_int_conversion",
    "flext_safe_int_conversion_with_default",
    "flext_track_performance",
    "flext_truncate",
    # Backward compatibility functions
    "generate_correlation_id",
    "generate_id",
    "generate_iso_timestamp",
    "generate_uuid",
    "is_not_none",
    "safe_call",
    "safe_int_conversion",
    "safe_int_conversion_with_default",
    "truncate",
]
