"""Common utility functions."""

from __future__ import annotations

import functools
import json
import re
import sys
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from inspect import signature
from typing import Generic, Protocol, cast, override

from flext_core.constants import FlextConstants
from flext_core.loggings import FlextLoggerFactory
from flext_core.result import FlextResult
from flext_core.typings import (
    FlextTypes,
    P,
    R,
    T,
)
from flext_core.validation import FlextValidators

logger = FlextLoggerFactory.get_logger(__name__)

# =============================================================================
# CONSTANTS - Default values
# =============================================================================

BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024 * 1024 * 1024

# Time constants
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
# =============================================================================
# CONSTANTS
# =============================================================================
PERFORMANCE_METRICS: dict[str, dict[str, dict[str, float | bool]]] = {}


class FlextConsole:
    """Console implementation - Tier 1 Compatibility Facade.

    This class provides backward compatibility by delegating to FlextUtilities.
    For new code, use FlextUtilities.print() directly.
    """

    def __init__(self) -> None:
        """Initialize console."""

    @staticmethod
    def print(*args: FlextTypes.Core.Value, **kwargs: FlextTypes.Core.Value) -> None:
        """Print to console - delegates to FlextUtilities."""
        return FlextUtilities.print(*args, **kwargs)

    def log(
        self, *args: FlextTypes.Core.Value, **kwargs: FlextTypes.Core.Value
    ) -> None:
        """Log to console - delegates to FlextUtilities."""
        return FlextUtilities.log(*args, **kwargs)


class FlextDecoratedFunction(Protocol):
    """Protocol for functions that can be decorated with performance tracking."""

    __name__: str

    def __call__(
        self, *args: FlextTypes.Core.Value, **kwargs: FlextTypes.Core.Value
    ) -> FlextTypes.Core.Value:
        """Call decorated function with provided arguments."""


# =============================================================================
# FLEXT UTILITIES - Single class consolidating all functionality
# =============================================================================


class FlextUtilities:
    """Unified utility system implementing Tier 1 Module Pattern.

    This class serves as the single main export consolidating ALL utility
    functionality from the flext-core utilities ecosystem. Provides comprehensive
    utilities while maintaining backward compatibility.

    Tier 1 Module Pattern: utilities.py -> FlextUtilities
    All utility functionality is accessible through this single interface.

    Consolidated Functionality:
    - ID Generation (from FlextGenerators, FlextIdGenerator)
    - Time Operations (from FlextTimeUtils)
    - Text Processing (from FlextTextProcessor)
    - Performance Tracking (from FlextPerformance)
    - Type Conversions (from FlextConversions)
    - Type Guards (from FlextTypeGuards)
    - Formatting (from FlextFormatters)
    - Processing Utils (from FlextProcessingUtils)
    - Console Operations (from FlextConsole)
    - Factory Patterns (from FlextBaseFactory, FlextGenericFactory, FlextUtilityFactory)
    - Result Operations (from FlextResultUtilities)
    - Core Utilities (from FlextCoreUtilities)
    - Type Utilities (from FlextTypeUtilities)
    """

    # =============================================================================
    # CONSTANTS - Time and sizing constants
    # =============================================================================
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = 3600
    BYTES_PER_KB = 1024
    BYTES_PER_MB = 1024 * 1024
    BYTES_PER_GB = 1024 * 1024 * 1024

    # =============================================================================
    # ID GENERATION - Consolidated from FlextGenerators, FlextIdGenerator
    # =============================================================================

    @classmethod
    def generate_uuid(cls) -> str:
        """Generate standard UUID4."""
        return str(uuid.uuid4())

    @classmethod
    def generate_id(cls) -> str:
        """Generate unique ID with flext_ prefix."""
        return f"flext_{uuid.uuid4().hex[:8]}"

    @classmethod
    def generate_timestamp(cls) -> float:
        """Generate Unix timestamp."""
        return time.time()

    @classmethod
    def generate_iso_timestamp(cls) -> str:
        """Generate ISO 8601 timestamp."""
        return datetime.now(UTC).isoformat()

    @classmethod
    def generate_correlation_id(cls) -> str:
        """Generate correlation ID with corr_ prefix."""
        return f"corr_{uuid.uuid4().hex[:16]}"

    @classmethod
    def generate_entity_id(cls) -> str:
        """Generate entity ID with entity_ prefix."""
        return f"entity_{uuid.uuid4().hex[:12]}"

    @classmethod
    def generate_session_id(cls) -> str:
        """Generate session ID with sess_ prefix."""
        return f"sess_{uuid.uuid4().hex[:10]}"

    @classmethod
    def generate_request_id(cls) -> str:
        """Generate request ID with req_ prefix."""
        return f"req_{uuid.uuid4().hex[:10]}"

    # =============================================================================
    # TEXT PROCESSING - Consolidated from FlextTextProcessor
    # =============================================================================

    @classmethod
    def truncate(cls, text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text to maximum length with suffix."""
        if len(text) <= max_length:
            return text
        if max_length <= 0:
            return ""  # Return empty string for zero or negative max_length
        if max_length < len(suffix):
            return text[:max_length]  # Can't fit suffix, just truncate
        cut = max(0, max_length - len(suffix))
        return text[:cut] + suffix

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Clean text by removing extra whitespace and normalizing."""
        return re.sub(r"\s+", " ", text.strip())

    @classmethod
    def extract_numbers(cls, text: str) -> list[str]:
        """Extract all numbers from text."""
        return re.findall(r"\d+", text)

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename by removing invalid characters."""
        return re.sub(r'[<>:"/\\|?*]', "_", filename)

    # =============================================================================
    # TIME OPERATIONS - Consolidated from FlextTimeUtils
    # =============================================================================

    @classmethod
    def format_duration(cls, seconds: float) -> str:
        """Format duration in human-readable format."""
        seconds_per_minute = 60
        seconds_per_hour = 3600

        if seconds < seconds_per_minute:
            return f"{seconds:.2f}s"
        if seconds < seconds_per_hour:
            minutes = seconds / seconds_per_minute
            return f"{minutes:.1f}m"
        hours = seconds / seconds_per_hour
        return f"{hours:.1f}h"

    @classmethod
    def parse_iso_timestamp(cls, timestamp: str) -> datetime:
        """Parse ISO timestamp to datetime object."""
        return datetime.fromisoformat(timestamp)

    @classmethod
    def get_elapsed_time(cls, start_time: float) -> float:
        """Get elapsed time from start timestamp."""
        return time.time() - start_time

    # =============================================================================
    # PERFORMANCE TRACKING - Consolidated from FlextPerformance
    # =============================================================================

    @classmethod
    def track_performance(
        cls, category: str
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Track function performance as decorator."""

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                except (RuntimeError, ValueError, TypeError):
                    execution_time = time.time() - start_time
                    cls.record_performance(
                        category, func.__name__, execution_time, success=False
                    )
                    raise
                else:
                    execution_time = time.time() - start_time
                    cls.record_performance(
                        category, func.__name__, execution_time, success=True
                    )
                    return result

            return wrapper

        return decorator

    @classmethod
    def record_performance(
        cls, category: str, operation: str, duration: float, *, success: bool = True
    ) -> None:
        """Record performance metric."""
        if category not in PERFORMANCE_METRICS:
            PERFORMANCE_METRICS[category] = {}

        PERFORMANCE_METRICS[category][operation] = {
            "duration": duration,
            "success": success,
            "timestamp": time.time(),
        }

    @classmethod
    def get_performance_metrics(
        cls,
    ) -> dict[str, dict[str, dict[str, dict[str, float | bool]]]]:
        """Get all performance metrics."""
        return {"metrics": PERFORMANCE_METRICS}

    @classmethod
    def clear_performance_metrics(cls) -> None:
        """Clear all performance metrics."""
        PERFORMANCE_METRICS.clear()

    # =============================================================================
    # TYPE CONVERSIONS - Consolidated from FlextConversions
    # =============================================================================

    @classmethod
    def safe_int(cls, value: FlextTypes.Core.Value, default: int = 0) -> int:
        """Safely convert value to integer."""
        try:
            if isinstance(value, str):
                return int(float(value))  # Handle "3.14" -> 3
            return int(value)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return default

    @classmethod
    def safe_float(cls, value: FlextTypes.Core.Value, default: float = 0.0) -> float:
        """Safely convert value to float."""
        try:
            return float(value)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            return default

    @classmethod
    def safe_str(cls, value: FlextTypes.Core.Value, default: str = "") -> str:
        """Safely convert value to string."""
        try:
            return str(value)
        except Exception:
            return default

    @classmethod
    def to_bool(cls, value: FlextTypes.Core.Value) -> bool:
        """Convert value to boolean with intelligent parsing."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"true", "yes", "1", "on", "enabled"}
        return bool(value)

    # =============================================================================
    # TYPE GUARDS - Consolidated from FlextTypeGuards
    # =============================================================================

    @classmethod
    def is_not_none(cls, value: object) -> bool:
        """Check if value is not None."""
        return value is not None

    @classmethod
    def is_string(cls, value: object) -> bool:
        """Check if value is a string."""
        return isinstance(value, str)

    @classmethod
    def is_non_empty_string(cls, value: object) -> bool:
        """Check if value is a non-empty string."""
        return isinstance(value, str) and len(value.strip()) > 0

    @classmethod
    def is_numeric(cls, value: object) -> bool:
        """Check if value is numeric (int or float)."""
        return isinstance(value, (int, float))

    @classmethod
    def is_list(cls, value: object) -> bool:
        """Check if value is a list."""
        return isinstance(value, list)

    @classmethod
    def is_dict(cls, value: object) -> bool:
        """Check if value is a dictionary."""
        return isinstance(value, dict)

    # =============================================================================
    # FORMATTING - Consolidated from FlextFormatters
    # =============================================================================

    @classmethod
    def format_bytes(cls, bytes_count: int) -> str:
        """Format byte count in human-readable format."""
        if bytes_count < cls.BYTES_PER_KB:
            return f"{bytes_count} B"
        if bytes_count < cls.BYTES_PER_MB:
            return f"{bytes_count / cls.BYTES_PER_KB:.1f} KB"
        if bytes_count < cls.BYTES_PER_GB:
            return f"{bytes_count / cls.BYTES_PER_MB:.1f} MB"
        return f"{bytes_count / cls.BYTES_PER_GB:.1f} GB"

    @classmethod
    def format_number(cls, number: float, precision: int = 2) -> str:
        """Format number with thousand separators."""
        return f"{number:,.{precision}f}"

    @classmethod
    def format_percentage(cls, value: float, precision: int = 1) -> str:
        """Format value as percentage."""
        return f"{value * 100:.{precision}f}%"

    # =============================================================================
    # CONSOLE OPERATIONS - Consolidated from FlextConsole
    # =============================================================================

    @classmethod
    def print(
        cls, *args: FlextTypes.Core.Value, **kwargs: FlextTypes.Core.Value
    ) -> None:
        """Print to console with markup removal."""
        text_parts: FlextTypes.Core.List = []
        for arg in args:
            text = str(arg)
            # Remove rich-style markup tags for plain output
            clean_text = re.sub(r"\[/?[^\]]*\]", "", text)
            text_parts.append(clean_text)

        sep = str(kwargs.get("sep", " "))
        end = str(kwargs.get("end", "\n"))
        string_parts = [str(part) for part in text_parts]
        output = sep.join(string_parts) + end
        sys.stdout.write(output)
        sys.stdout.flush()

    @classmethod
    def log(cls, *args: FlextTypes.Core.Value, **kwargs: FlextTypes.Core.Value) -> None:
        """Log to console (alias for print)."""
        cls.print(*args, **kwargs)

    # =============================================================================
    # PROCESSING UTILITIES - Consolidated from FlextProcessingUtils
    # =============================================================================

    @classmethod
    def safe_json_parse(
        cls, json_str: str, default: dict[str, object] | None = None
    ) -> dict[str, object]:
        """Safely parse JSON string."""
        try:
            result = json.loads(json_str)
            return result if isinstance(result, dict) else default or {}
        except (json.JSONDecodeError, TypeError):
            return default or {}

    @classmethod
    def safe_json_stringify(cls, obj: object, default: str = "{}") -> str:
        """Safely stringify object to JSON."""
        try:
            return json.dumps(obj, default=str)
        except (TypeError, ValueError):
            return default

    @classmethod
    def extract_model_data(cls, obj: object) -> dict[str, object]:
        """Extract data from Pydantic model or dict."""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()  # type: ignore[no-any-return]
        if hasattr(obj, "dict"):
            return obj.dict()  # type: ignore[no-any-return]
        if isinstance(obj, dict):
            return obj
        return {}

    # =============================================================================
    # RESULT UTILITIES - Consolidated from FlextResultUtilities
    # =============================================================================

    @classmethod
    def chain_results(cls, *results: FlextResult[T]) -> FlextResult[list[T]]:
        """Chain multiple FlextResults into a single result with list of values."""
        values: list[T] = []
        for result in results:
            if result.is_failure:
                return FlextResult[list[T]].fail(result.error or "Chain failed")
            values.append(result.value)
        return FlextResult[list[T]].ok(values)

    @classmethod
    def first_success(cls, *results: FlextResult[T]) -> FlextResult[T]:
        """Return the first successful result, or the last failure."""
        for result in results:
            if result.success:
                return result
        # Return the last result if all failed
        return results[-1] if results else FlextResult[T].fail("No results provided")

    @classmethod
    def collect_errors(cls, *results: FlextResult[T]) -> list[str]:
        """Collect all error messages from failed results."""
        errors: list[str] = [
            result.error or "Unknown error" for result in results if result.is_failure
        ]
        return errors

    @classmethod
    def partition_results(
        cls, results: list[FlextResult[T]]
    ) -> tuple[list[T], list[str]]:
        """Partition results into successes and failures."""
        successes: list[T] = []
        failures: list[str] = []
        for result in results:
            if result.success:
                successes.append(result.value)
            else:
                failures.append(result.error or "Unknown error")
        return successes, failures

    # =============================================================================
    # CLI ERROR HANDLING - Enhanced from existing implementation
    # =============================================================================

    @classmethod
    def handle_cli_errors(
        cls,
        cli_function: Callable[[], None],
        *,
        debug_mode: bool = False,
    ) -> None:
        """Handle CLI main function errors with consistent error handling."""
        try:
            cli_function()
        except KeyboardInterrupt:
            cls.print("\nOperation cancelled by user")
            sys.exit(1)
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            ConnectionError,
            TimeoutError,
        ) as e:
            cls.print(f"Error: {e}")
            if debug_mode:
                cls.print(f"Traceback: {traceback.format_exc()}")
            sys.exit(1)

    # =============================================================================
    # FACTORY UTILITIES - Consolidated from Factory classes
    # =============================================================================

    @classmethod
    def create_factory(cls, factory_type: str, **kwargs: object) -> dict[str, object]:
        """Create a factory instance based on type."""
        # Generic factory creation pattern
        if factory_type == "data":
            return cls._create_data_factory(**kwargs)
        if factory_type == "service":
            return cls._create_service_factory(**kwargs)
        if factory_type == "utility":
            return cls._create_utility_factory(**kwargs)
        msg = f"Unknown factory type: {factory_type}"
        raise ValueError(msg)

    @classmethod
    def _create_data_factory(cls, **kwargs: object) -> dict[str, object]:
        """Create data factory configuration."""
        return {
            "type": "data_factory",
            "config": kwargs,
            "created_at": cls.generate_iso_timestamp(),
        }

    @classmethod
    def _create_service_factory(cls, **kwargs: object) -> dict[str, object]:
        """Create service factory configuration."""
        return {
            "type": "service_factory",
            "config": kwargs,
            "created_at": cls.generate_iso_timestamp(),
        }

    @classmethod
    def _create_utility_factory(cls, **kwargs: object) -> dict[str, object]:
        """Create utility factory configuration."""
        return {
            "type": "utility_factory",
            "config": kwargs,
            "created_at": cls.generate_iso_timestamp(),
        }

    # =============================================================================
    # VALIDATION UTILITIES - Type checking and validation
    # =============================================================================

    @classmethod
    def validate_signature(
        cls, func: Callable[..., object], *args: object, **kwargs: object
    ) -> bool:
        """Validate function call signature."""
        try:
            sig = signature(func)
            sig.bind(*args, **kwargs)
            return True
        except TypeError:
            return False

    @classmethod
    def get_type_name(cls, obj: object) -> str:
        """Get readable type name of an object."""
        return type(obj).__name__

    @classmethod
    def has_method(cls, obj: object, method_name: str) -> bool:
        """Check if object has a callable method."""
        return hasattr(obj, method_name) and callable(getattr(obj, method_name))

    # =============================================================================
    # BACKWARDS COMPATIBILITY - Legacy function aliases
    # =============================================================================

    # Maintain compatibility with existing code while encouraging use of new API
    handle_cli_main_errors = handle_cli_errors  # Alias for backwards compatibility

    @classmethod
    def has_attribute(cls, obj: FlextTypes.Core.Value, attr: str) -> bool:
        """Check if an object has an attribute."""
        return hasattr(obj, attr)

    @classmethod
    def is_instance_of(cls, obj: FlextTypes.Core.Value, target_type: type) -> bool:
        """Check if an object is an instance of a type."""
        return isinstance(obj, target_type)

    @classmethod
    def safe_call(
        cls, func: Callable[[], object] | Callable[[object], object]
    ) -> FlextResult[object]:
        """Safely call function using signature inspection."""
        try:
            # Use signature inspection to determine parameter count safely
            try:
                sig = signature(func)
                param_count = len(sig.parameters)
            except (ValueError, TypeError, OSError):
                # If signature inspection fails, fall back to try-catch approach
                try:
                    # Try zero parameter call first
                    zero_param_func = cast("Callable[[], object]", func)
                    result = zero_param_func()
                    return FlextResult[object].ok(result)
                except TypeError:
                    # Try one parameter call
                    one_param_func = cast("Callable[[object], object]", func)
                    result = one_param_func(object())
                    return FlextResult[object].ok(result)

            # Call based on actual parameter count with proper casting
            if param_count == 0:
                zero_param_func = cast("Callable[[], object]", func)
                result = zero_param_func()
            else:
                one_param_func = cast("Callable[[object], object]", func)
                result = one_param_func(object())

            return FlextResult[object].ok(result)
        except (TypeError, ValueError, AttributeError, RuntimeError) as e:
            return FlextResult[object].fail(str(e))

    @classmethod
    def is_not_none_guard(cls, value: object | None) -> bool:
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
            Converted integer, default value, or None if conversion fails.

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
            except ValueError as e:
                # Log debug information but continue with other conversion methods
                logger.debug(f"String digit conversion failed for '{value}': {e}")
                # Fall through to try other methods

        # Float truncation
        if isinstance(value, float):
            try:
                return int(value)
            except (ValueError, OverflowError) as e:
                # Log debug information but continue with fallback
                logger.debug(f"Float truncation failed for '{value}': {e}")
                # Fall through to try other methods

        return None

    @classmethod
    def _try_string_int_conversion(cls, value: object) -> int | None:
        """Try string-based conversion as fallback."""
        try:
            return int(str(value))
        except (ValueError, TypeError, OverflowError) as e:
            # Log debug information for final fallback failure
            logger.debug(f"String conversion fallback failed for '{value}': {e}")
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
            Converted integer or default (never None).

        """
        converted = cls.safe_int_conversion(value, default)
        return converted if converted is not None else default

    # =================================================================
    # TYPE GUARD UTILITIES - Consolidated from FlextTypeGuards
    # =================================================================

    @classmethod
    def is_list_of(cls, obj: object, item_type: type) -> bool:
        """Check if an object is a list of specific type."""
        if not isinstance(obj, list):
            return False
        # After isinstance check, obj is narrowed to list type
        typed_list = cast("list[object]", obj)
        return all(isinstance(item, item_type) for item in typed_list)

    # =================================================================
    # PERFORMANCE UTILITIES - Consolidated from FlextPerformance
    # =================================================================

    @classmethod
    def clear_performance_metrics_duplicate(cls) -> None:
        """Clear performance metrics - REMOVED DUPLICATE METHOD."""
        # This method was duplicated - the real implementation is in the main FlextUtilities class above
        PERFORMANCE_METRICS.clear()

    # =================================================================
    # LEGACY DUPLICATE METHODS REMOVED
    # =================================================================
    # All duplicate methods that were delegating back to facades have been removed.
    # The correct implementations are in the main FlextUtilities class above.

    @classmethod
    def truncate_text_legacy(
        cls, text: str, max_length: int = 100, suffix: str = "..."
    ) -> str:
        """Truncate text (delegates to FlextFormatters)."""
        return FlextFormatters.truncate(text, max_length, suffix)

    @classmethod
    def format_duration_seconds(cls, seconds: float) -> str:
        """Format duration in seconds (delegates to FlextFormatters)."""
        return FlextFormatters.format_duration(seconds)

    @classmethod
    def get_last_duration_for_key(cls, key: str) -> float:
        """Return last duration for a metric key."""
        metrics = PERFORMANCE_METRICS.get(key)
        if isinstance(metrics, dict):
            duration = metrics.get("duration", 0.0)
            if isinstance(duration, (int, float)):
                return float(duration)
        return 0.0

    @classmethod
    def get_last_duration(cls, category: str, function_name: str) -> float:
        """Return last duration for category/function."""
        key = f"{category}.{function_name}"
        return cls.get_last_duration_for_key(key)

    @classmethod
    def get_last_duration_ms(cls, category: str, function_name: str) -> float:
        """Return last duration (milliseconds) for category/function."""
        return cls.get_last_duration(category, function_name) * 1000.0

    @classmethod
    def iter_metrics_items(
        cls,
    ) -> Iterator[tuple[str, dict[str, dict[str, float | bool]]]]:
        """Iterate over (key, data) metric items with precise typing."""
        metrics = PERFORMANCE_METRICS
        return iter(metrics.items())


# =============================================================================
# FLEXT PERFORMANCE - Static class for performance tracking
# =============================================================================


class FlextPerformance:
    """Performance monitoring utilities - Tier 1 Compatibility Facade.

    This class provides backward compatibility by delegating to FlextUtilities.
    For new code, use FlextUtilities.track_performance() directly.
    """

    @staticmethod
    def track_performance(category: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Track function performance - delegates to FlextUtilities."""
        return FlextUtilities.track_performance(category)

    @staticmethod
    def get_performance_metrics() -> dict[
        str, dict[str, dict[str, dict[str, float | bool]]]
    ]:
        """Get performance metrics - delegates to FlextUtilities."""
        return FlextUtilities.get_performance_metrics()

    @staticmethod
    def clear_performance_metrics() -> None:
        """Clear performance metrics - delegates to FlextUtilities."""
        return FlextUtilities.clear_performance_metrics()

    @staticmethod
    def record_performance(
        category: str,
        function_name: str,
        execution_time: float,
        *,
        _success: bool = True,
    ) -> None:
        """Record performance metrics - delegates to FlextUtilities."""
        return FlextUtilities.record_performance(
            category, function_name, execution_time, _success
        )

    @staticmethod
    def get_last_duration_for_key(key: str) -> float:
        """Return last duration for a metric key - compatibility method."""
        metrics = PERFORMANCE_METRICS.get(key)
        if isinstance(metrics, dict):
            duration = metrics.get("duration", 0.0)
            if isinstance(duration, (int, float)):
                return float(duration)
        return 0.0

    @staticmethod
    def get_last_duration(category: str, function_name: str) -> float:
        """Return last duration for category/function - compatibility method."""
        key = f"{category}.{function_name}"
        return FlextPerformance.get_last_duration_for_key(key)

    @staticmethod
    def get_last_duration_ms(category: str, function_name: str) -> float:
        """Return last duration (milliseconds) for category/function."""
        return FlextPerformance.get_last_duration(category, function_name) * 1000.0

    @staticmethod
    def iter_metrics_items() -> Iterator[
        tuple[str, dict[str, dict[str, float | bool]]]
    ]:
        """Iterate over (key, data) metric items with precise typing."""
        # Type cast is safe since we control the structure of PERFORMANCE_METRICS
        metrics = PERFORMANCE_METRICS
        return iter(metrics.items())


# =============================================================================
# FLEXT CONVERSIONS - Static class for safe conversions
# =============================================================================


class FlextConversions:
    """Type conversion utilities following Single Responsibility Principle.

    SOLID Compliance:
    - SRP: Responsible only for type conversions and data safety
    - OCP: Extensible through static method addition without modification
    - LSP: N/A (utility class, no inheritance)
    - ISP: Focused interface for conversion operations only
    - DIP: Depends on FlextResult abstraction, not concrete implementations
    """

    @staticmethod
    def safe_call(
        func: Callable[[], object] | Callable[[object], object],
    ) -> FlextResult[object]:
        """Safely call function with FlextResult error handling."""
        return FlextUtilities.safe_call(func)

    @staticmethod
    def is_not_none(value: object | None) -> bool:
        """Type guard to check if value is not None."""
        return FlextUtilities.is_not_none_guard(value)


# =============================================================================
# FLEXT PROCESSING UTILS - JSON and model helpers to reduce boilerplate
# =============================================================================


class FlextProcessingUtils:
    """Processing helpers to eliminate repetition in processors.

    - parse_json_object: safely loads JSON string and ensures object (dict)
    - parse_json_to_model: loads JSON and validates via Pydantic model_validate
    """

    @staticmethod
    def parse_json_object(json_text: str) -> FlextResult[dict[str, object]]:
        """Parse JSON string and ensure it is a dict-like object."""
        try:
            data = json.loads(json_text)
            if not isinstance(data, dict):
                return FlextResult[dict[str, object]].fail("JSON must be object")
            # Ensure value type is object for typing consistency
            typed: dict[str, object] = cast("dict[str, object]", data)
            return FlextResult[dict[str, object]].ok(typed)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(f"Invalid JSON: {e}")

    @staticmethod
    def parse_json_to_model[TModel](
        json_text: str, model_cls: type[TModel]
    ) -> FlextResult[TModel]:
        """Parse JSON and validate into the provided Pydantic model class."""
        obj_result = FlextProcessingUtils.parse_json_object(json_text)
        if obj_result.is_failure:
            return FlextResult[TModel].fail(obj_result.error or "Invalid JSON")
        data = obj_result.value

        try:
            # Prefer Pydantic v2 style model_validate
            model_validate = getattr(model_cls, "model_validate", None)
            if callable(model_validate):
                validated = cast("TModel", model_validate(data))
                return FlextResult[TModel].ok(validated)

            # Fallback: try to instantiate directly
            instance = model_cls(**data)
            return FlextResult[TModel].ok(instance)
        except Exception as e:
            return FlextResult[TModel].fail(f"Validation failed: {e}")


# Service classes moved to services.py to avoid circular imports


# =============================================================================
# FLEXT TEXT PROCESSOR - Single Responsibility: Text Operations Only
# =============================================================================


class FlextTextProcessor:
    """Text processing utilities following Single Responsibility Principle.

    SOLID Compliance:
    - SRP: Responsible only for text processing, normalization, and formatting
    - OCP: Extensible through adding new text processing methods
    - LSP: N/A (utility class, no inheritance)
    - ISP: Focused interface for text operations only
    - DIP: No external dependencies, pure text operations
    """

    @staticmethod
    def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text to maximum length with ellipsis.

        SOLID: Single responsibility for text truncation.
        """
        if len(text) <= max_length:
            return text
        cut = max(0, max_length - len(suffix))
        return text[:cut] + suffix

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text.

        SOLID: Single responsibility for whitespace normalization.
        """
        return " ".join(text.split())

    @staticmethod
    def slugify(text: str) -> str:
        """Convert text to URL-friendly slug.

        SOLID: Single responsibility for slug generation.
        """
        # Convert to lowercase and replace non-alphanumeric with hyphens
        slug = re.sub(r"[^\w\s-]", "", text.lower())
        return re.sub(r"[-\s]+", "-", slug).strip("-")

    @staticmethod
    def mask_sensitive(
        text: str,
        *,
        mask_char: str = "*",
        show_first: int = 2,
        show_last: int = 2,
    ) -> str:
        """Mask sensitive information in text.

        SOLID: Single responsibility for data masking.
        """
        if len(text) <= show_first + show_last:
            return mask_char * len(text)

        masked_part = mask_char * (len(text) - show_first - show_last)
        return text[:show_first] + masked_part + text[-show_last:]


# =============================================================================
# FLEXT TIME UTILS - Single Responsibility: Time Operations Only
# =============================================================================


class FlextTimeUtils:
    """Time utilities following Single Responsibility Principle.

    SOLID Compliance:
    - SRP: Responsible only for time formatting and duration operations
    - OCP: Extensible through adding new time formatting methods
    - LSP: N/A (utility class, no inheritance)
    - ISP: Focused interface for time operations only
    - DIP: No external dependencies beyond a standard library
    """

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format.

        SOLID: Single responsibility for duration formatting.
        """
        if seconds < FlextConstants.Performance.SECONDS_PER_MINUTE:
            return f"{seconds:.1f}s"
        if seconds < FlextConstants.Performance.SECONDS_PER_HOUR:
            minutes = seconds / FlextConstants.Performance.SECONDS_PER_MINUTE
            return f"{minutes:.1f}m"
        hours = seconds / FlextConstants.Performance.SECONDS_PER_HOUR
        return f"{hours:.1f}h"

    @staticmethod
    def generate_timestamp() -> float:
        """Generate current timestamp.

        SOLID: Single responsibility for timestamp generation.
        """
        return time.time()


# =============================================================================
# FLEXT ID GENERATOR - Single Responsibility: ID Generation Only
# =============================================================================


class FlextIdGenerator:
    """ID generation utilities following Single Responsibility Principle.

    SOLID Compliance:
    - SRP: Responsible only for generating various types of IDs
    - OCP: Extensible through adding new ID generation methods
    - LSP: N/A (utility class, no inheritance)
    - ISP: Focused interface for ID generation only
    - DIP: Depends on uuid abstraction, not specific implementations
    """

    @staticmethod
    def generate_id() -> str:
        """Generate unique ID."""
        return FlextUtilities.generate_id()

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate correlation ID."""
        return FlextUtilities.generate_correlation_id()

    @staticmethod
    def generate_entity_id() -> str:
        """Generate entity ID."""
        return FlextUtilities.generate_entity_id()

    @staticmethod
    def generate_uuid() -> str:
        """Generate UUID."""
        return FlextUtilities.generate_uuid()

    @staticmethod
    def safe_int_conversion(value: object, default: int | None = None) -> int | None:
        """Safely convert value to integer with optional default (public API).

        SOLID REFACTORING: Eliminates duplication across FLEXT ecosystem.

        Args:
            value: Value to convert to integer
            default: Default value if conversion fails

        Returns:
            Converted integer, default, or None if conversion fails

        """
        return FlextUtilities.safe_int_conversion(value, default)

    @staticmethod
    def safe_int_conversion_with_default(value: object, default: int) -> int:
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
    # SEMANTIC PATTERN UTILITIES - Following domain-specific naming
    # =============================================================================

    @staticmethod
    def generate_timestamp() -> float:
        """Get current UTC timestamp."""
        return datetime.now(UTC).timestamp()

    @staticmethod
    def safe_get(
        dictionary: dict[str, object],
        key: str,
        default: object = None,
    ) -> object:
        """Safely get value from dictionary with default."""
        return dictionary.get(key, default)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text."""
        return FlextTextProcessor.normalize_whitespace(text)

    @staticmethod
    def slugify(text: str) -> str:
        """Convert text to URL-safe slug."""
        return FlextTextProcessor.slugify(text)


# NOTE: mask_sensitive function moved to FlextTextProcessor.mask_sensitive
# NOTE: format_duration function moved to FlextTimeUtils.format_duration
# NOTE: safe_bool_conversion function moved to FlextConversions.safe_bool_conversion


# =============================================================================
# ALIASES FOR SPECIALIZED CLASSES - Direct access to base functionality
# =============================================================================


# Direct aliases to avoid duplication
class FlextTypeGuards:
    """Type guard utilities."""

    @staticmethod
    def has_attribute(obj: object, attr: str) -> bool:
        """Check if an object has an attribute."""
        return hasattr(obj, attr)

    @staticmethod
    def is_instance_of(obj: object, target_type: type) -> bool:
        """Check if an object is an instance of type."""
        return isinstance(obj, target_type)

    @staticmethod
    def is_list_of(obj: object, item_type: type) -> bool:
        """Check if an object is a list of specific type."""
        if not isinstance(obj, list):
            return False
        # After isinstance check, obj is narrowed to list type
        typed_list = cast("list[object]", obj)
        return all(isinstance(item, item_type) for item in typed_list)

    @staticmethod
    def is_not_none_guard(value: object | None) -> bool:
        """Return True if value is not None."""
        return value is not None


class FlextGenerators:
    """ID and timestamp generation utilities - Tier 1 Compatibility Facade.

    This class provides backward compatibility by delegating to FlextUtilities.
    For new code, use FlextUtilities.generate_*() methods directly.
    """

    # Constants for compatibility
    SECONDS_PER_MINUTE: int = 60
    SECONDS_PER_HOUR: int = 3600

    @classmethod
    def generate_uuid(cls) -> str:
        """Generate UUID - delegates to FlextUtilities."""
        return FlextUtilities.generate_uuid()

    @classmethod
    def generate_id(cls) -> str:
        """Generate unique ID - delegates to FlextUtilities."""
        return FlextUtilities.generate_id()

    @classmethod
    def generate_timestamp(cls) -> float:
        """Generate timestamp - delegates to FlextUtilities."""
        return FlextUtilities.generate_timestamp()

    @classmethod
    def generate_iso_timestamp(cls) -> str:
        """Generate ISO format timestamp - delegates to FlextUtilities."""
        return FlextUtilities.generate_iso_timestamp()

    @classmethod
    def generate_correlation_id(cls) -> str:
        """Generate correlation ID - delegates to FlextUtilities."""
        return FlextUtilities.generate_correlation_id()

    @classmethod
    def generate_entity_id(cls) -> str:
        """Generate entity ID - delegates to FlextUtilities."""
        return FlextUtilities.generate_entity_id()

    @classmethod
    def generate_session_id(cls) -> str:
        """Generate session ID - delegates to FlextUtilities."""
        return FlextUtilities.generate_session_id()

    @classmethod
    def format_duration(cls, seconds: float) -> str:
        """Format duration - delegates to FlextUtilities."""
        return FlextUtilities.format_duration(seconds)


class FlextFormatters:
    """Text formatting utilities - Tier 1 Compatibility Facade.

    This class provides backward compatibility by delegating to FlextUtilities.
    For new code, use FlextUtilities.truncate(), FlextUtilities.format_*() directly.
    """

    @staticmethod
    def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text - delegates to FlextUtilities."""
        return FlextUtilities.truncate(text, max_length, suffix)

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration - delegates to FlextUtilities."""
        return FlextUtilities.format_duration(seconds)

    @staticmethod
    def format_bytes(bytes_count: int) -> str:
        """Format bytes - delegates to FlextUtilities."""
        return FlextUtilities.format_bytes(bytes_count)

    @staticmethod
    def format_number(number: float, precision: int = 2) -> str:
        """Format number - delegates to FlextUtilities."""
        return FlextUtilities.format_number(number, precision)


# =============================================================================
# FLEXT FACTORY PATTERNS - SOLID consolidated factory implementation
# =============================================================================


class FlextBaseFactory(ABC, Generic[T]):  # noqa: UP046
    """Abstract base factory providing creation foundation across ecosystem.

    SOLID compliance: Single responsibility for object creation patterns.
    """

    @abstractmethod
    def create(self, **kwargs: object) -> FlextResult[object]:
        """Abstract creation method - must be implemented by concrete factories."""


class FlextGenericFactory(FlextBaseFactory[object]):
    """Generic factory for object creation with type safety.

    SOLID compliance: Open/Closed principle - extensible factory pattern.
    """

    def __init__(self, target_type: type[object]) -> None:
        """Initialize factory with a target type."""
        self._target_type = target_type

    @override
    def create(self, **kwargs: object) -> FlextResult[object]:
        """Create instance of a target type with error handling."""
        try:
            # NOTE: Caller is responsible for providing correct kwargs
            # that match the target type's constructor signature
            instance = self._target_type(**kwargs)
            return FlextResult[object].ok(instance)
        except (TypeError, ValueError, AttributeError, RuntimeError, OSError) as e:
            return FlextResult[object].fail(f"Factory creation failed: {e}")


class FlextUtilityFactory:
    """Concrete factory for creating utility instances following SOLID principles."""

    def __init__(self) -> None:
        """Initialize utility factory."""

    @staticmethod
    def create_generator() -> FlextGenerators:
        """Create generator."""
        return FlextGenerators()

    @staticmethod
    def create_formatter() -> FlextFormatters:
        """Create formatter."""
        return FlextFormatters()

    @staticmethod
    def create_converter() -> FlextConversions:
        """Create converter."""
        return FlextConversions()

    @staticmethod
    def create_performance_tracker() -> FlextPerformance:
        """Create performance tracker."""
        return FlextPerformance()

    @staticmethod
    def create_text_processor() -> FlextTextProcessor:
        """Create text processor."""
        return FlextTextProcessor()


# =============================================================================
# FUNCTION ALIASES - Convenience functions
# =============================================================================


def flext_safe_int_conversion(value: object, default: int | None = None) -> int | None:
    """Alias for safe_int_conversion (convenience function)."""
    return FlextUtilities.safe_int_conversion(value, default)


def generate_correlation_id() -> str:
    """Generate correlation ID for request tracking."""
    return FlextIdGenerator.generate_correlation_id()


def safe_int_conversion_with_default(value: object, default: int) -> int:
    """Safe int conversion with guaranteed default."""
    return FlextUtilities.safe_int_conversion_with_default(value, default)


def flext_clear_performance_metrics() -> None:
    """Clear all performance metrics."""
    FlextPerformance.clear_performance_metrics()


def generate_id() -> str:
    """Generate unique ID."""
    return FlextIdGenerator.generate_id()


def generate_uuid() -> str:
    """Generate UUID."""
    return FlextIdGenerator.generate_uuid()


def is_not_none(value: object) -> bool:
    """Check if the value is not None."""
    return FlextUtilities.is_not_none_guard(value)


# safe_call moved to result.py to avoid type conflicts with generic T version


def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text with suffix."""
    return FlextTextProcessor.truncate(text, max_length, suffix)


def flext_get_performance_metrics() -> dict[
    str, dict[str, dict[str, dict[str, float | bool]]]
]:
    """Get performance metrics."""
    return FlextPerformance.get_performance_metrics()


def flext_record_performance(
    category: str,
    function_name: str,
    execution_time: float,
    *,
    success: bool | None = None,
    _success: bool | None = None,
) -> None:
    """Record performance metrics.

    Backward-compatible: accept both ``success`` (new) and ``_success`` (old) flags.
    """
    flag = (
        success if success is not None else (_success if _success is not None else True)
    )
    FlextPerformance.record_performance(
        category,
        function_name,
        execution_time,
        _success=bool(flag),
    )


def flext_track_performance(
    category: str,
) -> Callable[[FlextDecoratedFunction], FlextDecoratedFunction]:
    """Track performance decorator."""
    return FlextPerformance.track_performance(category)


def generate_iso_timestamp() -> str:
    """Generate ISO timestamp."""
    return FlextGenerators.generate_iso_timestamp()


# Back-compat alias: some modules/tests import `Console`
Console = FlextConsole


# =============================================================================
# FLEXT CORE UTILITIES - Common patterns for codebase consistency
# =============================================================================


class FlextCoreUtilities:
    """Utility methods for common patterns throughout the FLEXT ecosystem.

    This class provides static methods for frequently used operations
    to reduce code duplication and maintain consistency.
    """

    @staticmethod
    def safe_unwrap_or[T](result: FlextResult[T], default: T) -> T:
        """Safely unwrap FlextResult with default value.

        Modern pattern replacing .unwrap_or() calls for cleaner code.

        Args:
            result: FlextResult to unwrap
            default: Default value if result is failure

        Returns:
            Success value or default

        Example:
            value = FlextCoreUtilities.safe_unwrap_or(result, "default")

        """
        return result.unwrap_or(default)

    @staticmethod
    def validate_and_convert[T](
        value: object,
        converter: Callable[[object], T],
        error_message: str = "Conversion failed",
    ) -> FlextResult[T]:
        """Validate and convert value with error handling.

        Args:
            value: Value to convert
            converter: Conversion function
            error_message: Error message if conversion fails

        Returns:
            FlextResult containing converted value or error

        Example:
            result = FlextCoreUtilities.validate_and_convert(
                "123", int, "Invalid integer"
            )

        """
        try:
            converted = converter(value)
            return FlextResult[T].ok(converted)
        except Exception as e:
            return FlextResult[T].fail(f"{error_message}: {e}")

    @staticmethod
    def safe_dict_get[T](
        data: dict[str, object], key: str, expected_type: type[T], default: T
    ) -> T:
        """Safely get and convert value from dict with type checking.

        Args:
            data: Dictionary to extract from
            key: Key to look up
            expected_type: Expected type for casting
            default: Default value if key missing or wrong type

        Returns:
            Typed value or default

        Example:
            log_level = FlextCoreUtilities.safe_dict_get(
                config, "log_level", str, "INFO"
            )

        """
        value = data.get(key, default)
        if isinstance(value, expected_type):
            return value
        return default

    @staticmethod
    def create_logger_with_context(
        module_name: str, context: dict[str, object] | None = None
    ) -> object:
        """Create logger with contextual information.

        Args:
            module_name: Module name for logger
            context: Optional context dict to include in logs

        Returns:
            Configured logger instance

        Example:
            logger = FlextCoreUtilities.create_logger_with_context(
                __name__, {"service": "api", "version": "1.0"}
            )

        """
        logger = FlextLoggerFactory.get_logger(module_name)
        if context:
            # Add context to logger if supported
            # For now, just return the logger - context can be added to individual log calls
            pass
        return logger

    @staticmethod
    def benchmark_operation[T](
        operation: Callable[[], T],
        description: str = "Operation",
        *,
        log_results: bool = True,
    ) -> tuple[T, float]:
        """Benchmark an operation and optionally log results.

        Args:
            operation: Function to benchmark
            description: Description for logging
            log_results: Whether to log timing results

        Returns:
            Tuple of (result, duration_seconds)

        Example:
            result, duration = FlextCoreUtilities.benchmark_operation(
                lambda: expensive_calculation(), "Calculation"
            )

        """
        start_time = time.perf_counter()
        result = operation()
        end_time = time.perf_counter()
        duration = end_time - start_time

        if log_results:
            logger.info(f"{description} completed in {duration:.4f}s")

        return result, duration


# =============================================================================
# FLEXT RESULT UTILITIES - Common patterns for better type safety
# =============================================================================


class FlextResultUtilities:
    """Utility functions for common FlextResult operations."""

    @staticmethod
    def safe_unwrap[T](result: FlextResult[T], default: T) -> T:
        """Safely unwrap FlextResult with default value.

        Modern replacement for: result.value if result.success else default
        """
        return result.unwrap_or(default)

    @staticmethod
    def chain_results[T](*results: FlextResult[T]) -> FlextResult[list[T]]:
        """Chain multiple results, failing on first failure."""
        values: list[T] = []
        for result in results:
            if result.is_failure:
                return FlextResult[list[T]].fail(result.error or "Chain failed")
            values.append(result.value)
        return FlextResult[list[T]].ok(values)

    @staticmethod
    def filter_successes[T](results: list[FlextResult[T]]) -> list[T]:
        """Extract all successful values from a list of results."""
        return [r.value for r in results if r.success]

    @staticmethod
    def collect_errors[T](results: list[FlextResult[T]]) -> list[str]:
        """Collect all error messages from failed results."""
        return [r.error for r in results if r.is_failure and r.error]


class FlextTypeUtilities:
    """Type-safe utility functions for common operations."""

    @staticmethod
    def safe_cast[T](value: object, target_type: type[T]) -> FlextResult[T]:
        """Safely cast value to target type."""
        if isinstance(value, target_type):
            return FlextResult[T].ok(value)
        return FlextResult[T].fail(
            f"Cannot cast {type(value).__name__} to {target_type.__name__}"
        )

    @staticmethod
    def validate_not_none[T](value: T | None) -> FlextResult[T]:
        """Validate value is not None."""
        if value is None:
            return FlextResult[T].fail("Value cannot be None")
        return FlextResult[T].ok(value)


# =============================================================================
# EXPORTS - Main architectural classes and utilities
# =============================================================================

# =============================================================================
# TIER 1 MODULE PATTERN - Single Export Only
# =============================================================================

__all__: list[str] = [
    "FlextUtilities",  # 🎯 SINGLE EXPORT: All utility functionality consolidated
    # =======================================================================
    # LEGACY COMPATIBILITY - Function aliases only (not classes)
    # =======================================================================
    "flext_clear_performance_metrics",  # → FlextUtilities.clear_performance_metrics()
    "flext_get_performance_metrics",  # → FlextUtilities.get_performance_metrics()
    "flext_record_performance",  # → FlextUtilities.record_performance()
    "flext_safe_int_conversion",  # → FlextUtilities.safe_int_conversion()
    "flext_track_performance",  # → FlextUtilities.track_performance()
    "generate_correlation_id",  # → FlextUtilities.generate_correlation_id()
    "generate_id",  # → FlextUtilities.generate_id()
    "generate_iso_timestamp",  # → FlextUtilities.generate_iso_timestamp()
    "generate_uuid",  # → FlextUtilities.generate_uuid()
    "is_not_none",  # → FlextUtilities.is_not_none()
    "safe_int_conversion_with_default",  # → FlextUtilities.safe_int_conversion_with_default()
    "truncate",  # → FlextUtilities.truncate()
    # =======================================================================
    # NOTE: All FlextXxx classes have been internalized into FlextUtilities
    # Use FlextUtilities methods directly for all functionality
    # =======================================================================
]
