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
PERFORMANCE_METRICS: FlextTypes.Core.Dict = {}


class FlextConsole:
    """Native console implementation without external dependencies.

    Simple console interface that provides basic printing capabilities
    following FLEXT's minimal dependency principle. Eliminates rich dependency
    as requested to remove fallback imports and use real implementation.
    """

    def __init__(self) -> None:
        """Initialize console."""

    @staticmethod
    def print(*args: FlextTypes.Core.Value, **kwargs: FlextTypes.Core.Value) -> None:
        """Print to console with standard print function.

        Supports rich-style markup but ignores it,
        focusing on the text content only.
        """
        text_parts: FlextTypes.Core.List = []
        for arg in args:
            text = str(arg)
            # Remove rich-style markup tags (e.g., [red]...[/red]) for plain output
            clean_text = re.sub(r"\[/?[^\]]*\]", "", text)
            text_parts.append(clean_text)

        # Use sys.stdout.write instead of print to avoid T201 linting error
        # Handle kwargs similar to standard print (sep, end, etc.)
        sep = str(kwargs.get("sep", " "))
        end = str(kwargs.get("end", "\n"))
        # Convert all parts to strings for proper joining
        string_parts = [str(part) for part in text_parts]
        output = sep.join(string_parts) + end
        sys.stdout.write(output)
        sys.stdout.flush()

    def log(
        self, *args: FlextTypes.Core.Value, **kwargs: FlextTypes.Core.Value
    ) -> None:
        """Log to console (alias for print)."""
        self.print(*args, **kwargs)


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
        """Truncate text (delegates to FlextTextProcessor)."""
        return FlextTextProcessor.truncate(text, max_length, suffix)

    @classmethod
    def handle_cli_main_errors(
        cls,
        cli_function: FlextTypes.Protocol.Callable[None],
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
        console = FlextConsole()

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


# =============================================================================
# FLEXT PERFORMANCE - Static class for performance tracking
# =============================================================================


class FlextPerformance:
    """Performance monitoring utilities following Single Responsibility Principle.

    SOLID Compliance:
    - SRP: Responsible only for performance tracking and metrics
    - OCP: Extensible through adding new metric types without modification
    - LSP: N/A (utility class, no inheritance)
    - ISP: Focused interface for performance operations only
    - DIP: Depends on abstractions (decorators, metrics dict) not implementations
    """

    @staticmethod
    def track_performance(category: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Track function performance as decorator, preserving type signatures."""

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                except (RuntimeError, ValueError, TypeError):
                    execution_time = time.time() - start_time
                    FlextPerformance.record_performance(
                        category,
                        getattr(func, "__name__", "unknown"),
                        execution_time,
                        _success=False,
                    )
                    raise
                else:
                    execution_time = time.time() - start_time
                    FlextPerformance.record_performance(
                        category,
                        getattr(func, "__name__", "unknown"),
                        execution_time,
                        _success=True,
                    )
                    return result

            return wrapper

        return decorator

    @staticmethod
    def get_performance_metrics() -> dict[str, dict[str, dict[str, int | float]]]:
        """Get performance metrics for observability with precise typing."""
        # Type cast is safe since we control the structure of PERFORMANCE_METRICS
        return {"metrics": cast("dict[str, dict[str, int | float]]", PERFORMANCE_METRICS)}

    @staticmethod
    def clear_performance_metrics() -> None:
        """Clear performance metrics (for testing)."""
        PERFORMANCE_METRICS.clear()

    @staticmethod
    def record_performance(
        category: str,
        function_name: str,
        execution_time: float,
        *,
        _success: bool,
    ) -> None:
        """Record performance metrics for observability."""
        # Note: success parameter reserved for future observability features
        _ = _success  # Reserved for future observability metrics
        key = f"{category}.{function_name}"
        data = PERFORMANCE_METRICS.get(key)
        if not isinstance(data, dict):
            PERFORMANCE_METRICS[key] = {
                "last_duration": execution_time,
                "count": 1,
                "success": int(bool(_success)),
                "failure": int(not _success),
            }
        else:
            data["last_duration"] = execution_time
            data["count"] = int(data.get("count", 0)) + 1
            if _success:
                data["success"] = int(data.get("success", 0)) + 1
            else:
                data["failure"] = int(data.get("failure", 0)) + 1

    # -----------------------------
    # Convenience helpers
    # -----------------------------

    @staticmethod
    def get_last_duration_for_key(key: str) -> float:
        """Return last duration (seconds) for a metric key or 0.0 if missing."""
        data = PERFORMANCE_METRICS.get(key)
        if isinstance(data, dict):
            value = data.get("last_duration", 0.0)
            try:
                return float(value)
            except Exception:
                return 0.0
        return 0.0

    @staticmethod
    def get_last_duration(category: str, function_name: str) -> float:
        """Return last duration (seconds) for category/function."""
        key = f"{category}.{function_name}"
        return FlextPerformance.get_last_duration_for_key(key)

    @staticmethod
    def get_last_duration_ms(category: str, function_name: str) -> float:
        """Return last duration (milliseconds) for category/function."""
        return FlextPerformance.get_last_duration(category, function_name) * 1000.0

    @staticmethod
    def iter_metrics_items() -> Iterator[tuple[str, dict[str, int | float]]]:
        """Iterate over (key, data) metric items with precise typing."""
        # Type cast is safe since we control the structure of PERFORMANCE_METRICS
        metrics = cast("dict[str, dict[str, int | float]]", PERFORMANCE_METRICS)
        return iter(list(metrics.items()))


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
            typed: dict[str, object] = dict(data)
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
        return all(isinstance(item, item_type) for item in obj)

    @staticmethod
    def is_not_none_guard(value: object | None) -> bool:
        """Return True if value is not None."""
        return value is not None


class FlextGenerators:
    """ID and timestamp generation utilities - SINGLE SOURCE OF TRUTH."""

    # Constants reused by formatting helpers
    SECONDS_PER_MINUTE: int = 60
    SECONDS_PER_HOUR: int = 3600

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
        return datetime.now(UTC).isoformat()

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


def flext_get_performance_metrics() -> dict[str, dict[str, dict[str, int | float]]]:
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
# EXPORTS - Clean public API following directives
# =============================================================================

__all__: list[str] = [
    "FlextBaseFactory",  # Factory patterns following SOLID principles
    "FlextConversions",  # Type conversions
    "FlextFormatters",  # Text formatting
    "FlextGenerators",  # ID/timestamp generation
    "FlextGenericFactory",
    "FlextIdGenerator",  # ID generation only (delegates to FlextGenerators)
    "FlextPerformance",  # Performance tracking
    "FlextProcessingUtils",  # Processing helpers (JSON/model)
    "FlextTextProcessor",  # Text processing
    "FlextTimeUtils",  # Time operations only
    "FlextTypeGuards",  # Type checking utilities
    "FlextUtilities",  # General utilities orchestration (Open/Closed)
    "FlextUtilityFactory",  # Concrete utility factory
    "flext_clear_performance_metrics",
    "flext_get_performance_metrics",
    "flext_record_performance",
    # Legacy function aliases
    "flext_safe_int_conversion",
    "flext_track_performance",
    "generate_correlation_id",
    "generate_id",
    "generate_iso_timestamp",
    "generate_uuid",
    "is_not_none",
    "safe_int_conversion_with_default",
    "truncate",
]
