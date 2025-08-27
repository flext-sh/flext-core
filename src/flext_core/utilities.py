"""Common utility functions.

SINGLE CONSOLIDATED MODULE following FLEXT architectural patterns.
All utility functionality consolidated into FlextUtilities.
"""

from __future__ import annotations

import functools
import json
import re
import sys
import time
import traceback
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from inspect import signature
from typing import Any, Protocol, cast, override

from flext_core.constants import FlextConstants
from flext_core.loggings import FlextLoggerFactory
from flext_core.result import FlextResult
from flext_core.typings import (
    FlextTypes,
    P,
    R,
    T,
)
from flext_core.validation import FlextPredicates

logger = FlextLoggerFactory.get_logger(__name__)

# Global storage for performance metrics
PERFORMANCE_METRICS: dict[str, dict[str, dict[str, float | bool]]] = {}


class FlextUtilities:
    """SINGLE CONSOLIDATED CLASS for all utility functionality.

    Following FLEXT architectural patterns - consolidates ALL utility functionality
    including ID generation, time operations, text processing, performance tracking,
    type conversions, formatting, and factory patterns into one main class
    with nested classes for organization.

    CONSOLIDATED CLASSES: FlextGenerators + FlextPerformance + FlextConversions +
    FlextProcessingUtils + FlextTextProcessor + FlextTimeUtils + FlextIdGenerator +
    FlextTypeGuards + FlextFormatters + FlextBaseFactory + FlextGenericFactory +
    FlextUtilityFactory + FlextResultUtilities + FlextTypeUtilities
    """

    # Constants for validation
    MIN_SERVICE_NAME_LENGTH = 2
    MIN_PORT = 1
    MAX_PORT = 65535
    MIN_PERCENTAGE = 0.0
    MAX_PERCENTAGE = 100.0

    # ==========================================================================
    # NESTED CLASSES FOR ORGANIZATION
    # ==========================================================================

    class DecoratedFunction(Protocol):
        """Protocol for functions that can be decorated with performance tracking."""

        __name__: str

        def __call__(
            self, *args: FlextTypes.Core.Value, **kwargs: FlextTypes.Core.Value
        ) -> FlextTypes.Core.Value:
            """Call decorated function with provided arguments."""

    class Generators:
        """Nested ID and timestamp generation utilities."""

        @staticmethod
        def generate_uuid() -> str:
            """Generate standard UUID4."""
            return str(uuid.uuid4())

        @staticmethod
        def generate_id() -> str:
            """Generate unique ID with flext_ prefix."""
            return f"flext_{uuid.uuid4().hex[:8]}"

        @staticmethod
        def generate_timestamp() -> float:
            """Generate Unix timestamp."""
            return time.time()

        @staticmethod
        def generate_iso_timestamp() -> str:
            """Generate ISO 8601 timestamp."""
            return datetime.now(UTC).isoformat()

        @staticmethod
        def generate_correlation_id() -> str:
            """Generate correlation ID with corr_ prefix."""
            return f"corr_{uuid.uuid4().hex[:16]}"

        @staticmethod
        def generate_entity_id() -> str:
            """Generate entity ID with entity_ prefix."""
            return f"entity_{uuid.uuid4().hex[:12]}"

        @staticmethod
        def generate_service_name(prefix: str = "flext") -> str:
            """Generate a service name with prefix."""
            return f"{prefix}_{uuid.uuid4().hex[:8]}"

        @staticmethod
        def generate_session_id() -> str:
            """Generate session ID with sess_ prefix."""
            return f"sess_{uuid.uuid4().hex[:10]}"

        @staticmethod
        def generate_request_id() -> str:
            """Generate request ID with req_ prefix."""
            return f"req_{uuid.uuid4().hex[:10]}"

    class LdapConverters:
        """Nested LDAP data conversion utilities."""

        @staticmethod
        def safe_convert_value_to_str(value: object) -> str:
            """Safely convert any value to string, handling bytes properly."""
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value)

        @staticmethod
        def safe_convert_list_to_strings(values: list[object]) -> list[str]:
            """Safely convert list of values to list of strings."""
            result: list[str] = []
            for item in values:
                try:
                    str_item = FlextUtilities.LdapConverters.safe_convert_value_to_str(
                        item
                    )
                    if str_item:  # Only add non-empty strings
                        result.append(str_item)
                except Exception as e:
                    # Log conversion failure and skip items that can't be converted
                    logger.debug(f"Failed to convert item to string: {item!r} - {e}")
                    continue
            return result

        @staticmethod
        def safe_convert_to_ldap_attribute_list(source_value: object) -> list[str]:
            """Safely convert unknown value to list of strings for LDAP attributes."""
            if isinstance(source_value, list):
                typed_list: list[object] = cast("list[object]", source_value)
                return FlextUtilities.LdapConverters.safe_convert_list_to_strings(
                    typed_list
                )
            str_value = FlextUtilities.LdapConverters.safe_convert_value_to_str(
                source_value
            )
            return [str_value] if str_value else []

        @staticmethod
        def safe_convert_external_dict_to_ldap_attributes(
            source_dict: dict[object, object] | object,
        ) -> dict[str, str | list[str]]:
            """Safely convert unknown dict to typed LDAP attributes.

            Handles the common case where external systems provide dictionaries
            with unknown value types that need to be converted to LDAP-compatible
            string or list[str] format.

            Args:
                source_dict: Dictionary from external source with unknown value types

            Returns:
                Dictionary with string keys and string/list[str] values

            Example:
                # Input from external system
                external_data = {
                    'uid': 'john',
                    'cn': ['John', 'Doe'],
                    'mail': 'john@example.com',
                    'binary_attr': b'binary_data'
                }

                # Safe conversion
                ldap_attrs = safe_convert_external_dict_to_ldap_attributes(external_data)
                # Result: {'uid': 'john', 'cn': ['John', 'Doe'], 'mail': 'john@example.com'}

            """
            if not isinstance(source_dict, dict):
                return {}

            # Use cast for proper typing since we checked isinstance
            typed_dict = cast("FlextTypes.Core.Dict", source_dict)
            result: dict[str, str | list[str]] = {}
            for key, value in typed_dict.items():
                str_key = str(key)  # Ensure key is string
                if isinstance(value, list):
                    # Convert list values to strings with proper typing
                    typed_list: list[object] = value
                    converted_list = (
                        FlextUtilities.LdapConverters.safe_convert_list_to_strings(
                            typed_list
                        )
                    )
                    if len(converted_list) == 1:
                        result[str_key] = converted_list[0]  # Single item as string
                    elif converted_list:
                        result[str_key] = converted_list  # Multiple items as list
                    # Skip empty lists
                else:
                    # Convert single value to string
                    converted_str = (
                        FlextUtilities.LdapConverters.safe_convert_value_to_str(value)
                    )
                    if converted_str:  # Only add non-empty strings
                        result[str_key] = converted_str

            return result

        @staticmethod
        def normalize_attributes(
            attrs: dict[str, object],
        ) -> dict[str, str | list[str]]:
            """Normalize mapping: lists -> list[str], scalars -> str."""
            if not attrs:
                return {}

            # Optimized with dictionary comprehension for better performance
            def coerce_value(value: object) -> str | list[str]:
                """Normalize attribute value to str or list[str]."""
                if isinstance(value, list):
                    return [str(item) for item in cast("list[object]", value)]
                return str(value)

            return {key: coerce_value(value) for key, value in attrs.items()}

    class TextProcessor:
        """Nested text processing utilities."""

        @staticmethod
        def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
            """Truncate text to maximum length with suffix."""
            if len(text) <= max_length:
                return text
            if max_length <= 0:
                return ""  # Return empty string for zero or negative max_length
            if max_length < len(suffix):
                return text[:max_length]  # Can't fit suffix, just truncate
            cut = max(0, max_length - len(suffix))
            return text[:cut] + suffix

        @staticmethod
        def clean_text(text: str) -> str:
            """Clean text by removing extra whitespace and normalizing."""
            return re.sub(r"\s+", " ", text.strip())

        @staticmethod
        def extract_numbers(text: str) -> list[str]:
            """Extract all numbers from text."""
            return re.findall(r"\d+", text)

        @staticmethod
        def sanitize_filename(text: str) -> str:
            """Sanitize filename by removing invalid characters."""
            return re.sub(r'[<>:"/\\|?*]', "_", text)

        @staticmethod
        def slugify(text: str) -> str:
            """Convert text to URL-friendly slug."""
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
            """Mask sensitive information in text."""
            if len(text) <= show_first + show_last:
                return mask_char * len(text)
            masked_part = mask_char * (len(text) - show_first - show_last)
            return text[:show_first] + masked_part + text[-show_last:]

    class TimeUtils:
        """Nested time operations utilities."""

        # Constants for time formatting
        SECONDS_PER_MINUTE = 60
        SECONDS_PER_HOUR = 3600

        @staticmethod
        def format_duration(seconds: float) -> str:
            """Format duration in human-readable format."""
            if seconds < FlextUtilities.TimeUtils.SECONDS_PER_MINUTE:
                return f"{seconds:.2f}s"
            if seconds < FlextUtilities.TimeUtils.SECONDS_PER_HOUR:
                minutes = seconds / FlextUtilities.TimeUtils.SECONDS_PER_MINUTE
                return f"{minutes:.1f}m"
            hours = seconds / FlextUtilities.TimeUtils.SECONDS_PER_HOUR
            return f"{hours:.1f}h"

        @staticmethod
        def parse_iso_timestamp(timestamp: str) -> datetime:
            """Parse ISO timestamp to datetime object."""
            return datetime.fromisoformat(timestamp)

        @staticmethod
        def get_elapsed_time(start_time: float) -> float:
            """Get elapsed time from start timestamp."""
            return time.time() - start_time

    class Performance:
        """Nested performance tracking utilities."""

        @staticmethod
        def track_performance(
            category: str,
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
                        FlextUtilities.Performance.record_performance(
                            category, func.__name__, execution_time, success=False
                        )
                        raise
                    else:
                        execution_time = time.time() - start_time
                        FlextUtilities.Performance.record_performance(
                            category, func.__name__, execution_time, success=True
                        )
                        return result

                return wrapper

            return decorator

        @staticmethod
        def record_performance(
            category: str, operation: str, duration: float, *, success: bool = True
        ) -> None:
            """Record performance metric."""
            if category not in PERFORMANCE_METRICS:
                PERFORMANCE_METRICS[category] = {}
            PERFORMANCE_METRICS[category][operation] = {
                "duration": duration,
                "success": success,
                "timestamp": time.time(),
            }

        @staticmethod
        def get_performance_metrics() -> FlextTypes.Core.PerformanceMetrics:
            """Get all performance metrics."""
            return {"metrics": PERFORMANCE_METRICS}

        @staticmethod
        def clear_performance_metrics() -> None:
            """Clear all performance metrics."""
            PERFORMANCE_METRICS.clear()

    class Conversions:
        """Nested type conversion utilities."""

        @staticmethod
        def safe_int(value: FlextTypes.Core.Value, default: int = 0) -> int:
            """Safely convert value to integer."""
            try:
                if value is None:
                    return default
                if isinstance(value, str):
                    return int(float(value))  # Handle "3.14" -> 3
                if isinstance(value, (int, float)):
                    return int(value)
                return int(str(value))  # Convert to string first for safety
            except (ValueError, TypeError):
                return default

        @staticmethod
        def safe_float(value: FlextTypes.Core.Value, default: float = 0.0) -> float:
            """Safely convert value to float."""
            try:
                if value is None:
                    return default
                if isinstance(value, (int, float, str)):
                    return float(value)
                return float(str(value))  # Convert to string first for safety
            except (ValueError, TypeError):
                return default

        @staticmethod
        def safe_str(value: FlextTypes.Core.Value, default: str = "") -> str:
            """Safely convert value to string."""
            try:
                return str(value)
            except Exception:
                return default

        @staticmethod
        def to_bool(value: FlextTypes.Core.Value) -> bool:
            """Convert value to boolean with intelligent parsing."""
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in {"true", "yes", "1", "on", "enabled"}
            return bool(value)

    class TypeGuards:
        """Nested type guard utilities."""

        @staticmethod
        def is_not_none(value: object) -> bool:
            """Check if value is not None."""
            return value is not None

        @staticmethod
        def is_string(value: object) -> bool:
            """Check if value is a string."""
            return isinstance(value, str)

        @staticmethod
        def is_non_empty_string(value: object) -> bool:
            """Check if value is a non-empty string."""
            return isinstance(value, str) and len(value.strip()) > 0

        @staticmethod
        def is_numeric(value: object) -> bool:
            """Check if value is numeric (int or float)."""
            return isinstance(value, (int, float))

        @staticmethod
        def is_list(value: object) -> bool:
            """Check if value is a list."""
            return isinstance(value, list)

        @staticmethod
        def is_dict(value: object) -> bool:
            """Check if value is a dictionary."""
            return isinstance(value, dict)

        @staticmethod
        def is_list_of(obj: object, item_type: type) -> bool:
            """Check if an object is a list of specific type."""
            if not isinstance(obj, list):
                return False
            typed_list = cast("list[object]", obj)
            return all(isinstance(item, item_type) for item in typed_list)

        @staticmethod
        def is_email(value: object) -> bool:
            """Check if value is a valid email address."""
            if not isinstance(value, str):
                return False
            pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            return bool(re.match(pattern, value))

        @staticmethod
        def is_uuid(value: object) -> bool:
            """Check if value is a valid UUID."""
            if not isinstance(value, str):
                return False
            pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
            return bool(re.match(pattern, value.lower()))

        @staticmethod
        def is_url(value: object) -> bool:
            """Check if value is a valid URL."""
            if not isinstance(value, str):
                return False
            return value.startswith(("http://", "https://"))

        @staticmethod
        def matches_pattern(value: object, pattern: str) -> bool:
            """Check if string matches regex pattern."""
            if not isinstance(value, str):
                return False
            return bool(re.match(pattern, value))

    class Formatters:
        """Nested formatting utilities."""

        # Constants for byte formatting
        BYTES_PER_KB = 1024
        BYTES_PER_MB = 1024 * 1024
        BYTES_PER_GB = 1024 * 1024 * 1024

        @staticmethod
        def format_bytes(bytes_count: int) -> str:
            """Format byte count in human-readable format."""
            if bytes_count < FlextUtilities.Formatters.BYTES_PER_KB:
                return f"{bytes_count} B"
            if bytes_count < FlextUtilities.Formatters.BYTES_PER_MB:
                return f"{bytes_count / FlextUtilities.Formatters.BYTES_PER_KB:.1f} KB"
            if bytes_count < FlextUtilities.Formatters.BYTES_PER_GB:
                return f"{bytes_count / FlextUtilities.Formatters.BYTES_PER_MB:.1f} MB"
            return f"{bytes_count / FlextUtilities.Formatters.BYTES_PER_GB:.1f} GB"

        @staticmethod
        def format_number(number: float, precision: int = 2) -> str:
            """Format number with thousand separators."""
            return f"{number:,.{precision}f}"

        @staticmethod
        def format_percentage(value: float, precision: int = 1) -> str:
            """Format value as percentage."""
            return f"{value * 100:.{precision}f}%"

    class ProcessingUtils:
        """Nested processing utilities for JSON and models."""

        @staticmethod
        def safe_json_parse(
            json_str: str, default: dict[str, object] | None = None
        ) -> dict[str, object]:
            """Safely parse JSON string."""
            try:
                result = json.loads(json_str)
                if isinstance(result, dict):
                    typed_result: dict[str, object] = result
                    return typed_result
                return default or {}
            except (json.JSONDecodeError, TypeError):
                return default or {}

        @staticmethod
        def safe_json_stringify(obj: object, default: str = "{}") -> str:
            """Safely stringify object to JSON."""
            try:
                return json.dumps(obj, default=str)
            except (TypeError, ValueError):
                return default

        @staticmethod
        def extract_model_data(obj: object) -> dict[str, object]:
            """Extract data from Pydantic model or dict."""
            if hasattr(obj, "model_dump"):
                result = obj.model_dump()  # type: ignore[attr-defined]
                return cast("dict[str, object]", result)
            if hasattr(obj, "dict"):
                result = obj.dict()  # type: ignore[attr-defined]
                return cast("dict[str, object]", result)
            if isinstance(obj, dict):
                typed_obj: dict[str, object] = obj
                return typed_obj
            return {}

    class ResultUtils:
        """Nested FlextResult utilities."""

        @staticmethod
        def chain_results(*results: FlextResult[T]) -> FlextResult[list[T]]:
            """Chain multiple FlextResults into a single result with list of values."""
            values: list[T] = []
            for result in results:
                if result.is_failure:
                    return FlextResult[list[T]].fail(result.error or "Chain failed")
                values.append(result.value)
            return FlextResult[list[T]].ok(values)

        @staticmethod
        def first_success(*results: FlextResult[T]) -> FlextResult[T]:
            """Return the first successful result, or the last failure."""
            for result in results:
                if result.success:
                    return result
            return (
                results[-1] if results else FlextResult[T].fail("No results provided")
            )

        @staticmethod
        def collect_errors(*results: FlextResult[T]) -> list[str]:
            """Collect all error messages from failed results."""
            errors: list[str] = [
                result.error or "Unknown error"
                for result in results
                if result.is_failure
            ]
            return errors

        @staticmethod
        def partition_results(
            results: list[FlextResult[T]],
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

    class BaseFactory[TFactory](Protocol):
        """Nested factory protocol for type safety."""

        def create(self, **kwargs: object) -> FlextResult[object]:
            """Factory creation method."""
            ...

    class GenericFactory(BaseFactory[object]):
        """Nested generic factory for object creation."""

        def __init__(self, target_type: type[object]) -> None:
            """Initialize factory with a target type."""
            self._target_type = target_type

        @override
        def create(self, **kwargs: object) -> FlextResult[object]:
            """Create instance of target type with error handling."""
            try:
                instance = self._target_type(**kwargs)
                return FlextResult[object].ok(instance)
            except (TypeError, ValueError, AttributeError, RuntimeError, OSError) as e:
                return FlextResult[object].fail(f"Factory creation failed: {e}")

    class SimpleFactory:
        """Simple factory pattern for any class with FlextResult error handling."""

        def __init__(self, target_class: type) -> None:
            """Initialize factory with target class."""
            self._target_class = target_class

        def create(self, **kwargs: object) -> FlextResult[object]:
            """Create instance with error handling."""
            try:
                instance = self._target_class(**kwargs)
                return FlextResult[object].ok(instance)
            except Exception as e:
                return FlextResult[object].fail(f"Factory failed: {e}")

    class SimpleBuilder:
        """Simple builder pattern for fluent object construction."""

        def __init__(self, target_class: type) -> None:
            """Initialize builder with target class."""
            self._target_class = target_class
            self._kwargs: dict[str, object] = {}

        def set(self, **kwargs: object) -> FlextUtilities.SimpleBuilder:
            """Set builder parameters fluently."""
            self._kwargs.update(kwargs)
            return self

        def build(self) -> FlextResult[object]:
            """Build instance with accumulated parameters."""
            try:
                instance = self._target_class(**self._kwargs)
                return FlextResult[object].ok(instance)
            except Exception as e:
                return FlextResult[object].fail(f"Builder failed: {e}")

    # ==========================================================================
    # CONSTANTS - Class constants
    # ==========================================================================
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = 3600
    BYTES_PER_KB = 1024
    BYTES_PER_MB = 1024 * 1024
    BYTES_PER_GB = 1024 * 1024 * 1024

    # ==========================================================================
    # MAIN UTILITY FUNCTIONALITY - Access to nested classes
    # ==========================================================================

    @classmethod
    def generate_uuid(cls) -> str:
        """Generate UUID - delegates to nested Generators."""
        return cls.Generators.generate_uuid()

    @classmethod
    def generate_id(cls) -> str:
        """Generate unique ID - delegates to nested Generators."""
        return cls.Generators.generate_id()

    @classmethod
    def generate_timestamp(cls) -> float:
        """Generate timestamp - delegates to nested Generators."""
        return cls.Generators.generate_timestamp()

    @classmethod
    def generate_iso_timestamp(cls) -> str:
        """Generate ISO timestamp - delegates to nested Generators."""
        return cls.Generators.generate_iso_timestamp()

    @classmethod
    def generate_correlation_id(cls) -> str:
        """Generate correlation ID - delegates to nested Generators."""
        return cls.Generators.generate_correlation_id()

    @classmethod
    def generate_entity_id(cls) -> str:
        """Generate entity ID - delegates to nested Generators."""
        return cls.Generators.generate_entity_id()

    @classmethod
    def generate_session_id(cls) -> str:
        """Generate session ID - delegates to nested Generators."""
        return cls.Generators.generate_session_id()

    @classmethod
    def generate_request_id(cls) -> str:
        """Generate request ID - delegates to nested Generators."""
        return cls.Generators.generate_request_id()

    @classmethod
    def truncate(cls, text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text - delegates to nested TextProcessor."""
        return cls.TextProcessor.truncate(text, max_length, suffix)

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Clean text - delegates to nested TextProcessor."""
        return cls.TextProcessor.clean_text(text)

    @classmethod
    def extract_numbers(cls, text: str) -> list[str]:
        """Extract numbers - delegates to nested TextProcessor."""
        return cls.TextProcessor.extract_numbers(text)

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
        """Sanitize filename - delegates to nested TextProcessor."""
        return cls.TextProcessor.sanitize_filename(filename)

    @classmethod
    def format_duration(cls, seconds: float) -> str:
        """Format duration - delegates to nested TimeUtils."""
        return cls.TimeUtils.format_duration(seconds)

    @classmethod
    def parse_iso_timestamp(cls, timestamp: str) -> datetime:
        """Parse ISO timestamp - delegates to nested TimeUtils."""
        return cls.TimeUtils.parse_iso_timestamp(timestamp)

    @classmethod
    def get_elapsed_time(cls, start_time: float) -> float:
        """Get elapsed time - delegates to nested TimeUtils."""
        return cls.TimeUtils.get_elapsed_time(start_time)

    @classmethod
    def track_performance(
        cls, category: str
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Track performance - delegates to nested Performance."""
        return cls.Performance.track_performance(category)

    @classmethod
    def record_performance(
        cls, category: str, operation: str, duration: float, *, success: bool = True
    ) -> None:
        """Record performance - delegates to nested Performance."""
        return cls.Performance.record_performance(
            category, operation, duration, success=success
        )

    @classmethod
    def get_performance_metrics(
        cls,
    ) -> FlextTypes.Core.PerformanceMetrics:
        """Get performance metrics - delegates to nested Performance."""
        return cls.Performance.get_performance_metrics()

    @classmethod
    def clear_performance_metrics(cls) -> None:
        """Clear performance metrics - delegates to nested Performance."""
        return cls.Performance.clear_performance_metrics()

    @classmethod
    def safe_int(cls, value: FlextTypes.Core.Value, default: int = 0) -> int:
        """Safe int conversion - delegates to nested Conversions."""
        return cls.Conversions.safe_int(value, default)

    @classmethod
    def safe_float(cls, value: FlextTypes.Core.Value, default: float = 0.0) -> float:
        """Safe float conversion - delegates to nested Conversions."""
        return cls.Conversions.safe_float(value, default)

    @classmethod
    def safe_str(cls, value: FlextTypes.Core.Value, default: str = "") -> str:
        """Safe string conversion - delegates to nested Conversions."""
        return cls.Conversions.safe_str(value, default)

    @classmethod
    def to_bool(cls, value: FlextTypes.Core.Value) -> bool:
        """Boolean conversion - delegates to nested Conversions."""
        return cls.Conversions.to_bool(value)

    @classmethod
    def is_not_none(cls, value: object) -> bool:
        """Type guard - delegates to nested TypeGuards."""
        return cls.TypeGuards.is_not_none(value)

    @classmethod
    def is_string(cls, value: object) -> bool:
        """Type guard - delegates to nested TypeGuards."""
        return cls.TypeGuards.is_string(value)

    @classmethod
    def is_non_empty_string(cls, value: object) -> bool:
        """Type guard - delegates to nested TypeGuards."""
        return cls.TypeGuards.is_non_empty_string(value)

    @classmethod
    def is_numeric(cls, value: object) -> bool:
        """Type guard - delegates to nested TypeGuards."""
        return cls.TypeGuards.is_numeric(value)

    @classmethod
    def is_list(cls, value: object) -> bool:
        """Type guard - delegates to nested TypeGuards."""
        return cls.TypeGuards.is_list(value)

    @classmethod
    def is_dict(cls, value: object) -> bool:
        """Type guard - delegates to nested TypeGuards."""
        return cls.TypeGuards.is_dict(value)

    @classmethod
    def format_bytes(cls, bytes_count: int) -> str:
        """Format bytes - delegates to nested Formatters."""
        return cls.Formatters.format_bytes(bytes_count)

    @classmethod
    def format_number(cls, number: float, precision: int = 2) -> str:
        """Format number - delegates to nested Formatters."""
        return cls.Formatters.format_number(number, precision)

    @classmethod
    def format_percentage(cls, value: float, precision: int = 1) -> str:
        """Format percentage - delegates to nested Formatters."""
        return cls.Formatters.format_percentage(value, precision)

    # =============================================================================
    # CONSOLE OPERATIONS - Consolidated from FlextUtilities
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
            if isinstance(result, dict):
                typed_result: dict[str, object] = result
                return typed_result
            return default or {}
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
            result = obj.model_dump()  # type: ignore[attr-defined]
            return cast("dict[str, object]", result)
        if hasattr(obj, "dict"):
            result = obj.dict()  # type: ignore[attr-defined]
            return cast("dict[str, object]", result)
        if isinstance(obj, dict):
            typed_obj: dict[str, object] = obj
            return typed_obj
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
    def make_factory(cls, target_class: type) -> FlextUtilities.SimpleFactory:
        """Create a simple factory for safe object construction."""
        return cls.SimpleFactory(target_class)

    @classmethod
    def make_builder(cls, target_class: type) -> FlextUtilities.SimpleBuilder:
        """Create a simple builder for fluent object construction."""
        return cls.SimpleBuilder(target_class)

    # =============================================================================
    # VALIDATION UTILITIES - Simple validation patterns
    # =============================================================================

    @classmethod
    def validate_and_create(
        cls,
        validator_func: Callable[[object], bool],
        value: object,
        error_message: str = "Validation failed",
    ) -> FlextResult[bool]:
        """Generic validation with FlextResult return."""
        try:
            if validator_func(value):
                return FlextResult[bool].ok(data=True)
            return FlextResult[bool].fail(error_message)
        except Exception as e:
            return FlextResult[bool].fail(f"Validation error: {e}")

    @classmethod
    def create_validator(
        cls, predicate: Callable[[object], bool]
    ) -> Callable[[object], FlextResult[bool]]:
        """Create a FlextResult-based validator from a simple predicate."""

        def validator(value: object) -> FlextResult[bool]:
            try:
                if predicate(value):
                    return FlextResult[bool].ok(data=True)
                return FlextResult[bool].fail("Validation failed")
            except Exception as e:
                return FlextResult[bool].fail(f"Validation error: {e}")

        return validator

    # =============================================================================
    # JSON PROCESSING UTILITIES - Consolidated from multiple modules
    # =============================================================================

    @classmethod
    def parse_json_safe(cls, json_text: str) -> FlextResult[dict[str, object]]:
        """Parse JSON string safely with FlextResult error handling."""
        try:
            data = json.loads(json_text)
            if not isinstance(data, dict):
                return FlextResult[dict[str, object]].fail("JSON must be object")
            # Ensure value type is object for typing consistency
            typed: dict[str, object] = cast("dict[str, object]", data)
            return FlextResult[dict[str, object]].ok(typed)
        except Exception as e:
            return FlextResult[dict[str, object]].fail(f"Invalid JSON: {e}")

    @classmethod
    def serialize_json_safe(cls, data: object) -> FlextResult[str]:
        """Serialize object to JSON safely with FlextResult error handling."""
        try:
            result = json.dumps(data, default=str, ensure_ascii=False)
            return FlextResult[str].ok(result)
        except Exception as e:
            return FlextResult[str].fail(f"JSON serialization failed: {e}")

    # =============================================================================
    # TYPE CONVERSION UTILITIES - Safe conversions with FlextResult
    # =============================================================================

    @classmethod
    def safe_cast_to_type[TCast](
        cls, value: object, target_type: type[TCast]
    ) -> FlextResult[TCast]:
        """Safely cast value to target type with FlextResult error handling."""
        try:
            if isinstance(value, target_type):
                return FlextResult[TCast].ok(value)
            # Try direct conversion - cast to Any for flexible type conversion
            converted = target_type(cast("Any", value))  # type: ignore[call-arg,explicit-any]
            return FlextResult[TCast].ok(converted)
        except Exception as e:
            return FlextResult[TCast].fail(
                f"Cannot cast {type(value).__name__} to {target_type.__name__}: {e}"
            )

    @classmethod
    def safe_int_convert(cls, value: object) -> FlextResult[int]:
        """Safely convert value to integer."""
        return cls.safe_cast_to_type(value, int)

    @classmethod
    def safe_str_convert(cls, value: object) -> FlextResult[str]:
        """Safely convert value to string."""
        return cls.safe_cast_to_type(value, str)

    @classmethod
    def safe_float_convert(cls, value: object) -> FlextResult[float]:
        """Safely convert value to float."""
        return cls.safe_cast_to_type(value, float)

    # Validation utilities without FlextResult - simple boolean checks
    @classmethod
    def validate_email_simple(cls, value: object) -> bool:
        """Simple email validation without FlextResult."""
        return cls.TypeGuards.is_email(value)

    @classmethod
    def validate_service_name_simple(cls, value: object) -> bool:
        """Simple service name validation without FlextResult."""
        if (
            not isinstance(value, str)
            or len(value.strip()) < cls.MIN_SERVICE_NAME_LENGTH
        ):
            return False
        return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", value.strip()))

    @classmethod
    def validate_port_range(cls, port: object) -> bool:
        """Simple port range validation without FlextResult."""
        if not isinstance(port, int):
            try:
                port = int(cast("Any", port))  # type: ignore[explicit-any]
            except (ValueError, TypeError):
                return False
        return cls.MIN_PORT <= port <= cls.MAX_PORT

    @classmethod
    def validate_percentage(cls, value: object) -> bool:
        """Simple percentage validation without FlextResult."""
        if not isinstance(value, (int, float)):
            try:
                value = float(cast("Any", value))  # type: ignore[explicit-any]
            except (ValueError, TypeError):
                return False
        return cls.MIN_PERCENTAGE <= value <= cls.MAX_PERCENTAGE

    @classmethod
    def validate_positive_number(cls, value: object) -> bool:
        """Simple positive number validation without FlextResult."""
        if not isinstance(value, (int, float)):
            try:
                value = float(cast("Any", value))  # type: ignore[explicit-any]
            except (ValueError, TypeError):
                return False
        return value > 0

    @classmethod
    def validate_in_range(cls, value: object, min_val: float, max_val: float) -> bool:
        """Simple range validation without FlextResult."""
        if not isinstance(value, (int, float)):
            try:
                value = float(cast("Any", value))  # type: ignore[explicit-any]
            except (ValueError, TypeError):
                return False
        return min_val <= value <= max_val

    # Type Adaptation utilities - simple serialization without FlextResult dependencies
    class SimpleTypeAdapters:
        """Simple type adaptation utilities without external dependencies."""

        @staticmethod
        def to_dict_safe(obj: object) -> dict[str, object]:
            """Safely convert object to dictionary."""
            if isinstance(obj, dict):
                return cast("dict[str, object]", obj)
            if hasattr(obj, "__dict__"):
                return dict(obj.__dict__)
            return {}

        @staticmethod
        def normalize_dict_values(data: dict[str, object]) -> dict[str, str]:
            """Normalize dictionary values to strings."""
            return {key: str(value) for key, value in data.items()}

        @staticmethod
        def validate_host_port_simple(host: object, port: object) -> bool:
            """Simple host/port validation without FlextResult."""
            if not isinstance(host, str) or not host.strip():
                return False
            return FlextUtilities.validate_port_range(port)

        @staticmethod
        def extract_entity_id(data: dict[str, object], key: str = "id") -> str | None:
            """Extract entity ID from dictionary, return None if not found."""
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            return None

        @staticmethod
        def create_simple_schema(fields: list[str]) -> dict[str, object]:
            """Create a simple schema definition."""
            return {
                "type": "object",
                "properties": {field: {"type": "string"} for field in fields},
                "required": fields,
            }

    # Simple schema processing utilities without FlextResult dependencies
    class SimpleSchemaProcessors:
        """Simple schema and entry processing utilities."""

        @staticmethod
        def extract_identifier_regex(content: str, pattern: str) -> str | None:
            """Extract identifier using regex pattern, return None if not found."""
            match = re.search(pattern, content)
            return match.group(1) if match else None

        @staticmethod
        def clean_content_prefix(content: str, prefix: str = "") -> str:
            """Clean content by removing prefix and whitespace."""
            if prefix:
                content = content.replace(f"{prefix}: ", "")
            return content.strip()

        @staticmethod
        def validate_required_attributes_simple(
            config: object, required: list[str]
        ) -> bool:
            """Simple validation of required attributes without FlextResult."""
            if isinstance(config, dict):
                typed_config = cast("dict[str, object]", config)
                return all(field in typed_config for field in required)
            # For non-dict objects, check attributes
            return all(hasattr(config, field) for field in required)

        @staticmethod
        def extract_config_dict_safe(config: object) -> dict[str, object]:
            """Extract configuration as dict safely."""
            if isinstance(config, dict):
                return cast("dict[str, object]", config)
            return getattr(config, "__dict__", {})

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
    def validate_signature(  # type: ignore[explicit-any]
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
        return FlextPredicates.is_not_none(value)

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
        logic across multiple FLEXT projects (algar-oud-mig, flext-tap-oracle-wms).

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
    # ADVANCED UTILITIES - From consolidated duplicate class
    # =============================================================================

    @classmethod
    def safe_unwrap_or(cls, result: FlextResult[T], default: T) -> T:
        """Safely unwrap FlextResult with default value.

        Modern pattern replacing .unwrap_or() calls for cleaner code.

        Args:
            result: FlextResult to unwrap
            default: Default value if result is failure

        Returns:
            Success value or default

        """
        return result.unwrap_or(default)

    @classmethod
    def validate_and_convert[TConvert](
        cls,
        value: object,
        converter: Callable[[object], TConvert],
        error_message: str = "Conversion failed",
    ) -> FlextResult[TConvert]:
        """Validate and convert value with error handling.

        Args:
            value: Value to convert
            converter: Conversion function
            error_message: Error message if conversion fails

        Returns:
            FlextResult containing converted value or error

        """
        try:
            converted = converter(value)
            return FlextResult[TConvert].ok(converted)
        except Exception as e:
            return FlextResult[TConvert].fail(f"{error_message}: {e}")

    @classmethod
    def safe_dict_get[TDict](
        cls,
        data: dict[str, object],
        key: str,
        expected_type: type[TDict],
        default: TDict,
    ) -> TDict:
        """Safely get and convert value from dict with type checking.

        Args:
            data: Dictionary to extract from
            key: Key to look up
            expected_type: Expected type for casting
            default: Default value if key missing or wrong type

        Returns:
            Typed value or default

        """
        value = data.get(key, default)
        if isinstance(value, expected_type):
            return value
        return default

    @classmethod
    def create_logger_with_context(
        cls, module_name: str, context: dict[str, object] | None = None
    ) -> object:
        """Create logger with contextual information.

        Args:
            module_name: Module name for logger
            context: Optional context dict to include in logs

        Returns:
            Configured logger instance

        """
        logger_instance = FlextLoggerFactory.get_logger(module_name)
        if context:
            # Add context to logger if supported
            # For now, just return the logger - context can be added to individual log calls
            pass
        return logger_instance

    @classmethod
    def benchmark_operation[TBench](
        cls,
        operation: Callable[[], TBench],
        description: str = "Operation",
        *,
        log_results: bool = True,
    ) -> tuple[TBench, float]:
        """Benchmark an operation and optionally log results.

        Args:
            operation: Function to benchmark
            description: Description for logging
            log_results: Whether to log timing results

        Returns:
            Tuple of (result, duration_seconds)

        """
        start_time = time.perf_counter()
        result = operation()
        end_time = time.perf_counter()
        duration = end_time - start_time

        if log_results:
            logger.info(f"{description} completed in {duration:.4f}s")

        return result, duration


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
    def get_performance_metrics() -> FlextTypes.Core.PerformanceMetrics:
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
            category, function_name, execution_time, success=_success
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
    def is_instance_of(obj: object, target_type: object) -> bool:
        """Check if an object is an instance of type."""
        # Handle non-type objects gracefully (generic type annotations, etc.)
        if not isinstance(target_type, type):
            return False
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


class FlextBaseFactory[T](Protocol):
    """Factory protocol providing creation foundation across ecosystem.

    SOLID compliance: Single responsibility for object creation patterns.
    """

    def create(self, **kwargs: object) -> FlextResult[object]:
        """Factory creation method - must be implemented by concrete factories."""
        ...


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


# MIGRATED TO LEGACY.PY: Standalone functions moved to centralized legacy compatibility
# Use FlextUtilities and FlextTypes hierarchy directly in new code:
# - FlextUtilities.safe_int_conversion() (instead of flext_safe_int_conversion)
# - FlextUtilities.Generators.generate_correlation_id() (instead of generate_correlation_id)
# - FlextUtilities.Generators.generate_id() (instead of generate_id)
# - FlextUtilities.Generators.generate_uuid() (instead of generate_uuid)
# - FlextUtilities.is_not_none_guard() (instead of is_not_none)
# - FlextUtilities.clear_performance_metrics() (instead of flext_clear_performance_metrics)
# - FlextUtilities.safe_int_conversion_with_default() (instead of safe_int_conversion_with_default)


# safe_call moved to result.py to avoid type conflicts with generic T version


def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text with suffix."""
    return FlextTextProcessor.truncate(text, max_length, suffix)


def flext_get_performance_metrics() -> FlextTypes.Core.PerformanceMetrics:
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


# MIGRATED TO LEGACY.PY: Additional standalone functions moved to legacy compatibility
# Use FlextUtilities hierarchy directly in new code:
# - FlextUtilities.Performance.track_performance() (instead of flext_track_performance)
# - FlextUtilities.Generators.generate_iso_timestamp() (instead of generate_iso_timestamp)
# - FlextUtilities directly (instead of Console alias)


# =============================================================================
# ADDITIONAL METHODS CONSOLIDATED - Into main FlextUtilities
# =============================================================================

# Methods from duplicate FlextUtilities have been added to the main class above
# This ensures single class with ALL functionality consolidated


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
    # ADVANCED VALIDATION UTILITIES - From validation.py patterns
    # =============================================================================

    class ValidationCore:
        """Core validation functions without FlextResult dependency."""

        @staticmethod
        def is_valid_email_format(email: str) -> bool:
            """Simple email validation without FlextResult."""
            return "@" in email and "." in email.rsplit("@", maxsplit=1)[-1]

        @staticmethod
        def is_valid_uuid_format(value: str) -> bool:
            """Simple UUID validation without FlextResult."""
            try:
                uuid.UUID(value)
                return True
            except (ValueError, TypeError):
                return False

        @staticmethod
        def is_valid_service_name(name: str) -> bool:
            """Simple service name validation without FlextResult."""
            if not name.strip():
                return False
            return (
                len(name) >= FlextUtilities.MIN_SERVICE_NAME_LENGTH
                and name.replace("_", "").replace("-", "").replace(".", "").isalnum()
            )

        @staticmethod
        def is_valid_port_range(port: int) -> bool:
            """Simple port validation without FlextResult."""
            return FlextUtilities.MIN_PORT <= port <= FlextUtilities.MAX_PORT

        @staticmethod
        def is_valid_percentage(value: float) -> bool:
            """Simple percentage validation without FlextResult."""
            return (
                FlextUtilities.MIN_PERCENTAGE <= value <= FlextUtilities.MAX_PERCENTAGE
            )

        @staticmethod
        def is_non_empty_string_type(value: object) -> bool:
            """Type guard for non-empty strings without FlextResult."""
            return isinstance(value, str) and len(value.strip()) > 0

        @staticmethod
        def extract_domain_from_email(email: str) -> str | None:
            """Extract domain from email without FlextResult."""
            if "@" not in email:
                return None
            return email.rsplit("@", maxsplit=1)[-1] or None

    # =============================================================================
    # SIMPLE TYPE PROCESSING - From type_adapters.py patterns
    # =============================================================================

    class TypeProcessing:
        """Simple type processing without FlextResult dependency."""

        @staticmethod
        def safe_json_loads(json_str: str) -> dict[str, object] | None:
            """Safe JSON parsing without FlextResult."""
            try:
                result = json.loads(json_str)
                if isinstance(result, dict):
                    typed_result: dict[str, object] = result
                    return typed_result
                return None
            except (json.JSONDecodeError, TypeError):
                return None

        @staticmethod
        def safe_json_dumps(obj: object) -> str | None:
            """Safe JSON serialization without FlextResult."""
            try:
                return json.dumps(obj)
            except (TypeError, ValueError):
                return None

        @staticmethod
        def validate_and_cast_int(value: object) -> int | None:
            """Safe int validation and casting without FlextResult."""
            try:
                if isinstance(value, int):
                    return value
                if isinstance(value, str) and value.strip().isdigit():
                    return int(value)
                if isinstance(value, float) and value.is_integer():
                    return int(value)
                return None
            except (ValueError, TypeError):
                return None

        @staticmethod
        def validate_and_cast_float(value: object) -> float | None:
            """Safe float validation and casting without FlextResult."""
            try:
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    return float(value.strip())
                return None
            except (ValueError, TypeError):
                return None

        @staticmethod
        def extract_type_name(obj: object) -> str:
            """Extract type name for debugging without FlextResult."""
            return type(obj).__name__

        @staticmethod
        def is_dict_like(obj: object) -> bool:
            """Check if object is dict-like without FlextResult."""
            return (
                hasattr(obj, "keys")
                and hasattr(obj, "values")
                and hasattr(obj, "items")
            )

    # =============================================================================
    # SCHEMA PROCESSING UTILITIES - From schema_processing.py patterns
    # =============================================================================

    class SchemaCore:
        """Simple schema processing without FlextResult dependency."""

        @staticmethod
        def extract_identifier_simple(content: str, pattern: str) -> str | None:
            """Extract identifier using regex pattern without FlextResult."""
            match = re.search(pattern, content)
            return match.group(1) if match and match.groups() else None

        @staticmethod
        def clean_content_simple(content: str, prefix: str = "") -> str:
            """Clean content by removing prefix without FlextResult."""
            if prefix and content.startswith(f"{prefix}: "):
                return content.replace(f"{prefix}: ", "").strip()
            return content.strip()

        @staticmethod
        def is_valid_content_line(line: str) -> bool:
            """Check if content line is valid without FlextResult."""
            return len(line.strip()) > 0

        @staticmethod
        def extract_field_value(
            data: dict[str, object], field: str, default: object = None
        ) -> object:
            """Extract field value with default without FlextResult."""
            return data.get(field, default)

        @staticmethod
        def validate_required_fields(
            data: dict[str, object], required: list[str]
        ) -> list[str]:
            """Validate required fields and return missing ones without FlextResult."""
            return [
                field for field in required if field not in data or data[field] is None
            ]

        @staticmethod
        def create_error_summary(errors: list[str], max_errors: int = 3) -> str:
            """Create error summary without FlextResult."""
            if not errors:
                return "No errors"
            if len(errors) <= max_errors:
                return "; ".join(errors)
            return (
                f"{'; '.join(errors[:max_errors])} and {len(errors) - max_errors} more"
            )

    # =============================================================================
    # ENHANCED TYPE ADAPTER UTILITIES
    # =============================================================================

    class SimpleTypeAdapterExtended:
        """Extended type adapter utilities without FlextResult."""

        @staticmethod
        def adapt_to_string_safe(value: object) -> str:
            """Safely adapt value to string."""
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            try:
                return str(value)
            except Exception:
                return f"<{type(value).__name__} object>"

        @staticmethod
        def adapt_to_dict_safe(obj: object) -> dict[str, object]:
            """Safely adapt object to dict."""
            if isinstance(obj, dict):
                typed_dict: dict[object, object] = obj
                return {str(k): v for k, v in typed_dict.items()}
            if hasattr(obj, "__dict__"):
                obj_dict: dict[str, object] = obj.__dict__  # type: ignore[attr-defined]
                return {str(k): v for k, v in obj_dict.items() if not k.startswith("_")}
            return {"value": obj, "type": type(obj).__name__}

        @staticmethod
        def adapt_list_items(
            items: list[object], adapter_func: Callable[[object], object]
        ) -> list[object]:
            """Safely adapt list items using adapter function."""
            result: list[object] = []
            for item in items:
                try:
                    adapted = adapter_func(item)
                    result.append(adapted)
                except Exception:
                    result.append(f"<adaptation_error: {type(item).__name__}>")
            return result

        @staticmethod
        def create_metadata_dict(obj: object) -> dict[str, str]:
            """Create metadata dict for object."""
            return {
                "type": type(obj).__name__,
                "module": getattr(type(obj), "__module__", "unknown"),
                "str_repr": str(obj)[:100],
                "has_dict": str(hasattr(obj, "__dict__")),
            }

    # =============================================================================
    # DELEGATION TO NESTED CLASSES - Extended validation and processing
    # =============================================================================

    @classmethod
    def is_valid_email_format(cls, email: str) -> bool:
        """Check if email format is valid - delegates to ValidationCore."""
        return cls.ValidationCore.is_valid_email_format(email)

    @classmethod
    def is_valid_uuid_format(cls, value: str) -> bool:
        """Check if UUID format is valid - delegates to ValidationCore."""
        return cls.ValidationCore.is_valid_uuid_format(value)

    @classmethod
    def is_valid_service_name(cls, name: str) -> bool:
        """Check if service name is valid - delegates to ValidationCore."""
        return cls.ValidationCore.is_valid_service_name(name)

    @classmethod
    def is_valid_port_range(cls, port: int) -> bool:
        """Check if port is in valid range - delegates to ValidationCore."""
        return cls.ValidationCore.is_valid_port_range(port)

    @classmethod
    def is_valid_percentage(cls, value: float) -> bool:
        """Check if percentage is valid - delegates to ValidationCore."""
        return cls.ValidationCore.is_valid_percentage(value)

    @classmethod
    def extract_domain_from_email(cls, email: str) -> str | None:
        """Extract domain from email - delegates to ValidationCore."""
        return cls.ValidationCore.extract_domain_from_email(email)

    @classmethod
    def safe_json_loads_simple(cls, json_str: str) -> dict[str, object] | None:
        """Safe JSON parsing - delegates to TypeProcessing."""
        return cls.TypeProcessing.safe_json_loads(json_str)

    @classmethod
    def safe_json_dumps_simple(cls, obj: object) -> str | None:
        """Safe JSON serialization - delegates to TypeProcessing."""
        return cls.TypeProcessing.safe_json_dumps(obj)

    @classmethod
    def validate_and_cast_int(cls, value: object) -> int | None:
        """Validate and cast to int - delegates to TypeProcessing."""
        return cls.TypeProcessing.validate_and_cast_int(value)

    @classmethod
    def validate_and_cast_float(cls, value: object) -> float | None:
        """Validate and cast to float - delegates to TypeProcessing."""
        return cls.TypeProcessing.validate_and_cast_float(value)

    @classmethod
    def extract_identifier_simple(cls, content: str, pattern: str) -> str | None:
        """Extract identifier using pattern - delegates to SchemaCore."""
        return cls.SchemaCore.extract_identifier_simple(content, pattern)

    @classmethod
    def clean_content_simple(cls, content: str, prefix: str = "") -> str:
        """Clean content by removing prefix - delegates to SchemaCore."""
        return cls.SchemaCore.clean_content_simple(content, prefix)

    @classmethod
    def validate_required_fields(
        cls, data: dict[str, object], required: list[str]
    ) -> list[str]:
        """Validate required fields - delegates to SchemaCore."""
        return cls.SchemaCore.validate_required_fields(data, required)

    @classmethod
    def create_error_summary(cls, errors: list[str], max_errors: int = 3) -> str:
        """Create error summary - delegates to SchemaCore."""
        return cls.SchemaCore.create_error_summary(errors, max_errors)

    @classmethod
    def adapt_to_string_safe(cls, value: object) -> str:
        """Safely adapt to string - delegates to SimpleTypeAdapterExtended."""
        return cls.SimpleTypeAdapterExtended.adapt_to_string_safe(value)

    @classmethod
    def adapt_to_dict_safe(cls, obj: object) -> dict[str, object]:
        """Safely adapt to dict - delegates to SimpleTypeAdapterExtended."""
        return cls.SimpleTypeAdapterExtended.adapt_to_dict_safe(obj)


# =============================================================================
# EXPORTS - Main architectural classes and utilities
# =============================================================================

# =============================================================================
# TIER 1 MODULE PATTERN - Single Export Only
# =============================================================================

__all__: list[str] = [
    "FlextUtilities",  # ONLY main class exported
]

# =============================================================================
# BACKWARD COMPATIBILITY ALIASES - Preserve existing APIs
# =============================================================================

# Main class is now FlextUtilities directly (no alias needed)
# All utility functionality consolidated in FlextUtilities class above
