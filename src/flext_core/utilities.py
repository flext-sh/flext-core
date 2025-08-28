"""Comprehensive utility functions for the FLEXT ecosystem.

Provides the FlextUtilities class with organized utility functions including
ID generation, text processing, validation, performance monitoring, JSON handling,
and decorator patterns. All utilities follow SOLID principles with nested class
organization for better code organization and discoverability.

Main Classes:
    FlextUtilities: Main container with nested utility classes
    FlextUtilities.Generators: ID and timestamp generation
    FlextUtilities.TextProcessor: Text processing and formatting
    FlextUtilities.Validators: Data validation utilities
    FlextUtilities.Performance: Performance monitoring and timing
    FlextUtilities.JSON: JSON processing and serialization
    FlextUtilities.Decorators: Utility decorators and function wrappers

Key Features:
    - Hierarchical organization with nested classes
    - Performance monitoring with metrics collection
    - Type-safe validation with FlextResult patterns
    - Comprehensive text processing utilities
    - ID generation with consistent prefixing
    - JSON handling with error recovery
    - Timing and performance decorators

Example:
    Basic utility usage::

        # Generate various types of IDs
        uuid = FlextUtilities.Generators.generate_uuid()
        entity_id = FlextUtilities.Generators.generate_entity_id()
        correlation_id = FlextUtilities.Generators.generate_correlation_id()

        # Text processing
        truncated = FlextUtilities.TextProcessor.truncate(long_text, 50)
        safe_str = FlextUtilities.TextProcessor.safe_string(any_value)

        # Validation
        email_result = FlextUtilities.Validators.validate_email(email)
        if email_result.success:
            process_valid_email(email)

Note:
    All utilities return FlextResult types for consistent error handling
    and integrate with the FLEXT logging and constants systems.

"""

from __future__ import annotations

import functools
import json
import re
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import cast

from flext_core.constants import FlextConstants
from flext_core.loggings import FlextLogger
from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult
from flext_core.typings import P, R, T

logger = FlextLogger(__name__)

# Performance metrics storage.
PERFORMANCE_METRICS: dict[str, dict[str, object]] = {}


class FlextUtilities:
    """Comprehensive utility functions organized by functional domain.

    This class serves as the central container for all FLEXT utility functions,
    organized into nested classes by functional area. Provides common operations
    for ID generation, text processing, validation, performance monitoring,
    JSON handling, and more.

    Architecture:
        The utilities are organized into nested classes following single
        responsibility principle:
        - Generators: ID and timestamp generation utilities
        - TextProcessor: Text manipulation and formatting
        - Validators: Data validation with FlextResult patterns
        - Performance: Timing and performance monitoring
        - JSON: JSON processing and serialization
        - Decorators: Function decorators and wrappers

    Examples:
        ID generation utilities::

            uuid = FlextUtilities.Generators.generate_uuid()
            entity_id = FlextUtilities.Generators.generate_entity_id()
            correlation_id = FlextUtilities.Generators.generate_correlation_id()

        Text processing utilities::

            truncated = FlextUtilities.TextProcessor.truncate(text, 100)
            cleaned = FlextUtilities.TextProcessor.safe_string(value)

        Validation utilities::

            email_result = FlextUtilities.Validators.validate_email(email)
            if email_result.success:
                process_valid_email(email)

    Note:
        All utilities follow FLEXT patterns including FlextResult for error
        handling, FlextConstants for configuration, and structured logging.

    """

    # ==========================================================================
    # CLASS CONSTANTS
    # ==========================================================================

    MIN_PORT: int = FlextConstants.Network.MIN_PORT
    MAX_PORT: int = FlextConstants.Network.MAX_PORT

    # ==========================================================================
    # NESTED CLASSES FOR ORGANIZATION
    # ==========================================================================

    class Generators:
        """ID and timestamp generation utilities with consistent prefixing.

        Provides various ID generation methods with semantic prefixes for
        different use cases. All IDs use UUID4 for uniqueness with shortened
        hex representations for readability.

        Examples:
            Generate different types of IDs::

                uuid = Generators.generate_uuid()  # Full UUID4
                entity_id = Generators.generate_entity_id()  # entity_xxxx
                correlation_id = Generators.generate_correlation_id()  # corr_xxxx
                session_id = Generators.generate_session_id()  # sess_xxxx

        """

        @staticmethod
        def generate_uuid() -> str:
            """Generate standard UUID4."""
            return str(uuid.uuid4())

        @staticmethod
        def generate_id() -> str:
            """Generate unique ID with flext_ prefix."""
            return f"flext_{uuid.uuid4().hex[:8]}"

        @staticmethod
        def generate_entity_id() -> str:
            """Generate entity ID with entity_ prefix."""
            return f"entity_{uuid.uuid4().hex[:12]}"

        @staticmethod
        def generate_correlation_id() -> str:
            """Generate correlation ID for request tracing."""
            return f"corr_{uuid.uuid4().hex[:16]}"

        @staticmethod
        def generate_iso_timestamp() -> str:
            """Generate ISO timestamp in UTC."""
            return datetime.now(UTC).isoformat()

        @staticmethod
        def generate_session_id() -> str:
            """Generate session ID for web sessions."""
            return f"sess_{uuid.uuid4().hex[:16]}"

        @staticmethod
        def generate_request_id() -> str:
            """Generate request ID for web requests."""
            return f"req_{uuid.uuid4().hex[:12]}"

    class TextProcessor:
        """Text processing, formatting, and safe conversion utilities.

        Provides safe text processing functions with proper error handling
        and consistent formatting. Handles edge cases and provides fallbacks
        for robust text manipulation.

        Examples:
            Text processing operations::

                truncated = TextProcessor.truncate(long_text, 50, '...')
                safe_str = TextProcessor.safe_string(any_object, 'default')

        """

        @staticmethod
        def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
            """Truncate text to maximum length with suffix."""
            if len(text) <= max_length:
                return text
            if max_length <= len(suffix):
                return text[:max_length]
            return text[: max_length - len(suffix)] + suffix

        @staticmethod
        def safe_string(value: object, default: str = "") -> str:
            """Convert any value to string safely."""
            try:
                return str(value) if value is not None else default
            except Exception:
                return default

        @staticmethod
        def clean_text(text: str) -> str:
            """Clean text by removing extra whitespace and controlling characters."""
            if not text:
                return ""
            # Remove control characters except tabs and newlines
            cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
            # Normalize whitespace
            cleaned = re.sub(r"\s+", " ", cleaned)
            return cleaned.strip()

        @staticmethod
        def slugify(text: str) -> str:
            """Convert text to URL-safe slug format."""
            if not text:
                return ""
            # Convert to lowercase and remove extra whitespace
            slug = text.lower().strip()
            # Replace non-alphanumeric characters with hyphens
            slug = re.sub(r"[^a-z0-9]+", "-", slug)
            # Remove leading/trailing hyphens
            return slug.strip("-")

        @staticmethod
        def mask_sensitive(
            text: str,
            mask_char: str = "*",
            visible_chars: int = 4,
            show_first: int | None = None,
            show_last: int | None = None,
        ) -> str:
            """Mask sensitive information with flexible visibility options.

            Args:
                text: Text to mask
                mask_char: Character to use for masking
                visible_chars: Number of characters to show at the end (legacy parameter)
                show_first: Number of characters to show at the start
                show_last: Number of characters to show at the end

            """
            if not text:
                return mask_char * 8  # Default masked length

            # Handle new API with show_first and show_last
            if show_first is not None or show_last is not None:
                first_chars = show_first or 0
                last_chars = show_last or 0

                if len(text) <= (first_chars + last_chars):
                    return mask_char * len(text)

                first_part = text[:first_chars] if first_chars > 0 else ""
                last_part = text[-last_chars:] if last_chars > 0 else ""
                middle_length = len(text) - first_chars - last_chars
                masked_part = mask_char * middle_length

                return first_part + masked_part + last_part

            # Legacy API - show only last characters
            if len(text) <= visible_chars:
                return mask_char * len(text)

            visible_part = text[-visible_chars:]
            masked_part = mask_char * (len(text) - visible_chars)
            return masked_part + visible_part

    class TimeUtils:
        """Time and duration utilities with formatting and conversion.

        Provides utilities for time formatting, duration calculations, and
        timestamp operations with proper timezone handling.

        Examples:
            Time operations::

                formatted = TimeUtils.format_duration(123.45)  # "2.1m"
                utc_now = TimeUtils.get_timestamp_utc()

        """

        @staticmethod
        def format_duration(seconds: float) -> str:
            """Format duration in human-readable format."""
            if seconds < 1:
                return f"{seconds * 1000:.1f}ms"
            if seconds < FlextConstants.Utilities.SECONDS_PER_MINUTE:
                return f"{seconds:.1f}s"
            if seconds < FlextConstants.Utilities.SECONDS_PER_HOUR:
                return f"{seconds / FlextConstants.Utilities.SECONDS_PER_MINUTE:.1f}m"
            return f"{seconds / FlextConstants.Utilities.SECONDS_PER_HOUR:.1f}h"

        @staticmethod
        def get_timestamp_utc() -> datetime:
            """Get current UTC timestamp."""
            return datetime.now(UTC)

    class Performance:
        """Performance tracking and monitoring utilities.

        Provides decorators and utilities for tracking function performance,
        recording metrics, and monitoring operation success rates.

        Examples:
            Performance tracking::

                @Performance.track_performance("user_creation")
                def create_user(data):
                    return process_user_data(data)

                metrics = Performance.get_metrics("user_creation")

        """

        @staticmethod
        def track_performance(
            operation_name: str,
        ) -> Callable[[Callable[P, R]], Callable[P, R]]:
            """Decorator for tracking performance metrics."""

            def decorator(func: Callable[P, R]) -> Callable[P, R]:
                @functools.wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                    start_time = time.perf_counter()
                    try:
                        result = func(*args, **kwargs)
                        duration = time.perf_counter() - start_time
                        FlextUtilities.Performance.record_metric(
                            operation_name, duration, success=True
                        )
                        return result
                    except Exception as e:
                        duration = time.perf_counter() - start_time
                        FlextUtilities.Performance.record_metric(
                            operation_name, duration, success=False, error=str(e)
                        )
                        raise

                return wrapper

            return decorator

        @staticmethod
        def record_metric(
            operation: str,
            duration: float,
            *,
            success: bool = True,
            error: str | None = None,
        ) -> None:
            """Record performance metric."""
            if operation not in PERFORMANCE_METRICS:
                PERFORMANCE_METRICS[operation] = {
                    "total_calls": 0,
                    "total_duration": 0.0,
                    "avg_duration": 0.0,
                    "success_count": 0,
                    "error_count": 0,
                }

            metrics = PERFORMANCE_METRICS[operation]
            total_calls = cast("int", metrics["total_calls"]) + 1
            total_duration = cast("float", metrics["total_duration"]) + duration

            metrics["total_calls"] = total_calls
            metrics["total_duration"] = total_duration
            metrics["avg_duration"] = total_duration / total_calls

            if success:
                metrics["success_count"] = cast("int", metrics["success_count"]) + 1
            else:
                metrics["error_count"] = cast("int", metrics["error_count"]) + 1
                if error:
                    metrics["last_error"] = error

        @staticmethod
        def get_metrics(operation: str | None = None) -> dict[str, object]:
            """Get performance metrics."""
            if operation:
                return PERFORMANCE_METRICS.get(operation, {})
            return dict(PERFORMANCE_METRICS)

    class Conversions:
        """Safe type conversion utilities with fallback handling.

        Provides robust type conversion functions that handle edge cases
        and provide sensible defaults when conversion fails.

        Examples:
            Safe conversions::

                num = Conversions.safe_int("123", 0)  # 123
                flag = Conversions.safe_bool("true")   # True
                val = Conversions.safe_float(None, 0.0)  # 0.0

        """

        @staticmethod
        def safe_int(value: object, default: int = 0) -> int:
            """Convert value to int safely."""
            if value is None:
                return default
            try:
                # Type narrowing: handle string and numeric types
                if isinstance(value, (str, int, float)):
                    return int(value)
                # For other objects, try str conversion first
                return int(str(value))
            except (ValueError, TypeError, OverflowError):
                return default

        @staticmethod
        def safe_float(value: object, default: float = 0.0) -> float:
            """Convert value to float safely."""
            if value is None:
                return default
            try:
                # Type narrowing: handle string and numeric types
                if isinstance(value, (str, int, float)):
                    return float(value)
                # For other objects, try str conversion first
                return float(str(value))
            except (ValueError, TypeError, OverflowError):
                return default

        @staticmethod
        def safe_bool(value: object, *, default: bool = False) -> bool:  # noqa: FBT001
            """Convert value to bool safely."""
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in {"true", "1", "yes", "on"}
            try:
                return bool(value)
            except (ValueError, TypeError):
                return default

    class TypeGuards:
        """Type checking and validation guard utilities.

        Provides type guard functions for runtime type checking and
        validation with proper type narrowing support.

        Examples:
            Type validation::

                if TypeGuards.is_string_non_empty(value):
                    process_string(value)  # value is now str

                if TypeGuards.is_dict_non_empty(data):
                    process_dict(data)  # data is now dict

        """

        @staticmethod
        def is_string_non_empty(value: object) -> bool:
            """Check if value is non-empty string."""
            return isinstance(value, str) and len(value) > 0

        @staticmethod
        def is_dict_non_empty(value: object) -> bool:
            """Check if value is non-empty dict."""
            if isinstance(value, dict):
                sized_dict = cast("FlextProtocols.Foundation.SizedDict", value)
                return len(sized_dict) > 0
            return False

        @staticmethod
        def is_list_non_empty(value: object) -> bool:
            """Check if value is non-empty list."""
            if isinstance(value, list):
                sized_list = cast("FlextProtocols.Foundation.SizedList", value)
                return len(sized_list) > 0
            return False

        @staticmethod
        def has_attribute(obj: object, attr: str) -> bool:
            """Check if object has attribute."""
            return hasattr(obj, attr)

        @staticmethod
        def is_not_none(value: object) -> bool:
            """Check if value is not None."""
            return value is not None

    class Formatters:
        """Data formatting utilities for human-readable output.

        Provides formatting functions for common data types including
        byte sizes, percentages, and other human-readable formats.

        Examples:
            Data formatting::

                size = Formatters.format_bytes(1024)  # "1.0 KB"
                percent = Formatters.format_percentage(0.85)  # "85.0%"

        """

        @staticmethod
        def format_bytes(bytes_count: int) -> str:
            """Format byte count in human-readable format."""
            kb = FlextConstants.Utilities.BYTES_PER_KB
            if bytes_count < kb:
                return f"{bytes_count} B"
            if bytes_count < kb**2:
                return f"{bytes_count / kb:.1f} KB"
            if bytes_count < kb**3:
                return f"{bytes_count / (kb**2):.1f} MB"
            return f"{bytes_count / (kb**3):.1f} GB"

        @staticmethod
        def format_percentage(value: float, precision: int = 1) -> str:
            """Format value as percentage."""
            return f"{value * 100:.{precision}f}%"

    class ProcessingUtils:
        """Data processing utilities for JSON, models, and structured data.

        Provides safe processing functions for JSON parsing, model extraction,
        and data validation with FlextResult error handling.

        Examples:
            Data processing::

                data = ProcessingUtils.safe_json_parse(json_str, {})
                json_str = ProcessingUtils.safe_json_stringify(obj)
                result = ProcessingUtils.parse_json_to_model(json_str, MyModel)

        """

        @staticmethod
        def safe_json_parse(
            json_str: str, default: dict[str, object] | None = None
        ) -> dict[str, object]:
            """Safely parse JSON string."""
            try:
                result: object = json.loads(json_str)
                if isinstance(result, dict):
                    return cast("dict[str, object]", result)
                return default or {}
            except (json.JSONDecodeError, TypeError):
                return default or {}

        @staticmethod
        def safe_json_stringify(obj: object, default: str = "{}") -> str:
            """Safely stringify object to JSON."""
            try:
                return json.dumps(obj, default=str, ensure_ascii=False)
            except (TypeError, ValueError):
                return default

        @staticmethod
        def extract_model_data(obj: object) -> dict[str, object]:
            """Extract data from Pydantic model or dict."""
            if hasattr(obj, "model_dump"):
                model_obj = cast("FlextProtocols.Foundation.HasModelDump", obj)
                return model_obj.model_dump()
            if hasattr(obj, "dict"):
                dict_obj = cast("FlextProtocols.Foundation.HasDict", obj)
                return dict_obj.dict()
            if isinstance(obj, dict):
                return cast("dict[str, object]", obj)
            return {}

        @staticmethod
        def parse_json_to_model(json_text: str, model_class: type[T]) -> FlextResult[T]:
            """Parse JSON and validate using Pydantic model."""
            try:
                data: object = json.loads(json_text)
                # Check if the class supports Pydantic v2 model_validate
                if hasattr(model_class, "model_validate"):
                    # Cast to HasModelValidate protocol for type safety
                    pydantic_model = cast(
                        "type[FlextProtocols.Foundation.HasModelValidate]",  # type: ignore[name-defined]
                        model_class,  # pyright: ignore[reportGeneralTypeIssues]
                    )
                    model_obj = pydantic_model.model_validate(data)
                    model = cast("T", model_obj)
                else:
                    # Fallback for regular classes - use DataConstructor protocol
                    constructor = cast(
                        "FlextProtocols.Foundation.DataConstructor",
                        model_class,  # type: ignore[name-defined]
                    )
                    model_obj = constructor(data)
                    model = cast("T", model_obj)
                return FlextResult[T].ok(model)
            except json.JSONDecodeError as e:
                return FlextResult[T].fail(f"Invalid JSON: {e}")
            except Exception as e:
                return FlextResult[T].fail(f"Model validation failed: {e}")

    class ResultUtils:
        """FlextResult processing and composition utilities.

        Provides utilities for working with FlextResult types including
        result chaining, batch processing, and error collection.

        Examples:
            Result operations::

                combined = ResultUtils.chain_results(result1, result2)
                successes, errors = ResultUtils.batch_process(items, processor)

        """

        @staticmethod
        def chain_results(*results: FlextResult[T]) -> FlextResult[list[T]]:
            """Chain multiple FlextResults into a single result."""
            values: list[T] = []
            for result in results:
                if result.is_failure:
                    return FlextResult[list[T]].fail(
                        result.error or "Chain operation failed"
                    )
                values.append(result.value)
            return FlextResult[list[T]].ok(values)

        @staticmethod
        def batch_process[TInput, TOutput](
            items: list[TInput],
            processor: Callable[[TInput], FlextResult[TOutput]],
        ) -> tuple[list[TOutput], list[str]]:
            """Process list of items, collecting successes and errors."""
            successes: list[TOutput] = []
            errors: list[str] = []

            for item in items:
                result = processor(item)
                if result.is_success:
                    successes.append(result.value)
                else:
                    errors.append(result.error or "Unknown error")

            return successes, errors

    # ==========================================================================
    # MAIN CLASS METHODS - Delegate to nested classes
    # ==========================================================================

    @classmethod
    def generate_uuid(cls) -> str:
        """Generate UUID (delegates to Generators)."""
        return cls.Generators.generate_uuid()

    @classmethod
    def generate_id(cls) -> str:
        """Generate ID (delegates to Generators)."""
        return cls.Generators.generate_id()

    @classmethod
    def generate_entity_id(cls) -> str:
        """Generate entity ID (delegates to Generators)."""
        return cls.Generators.generate_entity_id()

    @classmethod
    def generate_correlation_id(cls) -> str:
        """Generate correlation ID (delegates to Generators)."""
        return cls.Generators.generate_correlation_id()

    @classmethod
    def truncate(cls, text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text (delegates to TextProcessor)."""
        return cls.TextProcessor.truncate(text, max_length, suffix)

    @classmethod
    def format_duration(cls, seconds: float) -> str:
        """Format duration (delegates to TimeUtils)."""
        return cls.TimeUtils.format_duration(seconds)

    @classmethod
    def track_performance(
        cls, operation_name: str
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Track performance (delegates to Performance)."""
        return cls.Performance.track_performance(operation_name)

    @classmethod
    def safe_json_parse(
        cls, json_str: str, default: dict[str, object] | None = None
    ) -> dict[str, object]:
        """Parse JSON safely (delegates to ProcessingUtils)."""
        return cls.ProcessingUtils.safe_json_parse(json_str, default)

    @classmethod
    def safe_json_stringify(cls, obj: object, default: str = "{}") -> str:
        """Stringify object to JSON safely (delegates to ProcessingUtils)."""
        return cls.ProcessingUtils.safe_json_stringify(obj, default)

    @classmethod
    def parse_json_to_model(
        cls, json_text: str, model_class: type[T]
    ) -> FlextResult[T]:
        """Parse JSON to model (delegates to ProcessingUtils)."""
        return cls.ProcessingUtils.parse_json_to_model(json_text, model_class)

    @classmethod
    def safe_int(cls, value: object, default: int = 0) -> int:
        """Convert to int safely (delegates to Conversions)."""
        return cls.Conversions.safe_int(value, default)

    @classmethod
    def batch_process[TInput, TOutput](
        cls,
        items: list[TInput],
        processor: Callable[[TInput], FlextResult[TOutput]],
    ) -> tuple[list[TOutput], list[str]]:
        """Process batch (delegates to ResultUtils)."""
        return cls.ResultUtils.batch_process(items, processor)

    # Additional methods needed by legacy compatibility layer
    @classmethod
    def safe_int_conversion(
        cls, value: object, default: int | None = None
    ) -> int | None:
        """Convert value to int safely with optional default."""
        if value is None:
            return default
        try:
            # Type narrowing: handle string and numeric types
            if isinstance(value, (str, int, float)):
                return int(value)
            # For other objects, try str conversion first
            return int(str(value))
        except (ValueError, TypeError, OverflowError):
            return default

    @classmethod
    def safe_int_conversion_with_default(cls, value: object, default: int) -> int:
        """Convert value to int safely with guaranteed default."""
        return cls.safe_int_conversion(value, default) or default

    @classmethod
    def safe_bool_conversion(cls, value: object, *, default: bool = False) -> bool:  # noqa: FBT001
        """Convert value to bool safely (delegates to Conversions)."""
        return cls.Conversions.safe_bool(value, default=default)

    @classmethod
    def generate_iso_timestamp(cls) -> str:
        """Generate ISO timestamp (delegates to Generators)."""
        return cls.Generators.generate_iso_timestamp()

    @classmethod
    def get_performance_metrics(cls) -> dict[str, object]:
        """Get all performance metrics (delegates to Performance)."""
        return cls.Performance.get_metrics()

    @classmethod
    def record_performance(
        cls,
        operation: str,
        duration: float,
        *,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Record performance metric (delegates to Performance)."""
        return cls.Performance.record_metric(
            operation, duration, success=success, error=error
        )

    # Additional delegator methods needed by flext-cli
    @classmethod
    def is_non_empty_string(cls, value: object) -> bool:
        """Check if value is non-empty string (delegates to TypeGuards)."""
        return cls.TypeGuards.is_string_non_empty(value)

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Clean text (delegates to TextProcessor)."""
        return cls.TextProcessor.clean_text(text)

    @classmethod
    def generate_timestamp(cls) -> str:
        """Generate timestamp (delegates to Generators)."""
        return cls.Generators.generate_iso_timestamp()

    @classmethod
    def parse_iso_timestamp(cls, timestamp_str: str) -> datetime:
        """Parse ISO timestamp string to datetime."""
        try:
            return datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            # Return current UTC time as fallback
            return datetime.now(UTC)

    @classmethod
    def get_elapsed_time(cls, start_time: datetime) -> float:
        """Get elapsed time from start_time to now in seconds."""
        current_time = datetime.now(UTC)
        # Ensure both timestamps are timezone-aware
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=UTC)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=UTC)

        delta = current_time - start_time
        return delta.total_seconds()


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__: list[str] = [
    "FlextUtilities",  # ONLY main class exported
]
