"""FLEXT Utilities - Consolidated utility functions following architectural patterns.

SINGLE CONSOLIDATED MODULE following FLEXT architectural patterns.
All utility functionality consolidated into FlextUtilities main class.
"""

from __future__ import annotations

import functools
import json
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

# Global storage for performance metrics
PERFORMANCE_METRICS: dict[str, dict[str, object]] = {}


class FlextUtilities:
    """SINGLE CONSOLIDATED CLASS for all utility functionality.

    Following FLEXT architectural patterns - consolidates ALL utility functionality
    into one main class with nested organization.

    Architecture:
        - Single main class FlextUtilities
        - Nested classes for logical organization
        - No standalone helper classes
        - Direct method access patterns
        - Railway-oriented programming with FlextResult
    """

    # ==========================================================================
    # NESTED CLASSES FOR ORGANIZATION
    # ==========================================================================

    class Generators:
        """ID and timestamp generation utilities."""

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

    class TextProcessor:
        """Text processing and formatting utilities."""

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

    class TimeUtils:
        """Time and duration utilities."""

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
        """Performance tracking utilities."""

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
            metrics["total_calls"] = cast("int", metrics["total_calls"]) + 1
            metrics["total_duration"] = (
                cast("float", metrics["total_duration"]) + duration
            )
            metrics["avg_duration"] = cast("float", metrics["total_duration"]) / cast(
                "int", metrics["total_calls"]
            )

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
        """Type conversion utilities."""

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
                return value.lower() in ("true", "1", "yes", "on")
            try:
                return bool(value)
            except (ValueError, TypeError):
                return default

    class TypeGuards:
        """Type guard utilities."""

        @staticmethod
        def is_string_non_empty(value: object) -> bool:
            """Check if value is non-empty string."""
            return isinstance(value, str) and len(value) > 0

        @staticmethod
        def is_dict_non_empty(value: object) -> bool:
            """Check if value is non-empty dict."""
            return isinstance(value, dict) and len(value) > 0

        @staticmethod
        def is_list_non_empty(value: object) -> bool:
            """Check if value is non-empty list."""
            return isinstance(value, list) and len(value) > 0

        @staticmethod
        def has_attribute(obj: object, attr: str) -> bool:
            """Check if object has attribute."""
            return hasattr(obj, attr)

        @staticmethod
        def is_not_none(value: object) -> bool:
            """Check if value is not None."""
            return value is not None

    class Formatters:
        """Formatting utilities."""

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
        """Processing utilities for JSON and data."""

        @staticmethod
        def safe_json_parse(
            json_str: str, default: dict[str, object] | None = None
        ) -> dict[str, object]:
            """Safely parse JSON string."""
            try:
                result = json.loads(json_str)
                return result if isinstance(result, dict) else (default or {})
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
                data = json.loads(json_text)
                # Use getattr to handle generic type properly
                if hasattr(model_class, "model_validate"):
                    model = model_class.model_validate(data)  # type: ignore[attr-defined]
                else:
                    # Fallback for non-Pydantic models
                    model = model_class(data)  # type: ignore[call-arg]
                return FlextResult[T].ok(model)
            except json.JSONDecodeError as e:
                return FlextResult[T].fail(f"Invalid JSON: {e}")
            except Exception as e:
                return FlextResult[T].fail(f"Model validation failed: {e}")

    class ResultUtils:
        """FlextResult utilities."""

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


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__: list[str] = [
    "FlextUtilities",  # ONLY main class exported
]
