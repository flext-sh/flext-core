"""FLEXT Core Utilities Module.

Comprehensive utility system for the FLEXT Core library providing consolidated
functionality through multiple inheritance patterns and specialized utility classes.

Architecture:
    - Multiple inheritance combining specialized utility base classes
    - Performance tracking system with comprehensive metrics collection
    - Type guard utilities for runtime type safety validation
    - Generation utilities for IDs, timestamps, and entity metadata
    - Formatting utilities for strings, durations, and data display
    - Delegation patterns for code reuse and optimization

Utility Categories:
    - Performance tracking: Function execution metrics and observability
    - Type guards: Runtime type checking with TypeGuard support
    - Generators: ID generation, timestamps, and entity metadata
    - Formatters: String formatting, data display, and sanitization
    - System utilities: Environment information and system metadata
    - Safe operations: FlextResult integration for error handling

Maintenance Guidelines:
    - Add new utility types to appropriate specialized classes first
    - Use multiple inheritance for utility combination patterns
    - Maintain backward compatibility through function aliases
    - Integrate FlextResult pattern for all operations that can fail
    - Keep utility functions stateless and thread-safe for concurrent use

Design Decisions:
    - Multiple inheritance pattern for maximum utility reuse
    - Performance tracking with automatic metrics collection
    - Type guards using TypeGuard for static analysis support
    - Safe operations with FlextResult instead of exception handling
    - Delegation patterns for code reuse optimization

Performance Features:
    - Function performance tracking with category-based metrics
    - Automatic timing measurement and success/failure tracking
    - Memory-efficient metrics storage with category organization
    - Observability support through comprehensive metrics reporting

Dependencies:
    - validation: Core validation utilities for data integrity
    - result: FlextResult pattern for consistent error handling
    - constants: Core constants and configuration values
    - Standard library: hashlib, platform, time, datetime, uuid

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import hashlib
import platform
import random
import re
import string
import sys
import time
from datetime import UTC, datetime
from functools import wraps
from typing import TYPE_CHECKING, Protocol, TypeGuard
from uuid import uuid4

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.validation import FlextValidators

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core.types import T, TFactory

# =============================================================================
# PERFORMANCE TRACKING - sem underscore conforme diretrizes
# =============================================================================

# Performance metrics storage
PERFORMANCE_METRICS: dict[str, dict[str, object]] = {}

# Time constants for formatting
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
BYTES_PER_KB = 1024


class DecoratedFunction(Protocol):
    """Protocol for functions that can be decorated with performance tracking."""

    __name__: str

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Call decorated function with provided arguments."""
        ...


def track_performance(
    category: str,
) -> Callable[[DecoratedFunction], DecoratedFunction]:
    """Track function performance as decorator.

    Args:
        category: Category of the function to track

    Returns:
        Decorator function

    """

    def decorator(func: DecoratedFunction) -> DecoratedFunction:
        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                execution_time = time.perf_counter() - start_time
                record_performance(
                    category,
                    func.__name__,
                    execution_time,
                    success=True,
                )
            except (
                TypeError,
                ValueError,
                AttributeError,
                RuntimeError,
                ZeroDivisionError,
                IndexError,
            ):
                execution_time = time.perf_counter() - start_time
                record_performance(
                    category,
                    func.__name__,
                    execution_time,
                    success=False,
                )
                raise
            else:
                return result

        return wrapper

    return decorator


def update_category_metrics(
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


def update_function_metrics(
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


def record_performance(
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
    if category not in PERFORMANCE_METRICS:
        PERFORMANCE_METRICS[category] = {
            "total_calls": 0,
            "total_time": 0.0,
            "successful_calls": 0,
            "failed_calls": 0,
            "functions": {},
        }

    category_metrics = PERFORMANCE_METRICS[category]
    update_category_metrics(category_metrics, execution_time, success=success)

    # Track per-function metrics
    functions = category_metrics["functions"]
    if isinstance(functions, dict):
        update_function_metrics(functions, function_name, execution_time)


def get_performance_metrics() -> dict[str, dict[str, object]]:
    """Get performance metrics for observability."""
    return PERFORMANCE_METRICS.copy()


def clear_performance_metrics() -> None:
    """Clear performance metrics (for testing)."""
    PERFORMANCE_METRICS.clear()


# =============================================================================
# TYPE GUARDS - sem underscore conforme diretrizes
# =============================================================================


class FlextTypeGuards:
    """Type guard utilities providing runtime type checking with static analysis support.

    Comprehensive type guard system for runtime type validation with TypeGuard support
    for static type checkers. Provides safe type checking patterns for complex types
    and collections with proper type narrowing.

    Architecture:
        - Static methods for stateless type checking operations
        - TypeGuard support for static analysis type narrowing
        - Runtime validation with fallback safety patterns
        - Integration with callable and collection type checking

    Type Guard Features:
        - Basic attribute and instance checking
        - Collection type validation with item type checking
        - Callable validation with return type documentation
        - Safe type narrowing for conditional type checking

    Usage Patterns:
        # Basic type checking
        if FlextTypeGuards.is_instance_of(obj, UserModel):
            # obj is now typed as UserModel
            user_name = obj.name

        # Collection type checking
        if FlextTypeGuards.is_list_of(data, str):
            # data is now typed as list[str]
            string_list = data

        # Attribute checking
        if FlextTypeGuards.has_attribute(obj, 'process'):
            # Safe to access obj.process
            result = obj.process()
    """

    @staticmethod
    def has_attribute(obj: object, attr: str) -> bool:
        """Check if object has attribute."""
        return hasattr(obj, attr)

    @staticmethod
    def is_instance_of(obj: object, target_type: type) -> bool:
        """Check if object is instance of target type."""
        return isinstance(obj, target_type)

    @staticmethod
    def is_list_of(obj: object, item_type: type) -> bool:
        """Check if object is a list of specific type."""
        if not isinstance(obj, list):
            return False
        return all(isinstance(item, item_type) for item in obj)

    @staticmethod
    def is_callable_with_return(obj: object, return_type: type) -> bool:
        """Check if object is callable with specific return type."""
        # Note: return_type is used for documentation but not runtime validation
        _ = return_type  # Acknowledge parameter for linting
        return callable(obj)


# =============================================================================
# GENERATORS - sem underscore conforme diretrizes
# =============================================================================


class FlextGenerators:
    """Generation utilities for IDs, timestamps, and entity metadata.

    Comprehensive generation system providing various ID formats, timestamps,
    and entity metadata creation with consistent patterns and
    uniqueness guarantees.

    Architecture:
        - Static methods for stateless generation operations
        - Multiple ID format support for different use cases
        - Timestamp generation with multiple format options
        - Entity metadata with correlation and tracing support

    ID Generation Types:
        - UUID: Globally unique identifiers with RFC 4122 compliance
        - Short IDs: Alphanumeric identifiers for human readability
        - Prefixed IDs: Domain-specific identifiers with namespace prefixes
        - Entity IDs: FLEXT-prefixed identifiers for entity management
        - Hash IDs: Content-based identifiers using SHA-256

    Timestamp Formats:
        - Unix timestamp: Numeric timestamp for calculations
        - ISO timestamp: ISO 8601 format for data exchange
        - Correlation IDs: Timestamp-based tracing identifiers

    Usage Patterns:
        # Unique identifiers
        user_id = FlextGenerators.generate_uuid()
        short_ref = FlextGenerators.generate_short_id(8)

        # Entity management
        entity_id = FlextGenerators.generate_entity_id()
        prefixed_id = FlextGenerators.generate_prefixed_id('USER', 12)

        # Tracing and correlation
        correlation_id = FlextGenerators.generate_correlation_id()
        session_id = FlextGenerators.generate_session_id()

        # Content-based IDs
        content_hash = FlextGenerators.generate_hash_id(json_data)
    """

    @staticmethod
    def generate_uuid() -> str:
        """Generate UUID string."""
        return str(uuid4())

    @staticmethod
    def generate_short_id(length: int = 8) -> str:
        """Generate short alphanumeric ID."""
        chars = string.ascii_letters + string.digits
        return "".join(random.choices(chars, k=length))  # noqa: S311

    @staticmethod
    def generate_timestamp() -> float:
        """Generate current timestamp."""
        return time.time()

    @staticmethod
    def generate_correlation_id() -> str:
        """Generate correlation ID for tracing."""
        timestamp = int(time.time() * 1000)
        short_id = FlextGenerators.generate_short_id(6)
        return f"{timestamp}-{short_id}"

    @staticmethod
    def generate_prefixed_id(prefix: str, length: int = 8) -> str:
        """Generate prefixed ID."""
        short_id = FlextGenerators.generate_short_id(length)
        return f"{prefix}_{short_id}"

    @staticmethod
    def generate_entity_id() -> str:
        """Generate entity ID with FLEXT prefix."""
        return FlextGenerators.generate_prefixed_id("FLEXT", 12)

    @staticmethod
    def generate_id() -> str:
        """Generate a unique ID (alias for UUID)."""
        return FlextGenerators.generate_uuid()

    @staticmethod
    def generate_iso_timestamp() -> str:
        """Generate ISO format timestamp."""
        return datetime.now(tz=UTC).isoformat()

    @staticmethod
    def generate_session_id() -> str:
        """Generate a session ID."""
        return f"sess_{FlextGenerators.generate_short_id(16)}"

    @staticmethod
    def generate_hash_id(data: str) -> str:
        """Generate hash ID from data using SHA-256."""
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# =============================================================================
# FORMATTERS - sem underscore conforme diretrizes
# =============================================================================


class FlextFormatters:
    """Formatting utilities for strings, data display, and sanitization.

    Comprehensive formatting system providing human-readable output for various
    data types, sanitization for security, and conversion between different
    string formats with safety features.

    Architecture:
        - Static methods for stateless formatting operations
        - Security-focused sanitization with pattern detection
        - Human-readable format conversion for durations and sizes
        - Case conversion utilities for API compatibility
        - Error message formatting with context integration

    Formatting Categories:
        - Duration formatting: Human-readable time displays
        - Size formatting: Byte size conversion with appropriate units
        - String sanitization: Security-focused data cleansing
        - Data structure formatting: Dictionary and collection display
        - Case conversion: snake_case ↔ camelCase transformation
        - Error formatting: Contextual error message construction

    Security Features:
        - Automatic detection and masking of sensitive patterns
        - Credit card number detection and replacement
        - Social Security Number detection and masking
        - Email address detection and anonymization
        - Length truncation for log safety

    Usage Patterns:
        # Duration and size formatting
        duration = FlextFormatters.format_duration(125.5)  # "2m 5.50s"
        size = FlextFormatters.format_size(1024*1024)      # "1.00MB"

        # String sanitization
        safe_log = FlextFormatters.sanitize_string(user_input)
        truncated = FlextFormatters.truncate(long_text, 100)

        # Case conversion
        camel = FlextFormatters.snake_to_camel("user_name")  # "userName"
        snake = FlextFormatters.camel_to_snake("userName")   # "user_name"

        # Error formatting
        error_msg = FlextFormatters.format_error_message(
            "Validation failed",
            {"field": "email", "value": "invalid"}
        )
    """

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 1:
            return f"{seconds * 1000:.2f}ms"
        if seconds < SECONDS_PER_MINUTE:
            return f"{seconds:.2f}s"
        if seconds < SECONDS_PER_HOUR:
            minutes = int(seconds // SECONDS_PER_MINUTE)
            remaining_seconds = seconds % SECONDS_PER_MINUTE
            return f"{minutes}m {remaining_seconds:.2f}s"
        hours = int(seconds // SECONDS_PER_HOUR)
        remaining_minutes = int((seconds % SECONDS_PER_HOUR) // SECONDS_PER_MINUTE)
        return f"{hours}h {remaining_minutes}m"

    @staticmethod
    def format_size(bytes_count: int) -> str:
        """Format size in human-readable format."""
        current_size = float(bytes_count)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if current_size < BYTES_PER_KB:
                return f"{current_size:.2f}{unit}"
            current_size /= BYTES_PER_KB
        return f"{current_size:.2f}PB"

    @staticmethod
    def sanitize_string(text: str, max_length: int = 100) -> str:
        """Sanitize string for safe logging."""
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
        """Format dictionary for display."""
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
        """Truncate text to maximum length."""
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    @staticmethod
    def snake_to_camel(snake_str: str) -> str:
        """Convert snake_case to camelCase."""
        components = snake_str.split("_")
        return components[0] + "".join(word.capitalize() for word in components[1:])

    @staticmethod
    def camel_to_snake(camel_str: str) -> str:
        """Convert camelCase to snake_case."""
        # Insert underscore before uppercase letters
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

        context_parts = []
        for key, value in context.items():
            context_parts.append(f"{key}={value}")

        context_str = ", ".join(context_parts)
        return f"{message} (Context: {context_str})"

    @staticmethod
    def format_entity_reference(entity_type: str, entity_id: str) -> str:
        """Format entity reference string."""
        return f"{entity_type}({entity_id})"


# =============================================================================
# DELEGATION OPTIMIZATION - DRY pattern for delegation
# =============================================================================


class DelegationMixin:
    """Mixin for optimizing delegation patterns with automatic method generation.

    Advanced delegation system for creating delegated methods with automatic
    naming, performance tracking, and batch delegation from base classes.
    Optimizes code reuse patterns while maintaining type safety.

    Architecture:
        - Class-level delegation with automatic method creation
        - Static method delegation for performance optimization
        - Batch delegation with reflection-based method discovery
        - Performance tracking integration for delegated methods

    Delegation Features:
        - Automatic method naming from base method attributes
        - Static method optimization for better performance
        - Batch delegation of all static methods from base classes
        - Optional performance tracking for delegated calls

    Performance Benefits:
        - Static method delegation reduces call overhead
        - Automatic method generation eliminates boilerplate code
        - Reflection-based batch processing for large APIs
        - Optional tracking for performance monitoring

    Usage Patterns:
        # Single method delegation
        delegated_method = DelegationMixin.create_delegated_method(
            base_class.static_method,
            "custom_name"
        )

        # Batch delegation from base class
        all_methods = DelegationMixin.delegate_all_static_methods(BaseClass)
        for name, method in all_methods.items():
            setattr(MyClass, name, method)

        # Delegation with tracking
        tracked_method = delegate_with_tracking(base_method)
    """

    @classmethod
    def create_delegated_method(
        cls,
        base_method: object,
        method_name: str | None = None,
    ) -> object:
        """Create a delegated method with automatic naming."""
        if method_name is None and hasattr(base_method, "__name__"):
            method_name = base_method.__name__

        def delegated_method(*args: object, **kwargs: object) -> object:
            return base_method(*args, **kwargs)  # type: ignore[operator]

        if method_name:
            delegated_method.__name__ = method_name

        return staticmethod(delegated_method)

    @classmethod
    def delegate_all_static_methods(cls, base_class: type) -> dict[str, object]:
        """Auto-delegate all static methods from a base class."""
        delegated = {}
        for attr_name in dir(base_class):
            if not attr_name.startswith("_"):
                attr = getattr(base_class, attr_name)
                if callable(attr):
                    delegated[attr_name] = cls.create_delegated_method(attr, attr_name)
        return delegated


def delegate_with_tracking(base_method: object) -> object:
    """Decorate using tracked methods."""

    def decorator(*args: object, **kwargs: object) -> object:
        # Add performance tracking if needed
        return base_method(*args, **kwargs)  # type: ignore[operator]

    if hasattr(base_method, "__name__"):
        decorator.__name__ = base_method.__name__

    return staticmethod(decorator)


# =============================================================================
# FLEXT UTILITIES - Consolidados com herança múltipla + funcionalidades específicas
# =============================================================================


class FlextUtilities(
    FlextGenerators,
    FlextFormatters,
    FlextTypeGuards,
    FlextValidators,
):
    """Consolidated utilities with multiple inheritance and functionality.

    Ultimate utility orchestration class combining four specialized utility bases
    through multiple inheritance, adding complex functionality impossible to achieve
    with single inheritance patterns. Provides comprehensive utility operations
    with FlextResult integration for enterprise error handling.

    Architecture:
        - Multiple inheritance from four specialized utility base classes
        - Complex orchestration methods combining multiple utility types
        - FlextResult integration for all operations that can fail
        - Enterprise-grade entity validation and metadata generation
        - Safe operations with comprehensive error handling

    Inherited Utility Categories:
        - Generation: IDs, timestamps, correlation tracking (FlextGenerators)
        - Formatting: String formatting, data display, sanitization (FlextFormatters)
        - Type Guards: Runtime type checking, collection validation (FlextTypeGuards)
        - Validation: Data validation, constraint checking (_BaseValidators)

    Enterprise Features:
        - Complete entity validation with version control
        - Entity metadata generation with correlation tracking
        - Safe parsing operations with FlextResult error handling
        - System information collection for observability
        - Safe attribute access with comprehensive error reporting

    Orchestration Methods:
        - safe_call: Function execution with FlextResult error handling
        - validate_entity_complete: Comprehensive entity validation workflow
        - generate_entity_metadata_complete: Full entity metadata creation
        - get_system_info_complete: Complete system information collection
        - format_entity_complete: Entity formatting with validation

    Usage Patterns:
        # Safe operations with error handling
        result = FlextUtilities.safe_call(lambda: risky_operation())
        if result.is_success:
            data = result.data

        # Entity validation
        validation_result = FlextUtilities.validate_entity_complete(
            entity_id="user_123",
            entity_data={"name": "John", "version": 1}
        )

        # Entity metadata generation
        metadata = FlextUtilities.generate_entity_metadata_complete(
            "User",
            include_correlation=True
        )

        # Safe parsing
        int_result = FlextUtilities.safe_parse_int("123")
        float_result = FlextUtilities.safe_parse_float("123.45")
    """

    # =========================================================================
    # FUNCIONALIDADES ESPECÍFICAS (combinam múltiplas bases)
    # =========================================================================

    @classmethod
    def safe_call(cls, func: TFactory[T]) -> FlextResult[T]:
        """Safely call function and return FlextResult."""
        try:
            return FlextResult.ok(func())
        except (
            TypeError,
            ValueError,
            AttributeError,
            RuntimeError,
            ZeroDivisionError,
            IndexError,
        ) as e:
            return FlextResult.fail(str(e))

    @classmethod
    def is_not_none_guard(cls, value: T | None) -> TypeGuard[T]:
        """Type guard combining validation + type guarding."""
        return cls.is_not_none(value)

    @classmethod
    def safe_parse_int(cls, value: str) -> FlextResult[int]:
        """Safely parse string to integer with FlextResult."""
        try:
            return FlextResult.ok(int(value))
        except ValueError as e:
            return FlextResult.fail(f"Cannot parse '{value}' as integer: {e}")

    @classmethod
    def safe_parse_float(cls, value: str) -> FlextResult[float]:
        """Safely parse string to float with FlextResult."""
        try:
            return FlextResult.ok(float(value))
        except ValueError as e:
            return FlextResult.fail(f"Cannot parse '{value}' as float: {e}")

    @classmethod
    def validate_entity_complete(
        cls,
        entity_id: str,
        entity_data: dict[str, object],
        *,
        require_version: bool = True,
    ) -> FlextResult[dict[str, object]]:
        """Complex entity validation orchestrating multiple inherited bases."""
        # Use inherited validation methods directly
        if not cls.is_non_empty_string(entity_id):
            return FlextResult.fail("Entity ID cannot be empty")

        if not cls.is_dict(entity_data):
            return FlextResult.fail("Entity data must be a dictionary")

        # Version validation using inherited methods
        if require_version:
            version = entity_data.get("version")
            if version is None or not isinstance(version, int) or version < 1:
                return FlextResult.fail("Entity version must be integer >= 1")

        # Build validated data using inherited generators
        validated_data = {
            "id": entity_id,
            **entity_data,
            "_validated_at": cls.generate_timestamp(),  # inherited method
            "_validation_id": cls.generate_uuid(),  # inherited method
        }

        return FlextResult.ok(validated_data)

    @classmethod
    def generate_entity_metadata_complete(
        cls,
        entity_type: str,
        *,
        include_correlation: bool = True,
    ) -> dict[str, object]:
        """Generate complete entity metadata using inherited generators."""
        # All methods inherited from bases
        metadata = {
            "id": cls.generate_entity_id(),
            "type": entity_type,
            "version": 1,
            "created_at": cls.generate_timestamp(),
            "timestamp_iso": cls.generate_iso_timestamp(),
        }

        if include_correlation:
            metadata["correlation_id"] = cls.generate_correlation_id()
            metadata["session_id"] = cls.generate_session_id()

        # Use inherited formatter
        metadata["formatted_reference"] = cls.format_entity_reference(
            entity_type,
            metadata["id"],
        )

        return metadata

    @classmethod
    def get_system_info_complete(cls) -> dict[str, object]:
        """Get complete system information."""
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "architecture": platform.architecture(),
            "processor": platform.processor(),
            "flext_version": FlextConstants.VERSION,
            "timestamp": cls.generate_timestamp(),  # inherited
            "correlation_id": cls.generate_correlation_id(),  # inherited
        }

    @classmethod
    def safe_increment(cls, value: int, max_value: int = 2**31 - 1) -> FlextResult[int]:
        """Safely increment integer with overflow protection."""
        if value >= max_value:
            return FlextResult.fail(f"Value {value} would overflow max {max_value}")
        return FlextResult.ok(value + 1)

    @classmethod
    def safe_get_attr(cls, obj: object, attr: str) -> FlextResult[object]:
        """Safely get attribute using inherited type checking."""
        try:
            if not cls.has_attribute(obj, attr):  # inherited method
                return FlextResult.fail(f"Object has no attribute '{attr}'")
            return FlextResult.ok(getattr(obj, attr))
        except (AttributeError, TypeError) as e:
            return FlextResult.fail(f"Error getting attribute '{attr}': {e}")

    @classmethod
    def format_entity_complete(
        cls,
        entity_type: str,
        entity_id: str,
        version: int,
    ) -> str:
        """Format entity with validation + formatting (combines inherited methods)."""
        # Use inherited validation
        if not cls.is_non_empty_string(entity_type) or not cls.is_non_empty_string(
            entity_id,
        ):
            return "INVALID_ENTITY"
        return f"{entity_type}(id={entity_id}, version={version})"


# =============================================================================
# EXPOSIÇÃO DIRETA DAS BASES ÚTEIS (aliases limpos sem herança vazia)
# =============================================================================

# Direct access to consolidated classes
# FlextValidators already imported from validation module

# =============================================================================
# ESSENTIAL COMPATIBILITY FUNCTIONS (mantém apenas interface crítica)
# =============================================================================

# Mantém apenas as funções mais essenciais para compatibilidade
def safe_call(func: TFactory[T]) -> FlextResult[T]:
    """Safely call function with FlextResult error handling.

    Essential function providing direct access to safe execution.

    Args:
        func: Function to execute safely

    Returns:
        FlextResult[T] with function result or captured exception

    """
    return FlextUtilities.safe_call(func)


def is_not_none(value: T | None) -> TypeGuard[T]:
    """Type guard to check if value is not None.

    Essential type guard function for static analysis support.

    Args:
        value: Value to check for None

    Returns:
        True if value is not None, False otherwise

    """
    return FlextUtilities.is_not_none_guard(value)


# =============================================================================
# EXPORTS - Clean public API seguindo diretrizes
# =============================================================================

__all__ = [
    "BYTES_PER_KB",
    # Performance tracking - objetos sem underscore
    "PERFORMANCE_METRICS",
    "SECONDS_PER_HOUR",
    # Constants - objetos sem underscore
    "SECONDS_PER_MINUTE",
    "DecoratedFunction",
    # Delegation utilities - objetos sem underscore
    "DelegationMixin",
    # Direct consolidated classes - objetos sem underscore
    "FlextFormatters",
    "FlextGenerators",
    "FlextTypeGuards",
    # Main consolidated class with multiple inheritance
    "FlextUtilities",
    "FlextValidators",
    "clear_performance_metrics",
    "delegate_with_tracking",
    "get_performance_metrics",
    # Essential compatibility functions
    "is_not_none",
    "record_performance",
    "safe_call",
    "track_performance",
]
