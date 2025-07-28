"""FLEXT Core Utilities Module.

Comprehensive utility system for the FLEXT Core library providing consolidated
functionality through single inheritance pattern and direct delegation to base
implementations for maximum efficiency and minimum code duplication.

Architecture:
    - Single inheritance from _BaseUtilities for all core functionality
    - Direct delegation pattern eliminating unnecessary method wrapping
    - Consolidated class design following "deliver more with much less" principle
    - Clean separation between internal implementation and public API
    - Maximum reuse of base implementations without duplication

Utility Categories:
    - Performance tracking: Function execution metrics and observability
    - Type guards: Runtime type checking with TypeGuard support
    - Generators: ID generation, timestamps, and entity metadata
    - Formatters: String formatting, data display, and sanitization
    - System utilities: Environment information and system metadata
    - Safe operations: FlextResult integration for error handling

Maintenance Guidelines:
    - Add new utilities to _utilities_base.py following established patterns
    - Use direct delegation instead of method wrapping for performance
    - Maintain backward compatibility through function aliases
    - Integrate FlextResult pattern for all operations that can fail
    - Keep utilities stateless and thread-safe for concurrent use

Design Decisions:
    - Single class inheritance pattern instead of multiple inheritance
    - Direct access to base functionality through inheritance
    - Performance-optimized delegation patterns
    - Backward compatibility through module-level function aliases
    - Clean API design with minimal method overhead

Dependencies:
    - _utilities_base: All core utility implementations
    - validation: FlextValidators for data integrity
    - result: FlextResult pattern for consistent error handling
    - constants: Core constants and configuration values

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import platform
import sys
from typing import TYPE_CHECKING, Protocol, TypeGuard

from flext_core._utilities_base import (
    _BaseFormatters,
    _BaseGenerators,
    _BaseTypeGuards,
    _clear_performance_metrics,
    _DelegationMixin,
    _get_performance_metrics,
    _record_performance,
    _track_performance,
)
from flext_core.constants import FlextConstants
from flext_core.result import FlextResult, safe_call
from flext_core.validation import FlextValidators

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core.types import T, TFactory

# =============================================================================
# PERFORMANCE TRACKING - Direct delegation to base implementation
# =============================================================================

# Import constants from base - elimina duplicação (DRY)
from flext_core._utilities_base import (
    _BYTES_PER_KB,
    _SECONDS_PER_HOUR,
    _SECONDS_PER_MINUTE,
)

# Re-export without underscore (public API)
SECONDS_PER_MINUTE = _SECONDS_PER_MINUTE
SECONDS_PER_HOUR = _SECONDS_PER_HOUR
BYTES_PER_KB = _BYTES_PER_KB

# Direct access to performance metrics
PERFORMANCE_METRICS = _get_performance_metrics()


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
    """Consolidated utilities class providing all utility functionality.

    Composition-based pattern providing access to all utility functionality
    from specialized base classes. Eliminates multiple inheritance complexity
    while maintaining full functionality and adding enterprise-specific features.

    Architecture:
        - Composition-based delegation to specialized base classes
        - Enterprise-grade extensions for complex operations
        - FlextResult integration for all operations that can fail
        - Thread-safe operations for concurrent application environments
        - Performance-optimized implementation through direct delegation

    Core Features:
        - Type guards: Runtime type checking with static analysis support
        - Generators: ID, timestamp, and metadata generation utilities
        - Formatters: String formatting, sanitization, and display utilities
        - Validators: Integration with FlextValidators for data integrity
        - System utilities: Environment and system information collection

    Enterprise Extensions:
        - Safe operations with FlextResult error handling
        - Complex entity validation and metadata generation
        - System information collection for observability
        - Performance-optimized implementations for high-throughput scenarios

    Usage Patterns:
        # Direct utility access
        user_id = FlextUtilities.generate_uuid()
        formatted = FlextUtilities.format_duration(123.45)

        # Safe operations with error handling
        result = FlextUtilities.safe_call(lambda: risky_operation())
        if result.is_success:
            data = result.data

        # Type checking
        if FlextUtilities.is_instance_of(obj, UserService):
            user_service: UserService = obj

        # Entity validation
        validation_result = FlextUtilities.validate_entity_complete(
            entity_id="user_123",
            entity_data={"name": "John", "version": 1}
        )
    """

    # =========================================================================
    # DELEGATED BASE FUNCTIONALITY (composition-based access to base classes)
    # =========================================================================

    # Generator methods - delegate to _BaseGenerators
    @classmethod
    def generate_uuid(cls) -> str:
        """Generate UUID (delegates to base)."""
        return _BaseGenerators.generate_uuid()

    @classmethod
    def generate_id(cls) -> str:
        """Generate unique ID (delegates to base)."""
        return _BaseGenerators.generate_id()

    @classmethod
    def generate_timestamp(cls) -> float:
        """Generate timestamp (delegates to base)."""
        return _BaseGenerators.generate_timestamp()

    @classmethod
    def generate_correlation_id(cls) -> str:
        """Generate correlation ID (delegates to base)."""
        return _BaseGenerators.generate_correlation_id()

    @classmethod
    def generate_entity_id(cls) -> str:
        """Generate entity ID (delegates to base)."""
        return _BaseGenerators.generate_entity_id()

    @classmethod
    def generate_iso_timestamp(cls) -> str:
        """Generate ISO timestamp (delegates to base)."""
        return _BaseGenerators.generate_iso_timestamp()

    @classmethod
    def generate_session_id(cls) -> str:
        """Generate session ID (delegates to base)."""
        return _BaseGenerators.generate_session_id()

    # Formatter methods - delegate to _BaseFormatters
    @classmethod
    def truncate(cls, text: str, max_length: int = 100, suffix: str = "...") -> str:
        """Truncate text (delegates to base)."""
        return _BaseFormatters.truncate(text, max_length, suffix)

    @classmethod
    def format_duration(cls, seconds: float) -> str:
        """Format duration (delegates to base)."""
        return _BaseFormatters.format_duration(seconds)

    @classmethod
    def format_entity_reference(cls, entity_type: str, entity_id: str) -> str:
        """Format entity reference (delegates to base)."""
        return _BaseFormatters.format_entity_reference(entity_type, entity_id)

    # Type guard methods - delegate to _BaseTypeGuards
    @classmethod
    def has_attribute(cls, obj: object, attr: str) -> bool:
        """Check if object has attribute (delegates to base)."""
        return _BaseTypeGuards.has_attribute(obj, attr)

    @classmethod
    def is_instance_of(cls, obj: object, target_type: type) -> bool:
        """Check if object is instance of type (delegates to base)."""
        return _BaseTypeGuards.is_instance_of(obj, target_type)

    # =========================================================================
    # ENTERPRISE-SPECIFIC FUNCTIONALITY (extensions beyond base classes)
    # =========================================================================

    @classmethod
    def safe_call(cls, func: TFactory[T]) -> FlextResult[T]:
        """Safely call function and return FlextResult."""
        # Delegate to result.py source of truth - elimina duplicação
        return safe_call(func)

    @classmethod
    def is_not_none_guard(cls, value: T | None) -> TypeGuard[T]:
        """Type guard combining validation + type guarding."""
        return FlextValidators.is_not_none(value)

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
        """Complex entity validation orchestrating multiple inherited functionality."""
        # Use inherited validation methods directly
        if not FlextValidators.is_non_empty_string(entity_id):
            return FlextResult.fail(FlextConstants.MESSAGES["ENTITY_ID_EMPTY"])

        if not FlextValidators.is_dict(entity_data):
            return FlextResult.fail("Entity data must be a dictionary")

        # Version validation
        if require_version:
            version = entity_data.get("version")
            if version is None or not isinstance(version, int) or version < 1:
                return FlextResult.fail("Entity version must be integer >= 1")

        # Build validated data using inherited generators
        validated_data = {
            "id": entity_id,
            **entity_data,
            "_validated_at": _BaseGenerators.generate_timestamp(),  # composition
            "_validation_id": _BaseGenerators.generate_uuid(),  # composition
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
        # All methods using composition from bases
        metadata = {
            "id": _BaseGenerators.generate_entity_id(),
            "type": entity_type,
            "version": 1,
            "created_at": _BaseGenerators.generate_timestamp(),
            "timestamp_iso": _BaseGenerators.generate_iso_timestamp(),
        }

        if include_correlation:
            metadata["correlation_id"] = _BaseGenerators.generate_correlation_id()
            metadata["session_id"] = _BaseGenerators.generate_session_id()

        # Use composition for formatter
        metadata["formatted_reference"] = _BaseFormatters.format_entity_reference(
            entity_type,
            str(metadata["id"]),
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
            "timestamp": _BaseGenerators.generate_timestamp(),  # composition
            "correlation_id": _BaseGenerators.generate_correlation_id(),  # composition
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
            if not _BaseTypeGuards.has_attribute(obj, attr):  # composition
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
        if not FlextValidators.is_non_empty_string(
            entity_type,
        ) or not FlextValidators.is_non_empty_string(
            entity_id,
        ):
            return "INVALID_ENTITY"
        return f"{entity_type}(id={entity_id}, version={version})"


# =============================================================================
# PUBLIC API FUNCTIONS - Direct delegation to FlextUtilities
# =============================================================================


def flext_track_performance(
    category: str,
) -> Callable[[DecoratedFunction], DecoratedFunction]:
    """Track function performance as decorator."""
    return _track_performance(category)


def flext_get_performance_metrics() -> dict[str, dict[str, object]]:
    """Get performance metrics for observability."""
    return _get_performance_metrics()


def flext_clear_performance_metrics() -> None:
    """Clear performance metrics (for testing)."""
    _clear_performance_metrics()


def flext_record_performance(
    category: str,
    function_name: str,
    execution_time: float,
    *,
    success: bool,
) -> None:
    """Record performance metrics for observability."""
    _record_performance(category, function_name, execution_time, success=success)


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


def is_not_none(value: object) -> bool:
    """Check if value is not None (backward compatibility)."""
    return FlextValidators.is_not_none(value)


# safe_call is imported from result.py (single source of truth)
# and delegated through FlextUtilities.safe_call for consistency


# =============================================================================
# ALIASES FOR SPECIALIZED CLASSES - Direct access to base functionality
# =============================================================================

# Direct aliases to avoid duplication
FlextTypeGuards = _BaseTypeGuards
FlextGenerators = _BaseGenerators
FlextFormatters = _BaseFormatters
DelegationMixin = _DelegationMixin


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
    "DelegationMixin",
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
    "generate_uuid",
    "is_not_none",
    "safe_call",
    "truncate",
]
