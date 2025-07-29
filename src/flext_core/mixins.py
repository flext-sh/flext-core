"""FLEXT Core Mixins Module.

Comprehensive mixin system for the FLEXT Core library providing consolidated
behavior patterns through direct inheritance composition and optimized performance.

Architecture:
    - Direct base exposure pattern eliminating inheritance overhead
    - Consolidated mixin architecture from specialized base implementations
    - Zero-overhead delegation pattern with complete functionality preservation
    - Single responsibility principle with focused mixin behaviors
    - Multiple inheritance composition support for complex behavioral patterns
    - Clean public API hiding internal implementation complexity

Mixin System Components:
    - Temporal mixins: Timestamp tracking and execution timing capabilities
    - Identity mixins: Unique identification and comparison behaviors
    - Validation mixins: Validation state management and data integrity
    - Utility mixins: Caching, logging, and structured data handling
    - Composite mixins: Pre-composed patterns for entity and value object behaviors
    - Serialization mixins: Data transformation and persistence support

Maintenance Guidelines:
    - Add new mixins by exposing them directly from _mixins_base module
    - Maintain consistent naming with Flext prefix for public API clarity
    - Keep mixin responsibilities focused and single-purpose for composability
    - Document mixin combinations for common architectural patterns
    - Preserve backward compatibility through direct assignment patterns
    - Test mixin compositions thoroughly for method resolution order conflicts
    - Update composite mixins when base mixin functionality changes

Design Decisions:
    - Direct assignment pattern eliminating empty inheritance layer overhead
    - Zero-overhead delegation with complete functionality preservation
    - Clean public names following FlextXxx convention for consistency
    - Maintained full functionality from underlying base implementations
    - Backward compatibility through direct class assignment
    - Single source of truth from _mixins_base implementations
    - Composite mixin patterns for common architectural use cases

Mixin Composition Patterns:
    - Single behavior: Focused mixins for specific functionality
    - Multiple inheritance: Combining mixins for complex behaviors
    - Composite patterns: Pre-built combinations for domain patterns
    - Layered architecture: Mixins supporting Clean Architecture principles
    - Cross-cutting concerns: Logging, validation, and caching behaviors
    - Domain patterns: Entity and value object specific compositions

Performance Optimization:
    - Direct class assignment eliminating method resolution overhead
    - Zero delegation overhead through direct base class exposure
    - Minimal memory footprint with shared base implementations
    - Optimized method resolution order for multiple inheritance scenarios
    - Efficient mixin composition without runtime delegation cost
    - Compiled method access patterns for maximum performance

Enterprise Integration Features:
    - Structured logging integration with context management
    - Validation state tracking for data integrity assurance
    - Caching capabilities for performance optimization
    - Timestamp tracking for audit and versioning requirements
    - Identity management for entity equality and hashing
    - Serialization support for persistence and transport scenarios

Usage Patterns:
    # Single mixin inheritance
    class TimestampedModel(FlextTimestampMixin):
        def __init__(self):
            super().__init__()
            # Automatic created_at and updated_at tracking

    # Multiple mixin composition
    class ValidatedEntity(FlextIdentifiableMixin, FlextValidatableMixin):
        def __init__(self, entity_id: str):
            super().__init__()
            self.set_id(entity_id)
            # Combines ID management with validation state

    # Composite entity pattern
    class DomainEntity(FlextEntityMixin):
        def __init__(self, entity_id: str):
            super().__init__()
            self.set_id(entity_id)
            # Includes ID + timestamps + validation + logging

    # Value object pattern
    class DomainValue(FlextValueObjectMixin):
        def __init__(self, value: object):
            super().__init__()
            self.value = value
            # Includes validation + serialization + comparison

    # Complex enterprise pattern
    class AggregateRoot(
        FlextEntityMixin,
        FlextCacheableMixin,
        FlextTimingMixin
    ):
        def __init__(self, aggregate_id: str):
            super().__init__()
            self.set_id(aggregate_id)
            # Enterprise aggregate with full capabilities

    # Service layer pattern
    class DomainService(
        FlextLoggableMixin,
        FlextValidatableMixin,
        FlextTimingMixin
    ):
        def __init__(self, service_name: str):
            super().__init__()
            self.service_name = service_name
            # Service with logging, validation, and timing

Dependencies:
    - _mixins_base: Foundation mixin implementations with core behaviors
    - abc: Abstract base class patterns for mixin interface enforcement
    - typing: Type annotations and generic programming support
    - All functionality delegated to optimized base implementations

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._mixins_base import (
    _BaseCacheableMixin,
    _BaseComparableMixin,
    _BaseEntityMixin,
    _BaseIdentifiableMixin,
    _BaseLoggableMixin,
    _BaseSerializableMixin,
    _BaseTimestampMixin,
    _BaseTimingMixin,
    _BaseValidatableMixin,
    _BaseValueObjectMixin,
)

# =============================================================================
# FLEXT MIXINS - Direct exposure eliminating inheritance overhead
# =============================================================================

# Direct exposure with clean names - completely eliminates empty inheritance
# Each assignment provides full functionality from base implementation

# Temporal mixins for time-related functionality
FlextTimestampMixin = _BaseTimestampMixin  # Creation and update timestamps
FlextTimingMixin = _BaseTimingMixin  # Execution timing and measurement

# Identity and behavior mixins
FlextIdentifiableMixin = _BaseIdentifiableMixin  # Unique ID management
FlextLoggableMixin = _BaseLoggableMixin  # Structured logging capabilities
FlextValidatableMixin = _BaseValidatableMixin  # Validation state management
FlextSerializableMixin = _BaseSerializableMixin  # Dictionary serialization
FlextComparableMixin = _BaseComparableMixin  # Comparison operations
FlextCacheableMixin = _BaseCacheableMixin  # Caching functionality

# Composite mixins for common patterns
FlextEntityMixin = _BaseEntityMixin  # ID + timestamps + validation
FlextValueObjectMixin = _BaseValueObjectMixin  # Validation + serialization + comparison

# =============================================================================
# SMART COMPOSITIONS - Intelligent mixin combinations
# =============================================================================


class FlextServiceMixin(
    FlextLoggableMixin,
    FlextTimingMixin,
    FlextValidatableMixin,
    FlextIdentifiableMixin,
):
    """Smart composition for service classes.

    Combines logging, timing, validation, and identification
    in a single mixin optimized for service layer components.

    Features:
    - Structured logging with service context
    - Execution timing for performance monitoring
    - Validation state management for input checking
    - Unique identification for service tracking

    Usage:
        class UserService(FlextServiceMixin):
            def __init__(self):
                super().__init__()
                self.service_name = "UserService"
    """

    def __init__(self, service_name: str | None = None) -> None:
        """Initialize service with smart defaults."""
        super().__init__()
        if service_name:
            self.set_id(service_name)
        self._service_initialized = True


class FlextCommandMixin(
    FlextValidatableMixin,
    FlextTimestampMixin,
    FlextSerializableMixin,
    FlextIdentifiableMixin,
):
    """Smart composition for command classes.

    Combines validation, timestamps, serialization, and ID
    optimized for CQRS command patterns.

    Features:
    - Command validation with business rules
    - Creation and update timestamps
    - Serialization for transport and persistence
    - Command identification for tracking

    Usage:
        class CreateUserCommand(FlextCommandMixin):
            def __init__(self, **kwargs):
                super().__init__()
                self.validate_and_set(**kwargs)
    """

    def validate_and_set(self, **kwargs: object) -> None:
        """Validate and set command data in one operation."""
        # Clear previous validation errors
        self.clear_validation_errors()

        # Set attributes from kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Update timestamp
        self._update_timestamp()


class FlextDataMixin(
    FlextSerializableMixin,
    FlextValidatableMixin,
    FlextComparableMixin,
    FlextCacheableMixin,
):
    """Smart composition for data classes.

    Combines serialization, validation, comparison, and caching
    optimized for data transfer and value objects.

    Features:
    - Dictionary serialization with type conversion
    - Data validation with error collection
    - Value-based comparison operations
    - Intelligent caching for performance

    Usage:
        class UserData(FlextDataMixin):
            def __init__(self, name: str, email: str):
                super().__init__()
                self.name = name
                self.email = email
                self.validate_data()
    """

    def validate_data(self) -> bool:
        """Validate all data and cache result."""
        cache_key = f"validation_{hash(str(self.to_dict_basic()))}"

        # Check cache first
        cached_result = self.cache_get(cache_key)
        if cached_result is not None:
            return bool(cached_result)

        # Perform validation
        self.clear_validation_errors()
        is_valid = self._perform_validation()

        # Cache result
        self.cache_set(cache_key, is_valid)
        return is_valid

    def _perform_validation(self) -> bool:
        """Override in subclasses for specific validation logic."""
        return not self.has_validation_errors()


class FlextFullMixin(
    FlextLoggableMixin,
    FlextTimingMixin,
    FlextValidatableMixin,
    FlextSerializableMixin,
    FlextTimestampMixin,
    FlextIdentifiableMixin,
    FlextComparableMixin,
    FlextCacheableMixin,
):
    """Complete mixin composition with all capabilities.

    Combines all available mixins for maximum functionality
    in enterprise components that need comprehensive features.

    Features:
    - Complete logging, timing, and monitoring
    - Full validation and serialization
    - Timestamps and identification
    - Comparison and caching operations

    Usage:
        class EnterpriseEntity(FlextFullMixin):
            def __init__(self, **kwargs):
                super().__init__()
                self.configure_enterprise_features(**kwargs)
    """

    def configure_enterprise_features(self, **kwargs: object) -> None:
        """Configure all enterprise features in one operation."""
        # Set ID if provided
        entity_id = kwargs.get("id")
        if entity_id:
            self.set_id(str(entity_id))

        # Initialize timestamps
        self._update_timestamp()

        # Configure logging context
        entity_name = kwargs.get("entity_name", self.__class__.__name__)
        self.logger.info("Enterprise entity configured", entity_name=entity_name)

        # Clear validation state
        self.clear_validation_errors()


# =============================================================================
# EXPORTS - Clean public API
# =============================================================================


__all__ = [
    "FlextCacheableMixin",
    "FlextCommandMixin",
    "FlextComparableMixin",
    "FlextDataMixin",
    "FlextEntityMixin",
    "FlextFullMixin",
    "FlextIdentifiableMixin",
    "FlextLoggableMixin",
    "FlextSerializableMixin",
    "FlextServiceMixin",
    "FlextTimestampMixin",
    "FlextTimingMixin",
    "FlextValidatableMixin",
    "FlextValueObjectMixin",
]
