"""FLEXT Core Mixins - Extension Layer Behavior Composition.

Comprehensive mixin system providing behavior patterns through multiple inheritance
composition across the 32-project FLEXT ecosystem. Foundation for cross-cutting
concerns, domain-specific behaviors, and architectural pattern implementation in
data integration and business logic components.

Module Role in Architecture:
    Extension Layer → Behavior Composition → Cross-Cutting Concerns

    This module provides mixin patterns used throughout FLEXT projects:
    - Cross-cutting concerns like logging, validation, and timing
    - Domain-specific behaviors for entities, value objects, and aggregates
    - Utility mixins for caching, serialization, and data handling
    - Composite mixins for common architectural patterns

Mixin Architecture Patterns:
    Single Responsibility: Focused mixins for specific functionality
    Multiple Inheritance: Combining mixins for complex behaviors
    Composite Patterns: Pre-built combinations for domain patterns
    Zero Overhead: Direct class assignment eliminating delegation overhead

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Logging, validation, timing, serialization mixins
    🚧 Active Development: Domain-specific mixin patterns (Enhancement 3 - Med)
    📋 TODO Integration: Plugin mixin architecture (Priority 3)

Mixin System Components:
    Temporal Mixins: FlextTimingMixin for execution timing and performance tracking
    Identity Mixins: FlextIdentityMixin for unique identification and comparison
    Validation Mixins: FlextValidatableMixin for validation state management
    Utility Mixins: FlextLoggableMixin, FlextCachingMixin for cross-cutting concerns
    Composite Mixins: FlextEntityMixin, FlextValueObjectMixin for domain patterns

Ecosystem Usage Patterns:
    # FLEXT Service Domain Objects
    class User(BaseModel, FlextLoggableMixin, FlextValidatableMixin):
        name: str
        email: str

        def validate_business_rules(self) -> FlextResult[None]:
            return self.validate_email_format()

    # Singer Tap/Target Components
    class OracleExtractor(FlextTimingMixin, FlextLoggableMixin):
        def extract_data(self, table: str) -> FlextResult[list]:
            with self.time_operation("extract_data"):
                self.logger.info("Extracting data", table=table)
                return self.perform_extraction(table)

    # ALGAR Migration Components
    class LdapMigrator(FlextEntityMixin, FlextCachingMixin):
        def migrate_users(self, source_dn: str) -> FlextResult[int]:
            cached_result = self.get_cached("migration_" + source_dn)
            if cached_result:
                return cached_result
            return self.perform_migration(source_dn)

Mixin Composition Benefits:
    - Cross-cutting concerns without code duplication
    - Behavior composition through multiple inheritance
    - Domain pattern reuse across different entity types
    - Performance optimization through zero-overhead delegation

Quality Standards:
    - All mixins must be stateless or provide proper state management
    - Mixin combinations must be tested for method resolution order conflicts
    - Cross-cutting concerns must be implemented through mixins, not inheritance
    - Domain mixins should encapsulate related behaviors for specific patterns

See Also:
    docs/TODO.md: Enhancement 3 - Domain-specific mixin development
    _mixins_base.py: Foundation mixin implementations
    entities.py: Entity patterns using mixin composition

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
