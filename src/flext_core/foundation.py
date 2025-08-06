"""FLEXT Foundation - Harmonized Semantic Patterns (v1.0.0).

This module provides the harmonized, unified foundation patterns for the entire
FLEXT ecosystem, eliminating duplications and conflicts identified in the
semantic patterns harmonization process.

🎯 HARMONIZATION OBJECTIVES:
    - Single Source of Truth: Each concept has ONE canonical definition
    - Semantic Consistency: All follow Flext[Domain][Type][Context] pattern
    - Type Safety: Strict MyPy compliance with modern Python 3.13 features
    - Migration Support: Backward compatibility during transition period
    - Zero Duplication: Eliminates conflicts between models.py, entities.py, types.py

CANONICAL DEFINITIONS:
    This module establishes the definitive patterns that all 33 FLEXT projects
    will migrate to, providing consistency and eliminating confusion.

Usage - NEW HARMONIZED IMPORTS:
    # Foundation classes
    from flext_core.foundation import FlextModel, FlextEntity, FlextValue, FlextConfig

    # Semantic type system
    from flext_core.foundation import FlextTypes

    # Factory patterns
    from flext_core.foundation import FlextFactory

Migration Status:
    ✅ FlextModel: Unified base class established
    ✅ FlextEntity: Single entity definition (validate_business_rules)
    ✅ FlextValue: Immutable value object pattern
    ✅ FlextConfig: Environment-aware configuration base
    🚧 FlextTypes: Semantic type system (in progress)
    📋 Full ecosystem migration: Planned for Phase 2

Quality Standards:
    - Python 3.13+ exclusive (no backward compatibility)
    - MyPy strict mode compliance (zero tolerance)
    - FlextResult integration for all operations
    - Comprehensive type annotations
    - Business rule validation standard

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

from flext_core.result import FlextResult

# =============================================================================
# TYPE VARIABLES - Foundation for generic programming
# =============================================================================

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
E = TypeVar("E", bound=Exception)

# =============================================================================
# HARMONIZED FOUNDATION CLASSES - Single Source of Truth
# =============================================================================


class FlextModel(BaseModel):
    """Harmonized universal base class for all FLEXT models.

    HARMONIZATION: Consolidates patterns from models.py, eliminating duplication
    and establishing single source of truth for all FLEXT projects.

    Foundation Principles:
        - Universal base for ALL FLEXT models across 33 projects
        - Semantic validation through validate_business_rules (STANDARD name)
        - FlextResult integration for type-safe error handling
        - Metadata support for extensibility
        - Cross-language serialization support (Go bridge compatibility)

    Usage Pattern:
        class ProjectModel(FlextModel):
            name: str
            value: int

            def validate_business_rules(self) -> FlextResult[None]:
                if self.value < 0:
                    return FlextResult.fail("Value must be non-negative")
                return FlextResult.ok(None)
    """

    model_config = ConfigDict(
        # Type safety and validation
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
        # Serialization
        arbitrary_types_allowed=True,
        validate_default=True,
        # Mutability (overridden in subclasses)
        frozen=False,
    )

    # Universal metadata for extensibility
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Model metadata for cross-project extensibility",
    )

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate model-specific business rules.

        HARMONIZED STANDARD: All models use validate_business_rules (not domain_rules)

        Override in subclasses to implement domain-specific validation.
        Must return FlextResult for consistent error handling across ecosystem.

        Returns:
            FlextResult[None]: Success if valid, failure with descriptive error message

        """
        return FlextResult.ok(None)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary representation for serialization."""
        return self.model_dump()

    def to_typed_dict(self) -> dict[str, object]:
        """Convert to typed dictionary excluding unset values."""
        return self.model_dump(exclude_unset=True)


class FlextValue(FlextModel, ABC):
    """Harmonized immutable value objects with attribute-based equality.

    HARMONIZATION: Consolidates value object patterns, providing single
    definitive implementation for Domain-Driven Design value objects.

    Foundation Principles:
        - Immutable after creation (frozen=True)
        - Attribute-based equality (not identity-based)
        - REQUIRED business rule validation implementation
        - Value semantics with structural comparison
        - Thread-safe for concurrent usage across services

    Usage Pattern:
        class EmailAddress(FlextValue):
            address: str

            def validate_business_rules(self) -> FlextResult[None]:
                if "@" not in self.address:
                    return FlextResult.fail("Invalid email format")
                return FlextResult.ok(None)
    """

    model_config = ConfigDict(
        # Type safety and validation
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
        # Serialization
        arbitrary_types_allowed=True,
        validate_default=True,
        # Immutable value object
        frozen=True,
    )

    def __hash__(self) -> int:
        """Hash based on all attributes for value semantics."""

        def make_hashable(item: object) -> object:
            """Convert unhashable types to hashable equivalents."""
            if isinstance(item, dict):
                return tuple(sorted(item.items()))
            if isinstance(item, list):
                return tuple(item)
            if isinstance(item, set):
                return frozenset(item)
            return item

        dumped = self.model_dump()
        hashable_items = []
        for key, value in sorted(dumped.items()):
            if key != "metadata":  # Exclude metadata from hash
                hashable_items.append((key, make_hashable(value)))
        return hash(tuple(hashable_items))

    def __eq__(self, other: object) -> bool:
        """Equality based on all attributes (value semantics)."""
        if not isinstance(other, type(self)):
            return False
        # Compare without metadata for value equality
        self_data = {k: v for k, v in self.model_dump().items() if k != "metadata"}
        other_data = {k: v for k, v in other.model_dump().items() if k != "metadata"}
        return self_data == other_data

    @abstractmethod
    def validate_business_rules(self) -> FlextResult[None]:
        """REQUIRED: Abstract business rule validation implementation."""


class FlextEntity(FlextModel, ABC):
    """Harmonized identity-based entities with lifecycle management.

    HARMONIZATION: Single definitive entity implementation, replacing
    conflicting definitions from models.py and entities.py.

    Foundation Principles:
        - Identity-based equality (not attribute-based)
        - Mutable for lifecycle changes
        - Optimistic locking versioning support
        - Domain event collection for event sourcing
        - REQUIRED business rule validation implementation

    Usage Pattern:
        class User(FlextEntity):
            name: str
            email: str

            def validate_business_rules(self) -> FlextResult[None]:
                if not self.email:
                    return FlextResult.fail("Email is required")
                return FlextResult.ok(None)
    """

    model_config = ConfigDict(
        # Type safety and validation
        extra="forbid",
        validate_assignment=True,
        use_enum_values=True,
        str_strip_whitespace=True,
        # Serialization
        arbitrary_types_allowed=True,
        validate_default=True,
        # Mutable entity
        frozen=False,
    )

    # Core identity
    id: str = Field(description="Unique entity identifier")
    version: int = Field(default=1, description="Entity version for optimistic locking")

    # Domain events for event sourcing
    domain_events: list[dict[str, object]] = Field(default_factory=list, exclude=True)

    def __hash__(self) -> int:
        """Hash based on entity ID (identity-based equality)."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equality based on entity ID (identity-based equality)."""
        if not isinstance(other, FlextEntity):
            return False
        return self.id == other.id

    def increment_version(self) -> None:
        """Increment version for optimistic locking."""
        self.version += 1

    def add_domain_event(self, event: dict[str, object]) -> None:
        """Add domain event for event sourcing."""
        self.domain_events.append(event)

    def clear_domain_events(self) -> list[dict[str, object]]:
        """Clear and return domain events."""
        events = self.domain_events.copy()
        self.domain_events.clear()
        return events

    @abstractmethod
    def validate_business_rules(self) -> FlextResult[None]:
        """REQUIRED: Abstract business rule validation implementation."""


class FlextConfig(FlextValue):
    """Harmonized environment-aware configuration with validation.

    HARMONIZATION: Foundation configuration pattern that integrates with
    environment systems while maintaining immutable value object semantics.

    Foundation Principles:
        - Immutable configuration objects
        - Environment variable integration ready
        - Comprehensive business rule validation
        - Secure secret handling patterns
        - Cross-service serialization support

    Usage Pattern:
        class DatabaseConfig(FlextConfig):
            host: str = "localhost"
            port: int = 5432
            username: str
            password: SecretStr

            def validate_business_rules(self) -> FlextResult[None]:
                if not (1 <= self.port <= 65535):
                    return FlextResult.fail("Invalid port range")
                return FlextResult.ok(None)
    """

    def validate_business_rules(self) -> FlextResult[None]:
        """Default configuration validation - override in subclasses."""
        return FlextResult.ok(None)


# =============================================================================
# HARMONIZED SEMANTIC TYPE SYSTEM - Hierarchical Organization
# =============================================================================


class FlextTypes:
    """Harmonized semantic type system with hierarchical domain organization.

    HARMONIZATION: Consolidates type systems from types.py and semantic_types.py,
    providing single canonical type organization following semantic patterns.

    Semantic Pattern: Flext[Domain][Type][Context]
    Organization: Hierarchical namespaces by domain for maximum clarity

    Usage:
        # Core functional types
        predicate: FlextTypes.Core.Predicate[User] = lambda u: u.is_active
        factory: FlextTypes.Core.Factory[Connection] = ConnectionFactory()

        # Data domain types
        connection: FlextTypes.Data.Connection = oracle_connection
        record: FlextTypes.Data.Record = {"name": "value"}

        # Authentication domain types
        token: FlextTypes.Auth.Token = jwt_token
        user: FlextTypes.Auth.User = authenticated_user
    """

    class Core:
        """Core functional and architectural types."""

        # Functional types
        type Predicate[T] = Callable[[T], bool]
        type Factory[T] = Callable[[], T] | Callable[[object], T]
        type Transformer[T, R] = Callable[[T], R]
        type Validator[T] = Callable[[T], bool | str]
        type Serializer[T] = Callable[[T], str | bytes | dict[str, object]]

        # Result and error handling types
        type Result[T, E] = T | E
        type OptionalResult[T] = T | None
        type ErrorHandler[E] = Callable[[E], None]

        # Container and dependency types
        type Container = Mapping[str, object]
        type ServiceLocator = Callable[[str], object]
        type ServiceFactory[T] = Callable[[], T]

        # Event and messaging types
        type EventHandler[TEvent] = Callable[[TEvent], None]
        type EventBus = Callable[[object], None]
        type MessageHandler[T] = Callable[[T], object]

        # Metadata and configuration types
        type Metadata = dict[str, object]
        type Settings = dict[str, object]
        type Configuration = Mapping[str, object]

    class Data:
        """Data integration and storage domain types."""

        # Connection and database types
        type Connection = object  # Protocol-based, defined by each project
        type ConnectionString = str
        type DatabaseConnection = object
        type Credentials = dict[str, str]

        # Data processing types
        type Record = dict[str, object]
        type RecordBatch = Sequence[dict[str, object]]
        type Schema = dict[str, object]
        type Query = str | dict[str, object]

        # Serialization types
        type Serializable = (
            dict[str, object] | list[object] | str | int | float | bool | None
        )
        type JsonData = dict[str, object] | list[object]

    class Auth:
        """Authentication and authorization domain types."""

        # Token and credential types
        type Token = str
        type AccessToken = str
        type ApiKey = str
        type Secret = str
        type Password = str

        # Authentication types
        type AuthProvider = object  # Protocol-based
        type AuthenticatedUser = dict[str, object]
        type LoginCredentials = dict[str, str]
        type AuthContext = dict[str, object]

        # Authorization types
        type Permission = str
        type Role = str
        type Policy = dict[str, object]

    class Observability:
        """Monitoring and observability domain types."""

        # Logging types
        type Logger = object  # Protocol-based
        type LogLevel = str
        type LogMessage = str
        type LogContext = dict[str, object]

        # Metrics types
        type Metric = dict[str, object]
        type MetricValue = int | float
        type Counter = int
        type Gauge = float

        # Tracing types
        type Tracer = object  # Protocol-based
        type TraceId = str
        type CorrelationId = str


# =============================================================================
# HARMONIZED SEMANTIC ENUMS - Domain-specific enumerations
# =============================================================================


class FlextConnectionType(StrEnum):
    """Harmonized connection types across FLEXT ecosystem."""

    DATABASE = "database"
    REDIS = "redis"
    LDAP = "ldap"
    ORACLE = "oracle"
    POSTGRES = "postgres"
    REST_API = "rest_api"
    GRPC = "grpc"
    FILE = "file"
    STREAM = "stream"


class FlextDataFormat(StrEnum):
    """Harmonized data formats."""

    JSON = "json"
    XML = "xml"
    CSV = "csv"
    LDIF = "ldif"
    YAML = "yaml"
    PARQUET = "parquet"
    AVRO = "avro"
    PROTOBUF = "protobuf"


class FlextOperationStatus(StrEnum):
    """Harmonized operation status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class FlextLogLevel(StrEnum):
    """Harmonized log levels."""

    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# HARMONIZED FACTORY PATTERN - Semantic model creation
# =============================================================================


class FlextFactory:
    """Harmonized semantic factory for model creation across the ecosystem.

    HARMONIZATION: Consolidates factory patterns with validation and type safety,
    providing consistent model instantiation across all 33 FLEXT projects.

    Design Principles:
        - Semantic creation by type and domain
        - FlextResult integration for error handling
        - Type-safe factory methods with proper generics
        - Business rule validation integration
        - Extensible for project-specific needs
    """

    @staticmethod
    def create_model[T: FlextModel](
        model_class: type[T],
        **kwargs: object,
    ) -> FlextResult[T]:
        """Create model with validation.

        Args:
            model_class: Model class to instantiate
            **kwargs: Model parameters

        Returns:
            FlextResult[T]: Created and validated model instance

        """
        try:
            # Use model_validate for type-safe construction with dict
            instance = model_class.model_validate(kwargs)
            validation_result = instance.validate_business_rules()
            if validation_result.is_failure:
                return FlextResult.fail(
                    validation_result.error or "Business rule validation failed",
                )
            return FlextResult.ok(instance)
        except Exception as e:
            return FlextResult.fail(f"Failed to create {model_class.__name__}: {e}")

    @staticmethod
    def create_entity[T: FlextEntity](
        entity_class: type[T],
        entity_id: str,
        **kwargs: object,
    ) -> FlextResult[T]:
        """Create entity with ID validation.

        Args:
            entity_class: Entity class to instantiate
            entity_id: Unique identifier for entity
            **kwargs: Entity parameters

        Returns:
            FlextResult[T]: Created and validated entity instance

        """
        if not entity_id or not entity_id.strip():
            return FlextResult.fail("Entity ID cannot be empty")

        return FlextFactory.create_model(entity_class, id=entity_id, **kwargs)

    @staticmethod
    def create_value[T: FlextValue](
        value_class: type[T],
        **kwargs: object,
    ) -> FlextResult[T]:
        """Create value object with immutability validation.

        Args:
            value_class: Value object class to instantiate
            **kwargs: Value object parameters

        Returns:
            FlextResult[T]: Created and validated value object instance

        """
        return FlextFactory.create_model(value_class, **kwargs)

    @staticmethod
    def create_config[T: FlextConfig](
        config_class: type[T],
        **kwargs: object,
    ) -> FlextResult[T]:
        """Create configuration with environment integration.

        Args:
            config_class: Configuration class to instantiate
            **kwargs: Configuration parameters

        Returns:
            FlextResult[T]: Created and validated configuration instance

        """
        return FlextFactory.create_model(config_class, **kwargs)


# =============================================================================
# EXPORTS - Harmonized foundation patterns
# =============================================================================

__all__ = [
    "FlextConfig",
    # Semantic enums (4 items)
    "FlextConnectionType",
    "FlextDataFormat",
    "FlextEntity",
    # Factory patterns (1 item)
    "FlextFactory",
    "FlextLogLevel",
    # Foundation classes (4 items)
    "FlextModel",
    "FlextOperationStatus",
    # Semantic type system (1 item)
    "FlextTypes",
    "FlextValue",
]

# Total exports: 10 harmonized items (minimal, focused foundation)
