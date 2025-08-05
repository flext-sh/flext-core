"""FLEXT Core Types - Legacy Type System (Unified Pattern Migration).

UNIFIED PATTERN MIGRATION: This module provides legacy type definitions
and is being migrated to the FLEXT Unified Semantic Patterns system.

New Development: Use unified semantic patterns from semantic_types.py
    from flext_core.semantic_types import FlextTypes
    # Unified semantic usage
    predicate: FlextTypes.Core.Predicate[User] = lambda u: u.is_active
    connection: FlextTypes.Data.Connection = get_oracle_connection()

Legacy Support: This module maintains backward compatibility
    from flext_core.types import FlextTypes  # Legacy hierarchical types
    from flext_core.types import TAnyDict, TFactory  # Legacy flat aliases

Unified Migration Timeline:
    - Phase 1 (Current): Harmonization complete - unified patterns active
    - Phase 2 (Week 2-3): Project-by-project migration to unified system
    - Phase 3 (Week 4-5): Deprecation warnings for legacy usage
    - Phase 4 (Week 6): Complete migration to unified patterns only

For complete unified semantic patterns, see:
/home/marlonsc/flext/flext-core/docs/FLEXT_UNIFIED_SEMANTIC_PATTERNS.md

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import (
    Protocol,
    TypeGuard,
    TypeVar,
)

# =============================================================================
# CORE TYPE VARIABLES - Foundation Generic Programming
# =============================================================================

# Primary generics for universal usage
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
R = TypeVar("R")  # Result type
E = TypeVar("E")  # Error type
P = TypeVar("P")  # Payload type
F = TypeVar("F")  # Function type

# Constrained generics for specific patterns
TComparable = TypeVar("TComparable", bound="FlextTypes.Protocols.Comparable")
TSerializable = TypeVar("TSerializable", bound="FlextTypes.Protocols.Serializable")
TValidatable = TypeVar("TValidatable", bound="FlextTypes.Protocols.Validatable")


# =============================================================================
# MAIN TYPE SYSTEM - Hierarchical Organization
# =============================================================================


class FlextTypes:
    """Centralized type system with semantic hierarchical organization.

    Provides structured access to all FLEXT type definitions through nested
    classes that group related types by domain, purpose, and usage patterns.
    This approach minimizes namespace pollution while maximizing discoverability.
    """

    # =========================================================================
    # CORE TYPES - Fundamental programming constructs
    # =========================================================================

    class Core:
        """Core programming types used throughout the ecosystem."""

        # Result and error handling types
        Result = TypeVar("Result")  # FlextResult[T] generic
        Success = TypeVar("Success")  # Success value type
        Failure = TypeVar("Failure")  # Failure value type

        # Function composition types
        type Predicate[T] = Callable[[T], bool]
        type Transform[T, U] = Callable[[T], U]
        type Factory[T] = Callable[[], T] | Callable[[object], T]
        type Validator[T] = Callable[[T], bool]
        type Handler[T, R] = Callable[[T], R]

        # Container and dependency injection types
        type ServiceKey = str | type[object]
        type ServiceName = str
        ServiceInstance = TypeVar("ServiceInstance")

        # Context and correlation types
        type CorrelationId = str
        type RequestId = str
        type TraceId = str

    # =========================================================================
    # DOMAIN TYPES - Business domain modeling
    # =========================================================================

    class Domain:
        """Domain-Driven Design types for business modeling."""

        # Entity and aggregate types
        Entity = TypeVar("Entity")
        type EntityId = str
        AggregateRoot = TypeVar("AggregateRoot")
        ValueObject = TypeVar("ValueObject")

        # Domain service types
        DomainService = TypeVar("DomainService")
        Repository = TypeVar("Repository")

        # Event and messaging types
        DomainEvent = TypeVar("DomainEvent")
        type EventId = str
        type EventType = str
        type EventPayload = dict[str, object]

        # Business identifier types
        type BusinessId = str
        type BusinessCode = str
        type BusinessName = str
        type BusinessType = str
        type BusinessStatus = str

        # User and authentication types
        type UserId = str
        type UserRole = str
        type Permission = str

    # =========================================================================
    # CQRS TYPES - Command Query Responsibility Segregation
    # =========================================================================

    class CQRS:
        """CQRS pattern types for command-query separation."""

        # Command types
        Command = TypeVar("Command")
        CommandHandler = TypeVar("CommandHandler")
        CommandResult = TypeVar("CommandResult")

        # Query types
        Query = TypeVar("Query")
        QueryHandler = TypeVar("QueryHandler")
        QueryResult = TypeVar("QueryResult")

        # Message types
        Message = TypeVar("Message")
        type MessageHandler[T, R] = Callable[[T], R]

        # Request/response types
        Request = TypeVar("Request")
        Response = TypeVar("Response")

    # =========================================================================
    # DATA TYPES - Structured data handling
    # =========================================================================

    class Data:
        """Data structure and serialization types."""

        # Basic data structures
        type Dict = dict[str, object]
        type FlexibleDict = dict[str, object]
        type ConfigDict = dict[str, str | int | float | bool | None]

        # Value types
        type Value = str | int | float | bool | None
        type ConfigValue = str | int | float | bool | None

        # Collection types
        type List = list[object]
        type FlextSequence = Sequence[object]
        type FlextMapping = Mapping[str, object]

        # Field and metadata types
        type FieldValue = str | int | float | bool | None
        type FieldMetadata = dict[str, object]
        type FieldInfo = dict[str, object]

        # Serialization types
        type JsonData = dict[str, object]
        type SerializedData = str | bytes

    # =========================================================================
    # INFRASTRUCTURE TYPES - System and technical types
    # =========================================================================

    class Infrastructure:
        """Infrastructure and system-level types."""

        # File and path types
        type FilePath = str
        type DirectoryPath = str
        type URL = str

        # Connection types
        type ConnectionString = str
        type DatabaseURL = str

        # Cache types
        type CacheKey = str
        type CacheValue = object
        type CacheTTL = int

        # Logging types
        type LogLevel = str
        type LogMessage = str
        type LogContext = dict[str, object]

        # Error types
        type ErrorCode = str
        type ErrorMessage = str
        type ErrorContext = dict[str, object]

        # Configuration types
        type EnvVar = str
        type ConfigSection = str

    # =========================================================================
    # SINGER TYPES - Singer ecosystem integration
    # =========================================================================

    class Singer:
        """Singer specification and data pipeline types."""

        # Stream and schema types
        type StreamName = str
        type SchemaName = str
        type TableName = str

        # Record types
        type Record = dict[str, object]
        type RecordId = str

        # Tap and target types
        type TapConfig = dict[str, object]
        type TargetConfig = dict[str, object]

        # State and bookmark types
        type State = dict[str, object]
        type Bookmark = dict[str, object]

        # Catalog types
        type Catalog = dict[str, object]
        type Stream = dict[str, object]
        type Schema = dict[str, object]

    # =========================================================================
    # PROTOCOL DEFINITIONS - Structural typing interfaces
    # =========================================================================

    class Protocols:
        """Protocol definitions for structural typing."""

        class Comparable(Protocol):
            """Protocol for comparable objects."""

            @abstractmethod
            def __lt__(self, other: object) -> bool:
                """Less than comparison."""
                ...

            @abstractmethod
            def __le__(self, other: object) -> bool:
                """Less than or equal comparison."""
                ...

            @abstractmethod
            def __gt__(self, other: object) -> bool:
                """Greater than comparison."""
                ...

            @abstractmethod
            def __ge__(self, other: object) -> bool:
                """Greater than or equal comparison."""
                ...

        class Serializable(Protocol):
            """Protocol for serializable objects."""

            @abstractmethod
            def to_dict(self) -> dict[str, object]:
                """Convert to dictionary."""
                ...

            @abstractmethod
            def to_json(self) -> str:
                """Convert to JSON string."""
                ...

        class Validatable(Protocol):
            """Protocol for validatable objects."""

            @abstractmethod
            def validate(self) -> object:
                """Validate the object."""
                ...

            @abstractmethod
            def is_valid(self) -> bool:
                """Check if object is valid."""
                ...

        class Timestamped(Protocol):
            """Protocol for timestamped objects."""

            @abstractmethod
            def get_timestamp(self) -> float:
                """Get timestamp."""
                ...

            @abstractmethod
            def update_timestamp(self) -> None:
                """Update timestamp."""
                ...

        class Cacheable(Protocol):
            """Protocol for cacheable objects."""

            @abstractmethod
            def cache_key(self) -> str:
                """Get cache key."""
                ...

            @abstractmethod
            def cache_ttl(self) -> int:
                """Get cache TTL."""
                ...

        class Configurable(Protocol):
            """Protocol for configurable objects."""

            @abstractmethod
            def configure(self, config: dict[str, object]) -> object:
                """Configure the object."""
                ...

            @abstractmethod
            def get_config(self) -> dict[str, object]:
                """Get configuration."""
                ...

        # Function protocols
        type Validator[T] = Callable[[T], bool]
        type Transformer[T, U] = Callable[[T], U]
        type ErrorHandler = Callable[[Exception], str]
        type EventHandler = Callable[[object], None]

    # =========================================================================
    # TYPE GUARDS - Runtime type checking utilities
    # =========================================================================

    class Guards:
        """Type guard utilities for runtime type checking."""

        @staticmethod
        def is_instance_of(obj: object, expected_type: type[T]) -> TypeGuard[T]:
            """Type guard for instance checking."""
            return isinstance(obj, expected_type)

        @staticmethod
        def is_callable(obj: object) -> TypeGuard[Callable[..., object]]:
            """Type guard for callable objects."""
            return callable(obj)

        @staticmethod
        def is_dict_like(obj: object) -> TypeGuard[Mapping[str, object]]:
            """Type guard for dict-like objects."""
            return (
                hasattr(obj, "keys")
                and hasattr(obj, "values")
                and hasattr(obj, "items")
            )

        @staticmethod
        def is_sequence_like(obj: object) -> TypeGuard[Sequence[object]]:
            """Type guard for sequence-like objects."""
            return (
                hasattr(obj, "__iter__")
                and hasattr(obj, "__len__")
                and not isinstance(obj, str | bytes)
            )

        @staticmethod
        def is_non_empty_string(obj: object) -> TypeGuard[str]:
            """Type guard for non-empty strings."""
            return isinstance(obj, str) and len(obj) > 0

        @staticmethod
        def is_positive_int(obj: object) -> TypeGuard[int]:
            """Type guard for positive integers."""
            return isinstance(obj, int) and obj > 0


# =============================================================================
# COMPATIBILITY ALIASES - Migration support
# =============================================================================


class FlextTypesCompat:
    """Compatibility aliases for migration from flext_types.py.

    Provides temporary aliases to ease migration from the old flat type system.
    These will be deprecated and removed in a future version.
    """

    # Core type variables (maintain existing names)
    T = T
    U = U
    V = V
    R = R
    E = E
    P = P
    F = F

    # Legacy domain aliases
    TEntity = FlextTypes.Domain.Entity
    TEntityId = FlextTypes.Domain.EntityId
    TAggregateRoot = FlextTypes.Domain.AggregateRoot
    TValueObject = FlextTypes.Domain.ValueObject

    # Legacy CQRS aliases
    TCommand = FlextTypes.CQRS.Command
    TQuery = FlextTypes.CQRS.Query
    TEvent = FlextTypes.Domain.DomainEvent
    TMessage = FlextTypes.CQRS.Message
    TRequest = FlextTypes.CQRS.Request
    TResponse = FlextTypes.CQRS.Response

    # Legacy data aliases
    TData = FlextTypes.Data.Dict
    TValue = FlextTypes.Data.Value
    TAnyDict = FlextTypes.Data.Dict
    TConfigDict = FlextTypes.Data.ConfigDict

    # Legacy service aliases
    TService = FlextTypes.Core.ServiceInstance
    TServiceName = FlextTypes.Core.ServiceName
    TServiceKey = FlextTypes.Core.ServiceKey

    # Legacy infrastructure aliases
    TFilePath = FlextTypes.Infrastructure.FilePath
    TConnectionString = FlextTypes.Infrastructure.ConnectionString
    TCacheKey = FlextTypes.Infrastructure.CacheKey
    TCacheValue = FlextTypes.Infrastructure.CacheValue

    # Legacy function aliases
    TPredicate = FlextTypes.Core.Predicate
    TValidator = FlextTypes.Core.Validator
    TTransformer = FlextTypes.Core.Transform
    TFactory = FlextTypes.Core.Factory

    # Legacy business aliases
    TBusinessId = FlextTypes.Domain.BusinessId
    TBusinessCode = FlextTypes.Domain.BusinessCode
    TBusinessName = FlextTypes.Domain.BusinessName
    TUserId = FlextTypes.Domain.UserId

    # Legacy context aliases
    TCorrelationId = FlextTypes.Core.CorrelationId
    TRequestId = FlextTypes.Core.RequestId

    # Legacy protocol aliases
    Comparable = FlextTypes.Protocols.Comparable
    Serializable = FlextTypes.Protocols.Serializable
    Validatable = FlextTypes.Protocols.Validatable
    Timestamped = FlextTypes.Protocols.Timestamped


# =============================================================================
# CONTROLLED EXPORTS - Minimal namespace pollution
# =============================================================================

__all__ = [
    "E",
    "F",
    "FlextTypes",
    "FlextTypesCompat",
    "P",
    "R",
    "T",
    "TComparable",
    "TSerializable",
    "TValidatable",
    "U",
    "V",
]

# =============================================================================
# MIGRATION HELPERS - Temporary compatibility functions
# =============================================================================


def migrate_from_flext_types() -> dict[str, object]:
    """Helper function to map old flext_types imports to new structure.

    Returns:
        Dictionary mapping old names to new FlextTypes references

    Usage:
        from flext_core.types import migrate_from_flext_types
        type_mapping = migrate_from_flext_types()
        TEntityId = type_mapping["TEntityId"]  # Gets FlextTypes.Domain.EntityId

    """
    return {
        # Core generics
        "T": T,
        "U": U,
        "V": V,
        "R": R,
        "E": E,
        "P": P,
        "F": F,
        # Domain types
        "TEntity": FlextTypes.Domain.Entity,
        "TEntityId": FlextTypes.Domain.EntityId,
        "TAggregateRoot": FlextTypes.Domain.AggregateRoot,
        "TValueObject": FlextTypes.Domain.ValueObject,
        # CQRS types
        "TCommand": FlextTypes.CQRS.Command,
        "TQuery": FlextTypes.CQRS.Query,
        "TEvent": FlextTypes.Domain.DomainEvent,
        "TMessage": FlextTypes.CQRS.Message,
        # Data types
        "TData": FlextTypes.Data.Dict,
        "TValue": FlextTypes.Data.Value,
        "TAnyDict": FlextTypes.Data.Dict,
        # Service types
        "TService": FlextTypes.Core.ServiceInstance,
        "TServiceName": FlextTypes.Core.ServiceName,
        # Function types
        "TPredicate": FlextTypes.Core.Predicate,
        "TValidator": FlextTypes.Core.Validator,
        "TTransformer": FlextTypes.Core.Transform,
        # Protocol types
        "Comparable": FlextTypes.Protocols.Comparable,
        "Serializable": FlextTypes.Protocols.Serializable,
        "Validatable": FlextTypes.Protocols.Validatable,
    }
