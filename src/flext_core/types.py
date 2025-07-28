"""FLEXT Core Type System Module.

Comprehensive type definitions for the FLEXT Core library implementing enterprise-grade
type safety, protocol-based design patterns, and structural typing for maximum
flexibility and compile-time verification.

Architecture:
    - Protocol-based design for structural typing and duck typing support
    - Generic TypeVar definitions for type-safe generic programming patterns
    - Domain-specific type aliases for business clarity and semantic meaning
    - Consolidated type system eliminating base module duplication
    - No underscore prefixes on public types for clean API access
    - Runtime type checking integration with static analysis support

Type System Components:
    - Core type variables: T, U, V, R, E, P, F for generic programming foundations
    - Domain type variables: TEntity, TValue, TService for business logic typing
    - CQRS type variables: TCommand, TQuery, TEvent for architectural patterns
    - Protocol definitions: Structural typing interfaces for flexible duck typing
    - Functional type aliases: TPredicate, TTransformer, THandler for function
      signatures
    - Business type aliases: TEntityId, TBusinessCode, TCorrelationId for domain clarity

Maintenance Guidelines:
    - Add new protocols to support duck typing and structural interfaces
    - Keep type aliases descriptive and semantically meaningful for business domains
    - Use TypeVar bounds for constraint specification and type safety
    - Document protocol methods with proper signatures and usage examples
    - Group related types in logical sections for maintainability and discovery
    - Maintain backward compatibility through legacy type aliases
    - Follow naming conventions with T prefix for type aliases

Design Decisions:
    - Protocol over ABC for structural typing and maximum flexibility
    - Generic TypeVars for maximum type safety and reusability
    - Domain-specific aliases for business clarity and semantic understanding
    - Eliminated _types_base.py following "deliver more with much less" principle
    - Runtime type checking support through integration with utilities module
    - Extensive type coverage for all system domains and architectural patterns

Enterprise Features:
    - Comprehensive protocol definitions for structural typing across system components
    - Type-safe functional programming support through typed callable aliases
    - Domain-driven design support through business-specific type definitions
    - CQRS pattern support through command, query, and event type variables
    - Extensive business domain coverage including security, networking, and database
      types
    - Legacy compatibility through backward-compatible type aliases
    - Integration with static type checkers for compile-time verification

Protocol-Based Design:
    - FlextValidatable: Objects supporting validation operations with status reporting
    - FlextSerializable: Objects supporting dictionary serialization for transport
    - FlextIdentifiable: Objects with unique identification for tracking and lookup
    - FlextTimestamped: Objects with timestamp tracking for audit and temporal queries
    - FlextCacheable: Objects supporting caching operations for performance optimization
    - FlextConfigurable: Objects supporting configuration management for dependency
      injection

Type Safety Features:
    - Generic type variables with proper bounds for constraint enforcement
    - Protocol-based interfaces for structural typing without inheritance requirements
    - Domain-specific type aliases preventing type confusion and enhancing readability
    - Runtime type checking integration for dynamic validation scenarios
    - Static analysis support through TypeGuard patterns and proper type annotations
    - Comprehensive type coverage eliminating Any usage throughout the system

Dependencies:
    - typing: Core typing infrastructure and generic programming support
    - collections.abc: Abstract base classes for collection and callable interfaces
    - datetime: Time-related type definitions for temporal data management
    - flext_core.result: FlextResult type integration for error handling patterns

Warning:
    This module shares name with Python stdlib 'types' module.
    Always use explicit imports to avoid namespace conflicts and import errors.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Protocol, TypeVar

# =============================================================================
# CORE TYPE VARIABLES - sem underscore conforme diretrizes
# =============================================================================

# Core type variables - foundation for entire FLEXT ecosystem
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
R = TypeVar("R")
E = TypeVar("E", bound=Exception)
P = TypeVar("P")
F = TypeVar("F")  # Function type variable for decorators

# Domain-specific type variables (mantém prefixo T para clareza de domínio)
TEntity = TypeVar("TEntity")
TValue = TypeVar("TValue")
TService = TypeVar("TService")

# Command/Query type variables (mantém prefixo T para tipos CQRS)
TCommand = TypeVar("TCommand")
TData = TypeVar("TData")
TEvent = TypeVar("TEvent")
TMessage = TypeVar("TMessage")
TRequest = TypeVar("TRequest")
TResponse = TypeVar("TResponse")
TResult = TypeVar("TResult")
TQuery = TypeVar("TQuery")

# =============================================================================
# BASIC TYPE ALIASES - sem underscore conforme diretrizes
# =============================================================================

# Basic type aliases (mantém prefixo T para evitar conflitos)
TAnyDict = dict[str, object]
TAnyList = list[object]
TAnyMapping = Mapping[str, object]
TAnySequence = Sequence[object]

# Strong type aliases for domain (mantém prefixo T para IDs e strings tipadas)
TEntityId = str
TRequestId = str
TSessionId = str
TUserId = str
TFieldId = str
TFieldName = str
TErrorCode = str
TErrorMessage = str
TLogMessage = str
TContextDict = dict[str, object]
TServiceName = str
TCorrelationId = str
TOperationStatus = str
TValidationStatus = str

# Time types (mantém prefixo T para tipos temporais)
TTimestamp = datetime
TTimestampStr = str

# Functional types (mantém prefixo T para tipos funcionais)
TPredicate = Callable[[object], bool]
TTransformer = Callable[[object], object]
TValidator = Callable[[object], bool]
TFactory = Callable[[], T]
TBuilder = Callable[[object], T]
TSerializer = Callable[[object], dict[str, object]]
TDeserializer = Callable[[dict[str, object]], object]
THandler = Callable[[object], object]
TProcessor = Callable[[object], object]
TMapper = Callable[[T], U]
TFilter = Callable[[T], bool]
TReducer = Callable[[T, T], T]

# Error handling types
TErrorHandler = Callable[[Exception], object]
TExceptionMapper = Callable[[Exception], TErrorMessage]

# Configuration types
TConfigValue = str | int | float | bool | None
TConfigDict = dict[str, TConfigValue]
TSettingsDict = dict[str, object]

# Logging types
TLogLevel = str
TLogContext = dict[str, object]
TLogEntry = dict[str, object]

# Database types
TConnectionString = str
TQueryString = str
TTableName = str
TColumnName = str
TIndexName = str

# HTTP/API types
TUrl = str
THttpMethod = str
THttpHeaders = dict[str, str]
THttpStatus = int
TApiKey = str
TToken = str

# File/Path types
TFilePath = str
TFileName = str
TFileExtension = str
TDirectoryPath = str

# Cache types
TCacheKey = str
TCacheValue = object
TCacheTTL = int

# Container/DI types
TServiceKey = str
TServiceInstance = object
TDependency = object

# Metric/Monitoring types
TMetricName = str
TMetricValue = float
TMetricTags = dict[str, str]

# Business domain types
TBusinessId = str
TBusinessName = str
TBusinessCode = str
TBusinessStatus = str
TBusinessType = str

# Content types
TContentType = str
TEncoding = str
THashValue = str
TChecksum = str

# Network types
THostname = str
TPort = int
TIpAddress = str
TMacAddress = str

# Security types
TSecretKey = str
TPublicKey = str
TPrivateKey = str
TCertificate = str

# =============================================================================
# PROTOCOLS - sem underscore conforme diretrizes
# =============================================================================


class FlextValidatable(Protocol):
    """Protocol for objects that support validation operations.

    Defines the interface for objects that can validate their internal state
    and report validation status. Used throughout the system for consistent
    validation patterns and duck typing.

    Protocol Methods:
        validate() -> bool: Perform validation and return success status

    Usage:
        def process_validatable(obj: FlextValidatable) -> bool:
            return obj.validate()

        class User:
            def validate(self) -> bool:
                return self.email is not None
    """

    def validate(self) -> bool:
        """Validate object state and return validation result."""
        ...


class FlextSerializable(Protocol):
    """Protocol for objects that support dictionary serialization.

    Defines the interface for objects that can convert themselves to
    dictionary representations for storage, transmission, or API responses.
    Supports structured data exchange and persistence patterns.

    Protocol Methods:
        to_dict() -> dict[str, object]: Convert to dictionary representation

    Usage:
        def serialize_object(obj: FlextSerializable) -> dict[str, object]:
            return obj.to_dict()

        class UserModel:
            def to_dict(self) -> dict[str, object]:
                return {"id": self.id, "name": self.name}
    """

    def to_dict(self) -> dict[str, object]:
        """Convert object to dictionary representation."""
        ...


class FlextIdentifiable(Protocol):
    """Protocol for objects with unique identification capabilities.

    Defines the interface for objects that provide unique identifiers
    for tracking, caching, and referential integrity. Used throughout
    the system for entity management and lookup operations.

    Protocol Properties:
        id -> TEntityId: Unique identifier for the object

    Usage:
        def lookup_by_id(obj: FlextIdentifiable, registry: dict) -> object:
            return registry.get(obj.id)

        class Entity:
            @property
            def id(self) -> str:
                return self._id
    """

    @property
    def id(self) -> TEntityId:
        """Get unique identifier for this object."""
        ...


class FlextTimestamped(Protocol):
    """Protocol for objects with timestamp tracking capabilities.

    Defines the interface for objects that track creation and modification
    timestamps for audit trails, cache invalidation, and temporal queries.
    Essential for enterprise data management and tracking.

    Protocol Properties:
        created_at -> TTimestamp: Object creation timestamp
        updated_at -> TTimestamp: Last modification timestamp

    Usage:
        def check_freshness(obj: FlextTimestamped, max_age: int) -> bool:
            age = datetime.now() - obj.updated_at
            return age.seconds < max_age

        class Document:
            @property
            def created_at(self) -> datetime:
                return self._created_at
    """

    @property
    def created_at(self) -> TTimestamp:
        """Get creation timestamp."""
        ...

    @property
    def updated_at(self) -> TTimestamp:
        """Get last update timestamp."""
        ...


class FlextCacheable(Protocol):
    """Protocol for objects that support caching operations.

    Defines the interface for objects that can generate cache keys for
    efficient storage and retrieval. Supports performance optimization
    through intelligent caching strategies.

    Protocol Methods:
        cache_key() -> TCacheKey: Generate unique cache key

    Usage:
        def cache_object(obj: FlextCacheable, cache: dict) -> None:
            cache[obj.cache_key()] = obj

        class QueryResult:
            def cache_key(self) -> str:
                return f"query_{self.query_hash}_{self.params_hash}"
    """

    def cache_key(self) -> TCacheKey:
        """Generate cache key for this object."""
        ...


class FlextConfigurable(Protocol):
    """Protocol for objects that support configuration management.

    Defines the interface for objects that can be configured through
    dictionary-based configuration data. Supports dynamic configuration
    and dependency injection patterns.

    Protocol Methods:
        configure(config: TConfigDict) -> None: Apply configuration settings

    Usage:
        def setup_service(service: FlextConfigurable, config: dict) -> None:
            service.configure(config)

        class DatabaseService:
            def configure(self, config: dict) -> None:
                self.connection_string = config["connection_string"]
    """

    def configure(self, config: TConfigDict) -> None:
        """Configure object with provided configuration dictionary."""
        ...


class FlextExecutable(Protocol):
    """Protocol for executable objects."""

    def execute(self, *args: object, **kwargs: object) -> object:
        """Execute operation with provided arguments."""
        ...


class FlextValidator(Protocol):
    """Protocol for validator objects."""

    def validate(self, value: object) -> bool:
        """Validate provided value and return validation result."""
        ...


class FlextTransformer(Protocol):
    """Protocol for transformer objects."""

    def transform(self, value: object) -> object:
        """Transform provided value and return transformed result."""
        ...


class FlextHandler(Protocol):
    """Protocol for handler objects."""

    def handle(self, event: object) -> object:
        """Handle provided event and return handling result."""
        ...


class Comparable(Protocol):
    """Protocol for comparable objects."""

    def __lt__(self, other: object) -> bool:
        """Return True if self is less than other."""
        ...

    def __le__(self, other: object) -> bool:
        """Return True if self is less than or equal to other."""
        ...

    def __gt__(self, other: object) -> bool:
        """Return True if self is greater than other."""
        ...

    def __ge__(self, other: object) -> bool:
        """Return True if self is greater than or equal to other."""
        ...


# =============================================================================
# TYPE CHECKING ALIASES - consolidado sem underscore
# =============================================================================

# Type checking specific aliases (quando TYPE_CHECKING)
# FlextResult is defined in result.py module

# =============================================================================
# MAIN TYPES CLASS - consolidado sem base
# =============================================================================


class FlextTypes:
    """Consolidated type system providing single source of truth for FLEXT Core types.

    Central repository for all type definitions, type variables, protocols, and type
    utilities. Eliminates base type modules following the "deliver more with much less"
    principle while maintaining comprehensive type safety.

    Architecture:
        - Single source of truth for all type definitions
        - No underscore prefixes on public objects for clean API
        - Protocol-based design for structural typing
        - Generic type variables for maximum flexibility
        - Domain-specific type aliases for business clarity

    Type Categories:
        - Core type variables: T, U, V, R, E, P, F for generic programming
        - Domain type variables: TEntity, TValue, TService for business logic
        - CQRS type variables: TCommand, TQuery, TEvent for architectural patterns
        - Protocol definitions: FlextValidatable, FlextSerializable, etc.
        - Type aliases: TAnyDict, TEntityId, TErrorMessage for specific domains

    Nested Classes:
        - TypeGuards: Runtime type checking utilities
        - Meta: Type system metadata and version information

    Protocol Support:
        - Structural typing through Protocol definitions
        - Duck typing support for flexible interfaces
        - Type checking integration for static analysis
        - Runtime type validation capabilities

    Usage Patterns:
        # Generic programming
        def process_data[T](data: T) -> FlextResult[T]:
            return FlextResult.ok(data)

        # Protocol-based design
        def validate_object(obj: FlextValidatable) -> bool:
            return obj.validate()

        # Domain-specific types
        def create_entity(entity_id: TEntityId) -> TEntity:
            return Entity(id=entity_id)

        # Type guards
        if FlextTypes.TypeGuards.is_instance_of(obj, UserService):
            user_service: UserService = obj
    """

    # Type guards access via nested class
    class TypeGuards:
        """Runtime type checking utilities with static analysis support.

        Provides access to type guard utilities for runtime type validation
        with integration to static type checkers. Supports type narrowing
        and runtime safety patterns throughout the system.

        Type Guard Features:
            - Runtime type validation with isinstance checking
            - Static analysis support through TypeGuard return types
            - Integration with utilities module for comprehensive checking
            - Type narrowing support for conditional type checking

        Usage:
            if FlextTypes.TypeGuards.is_instance_of(service, UserService):
                # service is now typed as UserService
                user_service: UserService = service
        """

        @staticmethod
        def is_instance_of(obj: object, target_type: type) -> bool:
            """Check if object is instance of target type."""
            from flext_core.utilities import FlextTypeGuards

            return FlextTypeGuards.is_instance_of(obj, target_type)

    # Type variables
    T = T
    U = U
    V = V
    R = R
    E = E
    P = P
    F = F

    # Domain variables
    TEntity = TEntity
    TValue = TValue
    TService = TService
    TCommand = TCommand
    TData = TData
    TEvent = TEvent
    TMessage = TMessage
    TRequest = TRequest
    TResponse = TResponse
    TResult = TResult
    TQuery = TQuery

    # Basic aliases
    TAnyDict = TAnyDict
    TAnyList = TAnyList
    TAnyMapping = TAnyMapping
    TAnySequence = TAnySequence

    # Domain types
    TEntityId = TEntityId
    TRequestId = TRequestId
    TSessionId = TSessionId
    TUserId = TUserId
    TErrorCode = TErrorCode
    TErrorMessage = TErrorMessage
    TLogMessage = TLogMessage
    TContextDict = TContextDict

    # Functional types
    TPredicate = TPredicate
    TTransformer = TTransformer
    TValidator = TValidator
    TFactory = TFactory
    TBuilder = TBuilder
    THandler = THandler
    TProcessor = TProcessor
    TMapper = TMapper
    TFilter = TFilter

    # Configuration types
    TConfigValue = TConfigValue
    TConfigDict = TConfigDict
    TSettingsDict = TSettingsDict

    # Network types
    TUrl = TUrl
    THttpMethod = THttpMethod
    THttpHeaders = THttpHeaders
    THttpStatus = THttpStatus
    THostname = THostname
    TPort = TPort
    TIpAddress = TIpAddress

    # File types
    TFilePath = TFilePath
    TFileName = TFileName
    TDirectoryPath = TDirectoryPath

    # Database types
    TConnectionString = TConnectionString
    TQueryString = TQueryString
    TTableName = TTableName
    TColumnName = TColumnName

    # Business types
    TBusinessId = TBusinessId
    TBusinessName = TBusinessName
    TBusinessCode = TBusinessCode
    TBusinessStatus = TBusinessStatus

    class Meta:
        """Type system metadata and version information.

        Provides comprehensive information about the type system including
        version compatibility, type count statistics, and system requirements.

        Metadata Categories:
            - Version information for compatibility tracking
            - Type count statistics for system complexity metrics
            - Python version requirements for deployment planning
            - Compatibility information for integration planning
        """

        TYPE_COUNT = 100  # Approximate count of defined types
        VERSION = "1.0.0"
        COMPATIBILITY = "Python 3.13+"


# =============================================================================
# LEGACY COMPATIBILITY ALIASES - mantém interface existente
# =============================================================================

# For backward compatibility with existing code
FlextEntityId = TEntityId
FlextUserId = TUserId
FlextErrorCode = TErrorCode
FlextErrorMessage = TErrorMessage
FlextMessageType = TMessage

# Protocol aliases for legacy compatibility
Identifiable = FlextIdentifiable
Serializable = FlextSerializable
Timestamped = FlextTimestamped
Validatable = FlextValidatable

# =============================================================================
# EXPORTS - Clean public API seguindo diretrizes
# =============================================================================

__all__ = [
    "Comparable",
    "E",
    "F",
    "FlextCacheable",
    "FlextConfigurable",
    "FlextEntityId",
    "FlextErrorCode",
    "FlextErrorMessage",
    "FlextExecutable",
    "FlextHandler",
    "FlextIdentifiable",
    "FlextMessageType",
    "FlextSerializable",
    "FlextTimestamped",
    "FlextTransformer",
    "FlextTypes",
    "FlextUserId",
    "FlextValidatable",
    "FlextValidator",
    "Identifiable",
    "P",
    "R",
    "Serializable",
    "T",
    "TAnyDict",
    "TAnyList",
    "TAnyMapping",
    "TAnySequence",
    "TApiKey",
    "TBuilder",
    "TBusinessCode",
    "TBusinessId",
    "TBusinessName",
    "TBusinessStatus",
    "TBusinessType",
    "TCacheKey",
    "TCacheTTL",
    "TCacheValue",
    "TCertificate",
    "TChecksum",
    "TColumnName",
    "TCommand",
    "TConfigDict",
    "TConfigValue",
    "TConnectionString",
    "TContentType",
    "TContextDict",
    "TCorrelationId",
    "TData",
    "TDependency",
    "TDeserializer",
    "TDirectoryPath",
    "TEncoding",
    "TEntity",
    "TEntityId",
    "TErrorCode",
    "TErrorHandler",
    "TErrorMessage",
    "TEvent",
    "TExceptionMapper",
    "TFactory",
    "TFieldId",
    "TFieldName",
    "TFileExtension",
    "TFileName",
    "TFilePath",
    "TFilter",
    "THandler",
    "THashValue",
    "THostname",
    "THttpHeaders",
    "THttpMethod",
    "THttpStatus",
    "TIndexName",
    "TIpAddress",
    "TLogContext",
    "TLogEntry",
    "TLogLevel",
    "TLogMessage",
    "TMacAddress",
    "TMapper",
    "TMessage",
    "TMetricName",
    "TMetricTags",
    "TMetricValue",
    "TOperationStatus",
    "TPort",
    "TPredicate",
    "TPrivateKey",
    "TProcessor",
    "TPublicKey",
    "TQuery",
    "TQueryString",
    "TReducer",
    "TRequest",
    "TRequestId",
    "TResponse",
    "TResult",
    "TSecretKey",
    "TSerializer",
    "TService",
    "TServiceInstance",
    "TServiceKey",
    "TServiceName",
    "TSessionId",
    "TSettingsDict",
    "TTableName",
    "TTimestamp",
    "TTimestampStr",
    "TToken",
    "TTransformer",
    "TUrl",
    "TUserId",
    "TValidationStatus",
    "TValidator",
    "TValue",
    "Timestamped",
    "U",
    "V",
    "Validatable",
]
