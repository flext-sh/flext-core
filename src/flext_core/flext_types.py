"""FLEXT Core Types - Foundation Layer Type System.

Comprehensive type system providing the foundation for type-safe operations across
all 32 projects in the FLEXT ecosystem. Establishes type contracts, protocols,
and generic programming patterns used throughout the architectural layers.

Module Role in Architecture:
    Foundation Layer → Type System → All Other Layers

    This module provides the fundamental type definitions that enable:
    - Railway-oriented programming with FlextResult[T]
    - Domain-driven design with typed entities and value objects
    - CQRS patterns with command/query type contracts
    - Dependency injection with type-safe service location
    - Protocol-based interfaces for architectural flexibility

Type System Categories:
    Generic Variables: T, U, V, R, E, P for comprehensive generic programming
    Domain Types: TEntity, TEntityId, TValue for domain modeling
    CQRS Types: TCommand, TQuery, TEvent for command-query separation
    Service Types: TService, TServiceName for dependency injection
    Data Types: TData, TMessage, TPayload for structured data handling

Development Status (v0.9.0 → 1.0.0):
    ✅ Production Ready: Core generics, domain types, protocol definitions
    🚧 Future Enhancement: Event sourcing types, advanced query patterns
    📋 TODO Integration: Plugin architecture types (Plugin Priority 3)

Protocol Design Patterns:
    - Structural typing for interface segregation
    - Runtime type checking capabilities
    - Validation contract definitions
    - Serialization/deserialization support
    - Timestamping and comparison protocols

Usage Across Ecosystem:
    # Type-safe result handling (used in 15,000+ function signatures)
    def process_data(data: TData) -> FlextResult[TEntity]:
        return FlextResult.ok(create_entity(data))

    # Domain modeling with typed entities
    class User(FlextEntity):
        user_id: TEntityId
        name: str

    # Service contracts with protocol typing
    class UserService(Protocol):
        def find_user(self, user_id: TEntityId) -> FlextResult[User]: ...

Quality Standards:
    - All types must be compatible with strict MyPy checking
    - Protocol definitions must support structural typing
    - Generic constraints must be properly bounded
    - Type aliases must be semantically meaningful
    - Runtime type guards must be provided where applicable

See Also:
    docs/python-module-organization.md: Type system architecture details
    docs/TODO.md: Type system enhancement roadmap

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Protocol, TypeVar

# =============================================================================
# CORE TYPE VARIABLES - Generic programming foundation
# =============================================================================

# Primary generic type variables
T = TypeVar("T")  # Generic type parameter
U = TypeVar("U")  # Secondary generic type parameter
V = TypeVar("V")  # Tertiary generic type parameter
R = TypeVar("R")  # Result type parameter
E = TypeVar("E")  # Error type parameter
P = TypeVar("P")  # Payload type parameter
F = TypeVar("F")  # Function type parameter

# =============================================================================
# DOMAIN TYPE ALIASES - Business domain modeling
# =============================================================================

# Entity and aggregate types
TEntity = TypeVar("TEntity")  # Generic entity type
TEntityId = str  # Entity identifier type
TService = TypeVar("TService")  # Generic service type
TAggregateRoot = TypeVar("TAggregateRoot")  # Aggregate root type

# Value object and data types
TValueObject = TypeVar("TValueObject")  # Generic value object type
TData = dict[str, object]  # Generic data dictionary
TValue = object  # Generic value type

# =============================================================================
# CQRS TYPE DEFINITIONS - Command Query Responsibility Segregation
# =============================================================================

# Command and query types
TCommand = TypeVar("TCommand")  # Generic command type
TQuery = TypeVar("TQuery")  # Generic query type
TEvent = TypeVar("TEvent")  # Generic event type

# Request and response types
TRequest = TypeVar("TRequest")  # Generic request type
TResponse = TypeVar("TResponse")  # Generic response type
TResult = TypeVar("TResult")  # Generic result type

# Message and handler types
TMessage = TypeVar("TMessage")  # Generic message type
THandler = Callable[[object], object]  # Generic handler function type
TRequestId = str  # Request identifier type

# Tracking and identification types
TCorrelationId = str  # Correlation identifier type
TUserId = str  # User identifier type

# Business domain types
TBusinessId = str  # Business entity identifier type
TBusinessName = str  # Business entity name type
TBusinessCode = str  # Business code type
TBusinessStatus = str  # Business status type
TBusinessType = str  # Business type classifier

# Cache and storage types
TCacheKey = str  # Cache key type
TCacheValue = object  # Cache value type
TCacheTTL = int  # Cache time-to-live type

# Configuration types
TConfigDict = dict[str, object]  # Configuration dictionary type
TConfigValue = object  # Configuration value type

# Infrastructure types
TConnectionString = str  # Database connection string type
TFilePath = str  # File path type
TDirectoryPath = str  # Directory path type

# =============================================================================
# UTILITY TYPE ALIASES - Common data structures
# =============================================================================

# Collection types
TAnyDict = dict[str, object]  # Dictionary with string keys and any values
TAnyList = list[object]  # List with any value types
TAnyMapping = dict[str, object]  # Generic mapping type alias
TAnySequence = list[object]  # Generic sequence type alias

# Function and predicate types
TPredicate = Callable[[object], bool]  # Predicate function type
TTransformer = Callable[[T], U]  # Transformer function type
TValidator = Callable[[object], bool]  # General validator function type
FlextValidator = TValidator  # Validator function type alias
TErrorHandler = Callable[[Exception], object]  # Error handler function type
TFactory = Callable[[], T]  # Generic factory function type

# Error and message types
TErrorCode = str  # Error code type
TErrorMessage = str  # Error message type

# Service and configuration types
TServiceName = str  # Service name type
TServiceKey = str | type[object]  # Service key type (string or type)

# Logging and context types
TLogMessage = str  # Log message type
TContextDict = dict[str, object]  # Logging context dictionary

# Common example types for MyPy compatibility
TAnyObject = dict[str, object]  # Any object dictionary type (replaces Any)
TFlexibleDict = dict[str, object]  # Flexible dictionary for examples
TUserData = dict[str, object]  # User data type for examples

# =============================================================================
# PROTOCOL DEFINITIONS - Structural typing interfaces
# =============================================================================


class Comparable(Protocol):
    """Protocol for comparable objects."""

    @abstractmethod
    def __lt__(self, other: object) -> bool:
        """Less than comparison."""

    @abstractmethod
    def __le__(self, other: object) -> bool:
        """Less than or equal comparison."""

    @abstractmethod
    def __gt__(self, other: object) -> bool:
        """Greater than comparison."""

    @abstractmethod
    def __ge__(self, other: object) -> bool:
        """Greater than or equal comparison."""


class FlextValidatable(Protocol):
    """Protocol for objects that can be validated."""

    @abstractmethod
    def validate(self) -> object:
        """Validate the object."""


class FlextSerializable(Protocol):
    """Protocol for objects that can be serialized."""

    @abstractmethod
    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary."""

    @abstractmethod
    def to_json(self) -> str:
        """Convert to JSON string."""


class Serializable(Protocol):
    """Protocol for serializable objects."""

    @abstractmethod
    def serialize(self) -> dict[str, object]:
        """Serialize to dictionary."""


class Timestamped(Protocol):
    """Protocol for objects with timestamps."""

    @abstractmethod
    def get_timestamp(self) -> object:
        """Get timestamp."""


class Validatable(Protocol):
    """Protocol for validatable objects."""

    @abstractmethod
    def is_valid(self) -> bool:
        """Check if object is valid."""


# =============================================================================
# FLEXT SPECIFIC TYPE ALIASES - Domain-specific definitions
# =============================================================================

# Entity identifier alias for backward compatibility
FlextEntityId = TEntityId

# =============================================================================
# TYPE GUARDS AND UTILITIES - Runtime type checking
# =============================================================================


class FlextTypes:
    """Unified type system providing organized access to all FLEXT type functionality.

    Comprehensive type management framework providing centralized access to type
    definitions, type guards, and type utilities. Serves as the primary namespace
    for type-related functionality with nested class organization.
    """

    class TypeGuards:
        """Type guard utilities for runtime type checking."""

        @staticmethod
        def is_instance_of(obj: object, expected_type: type[T]) -> bool:
            """Check if object is instance of expected type.

            Args:
                obj: Object to check
                expected_type: Expected type

            Returns:
                True if object is instance of expected type

            """
            try:
                return isinstance(obj, expected_type)
            except (TypeError, AttributeError):
                return False

        @staticmethod
        def is_callable(obj: object) -> bool:
            """Check if object is callable.

            Args:
                obj: Object to check

            Returns:
                True if object is callable

            """
            return callable(obj)

        @staticmethod
        def is_dict_like(obj: object) -> bool:
            """Check if object is dict-like.

            Args:
                obj: Object to check

            Returns:
                True if object has dict-like interface

            """
            return (
                hasattr(obj, "keys")
                and hasattr(obj, "values")
                and hasattr(obj, "items")
            )

        @staticmethod
        def is_list_like(obj: object) -> bool:
            """Check if object is list-like.

            Args:
                obj: Object to check

            Returns:
                True if object has list-like interface

            """
            return (
                hasattr(obj, "__iter__")
                and hasattr(obj, "__len__")
                and not isinstance(obj, str | bytes)
            )


# Export API - Alphabetically organized for ruff compliance
__all__ = [
    "Comparable",
    "E",
    "F",
    "FlextEntityId",
    "FlextSerializable",
    "FlextTypes",
    "FlextValidatable",
    "FlextValidator",
    "P",
    "R",
    "Serializable",
    "T",
    "TAggregateRoot",
    "TAnyDict",
    "TAnyList",
    "TAnyMapping",
    "TAnyObject",
    "TAnySequence",
    "TBusinessCode",
    "TBusinessId",
    "TBusinessName",
    "TBusinessStatus",
    "TBusinessType",
    "TCacheKey",
    "TCacheTTL",
    "TCacheValue",
    "TCommand",
    "TConfigDict",
    "TConfigValue",
    "TConnectionString",
    "TContextDict",
    "TCorrelationId",
    "TData",
    "TDirectoryPath",
    "TEntity",
    "TEntityId",
    "TErrorCode",
    "TErrorHandler",
    "TErrorMessage",
    "TEvent",
    "TFactory",
    "TFilePath",
    "TFlexibleDict",
    "THandler",
    "TLogMessage",
    "TMessage",
    "TPredicate",
    "TQuery",
    "TRequest",
    "TRequestId",
    "TResponse",
    "TResult",
    "TService",
    "TServiceKey",
    "TServiceName",
    "TTransformer",
    "TUserData",
    "TUserId",
    "TValidator",
    "TValue",
    "TValueObject",
    "Timestamped",
    "U",
    "V",
    "Validatable",
]
