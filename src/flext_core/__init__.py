"""FLEXT Core - Enterprise Foundation Library.

Comprehensive foundation library for the FLEXT ecosystem providing
Clean Architecture, Domain-Driven Design, and CQRS patterns with type-safe operations.

Architecture:
    - Clean Architecture with clear separation of concerns
    - Domain-Driven Design patterns for business logic modeling
    - CQRS implementation for command-query responsibility segregation
    - Railway-oriented programming for error handling
    - Type-safe operations with comprehensive generic support

Core Components:
    - FlextResult: Type-safe error handling with railway programming
    - FlextEntity: Domain entities with rich behavior and validation
    - FlextValueObject: Immutable value objects with validation
    - FlextAggregateRoot: DDD aggregate roots with event sourcing
    - FlextDomainService: Domain services for complex business logic
    - FlextTypes: Comprehensive type system with protocols and aliases

Enterprise Features:
    - Type safety through extensive generic programming
    - Error handling without exception propagation
    - Validation system with Pydantic integration
    - Logging system with structured context management
    - Dependency injection container with type-safe operations
    - Configuration management with environment variable support

Public API Categories:
    - Core patterns: FlextResult, type system, constants
    - DDD patterns: Entities, value objects, aggregates, domain services
    - Type system: Generic types, protocols, and domain-specific aliases
    - Constants: Environment settings, field types, log levels
    - Version information: Library version and authorship

Design Principles:
    - Single responsibility principle with focused components
    - Open/closed principle through extensible patterns
    - Liskov substitution with compatible implementations
    - Interface segregation through protocol-based design
    - Dependency inversion through abstraction dependencies

Usage Patterns:
    # Basic error handling
    from flext_core import FlextResult
    result = FlextResult.ok("success")

    # Domain modeling
    from flext_core import FlextEntity, TEntityId
    class User(FlextEntity):
        def __init__(self, user_id: TEntityId, name: str):
            super().__init__(user_id)
            self.name = name

    # Type-safe operations
    from flext_core import T, U
    def process_data[T, U](data: T, transformer: Callable[[T], U]) -> U:
        return transformer(data)

Dependencies:
    - pydantic: Data validation and configuration management
    - typing: Type system and generic programming support
    - Standard library: Core Python functionality

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.aggregate_root import FlextAggregateRoot
from flext_core.constants import (
    DEFAULT_TIMEOUT,
    EMAIL_PATTERN,
    ERROR_CODES,
    VERSION,
    FlextConstants,
    FlextEnvironment,
    FlextFieldType,
    FlextLogLevel,
)
from flext_core.domain_services import FlextDomainService
from flext_core.entities import FlextEntity, FlextEntityFactory
from flext_core.result import FlextResult
from flext_core.types import (
    Comparable,
    E,
    FlextTypes,
    Identifiable,
    P,
    R,
    Serializable,
    T,
    TAnyDict,
    TAnyList,
    TCommand,
    TData,
    TEntity,
    TEntityId,
    TErrorCode,
    TErrorMessage,
    TEvent,
    Timestamped,
    TMessage,
    TPredicate,
    TQuery,
    TRequest,
    TRequestId,
    TResponse,
    TResult,
    TService,
    TTransformer,
    TValue,
    U,
    V,
    Validatable,
)
from flext_core.value_objects import FlextValueObject, FlextValueObjectFactory

# Version information
__version__ = "0.8.0"
__author__ = "FLEXT Contributors"

# Clean public API - organized by category (T<Type> convention)
__all__ = [
    # Core constants
    "DEFAULT_TIMEOUT",
    "EMAIL_PATTERN",
    "ERROR_CODES",
    "VERSION",
    # Basic types (T<Type> convention)
    "Comparable",
    "E",
    # DDD patterns
    "FlextAggregateRoot",
    # Constants and configuration
    "FlextConstants",
    "FlextDomainService",
    "FlextEntity",
    "FlextEntityFactory",
    "FlextEnvironment",
    "FlextFieldType",
    "FlextLogLevel",
    # Core patterns
    "FlextResult",
    # Type system
    "FlextTypes",
    "FlextValueObject",
    "FlextValueObjectFactory",
    "Identifiable",
    "P",
    "R",
    "Serializable",
    "T",
    "TAnyDict",
    "TAnyList",
    "TCommand",
    "TData",
    "TEntity",
    "TEntityId",
    "TErrorCode",
    "TErrorMessage",
    "TEvent",
    "TMessage",
    "TPredicate",
    "TQuery",
    "TRequest",
    "TRequestId",
    "TResponse",
    "TResult",
    "TService",
    "TTransformer",
    "TValue",
    "Timestamped",
    "U",
    "V",
    "Validatable",
    # Version info
    "__version__",
]
