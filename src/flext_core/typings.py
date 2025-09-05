"""Type definitions and aliases for the FLEXT core library.

Provides efficient type definitions, generic patterns, and type utilities for the
FLEXT ecosystem with hierarchical organization and Python 3.13+ syntax.

"""

from __future__ import annotations

from collections.abc import (
    Callable,
)
from typing import (
    Literal,
    ParamSpec,
    TypeVar,
)

from flext_core.result import FlextResult

# =============================================================================
# FLEXT TYPE SYSTEM - Hierarchical organization following FLEXT patterns
# =============================================================================


class FlextTypes:
    """Hierarchical type system organizing FLEXT types by domain and usage.

    This is the single consolidated class for all FLEXT Core type definitions,
    following the Flext[Area][Module] pattern where this represents FlextTypes.
    All other FLEXT libraries should reference types from this class hierarchy.

    """

    # =========================================================================
    # TYPE VARIABLES - Generic programming foundation
    # =========================================================================

    class TypeVars:
        """Generic type variables for ecosystem-wide use.

        Centralized location for all TypeVar and ParamSpec definitions
        following FLEXT patterns and Python 3.13+ syntax.
        """

        # Specialized type variables
        TEntity = TypeVar("TEntity")  # Entity types
        TValueObject = TypeVar("TValueObject")  # Value object types
        TAggregate = TypeVar("TAggregate")  # Aggregate root types
        TMessage = TypeVar("TMessage")  # Message types for handlers
        TRequest = TypeVar("TRequest")  # Request types
        TResponse = TypeVar("TResponse")  # Response types
        TCommand = TypeVar("TCommand")  # Command types for CQRS
        TQuery = TypeVar("TQuery")  # Query types for CQRS
        TResult = TypeVar("TResult")  # Result types for handlers
        TEntry = TypeVar("TEntry")  # Entry types for schema processing

        # Primary type variables for generic programming (Python 3.13+ syntax)
        T = TypeVar("T")  # Primary generic type for FLEXT ecosystem
        U = TypeVar("U")  # Secondary generic type for FLEXT ecosystem
        V = TypeVar("V")  # Tertiary generic type
        K = TypeVar("K")  # Key type for mappings
        R = TypeVar("R")  # Return type for functions
        E = TypeVar("E", bound=Exception)  # Exception type
        F = TypeVar("F")  # Function/field type variable
        P = ParamSpec("P")  # Parameter specification for callables

    # =========================================================================
    # CORE TYPES - Fundamental building blocks
    # =========================================================================

    class Core:
        """Core fundamental types used throughout the FLEXT ecosystem."""

        # Basic collection types
        type Dict = dict[str, object]
        type List = list[object]

        # JSON types
        type JsonValue = (
            str
            | int
            | float
            | bool
            | None
            | list[str | int | float | bool | None | list[object] | dict[str, object]]
            | dict[
                str,
                str | int | float | bool | None | list[object] | dict[str, object],
            ]
        )
        type JsonObject = dict[str, JsonValue]

        # Value type - Union type for domain operations
        type Value = str | int | float | bool | None | object

        # Operation callable
        type OperationCallable = Callable[[object], object]

        # Serialization
        type Serializer = Callable[[object], dict[str, object]]

    # =========================================================================
    # DOMAIN TYPES - Domain-Driven Design patterns
    # =========================================================================

    class Domain:
        """Domain types for DDD patterns and business logic."""

        # Entity types
        type EntityId = str
        type Entity = object

        # Value object types
        type ValueObject = object

        # Aggregate types
        type AggregateRoot = object

        # Domain event types
        type DomainEvent = dict[str, object]

    # =========================================================================
    # RESULT PATTERN TYPES - Railway-oriented programming
    # =========================================================================

    class Result:
        """Result pattern types optimized for processors.py and decorators.py implementation."""

        # =====================================================================
        # HEAVILY USED RESULT TYPES - Core result patterns (15 total usages)
        # =====================================================================

        # Primary result type
        type ResultType[T] = FlextResult[T]

        # Success type
        type Success[T] = FlextResult[T]

    # =========================================================================
    # SERVICE TYPES - Service layer patterns - REFACTORED with Pydantic 2.11+ & Python 3.13+
    # =========================================================================

    class Service:
        """Service layer types optimized for container.py and commands.py implementation."""

        # Container registry types
        type ServiceDict = dict[str, object]  # Services registry mapping
        type FactoryDict = dict[str, Callable[[], object]]  # Factory registry
        type ServiceListDict = dict[
            str,
            Literal["instance", "factory"],
        ]  # Service type mapping

    # =========================================================================
    # PAYLOAD TYPES - Message and payload patterns for integration
    # =========================================================================

    class Payload:
        """Payload types for message and event patterns.

        This class provides type definitions for payload patterns used in
        messaging, event handling, and cross-service communication.
        """

        # Basic payload types
        type Id = str
        type Data = dict[str, object]
        type Metadata = dict[str, str]

        # Message types
        type MessageId = Id
        type MessageData = Data
        type MessageType = str

        # Event types
        type EventId = Id
        type EventData = Data
        type EventMetadata = Metadata

        # Serialization types
        type Serialized = str
        type Deserialized = Data

    # =========================================================================
    # HANDLER TYPES - Handler patterns for CQRS
    # =========================================================================

    class Handler:
        """Handler types for CQRS and message processing.

        This class provides type definitions for command query responsibility
        segregation patterns and message processing handlers.
        """

        # Command and query types
        type Command = object  # Commands are specific to domain
        type Query = object  # Queries are specific to domain
        type Event = dict[str, object]

        # Handler function types
        type CommandHandler = Callable[[Command], object]
        type QueryHandler = Callable[[Query], object]
        type EventHandler = Callable[[Event], None]

        # Handler metadata
        type HandlerName = str
        type HandlerMetadata = dict[str, object]

        # Processing context
        type Context = dict[str, object]
        type ProcessingResult = object

    # =========================================================================
    # COMMANDS TYPES - CQRS command system types
    # =========================================================================

    class Commands:
        """Commands-specific types based on actual FlextCommands implementation.

        These types match the actual method signatures found in commands.py
        for better type coherence between definitions and implementations.
        """

        # Configuration types - matching configure_commands_system
        type CommandsConfig = FlextResult[
            dict[str, str | int | float | bool | list[object] | dict[str, object]]
        ]
        type CommandsConfigDict = dict[
            str,
            str | int | float | bool | list[object] | dict[str, object],
        ]

        # Command lifecycle types
        type CommandId = str
        type CommandName = str
        type CommandStatus = str
        type CommandResult = object

    # =========================================================================
    # AGGREGATES TYPES - Aggregate root system types
    # =========================================================================

    class Aggregates:
        """Aggregates-specific types for FlextModels implementation."""

        # Primary configuration dictionary type
        type ConfigValue = str | int | float | bool | list[object] | dict[str, object]
        # Primary aggregate config type
        type AggregatesConfigDict = dict[str, ConfigValue]
        # Primary config result
        type AggregatesConfig = FlextResult[AggregatesConfigDict]

        # System config result
        type SystemConfig = FlextResult[AggregatesConfigDict]

        # Performance optimization types
        type PerformanceLevel = Literal["low", "balanced", "high", "extreme"]
        type PerformanceConfig = FlextResult[AggregatesConfigDict]

    # =========================================================================
    # CONTAINER TYPES - Dependency injection
    # =========================================================================

    class Container:
        """Container types optimized for core.py implementation."""

        # Primary service key type (3 usages) - Enhanced service identifier
        type ServiceKey = str  # Service key identifier

        # Service operation types (5 usages) - Enhanced return types for container operations
        type ServiceInstance = object  # Service instance type
        type ServiceRegistration = FlextResult[None]  # Service registration result
        type ServiceRetrieval = FlextResult[object]  # Service retrieval result
        type FactoryFunction = Callable[[], object]  # Factory callable type
        type FactoryRegistration = FlextResult[None]  # Factory registration result

    # =========================================================================
    # CONFIG TYPES - Configuration and settings
    # =========================================================================

    class Config:
        """Configuration and settings types."""

        # Primary configuration types
        type ConfigValue = str | int | float | bool | list[object] | dict[str, object]
        type ConfigDict = dict[str, ConfigValue]  # Core config dictionary type

        # Environment type
        type Environment = Literal[
            "development",
            "production",
            "staging",
            "test",
            "local",
        ]

        # Logging levels
        type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        # Config serialization
        type ConfigSerializer = Callable[[ConfigDict], str]

    # =========================================================================
    # MODELS TYPES - Models system types - REFACTORED with Pydantic 2.11+ & Python 3.13+
    # =========================================================================

    class Models:
        """Models system types optimized for models.py implementation."""

        # =====================================================================
        # HEAVILY USED MODELS TYPES - Core model patterns (21 total usages)
        # =====================================================================

        # Primary configuration dictionary type (15 usages) - Enhanced with Python 3.13+ union syntax
        type ConfigValue = str | int | float | bool | list[object] | dict[str, object]
        type ConfigDict = dict[str, ConfigValue]  # Primary models config type

        # Configuration result types (6 usages) - Enhanced return types for model operations
        type Config = FlextResult[ConfigDict]  # Models config result
        type EnvironmentConfig = FlextResult[ConfigDict]  # Environment config result
        type SystemInfo = FlextResult[ConfigDict]  # System info result
        type PerformanceConfig = dict[str, ConfigValue]  # Performance config input
        type OptimizedPerformanceConfig = FlextResult[
            ConfigDict
        ]  # Optimized config result


# =============================================================================
# TYPE VARIABLES
# =============================================================================

# Generic type variables

T = TypeVar("T")  # Generic type
U = TypeVar("U")  # Generic type
V = TypeVar("V")  # Generic type
R = TypeVar("R")  # Generic result type
E = TypeVar("E", bound=Exception)  # Exception type
F = TypeVar("F")  # Generic function type
K = TypeVar("K")  # Generic key type
P = ParamSpec("P")  # Parameter specification

# =============================================================================
# EXPORTS - Hierarchical types only
# =============================================================================

__all__: list[str] = [
    "E",
    "F",
    "FlextTypes",
    "K",
    "P",
    "R",
    "T",
    "U",
    "V",
]
