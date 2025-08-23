"""Centralized type definitions for the FLEXT core library.

This module provides the single source of truth for all types used throughout
the FLEXT ecosystem. All types are organized within the FlextTypes class
hierarchy for consistent access and maintainability.

Built for Python 3.13+ with strict typing enforcement and no compatibility layers.

Usage:
    Import types directly from FlextTypes::

        from flext_core.typings import FlextTypes

        # Core types
        config: FlextTypes.Core.Config = {"debug": True}
        entity_id: FlextTypes.Domain.EntityId = "user_123"

        # Protocol types
        validator: FlextTypes.Protocol.Validator[str] = email_validator
        handler: FlextTypes.Protocol.Handler[Command, str] = command_handler

Architecture:
    - FlextTypes: Single hierarchical class containing ALL types
    - No compatibility layers or aliases outside FlextTypes
    - Domain-organized type system (Core, Domain, Service, etc.)
    - Modern Python 3.13 type alias syntax throughout

"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, ParamSpec, TypeVar

# Define ParamSpec and TypeVar for FlextCallable
P = ParamSpec("P")
T = TypeVar("T")

if TYPE_CHECKING:
    from flext_core.protocols import (
        FlextProtocols,
    )
    from flext_core.result import FlextResult

# =============================================================================
# DYNAMIC EXCEPTION TYPE STUBS
# =============================================================================

# Type stubs for dynamically generated exceptions to help mypy understand them
# These are created at runtime by FlextExceptions but need type definitions
if TYPE_CHECKING:
    # Base exception classes - dynamically generated
    class FlextError(RuntimeError): ...

    class FlextUserError(TypeError): ...

    class FlextValidationError(ValueError): ...

    class FlextConfigurationError(ValueError): ...

    class FlextConnectionError(ConnectionError): ...

    class FlextAuthenticationError(PermissionError): ...

    class FlextPermissionError(PermissionError): ...

    class FlextNotFoundError(FileNotFoundError): ...

    class FlextAlreadyExistsError(FileExistsError): ...

    class FlextTimeoutError(TimeoutError): ...

    class FlextProcessingError(RuntimeError): ...

    class FlextCriticalError(RuntimeError): ...

    class FlextOperationError(RuntimeError): ...

    class FlextTypeError(TypeError): ...

    class FlextAttributeError(AttributeError): ...

# =============================================================================
# FLEXT HIERARCHICAL TYPE SYSTEM - Organized by domain
# =============================================================================


class FlextTypes:
    """Hierarchical type system organizing FLEXT types by domain and functionality.

    This class provides a structured organization of all types used throughout
    the FLEXT ecosystem, grouped by domain and functionality for better
    maintainability and discoverability.

    The type system is organized into the following domains:
        - Protocol: Type aliases for protocol definitions
        - Core: Fundamental building blocks (Value, Data, Config, etc.)
        - Domain: Business domain modeling (Entity, Event, etc.)
        - Service: Dependency injection and service location
        - Config: Configuration management
        - Logging: Structured logging and observability
        - Auth: Authentication and authorization
        - Field: Field validation and metadata

    Examples:
        Using protocol aliases::

            from flext_core.typings import FlextTypes

            validator: FlextTypes.Protocol.Validator[str] = email_validator
            handler: FlextTypes.Protocol.Handler[Command, str] = command_handler

        Using the hierarchical type system::

            user_id: FlextTypes.Domain.EntityId = "user123"
            config: FlextTypes.Config.Dict = {"debug": True}
            event_data: FlextTypes.Domain.EventData = {"type": "UserCreated"}

    """

    # =========================================================================
    # PROTOCOL TYPE ALIASES - Modern Python 3.13 syntax
    # =========================================================================

    class Protocol:
        """Protocol type aliases using modern Python 3.13 syntax.

        This class contains all protocol-related type aliases that reference
        the consolidated FlextProtocols class. These aliases provide shorter,
        more convenient names for common protocol usage patterns.

        Examples:
            Using protocol aliases for cleaner type annotations::

                # Instead of FlextProtocols.Callable[str]
                processor: FlextTypes.Protocol.Callable[str] = string_processor

                # Instead of FlextProtocols.Handler[Command, str]
                handler: FlextTypes.Protocol.Handler[Command, str] = command_handler

                # Instead of FlextProtocols.Validator[dict]
                validator: FlextTypes.Protocol.Validator[dict] = dict_validator

        """

        # Foundation layer aliases
        type Callable[T] = FlextProtocols.Foundation.Callable[T]
        type DecoratedCallable[T] = FlextProtocols.Foundation.DecoratedCallable[T]
        type Validator[T] = FlextProtocols.Foundation.Validator[T]
        type ErrorHandler = FlextProtocols.Foundation.ErrorHandler
        type Factory[T] = FlextProtocols.Foundation.Factory[T]
        type AsyncFactory[T] = FlextProtocols.Foundation.AsyncFactory[T]

        # Domain layer aliases
        type Service = FlextProtocols.Domain.Service
        type Repository[T] = FlextProtocols.Domain.Repository[T]
        type DomainEvent = FlextProtocols.Domain.DomainEvent
        type EventStore = FlextProtocols.Domain.EventStore

        # Application layer aliases
        type Handler[TInput, TOutput] = FlextProtocols.Application.Handler[
            TInput, TOutput
        ]
        type MessageHandler = FlextProtocols.Application.MessageHandler
        type ValidatingHandler = FlextProtocols.Application.ValidatingHandler
        type AuthorizingHandler = FlextProtocols.Application.AuthorizingHandler
        type EventProcessor = FlextProtocols.Application.EventProcessor
        type UnitOfWork = FlextProtocols.Application.UnitOfWork

        # Infrastructure layer aliases
        type Connection = FlextProtocols.Infrastructure.Connection
        type Auth = FlextProtocols.Infrastructure.Auth
        type Configurable = FlextProtocols.Infrastructure.Configurable
        type LoggerProtocol = FlextProtocols.Infrastructure.LoggerProtocol

        # Extensions layer aliases
        type Plugin = FlextProtocols.Extensions.Plugin
        type PluginContext = FlextProtocols.Extensions.PluginContext
        type Middleware = FlextProtocols.Extensions.Middleware
        type AsyncMiddleware = FlextProtocols.Extensions.AsyncMiddleware
        type Observability = FlextProtocols.Extensions.Observability

        # Decorator patterns - using imported alias
        type DecoratedFunction[T] = FlextDecoratedFunction[T]

        # Utility callable types
        type SafeCallable[T] = FlextProtocols.Foundation.Callable[FlextResult[T]]

    # =========================================================================
    # CORE TYPES - Fundamental building blocks
    # =========================================================================

    class Core:
        """Core fundamental types used throughout the ecosystem.

        This class contains the most basic types that form the foundation
        of the FLEXT type system, including primitive types, identifiers,
        collections, and basic callable patterns.
        """

        # Basic value types
        type Value = str | int | float | bool | None
        type Data = dict[str, object]
        type Config = dict[str, str | int | float | bool | None]

        # Identifier types
        type Id = str
        type Key = str

        # Collection types
        type Dict = dict[str, object]
        type List = list[object]
        type StringDict = dict[str, str]
        type JsonDict = dict[str, object]

        # Connection and infrastructure
        type ConnectionString = str
        type LogMessage = str
        type ErrorCode = str
        type ErrorMessage = str

        # Factory types
        type Factory[T] = Callable[[], T]

        # Callable type union for generic callable patterns
        type TCallable = (
            Callable[[], object]
            | Callable[[str], object]
            | Callable[[str], str]
            | Callable[[str], dict[str, object]]
            | Callable[[object], object]
            | Callable[[object, object], object]
            | Callable[[int], int]
            | Callable[[int, int], int]
            | Callable[[], str]
            | Callable[[], dict[str, str]]
            | Callable[[], None]
            | Callable[[list[int]], dict[str, object]]
            | Callable[[str, str], dict[str, str]]
            | Callable[[int], dict[str, int]]
            | Callable[[float], dict[str, float]]
        )

        # Validator callable types
        type Validator = Callable[[object], bool]
        type DecoratorFunction[T] = Callable[
            [FlextCallable[T]], FlextDecoratedFunction[T]
        ]

        # Serializer callable types
        type Serializer = Callable[[object], dict[str, object]]

    # =========================================================================
    # DOMAIN TYPES - Business domain modeling
    # =========================================================================

    class Domain:
        """Domain modeling and business logic types.

        This class contains types used in Domain-Driven Design (DDD) patterns,
        including entities, aggregates, domain events, and value objects.
        """

        # Entity and aggregate types
        type EntityId = str
        type EntityVersion = int
        type EntityTimestamp = datetime
        type AggregateId = str
        type EntityMetadata = dict[str, object]

        # Event types
        type EventType = str
        type EventData = dict[str, object]
        type EventVersion = int
        type DomainEvents = list[object]

        # Backward compatibility aliases (T* pattern)
        type TEntityId = EntityId

    # =========================================================================
    # SERVICE TYPES - Dependency injection and service location
    # =========================================================================

    class Service:
        """Service-related types for dependency injection.

        This class contains types used in the dependency injection system,
        service location patterns, event handling, and service orchestration.
        """

        # Service identification
        type ServiceName = str
        type ServiceKey = str | type[object]
        type Container = Mapping[str, object]
        type ServiceFactory[T] = Callable[[], T]
        type ServiceInstance = object
        type ServiceDict = dict[str, object]
        type ServiceListDict = dict[str, str]  # For list_services return type
        type FactoryDict = dict[str, Callable[[], object]]
        type ServiceInfo = dict[str, object]
        type ServiceRegistrations = dict[str, object]

        # Context and correlation types
        type CorrelationId = str
        type RequestId = str
        type TraceId = str

    # =========================================================================
    # HANDLER TYPES - Handler patterns and registries
    # =========================================================================

    class Handler:
        """Handler pattern types for CQRS and chain of responsibility.

        This class contains types used in handler implementations,
        including handler registries, chains, and metadata.
        """

        # Handler registry and metadata
        type HandlerDict = dict[str, object]
        type HandlerMetadata = dict[str, object]
        type ValidationRules = list[object]
        type MetricsDict = dict[str, object]

        # Metrics value types - specific and type-safe
        type CounterMetric = int
        type TimeMetric = float
        type ErrorCounterMap = dict[str, int]
        type SizeList = list[int]
        type PerformanceMap = dict[str, dict[str, int | float]]

        # More flexible metrics types for compatibility
        type MetricsValue = (
            #           CounterMetric | TimeMetric | ErrorCounterMap | SizeList | PerformanceMap
            int | float | str | dict[str, object] | list[object] | object
        )
        type MetricsData = dict[str, MetricsValue]
        type NumericMetrics = dict[str, int | float]
        type CollectionMetrics = dict[str, list[object] | dict[str, object]]

        # Handler-specific types
        type HandlerName = str
        type MessageType = object

    # =========================================================================
    # CONFIG TYPES - Configuration management
    # =========================================================================

    class Config:
        """Configuration management types.

        This class contains types used in configuration management systems,
        including configuration keys, values, environments, and deployment
        strategies.
        """

        # Core configuration types
        type Key = str
        type Value = object
        type Path = str
        type Environment = str
        type Dict = dict[str, str | int | float | bool | None]
        type Settings = dict[str, object]

        # File system types
        type DirectoryPath = str
        type FilePath = str

    # =========================================================================
    # PAYLOAD TYPES - Data transport and serialization
    # =========================================================================

    class Payload:
        """Payload and message transport types.

        This class contains types used in payload systems for cross-service
        communication, including metadata, serialization, and transport formats.
        """

        # Payload metadata and transport
        type Metadata = dict[str, object]
        type SerializedData = dict[str, object]
        type TransformFunction = Callable[[object], object]
        type SerializerFunction = Callable[[object], dict[str, object] | object]

        # Collection types
        type CollectionType = list[object] | tuple[object, ...]
        type MappingType = dict[str, object] | Mapping[str, object]

    # =========================================================================
    # LOGGING TYPES - Structured logging and observability
    # =========================================================================

    class Logging:
        """Logging and observability types.

        This class contains types used in structured logging and observability
        systems, including log levels, correlation IDs, metrics, and tracing.
        """

        # Core logging types
        type LoggerName = str
        type Level = str
        type Format = str
        type Message = str

        # Context types
        type ContextDict = dict[str, object]
        type Record = dict[str, object]
        type Metrics = dict[str, object]

    # =========================================================================
    # AUTH TYPES - Authentication and authorization
    # =========================================================================

    class Auth:
        """Authentication and authorization types.

        This class contains types used in authentication and authorization
        systems, including tokens, credentials, users, roles, and permissions.
        """

        # Authentication types
        type Token = str
        type UserId = str
        type UserData = dict[str, object]
        type Credentials = dict[str, object]

        # Authorization types
        type Role = str
        type Permission = str

    # =========================================================================
    # FIELD TYPES - Field definitions and validation
    # =========================================================================

    class Field:
        """Field-related types for validation and metadata.

        This class contains types used in field definition, validation,
        and metadata management systems.
        """

        # Field definition types
        type Id = str
        type Name = str
        type TypeStr = str
        type Value = object
        type Info = dict[str, object]
        type Metadata = dict[str, object]

    # =========================================================================
    # RESULT TYPES - Railway-oriented programming support
    # =========================================================================

    class Result:
        """Result-specific types for railway-oriented programming.

        This class contains types used in the FlextResult implementation,
        including error data, context, and operation metadata.
        """

        # Error handling types
        type ErrorMessage = str | None
        type ErrorCode = str | None
        type ErrorData = dict[str, object] | None
        type ErrorContext = dict[str, object]

        # Operation types
        type OperationName = str | None
        type ExceptionType = type[BaseException] | None

        # Transformation functions
        type MapFunction[T, U] = Callable[[T], U]
        type FlatMapFunction[T, U] = Callable[[T], FlextResult[U]]
        type FilterPredicate[T] = Callable[[T], bool]
        type RecoverFunction[T] = Callable[[str], T]
        type RecoverWithFunction[T] = Callable[[str], FlextResult[T]]
        type TapFunction[T] = Callable[[T], None]
        type ZipFunction[T, U, V] = Callable[[T, U], V]

        # Utility types
        type DataOrNone[T] = T | None
        type ResultTuple = tuple[object | None, str | None]


# =============================================================================
# CORE TYPE VARIABLES - Foundation building blocks
# =============================================================================


# Primary generic type variables (most commonly used)
# T and P are already defined at the top of the file (lines 37-38)
U = TypeVar("U")  # Secondary generic type parameter
V = TypeVar("V")  # Tertiary generic type parameter
K = TypeVar("K")  # Key type parameter
R = TypeVar("R")  # Result type for operations
E = TypeVar("E")  # Error type for error handling


# TypeVar for preserving function signatures in decorator
F = TypeVar("F", bound=FlextTypes.Core.TCallable)

# Specialized type variables
TData = TypeVar("TData")  # Generic data type
TConfig = TypeVar("TConfig")  # Generic configuration type

# Result-specific type variable (for local type annotations)
TResultLocal = TypeVar("TResultLocal")  # Local type variable for result operations

# Command and handler specific type variables
CommandT = TypeVar("CommandT")  # Command types
ResultT = TypeVar("ResultT")  # Result types
QueryT = TypeVar("QueryT")  # Query types
QueryResultT = TypeVar("QueryResultT")  # Query result types
TInput = TypeVar("TInput")  # Input types
TOutput = TypeVar("TOutput")  # Output types
ServiceRequestT = TypeVar("ServiceRequestT")  # Service request types
ServiceDomainT = TypeVar("ServiceDomainT")  # Service domain types
ServiceResultT = TypeVar("ServiceResultT")  # Service result types
EntryT = TypeVar("EntryT")  # Entry types
InputT = TypeVar("InputT")  # Generic input types
OutputT = TypeVar("OutputT")  # Generic output types
TDomainResult = TypeVar("TDomainResult")  # Domain result types

# Query-specific type variables for handlers
TQuery = TypeVar("TQuery")  # Query types (alias for clarity)
TQueryResult = TypeVar("TQueryResult")  # Query result types (alias for clarity)


# =============================================================================
# BASIC TYPE DEFINITIONS
# =============================================================================

AnyCallable = FlextTypes.Core.TCallable

# Define FlextDecoratedFunction early since it's used in class definitions
# Compatible with variadic FlextCallable protocol (*args, **kwargs)
# Using Protocol-based callable that accepts any arguments
if TYPE_CHECKING:
    type FlextDecoratedFunction[T] = FlextProtocols.Foundation.DecoratedCallable[T]
else:
    type FlextDecoratedFunction[T] = Callable[..., T]

# =============================================================================
# CONVENIENCE ALIASES - For backward compatibility and shorter names
# =============================================================================

# Field aliases for current usage
FlextFieldId = FlextTypes.Field.Id
FlextFieldName = FlextTypes.Field.Name
FlextFieldTypeStr = FlextTypes.Field.TypeStr
TFieldInfo = FlextTypes.Field.Info  # Field info type

# Core type aliases for easy access
TAnyDict = FlextTypes.Core.Dict
TAnyList = FlextTypes.Core.List
TValue = FlextTypes.Core.Value
TFactory = FlextTypes.Core.Factory[object]

# Protocol aliases for easy access - proper generic forms
# Foundation layer aliases - use protocols from FlextProtocols
type FlextCallable[T] = FlextProtocols.Foundation.Callable[T]
type FlextErrorHandler = FlextProtocols.Foundation.ErrorHandler
type FlextFactory[T] = FlextProtocols.Foundation.Factory[T]
type FlextAsyncFactory[T] = FlextProtocols.Foundation.AsyncFactory[T]

# Domain layer aliases
type FlextService = FlextProtocols.Domain.Service
type FlextRepository[T] = FlextProtocols.Domain.Repository[T]
type FlextDomainEvent = FlextProtocols.Domain.DomainEvent
type FlextEventStore = FlextProtocols.Domain.EventStore

# Application layer aliases
type FlextHandler[TInput, TOutput] = FlextProtocols.Application.Handler[TInput, TOutput]
type FlextMessageHandler = FlextProtocols.Application.MessageHandler
type FlextValidatingHandler = FlextProtocols.Application.ValidatingHandler
type FlextAuthorizingHandler = FlextProtocols.Application.AuthorizingHandler
type FlextEventProcessor = FlextProtocols.Application.EventProcessor
type FlextUnitOfWork = FlextProtocols.Application.UnitOfWork

# Infrastructure layer aliases
type FlextConnection = FlextProtocols.Infrastructure.Connection
type FlextAuth = FlextProtocols.Infrastructure.Auth
# type FlextConfigurable = FlextProtocols.Infrastructure.Configurable  # Moved to protocols.py
type FlextLoggerProtocol = FlextProtocols.Infrastructure.LoggerProtocol

# Extensions layer aliases
type FlextPlugin = FlextProtocols.Extensions.Plugin
type FlextPluginContext = FlextProtocols.Extensions.PluginContext
type FlextMiddleware = FlextProtocols.Extensions.Middleware
type FlextAsyncMiddleware = FlextProtocols.Extensions.AsyncMiddleware
type FlextObservability = FlextProtocols.Extensions.Observability


# Domain aliases
TEntityId = FlextTypes.Domain.EntityId
TEntityMetadata = FlextTypes.Domain.EntityMetadata
TEventData = FlextTypes.Domain.EventData

# Service aliases
TServiceName = FlextTypes.Service.ServiceName
TCorrelationId = FlextTypes.Service.CorrelationId

# Logging aliases
TLogMessage = FlextTypes.Logging.Message
TContextDict = FlextTypes.Logging.ContextDict

# =============================================================================
# EXPORTS - Comprehensive centralized type system
# =============================================================================

__all__ = [  # noqa: RUF022
    # Core type variables
    "T",
    "U",
    "V",
    "K",
    "R",
    "E",
    "F",
    "P",
    "TData",
    "TConfig",
    "TResultLocal",
    "AnyCallable",
    # Command and handler specific type variables
    "CommandT",
    "ResultT",
    "QueryT",
    "QueryResultT",
    "TInput",
    "TOutput",
    "InputT",
    "OutputT",
    "ServiceRequestT",
    "ServiceDomainT",
    "ServiceResultT",
    "EntryT",
    "TDomainResult",
    "TQuery",
    "TQueryResult",
    # Main hierarchical classes
    "FlextTypes",
    # Foundation layer protocol aliases
    "FlextCallable",
    "FlextErrorHandler",
    "FlextFactory",
    "FlextAsyncFactory",
    # Domain layer protocol aliases
    "FlextService",
    "FlextRepository",
    "FlextDomainEvent",
    "FlextEventStore",
    # Application layer protocol aliases
    "FlextHandler",
    "FlextMessageHandler",
    "FlextValidatingHandler",
    "FlextAuthorizingHandler",
    "FlextEventProcessor",
    "FlextUnitOfWork",
    # Infrastructure layer protocol aliases
    "FlextConnection",
    "FlextAuth",
    # "FlextConfigurable",  # Moved to protocols.py
    "FlextLoggerProtocol",
    # Extensions layer protocol aliases
    "FlextPlugin",
    "FlextPluginContext",
    "FlextMiddleware",
    "FlextAsyncMiddleware",
    "FlextObservability",
    # Decorator patterns
    "FlextDecoratedFunction",
    # Field aliases
    "FlextFieldId",
    "FlextFieldName",
    "FlextFieldTypeStr",
    "TFieldInfo",
    # Core type aliases
    "TAnyDict",
    "TAnyList",
    "TValue",
    "TFactory",
    # Domain aliases
    "TEntityId",
    "TEntityMetadata",
    "TEventData",
    # Service aliases
    "TServiceName",
    "TCorrelationId",
    # Logging aliases
    "TLogMessage",
    "TContextDict",
    # Handler metrics aliases
    "MetricsValue",
    "MetricsData",
]

# Handler metrics aliases for easy access
MetricsValue = FlextTypes.Handler.MetricsValue
MetricsData = FlextTypes.Handler.MetricsData
