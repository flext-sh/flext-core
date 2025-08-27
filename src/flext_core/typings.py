"""Type definitions and aliases for the FLEXT core library.

This module provides comprehensive type definitions, generic patterns, and type
utilities for the FLEXT ecosystem. Built with Python 3.13+ syntax and strict
typing enforcement following Clean Architecture principles and FLEXT patterns.

Critical Patterns Applied:
    - FlextTypes hierarchical class with nested organization
    - FlextConstants integration for type-related constants
    - Centralized protocols import from protocols.py
    - Zero circular import dependencies through proper layering
    - Minimal compatibility facades in legacy.py

The module includes:
    - FlextTypes hierarchical type system with nested classes
    - Integration with FlextConstants for type constants
    - Centralized protocol imports from protocols.py
    - Core type definitions (T, U, V generic variables)
    - FLEXT-specific type aliases for common patterns
    - Complex generic types for Result patterns and containers
    - Type utility functions and validation helpers
    - Protocol-based type definitions for interfaces
    - Backward compatibility aliases for ecosystem migration

Examples:
    Modern hierarchical usage::

        from flext_core.typings import FlextTypes

        # Hierarchical type access
        result: FlextTypes.Result.Success[str] = FlextResult.ok("data")
        handler: FlextTypes.Protocol.Handler[str, int] = MyHandler()
        config: FlextTypes.Config.Dict = {"key": "value"}

    Protocol integration::

        from flext_core.typings import FlextTypes
        from flext_core.protocols import FlextProtocols

        # Use centralized protocols
        service: FlextProtocols.Domain.Service = MyService()
        validator: FlextTypes.Protocol.Validator[str] = MyValidator()

    Constant integration::

        from flext_core.typings import FlextTypes
        from flext_core.constants import FlextConstants

        # Use hierarchical constants with types
        timeout: int = FlextConstants.Defaults.TIMEOUT
        error_code: str = FlextConstants.Errors.VALIDATION_ERROR

Note:
    This module enforces Python 3.13+ requirements and follows FLEXT refactoring
    patterns with hierarchical organization, proper import layering to avoid
    circular dependencies, and centralized compatibility management.

"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Coroutine
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from pathlib import Path
from typing import (
    Literal,
    ParamSpec,
    TypeGuard,
    TypeVar,
)

from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult

# NOTE: FlextResult imported at end of file to avoid circular import

# =============================================================================
# CORE TYPE VARIABLES - Foundation for generic programming
# =============================================================================


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

# =============================================================================
# FLEXT TYPE SYSTEM - Hierarchical organization following FLEXT patterns
# =============================================================================


class FlextTypes:
    """Hierarchical type system organizing FLEXT types by domain and usage.

    Following FLEXT_REFACTORING_PROMPT.md requirements:
        - Hierarchical organization with nested classes
        - Integration with FlextConstants for type-related constants
        - Proper import layering to avoid circular dependencies
        - Centralized protocol imports from protocols.py
        - Clean Architecture compliance with domain separation

    This is the single consolidated class for all FLEXT Core type definitions,
    following the Flext[Area][Module] pattern where this represents FlextTypes.
    All other FLEXT libraries should reference types from this class hierarchy.

    The type system is organized into the following domains:
        - Core: Fundamental types (primitives, collections, unions)
        - Result: Result pattern types for railway-oriented programming
        - Protocol: Protocol-based interface types from centralized protocols.py
        - Domain: Domain modeling types (entities, value objects, aggregates)
        - Handler: Handler and processing types for CQRS
        - Config: Configuration and settings types
        - Network: Network and connectivity types
        - Async: Asynchronous operation types
        - Meta: Metaclass and advanced type utilities
        - Constants: Type-related constants from FlextConstants

    Architecture Principles:
        - Single Responsibility: Each nested class has a single domain focus
        - Open/Closed: Easy to extend with new type categories
        - Liskov Substitution: Consistent interface across all categories
        - Interface Segregation: Clients depend only on types they use
        - Dependency Inversion: High-level types don't depend on low-level details

    Examples:
        Using hierarchical types for better organization::

            # Core primitive types
            value: str = "hello"
            count: int = 42

            # Result pattern types
            result: FlextTypes.Result.Success[str] = FlextResult.ok("data")

            # Protocol types from centralized protocols.py
            handler: FlextTypes.Protocol.Handler[str, int] = MyHandler()

            # Domain modeling
            user_id: FlextTypes.Domain.EntityId = "user_123"

            # Constants integration
            timeout: int = FlextTypes.Constants.Timeout

    Note:
        All types use Python 3.13+ syntax and are designed for strict
        type checking. The hierarchical organization follows SOLID principles,
        Clean Architecture patterns, and FLEXT refactoring requirements.

    """

    # =========================================================================
    # CORE TYPES - Fundamental building blocks
    # =========================================================================

    class Core:
        """Core fundamental types used throughout the FLEXT ecosystem.

        This class contains the most basic type definitions that form the foundation
        of the FLEXT type system, including primitives, collections, and utilities
        following SOLID principles.

        Architecture Principles Applied:
            - Single Responsibility: Only core type definitions
            - Open/Closed: Easy to extend with new primitive types
            - Interface Segregation: Core types separated from domain-specific ones
        """

        # Primitive types with semantic meaning
        type String = str
        type Integer = int
        type Float = float
        type Boolean = bool
        type Bytes = bytes
        type Object = object

        # Numeric types
        type Number = int | float
        type PositiveInt = int  # Constrained in validation
        type NonNegativeInt = int  # Constrained in validation

        # Collection types
        type List = list[object]
        type Dict = dict[str, object]
        type Set = set[object]
        type Tuple = tuple[object, ...]
        type Data = dict[str, object]  # Generic data container type

        # JSON-compatible types
        type JsonPrimitive = str | int | float | bool | None
        type JsonValue = (
            JsonPrimitive
            | list[str | int | float | bool | None | list[object] | dict[str, object]]
            | dict[
                str, str | int | float | bool | None | list[object] | dict[str, object]
            ]
        )
        type JsonObject = dict[str, JsonValue]
        type JsonDict = dict[str, JsonValue]  # Alias for JsonObject compatibility
        type JsonArray = list[JsonValue]

        # Serialization types
        type Serializer = Callable[[object], dict[str, object]]
        type Deserializer = Callable[[dict[str, object]], object]

        # Identifier types
        type Identifier = str
        type Id = str  # Generic ID type
        type UUID = str  # UUID string representation

        # Path and filesystem types
        type PathLike = str | Path
        type FilePath = Path
        type DirectoryPath = Path

        # Time and date types
        type Timestamp = float
        type IsoDateTime = str

        # Function types - Fixed Python 3.13+ type alias syntax
        type FlextCallableType = Callable[[object], object]

        # Error and status types
        type ErrorMessage = str
        type ErrorCode = str
        type StatusCode = str
        type LogMessage = str

        # Operation types - Python 3.13+ compatible
        type OperationCallable = Callable[[object], object]
        type OperationCallableType = (
            Callable[[], object]
            | Callable[[object], object]
            | Callable[[object, object], object]
            | Callable[[object, object, object], object]
        )

        # Value types for domain operations
        type Value = str | int | float | bool | None | object

        # Performance metrics type for consistent signatures
        type PerformanceMetrics = dict[
            str, dict[str, dict[str, dict[str, float | bool]]]
        ]

    # =========================================================================
    # RESULT PATTERN TYPES - Railway-oriented programming support
    # =========================================================================

    class Result:
        """Result pattern types for railway-oriented programming.

        This class provides type definitions for the FlextResult pattern that
        enables railway-oriented programming throughout the FLEXT ecosystem.

        Architecture Principles Applied:
            - Single Responsibility: Only result pattern types
            - Open/Closed: Easy to extend with new result transformation types
            - Dependency Inversion: Result types don't depend on implementation details
        """

        # Direct hierarchical types following FLEXT dependency layers
        # String-based type annotations to avoid circular imports

        # Success and failure type aliases
        type Success[T] = FlextResult[T]
        type Failure = FlextResult[None]
        type ResultType[T] = FlextResult[T]

        # Result transformation types
        type ResultMapper[T, U] = Callable[[T], FlextResult[U]]
        type ResultPredicate[T] = Callable[[T], bool]

        # Async result types
        type AsyncResult[T] = Awaitable[FlextResult[T]]
        type AsyncResultMapper[T, U] = Callable[[T], Awaitable[FlextResult[U]]]

        # Error handling types
        type ErrorHandler = Callable[[Exception], str]
        type ErrorTransformer[T] = Callable[[T], Exception]

    # =========================================================================
    # PROTOCOL TYPES - Centralized protocol imports from protocols.py
    # =========================================================================

    class Protocol:
        """Protocol-based interface types for contracts.

        Following FLEXT refactoring requirements, this class imports and aliases
        protocols from the centralized protocols.py module to avoid circular
        dependencies and maintain proper layering.

        Architecture Principles Applied:
            - Single Responsibility: Only protocol type definitions
            - Dependency Inversion: Protocol types don't depend on implementations
            - Interface Segregation: Protocol types separated by layer
        """

        # Direct protocol type aliases to avoid circular dependencies
        # Following FLEXT strict rules: NO TYPE_CHECKING imports allowed

        # Foundation protocols - proper FlextProtocols references
        type CallableProtocol[T] = FlextProtocols.Foundation.Callable[T]
        type ValidatorProtocol[T] = FlextProtocols.Foundation.Validator[T]
        type FactoryProtocol[T] = FlextProtocols.Foundation.Factory[T]
        type ErrorHandlerProtocol = FlextProtocols.Foundation.ErrorHandler

        # Domain protocols - proper FlextProtocols references
        type ServiceProtocol = FlextProtocols.Domain.Service
        type RepositoryProtocol[T] = FlextProtocols.Domain.Repository[T]
        type DomainEventProtocol = FlextProtocols.Domain.DomainEvent

        # Application protocols - proper FlextProtocols references
        type HandlerProtocol[TRequest, TResponse] = FlextProtocols.Application.Handler[
            TRequest, TResponse
        ]
        type MessageHandlerProtocol = FlextProtocols.Application.MessageHandler
        type ValidatingHandlerProtocol = FlextProtocols.Application.ValidatingHandler
        type AuthorizingHandlerProtocol = FlextProtocols.Application.AuthorizingHandler
        type EventProcessorProtocol = FlextProtocols.Application.EventProcessor
        type UnitOfWorkProtocol = FlextProtocols.Application.UnitOfWork

        # Infrastructure protocols - proper FlextProtocols references
        type ConnectionProtocol = FlextProtocols.Infrastructure.Connection
        type LoggerProtocol = FlextProtocols.Infrastructure.LoggerProtocol
        type Configurable = FlextProtocols.Infrastructure.Configurable

        # Extensions protocols
        type Plugin = FlextProtocols.Extensions.Plugin
        type PluginContext = FlextProtocols.Extensions.PluginContext
        type Middleware = FlextProtocols.Extensions.Middleware
        type AsyncMiddleware = FlextProtocols.Extensions.AsyncMiddleware
        type Observability = FlextProtocols.Extensions.Observability

        # Legacy protocol aliases for backward compatibility
        type Validator[T] = FlextProtocols.Foundation.Validator[T]
        type Factory[T] = FlextProtocols.Foundation.Factory[T]
        type ErrorHandler = FlextProtocols.Foundation.ErrorHandler
        type Service = FlextProtocols.Domain.Service
        type Repository[T] = FlextProtocols.Domain.Repository[T]
        type DomainEvent = FlextProtocols.Domain.DomainEvent
        type Handler[TRequest, TResponse] = FlextProtocols.Application.Handler[
            TRequest, TResponse
        ]
        type MessageHandler = FlextProtocols.Application.MessageHandler
        type ValidatingHandler = FlextProtocols.Application.ValidatingHandler
        type AuthorizingHandler = FlextProtocols.Application.AuthorizingHandler
        type EventProcessor = FlextProtocols.Application.EventProcessor
        type UnitOfWork = FlextProtocols.Application.UnitOfWork
        type Connection = FlextProtocols.Infrastructure.Connection

    # =========================================================================
    # DOMAIN TYPES - Domain modeling and DDD patterns
    # =========================================================================

    class Domain:
        """Domain modeling types for DDD patterns.

        This class provides type definitions for domain-driven design patterns
        including entities, value objects, aggregates, and events.

        Architecture Principles Applied:
            - Single Responsibility: Only domain modeling types
            - Open/Closed: Easy to extend with new domain patterns
            - Domain Focus: Types reflect business domain concepts
        """

        # Entity types
        type EntityId = str
        type EntityState = dict[str, object]
        type EntityVersion = int
        type EntityTimestamp = datetime  # Add missing EntityTimestamp type

        # Value object types
        type ValueObjectData = dict[str, object]

        # Aggregate types
        type AggregateId = EntityId
        type AggregateVersion = int

        # Event types
        type EventId = str
        type EventType = str
        type EventData = dict[str, object]
        type EventMetadata = dict[str, object]

        # Correlation and tracing types
        type CorrelationId = str

        # Repository types
        type Specification[T] = Callable[[T], bool]
        type QueryCriteria = dict[str, object]

    # =========================================================================
    # SERVICE TYPES - Service layer patterns and application services
    # =========================================================================

    class Service:
        """Service layer types for application services and business operations.

        This class provides type definitions for service layer patterns
        including application services, domain services, and service contracts.

        Architecture Principles Applied:
            - Single Responsibility: Only service layer type definitions
            - Open/Closed: Easy to extend with new service patterns
            - Service Focus: Types support service layer patterns
        """

        # Service contract types
        type ServiceId = str
        type ServiceName = str
        type ServiceVersion = str

        # Service operation types
        type ServiceOperation = Callable[[object], object]
        type ServiceResult[T] = FlextTypes.Result.Success[T]

        # Service metadata
        type ServiceMetadata = dict[str, object]
        type ServiceConfig = dict[str, object]

        # Service lifecycle types
        type ServiceState = Literal["active", "inactive", "maintenance"]
        type ServiceHealth = Literal["healthy", "degraded", "unhealthy"]

        # Container service types for dependency injection
        type ServiceInstance = object  # Any service instance
        type ServiceDict = dict[str, ServiceInstance]  # Services registry
        type FactoryDict = dict[str, Callable[[], ServiceInstance]]  # Factory registry
        type ServiceListDict = dict[
            str, Literal["instance", "factory"]
        ]  # Service type mapping

    # =========================================================================
    # PAYLOAD TYPES - Message and payload patterns for integration
    # =========================================================================

    class Payload:
        """Payload types for message and event patterns.

        This class provides type definitions for payload patterns used in
        messaging, event handling, and cross-service communication.

        Architecture Principles Applied:
            - Single Responsibility: Only payload and messaging type definitions
            - Open/Closed: Easy to extend with new payload patterns
            - Integration Focus: Types support cross-service communication
        """

        # Basic payload types
        type PayloadData = dict[str, object]
        type PayloadMetadata = dict[str, str]
        type PayloadId = str

        # Message types
        type MessageId = str
        type MessageType = str
        type MessageData = PayloadData

        # Event types
        type EventId = str
        type EventPayload = PayloadData
        type EventMetadata = PayloadMetadata

        # Serialization types
        type SerializedPayload = str
        type DeserializedPayload = PayloadData

    # =========================================================================
    # AUTH TYPES - Authentication and authorization patterns
    # =========================================================================

    class Auth:
        """Authentication and authorization types.

        This class provides type definitions for authentication and authorization
        patterns including tokens, credentials, and user identity.

        Architecture Principles Applied:
            - Single Responsibility: Only authentication type definitions
            - Security Focus: Types support secure authentication patterns
        """

        # Identity types
        type UserId = str
        type Username = str
        type UserRole = str

        # Token types
        type AccessToken = str
        type RefreshToken = str
        type TokenPayload = dict[str, object]

        # Authorization types
        type Permission = str
        type Role = str
        type Scope = str

    # =========================================================================
    # LOGGING TYPES - Logging and observability patterns
    # =========================================================================

    class Logging:
        """Logging and observability types.

        This class provides type definitions for logging patterns including
        log levels, messages, and structured logging data.

        Architecture Principles Applied:
            - Single Responsibility: Only logging type definitions
            - Observability Focus: Types support structured logging patterns
        """

        # Log level types
        type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        type LogMessage = str
        type LogData = dict[str, object]

        # Structured logging types
        type LogEntry = dict[str, object]
        type LogContext = dict[str, str]
        type LogMetadata = dict[str, object]
        type ContextDict = dict[str, object]

    # =========================================================================
    # HANDLER TYPES - Handler patterns for CQRS
    # =========================================================================

    class Handler:
        """Handler types for CQRS and message processing.

        This class provides type definitions for command query responsibility
        segregation patterns and message processing handlers.

        Architecture Principles Applied:
            - Single Responsibility: Only handler pattern types
            - Open/Closed: Easy to extend with new handler types
            - Interface Segregation: Handler types separated by responsibility
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
    # CONFIG TYPES - Configuration and settings
    # =========================================================================

    class Config:
        """Configuration and settings types.

        This class provides type definitions for configuration management,
        environment settings, and deployment configuration.

        Architecture Principles Applied:
            - Single Responsibility: Only configuration types
            - Open/Closed: Easy to extend with new configuration patterns
            - Environment Independence: Configuration types don't depend on
              specific environments
        """

        # Configuration data types
        type ConfigValue = str | int | float | bool | list[object] | dict[str, object]
        type ConfigDict = dict[str, ConfigValue]
        type ConfigKey = str
        type ConfigPath = str

        # Alias for legacy compatibility
        type Config = ConfigDict

        # Environment and deployment
        type Environment = Literal["development", "production", "staging", "test"]
        type DeploymentMode = Literal["local", "cloud", "hybrid"]

        # Validation types
        type ConfigValidator = Callable[[ConfigDict], None]
        type ConfigTransformer = Callable[[ConfigDict], ConfigDict]

    # =========================================================================
    # FIELD TYPES - Field system types for FlextFields
    # =========================================================================

    class Fields:
        """Fields system types for the FlextFields system.

        This class provides type definitions for the field system,
        ensuring type safety when working with different field types
        and their configurations.

        Architecture Principles Applied:
            - Single Responsibility: Only field system types
            - Open/Closed: Easy to extend with new field types
            - Type Safety: Proper type compatibility for field operations
        """

        # Basic Python type aliases for field values
        type String = str
        type Integer = int
        type Float = float
        type Boolean = bool
        type Object = object
        type Dict = dict[str, object]
        type List = list[object]
        type Number = int | float

        # Field-specific type aliases
        type Email = str  # Email is a string with validation
        type Uuid = str  # UUID is a string with validation
        type DateTime = datetime  # DateTime field type

        # Field configuration and metadata type aliases
        type Config = dict[str, str | int | float | bool | datetime | None | object]
        type Metadata = dict[str, object]

        # Field instance union type - represents any field instance
        type Instance = object  # Any field instance (covariant compatibility)

        # Field validation type aliases
        type ValidationResult = object  # Result of field validation
        type ValidationError = str  # Validation error message

        # Field constraint type aliases
        type Constraints = dict[str, object]  # Field constraints configuration
        type Options = dict[str, object]  # Field options configuration

    # =========================================================================
    # NETWORK TYPES - Network and connectivity
    # =========================================================================

    class Network:
        """Network and connectivity types.

        This class provides type definitions for network operations,
        HTTP requests/responses, and connectivity patterns.

        Architecture Principles Applied:
            - Single Responsibility: Only network-related types
            - Open/Closed: Easy to extend with new network protocols
            - Protocol Independence: Network types don't depend on specific protocols
        """

        # Address types
        type IPAddress = str
        type PortNumber = int
        type HostName = str
        type URL = str
        type URI = str

        # Protocol types
        type Protocol = Literal["http", "https", "tcp", "udp", "ws", "wss"]
        type HttpMethod = Literal[
            "GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"
        ]

        # Request/Response types
        type Headers = dict[str, str]
        type QueryParams = dict[str, str | list[str]]
        type RequestBody = str | bytes | dict[str, object]
        type ResponseBody = str | bytes | dict[str, object]

        # Connection types
        type ConnectionString = str
        type ConnectionPool = object  # Implementation-specific

    # =========================================================================
    # ASYNC TYPES - Asynchronous operation support
    # =========================================================================

    class Async:
        """Asynchronous operation types.

        This class provides type definitions for asynchronous programming
        patterns including async/await, coroutines, and streaming.

        Architecture Principles Applied:
            - Single Responsibility: Only asynchronous operation types
            - Open/Closed: Easy to extend with new async patterns
            - Concurrency Focus: Types support safe concurrent operations
        """

        # Awaitable types
        type AsyncCallable[**P, T] = Callable[P, Awaitable[T]]
        type AsyncGenerator[T] = AsyncIterator[T]
        type AsyncContext[T] = AbstractAsyncContextManager[T]

        # Coroutine types
        type CoroFunction[**P, T] = Callable[P, Coroutine[object, object, T]]
        type AsyncResult[T] = Awaitable[T]

        # Streaming types
        type AsyncStream[T] = AsyncIterable[T]
        type StreamProcessor[T, U] = Callable[[AsyncIterable[T]], AsyncIterable[U]]

    # =========================================================================
    # META TYPES - Metaclass and advanced utilities
    # =========================================================================

    class Meta:
        """Metaclass and advanced type utilities.

        This class provides type definitions for metaprogramming,
        reflection, and advanced type manipulation.

        Architecture Principles Applied:
            - Single Responsibility: Only metaprogramming types
            - Open/Closed: Easy to extend with new metaprogramming patterns
            - Type Safety: Meta types support compile-time type checking
        """

        # Type checking utilities
        type TypeChecker[T] = Callable[[object], TypeGuard[T]]
        type TypeValidator = Callable[[object], bool]

        # Generic utilities
        type GenericAlias = object  # For generic type manipulation
        type TypeInfo = dict[str, object]  # Type metadata

        # Decorator types
        type DecoratorFactory[P, T] = Callable[
            [P], Callable[[Callable[[P], T]], Callable[[P], T]]
        ]
        type MethodDecorator[T] = Callable[
            [Callable[[object], T]], Callable[[object], T]
        ]

        # Reflection types
        type AttributeName = str
        type MethodName = str
        type ClassName = str

    # =========================================================================
    # CONSTANTS INTEGRATION - Type-related constants from FlextConstants
    # =========================================================================

    class Constants:
        """Type-related constants integrated from FlextConstants.

        Following FLEXT refactoring requirements, this class provides
        type-related constants that are frequently used with type definitions.

        Architecture Principles Applied:
            - Single Responsibility: Only type-related constants
            - Dependency Inversion: Constants don't depend on implementation details
            - Integration: Seamless integration with FlextConstants hierarchy
        """

        # Direct constant types - following FLEXT hierarchical pattern
        type Timeout = int  # FlextConstants.Defaults.TIMEOUT
        type MaxRetries = int  # FlextConstants.Defaults.MAX_RETRIES
        type PageSize = int  # FlextConstants.Defaults.PAGE_SIZE

        # Error codes for types
        type ValidationError = str  # FlextConstants.Errors.VALIDATION_ERROR
        type TypeError = str  # FlextConstants.Errors.TYPE_ERROR

        # Status values for types
        type Success = str  # FlextConstants.Status.SUCCESS
        type Failure = str  # FlextConstants.Status.FAILURE


# =============================================================================
# EXPORTS - Hierarchical types only
# =============================================================================

__all__: list[str] = [
    "E",
    "F",
    "FlextTypes",  # ONLY main class exported
    "K",
    "P",
    "R",
    "T",
    "U",
    "V",
]
