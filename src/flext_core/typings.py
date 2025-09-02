"""Type definitions and aliases for the FLEXT core library.

Provides efficient type definitions, generic patterns, and type utilities for the
FLEXT ecosystem with hierarchical organization and Python 3.13+ syntax.

Usage:
    # Basic type imports
    from flext_core.typings import FlextTypes, T, U, V

    # Using hierarchical types
    user_id: FlextTypes.Core.EntityId = "user-123"
    result: FlextTypes.Result.Success[User] = FlextResult.ok(user)

    # Generic function definitions
    def process_data[T, U](input_data: T) -> FlextResult[U]:
        # Implementation with generic type support

    # Protocol types
    repository: FlextTypes.Infrastructure.Repository[User] = UserRepository()

Features:
    - Hierarchical type organization (Core, Result, Domain, Infrastructure)
    - Integration with FlextConstants and FlextProtocols
    - Python 3.13+ generic syntax support
    - Type utility functions and validation helpers
    - Protocol-based type definitions for interfaces
    - Backward compatibility aliases for ecosystem migration
    FlextTypes.Service: Service layer types (ServiceDict, FactoryDict, etc.)
    FlextTypes.Payload: Message and payload types for integration patterns
    FlextTypes.Config: Configuration and settings types (ConfigDict, etc.)
    FlextTypes.Protocol: Protocol and interface types for contracts
    FlextTypes.Meta: Metaclass and advanced type utilities for framework development

Examples:
    Hierarchical type access:
    >>> from flext_core.typings import FlextTypes
    >>> result: FlextTypes.Result.Success[str] = FlextResult.ok("data")
    >>> config: FlextTypes.Config.Dict = {"key": "value"}
    >>> handler: FlextTypes.Protocol.Handler[str, int] = MyHandler()

    Generic type variables:
    >>> from flext_core.typings import T, U, V
    >>> def process_data(input_data: T) -> FlextResult[U]:
    ...     # Process and transform data with type safety
    ...     pass

    Protocol integration with centralized definitions:
    >>> from flext_core.typings import FlextTypes
    >>> from flext_core.protocols import FlextProtocols
    >>> service: FlextProtocols.Domain.Service = MyDomainService()
    >>> validator: FlextTypes.Protocol.Validator[str] = MyStringValidator()

Notes:
    - All types follow hierarchical organization for clear namespace separation
    - Generic type variables (T, U, V) are reused across the ecosystem for consistency
    - Protocol imports are centralized to avoid circular dependencies
    - Type definitions integrate with FlextConstants for configuration consistency
    - Backward compatibility is maintained through carefully designed aliases

"""

from __future__ import annotations

from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterator,
)
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import (
    Literal,
    ParamSpec,
    TypeGuard,
    TypeVar,
)

from flext_core.result import FlextResult

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
        following SOLID principles and Python 3.13+ syntax optimizations.

        Architecture Principles Applied:
            - Single Responsibility: Only core type definitions
            - Open/Closed: Easy to extend with new primitive types
            - Interface Segregation: Core types separated from domain-specific ones
            - Pydantic 2.11+ Integration: Enhanced validation and serialization support
        """

        # =====================================================================
        # FREQUENTLY USED CORE TYPES - Optimized with Python 3.13+ syntax
        # =====================================================================

        # Primary collection type (120 usages) - Enhanced with Python 3.13+ generics
        type Dict = dict[str, object]

        # Object type (28 usages) - Core foundation type
        type Object = object

        # String type (24 usages) - Enhanced with validation support
        type String = str

        # JSON types (21 usages) - Optimized for Pydantic 2.11+ serialization
        type JsonValue = (
            str
            | int
            | float
            | bool
            | None
            | list[str | int | float | bool | None | list[object] | dict[str, object]]
            | dict[
                str, str | int | float | bool | None | list[object] | dict[str, object]
            ]
        )
        type JsonObject = dict[str, JsonValue]

        # Value type (7 usages) - Enhanced union type for domain operations
        type Value = str | int | float | bool | None | object

        # Boolean type (6 usages) - Enhanced with strict validation
        type Boolean = bool

        # Function type (5 usages) - Python 3.13+ callable syntax
        type FlextCallableType = Callable[[object], object]

        # Operation callable (3 usages) - Enhanced for performance
        type OperationCallable = Callable[[object], object]

        # Logging type (3 usages) - Structured logging support
        type LogMessage = str

        # Identifier types (3 usages) - Enhanced with validation
        type Id = str
        type Identifier = str

        # Error handling (3 usages) - Enhanced error management
        type ErrorMessage = str

        # Serialization (2 usages) - Pydantic 2.11+ compatible
        type Serializer = Callable[[object], dict[str, object]]

        # Float type (2 usages) - Enhanced numeric support
        type Float = float

        # =====================================================================
        # MODERATELY USED TYPES - Kept for compatibility
        # =====================================================================

        # UUID type (1 usage) - Enhanced with validation
        type UUID = str

        # List type (1 usage) - Generic collection type
        type List = list[object]

        # =====================================================================
        # COMMENTED OUT - Low/No usage types (will be removed at the end)
        # =====================================================================

        # # Primitive types with low usage
        # type Integer = int  # Not directly used
        # type Bytes = bytes  # Not used

        # # Numeric types with no usage
        # type Number = int | float  # Not used
        # type PositiveInt = int  # Not used - validation should be in Pydantic models
        # type NonNegativeInt = int  # Not used - validation should be in Pydantic models

        # # Collection types with low/no usage
        # type Set = set[object]  # Not used
        # type Tuple = tuple[object, ...]  # Not used
        # type Data = dict[str, object]  # Duplicate of Dict
        # type JsonDict = dict[str, JsonValue]  # Duplicate of JsonObject
        # type JsonArray = list[JsonValue]  # Not used
        # type JsonPrimitive = str | int | float | bool | None  # Not used

        # # Serialization types with low usage
        # type Deserializer = Callable[[dict[str, object]], object]  # Not used

        # # Path and filesystem types - not used
        # type PathLike = str | Path  # Not used
        # type FilePath = Path  # Not used
        # type DirectoryPath = Path  # Not used

        # # Time and date types - not used
        # type Timestamp = float  # Not used
        # type IsoDateTime = str  # Not used

        # # Error and status types with low usage
        # type ErrorCode = str  # Not used
        # type StatusCode = str  # Not used

        # # Complex operation types - not used
        # type OperationCallableType = (
        #     Callable[[], object]
        #     | Callable[[object], object]
        #     | Callable[[object, object], object]
        #     | Callable[[object, object, object], object]
        # )  # Not used

        # # Performance metrics - overly complex, not used
        # type PerformanceMetrics = dict[
        #     str, dict[str, dict[str, dict[str, float | bool]]]
        # ]  # Not used - too complex and specific

    # =========================================================================
    # RESULT PATTERN TYPES - Railway-oriented programming - REFACTORED with Pydantic 2.11+ & Python 3.13+
    # =========================================================================

    class Result:
        """Result pattern types optimized for processors.py, decorators.py and legacy.py implementation.

        Refactored following FLEXT requirements with Pydantic 2.11+ and Python 3.13+
        enhancements. Only heavily used types are kept, unused types are commented out.

        Architecture Principles Applied:
            - Single Responsibility: Only result pattern type definitions
            - Open/Closed: Easy to extend with new result transformation types
            - Pydantic 2.11+ Integration: Enhanced validation and serialization
            - Python 3.13+ Type System: Modern union syntax and generic types
            - Railway-Oriented Programming: Types support FlextResult patterns
        """

        # =====================================================================
        # HEAVILY USED RESULT TYPES - Core result patterns (15 total usages)
        # =====================================================================

        # Primary result type (8 usages) - Core FlextResult type alias
        type ResultType[T] = FlextResult[
            T
        ]  # Generic result type for processors and legacy

        # Success type (5 usages) - Used in decorators.py and typings.py
        type Success[T] = FlextResult[T]  # Success result type alias

        # =====================================================================
        # COMMENTED OUT - Low usage types (will be removed at the end)
        # =====================================================================

        # # Failure and error types - not used directly in current implementation
        # type Failure = FlextResult[None]  # Not used

        # # Result transformation types - not used directly
        # type ResultMapper[T, U] = Callable[[T], FlextResult[U]]  # Not used
        # type ResultPredicate[T] = Callable[[T], bool]  # Not used

        # # Async result types - not used in current implementation
        # type AsyncResult[T] = Awaitable[FlextResult[T]]  # Not used
        # type AsyncResultMapper[T, U] = Callable[[T], Awaitable[FlextResult[U]]]  # Not used

        # # Error handling types - not used directly
        # type ErrorHandler = Callable[[Exception], str]  # Not used
        # type ErrorTransformer[T] = Callable[[T], Exception]  # Not used

    # =========================================================================
    # DOMAIN TYPES - Domain modeling - REFACTORED with Pydantic 2.11+ & Python 3.13+
    # =========================================================================

    class Domain:
        """Domain modeling types optimized for validation.py implementation.

        Refactored following FLEXT requirements with Pydantic 2.11+ and Python 3.13+
        enhancements. Only heavily used types are kept, unused types are commented out.

        Architecture Principles Applied:
            - Single Responsibility: Only domain modeling type definitions
            - Open/Closed: Easy to extend with new domain patterns
            - Pydantic 2.11+ Integration: Enhanced validation and serialization
            - Python 3.13+ Type System: Modern union syntax and literals
            - Domain-Driven Design: Types match DDD entity patterns
        """

        # =====================================================================
        # HEAVILY USED DOMAIN TYPES - Core DDD patterns (6 total usages)
        # =====================================================================

        # Primary entity identifier type (6 usages) - Enhanced with Python 3.13+ string type
        type EntityId = (
            str  # Entity identifier type (5 in validation.py, 1 in typings.py)
        )

        # =====================================================================
        # COMMENTED OUT - Low usage types (will be removed at the end)
        # =====================================================================

        # # Entity types - not used directly in current implementation
        # type EntityState = dict[str, object]  # Not used
        # type EntityVersion = int  # Not used
        # type EntityTimestamp = datetime  # Not used

        # # Value object types - not used directly
        # type ValueObjectData = dict[str, object]  # Not used

        # # Aggregate types - not used directly
        # type AggregateId = EntityId  # Not used
        # type AggregateVersion = int  # Not used

        # # Event types - not used directly
        # type EventId = str  # Not used
        # type EventType = str  # Not used
        # type EventData = dict[str, object]  # Not used
        # type EventMetadata = dict[str, object]  # Not used

        # # Correlation and tracing types - not used directly
        # type CorrelationId = str  # Not used

        # # Repository types - not used directly
        # type Specification[T] = Callable[[T], bool]  # Not used
        # type QueryCriteria = dict[str, object]  # Not used

    # =========================================================================
    # SERVICE TYPES - Service layer patterns - REFACTORED with Pydantic 2.11+ & Python 3.13+
    # =========================================================================

    class Service:
        """Service layer types optimized for container.py and commands.py implementation.

        Refactored following FLEXT requirements with Pydantic 2.11+ and Python 3.13+
        enhancements. Only heavily used types are kept, unused types are commented out.

        Architecture Principles Applied:
            - Single Responsibility: Only service layer type definitions
            - Open/Closed: Easy to extend with new service patterns
            - Pydantic 2.11+ Integration: Enhanced validation and serialization
            - Python 3.13+ Type System: Modern union syntax and literals
            - Container Integration: Types match dependency injection patterns
        """

        # =====================================================================
        # HEAVILY USED SERVICE TYPES - Core service patterns (21 total usages)
        # =====================================================================

        # Primary service instance type (11 usages) - Core service object type
        type ServiceInstance = object  # Generic service instance type

        # Container registry types (9 usages) - Enhanced with Python 3.13+ union syntax
        type ServiceDict = dict[str, ServiceInstance]  # Services registry mapping
        type FactoryDict = dict[str, Callable[[], ServiceInstance]]  # Factory registry
        type ServiceListDict = dict[
            str, Literal["instance", "factory"]
        ]  # Service type mapping

        # Service naming type (1 usage) - Used in commands.py
        type ServiceName = str  # Service identifier

        # =====================================================================
        # COMMENTED OUT - Low usage types (will be removed at the end)
        # =====================================================================

        # # Service contract types - not used in current implementation
        # type ServiceId = str  # Not used
        # type ServiceVersion = str  # Not used

        # # Service operation types - not used directly
        # type ServiceOperation = Callable[[object], object]  # Not used
        # type ServiceResult[T] = FlextTypes.Result.Success[T]  # Not used

        # # Service metadata - not used directly
        # type ServiceMetadata = dict[str, object]  # Not used
        # type ServiceConfig = dict[str, object]  # Not used

        # # Service lifecycle types - not used in current implementation
        # type ServiceState = Literal["active", "inactive", "maintenance"]  # Not used
        # type ServiceHealth = Literal["healthy", "degraded", "unhealthy"]  # Not used

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
    # VALIDATION TYPES - Specific validation system types
    # =========================================================================

    class Validation:
        """Validation-specific types based on actual FlextValidations implementation.

        These types match the actual method signatures found in validation.py
        for better type coherence between definitions and implementations.
        """

        # Email validation types
        type Email = str  # Validated email format
        type EmailValidationResult = FlextResult[
            str
        ]  # validate_email returns FlextResult[str]

        # URL validation types
        type Url = str  # Validated URL format
        type UrlValidationResult = FlextResult[str]

        # Phone validation types
        type Phone = str  # Validated phone format
        type PhoneValidationResult = FlextResult[str]

        # Numeric validation types
        type PositiveNumber = float | int  # For validate_positive
        type PositiveValidationResult = FlextResult[float | int]

        # Predicate types
        type PredicateFunction = Callable[[object], bool]
        type PredicateName = str
        type PredicateResult = FlextResult[None]

        # Pattern validation types
        type Pattern = str  # Regex pattern
        type PatternValidationResult = FlextResult[str]

        # Validation rule types
        type ValidationRule = str
        type ValidationMessage = str
        type ValidationCode = str

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
            str, str | int | float | bool | list[object] | dict[str, object]
        ]

        # Performance optimization types - matching optimize_commands_performance
        type PerformanceConfig = FlextResult[
            dict[str, str | int | float | bool | list[object] | dict[str, object]]
        ]
        type PerformanceLevel = str  # "high", "medium", "low"

        # Command bus types
        type CommandBus = object  # FlextCommands.Bus type
        type CommandHandler = object  # Handler instance

        # Command lifecycle types
        type CommandId = str
        type CommandName = str
        type CommandStatus = str
        type CommandResult = object

    # =========================================================================
    # HANDLERS TYPES - Handler system types
    # =========================================================================

    class Handlers:
        """Handlers-specific types based on actual FlextHandlers implementation.

        These types match the actual method signatures found in handlers.py
        for better type coherence between definitions and implementations.
        """

        # Thread safety types - matching thread_safe_operation
        type ThreadSafeOperation = Iterator[None]  # Context manager yield type
        type ThreadLock = object  # Threading lock object

        # Handler registry types
        type HandlerRegistry = object  # Registry instance
        type HandlerName = str
        type HandlerInstance = object

        # Handler state types
        type HandlerState = str  # "IDLE", "PROCESSING", "COMPLETED", "FAILED"
        type HandlerMetrics = dict[str, int | float | bool]

        # Handler processing types
        type HandlerRequest = dict[str, object]
        type HandlerResponse = FlextResult[object]
        type HandlerConfig = dict[str, object]

    # =========================================================================
    # AGGREGATES TYPES - Aggregate root system types - REFACTORED with Pydantic 2.11+ & Python 3.13+
    # =========================================================================

    class Aggregates:
        """Aggregates-specific types for FlextModels implementation.

        Refactored following FLEXT requirements with Pydantic 2.11+ and Python 3.13+
        enhancements. Only heavily used types are kept, unused types are commented out.

        Architecture Principles Applied:
            - Single Responsibility: Only aggregate-related type definitions
            - Open/Closed: Easy to extend with new aggregate patterns
            - Pydantic 2.11+ Integration: Enhanced validation and serialization
            - Python 3.13+ Type System: Modern union syntax and literals
            - Domain-Driven Design: Types match DDD aggregate patterns
        """

        # =====================================================================
        # HEAVILY USED AGGREGATE TYPES - Core aggregate patterns (23 total usages)
        # =====================================================================

        # Primary configuration dictionary type (17 usages) - Enhanced with Python 3.13+ union syntax
        type ConfigValue = str | int | float | bool | list[object] | dict[str, object]
        type AggregatesConfigDict = dict[
            str, ConfigValue
        ]  # Primary aggregate config type

        # Configuration result types (2 usages) - Enhanced return types for aggregate operations
        type AggregatesConfig = FlextResult[
            AggregatesConfigDict
        ]  # Primary config result
        type SystemConfig = FlextResult[AggregatesConfigDict]  # System config result

        # Environment configuration types (2 usages) - Enhanced with strict literals for Pydantic validation
        type Environment = Literal[
            "development", "production", "staging", "test", "local"
        ]
        type EnvironmentConfig = FlextResult[
            AggregatesConfigDict
        ]  # Environment-specific config result

        # Performance optimization types (2 usages) - Enhanced with strict literals
        type PerformanceLevel = Literal["low", "balanced", "high", "extreme"]
        type PerformanceConfig = FlextResult[
            AggregatesConfigDict
        ]  # Performance config result

        # =====================================================================
        # COMMENTED OUT - Low usage types (will be removed at the end)
        # =====================================================================

        # # Runtime metrics - rarely used directly
        # type RuntimeMetrics = dict[str, int | float | str]  # Runtime metrics for aggregates

        # # Domain event types - for future aggregate event sourcing
        # type EventId = str  # Not used
        # type EventType = str  # Not used
        # type EventData = dict[str, object]  # Not used
        # type AggregateId = str  # Not used
        # type AggregateVersion = int  # Not used

    # =========================================================================
    # ENHANCED CONFIG TYPES - Specific config system types
    # =========================================================================

    class ConfigSystem:
        """Config-specific types based on actual FlextConfig implementation.

        These types match the actual method signatures found in config.py
        for better type coherence between definitions and implementations.
        """

        # File loading types - matching load_and_validate_from_file
        type FilePath = str  # String file path
        type RequiredKeys = list[str] | None  # Optional required keys list
        type FileLoadResult = FlextResult[dict[str, object]]  # Load result

        # JSON operations types - matching safe_load_json_file
        type JsonPath = str | Path  # Path to JSON file
        type JsonDict = dict[str, object]  # JSON dictionary
        type JsonLoadResult = FlextResult[dict[str, object]]  # JSON load result

        # Environment variable types - matching safe_get_env_var
        type VarName = str  # Environment variable name
        type DefaultValue = str | None  # Optional default value
        type EnvResult = FlextResult[str]  # Environment result

        # Business validation types
        type BusinessRules = FlextResult[None]  # Validation result
        type ConfigDict = dict[str, object]  # Configuration dictionary

        # Settings types
        type SettingsInstance = object  # Settings instance
        type ConfigModel = object  # Config model instance

    # =========================================================================
    # CONTAINER TYPES - Dependency injection - REFACTORED with Pydantic 2.11+ & Python 3.13+
    # =========================================================================

    class Container:
        """Container types optimized for core.py implementation.

        Refactored following FLEXT requirements with Pydantic 2.11+ and Python 3.13+
        enhancements. Only heavily used types are kept, unused types are commented out.

        Architecture Principles Applied:
            - Single Responsibility: Only dependency injection type definitions
            - Open/Closed: Easy to extend with new container patterns
            - Pydantic 2.11+ Integration: Enhanced validation and serialization
            - Python 3.13+ Type System: Modern union syntax and generic types
            - Dependency Injection: Types match DI container patterns
        """

        # =====================================================================
        # HEAVILY USED CONTAINER TYPES - Core DI patterns (8 total usages)
        # =====================================================================

        # Primary service key type (3 usages) - Enhanced service identifier
        type ServiceKey = str  # Service key identifier

        # Service operation types (5 usages) - Enhanced return types for container operations
        type ServiceInstance = object  # Service instance type
        type ServiceRegistration = FlextResult[None]  # Service registration result
        type ServiceRetrieval = FlextResult[object]  # Service retrieval result
        type FactoryFunction = Callable[[], object]  # Factory callable type
        type FactoryRegistration = FlextResult[None]  # Factory registration result

        # =====================================================================
        # COMMENTED OUT - Low usage types (will be removed at the end)
        # =====================================================================

        # # Service metadata - not used directly in current implementation
        # type ServiceName = str  # Not used
        # type ServiceInfo = FlextResult[dict[str, object]]  # Not used

        # # Container management types - not used directly
        # type ContainerConfig = dict[str, object]  # Not used
        # type ServiceList = dict[str, object]  # Not used
        # type ServiceCount = int  # Not used
        # type ServiceNames = list[str]  # Not used

        # # Container operations types - not used directly
        # type ClearResult = FlextResult[None]  # Not used
        # type HasService = bool  # Not used

        # # Type-safe operations - not used in current implementation
        # type TypedRetrieval[T] = FlextResult[T]  # Not used

    # =========================================================================
    # UTILITIES TYPES - Utility functions types
    # =========================================================================

    class Utilities:
        """Utilities-specific types based on actual FlextUtilities implementation."""

        # String utilities types - matching truncate
        type Text = str
        type MaxLength = int
        type Suffix = str
        type TruncatedText = str

        # Type checking types - matching is_not_none
        type Value = object | None
        type TypeCheck = bool

        # Safe operations types
        type SafeCall[T] = FlextResult[T]
        type DefaultValue[T] = T

        # Generator types
        type UUID = str
        type Timestamp = str
        type CorrelationId = str
        type EntityId = str

    # =========================================================================
    # CONFIG TYPES - Configuration and settings
    # =========================================================================

    class Config:
        """Configuration and settings types - Optimized for Pydantic 2.11+ and Python 3.13+.

        This class provides type definitions for configuration management,
        environment settings, and deployment configuration following modern
        Python practices and enhanced validation patterns.

        Architecture Principles Applied:
            - Single Responsibility: Only configuration types
            - Open/Closed: Easy to extend with new configuration patterns
            - Pydantic 2.11+ Integration: Enhanced validation and serialization
            - Python 3.13+ Type System: Modern union syntax and literals
        """

        # =====================================================================
        # HEAVILY USED CONFIG TYPES - Primary configuration patterns
        # =====================================================================

        # Primary configuration types (360 usages) - Enhanced with Python 3.13+ syntax
        type ConfigValue = str | int | float | bool | list[object] | dict[str, object]
        type ConfigDict = dict[str, ConfigValue]  # Core config dictionary type

        # Environment type (20 usages) - Enhanced with strict literals for Pydantic validation
        type Environment = Literal[
            "development", "production", "staging", "test", "local"
        ]

        # Logging levels (8 usages) - Enhanced for structured logging
        type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        # Validation result (6 usages) - Enhanced return type for validation
        type ValidationResult = bool

        # =====================================================================
        # MODERATELY USED TYPES - Enhanced for modern usage
        # =====================================================================

        # Configuration processing (2 usages) - Enhanced result type
        type ProcessedConfig = dict[str, ConfigValue]

        # Dict alias (2 usages) - Legacy compatibility alias
        type Dict = ConfigDict

        # Configuration source and related types (used in config.py)
        type ConfigSource = str  # Configuration source type
        type ConfigPriority = int  # Configuration priority for ordering
        type ConfigNamespace = str  # Configuration namespace identifier
        type ConfigSerializer = Callable[[ConfigDict], str]  # Config serialization

        # =====================================================================
        # COMMENTED OUT - Low usage types (will be removed at the end)
        # =====================================================================

        # # Low usage configuration types (1 usage each)
        # type ConfigKey = str  # Rarely used
        # type ConfigPath = str  # Rarely used
        # type ConfigProvider = str  # Single usage
        # type ConfigFile = str  # Not used
        # type ConfigEnv = str  # Not used
        # type ConfigSection = str  # Not used

        # # Environment-specific types - mostly unused
        # type EnvironmentName = str  # Duplicate of Environment
        # type EnvironmentConfig = dict[str, ConfigValue]  # Not directly used
        # type EnvironmentVariable = str  # Not used
        # type EnvironmentValue = str  # Not used

        # # Configuration file types - not used
        # type JsonConfig = dict[str, object]  # Not used
        # type YamlConfig = dict[str, object]  # Not used
        # type TomlConfig = dict[str, object]  # Not used
        # type IniConfig = dict[str, dict[str, object]]  # Not used

        # # Configuration validation types - low usage
        # type ValidationRule = str  # Not used
        # type ValidationMessage = str  # Not used
        # type ValidationError = str  # Not used

        # # Configuration provider types - not used
        # type FileProvider = str  # Not used
        # type EnvProvider = str  # Not used
        # type CliProvider = str  # Not used
        # type DefaultProvider = str  # Not used

        # # Configuration management callable types - not used
        # type ConfigLoader = Callable[[str], ConfigDict]  # Not used
        # type ConfigMerger = Callable[[ConfigDict, ConfigDict], ConfigDict]  # Not used
        # type ConfigValidator = Callable[[ConfigDict], ValidationResult]  # Not used
        # type ConfigTransformer = Callable[[ConfigDict], ConfigDict]  # Not used
        # type ConfigDeserializer = Callable[[str], ConfigDict]  # Not used

        # # Directory and format types - low usage
        # type ConfigDir = str  # Not used
        # type ConfigFormat = Literal["json", "yaml", "toml", "ini"]  # Single usage

        # # Legacy aliases - should be phased out
        # type Config = ConfigDict  # Legacy alias - should use ConfigDict directly

        # # Deployment types - not used
        # type DeploymentMode = Literal["local", "cloud", "hybrid"]  # Not used

        # # Validation level - should be integrated with constants
        # type ValidationLevel = Literal["strict", "normal", "minimal"]  # Not used

    # =========================================================================
    # FIELD TYPES - Field system types - REFACTORED with Pydantic 2.11+ & Python 3.13+
    # =========================================================================

    class Fields:
        """Fields system types optimized for fields.py implementation.

        Refactored following FLEXT requirements with Pydantic 2.11+ and Python 3.13+
        enhancements. Only heavily used types are kept, unused types are commented out.

        Architecture Principles Applied:
            - Single Responsibility: Only field system type definitions
            - Open/Closed: Easy to extend with new field types
            - Pydantic 2.11+ Integration: Enhanced validation and serialization
            - Python 3.13+ Type System: Modern union syntax and literals
            - Type Safety: Proper type compatibility for field operations
        """

        # =====================================================================
        # HEAVILY USED FIELD TYPES - Core field patterns (76 total usages)
        # =====================================================================

        # Primary field value types (68 usages) - Enhanced with Python 3.13+ union syntax
        type String = str  # String field type (30 usages)
        type Object = object  # Generic object type (18 usages)
        type Boolean = bool  # Boolean field type (11 usages)
        type Integer = int  # Integer field type (8 usages)
        type Instance = object  # Field instance type (7 usages)

        # Secondary field types (2 usages)
        type Float = float  # Float field type (1 usage)
        type Dict = dict[str, object]  # Dictionary field type (1 usage)

        # =====================================================================
        # COMMENTED OUT - Low usage types (will be removed at the end)
        # =====================================================================

        # # Basic type aliases - not used directly in fields.py
        # type List = list[object]  # Not used
        # type Number = int | float  # Not used

        # # Field-specific type aliases - not used directly
        # type Email = str  # Not used in current implementation
        # type Uuid = str  # Not used in current implementation
        # type DateTime = datetime  # Not used in current implementation

        # # Field configuration and metadata - not used directly
        # type Config = dict[str, str | int | float | bool | datetime | None | object]  # Not used
        # type Metadata = dict[str, object]  # Not used

        # # Field validation types - not used directly
        # type ValidationResult = object  # Not used
        # type ValidationError = str  # Not used

        # # Field constraint types - not used directly
        # type Constraints = dict[str, object]  # Not used
        # type Options = dict[str, object]  # Not used

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
    # MODELS TYPES - Models system types - REFACTORED with Pydantic 2.11+ & Python 3.13+
    # =========================================================================

    class Models:
        """Models system types optimized for models.py implementation.

        Refactored following FLEXT requirements with Pydantic 2.11+ and Python 3.13+
        enhancements. Only heavily used types are kept, unused types are commented out.

        Architecture Principles Applied:
            - Single Responsibility: Only models system type definitions
            - Open/Closed: Easy to extend with new model patterns
            - Pydantic 2.11+ Integration: Enhanced validation and serialization
            - Python 3.13+ Type System: Modern union syntax and literals
            - Domain-Driven Design: Types match DDD model patterns
        """

        # =====================================================================
        # HEAVILY USED MODELS TYPES - Core model patterns (21 total usages)
        # =====================================================================

        # Primary configuration dictionary type (15 usages) - Enhanced with Python 3.13+ union syntax
        type ConfigValue = str | int | float | bool | list[object] | dict[str, object]
        type ModelsConfigDict = dict[str, ConfigValue]  # Primary models config type

        # Configuration result types (6 usages) - Enhanced return types for model operations
        type ModelsConfig = FlextResult[ModelsConfigDict]  # Models config result
        type EnvironmentModelsConfig = FlextResult[
            ModelsConfigDict
        ]  # Environment config result
        type ModelsSystemInfo = FlextResult[ModelsConfigDict]  # System info result
        type PerformanceConfig = dict[str, ConfigValue]  # Performance config input
        type OptimizedPerformanceConfig = FlextResult[
            ModelsConfigDict
        ]  # Optimized config result
        type Environment = str  # Environment name

        # =====================================================================
        # COMMENTED OUT - Low usage types (will be removed at the end)
        # =====================================================================

        # # Factory method types - not used directly in current models.py
        # type EntityData = dict[str, object]  # Not used
        # type ValueObjectData = dict[str, object]  # Not used
        # type EntityCreationResult = FlextResult[object]  # Not used
        # type ValueObjectCreationResult = FlextResult[object]  # Not used

        # # Payload creation types - not used in current implementation
        # type PayloadData = object  # Not used
        # type MessageType = str  # Not used
        # type SourceService = str  # Not used
        # type TargetService = str | None  # Not used
        # type CorrelationId = str | None  # Not used
        # type PayloadCreationResult = FlextResult[object]  # Not used

        # # Domain event creation types - not used directly
        # type EventType = str  # Not used
        # type AggregateId = str  # Not used
        # type AggregateType = str  # Not used
        # type EventData = dict[str, object]  # Not used
        # type SequenceNumber = int  # Not used
        # type DomainEventCreationResult = FlextResult[object]  # Not used

        # # Validation types - not used directly in current implementation
        # type JsonSerializableData = object  # Not used
        # type JsonValidationResult = FlextResult[object]  # Not used
        # type DateTimeValue = str | datetime  # Not used
        # type DateTimeParseResult = FlextResult[datetime]  # Not used

        # # Primitive validation types - not used in current implementation
        # type EntityIdValue = str  # Not used
        # type VersionValue = int  # Not used
        # type TimestampValue = datetime  # Not used
        # type EmailAddressValue = str  # Not used
        # type PortValue = int  # Not used
        # type HostValue = str  # Not used
        # type UrlValue = str  # Not used
        # type JsonDataValue = dict[str, object]  # Not used
        # type MetadataValue = dict[str, str]  # Not used

        # # Business rules validation types - not used directly
        # type BusinessRulesResult = FlextResult[None]  # Not used
        # type ValidationError = str  # Not used

        # # Entity and Value Object types - not used directly
        # type EntityInstance = object  # Not used
        # type ValueObjectInstance = object  # Not used
        # type AggregateRootInstance = object  # Not used

        # # Payload and Message types - not used directly
        # type PayloadInstance = object  # Not used
        # type MessageInstance = object  # Not used
        # type EventInstance = object  # Not used

    # =========================================================================
    # OBSERVABILITY TYPES - Specific observability system types
    # =========================================================================

    class Observability:
        """Observability-specific types based on actual FlextObservability implementation.

        These types match the actual method signatures found in observability.py
        for better type coherence between definitions and implementations.
        """

        # Configuration types - matching configure_observability_system
        type ObservabilityConfig = FlextResult[
            dict[str, object]
        ]  # Observability system configuration
        type ObservabilityConfigDict = dict[str, object]  # Raw configuration dictionary
        type ObservabilitySystemInfo = FlextResult[
            dict[str, object]
        ]  # System information

        # Environment configuration types - matching create_environment_observability_config
        type Environment = str  # Environment name (development, production, etc.)
        type EnvironmentObservabilityConfig = FlextResult[
            dict[str, object]
        ]  # Environment-specific config

        # Performance optimization types - matching optimize_observability_performance
        type PerformanceConfig = dict[str, object]  # Performance configuration input
        type OptimizedPerformanceConfig = FlextResult[
            dict[str, object]
        ]  # Optimized config result

        # Metrics types - matching Metrics.get_metrics
        type MetricsDict = dict[str, object]  # Metrics data structure
        type MetricName = str  # Metric identifier
        type MetricValue = int | float  # Metric value
        type MetricTags = dict[str, str]  # Metric tags

        # Tracing types
        type SpanName = str  # Trace span name
        type TraceId = str  # Trace identifier
        type SpanContext = dict[str, object]  # Span context data
        type TracingResult = FlextResult[None]  # Tracing operation result

        # Health monitoring types
        type HealthStatus = str  # Health status (healthy, degraded, unhealthy)
        type HealthCheckResult = FlextResult[dict[str, object]]  # Health check result
        type HealthMetrics = dict[str, object]  # Health metrics data

        # Alert system types
        type AlertLevel = str  # Alert severity level (info, warning, error, critical)
        type AlertMessage = str  # Alert message text
        type AlertContext = dict[str, object]  # Alert context information
        type AlertResult = FlextResult[None]  # Alert operation result

        # Console logging types
        type LogLevel = str  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        type LogMessage = str  # Log message text
        type LogContext = dict[str, object]  # Logging context data
        type LogResult = FlextResult[None]  # Logging operation result

        # Observability instance types
        type ObservabilityInstance = object  # Observability instance type
        type GlobalObservability = object  # Global observability instance

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
# PUBLIC TYPE ALIASES - For backward compatibility and ease of use
# =============================================================================

# Generic callable type with type parameter - for decorator compatibility
# Use FlextTypes.Core.FlextCallableType directly - no aliases


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
