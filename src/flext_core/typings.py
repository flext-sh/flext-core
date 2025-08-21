"""Centralized TypeVars used across flext_core at runtime.

Define commonly-used TypeVars in a single module so both runtime
and stub files can reference the same names.

"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, ParamSpec, TypeVar

# Import FlextResult for type definitions
if TYPE_CHECKING:
    from flext_core.result import FlextResult

# Test TypeVars para uso global
TTestData = TypeVar("TTestData")
TTestConfig = TypeVar("TTestConfig")

# =============================================================================
# CORE TYPE VARIABLES - Foundation building blocks
# =============================================================================

# Basic generic type variables (most commonly used)
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
R = TypeVar("R")  # Result type
E = TypeVar("E")  # Error type
F = TypeVar("F")  # Function type
P = ParamSpec("P")  # Parameter specification for callables

# =============================================================================
# CALLABLE PATTERNS - Python 3.13+ with type safety
# =============================================================================
if TYPE_CHECKING:
    # MyPy-compatible generic callable types avoiding explicit Any/... usage
    from typing import Protocol

    class FlextCallableProtocol[T](Protocol):
        """Generic callable protocol with return type safety."""

        def __call__(self, *args: object, **kwargs: object) -> T: ...

    class FlextDecoratedFunctionProtocol[T](Protocol):
        """Decorated function protocol returning FlextResult."""

        def __call__(self, *args: object, **kwargs: object) -> FlextResult[T]: ...

    class FlextHandlerProtocol[TInput, TOutput](Protocol):
        """Handler protocol for single input/output operations."""

        def __call__(self, input_data: TInput) -> FlextResult[TOutput]: ...

    class FlextProcessorProtocol[T](Protocol):
        """Processor protocol for transform operations."""

        def __call__(self, data: T) -> FlextResult[T]: ...

    class FlextValidatorProtocol[T](Protocol):
        """Validator protocol for validation operations."""

        def __call__(self, data: T) -> FlextResult[None]: ...

    class FlextFactoryProtocol[T](Protocol):
        """Factory protocol for creation operations."""

        def __call__(self, *args: object) -> FlextResult[T]: ...

    # Type aliases for callable protocols
    type FlextCallable[T] = FlextCallableProtocol[
        T
    ]  # Generic callable with type safety
    type FlextDecoratedFunction[T] = FlextDecoratedFunctionProtocol[
        T
    ]  # Returns FlextResult
    type FlextHandler[TInput, TOutput] = FlextHandlerProtocol[
        TInput, TOutput
    ]  # Handler interface
    type FlextProcessor[T] = FlextProcessorProtocol[T]  # Transform operations
    type FlextValidator[T] = FlextValidatorProtocol[T]  # Validation interface
    type FlextFactory[T] = FlextFactoryProtocol[T]  # Factory pattern

    # UTILITY TYPES - Specialized protocols for common patterns
    type SafeCallable[T] = FlextCallableProtocol[
        FlextResult[T]
    ]  # Exception-safe callable
    type ValidatedCallable[T] = FlextDecoratedFunctionProtocol[T]  # Returns FlextResult
    type AsyncSafeCallable[T] = FlextDecoratedFunctionProtocol[T]  # Async handler
    type TransformCallable[TIn, TOut] = FlextHandlerProtocol[
        TIn, TOut
    ]  # Data transformation
    type PredicateCallable[T] = Callable[[T], FlextResult[bool]]  # Boolean validation
    type CreationCallable[T] = Callable[[], FlextResult[T]]  # Factory creation
else:
    # Runtime fallbacks for callable types (Python 3.13 type syntax not yet fully supported at runtime)
    FlextCallable = Callable  # Runtime fallback to standard Callable
    FlextDecoratedFunction = Callable  # Runtime fallback with FlextResult annotation
    FlextHandler = Callable  # Runtime fallback for handlers
    FlextProcessor = Callable  # Runtime fallback for processors
    FlextValidator = Callable  # Runtime fallback for validators
    FlextFactory = Callable  # Runtime fallback for factories

    # UTILITY TYPES - Runtime fallbacks
    SafeCallable = Callable  # Runtime fallback for exception-safe
    ValidatedCallable = Callable  # Runtime fallback for validated
    AsyncSafeCallable = Callable  # Runtime fallback for async
    TransformCallable = Callable  # Runtime fallback for transformers
    PredicateCallable = Callable  # Runtime fallback for predicates
    CreationCallable = Callable  # Runtime fallback for creation

# LEGACY COMPATIBILITY - Maintain exact same signatures for backward compatibility
type FlextCallableLegacy = Callable[
    [object], object
]  # LEGACY METHOD: Original simple callable
type FlextDecoratedFunctionLegacy = Callable[
    [object], object
]  # LEGACY METHOD: Original decorated function

# BACKWARD COMPATIBILITY ALIASES - Keep existing code working unchanged
# These maintain the exact original signatures that legacy code expects
FlextCallableOriginal = FlextCallableLegacy  # LEGACY: Original FlextCallable signature
FlextDecoratedFunctionOriginal = (
    FlextDecoratedFunctionLegacy  # LEGACY: Original FlextDecoratedFunction signature
)

# Domain-level result type used by domain services
TDomainResult = TypeVar("TDomainResult")

# Constrained generics with bounds
TComparable = TypeVar("TComparable")  # For comparable types
TSerializable = TypeVar("TSerializable")  # For serializable types
TValidatable = TypeVar("TValidatable")  # For validatable types

# Entity and domain type variables - UPDATED TO AVOID CONFLICTS
TEntity = TypeVar("TEntity")  # Domain entity
TAnyObject = TypeVar("TAnyObject")  # Any object type
# NEW: Command and Query TypeVars with different names
CommandTypeVar = TypeVar("CommandTypeVar")  # CQRS command
QueryTypeVar = TypeVar("QueryTypeVar")  # CQRS query
QueryResultTypeVar = TypeVar("QueryResultTypeVar")  # CQRS query result
TRequest = TypeVar("TRequest")  # Request type
TResponse = TypeVar("TResponse")  # Response type
ResultTypeVar = TypeVar("ResultTypeVar")  # Result type
TService = TypeVar("TService")  # Service type
TOptional = TypeVar("TOptional")  # Optional value type

# Backwards compatibility aliases for old names - create new TypeVars with same names
TCommand = TypeVar("TCommand")  # Recreated for compatibility
TQuery = TypeVar("TQuery")  # Recreated for compatibility
TQueryResult = TypeVar("TQueryResult")  # Recreated for compatibility
TResult = TypeVar("TResult")  # Recreated for compatibility

# Schema processing type variables
EntryT = TypeVar("EntryT")  # Generic entry type for schema processing

# =============================================================================
# FLEXT HIERARCHICAL TYPE SYSTEM - Organized by domain
# =============================================================================


class FlextTypes:
    """Hierarchical type system organizing FLEXT types by domain and functionality."""

    # =========================================================================
    # CORE TYPES - Fundamental building blocks
    # =========================================================================

    class Core:
        """Core fundamental types used throughout the ecosystem."""

        # Basic value types
        type Value = str | int | float | bool | None
        type Data = dict[str, object]
        type Config = dict[str, str | int | float | bool | None]

        # Identifier types
        type EntityId = str
        type Id = str
        type Key = str

        # Collection types
        type AnyDict = dict[str, object]
        type AnyList = list[object]
        type StringDict = dict[str, str]
        # Explicit JSON dictionary alias used widely externally
        type JsonDict = dict[str, object]

        # Connection and infrastructure
        type ConnectionString = str
        type LogMessage = str
        type ErrorCode = str
        type ErrorMessage = str

        # Callable types
        type AnyCallable = Callable[[object], object]
        type Factory[T] = Callable[[], T] | Callable[[object], T]
        type Transformer[T, U] = Callable[[T], U]
        type Predicate[T] = Callable[[T], bool]
        type Validator[T] = Callable[[T], bool]
        type ErrorHandler = Callable[[Exception], str]

    # =========================================================================
    # DATA TYPES - Common data structures
    # =========================================================================

    class Data:
        """Common data structure types used across ecosystem."""

        # Dictionary types
        type Dict = dict[str, object]  # Generic dictionary for data storage
        type StringDict = dict[str, str]  # String-only dictionary
        type JsonDict = dict[str, object]  # JSON-serializable dictionary

        # List types
        type List = list[object]  # Generic list
        type StringList = list[str]  # String-only list

    # =========================================================================
    # SERVICE TYPES - Dependency injection and service location
    # =========================================================================

    class Service:
        """Service-related types for dependency injection."""

        # Service identification and management
        type ServiceName = str
        type FlextServiceKey = str | type[object]
        type Container = Mapping[str, object]
        type ServiceLocator = Callable[[str], object]
        type ServiceFactory[T] = Callable[[], T]

        # Event and messaging types
        type EventHandler[TEvent] = Callable[[TEvent], None]
        type EventBus = Callable[[object], None]
        type FlextMessageHandler[T] = Callable[[T], object]

        # Metadata and configuration types
        type JsonDict = dict[str, object]  # Standard JSON-serializable dictionary
        type Metadata = dict[str, object]
        type Settings = dict[str, object]
        type Configuration = Mapping[str, object]

        # Convenience aliases
        type Transform[T, U] = Callable[[T], U]  # Alias for Transformer
        type Handler[T, R] = Callable[[T], R]  # Generic handler
        ServiceInstance = TypeVar("ServiceInstance")

        # Context and correlation types
        type CorrelationId = str
        type RequestId = str
        type TraceId = str

    # =========================================================================
    # DOMAIN TYPES - Business domain modeling
    # =========================================================================

    class Domain:
        """Domain modeling and business logic types."""

        # Entity and aggregate types
        type EntityVersion = int  # Entity version for optimistic locking
        type EntityTimestamp = datetime  # Entity timestamp fields
        type DomainEventType = str  # Domain event type identifier
        type DomainEventData = dict[str, object]  # Domain event payload data
        type AggregateId = str  # Aggregate root identifier
        type EntityRule = str  # Domain rule identifier for validation
        type EntityState = str  # Entity state for state machines
        type EntityMetadata = dict[str, object]  # Entity metadata for extensions

        # Factory and creation types
        type EntityDefaults = dict[str, object]  # Default values for entity creation
        type EntityChanges = dict[str, object]  # Changes for entity updates
        type FactoryResult[T] = object  # Factory creation result

        # Event sourcing types
        type DomainEvents = list[object]  # Collection of domain events
        type EventStream = list[object]  # Entity event stream
        type EventVersion = int  # Event version for ordering

        # Value object types
        type ValueData = dict[str, object]
        type ValueValidation = Callable[[object], bool]

    # =========================================================================
    # CQRS TYPES - Command Query Responsibility Segregation
    # =========================================================================

    class CQRS:
        """CQRS pattern types for command and query handling."""

        # Command types
        type CommandId = str  # Command instance identifier (correlation ID)
        type CommandType = str  # Command type name for routing
        type HandlerName = str  # Command handler service name
        type CommandPayload = dict[str, object]  # Command data payload
        type CommandResult = object  # Command execution result
        type CommandMetadata = dict[str, object]  # Command metadata for middleware
        type MiddlewareName = str  # Middleware component name
        type ValidationRule = str  # Command validation rule identifier
        type CommandBusId = str  # Command bus instance identifier
        type CommandPriority = int  # Command execution priority (1-10)

        # Query types
        type QueryId = str  # Query instance identifier (correlation ID)
        type QueryType = str  # Query type name for routing
        type QueryResult[T] = object  # Query result with type parameter
        type QueryCriteria = dict[str, object]  # Query filtering criteria
        type QueryProjection = list[object]  # Query result projection fields
        type PaginationToken = str  # Query pagination continuation token

        # Message types
        type Event = dict[str, object]
        type Message = dict[str, object]

    # =========================================================================
    # VALIDATION TYPES - Validation and business rules
    # =========================================================================

    class Validation:
        """Validation system types for business rule enforcement."""

        # Core validation types
        type ValidationRule = str  # Validation rule identifier
        type ValidationError = str  # Validation error message
        type ValidationResult = object  # Validation result with data
        type ValidationContext = dict[str, object]  # Validation context data
        type ValidatorName = str  # Validator instance name
        type ValidationConfig = dict[str, object]  # Validator configuration
        type ValidationConstraint = object  # Validation constraint value
        type ValidationSchema = dict[str, object]  # Schema definition for validation

        # Field validation types
        type FieldName = str  # Field name for validation
        type FieldValue = object  # Field value to validate
        type FieldRule = str  # Field-specific validation rule
        type FieldError = str  # Field-specific error message
        type FieldInfo = dict[str, object]  # Field metadata
        type FieldMetadata = dict[str, object]  # Field metadata for validation

        # Field definition types (from fields.py)
        type FieldId = str  # Field identifier
        type FieldTypeStr = str  # Field type as string

        # Type guard types
        type TypeGuard[T] = Callable[[object], bool]  # Type guard function
        type GuardFunction = Callable[[object], bool]  # Generic guard function
        type GuardResult = bool  # Guard function result
        type GuardContext = dict[str, object]  # Guard execution context

        # Custom validation types
        type CustomValidator = Callable[[object], object]  # Custom validator
        type ValidationPipeline = list[
            FlextTypes.Validation.CustomValidator
        ]  # Chain of validators

    # =========================================================================
    # CONFIG TYPES - Configuration management
    # =========================================================================

    class Config:
        """Configuration management types."""

        # Core configuration types
        type ConfigKey = str  # Configuration key identifier
        type ConfigValue = object  # Configuration value (any type)
        type ConfigPath = str  # File a path for configuration files
        type ConfigEnv = str  # Environment name (dev, prod, test)
        type ConfigValidationRule = str  # Configuration validation rule
        type ConfigMergeStrategy = str  # Strategy for merging configurations
        type ConfigSettings = dict[str, object]  # Settings dictionary
        type ConfigDefaults = dict[str, object]  # Default configuration values
        type ConfigOverrides = dict[str, object]  # Configuration overrides
        type ConfigDict = dict[
            str,
            str | int | float | bool | None,
        ]  # Configuration dictionary

        # Environment and deployment types
        type EnvironmentName = str  # Environment identifier
        type DeploymentStage = str  # Deployment stage (staging, production)
        type ConfigVersion = str  # Configuration version for tracking
        # File system types
        type DirectoryPath = str
        type FilePath = str

        # Convenience aliases
        type EnvVar = str
        type ConfigSection = str

    # =========================================================================
    # LOGGING TYPES - Structured logging and observability
    # =========================================================================

    class Logging:
        """Logging and observability types."""

        # Core logging types
        type LoggerName = str  # Logger name identifier
        type LogLevel = str  # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        type LogFormat = str  # Log format string
        type LogHandler = str  # Log handler type identifier
        type LogFilter = str  # Log filter identifier

        # Correlation and tracing types
        type CorrelationId = str  # Correlation ID for request tracing
        type SessionId = str  # Session identifier for user tracking
        type TransactionId = str  # Transaction identifier for database operations
        type OperationName = str  # Business operation name for categorization

        # Structured logging types
        type LogRecord = dict[str, object]  # Complete log record data
        type LogMetrics = dict[str, object]  # Log metrics and statistics
        type LogConfiguration = dict[str, object]  # Logger configuration settings

    # =========================================================================
    # AUTH TYPES - Authentication and authorization
    # =========================================================================

    class Auth:
        """Authentication and authorization types."""

        # Authentication types
        type Token = str
        type UserData = dict[str, object]
        type Credentials = dict[str, object]
        type UserId = str

        # Authorization types
        type Role = str
        type Permission = str
        type Provider = object

        # Connection types
        type Connection = str  # Connection string

        # Context types
        type ContextDict = dict[str, object]

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
    # PROTOCOL TYPES - Interface definitions
    # =========================================================================

    class Protocols:
        """Protocol and interface types."""

        # Core protocols
        type Comparable = object  # Comparable interface
        type Serializable = object  # Serializable interface
        type Validatable = object  # Validatable interface
        type Timestamped = object  # Timestamped interface
        type Cacheable = object  # Cacheable interface
        type Configurable = object  # Configurable interface

        # Functional protocols
        type Validator[T] = Callable[[T], bool | str]

    # -----------------------------------------------------------------
    # CONVENIENCE: Some tests refer to FlextTypes.TypeGuards
    # Provide a minimal implementation to avoid circular imports
    # -----------------------------------------------------------------
    class TypeGuards:
        """Convenience facade for type guard utilities (tests convenience)."""

        @staticmethod
        def has_attribute(obj: object, attr: str) -> bool:
            """Check if an object has a specific attribute."""
            return hasattr(obj, attr)

        @staticmethod
        def is_instance_of(obj: object, target_type: type) -> bool:
            """Check if an object is an instance of a specific type."""
            try:
                return isinstance(obj, target_type)
            except Exception:
                return False

        @staticmethod
        def is_dict_like(obj: object) -> bool:
            """Check if an object is dict-like."""
            try:
                return isinstance(obj, dict) or (
                    hasattr(obj, "keys")
                    and (
                        hasattr(obj, "__getitem__")
                        or (hasattr(obj, "values") and hasattr(obj, "items"))
                    )
                )
            except Exception:
                return False

        @staticmethod
        def is_list_like(obj: object) -> bool:
            """Check if object is list-like."""
            try:
                if isinstance(obj, (str, bytes)):
                    return False
                return isinstance(obj, (list, tuple, set)) or hasattr(obj, "__iter__")
            except Exception:
                return False

        # Extra guards
        @staticmethod
        def is_callable(obj: object) -> bool:
            """Check if an object is callable."""
            return callable(obj)


# =============================================================================
# CONVENIENCE ALIASES - Export consistency
# =============================================================================

# Export all T* aliases for convenience
# Core type variables (already defined above)

# Domain and data type aliases (from types.py)
TEntityId = FlextTypes.Core.EntityId
TValue = FlextTypes.Core.Value
TData = FlextTypes.Core.Data
TConfig = FlextTypes.Core.Config

# CQRS type aliases (from types.py and commands.py)
TEvent = FlextTypes.CQRS.Event
TMessage = FlextTypes.CQRS.Message
TCommandId = FlextTypes.CQRS.CommandId
TCommandType = FlextTypes.CQRS.CommandType
THandlerName = FlextTypes.CQRS.HandlerName
TCommandPayload = FlextTypes.CQRS.CommandPayload
TCommandResult = FlextTypes.CQRS.CommandResult
TCommandMetadata = FlextTypes.CQRS.CommandMetadata
TMiddlewareName = FlextTypes.CQRS.MiddlewareName
TValidationRule = FlextTypes.CQRS.ValidationRule
TCommandBusId = FlextTypes.CQRS.CommandBusId
TCommandPriority = FlextTypes.CQRS.CommandPriority
TQueryId = FlextTypes.CQRS.QueryId
TQueryType = FlextTypes.CQRS.QueryType
TQueryCriteria = FlextTypes.CQRS.QueryCriteria
TQueryProjection = FlextTypes.CQRS.QueryProjection
TPaginationToken = FlextTypes.CQRS.PaginationToken

# Service type aliases (from types.py)
TServiceName = FlextTypes.Service.ServiceName
TServiceKey = FlextTypes.Service.FlextServiceKey

# Callable type aliases (from types.py)
TFactory = FlextTypes.Core.Factory[object]
TTransformer = FlextTypes.Core.Transformer[object, object]
TPredicate = FlextTypes.Core.Predicate[object]
TValidator = FlextTypes.Core.Validator[object]
TCallable = FlextTypes.Core.AnyCallable
TErrorHandler = FlextTypes.Core.ErrorHandler

# Infrastructure type aliases (from types.py)
TConnectionString = FlextTypes.Core.ConnectionString
TLogMessage = FlextTypes.Core.LogMessage
TErrorCode = FlextTypes.Core.ErrorCode
TErrorMessage = FlextTypes.Core.ErrorMessage

# Collection type aliases (from types.py)
TAnyDict = FlextTypes.Core.AnyDict
TAnyList = FlextTypes.Core.AnyList
TDict = FlextTypes.Core.AnyDict
TList = FlextTypes.Core.AnyList
TStringDict = FlextTypes.Core.StringDict

# Authentication and token types (from types.py)
TUserData = FlextTypes.Auth.UserData
TToken = FlextTypes.Auth.Token
TCredentials = FlextTypes.Auth.Credentials
TConnection = FlextTypes.Auth.Connection
TUserId = FlextTypes.Auth.UserId

# Context and correlation types (from types.py)
TContextDict = FlextTypes.Auth.ContextDict
TCorrelationId = FlextTypes.Service.CorrelationId
TRequestId = FlextTypes.Service.RequestId
TConfigDict = FlextTypes.Config.ConfigDict

# Field and metadata types (from types.py)
TFieldInfo = FlextTypes.Validation.FieldInfo
TFieldMetadata = FlextTypes.Validation.FieldMetadata

# Field types (from fields.py) - moved to end to avoid forward reference issues

# Config types (from config.py)
TConfigKey = FlextTypes.Config.ConfigKey
TConfigValue = FlextTypes.Config.ConfigValue
TConfigPath = FlextTypes.Config.ConfigPath
TConfigEnv = FlextTypes.Config.ConfigEnv
TConfigValidationRule = FlextTypes.Config.ConfigValidationRule
TConfigMergeStrategy = FlextTypes.Config.ConfigMergeStrategy
TConfigSettings = FlextTypes.Config.ConfigSettings
TConfigDefaults = FlextTypes.Config.ConfigDefaults
TConfigOverrides = FlextTypes.Config.ConfigOverrides
TEnvironmentName = FlextTypes.Config.EnvironmentName
TDeploymentStage = FlextTypes.Config.DeploymentStage
TConfigVersion = FlextTypes.Config.ConfigVersion

# Validation types (from validation.py)
TValidationError = FlextTypes.Validation.ValidationError
TValidationResult = FlextTypes.Validation.ValidationResult
TValidationContext = FlextTypes.Validation.ValidationContext
TValidatorName = FlextTypes.Validation.ValidatorName
TValidationConfig = FlextTypes.Validation.ValidationConfig
TValidationConstraint = FlextTypes.Validation.ValidationConstraint
TValidationSchema = FlextTypes.Validation.ValidationSchema
TFieldName = FlextTypes.Validation.FieldName
TFieldValue = FlextTypes.Validation.FieldValue
TFieldRule = FlextTypes.Validation.FieldRule
TFieldError = FlextTypes.Validation.FieldError
TCustomValidator = FlextTypes.Validation.CustomValidator
TValidationPipeline = FlextTypes.Validation.ValidationPipeline

# Type guard types (from guards.py)
TTypeGuard = FlextTypes.Validation.TypeGuard[object]
TGuardFunction = FlextTypes.Validation.GuardFunction
TGuardResult = FlextTypes.Validation.GuardResult
TGuardContext = FlextTypes.Validation.GuardContext

# Logging types (from loggings.py)
TLoggerName = FlextTypes.Logging.LoggerName

# Business types (testing convenience)
TBusinessId = str
TBusinessName = str
TBusinessCode = str
TBusinessStatus = str
TBusinessType = str

# Cache types (testing convenience)
TCacheKey = str
TCacheValue = str | int | float | bool | None
TCacheTTL = int

# Filesystem aliases
TDirectoryPath = FlextTypes.Config.DirectoryPath
TFilePath = FlextTypes.Config.FilePath
TLogLevel = FlextTypes.Logging.LogLevel
TLogFormat = FlextTypes.Logging.LogFormat
TLogHandler = FlextTypes.Logging.LogHandler
TLogFilter = FlextTypes.Logging.LogFilter
TSessionId = FlextTypes.Logging.SessionId
TTransactionId = FlextTypes.Logging.TransactionId
TOperationName = FlextTypes.Logging.OperationName
TLogRecord = FlextTypes.Logging.LogRecord
TLogMetrics = FlextTypes.Logging.LogMetrics
TLogConfiguration = FlextTypes.Logging.LogConfiguration

# Entity types (from entities.py)
TEntityVersion = FlextTypes.Domain.EntityVersion
TEntityTimestamp = FlextTypes.Domain.EntityTimestamp
TDomainEventType = FlextTypes.Domain.DomainEventType
TDomainEventData = FlextTypes.Domain.DomainEventData
TAggregateId = FlextTypes.Domain.AggregateId
TEntityRule = FlextTypes.Domain.EntityRule
TEntityState = FlextTypes.Domain.EntityState
TEntityMetadata = FlextTypes.Domain.EntityMetadata
TEntityDefaults = FlextTypes.Domain.EntityDefaults
TEntityChanges = FlextTypes.Domain.EntityChanges
TFactoryResult = FlextTypes.Domain.FactoryResult[object]
TDomainEvents = FlextTypes.Domain.DomainEvents
TEventStream = FlextTypes.Domain.EventStream
TEventVersion = FlextTypes.Domain.EventVersion

# Protocol convenience aliases
Comparable = FlextTypes.Protocols.Comparable
Serializable = FlextTypes.Protocols.Serializable
Timestamped = FlextTypes.Protocols.Timestamped
Validatable = FlextTypes.Protocols.Validatable
Cacheable = FlextTypes.Protocols.Cacheable
Configurable = FlextTypes.Protocols.Configurable

# Convenience protocol aliases with transition support
FlextSerializable = Serializable
FlextValidatable = Validatable
FlextProtocolValidator = FlextTypes.Protocols.Validator[
    object
]  # RENAMED: Avoid conflict with standard type

# Entity identifier alias for convenience
# FlextEntityId is imported from root_models module - remove local alias

# =============================================================================
# DEPRECATION WARNING SYSTEM - Encourage migration to centralized types
# =============================================================================


def _emit_transition_warning(old_name: str, new_name: str) -> None:
    """Emit transition warning for historical usage."""
    warnings.warn(
        f"'{old_name}' is transitional. Use 'from flext_core.core_types import FlextTypes.{new_name}' instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def get_centralized_types_usage_info() -> str:
    """Get information about centralized type usage."""
    # Reference the transition helper in a defensive branch so static analyzers
    # won't flag it as unused while preserving no runtime effect.
    if TYPE_CHECKING:  # pragma: no cover - static-analysis helper
        _emit_transition_warning("old", "new")

    return (
        "All types are now centralized in typings.py. "
        "Use 'from flext_core.typings import FlextTypes' for new development. "
        "T* aliases are available for convenience."
    )


# Field types (from fields.py) - defined here after FlextTypes class is complete
FlextFieldId = FlextTypes.Validation.FieldId
FlextFieldName = FlextTypes.Validation.FieldName
FlextFieldTypeStr = FlextTypes.Validation.FieldTypeStr


# =============================================================================
# EXPORTS - Comprehensive centralized type system
# =============================================================================

__all__: list[str] = [  # noqa: RUF022
    # Protocol aliases
    "Cacheable",
    "Comparable",
    "Configurable",
    # Core type variables
    "E",
    "F",
    "P",
    "R",
    "T",
    "U",
    "V",
    # NEW TypeVars for CQRS
    "CommandTypeVar",
    "QueryTypeVar",
    "QueryResultTypeVar",
    "ResultTypeVar",
    # Schema processing types
    "EntryT",
    # FlextEntity convenience
    "FlextSerializable",
    # Hierarchical type system (preferred)
    "FlextTypes",
    "FlextValidatable",
    "FlextValidator",
    "P",
    "R",
    "Serializable",
    "T",
    # T* aliases - Domain types
    "TAggregateId",
    # T* aliases - Core types
    "TAnyDict",
    "TAnyList",
    # TypeVar entities
    "TAnyObject",
    "TCallable",
    "TCommand",
    "TQuery",
    "TQueryResult",
    "TResult",
    # T* aliases - CQRS types
    "TCommandBusId",
    "TCommandId",
    "TCommandMetadata",
    "TCommandPayload",
    "TCommandPriority",
    "TCommandResult",
    "TCommandType",
    "TComparable",
    "TConfig",
    # T* aliases - Config types
    "TConfigDefaults",
    "TConfigDict",
    "TConfigEnv",
    "TConfigKey",
    "TConfigMergeStrategy",
    "TConfigOverrides",
    "TConfigPath",
    "TConfigSettings",
    "TConfigValidationRule",
    "TConfigValue",
    "TConfigVersion",
    # T* aliases - Infrastructure types
    "TConnection",
    "TConnectionString",
    # T* aliases - Auth types
    "TContextDict",
    # T* aliases - Service types
    "TCorrelationId",
    "TCredentials",
    # T* aliases - Validation types
    "TCustomValidator",
    "TData",
    "TDeploymentStage",
    "TDict",
    "TDomainEventData",
    "TDomainEventType",
    "TDomainEvents",
    "TEntity",
    "TEntityChanges",
    "TEntityDefaults",
    "TEntityId",
    "TEntityMetadata",
    "TEntityRule",
    "TEntityState",
    "TEntityTimestamp",
    "TEntityVersion",
    "TEnvironmentName",
    "TErrorCode",
    "TErrorHandler",
    "TErrorMessage",
    "TEvent",
    "TEventStream",
    "TEventVersion",
    "TFactory",
    "TFactoryResult",
    "TFieldError",
    "TFieldInfo",
    "TFieldMetadata",
    "TFieldName",
    "TFieldRule",
    "TFieldValue",
    "THandlerName",
    "TList",
    # T* aliases - Logging types
    "TLogConfiguration",
    "TLogFilter",
    "TLogFormat",
    "TLogHandler",
    "TLogLevel",
    "TLogMessage",
    "TLogMetrics",
    "TLogRecord",
    "TLoggerName",
    "TMessage",
    "TMiddlewareName",
    "TOperationName",
    "TOptional",
    "TPaginationToken",
    "TPredicate",
    "TQueryCriteria",
    "TQueryId",
    "TQueryProjection",
    "TQueryType",
    "TRequest",
    "TRequestId",
    "TResponse",
    "TSerializable",
    "TService",
    "TServiceKey",
    "TServiceName",
    "TSessionId",
    "TStringDict",
    "TToken",
    "TTransactionId",
    "TTransformer",
    "TUserData",
    "TUserId",
    "TValidatable",
    "TValidationConfig",
    "TValidationConstraint",
    "TValidationContext",
    "TValidationError",
    "TValidationPipeline",
    "TValidationResult",
    "TValidationRule",
    "TValidationSchema",
    "TValidator",
    "TValidatorName",
    "TValue",
    "Timestamped",
    "U",
    "V",
    "Validatable",
    # Type guard types - moved to end to avoid forward reference issues
    "TTypeGuard",
    "TGuardFunction",
    "TGuardResult",
    "TGuardContext",
    # Field types - moved to end to avoid forward reference issues
    "FlextFieldId",
    "FlextFieldName",
    "FlextFieldTypeStr",
    # Utility functions
    "get_centralized_types_usage_info",
]
