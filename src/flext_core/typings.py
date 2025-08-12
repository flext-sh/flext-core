"""Centralized type definitions for the FLEXT ecosystem.

Provides comprehensive type system following PEP8 naming conventions.
Single source of truth for all type definitions across 32+ FLEXT projects.

Architecture:
    - FlextTypes: Hierarchical type system organized by domain
    - Type Variables: Generic type parameters (T, U, V, etc.)
    - Legacy Aliases: Backward compatibility T* naming convention
    - Protocol Types: Interface definitions and contracts

Usage:
    from flext_core.typings import FlextTypes, T, TAnyDict
    # Preferred: FlextTypes.Core.AnyDict
    # Legacy: TAnyDict
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TypeVar

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
P = TypeVar("P")  # Predicate type

# Constrained generics with bounds
TComparable = TypeVar("TComparable")  # For comparable types
TSerializable = TypeVar("TSerializable")  # For serializable types
TValidatable = TypeVar("TValidatable")  # For validatable types

# Entity and domain type variables
TEntity = TypeVar("TEntity")  # Domain entity
TAnyObject = TypeVar("TAnyObject")  # Any object type
TCommand = TypeVar("TCommand")  # CQRS command
TQuery = TypeVar("TQuery")  # CQRS query
TRequest = TypeVar("TRequest")  # Request type
TResponse = TypeVar("TResponse")  # Response type
TResult = TypeVar("TResult")  # Result type
TService = TypeVar("TService")  # Service type
TOptional = TypeVar("TOptional")  # Optional value type

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

        # Legacy aliases
        type Dict = FlextTypes.Core.AnyDict  # Backward compatibility
        type List = FlextTypes.Core.AnyList  # Backward compatibility

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
        type JsonDict = dict[str, object]  # JSON-compatible dictionary

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
        type JsonDict = dict[str, object]  # Standard JSON-compatible dictionary
        type Metadata = dict[str, object]
        type Settings = dict[str, object]
        type Configuration = Mapping[str, object]

        # Legacy compatibility
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
        # File system types (legacy)
        type DirectoryPath = str
        type FilePath = str

        # Legacy compatibility
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
    # LEGACY: Some tests refer to FlextTypes.TypeGuards
    # Provide a minimal implementation to avoid circular imports
    # -----------------------------------------------------------------
    class TypeGuards:
        """Legacy facade for type guard utilities (tests compatibility)."""

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

        # Extra guards used by tests
        @staticmethod
        def is_callable(obj: object) -> bool:
            """Check if an object is callable."""
            return callable(obj)


# =============================================================================
# BACKWARD COMPATIBILITY ALIASES - Legacy support during migration
# =============================================================================

# Export all legacy T* aliases for backward compatibility
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
TQueryResult = FlextTypes.CQRS.QueryResult[object]
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
TDict = FlextTypes.Core.Dict
TList = FlextTypes.Core.List
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

# Field types (from fields.py)
FlextFieldId = FlextTypes.Validation.FieldId
FlextFieldName = FlextTypes.Validation.FieldName
FlextFieldTypeStr = FlextTypes.Validation.FieldTypeStr

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

# Logging types (from loggings.py)
TLoggerName = FlextTypes.Logging.LoggerName

# Business types (legacy/testing convenience)
TBusinessId = str
TBusinessName = str
TBusinessCode = str
TBusinessStatus = str
TBusinessType = str

# Cache types (legacy/testing convenience)
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

# Protocol compatibility aliases
Comparable = FlextTypes.Protocols.Comparable
Serializable = FlextTypes.Protocols.Serializable
Timestamped = FlextTypes.Protocols.Timestamped
Validatable = FlextTypes.Protocols.Validatable
Cacheable = FlextTypes.Protocols.Cacheable
Configurable = FlextTypes.Protocols.Configurable

# Legacy protocol aliases with deprecation support
FlextSerializable = Serializable
FlextValidatable = Validatable
FlextValidator = FlextTypes.Protocols.Validator[object]

# Entity identifier alias for backward compatibility
FlextEntityId = TEntityId

# =============================================================================
# DEPRECATION WARNING SYSTEM - Encourage migration to centralized types
# =============================================================================


def _emit_deprecation_warning(old_name: str, new_name: str) -> None:
    """Emit deprecation warning for legacy usage."""
    warnings.warn(
        f"'{old_name}' is deprecated. "
        f"Use 'from flext_core.core_types import FlextTypes.{new_name}' instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def get_centralized_types_usage_info() -> str:
    """Get information about centralized type usage."""
    return (
        "All types are now centralized in typings.py. "
        "Use 'from flext_core.typings import FlextTypes' for new development. "
        "Legacy T* aliases are available for backward compatibility."
    )


# =============================================================================
# EXPORTS - Comprehensive centralized type system
# =============================================================================

__all__: list[str] = [
    # Protocol aliases
    "Cacheable",
    "Comparable",
    "Configurable",
    # Core type variables
    "E",
    # Schema processing types
    "EntryT",
    "F",
    # FlextEntity compatibility
    "FlextEntityId",
    "FlextSerializable",
    # Hierarchical type system (preferred)
    "FlextTypes",
    "FlextValidatable",
    "FlextValidator",
    "P",
    "R",
    "Serializable",
    "T",
    # Legacy T* aliases - Domain types
    "TAggregateId",
    # Legacy T* aliases - Core types
    "TAnyDict",
    "TAnyList",
    # TypeVar entities
    "TAnyObject",
    "TCallable",
    "TCommand",
    # Legacy T* aliases - CQRS types
    "TCommandBusId",
    "TCommandId",
    "TCommandMetadata",
    "TCommandPayload",
    "TCommandPriority",
    "TCommandResult",
    "TCommandType",
    "TComparable",
    "TConfig",
    # Legacy T* aliases - Config types
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
    # Legacy T* aliases - Infrastructure types
    "TConnection",
    "TConnectionString",
    # Legacy T* aliases - Auth types
    "TContextDict",
    # Legacy T* aliases - Service types
    "TCorrelationId",
    "TCredentials",
    # Legacy T* aliases - Validation types
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
    # Legacy T* aliases - Logging types
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
    "TQuery",
    "TQueryCriteria",
    "TQueryId",
    "TQueryProjection",
    "TQueryResult",
    "TQueryType",
    "TRequest",
    "TRequestId",
    "TResponse",
    "TResult",
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
    # Utility functions
    "get_centralized_types_usage_info",
]

# Total exports: 100+ items - comprehensive coverage of all ecosystem types
# This module now serves as THE SINGLE SOURCE OF TRUTH for all types

# =============================================================================
# MODULE VALIDATION - Ensure centralization compliance
# =============================================================================

# Validation: This module should contain ALL type definitions used across the ecosystem
# All other modules should import from here to maintain centralization
# This satisfies the CENTRALIZED pattern: all types in one authoritative location
