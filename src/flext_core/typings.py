"""Layer 0: Centralized type system for the entire FLEXT ecosystem.

This module provides the complete type system that ALL other flext_core modules depend on.
As Layer 0 (pure Python), it has ZERO dependencies on other flext_core modules,
making it safe to import from anywhere without circular dependency risks.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import (
    ParamSpec,
    TypedDict,
    TypeVar,
)

from flext_core.constants import FlextConstants

# =============================================================================
# FLEXT TYPES NAMESPACE - Centralized type system for the FLEXT ecosystem
# =============================================================================

# Core TypeVars for generic programming
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

# Additional TypeVars for ecosystem compatibility
E = TypeVar("E")
F = TypeVar("F")
K = TypeVar("K")
P = ParamSpec("P")
R = TypeVar("R")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")

# Covariant TypeVars
T1_co = TypeVar("T1_co", covariant=True)
T2_co = TypeVar("T2_co", covariant=True)
T3_co = TypeVar("T3_co", covariant=True)
TAggregate_co = TypeVar("TAggregate_co", covariant=True)
TCacheValue_co = TypeVar("TCacheValue_co", covariant=True)
TDomainEvent_co = TypeVar("TDomainEvent_co", covariant=True)
TEntity_co = TypeVar("TEntity_co", covariant=True)
TResult_co = TypeVar("TResult_co", covariant=True)
TState_co = TypeVar("TState_co", covariant=True)
TValue_co = TypeVar("TValue_co", covariant=True)
TValueObject_co = TypeVar("TValueObject_co", covariant=True)

# Contravariant TypeVars
TCacheKey_contra = TypeVar("TCacheKey_contra", contravariant=True)
TCommand_contra = TypeVar("TCommand_contra", contravariant=True)
TConfigKey_contra = TypeVar("TConfigKey_contra", contravariant=True)
TEvent_contra = TypeVar("TEvent_contra", contravariant=True)
TInput_contra = TypeVar("TInput_contra", contravariant=True)
TItem_contra = TypeVar("TItem_contra", contravariant=True)
TQuery_contra = TypeVar("TQuery_contra", contravariant=True)
TResult_contra = TypeVar("TResult_contra", contravariant=True)
TUtil_contra = TypeVar("TUtil_contra", contravariant=True)

# Domain-specific TypeVars
Command = TypeVar("Command")
Event = TypeVar("Event")
Message = TypeVar("Message")
Query = TypeVar("Query")
ResultT = TypeVar("ResultT")


class FlextTypes:
    """Centralized type system namespace for the FLEXT ecosystem.

    Provides the single source of truth for all type definitions
    across the entire FLEXT ecosystem with proper variance annotations
    and modern Python 3.13+ syntax.

    Includes:
    - Core: Core fundamental types with Python 3.13+ enhancements
    - Async: Modern async and concurrent patterns
    - ErrorHandling: Enhanced error handling with recovery strategies
    - Service: Comprehensive service layer types
    - Headers: HTTP headers and metadata types
    - Message: Message types for CQRS patterns
    - Handler: Handler types for command/query processing
    - Protocol: Protocol types for interfaces
    - Plugin: Plugin types for extensibility

    All types use modern Python 3.13+ union syntax (X | Y) and provide
    80+ TypeVars with proper variance annotations for complete type safety.
    """

    # Basic collection types - Direct access for backward compatibility
    type Dict = dict[str, object]
    type List = list[object]
    type StringList = list[str]

    type IntList = list[int]
    type FloatList = list[float]
    type BoolList = list[bool]

    # Component types for normalization operations - backward compatibility
    # Must be hashable for set operations, so we exclude generic object
    type ComponentType = (
        Dict | List | set[object] | tuple[object, ...] | str | int | float | bool | None
    )

    # Sortable types for sorting operations - backward compatibility
    type SortableType = str | int | float

    # Serializable types for serialization operations - backward compatibility
    type SerializableType = (
        FlextTypes.Dict | FlextTypes.List | str | int | float | bool | None
    )

    # Type variable references moved to module level below

    # Collection types for backward compatibility
    type OrderedDict = (
        FlextTypes.Dict
    )  # Use dict[str, object] for type annotation (modern Python 3.7+)

    # Core types reference for backward compatibility

    # Advanced collection types
    type NestedDict = dict[str, FlextTypes.Dict]
    type StringDict = dict[str, str]
    type IntDict = dict[str, int]
    type FloatDict = dict[str, float]
    type BoolDict = dict[str, bool]

    # Configuration types
    type ConfigValue = (
        str | int | float | bool | FlextTypes.List | FlextTypes.Dict | None
    )
    type ConfigDict = dict[str, FlextTypes.ConfigValue]

    # JSON types with modern Python 3.13+ syntax
    type JsonValue = FlextTypes.ConfigValue
    type JsonArray = list[FlextTypes.JsonValue]
    type JsonDict = dict[str, FlextTypes.JsonValue]

    # Value types
    type Value = str | int | float | bool | object | None
    type Success = str  # Generic success type without dependencies

    # Collection types with ordering (duplicate removed - use dict[str, object] for type hints)
    # Note: Use FlextTypes.Dict for OrderedDict type annotations in modern Python

    # Assign for backward compatibility - moved outside class
    # OrderedDict = OrderedFlextTypes.Dict  # Will be assigned after class definition

    # =========================================================================
    # CORE TYPES - Fundamental types for ecosystem
    # =========================================================================

    class Core:
        """Core fundamental types for the FLEXT ecosystem with Python 3.13+ enhancements.

        This namespace provides the most basic, frequently used types
        that form the foundation of all other type definitions, enhanced
        with modern Python 3.13+ patterns and better type safety.

        Examples:
            Basic data structures with enhanced typing:

            >>> data: FlextTypes.Core.Dict = {"key": "value"}
            >>> items: FlextTypes.Core.List = [1, 2, 3]
            >>> mapping: FlextTypes.Core.Mapping = {"a": 1, "b": 2}
            >>> items_list: FlextTypes.Core.List = [1, 2, 3]

        Enhanced patterns:
            >>> # Generic sync operations
            >>> def process_data[T](data: T) -> T:
            ...     return data
            >>>
            >>> # Enhanced error handling
            >>> def safe_operation[T](
            ...     data: T,
            ... ) -> T:
            ...     return data

        """

        type Dict = FlextTypes.Dict
        type List = FlextTypes.List
        type Mapping = FlextTypes.Dict
        type Sequence = FlextTypes.List
        type Set = set[object]
        type Tuple = tuple[object, ...]

        # Component types for normalization operations
        # Must be hashable for set operations, so we exclude generic object
        type ComponentType = Dict | List | Set | Tuple | str | int | float | bool | None

        # Sortable types for sorting operations
        type SortableType = str | int | float

        # Serializable types for serialization operations
        type SerializableType = Dict | List | str | int | float | bool | None

        # Enhanced async and concurrent types
        type AsyncDict = dict[str, Awaitable[object]]
        type AsyncList = list[Awaitable[object]]
        type ConcurrentDict[TConcurrent] = dict[str, TConcurrent]
        type ConcurrentList[TConcurrent] = list[TConcurrent]

        # Enhanced optional and union types
        type OptionalDict = Dict | None
        type OptionalList = List | None
        type UnionDict = Dict | Mapping

        # Enhanced typed collections
        type TypedDict[K, V] = dict[K, V]
        type TypedList[T] = list[T]
        type TypedSet[T] = set[T]

    # Headers types - Direct access for backward compatibility
    HeadersDict = dict[str, str]
    HeadersMetadata = dict[str, str | int | float | bool]
    HeadersRequest = dict[str, str]
    HeadersResponse = dict[str, str]

    class Headers:
        """HTTP headers and metadata types.

        This namespace provides types for HTTP headers, metadata,
        and protocol-level information exchange.

        Examples:
            HTTP headers for API requests:

            >>> headers: FlextTypes.Headers.Dict = {
            ...     "Content-Type": "application/json",
            ...     "Authorization": "Bearer token123",
            ... }
            >>> metadata: FlextTypes.Headers.Metadata = {
            ...     "request_id": "req-123",
            ...     "timestamp": "2025-01-01T00:00:00Z",
            ... }

        """

        type Dict = dict[str, str]
        type Metadata = dict[str, str | int | float | bool]
        type RequestHeaders = dict[str, str]
        type ResponseHeaders = dict[str, str]

    # =========================================================================
    # DOMAIN TYPES - Domain-Driven Design patterns (ENHANCED for event sourcing)
    # =========================================================================

    class Domain:
        """Domain types for DDD and event sourcing patterns.

        This namespace provides types for entities, value objects, aggregates,
        and complex event sourcing scenarios.

        Examples:
            Event sourcing with typed events:

            >>> events: FlextTypes.Domain.EventStream = [
            ...     {
            ...         "type": "UserCreated",
            ...         "payload": {"id": "123"},
            ...         "timestamp": 1234567890,
            ...     },
            ...     {
            ...         "type": "UserUpdated",
            ...         "payload": {"name": "John"},
            ...         "timestamp": 1234567891,
            ...     },
            ... ]

            Event handler registry:

            >>> handlers: FlextTypes.Domain.EventHandlerRegistry = {
            ...     "UserCreated": [handle_user_created, notify_user_created],
            ...     "UserUpdated": [handle_user_updated],
            ... }

        """

        # Event sourcing types
        type EventType = str
        type EventPayload = FlextTypes.Dict
        type EventMetadata = dict[str, str | int | float]
        type EventTyped = dict[
            str,
            FlextTypes.Domain.EventType
            | FlextTypes.Domain.EventPayload
            | FlextTypes.Domain.EventMetadata,
        ]
        type EventStream = list[FlextTypes.Domain.EventTyped]

        # Event handler types
        type EventHandler = Callable[[FlextTypes.Domain.EventTyped], None]
        type EventHandlerList = list[FlextTypes.Domain.EventHandler]
        type EventHandlerRegistry = dict[
            FlextTypes.Domain.EventType, FlextTypes.Domain.EventHandlerList
        ]

        type AggregateState = dict[str, object | FlextTypes.List]
        type AggregateVersion = int

    # =========================================================================
    # ASYNC AND CONCURRENT TYPES - Modern async patterns
    # =========================================================================

    class Async:
        """Enhanced async and concurrent types for Python 3.13+ patterns.

        This namespace provides advanced async types for modern concurrent
        programming patterns including async iterators, async generators,
        and concurrent data structures.

        Examples:
            Sync operations with proper typing:

            >>> def process_data(data: dict) -> dict[str, object]:
            ...     results = []
            ...     for key, value in data.items():
            ...         result = process_item(key, value)
            ...         results.append(result)
            ...     return {"processed": len(results)}

            Concurrent processing:

            >>> def process_concurrent[T](
            ...     items: list[T],
            ... ) -> list[T]:
            ...     return [process(item) for item in items]

        """

        type AsyncDict = dict[str, Awaitable[object]]
        type AsyncList = list[Awaitable[object]]
        type AsyncResult[T] = Awaitable[T]

        type ConcurrentDict[TConcurrent] = dict[str, TConcurrent]
        type ConcurrentList[TConcurrent] = list[TConcurrent]
        type ConcurrentResult[T] = Awaitable[list[T]]

        type TaskDict = dict[str, Awaitable[object]]
        type TaskList = list[Awaitable[object]]
        type TaskResult[T] = Awaitable[T]

    # =========================================================================
    # ERROR HANDLING TYPES - Enhanced error patterns
    # =========================================================================

    class ErrorHandling:
        """Enhanced error handling types for robust error management.

        This namespace provides types for comprehensive error handling
        including error classification, error recovery, and error reporting
        patterns integrated with result types.

        Examples:
            Error classification and handling:

            >>> def classify_error(
            ...     error: ErrorHandling.ErrorType,
            ... ) -> ErrorHandling.ErrorCategory:
            ...     if isinstance(error, ValidationError):
            ...         return "validation"
            ...     elif isinstance(error, NetworkError):
            ...         return "network"
            ...     return "unknown"

            Error recovery patterns:

            >>> def recover_from_error[T](
            ...     operation: Callable[[], T],
            ...     recovery: Callable[[Exception], T],
            ... ) -> T:
            ...     try:
            ...         return operation()
            ...     except Exception as e:
            ...         return recovery(e)

        """

        type ErrorType = Exception | BaseException
        type ErrorCategory = FlextConstants.ErrorCategory
        type ErrorSeverity = FlextConstants.ErrorSeverity
        type ErrorReport = dict[str, Exception | str]
        type ErrorChain = list[Exception]

        type RecoveryStrategy[T] = Callable[[Exception | BaseException], Awaitable[T]]
        type RecoveryResult[T] = T | Exception | BaseException
        type ErrorHandler[T] = Callable[
            [Exception | BaseException], T | Exception | BaseException
        ]

    # =========================================================================
    # SERVICE TYPES - Service layer patterns (Enhanced)
    # =========================================================================

    class Service:
        """Enhanced service layer types with flext-core integration.

        This namespace provides comprehensive service types that integrate
        with FlextContainer, and other flext-core components
        for robust service-oriented architecture.

        Examples:
            Service registration and discovery:

            >>> services: Service.ServiceRegistry = {
            ...     "user_service": UserService(),
            ...     "auth_service": AuthService(),
            ...     "notification_service": NotificationService(),
            ... }

            Service factories with dependency injection:

            >>> factories: Service.ServiceFactory = {
            ...     "database": lambda config: DatabaseService(config.database_url),
            ...     "cache": lambda config: CacheService(config.cache_ttl),
            ... }

        """

        type Dict = FlextTypes.Dict
        type Type = FlextConstants.ServiceType
        type FactoryDict = dict[str, Callable[[], object]]
        type ServiceRegistry = FlextTypes.Dict
        type ServiceFactory = dict[str, Callable[[FlextTypes.Dict], object]]

        # Enhanced service lifecycle types
        type LifecycleState = FlextConstants.ServiceLifecycleState
        type ServiceConfig = dict[str, object | FlextTypes.Dict]

        # Enhanced service communication types
        type ServiceEndpoint = str
        type ServiceProtocol = FlextConstants.ServiceProtocol
        type ServiceContract = dict[str, ServiceEndpoint | ServiceProtocol]

        # Enhanced service monitoring types
        type ServiceMetrics = dict[str, int | float | str]
        type ServiceHealth = dict[str, bool | str | int]
        type ServiceStatus = dict[str, LifecycleState | ServiceMetrics | ServiceHealth]

    # =========================================================================
    # CONTEXT TYPES - Context management patterns
    # =========================================================================

    class Context:
        """Context management types for FlextContext operations.

        This namespace provides types for context scopes, correlation IDs,
        context dictionaries, and context export operations aligned with
        FlextConstants.Context configuration.

        Examples:
            Context scope definitions:

            >>> scope: FlextTypes.Context.Scope = "request"
            >>> correlation_id: FlextTypes.Context.CorrelationId = "flext-abc123def456"

            Context data structures:

            >>> context_data: FlextTypes.Context.Dict = {
            ...     "user_id": "user-123",
            ...     "request_id": "req-456",
            ...     "timestamp": 1234567890,
            ... }

            Context export formats:

            >>> export_format: FlextTypes.Context.ExportFormat = "json"

        """

        # Scope types (aligned with FlextConstants.Context scope literals)
        type Scope = FlextConstants.ContextScope

        # Correlation ID type (aligned with FlextConstants.Context correlation config)
        type CorrelationId = str

        # Context value types (support multiple primitive types)
        type ContextValue = str | int | float | bool | None
        type ContextDict = dict[str, FlextTypes.Context.ContextValue]

        # Context metadata types
        type ContextMetadata = dict[str, str | int | float]
        type ContextTimestamp = int  # Milliseconds since epoch

        # Context export types (aligned with FlextConstants.Context export formats)
        type ExportFormat = FlextConstants.ContextExportFormat
        type ExportedContext = FlextTypes.Dict | str

        # Context depth and size types (aligned with FlextConstants.Context limits)
        type ContextDepth = int
        type ContextSize = int

        # Scope management types (for nested context scopes)
        type ScopeRegistry = dict[str, FlextTypes.Dict]

        # Context hooks types (for context lifecycle hooks)
        type HookFunc = Callable[..., object]
        type HookList = list[FlextTypes.Context.HookFunc]
        type HookRegistry = dict[str, FlextTypes.Context.HookList]

    # =========================================================================
    # UTILITIES TYPES - Utility function patterns
    # =========================================================================

    class Utilities:
        """Utility function types for FlextUtilities operations.

        This namespace provides types for validation, caching, secret generation,
        string/number processing, and collection operations aligned with
        FlextConstants.Utilities configuration.

        Examples:
            Validation types:

            >>> validation_result: FlextTypes.Utilities.ValidationResult = True
            >>> validation_error: FlextTypes.Utilities.ValidationError = "Invalid input"

            Cache types:

            >>> cache_key: FlextTypes.Utilities.CacheKey = "user:123"
            >>> cache_value: FlextTypes.Utilities.CacheValue = {"name": "John"}
            >>> cache_ttl: FlextTypes.Utilities.CacheTTL = 300

            Secret generation:

            >>> secret: FlextTypes.Utilities.SecretString = "abc123def456..."
            >>> secret_length: FlextTypes.Utilities.SecretLength = 32

            String/number processing:

            >>> encoding: FlextTypes.Utilities.Encoding = "utf-8"
            >>> decimal_places: FlextTypes.Utilities.DecimalPlaces = 2

            Collection operations:

            >>> batch_size: FlextTypes.Utilities.BatchSize = 100
            >>> collection_size: FlextTypes.Utilities.CollectionSize = 1000

        """

        # Validation types
        type ValidationResult = bool
        type ValidationError = str | None
        type ValidationReport = dict[str, FlextTypes.Utilities.ValidationError]

        # Cache types (aligned with FlextConstants.Utilities cache config)
        type CacheKey = str
        type CacheValue = object
        type CacheTTL = int  # Time-to-live in seconds
        type CacheSize = int

        # Secret generation types (aligned with FlextConstants.Utilities secret config)
        type SecretString = str
        type SecretLength = int
        type SecretBytes = bytes

        # String processing types (aligned with FlextConstants.Utilities string config)
        type Encoding = str
        type StringLength = int
        type ProcessedString = str

        # Number processing types (aligned with FlextConstants.Utilities decimal config)
        type DecimalPlaces = int
        type ProcessedNumber = int | float

        # Collection types (aligned with FlextConstants.Utilities collection limits)
        type BatchSize = int
        type CollectionSize = int
        type CollectionIndex = int

        # Timeout types (aligned with FlextConstants.Utilities timeout config)
        type TimeoutSeconds = int
        type OperationTimeout = int

    # =========================================================================
    # CONFIG TYPES - Configuration and settings
    # =========================================================================

    class Config:
        """Configuration types."""

        type LogLevel = FlextConstants.LoggingLevel
        type Serializer = Callable[
            [
                dict[
                    str,
                    str | int | float | bool | FlextTypes.List | FlextTypes.Dict | None,
                ],
            ],
            str,
        ]

        # Core-specific type aliases
        type ConfigData = dict[str, FlextTypes.ConfigValue]
        type ModuleConfig = dict[str, FlextTypes.Config.ConfigData]
        type ServiceRegistry = FlextTypes.Dict
        type ComponentMap = dict[str, type | object]

    # =========================================================================
    # VALIDATION TYPES - Validation patterns (ENHANCED for complex scenarios)
    # =========================================================================

    class Validation:
        """Validation types for complex validation scenarios.

        This namespace provides types for field-level validators, business rules,
        invariant checking, and consistency rules across domain models.

        Examples:
            Field validators with result types:

            >>> field_validator: FlextTypes.Validation.FieldValidator[str] = (
            ...     lambda value: None  # Simplified for typing example
            ...     if "@" in value
            ...     else ValueError("Invalid email")
            ... )

        """

        # Basic validators
        type Rule = Callable[[object], bool]
        type Validator = Rule
        type BusinessRule = Rule

        # Complex validation patterns (NEW - high value)
        type FieldValidator[T] = Callable[
            [T], object
        ]  # Simplified to break circular import
        type FieldValidators[T] = list[FlextTypes.Validation.FieldValidator[T]]
        type FieldValidatorRegistry[T] = dict[
            str, FlextTypes.Validation.FieldValidators[T]
        ]

        type EntityValidator[T] = FlextTypes.Validation.FieldValidator[T]
        type BusinessRuleFunc[T] = FlextTypes.Validation.EntityValidator[T]
        type BusinessRuleRegistry[T] = dict[
            str, FlextTypes.Validation.BusinessRuleFunc[T]
        ]

        type Invariant[T] = FlextTypes.Validation.FieldValidator[T]
        type InvariantList[T] = list[FlextTypes.Validation.Invariant[T]]

        type ConsistencyRule[T, U] = Callable[
            [T, U], object
        ]  # Simplified to break circular import
        type ConsistencyRuleRegistry[T] = dict[
            str, dict[str, FlextTypes.Validation.ConsistencyRule[T, object]]
        ]

    # =========================================================================
    # OUTPUT TYPES - Generic output formatting types
    # =========================================================================

    class Output:
        """Generic output formatting types."""

        type OutputFormat = FlextConstants.ProcessingOutputFormat
        type SerializationFormat = FlextConstants.ProcessingSerializationFormat
        type CompressionFormat = FlextConstants.ProcessingCompressionFormat

    # =========================================================================
    # SERVICE ORCHESTRATION TYPES - Service orchestration patterns
    # =========================================================================

    class ServiceOrchestration:
        """Service orchestration types."""

        type ServiceOrchestrator = FlextTypes.Dict
        type AdvancedServiceOrchestrator = FlextTypes.Dict

    # =========================================================================
    # PROJECT TYPES - Project management types
    # =========================================================================

    class Project:
        """Project management types."""

        type ProjectType = FlextConstants.ProjectType
        type ProjectStatus = FlextConstants.ProjectStatus
        type ProjectConfig = FlextTypes.Dict

    # =========================================================================
    # PROCESSING TYPES - Generic processing patterns
    # =========================================================================

    class Processing:
        """Generic processing types for ecosystem patterns."""

        # Processing status types
        type ProcessingStatus = FlextConstants.WorkflowProcessingStatus
        type ProcessingMode = FlextConstants.WorkflowProcessingMode
        type ValidationLevel = FlextConstants.WorkflowValidationLevel
        type ProcessingPhase = FlextConstants.WorkflowProcessingPhase
        type HandlerType = FlextConstants.WorkflowHandlerType
        type WorkflowStatus = FlextConstants.WorkflowStatus

        WorkspaceStatus = FlextConstants.WorkspaceStatus

        type StepStatus = FlextConstants.WorkflowStepStatus

    # =========================================================================
    # HANDLERS TYPES - CQRS handler registries and pipelines (NEW)
    # =========================================================================

    class Handlers:
        """Complex CQRS handler types for command/query/event processing.

        This namespace provides types for handler registries, middleware pipelines,
        message routing, and handler configuration in CQRS patterns.

        Examples:
            Handler registry with callbacks:

            >>> handlers: FlextTypes.Handlers.HandlerRegistry = {
            ...     "CreateUser": [create_user_handler, log_user_created],
            ...     "UpdateUser": [update_user_handler],
            ... }

            Middleware pipeline:

            >>> pipeline: FlextTypes.Handlers.MiddlewarePipeline = [
            ...     validate_middleware,
            ...     auth_middleware,
            ...     logging_middleware,
            ... ]

        """

        # Handler identification
        type Mode = FlextConstants.CqrsMode

        # Handler functions
        type HandlerFunc = Callable[
            [object], object
        ]  # Simplified to break circular import
        type HandlerList = list[FlextTypes.Handlers.HandlerFunc]
        type HandlerRegistry = dict[str, FlextTypes.Handlers.HandlerList]

        # Middleware types
        type MiddlewareFunc = Callable[[FlextTypes.Dict], FlextTypes.Dict]
        type MiddlewarePipeline = list[FlextTypes.Handlers.MiddlewareFunc]
        type MiddlewareConfig = dict[str, object | int | str]

        # Handler configuration
        type HandlerConfig = dict[str, object | dict[str, int | float | bool]]
        type MessageRouter = dict[
            str, FlextTypes.Handlers.HandlerFunc | FlextTypes.Handlers.HandlerList
        ]

        # Saga types (complex multi-step handlers)
        type SagaStep[T] = Callable[[T], object]  # Simplified to break circular import
        type SagaSteps[T] = list[FlextTypes.Handlers.SagaStep[T]]
        type CompensationStep[T] = FlextTypes.Handlers.SagaStep[T]
        type CompensationSteps[T] = list[FlextTypes.Handlers.CompensationStep[T]]

    # =========================================================================
    # RELIABILITY TYPES - Circuit breaker, retry, and rate limiting (NEW)
    # =========================================================================

    class Reliability:
        """Reliability pattern types for circuit breakers, retries, and rate limiting.

        This namespace provides types for complex stateful reliability mechanisms
        including circuit breakers, retry policies, and rate limiters.

        Examples:
            Circuit breaker state registry:

            >>> breakers: FlextTypes.Reliability.CircuitBreakerRegistry = {
            ...     "payment_service": {
            ...         "state": "closed",
            ...         "failure_count": 0,
            ...         "last_failure": 0.0,
            ...     }
            ... }

            Retry policy with strategy:

            >>> policy: FlextTypes.Reliability.RetryPolicy = {
            ...     "max_attempts": 3,
            ...     "base_delay": 1.0,
            ...     "strategy": exponential_backoff,
            ... }

        """

        # Circuit breaker types
        type CircuitState = FlextConstants.CircuitBreakerState
        type CircuitStats = dict[
            str, bool | int | float | str | FlextTypes.FloatList | None
        ]
        type CircuitBreakerRegistry = dict[str, FlextTypes.Reliability.CircuitStats]

        # Retry types
        type RetryStrategy = Callable[[int], float]  # attempt -> delay
        type RetryPolicy = dict[str, int | float | FlextTypes.Reliability.RetryStrategy]
        type RetryPolicyRegistry = dict[str, FlextTypes.Reliability.RetryPolicy]

        class RateLimiterState(TypedDict):
            """Rate limiter state tracking structure.

            Used by FlextExceptions for tracking rate limiting state across
            exception handling operations.
            """

            requests: FlextTypes.FloatList
            last_reset: float

        type RateLimiterRegistry = dict[str, FlextTypes.Reliability.RateLimiterState]

        class DispatcherRateLimiterState(TypedDict):
            """Rate limiter state for FlextDispatcher sliding window implementation.

            Used by FlextDispatcher for tracking rate limiting state with
            count-based sliding window algorithm.
            """

            count: int
            window_start: float
            block_until: float

        # Performance metrics
        type PerformanceMetrics = dict[str, dict[str, int | float]]


__all__: FlextTypes.StringList = [
    "Command",
    "E",
    "Event",
    "F",
    "FlextTypes",
    "K",
    "Message",
    "P",  # ParamSpec for generic callable types
    "Query",
    "R",
    "ResultT",
    "T",
    "T1_co",
    "T2_co",
    "T3_co",
    "TAggregate_co",
    "TCacheKey_contra",
    "TCacheValue_co",
    "TCommand_contra",
    "TConfigKey_contra",
    "TDomainEvent_co",
    "TEntity_co",
    "TEvent_contra",
    "TInput_contra",
    "TItem_contra",
    "TQuery_contra",
    "TResult_co",
    "TResult_contra",
    "TState_co",
    "TUtil_contra",
    "TValueObject_co",
    "TValue_co",
    "T_co",
    "T_contra",
    "U",
    "V",
    "W",
]
