"""Centralized type system for FLEXT ecosystem with Python 3.13+ patterns.

This module provides the complete type system for the FLEXT ecosystem,
centralizing all TypeVars and type aliases in a single location following
strict Flext standards and Clean Architecture principles.

Key Principles:
- Single source of truth for all types
- Python 3.13+ syntax with modern union types
- Pydantic 2 patterns and validation
- No wrappers, aliases, or legacy patterns
- Centralized TypeVars exported as public API

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from typing import (
    Literal,
    ParamSpec,
    TypedDict,
    TypeVar,
)

# =============================================================================
# FLEXT TYPES NAMESPACE - Centralized type system for the FLEXT ecosystem
# =============================================================================


class FlextTypes:
    """Centralized type system namespace for the FLEXT ecosystem.

    FLEXT-CORE OPTIMIZATION PATTERNS DEMONSTRATED:

    🚀 NAMESPACE CLASS PATTERN
    Single unified class with nested specialized namespaces AND centralized TypeVars:
    All 80+ TypeVars defined as class attributes for strict module organization
    Nested namespaces for specialized type groups (Core, Async, ErrorHandling, etc.)
    No loose TypeVar definitions outside the class

    **Function**: Type definitions for ecosystem-wide consistency
        - Core fundamental types (Dict, List, Headers) with Python 3.13+ patterns
        - Configuration types (ConfigValue, ConfigDict) with modern syntax
        - JSON types with Python 3.13+ union syntax (X | Y)
        - Message types for CQRS patterns with proper variance
        - Handler types for command/query handlers with contravariant inputs
        - Service types for domain services with covariant outputs
        - Protocol types for interface definitions
        - Plugin types for extensibility
        - Generic type variables with 40+ specialized variants
        - Covariant and contravariant type variables for type safety

    **Uses**: Modern Python type system infrastructure
        - typing.TypeVar for generic type variables with variance
        - typing.ParamSpec for parameter specifications
        - typing.Literal for literal type hints
        - collections.abc.Callable and Awaitable for callable types
        - Python 3.13+ union syntax (X | Y) for modern type expressions
        - Modern type aliases with type keyword
        - Covariant and contravariant variance for type safety
        - Generic type constraints for proper type relationships

    **Attributes**:
        Core: Core fundamental types for ecosystem with Python 3.13+ enhancements.
        Async: Modern async and concurrent patterns for Python 3.13+.
        ErrorHandling: Enhanced error handling with recovery strategies.
        Service: Comprehensive service layer types with flext-core integration.
        Headers: HTTP headers and metadata types.
        Message: Message types for CQRS patterns.
        Handler: Handler types for command/query processing.
        Protocol: Protocol types for interfaces.
        Plugin: Plugin types for extensibility.

    **Note**:
        All types use Python 3.13+ modern syntax with union operators (X | Y).
        Type variables support covariance/contravariance for type safety.
        Type aliases use modern type keyword syntax.
        No wrappers or legacy patterns allowed.
        Single source of truth for all ecosystem types.
        40+ TypeVars exported for complete type safety coverage.

    **Warning**:
        Type changes may impact entire ecosystem.
        Generic type variables must match usage patterns correctly.
        Covariance/contravariance must be used appropriately for type safety.
        Type aliases should not wrap existing types unnecessarily.

    **See Also**:
        FlextResult: For result type patterns and railway composition.
        FlextModels: For domain model types and DDD patterns.
        FlextHandlers: For handler type usage and CQRS patterns.
        FlextProtocols: For protocol definitions and interface contracts.
        FlextUtilities: For type validation and transformation utilities.
    """

    # Type Variables - All centralized in this class
    P = ParamSpec("P")
    T1 = TypeVar("T1")
    T2 = TypeVar("T2")
    T3 = TypeVar("T3")
    T1_co = TypeVar("T1_co", covariant=True)
    T2_co = TypeVar("T2_co", covariant=True)
    T3_co = TypeVar("T3_co", covariant=True)
    TItem = TypeVar("TItem")
    TItem_contra = TypeVar("TItem_contra", contravariant=True)
    TResult = TypeVar("TResult")
    TResult_contra = TypeVar("TResult_contra", contravariant=True)
    TUtil = TypeVar("TUtil")
    TUtil_contra = TypeVar("TUtil_contra", contravariant=True)
    T_contra = TypeVar("T_contra", contravariant=True)
    MessageT = TypeVar("MessageT")
    MessageT_contra = TypeVar("MessageT_contra", contravariant=True)
    TCommand_contra = TypeVar("TCommand_contra", contravariant=True)
    TEvent_contra = TypeVar("TEvent_contra", contravariant=True)
    TInput_contra = TypeVar("TInput_contra", contravariant=True)
    TQuery_contra = TypeVar("TQuery_contra", contravariant=True)
    TState_co = TypeVar("TState_co", covariant=True)
    TState = TypeVar("TState")
    E = TypeVar("E")
    F = TypeVar("F")
    K = TypeVar("K")
    R = TypeVar("R")
    TAsync = TypeVar("TAsync")
    TAsync_co = TypeVar("TAsync_co", covariant=True)
    TAwaitable = TypeVar("TAwaitable")
    TConcurrent = TypeVar("TConcurrent")
    TParallel = TypeVar("TParallel")
    TSync = TypeVar("TSync")
    TData = TypeVar("TData")
    TData_co = TypeVar("TData_co", covariant=True)
    TProcessor = TypeVar("TProcessor")
    TPipeline = TypeVar("TPipeline")
    TTransform = TypeVar("TTransform")
    TTransform_co = TypeVar("TTransform_co", covariant=True)
    TError = TypeVar("TError")
    TError_co = TypeVar("TError_co", covariant=True)
    TException = TypeVar("TException")
    TException_co = TypeVar("TException_co", covariant=True)
    TConfig = TypeVar("TConfig")
    TConfig_co = TypeVar("TConfig_co", covariant=True)
    TSettings = TypeVar("TSettings")
    TSettings_co = TypeVar("TSettings_co", covariant=True)
    TService = TypeVar("TService")
    TService_co = TypeVar("TService_co", covariant=True)
    TClient = TypeVar("TClient")
    TClient_co = TypeVar("TClient_co", covariant=True)
    Message = TypeVar("Message")
    Command = TypeVar("Command")
    Query = TypeVar("Query")
    Event = TypeVar("Event")
    ResultT = TypeVar("ResultT")
    TCommand = TypeVar("TCommand")
    TEvent = TypeVar("TEvent")
    TQuery = TypeVar("TQuery")
    U = TypeVar("U")
    V = TypeVar("V")
    T = TypeVar("T")
    T_co = TypeVar("T_co", covariant=True)
    TAccumulate = TypeVar("TAccumulate")
    TAggregate = TypeVar("TAggregate")
    TAggregate_co = TypeVar("TAggregate_co", covariant=True)
    TCacheKey = TypeVar("TCacheKey")
    TCacheKey_contra = TypeVar("TCacheKey_contra", contravariant=True)
    TCacheValue = TypeVar("TCacheValue")
    TCacheValue_co = TypeVar("TCacheValue_co", covariant=True)
    W = TypeVar("W")
    TConfigKey = TypeVar("TConfigKey")
    TConfigKey_contra = TypeVar("TConfigKey_contra", contravariant=True)
    TConfigValue = TypeVar("TConfigValue")
    TConfigValue_co = TypeVar("TConfigValue_co", covariant=True)
    TDomainEvent = TypeVar("TDomainEvent")
    TDomainEvent_co = TypeVar("TDomainEvent_co", covariant=True)
    TEntity = TypeVar("TEntity")
    TEntity_co = TypeVar("TEntity_co", covariant=True)
    TKey = TypeVar("TKey")
    TKey_contra = TypeVar("TKey_contra", contravariant=True)
    TMessage = TypeVar("TMessage")
    TResource = TypeVar("TResource")
    TResult_co = TypeVar("TResult_co", covariant=True)
    TTimeout = TypeVar("TTimeout")
    TValue = TypeVar("TValue")
    TValue_co = TypeVar("TValue_co", covariant=True)
    TValueObject_co = TypeVar("TValueObject_co", covariant=True)
    UParallel = TypeVar("UParallel")
    UResource = TypeVar("UResource")
    TPlugin = TypeVar("TPlugin")
    TPluginConfig = TypeVar("TPluginConfig")

    # Basic collection types - Direct access for backward compatibility
    Dict = dict[str, object]
    List = list[object]
    StringList = list[str]

    IntList = list[int]
    FloatList = list[float]
    BoolList = list[bool]

    # Type variable references moved to module level below

    # Collection types for backward compatibility
    OrderedDict = dict[str, object]  # Use dict for type annotation (modern Python 3.7+)

    # Core types reference for backward compatibility

    # Advanced collection types
    type NestedDict = dict[str, FlextTypes.Dict]
    type StringDict = dict[str, str]
    type IntDict = dict[str, int]
    type FloatDict = dict[str, float]

    # Basic collection types (legacy names for compatibility)
    type DictType = FlextTypes.Dict
    type ListType = list[object]
    type StringListType = list[str]
    type IntListType = list[int]
    type FloatListType = list[float]
    type BoolListType = list[bool]

    # Advanced collection types (legacy names for compatibility)
    type NestedDictType = dict[str, DictType]
    type StringDictType = dict[str, str]
    type IntDictType = dict[str, int]
    type FloatDictType = dict[str, float]
    type BoolDict = dict[str, bool]
    type OrderedDictType = FlextTypes.Dict  # Use dict for type annotation

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

    # Collection types with ordering (duplicate removed - use dict for type hints)
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
        type List = list[object]
        type Mapping = FlextTypes.Dict
        type Sequence = list[object]
        type Set = set[object]
        type Tuple = tuple[object, ...]

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

            >>> def process_data(data: dict) -> dict:
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
        type AsyncIteratorAlias[T] = AsyncIterator[T]
        type AsyncGeneratorAlias[T] = AsyncGenerator[T]

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
        type ErrorCategory = Literal[
            "validation", "network", "database", "auth", "system", "unknown"
        ]
        type ErrorSeverity = Literal["low", "medium", "high", "critical"]
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
        with FlextService, FlextContainer, and other flext-core components
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
        type Type = Literal["instance", "factory", "singleton"]
        type FactoryDict = dict[str, Callable[[], object]]
        type ServiceRegistry = FlextTypes.Dict
        type ServiceFactory = dict[str, Callable[[FlextTypes.Dict], object]]

        # Enhanced service lifecycle types
        type LifecycleState = Literal[
            "initializing", "ready", "running", "stopping", "stopped", "error"
        ]
        type ServiceConfig = dict[str, object | FlextTypes.Dict]

        # Enhanced service communication types
        type ServiceEndpoint = str
        type ServiceProtocol = Literal["http", "grpc", "websocket", "message_queue"]
        type ServiceContract = dict[str, ServiceEndpoint | ServiceProtocol]

        # Enhanced service monitoring types
        type ServiceMetrics = dict[str, int | float | str]
        type ServiceHealth = dict[str, bool | str | int]
        type ServiceStatus = dict[str, LifecycleState | ServiceMetrics | ServiceHealth]

    # =========================================================================
    # CONFIG TYPES - Configuration and settings
    # =========================================================================

    class Config:
        """Configuration types."""

        type Environment = Literal[
            "development",
            "staging",
            "production",
            "testing",
            "test",
            "local",
        ]
        type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
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
        type ServiceRegistry = dict[str, object]
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
        type FieldName = str
        type FieldValidator[T] = Callable[
            [T], object
        ]  # Simplified to break circular import
        type FieldValidators[T] = list[FlextTypes.Validation.FieldValidator[T]]
        type FieldValidatorRegistry[T] = dict[
            FlextTypes.Validation.FieldName, FlextTypes.Validation.FieldValidators[T]
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

        type OutputFormat = Literal["json", "yaml", "table", "csv", "text", "xml"]
        type SerializationFormat = Literal["json", "yaml", "toml", "ini", "xml"]
        type CompressionFormat = Literal["gzip", "bzip2", "xz", "lzma"]

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

        type ProjectType = Literal[
            "library",
            "application",
            "service",
            "cli",
            "web",
            "api",
            "PYTHON",
            "GO",
            "JAVASCRIPT",
        ]
        type ProjectStatus = Literal["active", "inactive", "deprecated", "archived"]
        type ProjectConfig = FlextTypes.Dict

    # =========================================================================
    # PROCESSING TYPES - Generic processing patterns
    # =========================================================================

    class Processing:
        """Generic processing types for ecosystem patterns."""

        # Processing status types
        type ProcessingStatus = Literal[
            "pending",
            "running",
            "completed",
            "failed",
            "cancelled",
        ]
        type ProcessingMode = Literal["batch", "stream", "parallel", "sequential"]
        type ValidationLevel = Literal["strict", "lenient", "standard"]
        type ProcessingPhase = Literal["prepare", "execute", "validate", "complete"]
        type HandlerType = Literal["command", "query", "event", "processor"]
        type WorkflowStatus = Literal[
            "pending",
            "running",
            "completed",
            "failed",
            "cancelled",
        ]

        WorkspaceStatus = Literal[
            "initializing",
            "ready",
            "error",
            "maintenance",
        ]

        type StepStatus = Literal[
            "pending",
            "running",
            "completed",
            "failed",
            "skipped",
        ]

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
        type Mode = Literal["command", "query", "event", "saga"]

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
        type CircuitState = Literal["closed", "open", "half_open"]
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

    # =========================================================================
    # CONTEXT TYPES - Context and scope management (NEW)
    # =========================================================================

    class Context:
        """Context and scope management types for cross-cutting concerns.

        This namespace provides types for managing execution context, scopes,
        and context propagation across service boundaries.

        Examples:
            Nested scope context:

            >>> scopes: FlextTypes.Context.ScopeRegistry = {
            ...     "global": {"user_id": "123", "tenant": "acme"},
            ...     "request": {"request_id": "abc", "path": "/api/users"},
            ... }

            Context hooks:

            >>> hooks: FlextTypes.Context.HookRegistry = {
            ...     "before_request": [validate_hook, auth_hook],
            ...     "after_request": [log_hook, metrics_hook],
            ... }

        """

        # Scope management
        type ScopeRegistry = dict[str, FlextTypes.Dict]

        # Context hooks
        type HookFunc = Callable[..., object]
        type HookList = list[FlextTypes.Context.HookFunc]
        type HookRegistry = dict[str, FlextTypes.Context.HookList]


# Module level exports for backward compatibility (temporary - to be removed)

T1 = FlextTypes.T1
T2 = FlextTypes.T2
T3 = FlextTypes.T3
Command = FlextTypes.Command
E = FlextTypes.E
Event = FlextTypes.Event
F = FlextTypes.F
K = FlextTypes.K
Message = FlextTypes.Message
MessageT = FlextTypes.MessageT
MessageT_contra = FlextTypes.MessageT_contra
Query = FlextTypes.Query
R = FlextTypes.R
ResultT = FlextTypes.ResultT
T = FlextTypes.T
T1_co = FlextTypes.T1_co
T2_co = FlextTypes.T2_co
T3_co = FlextTypes.T3_co
TAccumulate = FlextTypes.TAccumulate
TAggregate = FlextTypes.TAggregate
TAggregate_co = FlextTypes.TAggregate_co
TAsync = FlextTypes.TAsync
TAsync_co = FlextTypes.TAsync_co
TAwaitable = FlextTypes.TAwaitable
TCacheKey = FlextTypes.TCacheKey
TCacheKey_contra = FlextTypes.TCacheKey_contra
TCacheValue = FlextTypes.TCacheValue
TCacheValue_co = FlextTypes.TCacheValue_co
TClient = FlextTypes.TClient
TClient_co = FlextTypes.TClient_co
TCommand = FlextTypes.TCommand
TCommand_contra = FlextTypes.TCommand_contra
TConcurrent = FlextTypes.TConcurrent
TConfig = FlextTypes.TConfig
TConfigKey = FlextTypes.TConfigKey
TConfigKey_contra = FlextTypes.TConfigKey_contra
TConfigValue = FlextTypes.TConfigValue
TConfigValue_co = FlextTypes.TConfigValue_co
TConfig_co = FlextTypes.TConfig_co
TData = FlextTypes.TData
TData_co = FlextTypes.TData_co
TDomainEvent = FlextTypes.TDomainEvent
TDomainEvent_co = FlextTypes.TDomainEvent_co
TEntity = FlextTypes.TEntity
TEntity_co = FlextTypes.TEntity_co
TError = FlextTypes.TError
TError_co = FlextTypes.TError_co
TEvent = FlextTypes.TEvent
TEvent_contra = FlextTypes.TEvent_contra
TException = FlextTypes.TException
TException_co = FlextTypes.TException_co
TInput_contra = FlextTypes.TInput_contra
TItem = FlextTypes.TItem
TItem_contra = FlextTypes.TItem_contra
TKey = FlextTypes.TKey
TKey_contra = FlextTypes.TKey_contra
TMessage = FlextTypes.TMessage
TParallel = FlextTypes.TParallel
TPipeline = FlextTypes.TPipeline
TPlugin = FlextTypes.TPlugin
TPluginConfig = FlextTypes.TPluginConfig
TProcessor = FlextTypes.TProcessor
TQuery = FlextTypes.TQuery
TQuery_contra = FlextTypes.TQuery_contra
TResource = FlextTypes.TResource
TResult = FlextTypes.TResult
TResult_co = FlextTypes.TResult_co
TResult_contra = FlextTypes.TResult_contra
TService = FlextTypes.TService
TService_co = FlextTypes.TService_co
TSettings = FlextTypes.TSettings
TSettings_co = FlextTypes.TSettings_co
TState = FlextTypes.TState
TState_co = FlextTypes.TState_co
TSync = FlextTypes.TSync
TTimeout = FlextTypes.TTimeout
TTransform = FlextTypes.TTransform
TTransform_co = FlextTypes.TTransform_co
TUtil = FlextTypes.TUtil
TUtil_contra = FlextTypes.TUtil_contra
TValue = FlextTypes.TValue
TValueObject_co = FlextTypes.TValueObject_co
TValue_co = FlextTypes.TValue_co
T_co = FlextTypes.T_co
T_contra = FlextTypes.T_contra
U = FlextTypes.U
UParallel = FlextTypes.UParallel
UResource = FlextTypes.UResource
V = FlextTypes.V
W = FlextTypes.W
P = FlextTypes.P

__all__: list[str] = [
    # TypeVars for backward compatibility
    "T1",
    "T2",
    "T3",
    "Command",
    "E",
    "Event",
    "F",
    "FlextTypes",
    "K",
    "Message",
    "MessageT",
    "MessageT_contra",
    "P",
    "Query",
    "R",
    "ResultT",
    "T",
    "T1_co",
    "T2_co",
    "T3_co",
    "TAccumulate",
    "TAggregate",
    "TAggregate_co",
    "TAsync",
    "TAsync_co",
    "TAwaitable",
    "TCacheKey",
    "TCacheKey_contra",
    "TCacheValue",
    "TCacheValue_co",
    "TClient",
    "TClient_co",
    "TCommand",
    "TCommand_contra",
    "TConcurrent",
    "TConfig",
    "TConfigKey",
    "TConfigKey_contra",
    "TConfigValue",
    "TConfigValue_co",
    "TConfig_co",
    "TData",
    "TData_co",
    "TDomainEvent",
    "TDomainEvent_co",
    "TEntity",
    "TEntity_co",
    "TError",
    "TError_co",
    "TEvent",
    "TEvent_contra",
    "TException",
    "TException_co",
    "TInput_contra",
    "TItem",
    "TItem_contra",
    "TKey",
    "TKey_contra",
    "TMessage",
    "TParallel",
    "TPipeline",
    "TPlugin",
    "TPluginConfig",
    "TProcessor",
    "TQuery",
    "TQuery_contra",
    "TResource",
    "TResult",
    "TResult_co",
    "TResult_contra",
    "TService",
    "TService_co",
    "TSettings",
    "TSettings_co",
    "TState",
    "TState_co",
    "TSync",
    "TTimeout",
    "TTransform",
    "TTransform_co",
    "TUtil",
    "TUtil_contra",
    "TValue",
    "TValueObject_co",
    "TValue_co",
    "T_co",
    "T_contra",
    "U",
    "UParallel",
    "UResource",
    "V",
    "W",
]
