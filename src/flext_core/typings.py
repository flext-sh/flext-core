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

from collections import OrderedDict
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Literal,
    ParamSpec,
    TypedDict,
    TypeVar,
)

if TYPE_CHECKING:
    from flext_core.result import FlextResult

# =============================================================================
# CENTRALIZED TYPE VARIABLES - All TypeVars for the entire FLEXT ecosystem
# =============================================================================

# Core generic type variables
P = ParamSpec("P")

# Core TypeVars
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")

# Covariant type variables (read-only)
T1_co = TypeVar("T1_co", covariant=True)
T2_co = TypeVar("T2_co", covariant=True)
T3_co = TypeVar("T3_co", covariant=True)

# Contravariant type variables (write-only)
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

# Domain-specific type variables
Message = TypeVar("Message")
Command = TypeVar("Command")
Query = TypeVar("Query")
Event = TypeVar("Event")
ResultT = TypeVar("ResultT")
TCommand = TypeVar("TCommand")
TEvent = TypeVar("TEvent")
TQuery = TypeVar("TQuery")

# Service and infrastructure type variables
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
TConcurrent = TypeVar("TConcurrent")
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
TParallel = TypeVar("TParallel")
TResource = TypeVar("TResource")
TResult_co = TypeVar("TResult_co", covariant=True)
TService = TypeVar("TService")
TTimeout = TypeVar("TTimeout")
TValue = TypeVar("TValue")
TValue_co = TypeVar("TValue_co", covariant=True)
TValueObject_co = TypeVar("TValueObject_co", covariant=True)
UParallel = TypeVar("UParallel")
UResource = TypeVar("UResource")
TPlugin = TypeVar("TPlugin")
TPluginConfig = TypeVar("TPluginConfig")

# =============================================================================
# FLEXT TYPES NAMESPACE - Centralized type system for the FLEXT ecosystem
# =============================================================================


class FlextTypes:
    """Centralized type system namespace for the FLEXT ecosystem.

    Provides comprehensive, organized type system following strict
    Flext standards with single source of truth for all definitions.

    **Function**: Type definitions for ecosystem-wide consistency
        - Core fundamental types (Dict, List, Headers)
        - Configuration types (ConfigValue, ConfigDict)
        - JSON types with Python 3.13+ syntax
        - Message types for CQRS patterns
        - Handler types for command/query handlers
        - Service types for domain services
        - Protocol types for interface definitions
        - Plugin types for extensibility
        - Generic type variables (T, U, V, W)
        - Covariant and contravariant type variables

    **Uses**: Core Python type system infrastructure
        - typing.TypeVar for generic type variables
        - typing.ParamSpec for parameter specifications
        - typing.Literal for literal type hints
        - collections.abc.Callable for callable types
        - Python 3.13+ union syntax (X | Y)
        - Modern type aliases with type keyword
        - Covariant and contravariant variance
        - Generic type constraints

    **How to use**: Type annotations and type safety
        ```python
        from flext_core import FlextTypes


        # Example 1: Use core dict type
        def process_data(data: FlextTypes.Dict) -> FlextTypes.Dict:
            return {"processed": True, **data}


        # Example 2: Use configuration types
        def load_config() -> FlextTypes.ConfigDict:
            return {
                "timeout": FlextConstants.Network.DEFAULT_TIMEOUT,
                "retries": FlextConstants.Reliability.DEFAULT_MAX_RETRIES,
            }


        # Example 3: Use headers type
        def build_headers(token: str) -> FlextTypes.StringDict:
            return {"Authorization": f"Bearer {token}"}


        # Example 4: Use message types for CQRS
        class CreateUserCommand(FlextTypes.Message.Command):
            email: str
            name: str


        # Example 5: Use handler types
        def create_handler(
            cmd: FlextTypes.Message.Command,
        ) -> FlextTypes.Handler.HandlerResult:
            return FlextResult[User].ok(User())


        # Example 6: Use generic type variables
        def transform[T](items: list[T]) -> list[T]:
            return [item for item in items]


        # Example 7: Use string lists
        tags: FlextTypes.StringList = ["tag1", "tag2"]
        ```

        - [ ] Add type validation decorators
        - [ ] Implement runtime type checking utilities
        - [ ] Support type narrowing helpers
        - [ ] Add type transformation utilities
        - [ ] Implement type compatibility checking
        - [ ] Support type documentation generation
        - [ ] Add type migration tools
        - [ ] Implement type testing utilities
        - [ ] Support type versioning
        - [ ] Add type analysis and inspection

    Attributes:
        Core: Core fundamental types for ecosystem.
        Message: Message types for CQRS patterns.
        Handler: Handler types for command/query.
        Service: Service types for domain logic.
        Protocol: Protocol types for interfaces.
        Plugin: Plugin types for extensibility.

    Note:
        All types use Python 3.13+ modern syntax.
        Type variables support covariance/contravariance.
        Type aliases use modern type keyword syntax.
        No wrappers or legacy patterns allowed.
        Single source of truth for all ecosystem types.

    Warning:
        Type changes may impact entire ecosystem.
        Generic type variables must match usage patterns.
        Covariance/contravariance must be used correctly.
        Type aliases should not wrap existing types.

    Example:
        Complete type usage with generics:

        >>> def process[T](data: "Dict") -> "Dict":
        ...     return data
        >>> result = process({"key": "value"})
        >>> print(result)
        {'key': 'value'}

    See Also:
        FlextResult: For result type patterns.
        FlextModels: For domain model types.
        FlextHandlers: For handler type usage.
        FlextProtocols: For protocol definitions.

    """

    # =========================================================================
    # CORE TYPES - Fundamental building blocks
    # =========================================================================

    # Basic collection types
    type Dict = dict[str, object]
    type List = list[object]
    type StringList = list[str]
    type IntList = list[int]
    type FloatList = list[float]
    type BoolList = list[bool]

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

    # Collection types with ordering
    type OrderedDictType = OrderedDict[str, object]

    # Assign for backward compatibility - moved outside class
    # OrderedDict = OrderedDict[str, object]  # Will be assigned after class definition

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

        type EventHandler = Callable[[FlextTypes.Domain.EventTyped], FlextResult[None]]
        type EventHandlerList = list[FlextTypes.Domain.EventHandler]
        type EventHandlerRegistry = dict[
            FlextTypes.Domain.EventType, FlextTypes.Domain.EventHandlerList
        ]

        type AggregateState = dict[str, object | FlextTypes.List]
        type AggregateVersion = int

    # =========================================================================
    # SERVICE TYPES - Service layer patterns
    # =========================================================================

    class Service:
        """Service layer types."""

        type Dict = FlextTypes.Dict
        type Type = Literal["instance", "factory"]
        type FactoryDict = dict[str, Callable[[], object]]

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

    # =========================================================================
    # VALIDATION TYPES - Validation patterns (ENHANCED for complex scenarios)
    # =========================================================================

    class Validation:
        """Validation types for complex validation scenarios.

        This namespace provides types for field-level validators, business rules,
        invariant checking, and consistency rules across domain models.

        Examples:
            Field validators with FlextResult:

            >>> field_validator: FlextTypes.Validation.FieldValidator[str] = (
            ...     lambda value: FlextResult[None].ok(None)
            ...     if "@" in value
            ...     else FlextResult[None].fail("Invalid email")
            ... )

        """

        # Basic validators
        type Rule = Callable[[object], bool]
        type Validator = Rule
        type BusinessRule = Rule

        # Complex validation patterns (NEW - high value)
        type FieldName = str
        type FieldValidator[T] = Callable[[T], FlextResult[None]]
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

        type ConsistencyRule[T, U] = Callable[[T, U], FlextResult[None]]
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
        type HandlerFunc = Callable[[object], FlextResult[object]]
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
        type SagaStep[T] = Callable[[T], FlextResult[T]]
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

        # Note: Type variables are assigned at module level below


# Assign type variables to FlextTypes class for backward compatibility
FlextTypes.T = T
FlextTypes.U = U
FlextTypes.V = V
FlextTypes.W = W
FlextTypes.OrderedDict = OrderedDict[str, object]


__all__: list[str] = [
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
    "TCacheKey",
    "TCacheKey_contra",
    "TCacheValue",
    "TCacheValue_co",
    "TCommand",
    "TCommand_contra",
    "TConcurrent",
    "TConfigKey",
    "TConfigKey_contra",
    "TConfigValue",
    "TConfigValue_co",
    "TDomainEvent",
    "TDomainEvent_co",
    "TEntity",
    "TEntity_co",
    "TEvent",
    "TEvent_contra",
    "TInput_contra",
    "TItem",
    "TItem_contra",
    "TKey",
    "TKey_contra",
    "TMessage",
    "TParallel",
    "TPlugin",
    "TPluginConfig",
    "TQuery",
    "TQuery_contra",
    "TResource",
    "TResult",
    "TResult_co",
    "TResult_contra",
    "TService",
    "TState",
    "TState_co",
    "TTimeout",
    "TUtil",
    "TUtil_contra",
    "TValue",
    "TValueObject_co",
    "TValue_co",
    "T_contra",
    "U",
    "UParallel",
    "UResource",
    "V",
    "W",
]
