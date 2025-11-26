"""Type system foundation module for FLEXT ecosystem.

Provides FlextTypes, a comprehensive type system foundation defining TypeVars at
module level and complex type aliases organized in nested namespaces within the
FlextTypes class. Uses Pydantic Field annotations for domain validation types and
provides type aliases for CQRS patterns, JSON serialization, handlers, processors,
factories, predicates, decorators, utilities, logging, hooks, and configuration.

Scope: Module-level TypeVars for generic programming (T, T_co, T_contra, P, R,
domain-specific TypeVars), and FlextTypes class with nested namespaces (Validation,
Json, Handler, Processor, Factory, Predicate, Decorator, Utility, Bus, Logging,
Hook, Config, Cqrs) containing type aliases and validation models. Type aliases
are also exported at module level for convenience. Follows single-class pattern
with nested namespaces for organization. TypeVars use appropriate variance (covariant,
contravariant) for protocol compliance and type safety.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import socket
from collections.abc import Callable
from typing import Annotated, ParamSpec, TypeAlias, TypeVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

# ============================================================================
# Module-Level TypeVars - Core and Domain-Specific
# ============================================================================
# All TypeVars are defined at module level for clarity and accessibility

# Core TypeVars - covariant, contravariant, and generic type variables
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
P = ParamSpec("P")  # ParamSpec for decorator patterns
R = TypeVar("R")  # Return type for decorators

# Handler-specific TypeVars for protocol compliance
TInput_Handler_Protocol_contra = TypeVar(
    "TInput_Handler_Protocol_contra", contravariant=True
)
TResult_Handler_Protocol = TypeVar("TResult_Handler_Protocol")
CallableInputT = TypeVar("CallableInputT")  # Input type for callable handlers
CallableOutputT = TypeVar("CallableOutputT")  # Output type for callable handlers

# Domain TypeVars for CQRS and domain-driven design patterns
TEntity_co = TypeVar("TEntity_co", covariant=True)
TResult_co = TypeVar("TResult_co", covariant=True)
TCommand_contra = TypeVar("TCommand_contra", contravariant=True)
TQuery_contra = TypeVar("TQuery_contra", contravariant=True)
MessageT_contra = TypeVar("MessageT_contra", contravariant=True)
TResult_Handler_co = TypeVar("TResult_Handler_co", covariant=True)
TInput_Handler_contra = TypeVar("TInput_Handler_contra", contravariant=True)

# Generic TypeVars for utility and collection operations
E = TypeVar("E")  # Element type
F = TypeVar("F")  # Function type
K = TypeVar("K")  # Key type
ResultT = TypeVar("ResultT")  # Result type
U = TypeVar("U")  # Utility type
V = TypeVar("V")  # Value type
W = TypeVar("W")  # Wrapper type

# Additional domain TypeVars for advanced patterns
T1_co = TypeVar("T1_co", covariant=True)
T2_co = TypeVar("T2_co", covariant=True)
T3_co = TypeVar("T3_co", covariant=True)
TAggregate_co = TypeVar("TAggregate_co", covariant=True)
TCacheKey_contra = TypeVar("TCacheKey_contra", contravariant=True)
TCacheValue_co = TypeVar("TCacheValue_co", covariant=True)
TConfigKey_contra = TypeVar("TConfigKey_contra", contravariant=True)
TDomainEvent_co = TypeVar("TDomainEvent_co", covariant=True)
TEvent_contra = TypeVar("TEvent_contra", contravariant=True)
TInput_contra = TypeVar("TInput_contra", contravariant=True)
TItem_contra = TypeVar("TItem_contra", contravariant=True)
TState_co = TypeVar("TState_co", covariant=True)
TUtil_contra = TypeVar("TUtil_contra", contravariant=True)
TValue_co = TypeVar("TValue_co", covariant=True)
TValueObject_co = TypeVar("TValueObject_co", covariant=True)
TResult_contra = TypeVar("TResult_contra", contravariant=True)

# Factory TypeVar for dependency injection patterns
FactoryT = TypeVar("FactoryT", bound=Callable[[], object])

# Config-specific TypeVars
T_Config = TypeVar("T_Config")  # Bound removed - forward ref causes circular import
T_Namespace = TypeVar("T_Namespace")


class FlextTypes:
    """Type system foundation for FLEXT ecosystem - Complex Type Aliases Only.

    NOTE: All TypeVars are now defined at module level (see above).
    This class contains only complex type aliases, domain validation types,
    and nested helper classes.

    Architecture: Nested type definitions for:
    - Domain validation types (PortNumber, TimeoutSeconds, etc.)
    - JSON types (JsonPrimitive, JsonValue, etc.)
    - Handler types (HandlerCallable, BusHandlerType, etc.)
    - Utility types (SerializableType, MetadataDict, etc.)
    - CQRS pattern types (Command, Event, Message, Query)
    - Configuration models (RetryConfig)
    """

    # Direct type aliases for compatibility (used by pyrefly)
    ObjectList: TypeAlias = list[object]
    GenericDetailsType: TypeAlias = object | dict[str, object]
    SortableObjectType: TypeAlias = object
    CachedObjectType: TypeAlias = object
    ParameterValueType: TypeAlias = object
    MessageTypeSpecifier: TypeAlias = object
    TypeOriginSpecifier: TypeAlias = object
    SerializableType: TypeAlias = (
        object | dict[str, object] | list[object] | str | int | float | bool | None
    )
    AcceptableMessageType: TypeAlias = (
        object | dict[str, object] | str | int | float | bool
    )
    HookRegistry: TypeAlias = dict[str, list[Callable[[object], object]]]
    ScopeRegistry: TypeAlias = dict[str, object]
    HookCallableType: TypeAlias = Callable[[object], object]

    class Validation:
        """Domain validation types using Pydantic Field annotations."""

        # Network validation types
        type PortNumber = Annotated[
            int, Field(ge=1, le=65535, description="Network port")
        ]
        type TimeoutSeconds = Annotated[
            float, Field(gt=0, le=300, description="Timeout in seconds")
        ]
        type RetryCount = Annotated[
            int, Field(ge=0, le=10, description="Retry attempts")
        ]

        # String validation types
        type NonEmptyStr = Annotated[
            str, Field(min_length=1, description="Non-empty string")
        ]

        @staticmethod
        def _validate_hostname(value: str) -> str:
            """Validate hostname by attempting DNS resolution."""
            try:
                socket.gethostbyname(value)
                return value
            except socket.gaierror as e:
                msg = f"Cannot resolve hostname '{value}': {e}"
                raise ValueError(msg) from e

        type HostName = Annotated[
            str,
            Field(min_length=1, max_length=253, description="Valid hostname"),
            AfterValidator(_validate_hostname),
        ]

    class Json:
        """JSON serialization types for API and data exchange."""

        # Primitive JSON types
        type JsonPrimitive = str | int | float | bool | None

        # Complex JSON types - direct reference (no string annotation)
        JsonValue: TypeAlias = JsonPrimitive | dict[str, object] | list[object]
        type JsonList = list[JsonValue]
        type JsonDict = dict[str, JsonValue]

    # JSON types for convenience (after Json class definition)
    JsonValue: TypeAlias = Json.JsonValue
    JsonDict: TypeAlias = Json.JsonDict

    class Handler:
        """Handler and middleware type definitions for CQRS patterns."""

        # Protocol-based handler types using simplified type aliases
        ValidationRule: TypeAlias = Callable[[object], object]
        CrossFieldValidator: TypeAlias = Callable[[object], object]
        ValidatorFunction: TypeAlias = Callable[[object], object]

        # Callable handler types with contravariant input, covariant output
        HandlerCallable: TypeAlias = Callable[[object], object]
        CallableHandlerType: TypeAlias = Callable[[object], object]
        BusHandlerType: TypeAlias = Callable[[object], object]
        MiddlewareType: TypeAlias = Callable[[object], object]

        # Middleware configuration types
        MiddlewareConfig: TypeAlias = dict[str, object]

    class Processor:
        """Processor types for data transformation pipelines."""

        # Input/output types for processing operations
        type ProcessorInputType = object | dict[str, object] | str | int | float | bool
        type ProcessorOutputType = (
            object | dict[str, object] | str | int | float | bool | None
        )

    class Factory:
        """Factory pattern types for dependency injection."""

        # Factory callable and type variable definitions
        type FactoryCallableType = Callable[[], object]
        FactoryT = TypeVar("FactoryT", bound=Callable[[], object])

    class Predicate:
        """Predicate types for filtering and validation operations."""

        type PredicateType = Callable[[object], bool]

    class Decorator:
        """Decorator pattern types for function enhancement."""

        type DecoratorReturnType = Callable[
            [Callable[[object], object]], Callable[[object], object]
        ]

    class Utility:
        """General utility types for common operations."""

        # Serialization and type hinting types
        type SerializableType = (
            object | dict[str, object] | list[object] | str | int | float | bool | None
        )
        type TypeOriginSpecifier = object
        type TypeHintSpecifier = object
        type GenericTypeArgument = str | object
        type ContextualObjectType = object
        type ContainerServiceType = object

        # Collection and caching types
        type ObjectList = list[object]
        CachedObjectType: TypeAlias = object
        type ParameterValueType = object
        type SortableObjectType = object
        type GenericDetailsType = object | dict[str, object]

        # Additional utility types for compatibility

        # Registry and configuration types
        type MetadataDict = dict[str, object]
        type ServiceRegistry = dict[str, object]
        type FactoryRegistry = dict[str, Callable[[], object]]
        type ContainerConfig = dict[str, object]

        # Event and payload types
        type EventPayload = dict[str, object]
        type CommandPayload = dict[str, object]
        type QueryPayload = dict[str, object]

    class Bus:
        """Message bus and handler type definitions."""

        # Message and handler types for bus operations
        type BusMessageType = object | dict[str, object]
        type MessageTypeOrHandlerType = type | str | Callable[[object], object]
        type HandlerOrCallableType = Callable[[object], object] | object
        type HandlerConfigurationType = dict[str, object] | None
        type HandlerCallableType = Callable[[object], object]
        type AcceptableMessageType = object | dict[str, object]

    class Logging:
        """Logging and context type definitions."""

        # Context and processor types for logging operations
        type LoggingContextType = dict[str, object]
        type LoggingContextValueType = object | str | int | float | bool | None
        type LoggerContextType = dict[str, object]
        type LoggingProcessorType = Callable[[object, str, dict[str, object]], None]
        type LoggingArgType = object
        type LoggingKwargsType = dict[str, object]
        type BoundLoggerType = object

    class Hook:
        """Hook and registry types for extensibility."""

        # Hook callable and registry types
        type HookCallableType = Callable[[object], object]
        type HookRegistry = dict[str, list[Callable[[object], object]]]
        type ScopeRegistry = dict[str, object]

    class Config:
        """Configuration models for operational settings."""

        class RetryConfig(BaseModel):
            """Configuration for retry operations with exponential backoff.

            Defines retry behavior for operations that may fail and need
            automatic retry with configurable backoff strategies.
            """

            model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

            # Retry attempt limits
            max_attempts: int = Field(ge=1, description="Maximum retry attempts")

            # Timing configuration
            initial_delay_seconds: float = Field(
                gt=0.0, description="Initial delay in seconds"
            )
            max_delay_seconds: float = Field(
                gt=0.0, description="Maximum delay in seconds"
            )

            # Backoff strategy configuration
            exponential_backoff: bool = Field(description="Enable exponential backoff")
            backoff_multiplier: float = Field(
                default=2.0, gt=0.0, description="Backoff multiplier"
            )

            # Exception handling configuration
            retry_on_exceptions: list[type[Exception]] = Field(
                description="Exception types to retry on"
            )

    class Cqrs:
        """CQRS pattern type aliases with simplified forward references."""

        # Simplified forward references to avoid circular imports
        Command: TypeAlias = object  # Simplified for type resolution
        Event: TypeAlias = object
        Message: TypeAlias = object
        Query: TypeAlias = object


# NOTE: All TypeVars are defined at module level (lines 28-81)
# No re-export needed - they're already available for direct import

# Type aliases from nested classes (these ARE inside FlextTypes)
Command: TypeAlias = FlextTypes.Cqrs.Command
Event: TypeAlias = FlextTypes.Cqrs.Event
Message: TypeAlias = FlextTypes.Cqrs.Message
Query: TypeAlias = FlextTypes.Cqrs.Query

# Validation types
RetryCount: TypeAlias = FlextTypes.Validation.RetryCount
TimeoutSeconds: TypeAlias = FlextTypes.Validation.TimeoutSeconds

# Utility types
ObjectList: TypeAlias = FlextTypes.Utility.ObjectList
GenericDetailsType: TypeAlias = FlextTypes.Utility.GenericDetailsType
SortableObjectType: TypeAlias = FlextTypes.Utility.SortableObjectType
CachedObjectType: TypeAlias = FlextTypes.Utility.CachedObjectType
ParameterValueType: TypeAlias = FlextTypes.Utility.ParameterValueType
AcceptableMessageType: TypeAlias = FlextTypes.AcceptableMessageType

# Bus types
MessageTypeSpecifier: TypeAlias = FlextTypes.MessageTypeSpecifier
TypeOriginSpecifier: TypeAlias = FlextTypes.Utility.TypeOriginSpecifier

# JSON types
JsonValue: TypeAlias = FlextTypes.Json.JsonValue
JsonDict: TypeAlias = FlextTypes.Json.JsonDict

# Hook types
HookRegistry: TypeAlias = FlextTypes.Hook.HookRegistry
ScopeRegistry: TypeAlias = FlextTypes.Hook.ScopeRegistry
HookCallableType: TypeAlias = FlextTypes.Hook.HookCallableType

# Config types
RetryConfig: TypeAlias = FlextTypes.Config.RetryConfig

__all__ = [
    "AcceptableMessageType",
    "CachedObjectType",
    "CallableInputT",
    "CallableOutputT",
    "Command",
    "E",
    "Event",
    "F",
    "FactoryT",
    "FlextTypes",
    "GenericDetailsType",
    "HookCallableType",
    "HookRegistry",
    "JsonDict",
    "JsonValue",
    "K",
    "Message",
    "MessageT_contra",
    "MessageTypeSpecifier",
    "ObjectList",
    "P",
    "ParameterValueType",
    "Query",
    "R",
    "ResultT",
    "RetryConfig",
    "RetryCount",
    "ScopeRegistry",
    "SortableObjectType",
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
    "TInput_Handler_Protocol_contra",
    "TInput_Handler_contra",
    "TInput_contra",
    "TItem_contra",
    "TQuery_contra",
    "TResult_Handler_Protocol",
    "TResult_Handler_co",
    "TResult_co",
    "TResult_contra",
    "TState_co",
    "TUtil_contra",
    "TValueObject_co",
    "TValue_co",
    "T_co",
    "T_contra",
    "TimeoutSeconds",
    "TypeOriginSpecifier",
    "U",
    "V",
    "W",
]
