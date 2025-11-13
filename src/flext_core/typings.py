"""Layer 0: Thin Layer Type System Using Pydantic Native Types.

**ARCHITECTURE LAYER 0** - Pure Constants (Zero Dependencies)

This module provides a THIN LAYER over Pydantic's native type system:
- TypeVars at module level (Python 3.13+ native support)
- FlextTypes class with ONLY domain-specific complex types
- Direct use of Pydantic constrained types (PositiveInt, EmailStr, etc.)
- NO unnecessary type aliases - use Python native types directly

**THIN LAYER PRINCIPLE**:
- Remove: type Dict = dict[str, object] → Use dict[str, object] directly
- Remove: type StringList = list[str] → Use list[str] directly
- Add: Only domain-specific types not covered by Pydantic
- Use: Pydantic native types (PositiveInt, EmailStr, HttpUrl, etc.)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import socket
from collections.abc import Callable
from dataclasses import dataclass
from types import UnionType
from typing import (
    Annotated,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    TypeVar,
)

# Pydantic utilities - ONLY validators and Field (NOT re-exporting types)
# Users should import specialized types directly from pydantic:
#   from pydantic import EmailStr, PositiveInt, HttpUrl, etc.
from pydantic import (
    AfterValidator,
    Field,
)

# =============================================================================
# CORE TYPEVARS - Python 3.13+ Native (PEP 696 defaults)
# =============================================================================

# Core invariant TypeVars
T = TypeVar("T")
E = TypeVar("E")
F = TypeVar("F")
K = TypeVar("K")
P = ParamSpec("P")
R = TypeVar("R")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")

# Covariant TypeVars (_co suffix)
T_co = TypeVar("T_co", covariant=True)
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
T_Service_co = TypeVar("T_Service_co", covariant=True)
TResult_Handler_co = TypeVar("TResult_Handler_co", covariant=True)

# TypeVars specifically for Protocol definitions (Pyright requirement)
# Input must be contravariant, Result must be invariant for Protocol
TInput_Handler_Protocol_contra = TypeVar(
    "TInput_Handler_Protocol_contra", contravariant=True
)
TResult_Handler_Protocol = TypeVar("TResult_Handler_Protocol")

# Contravariant TypeVars (_contra suffix)
T_contra = TypeVar("T_contra", contravariant=True)
TCacheKey_contra = TypeVar("TCacheKey_contra", contravariant=True)
TCommand_contra = TypeVar("TCommand_contra", contravariant=True)
TConfigKey_contra = TypeVar("TConfigKey_contra", contravariant=True)
TEvent_contra = TypeVar("TEvent_contra", contravariant=True)
TInput_contra = TypeVar("TInput_contra", contravariant=True)
TItem_contra = TypeVar("TItem_contra", contravariant=True)
TQuery_contra = TypeVar("TQuery_contra", contravariant=True)
TResult_contra = TypeVar("TResult_contra", contravariant=True)
TUtil_contra = TypeVar("TUtil_contra", contravariant=True)
T_Validator_contra = TypeVar("T_Validator_contra", contravariant=True)
T_Repository_contra = TypeVar("T_Repository_contra", contravariant=True)
TInput_Handler_contra = TypeVar("TInput_Handler_contra", contravariant=True)
MessageT_contra = TypeVar("MessageT_contra", contravariant=True)

# Invariant advanced TypeVars
T_ResultProtocol = TypeVar("T_ResultProtocol")
CallableInputT = TypeVar("CallableInputT")
CallableOutputT = TypeVar("CallableOutputT")

# Domain-specific TypeVars
Command = TypeVar("Command")
Event = TypeVar("Event")
Message = TypeVar("Message")
Query = TypeVar("Query")
ResultT = TypeVar("ResultT")

# Module-level TypeVars (for compatibility)
MessageT = TypeVar("MessageT")
FactoryT = TypeVar("FactoryT")
TValidateAll = TypeVar("TValidateAll")

# TypeVars for generic processor operations
ProcessedDataT = TypeVar("ProcessedDataT")
ProcessorResultT = TypeVar("ProcessorResultT")
RegistryHandlerT = TypeVar("RegistryHandlerT")


# =============================================================================
# FLEXT TYPES - THIN LAYER (Domain-Specific Complex Types Only)
# =============================================================================


class FlextTypes:
    """THIN LAYER over Pydantic types - domain-specific complex types ONLY.

    This class contains ONLY domain-specific types that are NOT covered by
    Pydantic's native type system. For standard validation, use Pydantic
    types directly:

    **USE PYDANTIC DIRECTLY** (not FlextTypes):
    - PositiveInt, NegativeInt, NonNegativeInt
    - PositiveFloat, NegativeFloat, NonPositiveFloat
    - EmailStr, HttpUrl, AnyUrl
    - SecretStr, UUID4
    - FilePath, DirectoryPath
    - constr(min_length=1), conint(gt=0, lt=100)
    - conlist(str, min_length=1, max_length=10)

    **USE PYTHON NATIVE** (not FlextTypes):
    - dict[str, object] instead of FlextTypes.Dict
    - list[str] instead of FlextTypes.StringList
    - dict[str, str] instead of FlextTypes.StringDict

    **ONLY IN FLEXT TYPES**:
    - FlextResult-specific types
    - Railway pattern types
    - CQRS/Event Sourcing types
    - Complex handler/callable signatures
    """

    # =========================================================================
    # DOMAIN VALIDATION TYPES - Annotated constraints (Phase 3 - Best Practices)
    # =========================================================================

    # Network types with Pydantic v2 constraints
    type PortNumber = Annotated[
        int,
        Field(ge=1, le=65535, description="Network port (1-65535)"),
    ]

    type TimeoutSeconds = Annotated[
        float,
        Field(gt=0, le=300, description="Timeout in seconds (max 5 min)"),
    ]

    type RetryCount = Annotated[
        int,
        Field(ge=0, le=10, description="Retry attempts (0-10)"),
    ]

    # String types with constraints
    type NonEmptyStr = Annotated[
        str,
        Field(min_length=1, description="Non-empty string"),
    ]

    type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    # Complex validators (with AfterValidator for DNS checks)
    @staticmethod
    def _validate_hostname(value: str) -> str:
        """Validate hostname can be resolved via DNS."""
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

    # =========================================================================
    # JSON TYPES - Python native with JSON compatibility
    # =========================================================================

    type JsonPrimitive = str | int | float | bool | None
    # JSON recursive type using forward reference (Pyrefly compatible)
    # Python 3.13 native syntax would be: JsonPrimitive | dict[str, JsonValue] | list[JsonValue]
    # But Pyrefly doesn't support PEP 695 recursive types yet, so we use string forward reference
    type JsonValue = JsonPrimitive | dict[str, JsonValue] | list[JsonValue]
    type JsonList = list[JsonValue]
    type JsonDict = dict[str, JsonValue]

    # =========================================================================
    # FLEXT RESULT TYPES - Railway pattern (domain-specific)
    # =========================================================================
    # Note: We define a minimal structural protocol to avoid circular imports
    # with FlextResult (Tier 1) while providing useful typing guarantees.

    class ResultLike(Protocol[T_co]):
        """Structural protocol satisfied by FlextResult[T]."""

        @property
        def is_success(self) -> bool:
            """Check if result represents success."""
            ...

        @property
        def is_failure(self) -> bool:
            """Check if result represents failure."""
            ...

        @property
        def value(self) -> T_co:
            """Get the success value."""
            ...

        @property
        def error(self) -> str | None:
            """Get the error message if failure."""
            ...

        def unwrap(self) -> T_co:
            """Unwrap the result value or raise exception."""
            ...

    # TypeAlias is correct inside class scope - PEP 695 type statement doesn't work here
    FlextResultType: TypeAlias = "ResultLike[T]"
    # Validator type aliases - these are callable types, not generic type constructors
    # Use them directly without subscripting (they're already parameterized by T)
    ValidationRule: TypeAlias = Callable[[T], "ResultLike[T]"]
    CrossFieldValidator: TypeAlias = Callable[[T], "ResultLike[T]"]
    ValidatorFunction: TypeAlias = Callable[[T], "ResultLike[T]"]

    # =========================================================================
    # HANDLER TYPES - CQRS/Bus patterns (domain-specific)
    # =========================================================================

    HandlerCallable: TypeAlias = (
        "Callable[[CallableInputT], ResultLike[CallableOutputT]]"
    )
    # Generic handler types with explicit TypeVar declarations
    CallableHandlerType: TypeAlias = (
        "Callable[[TInput_Handler_contra], ResultLike[TResult_Handler_co]]"
    )
    BusHandlerType: TypeAlias = (
        "Callable[[MessageT_contra], ResultLike[TResult_Handler_co]]"
    )
    MiddlewareType: TypeAlias = (
        "Callable[[MessageT_contra], ResultLike[TResult_Handler_co]]"
    )
    MiddlewareConfig: TypeAlias = JsonDict

    # =========================================================================
    # PROCESSOR TYPES - Data processing (domain-specific)
    # =========================================================================

    type ProcessorInputType = object | dict[str, object] | str | int | float | bool
    type ProcessorOutputType = (
        object | dict[str, object] | str | int | float | bool | None
    )

    # =========================================================================
    # FACTORY TYPES - Dependency injection (domain-specific)
    # =========================================================================

    type FactoryCallableType = Callable[[], object]

    # =========================================================================
    # PREDICATE TYPES - Business rules (domain-specific)
    # =========================================================================

    type PredicateType = Callable[[object], bool]

    # =========================================================================
    # DECORATOR TYPES - Cross-cutting concerns (domain-specific)
    # =========================================================================

    type DecoratorReturnType = Callable[
        [Callable[[object], object]], Callable[[object], object]
    ]

    # =========================================================================
    # UTILITY TYPES - FlextUtilities domain-specific types
    # =========================================================================

    # Complex types that provide semantic value
    type SerializableType = (
        object | dict[str, object] | list[object] | str | int | float | bool | None
    )
    type TypeOriginSpecifier = object
    type GenericTypeArgument = type | str | object
    type TypeHintSpecifier = type | str | UnionType | None
    type ContextualObjectType = object
    type ContainerServiceType = object

    # Cache and validation utility types
    type ObjectList = list[object]
    type CachedObjectType = (
        object  # Object with cache attributes (e.g., _cache, __cache__)
    )
    type ParameterValueType = object  # Configuration parameter value
    type SortableObjectType = object  # Object that can be sorted/ordered
    type GenericDetailsType = object | dict[str, object]  # Generic details/metadata
    type MetadataDict = dict[str, object]  # Generic metadata dictionary

    # Container/DI types
    type ServiceRegistry = dict[str, object]  # Service instance registry
    type FactoryRegistry = dict[str, Callable[[], object]]  # Factory function registry
    type ContainerConfig = dict[str, object]  # Container configuration

    # CQRS Payload types
    type EventPayload = dict[str, object]  # Event data payload
    type CommandPayload = dict[str, object]  # Command data payload
    type QueryPayload = dict[str, object]  # Query data payload

    # =========================================================================
    # BUS/HANDLER TYPES - Extended CQRS/Event Sourcing types
    # =========================================================================

    type BusMessageType = object | dict[str, object]
    # Accepts plain types, generic aliases (e.g., dict[str, object]), and string references
    type MessageTypeSpecifier = object
    type MessageTypeOrHandlerType = type | str | Callable[[object], object]
    type HandlerOrCallableType = Callable[[object], object] | object
    type HandlerConfigurationType = dict[str, object] | None
    type HandlerCallableType = Callable[[object], object]
    type AcceptableMessageType = object | dict[str, object]

    # =========================================================================
    # LOGGING TYPES - FlextLogger domain-specific types
    # =========================================================================

    type LoggingContextType = dict[str, object]
    type LoggingContextValueType = object | str | int | float | bool | None
    type LoggerContextType = dict[str, object]
    type LoggingProcessorType = Callable[[object, str, dict[str, object]], None]
    type LoggingArgType = object
    type LoggingKwargsType = dict[str, object]
    type BoundLoggerType = object  # structlog.BoundLogger type

    # =========================================================================
    # HOOK/REGISTRY TYPES - Plugin/Extension system types
    # =========================================================================

    type HookCallableType = Callable[[object], object]
    type HookRegistry = dict[str, list[Callable[[object], object]]]
    type ScopeRegistry = dict[str, object]

    # =========================================================================
    # VALIDATOR TYPES - Validation framework types
    # =========================================================================
    # Note: Validator types defined above using forward references to FlextResult

    @dataclass(frozen=True)
    class RetryConfig:
        """Configuration for retry operations.

        Attributes:
            max_attempts: Maximum number of retry attempts
            initial_delay_seconds: Initial delay between retries
            max_delay_seconds: Maximum delay between retries
            exponential_backoff: Whether to use exponential backoff
            retry_on_exceptions: List of exception types to retry on
            backoff_multiplier: Optional multiplier for backoff calculation

        """

        max_attempts: int
        initial_delay_seconds: float
        max_delay_seconds: float
        exponential_backoff: bool
        retry_on_exceptions: list[type[Exception]]
        backoff_multiplier: float | None = None

    # =========================================================================


# =============================================================================
# MODULE-LEVEL EXPORTS - Domain validation types for convenient importing
# =============================================================================
# These are re-exported from FlextTypes for direct import:
# from flext_core import PortNumber, TimeoutSeconds, RetryCount, NonEmptyStr, LogLevel, HostName

PortNumber = FlextTypes.PortNumber
TimeoutSeconds = FlextTypes.TimeoutSeconds
RetryCount = FlextTypes.RetryCount
NonEmptyStr = FlextTypes.NonEmptyStr
LogLevel = FlextTypes.LogLevel
HostName = FlextTypes.HostName


__all__: list[str] = [
    # TypeVars (domain-specific for FlextCore architecture)
    "CallableInputT",
    "CallableOutputT",
    "Command",
    "E",
    "Event",
    "F",
    "FactoryT",
    "FlextTypes",
    # Domain validation types (Phase 3 - Annotated constraints)
    "HostName",
    "K",
    "LogLevel",
    "Message",
    "MessageT",
    "MessageT_contra",
    "NonEmptyStr",
    "P",
    "PortNumber",
    "ProcessedDataT",
    "ProcessorResultT",
    "Query",
    "R",
    "RegistryHandlerT",
    "ResultT",
    "RetryCount",
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
    "TInput_Handler_contra",
    "TInput_contra",
    "TItem_contra",
    "TQuery_contra",
    "TResult_Handler_co",
    "TResult_co",
    "TResult_contra",
    "TState_co",
    "TUtil_contra",
    "TValidateAll",
    "TValueObject_co",
    "TValue_co",
    "T_Repository_contra",
    "T_ResultProtocol",
    "T_Service_co",
    "T_Validator_contra",
    "T_co",
    "T_contra",
    "TimeoutSeconds",
    "U",
    "V",
    "W",
]
