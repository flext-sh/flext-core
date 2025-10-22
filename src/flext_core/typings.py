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
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Literal,
    ParamSpec,
    TypeVar,
)

# Pydantic native types for direct use
from pydantic import (
    # UUID types
    UUID1,
    UUID3,
    UUID4,
    UUID5,
    AfterValidator,
    AnyUrl,
    AwareDatetime,
    DirectoryPath,
    EmailStr,
    Field,
    FilePath,
    FileUrl,
    FutureDate,
    HttpUrl,
    NaiveDatetime,
    NegativeFloat,
    NegativeInt,
    NewPath,
    NonNegativeFloat,
    NonNegativeInt,
    NonPositiveFloat,
    NonPositiveInt,
    PastDate,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    confloat,
    conint,
    conlist,
    conset,
    constr,
)

if TYPE_CHECKING:
    from flext_core.result import FlextResult

# =============================================================================
# CORE TYPEVARS - Python 3.13+ Native (PEP 696 defaults)
# =============================================================================

# Core invariant TypeVars
T = TypeVar("T", default=object)
E = TypeVar("E", default=object)
F = TypeVar("F", default=object)
K = TypeVar("K", default=object)
P = ParamSpec("P")
R = TypeVar("R", default=object)
U = TypeVar("U", default=object)
V = TypeVar("V", default=object)
W = TypeVar("W", default=object)

# Covariant TypeVars (_co suffix)
T_co = TypeVar("T_co", covariant=True, default=object)
T1_co = TypeVar("T1_co", covariant=True, default=object)
T2_co = TypeVar("T2_co", covariant=True, default=object)
T3_co = TypeVar("T3_co", covariant=True, default=object)
TAggregate_co = TypeVar("TAggregate_co", covariant=True, default=object)
TCacheValue_co = TypeVar("TCacheValue_co", covariant=True, default=object)
TDomainEvent_co = TypeVar("TDomainEvent_co", covariant=True, default=object)
TEntity_co = TypeVar("TEntity_co", covariant=True, default=object)
TResult_co = TypeVar("TResult_co", covariant=True, default=object)
TState_co = TypeVar("TState_co", covariant=True, default=object)
TValue_co = TypeVar("TValue_co", covariant=True, default=object)
TValueObject_co = TypeVar("TValueObject_co", covariant=True, default=object)
T_Service_co = TypeVar("T_Service_co", covariant=True, default=object)
TResult_Handler_co = TypeVar("TResult_Handler_co", covariant=True, default=object)

# Contravariant TypeVars (_contra suffix)
T_contra = TypeVar("T_contra", contravariant=True, default=object)
TCacheKey_contra = TypeVar("TCacheKey_contra", contravariant=True, default=object)
TCommand_contra = TypeVar("TCommand_contra", contravariant=True, default=object)
TConfigKey_contra = TypeVar("TConfigKey_contra", contravariant=True, default=object)
TEvent_contra = TypeVar("TEvent_contra", contravariant=True, default=object)
TInput_contra = TypeVar("TInput_contra", contravariant=True, default=object)
TItem_contra = TypeVar("TItem_contra", contravariant=True, default=object)
TQuery_contra = TypeVar("TQuery_contra", contravariant=True, default=object)
TResult_contra = TypeVar("TResult_contra", contravariant=True, default=object)
TUtil_contra = TypeVar("TUtil_contra", contravariant=True, default=object)
T_Validator_contra = TypeVar("T_Validator_contra", contravariant=True, default=object)
T_Repository_contra = TypeVar("T_Repository_contra", contravariant=True, default=object)
TInput_Handler_contra = TypeVar(
    "TInput_Handler_contra", contravariant=True, default=object
)
MessageT_contra = TypeVar("MessageT_contra", contravariant=True, default=object)

# Invariant advanced TypeVars
T_ResultProtocol = TypeVar("T_ResultProtocol", default=object)
CallableInputT = TypeVar("CallableInputT", default=object)
CallableOutputT = TypeVar("CallableOutputT", default=object)

# Domain-specific TypeVars
Command = TypeVar("Command", default=object)
Event = TypeVar("Event", default=object)
Message = TypeVar("Message", default=object)
Query = TypeVar("Query", default=object)
ResultT = TypeVar("ResultT", default=object)

# Module-level TypeVars (for compatibility)
MessageT = TypeVar("MessageT", default=object)
FactoryT = TypeVar("FactoryT", default=object)
TValidateAll = TypeVar("TValidateAll", default=object)

# TypeVars for generic processor operations
ProcessedDataT = TypeVar("ProcessedDataT", default=object)
ProcessorResultT = TypeVar("ProcessorResultT", default=object)
RegistryHandlerT = TypeVar("RegistryHandlerT", default=object)


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

    # Use these ONLY when you need the union type - otherwise use dict/list
    # Using object for dict/list values to avoid Pydantic recursion issues
    # while maintaining type compatibility with nested JSON structures
    type JsonValue = dict[str, object] | list[object] | str | int | float | bool | None
    type JsonDict = dict[str, JsonValue]

    # =========================================================================
    # FLEXT RESULT TYPES - Railway pattern (domain-specific)
    # =========================================================================
    # Note: Using TYPE_CHECKING to avoid runtime circular imports

    if TYPE_CHECKING:
        type FlextResultType[T] = FlextResult[T]
        type ValidationRule[T] = Callable[[T], FlextResult[bool]]
        type CrossFieldValidator[T] = Callable[[T], FlextResult[bool]]
        type ValidatorFunctionType = Callable[[object], FlextResult[bool]]

    # =========================================================================
    # HANDLER TYPES - CQRS/Bus patterns (domain-specific)
    # =========================================================================

    type HandlerCallable[T, M] = Callable[[M], T]
    type CallableHandlerType = Callable[[object], object]
    type BusHandlerType = Callable[[object], object]
    type MiddlewareType = Callable[[object], object]
    type MiddlewareConfig = dict[str, object]

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

    type GenericDetailsType = dict[str, object] | object
    type SortableObjectType = dict[str, object] | object
    type CachedObjectType = object
    type SerializableType = (
        object | dict[str, object] | list[object] | str | int | float | bool | None
    )
    type SerializableObjectType = (
        object | dict[str, object] | list[object] | str | int | float | bool | None
    )
    # Accepts plain types, generic aliases (e.g., dict[str, object]), and string references
    type TypeOriginSpecifier = object
    type ParameterValueType = object
    type IntList = list[int]
    type FloatList = list[float]
    type BoolList = list[bool]
    type ObjectList = list[object]
    type FloatDict = dict[str, float]
    type NestedDict = dict[str, dict[str, object]]
    type GenericTypeArgument = type | str | object
    type TypeHintSpecifier = type | str | UnionType | None
    type ValidatableInputType = object
    type ContextualObjectType = object
    type ContainerServiceType = object

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
    # Note: Validator types defined above with FlextResult types in TYPE_CHECKING block

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
    "UUID1",
    "UUID3",
    "UUID4",
    "UUID5",
    # Pydantic native types (re-exported for convenience)
    "AnyUrl",
    "AwareDatetime",
    # TypeVars (domain-specific for FlextCore architecture)
    "CallableInputT",
    "CallableOutputT",
    "Command",
    "DirectoryPath",
    "E",
    "EmailStr",
    "Event",
    "F",
    "FactoryT",
    "FilePath",
    "FileUrl",
    "FlextTypes",
    "FutureDate",
    # Domain validation types (Phase 3 - Annotated constraints)
    "HostName",
    "HttpUrl",
    "K",
    "LogLevel",
    "Message",
    "MessageT",
    "MessageT_contra",
    "NaiveDatetime",
    "NegativeFloat",
    "NegativeInt",
    "NewPath",
    "NonEmptyStr",
    "NonNegativeFloat",
    "NonNegativeInt",
    "NonPositiveFloat",
    "NonPositiveInt",
    "P",
    "PastDate",
    "PortNumber",
    "PositiveFloat",
    "PositiveInt",
    "ProcessedDataT",
    "ProcessorResultT",
    "Query",
    "R",
    "RegistryHandlerT",
    "ResultT",
    "RetryCount",
    "SecretStr",
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
    "confloat",
    "conint",
    "conlist",
    "conset",
    "constr",
]
