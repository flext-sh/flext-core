"""Layer 0: Centralized Type System Foundation for FLEXT Ecosystem.

**ARCHITECTURE LAYER 0** - Pure Constants (Zero Dependencies)

This module provides ONLY types actually used by src/flext_core modules:
- ALL TypeVars at module level (Python 3.13+ native support)
- FlextTypes class with ONLY types used by core 23 modules
- NO nested classes, NO redundant aliases, NO ecosystem-only types
- ~200-250 lines total (~80% reduction from 1,193)

**Core Philosophy**:
✅ KEEP: TypeVars (40+ types)
✅ KEEP: Types used in src/flext_core/ (Dict, List, HandlerRegistry, etc.)
❌ REMOVE: Unused types (no ecosystem types)
❌ REMOVE: Nested classes (FlextTypes.Service, FlextTypes.Reliability, etc.)
❌ REMOVE: Redundant aliases
✅ USE: FlextConstants enums directly (no aliases)

**TypeVar System** (40+ specialized types):
- All at module level: `from flext_core.typings import T, T_co, T_contra`
- **Covariant** (_co): Output types
- **Contravariant** (_contra): Input types
- **Invariant** (no suffix): Strict matching

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    ParamSpec,
    TypeVar,
)

if TYPE_CHECKING:
    from flext_core.result import FlextResult

# =============================================================================
# LAYER 0: TYPEVARS - ALL loose at module level (Python 3.13+)
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
# LAYER 1: FLEXTYPES - ONLY types used by src/flext_core modules
# =============================================================================


class FlextTypes:
    """Types used by src/flext_core modules - LEAN, NO nesting.

    This class contains ONLY types that are actually used in src/flext_core/
    modules. No ecosystem-specific types, no nested namespaces.
    """

    # =========================================================================
    # CORE WEAK TYPES - Foundation (most frequently used)
    # =========================================================================

    type Dict = dict[str, object]
    type List = list[object]
    type StringList = list[str]
    type StringDict = dict[str, str]
    type IntList = list[int]
    type FloatList = list[float]
    type BoolList = list[bool]
    type ObjectList = list[object]
    type NestedDict = dict[str, dict[str, object]]
    type FloatDict = dict[str, float]
    type BoolDict = dict[str, bool]
    type IntDict = dict[str, int]
    type JsonValue = dict[str, object] | list[object] | str | int | float | bool | None
    type JsonDict = dict[str, JsonValue]
    type Dicts = dict[str, dict[str, object]]
    type SerializableType = Dict | List | str | int | float | bool | None
    type OutputFormat = str
    type ConfigDict = dict[str, object]
    type Mapping = dict[str, object]
    type ProcessingStatus = str

    # =========================================================================
    # DOMAIN MODELING TYPES - Used in models.py (23+ uses)
    # =========================================================================

    type PydanticContextType = object
    type ComparableObjectType = object
    type ValidatorInputType = object
    type DomainObjectType = object
    type OptionalDomainObjectType = object | None
    type CallableHandlerType = Callable[..., object]
    type EventPayload = dict[str, object]  # Will be replaced with model in Phase 4.4
    type EventMetadata = dict[str, str | int | float]

    # =========================================================================
    # PROCESSOR TYPES - Used in processors.py (52-67 uses combined)
    # =========================================================================

    type ProcessorInputType = object | dict[str, object] | str | int | float | bool
    type ProcessorOutputType = (
        object | dict[str, object] | str | int | float | bool | None
    )

    # =========================================================================
    # HANDLER TYPES - Used in handlers.py, bus.py (13-14 uses combined)
    # =========================================================================

    type BusHandlerType = Callable[..., object]
    type BusMessageType = object
    type AcceptableMessageType = object | dict[str, object] | str | int | float | bool
    type MiddlewareType = Callable[[object], object]
    type MiddlewareConfig = dict[
        str, object | int | str
    ]  # Will be replaced with model in Phase 4.4

    # =========================================================================
    # LOGGING TYPES - Used in loggings.py (10-21 uses combined)
    # =========================================================================

    type LoggingContextValueType = object
    type LoggingArgType = object
    type LoggingContextType = object
    type LoggingKwargsType = object
    type BoundLoggerType = object
    type LoggingProcessorType = Callable[..., object]

    # =========================================================================
    # RUNTIME TYPES - Used in runtime.py (4-15 uses combined)
    # =========================================================================

    type ValidatableInputType = object
    type TypeHintSpecifier = object
    type SerializableObjectType = object
    type GenericTypeArgument = object
    type LoggerContextType = object
    type FactoryCallableType = Callable[[], object]
    type ContextualObjectType = object

    # =========================================================================
    # CONTAINER TYPES - Used in container.py (1-2 uses)
    # =========================================================================

    type ValidatorFunctionType = Callable[[object], object]
    type ContainerServiceType = object

    # =========================================================================
    # VALIDATION TYPES - Used in models.py (1-7 uses)
    # =========================================================================

    type ConfigValue = object
    type ValidationRule = Callable[[object], FlextResult[object]]
    type CrossFieldValidator = Callable[[object], FlextResult[object]]

    # =========================================================================
    # CONTEXT TYPES - Used in context.py
    # =========================================================================

    type HookRegistry = dict[str, list[Callable[..., object]]]
    type HookCallableType = Callable[..., object]
    type ScopeRegistry = dict[str, dict[str, object]]
    type PredicateType = Callable[[object], bool]

    # =========================================================================
    # DISPATCHER TYPES - Used in dispatcher.py
    # =========================================================================

    type HandlerCallableType = Callable[[object], object | FlextResult[object]]
    type MessageTypeOrHandlerType = str | object | HandlerCallableType
    type HandlerOrCallableType = object | HandlerCallableType

    # =========================================================================
    # DECORATOR TYPES - Used in decorators.py
    # =========================================================================

    type DecoratorReturnType = Callable[[Callable[..., object]], Callable[..., object]]

    # =========================================================================
    # BUS TYPES - Complex handler patterns for bus.py
    # =========================================================================

    type HandlerConfigurationType = Dict | None

    # =========================================================================
    # CORE NESTED CLASS - For external projects (Core types)
    # =========================================================================

    class Core:
        """Core domain types for external project extension.

        Provides foundational types that external projects (FlextMeltanoTypes,
        FlextOracleWmsTypes) can reference when building their own type systems.
        """

        # =====================================================================
        # BASIC COLLECTION TYPES - Foundation for external projects
        # =====================================================================

        # Direct references to FlextTypes collection types
        Dict = dict[str, object]  # Generic dict type
        List = list[object]  # Generic list type
        StringList = list[str]  # String list type
        StringDict = dict[str, str]  # String-keyed string value dict
        IntDict = dict[str, int]  # String-keyed int value dict
        FloatDict = dict[str, float]  # String-keyed float value dict
        BoolDict = dict[str, bool]  # String-keyed bool value dict
        NestedDict = dict[str, dict[str, object]]  # Nested dict type

        # =====================================================================
        # CONFIGURATION AND VALIDATION TYPES - For domain models
        # =====================================================================

        ConfigValue = object  # Generic config value type
        ValidationInput = object  # Generic validation input
        ValidationResult = object  # Generic validation result
        DomainValue = object  # Generic domain value
        EntityId = str  # Standard entity identifier type

        # =====================================================================
        # JSON AND SERIALIZATION TYPES - For data exchange
        # =====================================================================

        JsonValue = (
            dict[str, object] | list[object] | str | int | float | bool | None
        )  # JSON-compatible type
        JsonDict = dict[str, JsonValue]  # Dict with JSON values
        SerializableValue = object  # Any serializable value

        # =====================================================================
        # DATA PROCESSING TYPES - For operations and transformations
        # =====================================================================

        RecordDict = dict[str, object]  # Record dictionary
        FilterDict = dict[str, object]  # Filter configuration
        ResultDict = dict[str, object]  # Result data
        ContextDict = dict[str, object]  # Context data
        EntityDict = dict[str, object]  # Entity representation

        # =====================================================================
        # COLLECTION VARIANTS - Multiple container types
        # =====================================================================

        RecordList = list[dict[str, object]]  # List of records
        EntityList = list[dict[str, object]]  # List of entities
        ResultList = list[dict[str, object]]  # List of results


__all__: list[str] = [
    "CallableInputT",
    "CallableOutputT",
    "Command",
    "E",
    "Event",
    "F",
    "FactoryT",
    "FlextTypes",
    "K",
    "Message",
    "MessageT",
    "MessageT_contra",
    "P",
    "ProcessedDataT",
    "ProcessorResultT",
    "Query",
    "R",
    "RegistryHandlerT",
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
    "U",
    "V",
    "W",
]
