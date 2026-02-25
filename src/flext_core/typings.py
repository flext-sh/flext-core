"""Type aliases and generics for the FLEXT ecosystem.

Centralizes TypeVar declarations and type aliases for CQRS messages, handlers,
utilities, logging, and validation across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
from collections.abc import Callable, ItemsView, KeysView, Mapping, Sequence, ValuesView
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from re import Pattern
from typing import (
    Annotated,
    ClassVar,
    Literal,
    ParamSpec,
    TypeAlias,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field, RootModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from structlog.typing import BindableLogger

# LaxStr compatibility for external integrations (LDAP, etc.)

# ============================================================================
# Module-Level TypeVars - Used in flext-core src/
# ============================================================================
# All TypeVars are defined at module level for clarity and accessibility.
# Only TypeVars actually used in flext-core src/ are declared here.
# Note: ParamSpec cannot be used in type aliases within nested classes.

# ============================================================================
# Core Generic TypeVars
# ============================================================================
# Fundamental type variables used throughout flext-core.
T = TypeVar("T")
"""Generic type variable - most commonly used type parameter.
Used in: decorators, container, mixins, pagination, protocols."""
T_co = TypeVar("T_co", covariant=True)
"""Covariant generic type variable - for read-only types.
Used in: protocols, result."""
T_contra = TypeVar("T_contra", contravariant=True)
"""Contravariant generic type variable - for write-only types.
Used in: protocols (for future extensions)."""
E = TypeVar("E")
"""Element type - for collections and sequences.
Used in: collection, enum, args utilities."""
U = TypeVar("U")
"""Utility type - for utility functions and helpers.
Used in: result (for map/flat_map operations)."""
R = TypeVar("R")
"""Return type - for function return values and decorators.
Used in: decorators, args utilities."""
DictValueT = TypeVar("DictValueT")
"""Dictionary value type for RootModel dict-like mixins."""

# ============================================================================
# ParamSpec
# ============================================================================
P = ParamSpec("P")
"""ParamSpec for decorator patterns and variadic function signatures.
Used in: decorators, args utilities."""

# ============================================================================
# Handler TypeVars
# ============================================================================
# Type variables for CQRS handlers and message processing.
MessageT_contra = TypeVar("MessageT_contra", contravariant=True)
"""Contravariant message type - for message objects.
Used in: handlers (CQRS message processing)."""
ResultT = TypeVar("ResultT")
"""Result type - generic result type variable.
Used in: handlers (handler return types)."""

# ============================================================================
# Config/Model TypeVars
# ============================================================================
# Type variables for configuration and model types.
T_Model = TypeVar("T_Model", bound=BaseModel)
"""Model type - for Pydantic model types (bound to BaseModel).
Used in: configuration utilities."""
T_Namespace = TypeVar("T_Namespace")
"""Namespace type - for namespace objects.
Used in: config (namespace configuration)."""
T_Settings = TypeVar("T_Settings", bound=BaseSettings)
"""Settings type - for Pydantic settings types (bound to BaseSettings).
Used in: config (settings configuration)."""


# ============================================================================
# Module-Level Types (PEP 695) - model-centric scalar/payload aliases
# ============================================================================

# Internal only: recursive container value for RootModel Dict/ConfigMap/ServiceMap.
# Not exported. Public API uses m.ConfigMap, m.Dict, ScalarValue.
type _ContainerValue = (
    str
    | int
    | float
    | bool
    | datetime
    | None
    | BaseModel
    | Path
    | Sequence[_ContainerValue]
    | Mapping[str, _ContainerValue]
)

type JsonPrimitive = str | int | float | bool | None

type JsonValue = (
    JsonPrimitive | Sequence[_ContainerValue] | Mapping[str, _ContainerValue]
)

type JsonDict = Mapping[str, JsonValue]

# Scalar at module level for use in callable types (not exported)
type _ScalarML = str | int | float | bool | datetime | None
type ScalarValue = _ScalarML
type PayloadValue = _ContainerValue | BaseModel | Path
type RegisterableService = (
    _ContainerValue | BindableLogger | Callable[..., _ContainerValue]
)

# ============================================================================
# Core Callables - strict DI-compatible service types
# ============================================================================
type ServiceInstanceType = _ContainerValue
type FactoryCallable = Callable[[], RegisterableService]
type ResourceCallable = Callable[[], _ContainerValue]

type FactoryRegistrationCallable = Callable[[], _ScalarML | Sequence[_ScalarML]]


class FlextTypes:
    """Type system foundation for FLEXT ecosystem - Complex Type Aliases Only.

    This class serves as the centralized type system for all FLEXT projects.
    All complex types (type aliases) are organized within this class.
    TypeVars are defined at module level (see above).

    All aliases use ``TypeAlias`` annotation (not PEP 695 ``type`` statement)
    so that basedpyright resolves them as proper class attributes at workspace scope.
    """

    # =====================================================================
    # COMPLEX TYPE ALIASES (TypeAlias for basedpyright compatibility)
    # =====================================================================

    LaxStr: TypeAlias = str | bytes | bytearray
    """LaxStr compatibility for ldap3 integration."""

    # =========================================================================
    # ALIGNED TYPE HIERARCHY (Pydantic-safe, no dynamic top types)
    # =========================================================================

    # Tier 1: Scalar primitives (immutable, JSON-safe)
    ScalarValue: TypeAlias = str | int | float | bool | datetime | None
    ScalarAlias: TypeAlias = ScalarValue

    # Tier 2: Pydantic-safe metadata values
    MetadataScalarValue: TypeAlias = str | int | float | bool | None
    MetadataListValue: TypeAlias = list[str | int | float | bool | None]

    # Tier 2.5: Pydantic-safe config types for Field() annotations
    PydanticConfigValue: TypeAlias = (
        str | int | float | bool | None | list[str | int | float | bool | None]
    )

    # Tier 3: General value types (superset including BaseModel, Path, datetime)
    GeneralScalarValue: TypeAlias = str | int | float | bool | datetime | None
    GeneralListValue: TypeAlias = list[str | int | float | bool | datetime | None]

    # Input type for guard functions (flat union, no recursion; prefer models at API)
    GuardInputValue: TypeAlias = _ContainerValue

    # Recursive value type for m.ConfigMap / m.Dict root (no isinstance; use models)
    ConfigMapValue: TypeAlias = _ContainerValue

    FlexibleValue: TypeAlias = GuardInputValue

    # RegisterableService - fixed types only
    RegisterableService: TypeAlias = RegisterableService

    # RegistrablePlugin - fixed types only
    RegistrablePlugin: TypeAlias = (
        ScalarValue | BaseModel | Callable[..., ScalarValue | BaseModel]
    )

    # Constant value type - all possible constant types in FlextConstants
    ConstantValue: TypeAlias = (
        str
        | int
        | float
        | bool
        | ConfigDict
        | SettingsConfigDict
        | frozenset[str]
        | tuple[str, ...]
        | Mapping[str, str | int]
        | StrEnum
        | type[StrEnum]
        | Pattern[str]
        | type
    )

    class ObjectList(RootModel[list[_ContainerValue]]):
        """Sequence of container values for batch operations. Use m.* models when possible."""

        root: list[_ContainerValue]

    # File content type supporting all serializable formats
    FileContent: TypeAlias = str | bytes | BaseModel | Sequence[Sequence[str]]

    # Sortable value type
    SortableObjectType: TypeAlias = str | int | float

    # =========================================================================
    # Conversion Mode Types
    # =========================================================================
    ConversionMode: TypeAlias = Literal[
        "to_str",
        "to_str_list",
        "normalize",
        "join",
    ]

    # MetadataAttributeValue - fixed scalar/mapping/list (Mapping, no dict)
    MetadataAttributeValue: TypeAlias = (
        str
        | int
        | float
        | bool
        | datetime
        | None
        | Mapping[
            str,
            str
            | int
            | float
            | bool
            | datetime
            | list[str | int | float | bool | datetime | None]
            | None,
        ]
        | list[str | int | float | bool | datetime | None]
    )

    # Type hint specifier - used for type introspection
    TypeHintSpecifier: TypeAlias = type | str | Callable[[ScalarValue], ScalarValue]

    # Generic type argument
    GenericTypeArgument: TypeAlias = str | type[ScalarValue]

    # Message type specifier
    MessageTypeSpecifier: TypeAlias = str | type

    # Type origin specifier
    TypeOriginSpecifier: TypeAlias = (
        str | type[ScalarValue] | Callable[[ScalarValue], ScalarValue]
    )

    # JSON types - re-exported from module level
    JsonPrimitive: TypeAlias = JsonPrimitive
    JsonValue: TypeAlias = JsonValue
    JsonDict: TypeAlias = JsonDict

    # General value type for handlers and config boundaries
    GeneralValueType: TypeAlias = _ContainerValue

    # Single consolidated callable type for handlers and validators
    HandlerCallable: TypeAlias = Callable[[ScalarValue], ScalarValue]

    # Acceptable message types for handlers - scalars and models only
    AcceptableMessageType: TypeAlias = ScalarValue | BaseModel | Sequence[ScalarValue]

    # Conditional execution callable types
    ConditionCallable: TypeAlias = Callable[[ScalarValue], bool]

    # Handler type union
    HandlerType: TypeAlias = HandlerCallable | BaseModel

    # =========================================================================
    # GENERIC CONTAINERS (RootModels - use m.ConfigMap, m.Dict)
    # =========================================================================
    # Replaces raw dict aliases with strict typed models
    # =========================================================================

    @typing.runtime_checkable
    class _RootDictProtocol[RootValueT](typing.Protocol):
        root: dict[str, RootValueT]

    class _RootDictModel[DictValueT](RootModel[dict[str, DictValueT]]):
        root: dict[str, DictValueT] = Field(default_factory=dict)

        def __getitem__(self, key: str) -> DictValueT:
            return self.root[key]

        def __setitem__(self, key: str, value: DictValueT) -> None:
            self.root[key] = value

        def __delitem__(self, key: str) -> None:
            del self.root[key]

        def __len__(self) -> int:
            return len(self.root)

        def __contains__(self, key: str) -> bool:
            return key in self.root

        def get(self, key: str, default: DictValueT | None = None) -> DictValueT | None:
            return self.root.get(key, default)

        def items(self) -> ItemsView[str, DictValueT]:
            return self.root.items()

        def keys(self) -> KeysView[str]:
            return self.root.keys()

        def values(self) -> ValuesView[DictValueT]:
            return self.root.values()

        def update(self, other: Mapping[str, DictValueT]) -> None:
            self.root.update(other)

        def clear(self) -> None:
            self.root.clear()

        def pop(self, key: str, default: DictValueT | None = None) -> DictValueT | None:
            return self.root.pop(key, default)

        def popitem(self) -> tuple[str, DictValueT]:
            return self.root.popitem()

        def setdefault(self, key: str, default: DictValueT) -> DictValueT:
            return self.root.setdefault(key, default)

    class Dict(_RootDictModel[_ContainerValue]):
        """Generic dictionary container. Prefer m.Dict in public API."""

    class ConfigMap(_RootDictModel[_ContainerValue]):
        """Configuration map container. Prefer m.ConfigMap in public API."""

    ConfigurationMapping: TypeAlias = ConfigMap
    ConfigurationDict: TypeAlias = ConfigMap

    class ServiceMap(_RootDictModel[_ContainerValue]):
        """Service registry map container. Prefer m.ServiceMap in public API."""

    class ErrorMap(_RootDictModel[int | str | dict[str, int]]):
        """Error type mapping container.

        Replaces: ErrorTypeMapping
        """

    IncEx: TypeAlias = set[str] | Mapping[str, set[str] | bool]

    DecoratorType: TypeAlias = Callable[[HandlerCallable], HandlerCallable]

    FactoryCallable: TypeAlias = Callable[[], RegisterableService]
    ResourceCallable: TypeAlias = Callable[[], _ContainerValue]
    FactoryRegistrationCallable: TypeAlias = FactoryRegistrationCallable

    class FactoryMap(_RootDictModel[FactoryRegistrationCallable]):
        """Map of factory registration callables.

        Replaces: Mapping[str, FactoryRegistrationCallable]
        """

    class ResourceMap(_RootDictModel[ResourceCallable]):
        """Map of resource callables.

        Replaces: Mapping[str, ResourceCallable]
        """

    # =========================================================================
    # Validation mapping types (used in _models/validation.py)
    # =========================================================================
    class ValidatorCallable(
        RootModel[Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]],
    ):
        """Callable validator container. Fixed types: ScalarValue | BaseModel."""

        root: Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]

        def __call__(self, value: _ScalarML | BaseModel) -> _ScalarML | BaseModel:
            """Execute validator."""
            return self.root(value)

    class _RootValidatorMapModel(
        RootModel[
            dict[
                str,
                Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],
            ]
        ],
    ):
        """Shared API for validator map containers."""

        root: dict[
            str,
            Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],
        ] = Field(default_factory=dict)

        def items(
            self,
        ) -> ItemsView[
            str,
            Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],
        ]:
            """Get validator items."""
            validated: dict[
                str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]
            ] = {key: value for key, value in self.root.items() if callable(value)}
            return validated.items()

        def values(
            self,
        ) -> ValuesView[Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],]:
            """Get validator values."""
            validated: dict[
                str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]
            ] = {key: value for key, value in self.root.items() if callable(value)}
            return validated.values()

    class FieldValidatorMap(_RootValidatorMapModel):
        """Map of field validators."""

    class ConsistencyRuleMap(_RootValidatorMapModel):
        """Map of consistency rules."""

    class EventValidatorMap(_RootValidatorMapModel):
        """Map of event validators."""

    # Error/Exception types (used in exceptions.py)
    # ErrorTypeMapping removed - Use m.ErrorMap
    ExceptionKwargsType: TypeAlias = str | int | float | bool | datetime | None

    # General nested dict type (used in guards.py)
    # GeneralNestedDict removed - Use m.Dict

    # ServiceMapping for service registration
    # ServiceMapping removed - Use m.ServiceMap

    # =====================================================================
    # Pydantic Models
    # =====================================================================
    class BatchResultDict(BaseModel):
        """Result payload model for batch operation outputs."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
            extra="forbid",
        )

        results: list[_ScalarML] = Field(default_factory=list)
        errors: list[tuple[int, str]] = Field(default_factory=list)
        total: int = Field(default=0)
        success_count: int = Field(default=0)
        error_count: int = Field(default=0)

    # =====================================================================
    # VALIDATION TYPES (Annotated with Pydantic constraints)
    # =====================================================================

    class Validation:
        """Validation type aliases with Pydantic constraints."""

        PortNumber: TypeAlias = Annotated[int, Field(ge=1, le=65535)]
        """Port number type alias (1-65535)."""

        PositiveTimeout: TypeAlias = Annotated[float, Field(gt=0.0, le=300.0)]
        """Positive timeout in seconds (0-300)."""

        RetryCount: TypeAlias = Annotated[int, Field(ge=0, le=10)]
        """Retry count (0-10)."""

        WorkerCount: TypeAlias = Annotated[int, Field(ge=1, le=100)]
        """Worker count (1-100)."""

        # =====================================================================
        # EXTENDED VALIDATION TYPES - Reusable Annotated aliases
        # =====================================================================

        NonEmptyStr: TypeAlias = Annotated[str, Field(min_length=1)]
        """Non-empty string (minimum 1 character)."""

        StrippedStr: TypeAlias = Annotated[str, Field(min_length=1)]
        """String with whitespace stripped (minimum 1 character after strip)."""

        UriString: TypeAlias = Annotated[str, Field(min_length=1)]
        """URI string with scheme validation (e.g., http://, https://)."""

        HostnameStr: TypeAlias = Annotated[str, Field(min_length=1)]
        """Hostname string with format validation."""

        PositiveInt: TypeAlias = Annotated[int, Field(gt=0)]
        """Positive integer (> 0)."""

        NonNegativeInt: TypeAlias = Annotated[int, Field(ge=0)]
        """Non-negative integer (>= 0)."""

        BoundedStr: TypeAlias = Annotated[str, Field(min_length=1, max_length=255)]
        """Bounded string (1-255 characters)."""

        TimestampStr: TypeAlias = Annotated[str, Field(min_length=1)]
        """ISO 8601 timestamp string."""


# ============================================================================
# FACTORY FUNCTIONS FOR DYNAMIC ANNOTATED TYPES
# ============================================================================


def bounded_int_factory(min_val: int, max_val: int) -> object:
    """Factory for bounded integer type.

    Args:
        min_val: Minimum value (inclusive)
        max_val: Maximum value (inclusive)

    Returns:
        Annotated type alias for bounded integer

    """
    return Annotated[int, Field(ge=min_val, le=max_val)]


def choice_str_factory(allowed_values: list[str]) -> object:
    """Factory for choice string type.

    Args:
        allowed_values: List of allowed string values

    Returns:
        Annotated type alias for choice string

    """
    return Annotated[str, Field(pattern=f"^({'|'.join(allowed_values)})$")]


def pattern_str_factory(pattern: str) -> object:
    """Factory for regex pattern string type.

    Args:
        pattern: Regular expression pattern

    Returns:
        Annotated type alias for pattern string

    """
    return Annotated[str, Field(pattern=pattern)]


t = FlextTypes

# ============================================================================
# Module-Level Re-exports of Validation Types
# ============================================================================
# These are re-exported from FlextTypes.Validation for convenient access
PortNumber = FlextTypes.Validation.PortNumber
NonEmptyStr = FlextTypes.Validation.NonEmptyStr
StrippedStr = FlextTypes.Validation.StrippedStr
UriString = FlextTypes.Validation.UriString
HostnameStr = FlextTypes.Validation.HostnameStr
PositiveInt = FlextTypes.Validation.PositiveInt
NonNegativeInt = FlextTypes.Validation.NonNegativeInt
BoundedStr = FlextTypes.Validation.BoundedStr
TimestampStr = FlextTypes.Validation.TimestampStr
PositiveTimeout = FlextTypes.Validation.PositiveTimeout
RetryCount = FlextTypes.Validation.RetryCount
WorkerCount = FlextTypes.Validation.WorkerCount

__all__ = [
    "BoundedStr",
    "FlextTypes",
    "HostnameStr",
    "JsonDict",
    "JsonPrimitive",
    "JsonValue",
    "MessageT_contra",
    "NonEmptyStr",
    "NonNegativeInt",
    "P",
    # Validation types from FlextTypes.Validation
    "PortNumber",
    "PositiveInt",
    "PositiveTimeout",
    "R",
    "ResultT",
    "RetryCount",
    "StrippedStr",
    "T",
    "T_Model",
    "T_Namespace",
    "T_Settings",
    "T_co",
    "T_contra",
    "TimestampStr",
    "U",
    "UriString",
    "WorkerCount",
    "bounded_int_factory",
    "choice_str_factory",
    "pattern_str_factory",
    "t",
]
