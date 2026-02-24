"""Type aliases and generics for the FLEXT ecosystem.

Centralizes TypeVar declarations and type aliases for CQRS messages, handlers,
utilities, logging, and validation across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from re import Pattern
from typing import (
    Annotated,
    Literal,
    ParamSpec,
    TypeAlias,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field, RootModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from structlog.typing import BindableLogger

from flext_core.constants import FlextConstants

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
    GuardInputValue: TypeAlias = (
        ScalarValue
        | BaseModel
        | Path
        | Sequence[ScalarValue | BaseModel]
        | Mapping[str, ScalarValue | BaseModel]
    )

    # Recursive value type for m.ConfigMap / m.Dict root (no isinstance; use models)
    ConfigMapValue: TypeAlias = (
        ScalarValue
        | BaseModel
        | Path
        | Sequence["ConfigMapValue"]
        | Mapping[str, "ConfigMapValue"]
    )

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
        FlextConstants.Utilities.ConversionMode.TO_STR,
        FlextConstants.Utilities.ConversionMode.TO_STR_LIST,
        FlextConstants.Utilities.ConversionMode.NORMALIZE,
        FlextConstants.Utilities.ConversionMode.JOIN,
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

    class _DictMixin(typing.Generic[DictValueT]):
        """Shared dict-like API for all RootModel containers.

        Provides standard dict interface methods delegating to ``self.root``.
        Subclasses must define ``root: dict[str, V]`` via ``RootModel``.
        """

        root: dict[str, DictValueT]

        def __getitem__(self, key: str) -> DictValueT:
            """Get item by key."""
            return self.root[key]

        def __setitem__(self, key: str, value: DictValueT) -> None:
            """Set item by key."""
            self.root[key] = value

        def __delitem__(self, key: str) -> None:
            """Delete item by key."""
            del self.root[key]

        def __len__(self) -> int:
            """Get length."""
            return len(self.root)

        def __iter__(self) -> typing.Iterator[str]:
            """Iterate over keys (dict semantics, not model fields)."""
            return iter(self.root)

        def __contains__(self, key: str) -> bool:
            """Check if key exists."""
            return key in self.root

        def get(self, key: str, default: DictValueT | None = None) -> DictValueT | None:
            """Get item with default."""
            return self.root.get(key, default)

        def items(self) -> typing.ItemsView[str, DictValueT]:
            """Get items view."""
            return self.root.items()

        def keys(self) -> typing.KeysView[str]:
            """Get keys view."""
            return self.root.keys()

        def values(self) -> typing.ValuesView[DictValueT]:
            """Get values view."""
            return self.root.values()

        def update(self, other: Mapping[str, DictValueT]) -> None:
            """Update with other mapping."""
            self.root.update(other)

        def clear(self) -> None:
            """Clear all items."""
            self.root.clear()

        def pop(self, key: str, default: DictValueT | None = None) -> DictValueT | None:
            """Pop item by key."""
            return self.root.pop(key, default)

        def popitem(self) -> tuple[str, DictValueT]:
            """Pop last item."""
            return self.root.popitem()

        def setdefault(self, key: str, default: DictValueT) -> DictValueT:
            """Set default value for key."""
            return self.root.setdefault(key, default)

    class Dict(_DictMixin[_ContainerValue], RootModel[dict[str, _ContainerValue]]):
        """Generic dictionary container. Prefer m.Dict in public API."""

        root: dict[str, _ContainerValue] = Field(default_factory=dict)

    class ConfigMap(
        _DictMixin[_ContainerValue],
        RootModel[dict[str, _ContainerValue]],
    ):
        """Configuration map container. Prefer m.ConfigMap in public API."""

        root: dict[str, _ContainerValue] = Field(default_factory=dict)

    ConfigurationMapping: TypeAlias = ConfigMap
    ConfigurationDict: TypeAlias = ConfigMap

    class ServiceMap(
        _DictMixin[_ContainerValue],
        RootModel[dict[str, _ContainerValue]],
    ):
        """Service registry map container. Prefer m.ServiceMap in public API."""

        root: dict[str, _ContainerValue]

    class ErrorMap(
        _DictMixin[int | str | dict[str, int]],
        RootModel[dict[str, int | str | dict[str, int]]],
    ):
        """Error type mapping container.

        Replaces: ErrorTypeMapping
        """

        root: dict[str, int | str | dict[str, int]]

    IncEx: TypeAlias = set[str] | Mapping[str, set[str] | bool]

    DecoratorType: TypeAlias = Callable[[HandlerCallable], HandlerCallable]

    FactoryCallable: TypeAlias = Callable[[], RegisterableService]
    ResourceCallable: TypeAlias = Callable[[], _ContainerValue]
    FactoryRegistrationCallable: TypeAlias = FactoryRegistrationCallable

    class FactoryMap(
        _DictMixin[FactoryRegistrationCallable],
        RootModel[dict[str, FactoryRegistrationCallable]],
    ):
        """Map of factory registration callables.

        Replaces: Mapping[str, FactoryRegistrationCallable]
        """

        root: dict[str, FactoryRegistrationCallable]

    class ResourceMap(
        _DictMixin[ResourceCallable],
        RootModel[dict[str, ResourceCallable]],
    ):
        """Map of resource callables.

        Replaces: Mapping[str, ResourceCallable]
        """

        root: dict[str, ResourceCallable]

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

    class _ValidatorMapMixin:
        """Shared API for validator map containers."""

        root: dict[
            str,
            Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],
        ]

        def items(
            self,
        ) -> typing.ItemsView[
            str,
            Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],
        ]:
            """Get validator items."""
            return self.root.items()

        def values(
            self,
        ) -> typing.ValuesView[
            Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],
        ]:
            """Get validator values."""
            return self.root.values()

    class FieldValidatorMap(
        _ValidatorMapMixin,
        RootModel[
            dict[
                str,
                Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],
            ]
        ],
    ):
        """Map of field validators."""

        root: dict[
            str,
            Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],
        ]

    class ConsistencyRuleMap(
        _ValidatorMapMixin,
        RootModel[
            dict[
                str,
                Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],
            ]
        ],
    ):
        """Map of consistency rules."""

        root: dict[
            str,
            Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],
        ]

    class EventValidatorMap(
        _ValidatorMapMixin,
        RootModel[
            dict[
                str,
                Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],
            ]
        ],
    ):
        """Map of event validators."""

        root: dict[
            str,
            Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel],
        ]

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

        model_config = ConfigDict(
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


t_core = FlextTypes
t = FlextTypes

__all__ = [
    "FlextTypes",
    "JsonDict",
    "JsonPrimitive",
    "JsonValue",
    "MessageT_contra",
    "P",
    "R",
    "ResultT",
    "ScalarAlias",
    "T",
    "T_Model",
    "T_Namespace",
    "T_Settings",
    "T_co",
    "T_contra",
    "U",
    "t",
    "t_core",
]
