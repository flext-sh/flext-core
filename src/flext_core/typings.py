"""Type aliases and generics for the FLEXT ecosystem.

Centralizes TypeVar declarations and type aliases for CQRS messages, handlers,
utilities, logging, and validation across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
from collections.abc import Callable, Mapping, MutableMapping, Sequence
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
# Module-Level Recursive Types (PEP 695 required for Pydantic recursive compat)
# ============================================================================

type GeneralValueType = (
    str
    | int
    | float
    | bool
    | datetime
    | None
    | BaseModel
    | Path
    | Sequence[GeneralValueType]
    | Mapping[str, GeneralValueType]
)

type JsonPrimitive = str | int | float | bool | None

type JsonValue = (
    JsonPrimitive | Sequence[GeneralValueType] | Mapping[str, GeneralValueType]
)

# ============================================================================
# Core Callables & Instance Types (Module Level for scoping)
# ============================================================================
type ServiceInstanceType = GeneralValueType
type FactoryCallable = Callable[[], GeneralValueType]
type ResourceCallable = Callable[[], GeneralValueType]

type FactoryRegistrationCallable = Callable[
    [], FlextTypes.ScalarValue | Sequence[FlextTypes.ScalarValue]
]


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
    # ALIGNED TYPE HIERARCHY (Pydantic-safe, no 'object' types)
    # =========================================================================

    # Tier 1: Scalar primitives (immutable, JSON-safe)
    ScalarValue: TypeAlias = str | int | float | bool | datetime | None

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

    # =========================================================================
    # GeneralValueType - Re-exported from module level (recursive)
    # =========================================================================
    GeneralValueType: TypeAlias = GeneralValueType

    # RegisterableService - Type for services registerable in FlextContainer
    RegisterableService: TypeAlias = GeneralValueType | object

    # RegistrablePlugin - Type for plugins registerable in FlextRegistry
    RegistrablePlugin: TypeAlias = GeneralValueType | Callable[..., GeneralValueType]

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

    class ObjectList(RootModel[list[GeneralValueType]]):
        """Sequence of general value types for batch operations.

        Replaces: Sequence[GeneralValueType]
        """

        root: list[GeneralValueType]

    # File content type supporting all serializable formats
    FileContent: TypeAlias = str | bytes | BaseModel | Sequence[Sequence[str]]

    # Sortable object type
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

    # MetadataAttributeValue - ALIGNED with GeneralValueType primitive types
    MetadataAttributeValue: TypeAlias = (
        str
        | int
        | float
        | bool
        | datetime
        | None
        | list[str | int | float | bool | datetime | None]
    )

    # Flexible value type for protocol methods
    FlexibleValue: TypeAlias = (
        str
        | int
        | float
        | bool
        | datetime
        | None
        | Sequence[str | int | float | bool | datetime | None]
    )

    # Type hint specifier - used for type introspection
    TypeHintSpecifier: TypeAlias = (
        type | str | Callable[[GeneralValueType], GeneralValueType]
    )

    # Generic type argument
    GenericTypeArgument: TypeAlias = str | type[GeneralValueType]

    # Message type specifier
    MessageTypeSpecifier: TypeAlias = str | type

    # Type origin specifier
    TypeOriginSpecifier: TypeAlias = (
        str | type[GeneralValueType] | Callable[[GeneralValueType], GeneralValueType]
    )

    # JSON types - re-exported from module level
    JsonPrimitive: TypeAlias = JsonPrimitive
    JsonValue: TypeAlias = JsonValue

    # Single consolidated callable type for handlers and validators
    HandlerCallable: TypeAlias = Callable[[GeneralValueType], GeneralValueType]

    # Acceptable message types for handlers
    AcceptableMessageType: TypeAlias = (
        GeneralValueType | Sequence[GeneralValueType] | ScalarValue
    )

    # Conditional execution callable types
    ConditionCallable: TypeAlias = Callable[[GeneralValueType], bool]

    # Handler type union
    HandlerType: TypeAlias = HandlerCallable | BaseModel

    # =========================================================================
    # Configuration mapping types (REMOVED - Use m.ConfigMap)
    # =========================================================================
    # ConfigurationMapping: TypeAlias = Mapping[str, GeneralValueType]
    # ConfigurationDict: TypeAlias = dict[str, GeneralValueType]
    # PydanticConfigDict: TypeAlias = dict[str, GeneralValueType]

    # =========================================================================
    # GENERIC CONTAINERS (RootModels - Pydantic-validated dictionaries)
    # Replaces raw dict aliases with strict typed models
    # =========================================================================

    class Dict(
        RootModel[dict[str, GeneralValueType]], MutableMapping[str, GeneralValueType]
    ):
        """Generic dictionary container.

        Replaces: dict[str, Any], dict[str, GeneralValueType]
        """

        root: dict[str, GeneralValueType]

        def __getitem__(self, key: str) -> GeneralValueType:
            """Get item by key."""
            return self.root[key]

        def __setitem__(self, key: str, value: GeneralValueType) -> None:
            """Set item by key."""
            self.root[key] = value

        def __delitem__(self, key: str) -> None:
            """Delete item by key."""
            del self.root[key]

        def __len__(self) -> int:
            """Get dictionary length."""
            return len(self.root)

        def get(self, key: str, default: GeneralValueType = None) -> GeneralValueType:
            """Get item with default."""
            return self.root.get(key, default)

        def items(self) -> typing.ItemsView[str, GeneralValueType]:
            """Get dictionary items."""
            return self.root.items()

        def keys(self) -> typing.KeysView[str]:
            """Get dictionary keys."""
            return self.root.keys()

        def values(self) -> typing.ValuesView[GeneralValueType]:
            """Get dictionary values."""
            return self.root.values()

    class ConfigMap(
        RootModel[dict[str, GeneralValueType]], MutableMapping[str, GeneralValueType]
    ):
        """Configuration map container.

        Replaces: ConfigurationDict, ConfigurationMapping
        """

        root: dict[str, GeneralValueType]

        def __getitem__(self, key: str) -> GeneralValueType:
            """Get configuration item by key."""
            return self.root[key]

        def __len__(self) -> int:
            """Get configuration length."""
            return len(self.root)

        def get(self, key: str, default: GeneralValueType = None) -> GeneralValueType:
            """Get configuration item with default."""
            return self.root.get(key, default)

        def items(self) -> typing.ItemsView[str, GeneralValueType]:
            """Get configuration items."""
            return self.root.items()

        def keys(self) -> typing.KeysView[str]:
            """Get configuration keys."""
            return self.root.keys()

        def values(self) -> typing.ValuesView[GeneralValueType]:
            """Get configuration values."""
            return self.root.values()

        def __setitem__(self, key: str, value: GeneralValueType) -> None:
            """Set configuration item by key."""
            self.root[key] = value

        def __delitem__(self, key: str) -> None:
            """Delete configuration item by key."""
            del self.root[key]

        def __contains__(self, key: object) -> bool:
            """Check if key exists in configuration."""
            return key in self.root

        def update(self, other: Mapping[str, GeneralValueType]) -> None:
            """Update configuration with other mapping."""
            self.root.update(other)

        def clear(self) -> None:
            """Clear all configuration items."""
            self.root.clear()

        def pop(self, key: str, default: typing.Any = ...) -> typing.Any:
            """Pop item from configuration."""
            if default is ...:
                return self.root.pop(key)
            return self.root.pop(key, default)

        def popitem(self) -> tuple[str, GeneralValueType]:
            """Pop item from configuration."""
            return self.root.popitem()

        def setdefault(
            self, key: str, default: GeneralValueType = None
        ) -> GeneralValueType:
            """Set default value for key."""
            return self.root.setdefault(key, default)

    class ServiceMap(
        RootModel[dict[str, GeneralValueType]], MutableMapping[str, GeneralValueType]
    ):
        """Service registry map container.

        Replaces: ServiceMapping
        """

        root: dict[str, GeneralValueType]

        def __getitem__(self, key: str) -> GeneralValueType:
            """Get service by key."""
            return self.root[key]

        def __len__(self) -> int:
            """Get service map length."""
            return len(self.root)

        def get(self, key: str, default: GeneralValueType = None) -> GeneralValueType:
            """Get service with default."""
            return self.root.get(key, default)

        def items(self) -> typing.ItemsView[str, GeneralValueType]:
            """Get service items."""
            return self.root.items()

        def keys(self) -> typing.KeysView[str]:
            """Get service keys."""
            return self.root.keys()

        def values(self) -> typing.ValuesView[GeneralValueType]:
            """Get service values."""
            return self.root.values()

        def __setitem__(self, key: str, value: GeneralValueType) -> None:
            """Set service by key."""
            self.root[key] = value

        def __delitem__(self, key: str) -> None:
            """Delete service by key."""
            del self.root[key]

        def __contains__(self, key: object) -> bool:
            """Check if key exists in service map."""
            return key in self.root

        def update(self, other: Mapping[str, GeneralValueType]) -> None:
            """Update service map with other mapping."""
            self.root.update(other)

        def clear(self) -> None:
            """Clear all service items."""
            self.root.clear()

        def pop(self, key: str, default: typing.Any = ...) -> typing.Any:
            """Pop item from service map."""
            if default is ...:
                return self.root.pop(key)
            return self.root.pop(key, default)

        def popitem(self) -> tuple[str, GeneralValueType]:
            """Pop item from service map."""
            return self.root.popitem()

        def setdefault(
            self, key: str, default: GeneralValueType = None
        ) -> GeneralValueType:
            """Set default value for key."""
            return self.root.setdefault(key, default)

    class ErrorMap(
        RootModel[dict[str, int | str | dict[str, int]]],
        MutableMapping[str, int | str | dict[str, int]],
    ):
        """Error type mapping container.

        Replaces: ErrorTypeMapping
        """

        root: dict[str, int | str | dict[str, int]]

        def __getitem__(self, key: str) -> int | str | dict[str, int]:
            """Get error item by key."""
            return self.root[key]

        def __len__(self) -> int:
            """Get error map length."""
            return len(self.root)

        def get(
            self, key: str, default: int | str | dict[str, int] | None = None
        ) -> int | str | dict[str, int] | None:
            """Get error item with default."""
            return self.root.get(key, default)

        def items(self) -> typing.ItemsView[str, int | str | dict[str, int]]:
            """Get dictionary items."""
            return self.root.items()

        def keys(self) -> typing.KeysView[str]:
            """Get dictionary keys."""
            return self.root.keys()

        def values(self) -> typing.ValuesView[int | str | dict[str, int]]:
            """Get dictionary values."""
            return self.root.values()

        def __setitem__(self, key: str, value: int | str | dict[str, int]) -> None:
            """Set item by key."""
            self.root[key] = value

        def __delitem__(self, key: str) -> None:
            """Delete item by key."""
            del self.root[key]

        def __contains__(self, key: object) -> bool:
            """Check if key exists."""
            return key in self.root

        def update(self, other: Mapping[str, int | str | dict[str, int]]) -> None:
            """Update with other mapping."""
            self.root.update(other)

        def clear(self) -> None:
            """Clear all items."""
            self.root.clear()

        def pop(self, key: str, default: typing.Any = ...) -> typing.Any:
            """Pop item."""
            if default is ...:
                return self.root.pop(key)
            return self.root.pop(key, default)

        def popitem(self) -> tuple[str, int | str | dict[str, int]]:
            """Pop item."""
            return self.root.popitem()

        def setdefault(
            self, key: str, default: int | str | dict[str, int] | None = None
        ) -> int | str | dict[str, int] | None:
            """Set default."""
            return self.root.setdefault(key, default)

    # Aliases for backward compatibility in Models
    ConfigurationMapping: TypeAlias = ConfigMap
    ConfigurationDict: TypeAlias = ConfigMap
    PydanticConfigDict: TypeAlias = ConfigMap

    class Dispatcher:
        """Dispatcher configuration types namespace."""

    IncEx: TypeAlias = set[str] | dict[str, set[str] | bool]

    DecoratorType: TypeAlias = Callable[[HandlerCallable], HandlerCallable]

    FactoryCallable: TypeAlias = FactoryCallable
    ResourceCallable: TypeAlias = ResourceCallable
    FactoryRegistrationCallable: TypeAlias = FactoryRegistrationCallable

    class FactoryMap(RootModel[dict[str, FactoryRegistrationCallable]]):
        """Map of factory registration callables.

        Replaces: Mapping[str, FactoryRegistrationCallable]
        """

        root: dict[str, FactoryRegistrationCallable]

        def __getitem__(self, key: str) -> FactoryRegistrationCallable:
            """Get factory item by key."""
            return self.root[key]

        def __iter__(self) -> typing.Iterator[str]:  # type: ignore[override]
            """Iterate over factory keys."""
            return iter(self.root)

        def __len__(self) -> int:
            """Get factory map length."""
            return len(self.root)

        def get(
            self, key: str, default: FactoryRegistrationCallable | None = None
        ) -> FactoryRegistrationCallable | None:
            """Get factory item with default."""
            return self.root.get(key, default)

        def items(self) -> typing.ItemsView[str, FactoryRegistrationCallable]:
            """Get factory items."""
            return self.root.items()

        def keys(self) -> typing.KeysView[str]:
            """Get factory keys."""
            return self.root.keys()

        def values(self) -> typing.ValuesView[FactoryRegistrationCallable]:
            """Get factory values."""
            return self.root.values()

    class ResourceMap(RootModel[dict[str, ResourceCallable]]):
        """Map of resource callables.

        Replaces: Mapping[str, ResourceCallable]
        """

        root: dict[str, ResourceCallable]

        def __getitem__(self, key: str) -> ResourceCallable:
            """Get resource item by key."""
            return self.root[key]

        def __iter__(self) -> typing.Iterator[str]:  # type: ignore[override]
            """Iterate over resource keys."""
            return iter(self.root)

        def __len__(self) -> int:
            """Get resource map length."""
            return len(self.root)

        def get(
            self, key: str, default: ResourceCallable | None = None
        ) -> ResourceCallable | None:
            """Get resource item with default."""
            return self.root.get(key, default)

        def items(self) -> typing.ItemsView[str, ResourceCallable]:
            """Get resource items."""
            return self.root.items()

        def keys(self) -> typing.KeysView[str]:
            """Get resource keys."""
            return self.root.keys()

        def values(self) -> typing.ValuesView[ResourceCallable]:
            """Get resource values."""
            return self.root.values()

    FactoryMapping: TypeAlias = FactoryMap
    ResourceMapping: TypeAlias = ResourceMap

    # =========================================================================
    # Validation mapping types (used in _models/validation.py)
    # =========================================================================
    class ValidatorCallable(RootModel[Callable[[GeneralValueType], typing.Any]]):
        """Callable validator container."""

        root: Callable[[GeneralValueType], typing.Any]

        def __call__(self, value: GeneralValueType) -> typing.Any:
            """Execute validator."""
            return self.root(value)

    class FieldValidatorMap(
        RootModel[dict[str, Callable[[GeneralValueType], typing.Any]]]
    ):
        """Map of field validators."""

        root: dict[str, Callable[[GeneralValueType], typing.Any]]

        def items(
            self,
        ) -> typing.ItemsView[str, Callable[[GeneralValueType], typing.Any]]:
            return self.root.items()

        def values(self) -> typing.ValuesView[Callable[[GeneralValueType], typing.Any]]:
            return self.root.values()

    class ConsistencyRuleMap(
        RootModel[dict[str, Callable[[GeneralValueType], typing.Any]]]
    ):
        """Map of consistency rules."""

        root: dict[str, Callable[[GeneralValueType], typing.Any]]

        def items(
            self,
        ) -> typing.ItemsView[str, Callable[[GeneralValueType], typing.Any]]:
            return self.root.items()

        def values(self) -> typing.ValuesView[Callable[[GeneralValueType], typing.Any]]:
            return self.root.values()

    class EventValidatorMap(
        RootModel[dict[str, Callable[[GeneralValueType], typing.Any]]]
    ):
        """Map of event validators."""

        root: dict[str, Callable[[GeneralValueType], typing.Any]]

        def items(
            self,
        ) -> typing.ItemsView[str, Callable[[GeneralValueType], typing.Any]]:
            return self.root.items()

        def values(self) -> typing.ValuesView[Callable[[GeneralValueType], typing.Any]]:
            return self.root.values()

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

        results: list[GeneralValueType] = Field(default_factory=list)
        errors: list[tuple[int, str]] = Field(default_factory=list)
        total: int = Field(default=0)
        success_count: int = Field(default=0)
        error_count: int = Field(default=0)

        def __getitem__(
            self,
            key: str,
        ) -> list[GeneralValueType] | list[tuple[int, str]] | int:
            """Provide mapping-style compatibility for legacy batch result access."""
            if key == "results":
                return self.results
            if key == "errors":
                return self.errors
            if key == "total":
                return self.total
            if key == "success_count":
                return self.success_count
            if key == "error_count":
                return self.error_count
            msg = f"Invalid key: {key}"
            raise KeyError(msg)

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
    "GeneralValueType",
    "JsonPrimitive",
    "JsonValue",
    "MessageT_contra",
    "P",
    "R",
    "ResultT",
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
