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
from typing import Annotated, ClassVar, Literal, ParamSpec, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, RootModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from structlog.typing import BindableLogger

T = TypeVar("T")
"Generic type variable - most commonly used type parameter.\nUsed in: decorators, container, mixins, pagination, protocols."
T_co = TypeVar("T_co", covariant=True)
"Covariant generic type variable - for read-only types.\nUsed in: protocols, result."
T_contra = TypeVar("T_contra", contravariant=True)
"Contravariant generic type variable - for write-only types.\nUsed in: protocols (for future extensions)."
E = TypeVar("E")
"Element type - for collections and sequences.\nUsed in: collection, enum, args utilities."
U = TypeVar("U")
"Utility type - for utility functions and helpers.\nUsed in: result (for map/flat_map operations)."
R = TypeVar("R")
"Return type - for function return values and decorators.\nUsed in: decorators, args utilities."
DictValueT = TypeVar("DictValueT")
"Dictionary value type for RootModel dict-like mixins."
P = ParamSpec("P")
"ParamSpec for decorator patterns and variadic function signatures.\nUsed in: decorators, args utilities."
MessageT_contra = TypeVar("MessageT_contra", contravariant=True)
"Contravariant message type - for message objects.\nUsed in: handlers (CQRS message processing)."
ResultT = TypeVar("ResultT")
"Result type - generic result type variable.\nUsed in: handlers (handler return types)."
T_Model = TypeVar("T_Model", bound=BaseModel)
"Model type - for Pydantic model types (bound to BaseModel).\nUsed in: configuration utilities."
T_Namespace = TypeVar("T_Namespace")
"Namespace type - for namespace objects.\nUsed in: config (namespace configuration)."
T_Settings = TypeVar("T_Settings", bound=BaseSettings)
"Settings type - for Pydantic settings types (bound to BaseSettings).\nUsed in: config (settings configuration)."
TModel = TypeVar("TModel")
R2 = TypeVar("R2")
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
type _ScalarML = str | int | float | bool | datetime | None
type ScalarValue = _ScalarML
type PayloadValue = _ContainerValue | BaseModel | Path
type RegisterableService = (
    _ContainerValue
    | BindableLogger
    | Callable[..., _ContainerValue]
)
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

    type LaxStr = str | bytes | bytearray
    "LaxStr compatibility for ldap3 integration."
    Any: TypeAlias = typing.Any
    TYPE_CHECKING: TypeAlias = bool
    ScalarValue: TypeAlias = _ScalarML
    ScalarAlias: TypeAlias = ScalarValue
    type MetadataScalarValue = str | int | float | bool | None
    type MetadataListValue = list[str | int | float | bool | None]
    type PydanticConfigValue = (
        str | int | float | bool | None | list[str | int | float | bool | None]
    )
    type GeneralScalarValue = str | int | float | bool | datetime | None
    type GeneralListValue = list[str | int | float | bool | datetime | None]
    GuardInputValue: TypeAlias = _ContainerValue
    ConfigMapValue: TypeAlias = _ContainerValue
    FlexibleValue: TypeAlias = _ContainerValue
    PayloadValue: TypeAlias = PayloadValue
    RegisterableService: TypeAlias = RegisterableService
    type RegistrablePlugin = (
        ScalarValue | BaseModel | Callable[..., ScalarValue | BaseModel]
    )
    type ConstantValue = (
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

    type FileContent = str | bytes | BaseModel | Sequence[Sequence[str]]
    type SortableObjectType = str | int | float
    type ConversionMode = Literal["to_str", "to_str_list", "normalize", "join"]
    type MetadataAttributeValue = (
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
    type TypeHintSpecifier = type | str | Callable[[ScalarValue], ScalarValue]
    type GenericTypeArgument = str | type[ScalarValue]
    type MessageTypeSpecifier = str | type
    type TypeOriginSpecifier = (
        str | type[ScalarValue] | Callable[[ScalarValue], ScalarValue]
    )
    JsonPrimitive: TypeAlias = JsonPrimitive
    JsonValue: TypeAlias = JsonValue
    JsonDict: TypeAlias = JsonDict
    GeneralValueType: TypeAlias = _ContainerValue
    HandlerCallable: TypeAlias = Callable[[ScalarValue], ScalarValue]
    type AcceptableMessageType = ScalarValue | BaseModel | Sequence[ScalarValue]
    type ConditionCallable = Callable[[ScalarValue], bool]
    type HandlerType = Callable[..., Any] | object

    @typing.runtime_checkable
    class _RootDictProtocol[RootValueT](typing.Protocol):
        root: dict[str, RootValueT]

    class _RootDictModel[DictValueT](RootModel[dict[str, DictValueT]]):
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

        root: dict[str, _ContainerValue] = Field(default_factory=dict)

    class ConfigMap(_RootDictModel[_ContainerValue]):
        """Configuration map container. Prefer m.ConfigMap in public API."""

        root: dict[str, _ContainerValue] = Field(default_factory=dict)

    type ConfigurationMapping = ConfigMap
    type ConfigurationDict = ConfigMap

    class ServiceMap(_RootDictModel[_ContainerValue]):
        """Service registry map container. Prefer m.ServiceMap in public API."""

        root: dict[str, _ContainerValue] = Field(default_factory=dict)

    class ErrorMap(_RootDictModel[int | str | dict[str, int]]):
        """Error type mapping container.

        Replaces: ErrorTypeMapping
        """

        root: dict[str, int | str | dict[str, int]] = Field(default_factory=dict)

    type IncEx = set[str] | Mapping[str, set[str] | bool]
    type FactoryCallable = Callable[[], RegisterableService]
    type ResourceCallable = Callable[[], _ContainerValue]
    type FactoryRegistrationCallable = Callable[[], _ScalarML | Sequence[_ScalarML]]

    class FactoryMap(_RootDictModel[FactoryRegistrationCallable]):
        """Map of factory registration callables.

        Replaces: Mapping[str, FactoryRegistrationCallable]
        """

        root: dict[str, FactoryRegistrationCallable] = Field(default_factory=dict)

    class ResourceMap(_RootDictModel[ResourceCallable]):
        """Map of resource callables.

        Replaces: Mapping[str, ResourceCallable]
        """

        root: dict[str, ResourceCallable] = Field(default_factory=dict)

    class ValidatorCallable(
        RootModel[Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]]
    ):
        """Callable validator container. Fixed types: ScalarValue | BaseModel."""

        root: Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]

        def __call__(self, value: _ScalarML | BaseModel) -> _ScalarML | BaseModel:
            """Execute validator."""
            return self.root(value)

    class _RootValidatorMapModel(
        RootModel[dict[str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]]]
    ):
        """Shared API for validator map containers."""

        def items(
            self,
        ) -> ItemsView[str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]]:
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

        root: dict[str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]] = (
            Field(default_factory=dict)
        )

    class ConsistencyRuleMap(_RootValidatorMapModel):
        """Map of consistency rules."""

        root: dict[str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]] = (
            Field(default_factory=dict)
        )

    class EventValidatorMap(_RootValidatorMapModel):
        """Map of event validators."""

        root: dict[str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]] = (
            Field(default_factory=dict)
        )

    type ExceptionKwargsType = str | int | float | bool | datetime | None

    class BatchResultDict(BaseModel):
        """Result payload model for batch operation outputs."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True, extra="forbid"
        )
        results: list[_ScalarML] = []
        errors: list[tuple[int, str]] = []
        total: int = Field(default=0)
        success_count: int = Field(default=0)
        error_count: int = Field(default=0)

    class Validation:
        """Validation type aliases with Pydantic constraints."""

        type PortNumber = Annotated[int, Field(ge=1, le=65535)]
        "Port number type alias (1-65535)."
        type PositiveTimeout = Annotated[float, Field(gt=0.0, le=300.0)]
        "Positive timeout in seconds (0-300)."
        type RetryCount = Annotated[int, Field(ge=0, le=10)]
        "Retry count (0-10)."
        type WorkerCount = Annotated[int, Field(ge=1, le=100)]
        "Worker count (1-100)."
        type NonEmptyStr = Annotated[str, Field(min_length=1)]
        "Non-empty string (minimum 1 character)."
        type StrippedStr = Annotated[str, Field(min_length=1)]
        "String with whitespace stripped (minimum 1 character after strip)."
        type UriString = Annotated[str, Field(min_length=1)]
        "URI string with scheme validation (e.g., http://, https://)."
        type HostnameStr = Annotated[str, Field(min_length=1)]
        "Hostname string with format validation."
        type PositiveInt = Annotated[int, Field(gt=0)]
        "Positive integer (> 0)."
        type NonNegativeInt = Annotated[int, Field(ge=0)]
        "Non-negative integer (>= 0)."
        type BoundedStr = Annotated[str, Field(min_length=1, max_length=255)]
        "Bounded string (1-255 characters)."
        type TimestampStr = Annotated[str, Field(min_length=1)]
        "ISO 8601 timestamp string."


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
    "T",
    "T_Model",
    "T_Namespace",
    "T_Settings",
    "T_co",
    "T_contra",
    "U",
    "t",
]
