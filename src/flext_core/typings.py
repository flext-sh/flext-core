"""Type aliases and generics for the FLEXT ecosystem.

Centralizes TypeVar declarations and type aliases for CQRS messages, handlers,
utilities, logging, and validation across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from re import Pattern
from typing import Annotated, Literal, ParamSpec, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from structlog.typing import BindableLogger

from flext_core._models.containers import FlextModelsContainers

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
TModel = TypeVar("TModel", bound=BaseModel)
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
type _ScalarML = str | int | float | bool | datetime | None
type JsonPrimitive = str | int | float | bool | None
type JsonValue = (
    JsonPrimitive | Sequence[_ContainerValue] | Mapping[str, _ContainerValue]
)
type JsonDict = Mapping[str, JsonValue]
type StrictJsonScalar = str | int | float | bool | datetime | None
type StrictJsonValue = (
    StrictJsonScalar | list[StrictJsonValue] | Mapping[str, StrictJsonValue]
)
type ScalarValue = _ScalarML
type PayloadValue = _ContainerValue | BaseModel | Path
type RegisterableService = (
    _ContainerValue | BindableLogger | Callable[..., _ContainerValue]
)
type ServiceInstanceType = _ContainerValue
type FactoryCallable = Callable[[], RegisterableService]
type ResourceCallable = Callable[[], _ContainerValue]
type FactoryRegistrationCallable = Callable[[], _ScalarML | Sequence[_ScalarML]]
type GeneralValueType = _ContainerValue


class FlextTypes:
    """Type system foundation for FLEXT ecosystem - Type Aliases Only.

    This class serves as the centralized type system for all FLEXT projects.
    All type aliases are organized within this class as ``type`` statements.
    TypeVars are defined at module level (see above).
    Pydantic container models live in ``_models/containers.py`` — use ``m.*``.
    """

    _ContainerValue: TypeAlias = _ContainerValue
    _ScalarML: TypeAlias = _ScalarML
    JsonPrimitive: TypeAlias = JsonPrimitive
    JsonValue: TypeAlias = JsonValue
    JsonDict: TypeAlias = JsonDict
    StrictJsonScalar: TypeAlias = StrictJsonScalar
    StrictJsonValue: TypeAlias = StrictJsonValue

    # =========================================================================
    # SCALAR AND VALUE TYPES
    # =========================================================================

    ScalarValue: TypeAlias = ScalarValue
    PayloadValue: TypeAlias = PayloadValue
    MetadataScalarValue: TypeAlias = str | int | float | bool | None
    MetadataListValue: TypeAlias = list[str | int | float | bool | None]
    PydanticConfigValue: TypeAlias = (
        str | int | float | bool | None | list[str | int | float | bool | None]
    )
    GeneralScalarValue: TypeAlias = str | int | float | bool | datetime | None
    GeneralListValue: TypeAlias = list[str | int | float | bool | datetime | None]
    ExceptionKwargsType: TypeAlias = str | int | float | bool | datetime | None

    # =========================================================================
    # SERVICE AND CALLABLE TYPES
    # =========================================================================

    RegisterableService: TypeAlias = RegisterableService
    ServiceInstanceType: TypeAlias = ServiceInstanceType
    FactoryCallable: TypeAlias = FactoryCallable
    ResourceCallable: TypeAlias = ResourceCallable
    FactoryRegistrationCallable: TypeAlias = FactoryRegistrationCallable

    # =========================================================================
    # TYPE ALIAS ANNOTATIONS (public API surface)
    # =========================================================================

    TYPE_CHECKING: TypeAlias = bool
    GuardInputValue: TypeAlias = _ContainerValue
    ConfigMapValue: TypeAlias = _ContainerValue
    GeneralValueType: TypeAlias = GeneralValueType
    HandlerCallable: TypeAlias = Callable[[ScalarValue], ScalarValue]

    # =========================================================================
    # CONTAINER TYPE CONTRACTS (read-only aliases)
    # Concrete Pydantic models: use m.ConfigMap, m.Dict, etc. from FlextModels
    # =========================================================================

    ConfigMap: TypeAlias = FlextModelsContainers.ConfigMap
    Dict: TypeAlias = FlextModelsContainers.Dict
    ServiceMap: TypeAlias = FlextModelsContainers.ServiceMap
    ErrorMap: TypeAlias = FlextModelsContainers.ErrorMap
    ObjectList: TypeAlias = FlextModelsContainers.ObjectList
    FactoryMap: TypeAlias = FlextModelsContainers.FactoryMap
    ResourceMap: TypeAlias = FlextModelsContainers.ResourceMap
    FieldValidatorMap: TypeAlias = Mapping[
        str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]
    ]
    ConsistencyRuleMap: TypeAlias = Mapping[
        str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]
    ]
    EventValidatorMap: TypeAlias = Mapping[
        str, Callable[[_ScalarML | BaseModel], _ScalarML | BaseModel]
    ]
    ConfigurationMapping: TypeAlias = Mapping[str, _ContainerValue]
    ConfigurationDict: TypeAlias = Mapping[str, _ContainerValue]
    BatchResultDict: TypeAlias = FlextModelsContainers.BatchResultDict

    # =========================================================================
    # PLUGIN AND CONSTANT TYPES
    # =========================================================================

    RegistrablePlugin: TypeAlias = (
        ScalarValue | BaseModel | Callable[..., ScalarValue | BaseModel]
    )
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

    # =========================================================================
    # FILE AND CONVERSION TYPES
    # =========================================================================

    FileContent: TypeAlias = str | bytes | BaseModel | Sequence[Sequence[str]]
    SortableObjectType: TypeAlias = str | int | float
    ConversionMode: TypeAlias = Literal["to_str", "to_str_list", "normalize", "join"]

    # =========================================================================
    # METADATA AND HANDLER TYPES
    # =========================================================================

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
    TypeHintSpecifier: TypeAlias = type | str | Callable[[ScalarValue], ScalarValue]
    GenericTypeArgument: TypeAlias = str | type[ScalarValue]
    MessageTypeSpecifier: TypeAlias = str | type
    TypeOriginSpecifier: TypeAlias = (
        str | type[ScalarValue] | Callable[[ScalarValue], ScalarValue]
    )
    HandlerLike: TypeAlias = Callable[..., _ContainerValue | None]

    # =========================================================================
    # INCLUSION TYPES
    # =========================================================================

    IncEx: TypeAlias = set[str] | Mapping[str, set[str] | bool]

    # =========================================================================
    # VALIDATION TYPES
    # =========================================================================

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
