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
from typing import Annotated, Literal, ParamSpec, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from structlog.typing import BindableLogger

from flext_core._models.containers import FlextModelsContainers

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
E = TypeVar("E")
U = TypeVar("U")
R = TypeVar("R")
DictValueT = TypeVar("DictValueT")
P = ParamSpec("P")
MessageT_contra = TypeVar("MessageT_contra", contravariant=True)
ResultT = TypeVar("ResultT")
T_Model = TypeVar("T_Model", bound=BaseModel)
T_Namespace = TypeVar("T_Namespace")
T_Settings = TypeVar("T_Settings", bound=BaseSettings)
TModel = TypeVar("TModel", bound=BaseModel)
R2 = TypeVar("R2")


class FlextTypes:
    """Type system foundation for FLEXT ecosystem."""

    type _ContainerValue = (
        str
        | int
        | float
        | bool
        | datetime
        | None
        | BaseModel
        | Path
        | Sequence[FlextTypes._ContainerValue]
        | Mapping[str, FlextTypes._ContainerValue]
    )
    type _ScalarML = str | int | float | bool | datetime | None

    type JsonPrimitive = str | int | float | bool | None
    type JsonValue = (
        JsonPrimitive | Sequence[_ContainerValue] | Mapping[str, _ContainerValue]
    )
    type JsonDict = Mapping[str, JsonValue]
    type StrictJsonScalar = str | int | float | bool | datetime | None
    type StrictJsonValue = (
        StrictJsonScalar
        | list[FlextTypes.StrictJsonValue]
        | Mapping[str, FlextTypes.StrictJsonValue]
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

    type MetadataScalarValue = str | int | float | bool | None
    type MetadataListValue = list[str | int | float | bool | None]
    type PydanticConfigValue = (
        str | int | float | bool | None | list[str | int | float | bool | None]
    )
    type GeneralScalarValue = str | int | float | bool | datetime | None
    type GeneralListValue = list[str | int | float | bool | datetime | None]
    type ExceptionKwargsType = str | int | float | bool | datetime | None

    type TYPE_CHECKING = bool
    type GuardInputValue = _ContainerValue
    type ConfigMapValue = _ContainerValue
    type HandlerCallable = Callable[[ScalarValue], ScalarValue]

    type ConfigMap = FlextModelsContainers.ConfigMap
    type Dict = FlextModelsContainers.Dict
    type ServiceMap = FlextModelsContainers.ServiceMap
    type ErrorMap = FlextModelsContainers.ErrorMap
    type ObjectList = FlextModelsContainers.ObjectList
    type FactoryMap = FlextModelsContainers.FactoryMap
    type ResourceMap = FlextModelsContainers.ResourceMap
    type ValidatorCallable = FlextModelsContainers.ValidatorCallable
    type FieldValidatorMap = FlextModelsContainers.FieldValidatorMap
    type ConsistencyRuleMap = FlextModelsContainers.ConsistencyRuleMap
    type EventValidatorMap = FlextModelsContainers.EventValidatorMap
    type ConfigurationMapping = Mapping[str, _ContainerValue]
    type ConfigurationDict = Mapping[str, _ContainerValue]
    type BatchResultDict = FlextModelsContainers.BatchResultDict

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
    type HandlerLike = Callable[..., _ContainerValue | None]

    type IncEx = set[str] | Mapping[str, set[str] | bool]

    class Validation:
        """Validation type aliases with Pydantic constraints."""

        type PortNumber = Annotated[int, Field(ge=1, le=65535)]
        type PositiveTimeout = Annotated[float, Field(gt=0.0, le=300.0)]
        type RetryCount = Annotated[int, Field(ge=0, le=10)]
        type WorkerCount = Annotated[int, Field(ge=1, le=100)]
        type NonEmptyStr = Annotated[str, Field(min_length=1)]
        type StrippedStr = Annotated[str, Field(min_length=1)]
        type UriString = Annotated[str, Field(min_length=1)]
        type HostnameStr = Annotated[str, Field(min_length=1)]
        type PositiveInt = Annotated[int, Field(gt=0)]
        type NonNegativeInt = Annotated[int, Field(ge=0)]
        type BoundedStr = Annotated[str, Field(min_length=1, max_length=255)]
        type TimestampStr = Annotated[str, Field(min_length=1)]


__all__ = [
    "FlextTypes",
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
]
