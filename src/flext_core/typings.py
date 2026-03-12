"""Type aliases and generics for the FLEXT ecosystem.

Zero internal imports — depends only on stdlib, pydantic, pydantic-settings,
and structlog.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from re import Pattern
from types import ModuleType
from typing import (
    Annotated,
    Literal,
    ParamSpec,
    TypeAlias,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from structlog.typing import BindableLogger

DictValueT = TypeVar("DictValueT")
E = TypeVar("E")
EnumT = TypeVar("EnumT", bound=StrEnum)
MessageT_contra = TypeVar("MessageT_contra", contravariant=True)
P = ParamSpec("P")
R = TypeVar("R")
ResultT = TypeVar("ResultT")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
T_Model = TypeVar("T_Model", bound=BaseModel)
T_Namespace = TypeVar("T_Namespace")
T_Settings = TypeVar("T_Settings", bound=BaseSettings)
TK = TypeVar("TK")
TModel = TypeVar("TModel", bound=BaseModel)
TV = TypeVar("TV")
TValue = TypeVar("TValue")
type RegistryBindingKey = str | type
U = TypeVar("U")


class FlextTypes:
    """Type system foundation for FLEXT ecosystem.

    Three core layers — each builds on the previous:

        Primitives  ⊂  Scalar  ⊂  object  ⊂  Container

    ``None`` is **never** baked into definitions.
    Use ``X | None`` at call-sites when needed.
    """

    Primitives: TypeAlias = str | int | float | bool
    Scalar: TypeAlias = str | int | float | bool | datetime
    Container: TypeAlias = str | int | float | bool | datetime | BaseModel | Path
    type RegisterableService = object | BindableLogger | Callable[..., object]
    type FactoryCallable = Callable[[], RegisterableService]
    type ResourceCallable = Callable[[], object]
    type MetadataValue = Scalar | Mapping[str, Scalar | list[Scalar]] | list[Scalar]
    type MetadataAttributeValue = MetadataValue
    type HandlerCallable = Callable[[object], object]
    type HandlerLike = Callable[..., object]
    type RegistrablePlugin = Scalar | BaseModel | Callable[..., Scalar | BaseModel]
    type ConstantValue = (
        Primitives
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
    type TypeHintSpecifier = type | str | Callable[[Scalar], Scalar]
    type TypeOriginSpecifier = TypeHintSpecifier
    type GenericTypeArgument = str | type[Scalar]
    type MessageTypeSpecifier = str | type
    type IncEx = set[str] | Mapping[str, set[str] | bool]
    type TYPE_CHECKING = bool
    type Dict = Mapping[str, object]
    type ModuleExport = type | ModuleType | Callable[..., object] | object

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


t = FlextTypes

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
    "t",
]
