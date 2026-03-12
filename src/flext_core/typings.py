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
    TYPE_CHECKING as _TYPE_CHECKING,
    Annotated,
    Literal,
    ParamSpec,
    TypeVar,
    cast as _cast,
    override as _override,
)

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from structlog.typing import BindableLogger

P = ParamSpec("P")
R = TypeVar("R")
ResultT = TypeVar("ResultT")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
T_Model = TypeVar("T_Model", bound=BaseModel)
T_Settings = TypeVar("T_Settings", bound=BaseSettings)
T_Namespace = TypeVar("T_Namespace")
TRuntime = TypeVar("TRuntime")
U = TypeVar("U")
TV = TypeVar("TV")
TV_co = TypeVar("TV_co", covariant=True)
MessageT_contra = TypeVar("MessageT_contra", contravariant=True)
EnumT = TypeVar("EnumT", bound=StrEnum)
type RegistryBindingKey = str | type


class FlextTypes:
    """Type system foundation for FLEXT ecosystem.

    Strictly tiered layers — Primitives ⊂ Scalar ⊂ Container.
    ``object`` and ``Any`` are strictly forbidden in domain state.
    ``None`` is **never** baked into definitions.
    """

    # --- NON-RECURSIVE TYPES (TypeAlias — isinstance-safe) ---

    type Primitives = str | int | float | bool
    type Scalar = str | int | float | bool | datetime
    type Container = Scalar | Path

    # --- RUNTIME isinstance() TUPLES ---
    # PEP 695 `type` aliases are TypeAliasType and CANNOT be used with isinstance().
    # Use these tuples for ALL runtime isinstance() checks.
    PRIMITIVES_TYPES: tuple[type, ...] = (str, int, float, bool)
    SCALAR_TYPES: tuple[type, ...] = (str, int, float, bool, datetime)
    CONTAINER_TYPES: tuple[type, ...] = (str, int, float, bool, datetime, Path)

    # --- RECURSIVE TYPES (PEP 695 - Annotation-only, NEVER with isinstance) ---

    type JsonValue = (
        Scalar | list[FlextTypes.JsonValue] | dict[str, FlextTypes.JsonValue]
    )
    type Serializable = (
        Scalar | list[FlextTypes.Serializable] | dict[str, FlextTypes.Serializable]
    )
    type ContainerValue = (
        Scalar | list[FlextTypes.ContainerValue] | dict[str, FlextTypes.ContainerValue]
    )
    type GeneralValueType = (
        Scalar
        | Path
        | list[FlextTypes.GeneralValueType]
        | dict[str, FlextTypes.GeneralValueType]
    )

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
    type FileContent = str | bytes | Sequence[Sequence[str]]
    type GeneralValueTypeMapping = Mapping[str, Scalar]

    type RegisterableService = Container | BindableLogger | Callable[..., Container]
    type FactoryCallable = Callable[[], RegisterableService]
    type ResourceCallable = Callable[[], Container]
    type MetadataValue = (
        Scalar | Mapping[str, Scalar | Sequence[Scalar]] | Sequence[Scalar]
    )
    type MetadataAttributeValue = MetadataValue
    type HandlerCallable = Callable[[Container], Container]
    type HandlerLike = Callable[..., Container]
    type RegistrablePlugin = Scalar | Callable[..., Scalar | BaseModel]

    # Other Types
    type SortableObjectType = str | int | float
    type ConversionMode = Literal["to_str", "to_str_list", "normalize", "join"]
    type TypeHintSpecifier = type | str | Callable[[Scalar], Scalar]
    type TypeOriginSpecifier = TypeHintSpecifier
    type GenericTypeArgument = str | type[Scalar]
    type MessageTypeSpecifier = str | type
    type IncEx = set[str] | Mapping[str, set[str] | bool]
    TYPE_CHECKING = _TYPE_CHECKING
    cast = staticmethod(_cast)
    override = staticmethod(_override)

    type ConfigurationMapping = Mapping[str, Scalar]
    type ResultErrorData = BaseModel | Mapping[str, Container]
    type Dict = Mapping[str, Scalar | BaseModel]
    type ConfigMap = Mapping[str, Scalar | BaseModel]
    type ServiceMap = Mapping[str, RegisterableService]
    type ObjectList = Sequence[Container]
    type ModuleExport = Container | ModuleType | type | Callable[..., Container]

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
    "T",
    "T_Model",
    "T_Settings",
    "T_co",
    "T_contra",
    "t",
]
