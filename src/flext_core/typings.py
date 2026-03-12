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
    TypeAlias,
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

# --- MODULE LEVEL TYPES (Recursive types must be defined here for reliability) ---

type _Primitives = str | int | float | bool
type _Scalar = str | int | float | bool | datetime
type _Container = _Scalar | BaseModel | Path

# --- RECURSIVE TYPES (PEP 695 - Annotation-only, NEVER with isinstance) ---

type _JsonValue = _Scalar | list[_JsonValue] | dict[str, _JsonValue]
type _Serializable = _Scalar | list[_Serializable] | dict[str, _Serializable]
type _ContainerValue = (
    _Scalar | BaseModel | list[_ContainerValue] | dict[str, _ContainerValue]
)
type _GeneralValueType = (
    _Scalar | BaseModel | Path | list[_GeneralValueType] | dict[str, _GeneralValueType]
)

type _ConstantValue = (
    _Primitives
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
type _FileContent = str | bytes | BaseModel | Sequence[Sequence[str]]
type _GeneralValueTypeMapping = Mapping[str, _GeneralValueType]


class FlextTypes:
    """Type system foundation for FLEXT ecosystem.

    Strictly tiered layers — Primitives ⊂ Scalar ⊂ Container.
    ``object`` and ``Any`` are strictly forbidden in domain state.
    ``None`` is **never** baked into definitions.
    """

    Primitives: TypeAlias = _Primitives
    Scalar: TypeAlias = _Scalar
    Container: TypeAlias = _Container

    RegisterableService: TypeAlias = (
        Container | BindableLogger | Callable[..., Container]
    )
    FactoryCallable: TypeAlias = Callable[[], RegisterableService]
    ResourceCallable: TypeAlias = Callable[[], Container]
    MetadataValue: TypeAlias = (
        Scalar | Mapping[str, Scalar | Sequence[Scalar]] | Sequence[Scalar]
    )
    MetadataAttributeValue: TypeAlias = MetadataValue
    HandlerCallable: TypeAlias = Callable[[Container], Container]
    HandlerLike: TypeAlias = Callable[..., Container]
    RegistrablePlugin: TypeAlias = (
        Scalar | BaseModel | Callable[..., Scalar | BaseModel]
    )

    # RECURSIVE types (Annotation-only, use guards.py for narrowing)
    GeneralValueType: TypeAlias = _GeneralValueType
    Serializable: TypeAlias = _Serializable
    JsonValue: TypeAlias = _JsonValue
    ContainerValue: TypeAlias = _ContainerValue

    # Other Types
    ConstantValue: TypeAlias = _ConstantValue
    FileContent: TypeAlias = _FileContent
    SortableObjectType: TypeAlias = str | int | float
    ConversionMode: TypeAlias = Literal["to_str", "to_str_list", "normalize", "join"]
    TypeHintSpecifier: TypeAlias = type | str | Callable[[_Scalar], _Scalar]
    TypeOriginSpecifier: TypeAlias = TypeHintSpecifier
    GenericTypeArgument: TypeAlias = str | type[_Scalar]
    MessageTypeSpecifier: TypeAlias = str | type
    IncEx: TypeAlias = set[str] | Mapping[str, set[str] | bool]
    TYPE_CHECKING = _TYPE_CHECKING
    cast = staticmethod(_cast)
    override = staticmethod(_override)

    ConfigurationMapping: TypeAlias = Mapping[str, Scalar]
    ResultErrorData: TypeAlias = BaseModel | Mapping[str, Container]
    Dict: TypeAlias = Mapping[str, Scalar | BaseModel]
    ConfigMap: TypeAlias = Mapping[str, Scalar | BaseModel]
    ServiceMap: TypeAlias = Mapping[str, RegisterableService]
    ObjectList: TypeAlias = Sequence[Container]
    GeneralValueTypeMapping: TypeAlias = _GeneralValueTypeMapping
    ModuleExport: TypeAlias = Container | ModuleType | type | Callable[..., Container]

    class Validation:
        """Validation type aliases with Pydantic constraints."""

        PortNumber: TypeAlias = Annotated[int, Field(ge=1, le=65535)]
        PositiveTimeout: TypeAlias = Annotated[float, Field(gt=0.0, le=300.0)]
        RetryCount: TypeAlias = Annotated[int, Field(ge=0, le=10)]
        WorkerCount: TypeAlias = Annotated[int, Field(ge=1, le=100)]
        NonEmptyStr: TypeAlias = Annotated[str, Field(min_length=1)]
        StrippedStr: TypeAlias = Annotated[str, Field(min_length=1)]
        UriString: TypeAlias = Annotated[str, Field(min_length=1)]
        HostnameStr: TypeAlias = Annotated[str, Field(min_length=1)]
        PositiveInt: TypeAlias = Annotated[int, Field(gt=0)]
        NonNegativeInt: TypeAlias = Annotated[int, Field(ge=0)]
        BoundedStr: TypeAlias = Annotated[str, Field(min_length=1, max_length=255)]
        TimestampStr: TypeAlias = Annotated[str, Field(min_length=1)]


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
