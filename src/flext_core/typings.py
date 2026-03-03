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
    """Type system foundation for FLEXT ecosystem.

    Canonical types:

        ScalarValue     = str | int | float | bool | datetime
        ContainerValue  = ScalarValue | BaseModel | Path | Sequence | Mapping
        JsonPrimitive   = str | int | float | bool
    """

    # ── Scalar ────────────────────────────────────────────────────────
    type ScalarValue = str | int | float | bool | datetime

    # ── Container (recursive, includes None) ──────────────────────────
    type ContainerValue = (
        FlextTypes.ScalarValue
        | BaseModel
        | Path
        | Sequence[FlextTypes.ContainerValue]
        | Mapping[str, FlextTypes.ContainerValue]
    )

    # ── JSON ──────────────────────────────────────────────────────────
    type JsonPrimitive = FlextTypes.JsonPrimitive
    type JsonValue = (
        JsonPrimitive
        | Sequence[FlextTypes.JsonValue]
        | Mapping[str, FlextTypes.JsonValue]
    )
    type JsonDict = Mapping[str, JsonValue]
    type StrictJsonValue = (
        ScalarValue
        | list[FlextTypes.StrictJsonValue]
        | Mapping[str, FlextTypes.StrictJsonValue]
    )

    # ── Service / DI ──────────────────────────────────────────────────
    type RegisterableService = (
        ContainerValue | BindableLogger | Callable[..., ContainerValue]
    )
    type FactoryCallable = Callable[[], RegisterableService]
    type ResourceCallable = Callable[[], ContainerValue]
    type FactoryRegistrationCallable = Callable[[], ScalarValue | Sequence[ScalarValue]]

    # ── Metadata ──────────────────────────────────────────────────────
    type MetadataAttributeValue = (
        ScalarValue | Mapping[str, ScalarValue | list[ScalarValue]] | list[ScalarValue]
    )

    # ── Configuration ─────────────────────────────────────────────────
    type ConfigurationMapping = Mapping[str, ContainerValue]

    # ── Handlers ──────────────────────────────────────────────────────
    type HandlerCallable = Callable[[ContainerValue], ContainerValue]
    type HandlerLike = Callable[..., ContainerValue | None]

    # ── Plugin / Constants ────────────────────────────────────────────
    type RegistrablePlugin = (
        ScalarValue | BaseModel | Callable[..., ScalarValue | BaseModel]
    )
    type ConstantValue = (
        FlextTypes.JsonPrimitive
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

    # ── File / misc ───────────────────────────────────────────────────
    type FileContent = str | bytes | BaseModel | Sequence[Sequence[str]]
    type SortableObjectType = str | int | float
    type ConversionMode = Literal["to_str", "to_str_list", "normalize", "join"]
    type TypeHintSpecifier = type | str | Callable[[ScalarValue], ScalarValue]
    type GenericTypeArgument = str | type[ScalarValue]
    type MessageTypeSpecifier = str | type
    type TypeOriginSpecifier = (
        str | type[ScalarValue] | Callable[[ScalarValue], ScalarValue]
    )
    type IncEx = set[str] | Mapping[str, set[str] | bool]
    type PydanticConfigValue = ScalarValue | list[ScalarValue]

    type TYPE_CHECKING = bool

    # ── Validation (Pydantic-annotated) ───────────────────────────────
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
    "R2",
    "ResultT",
    "T",
    "T_Model",
    "T_Namespace",
    "T_Settings",
    "T_co",
    "T_contra",
    "U",
]
