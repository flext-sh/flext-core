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
from typing import Annotated, Literal, ParamSpec, TypeVar

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from structlog.typing import BindableLogger

# ---------------------------------------------------------------------------
# TypeVars
# ---------------------------------------------------------------------------
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
E = TypeVar("E")
U = TypeVar("U")
R = TypeVar("R")
R2 = TypeVar("R2")
DictValueT = TypeVar("DictValueT")
P = ParamSpec("P")
MessageT_contra = TypeVar("MessageT_contra", contravariant=True)
ResultT = TypeVar("ResultT")
T_Model = TypeVar("T_Model", bound=BaseModel)
T_Namespace = TypeVar("T_Namespace")
T_Settings = TypeVar("T_Settings", bound=BaseSettings)
TModel = TypeVar("TModel", bound=BaseModel)


class FlextTypes:
    """Type system foundation for FLEXT ecosystem.

    Three core layers — each builds on the previous:

        Primitives  ⊂  Scalar  ⊂  Serializable  ⊂  Container

    ``None`` is **never** baked into definitions.
    Use ``X | None`` at call-sites when needed.
    """

    # ── Core type layers ──────────────────────────────────────────────
    type Primitives = str | int | float | bool
    type Scalar = Primitives | datetime
    type Serializable = (
        Scalar | list[FlextTypes.Serializable] | dict[str, FlextTypes.Serializable]
    )
    type Container = Serializable | BaseModel | Path

    type ContainerValue = (
        Container
        | Sequence[FlextTypes.ContainerValue]
        | Mapping[str, FlextTypes.ContainerValue]
        | None
    )

    # ── JSON ──────────────────────────────────────────────────────────
    type JsonValue = (
        Scalar | Sequence[FlextTypes.JsonValue] | Mapping[str, FlextTypes.JsonValue]
    )
    type JsonDict = Mapping[str, JsonValue]

    # ── Config ────────────────────────────────────────────────────────
    type ConfigurationMapping = Mapping[str, Container]

    # ── Service / DI ──────────────────────────────────────────────────
    type RegisterableService = Container | BindableLogger | Callable[..., Container]
    type FactoryCallable = Callable[[], RegisterableService]
    type ResourceCallable = Callable[[], Container]

    # ── Metadata ──────────────────────────────────────────────────────
    type MetadataValue = Scalar | Mapping[str, Scalar | list[Scalar]] | list[Scalar]

    # ── Handlers ──────────────────────────────────────────────────────
    type HandlerCallable = Callable[[Container], Container]
    type HandlerLike = Callable[..., Container]

    # ── Plugin / Constants ────────────────────────────────────────────
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

    # ── File / misc ───────────────────────────────────────────────────
    type FileContent = str | bytes | BaseModel | Sequence[Sequence[str]]
    type SortableObjectType = str | int | float
    type ConversionMode = Literal["to_str", "to_str_list", "normalize", "join"]
    type TypeHintSpecifier = type | str | Callable[[Scalar], Scalar]
    type GenericTypeArgument = str | type[Scalar]
    type MessageTypeSpecifier = str | type
    type IncEx = set[str] | Mapping[str, set[str] | bool]

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
    "R2",
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
