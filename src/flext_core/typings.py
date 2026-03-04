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
from typing import Annotated, Literal, ParamSpec, TypeAlias, TypeVar

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

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║  ⚠️  AGENT STOP — READ BEFORE EDITING ANY TYPE ALIAS BELOW  ⚠️  ║
    # ║                                                                ║
    # ║  Non-recursive aliases MUST use `X: TypeAlias = ...`           ║
    # ║  DO NOT convert to `type X = ...` (PEP 695).                   ║
    # ║  PEP 695 creates TypeAliasType → isinstance() CRASHES.         ║
    # ║  See CLAUDE.md §3 AXIOMATIC rule. VIOLATION = REJECTION.       ║
    # ║                                                                ║
    # ║  Recursive aliases (Serializable, ContainerValue, JsonValue)   ║
    # ║  MUST use `type` statement — they need forward references.     ║
    # ╚══════════════════════════════════════════════════════════════════╝
    # ── Core type layers ──────────────────────────────────────────────
    Primitives: TypeAlias = str | int | float | bool
    Scalar: TypeAlias = str | int | float | bool | datetime
    Container: TypeAlias = str | int | float | bool | datetime | BaseModel | Path
    type Serializable = (
        Scalar | list[FlextTypes.Serializable] | dict[str, FlextTypes.Serializable]
    )

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
    JsonDict: TypeAlias = Mapping[str, JsonValue]

    # ── Config ────────────────────────────────────────────────────────
    ConfigurationMapping: TypeAlias = Mapping[str, Container]

    # ── Service / DI ──────────────────────────────────────────────────
    RegisterableService: TypeAlias = (
        Container | BindableLogger | Callable[..., Container]
    )
    FactoryCallable: TypeAlias = Callable[[], RegisterableService]
    ResourceCallable: TypeAlias = Callable[[], Container]

    # ── Metadata ──────────────────────────────────────────────────────
    MetadataValue: TypeAlias = (
        Scalar | Mapping[str, Scalar | list[Scalar]] | list[Scalar]
    )
    MetadataAttributeValue: TypeAlias = MetadataValue

    # ── Handlers ──────────────────────────────────────────────────────
    HandlerCallable: TypeAlias = Callable[[Container], Container]
    HandlerLike: TypeAlias = Callable[..., Container]

    # ── Plugin / Constants ────────────────────────────────────────────
    RegistrablePlugin: TypeAlias = (
        Scalar | BaseModel | Callable[..., Scalar | BaseModel]
    )
    ConstantValue: TypeAlias = (
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
    FileContent: TypeAlias = str | bytes | BaseModel | Sequence[Sequence[str]]
    SortableObjectType: TypeAlias = str | int | float
    ConversionMode: TypeAlias = Literal["to_str", "to_str_list", "normalize", "join"]
    TypeHintSpecifier: TypeAlias = type | str | Callable[[Scalar], Scalar]
    TypeOriginSpecifier: TypeAlias = TypeHintSpecifier
    GenericTypeArgument: TypeAlias = str | type[Scalar]
    MessageTypeSpecifier: TypeAlias = str | type
    IncEx: TypeAlias = set[str] | Mapping[str, set[str] | bool]

    TYPE_CHECKING: TypeAlias = bool

    # ── Collection convenience ────────────────────────────────────────
    Dict: TypeAlias = Mapping[str, Container]

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
