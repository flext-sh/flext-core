"""FlextTypesServices - service, mapping, and runtime helper type aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
    MutableMapping,
    Sequence,
    Set as AbstractSet,
)
from datetime import date, time, tzinfo
from enum import Enum
from pathlib import Path
from types import GenericAlias, ModuleType, UnionType
from typing import TypeAliasType

from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._protocols.base import FlextProtocolsBase as p
from flext_core._protocols.container import FlextProtocolsContainer as pc
from flext_core._protocols.context import FlextProtocolsContext as pcx
from flext_core._protocols.handler import FlextProtocolsHandler as ph
from flext_core._protocols.logging import FlextProtocolsLogging as pl
from flext_core._protocols.registry import FlextProtocolsRegistry as pr
from flext_core._protocols.result import FlextProtocolsResult as prt
from flext_core._protocols.settings import FlextProtocolsSettings as ps
from flext_core._typings.base import FlextTypingBase as t
from flext_core._typings.pydantic import FlextTypesPydantic as tp


class FlextTypesServices:
    """Type aliases for service registration and runtime mappings."""

    JsonPayload = tp.JsonValue | mp.BaseModel
    type RegistryDict[T] = MutableMapping[str, T]
    DomainModelCarrier = mp.BaseModel | p.Model
    StructuredValue = tp.JsonValue
    ScalarOrModel = t.Scalar | mp.BaseModel
    type ModelClass[T: mp.BaseModel] = type[T]
    LogArgument = tp.JsonValue | p.Model
    LogValue = LogArgument | Exception
    LogResult = prt.Result[bool]
    MetadataData = StructuredValue
    RuntimeData = StructuredValue | mp.BaseModel
    PresentRuntimeData = RuntimeData
    BootstrapInput = mp.BaseModel | t.JsonMapping
    RegisterableService = (
        tp.JsonValue
        | mp.BaseModel
        | pl.Logger
        | Callable[
            ...,
            tp.JsonValue | mp.BaseModel | pl.Logger,
        ]
    )
    FactoryCallable = Callable[[], RegisterableService]
    ResourceCallable = Callable[[], RegisterableService]
    ModelInput = tp.JsonValue | mp.BaseModel | Mapping[str, RuntimeData]
    type ConfigModelInput = mp.BaseModel | Mapping[str, RuntimeData]
    MetadataInput = mp.BaseModel | Mapping[str, tp.JsonValue]
    type ServiceMap = Mapping[str, RegisterableService]
    type FactoryMap = Mapping[str, FactoryCallable]
    type ResourceMap = Mapping[str, ResourceCallable]
    type ContextHookCallable = Callable[[t.Scalar], RuntimeData]
    type ContextHookMap = Mapping[str, Sequence[ContextHookCallable]]

    type HandlerCallable = Callable[
        ...,
        mp.BaseModel | prt.ResultLike[ScalarOrModel],
    ]
    type HandlerLike = Callable[
        ...,
        mp.BaseModel | prt.ResultLike[ScalarOrModel],
    ]
    type DispatchableHandler = (
        mp.BaseModel
        | Callable[
            ...,
            mp.BaseModel | RuntimeData | prt.ResultLike[RuntimeData],
        ]
    )
    type HandlerProtocolVariant = DispatchableHandler
    type ResolvedHandlerCallable = Callable[
        ...,
        mp.BaseModel | RuntimeData | prt.ResultLike[RuntimeData],
    ]
    type RoutedHandlerCallable = Callable[
        [p.Routable],
        RuntimeData | prt.ResultLike[RuntimeData],
    ]
    type RegisteredHandler = tuple[
        HandlerProtocolVariant,
        ResolvedHandlerCallable,
    ]
    type AutoHandlerRegistration = tuple[
        HandlerProtocolVariant,
        ResolvedHandlerCallable,
        tuple[FlextTypesServices.MessageTypeSpecifier, ...],
    ]
    type RegistrablePlugin = ScalarOrModel | Callable[..., ScalarOrModel]
    type LoggerFactory = Callable[..., pl.OutputLogger] | None
    type LoggerWrapperFactory = Callable[[], type[pl.Logger]]

    type SortableObjectType = str | int | float
    type ValueAdapter[T] = mp.TypeAdapter[T]
    type TypeOriginSpecifier = t.TypeHintSpecifier
    type GenericTypeArgument = str | type[t.Scalar]
    type MessageTypeSpecifier = type | str | UnionType | GenericAlias | TypeAliasType
    type IncEx = AbstractSet[str] | Mapping[str, AbstractSet[str] | bool]

    type LazyImportIndex = Mapping[str, str | t.StrSequence]
    type ConfigurationMapping = Mapping[str, t.Scalar]
    ResultErrorData = t.JsonMapping
    type MutableConfigurationMapping = MutableMapping[str, t.Scalar]
    type ScopedContainerRegistry = MutableMapping[
        str,
        t.MutableJsonMapping,
    ]
    type ScopedScalarRegistry = MutableMapping[
        str,
        MutableMapping[str, t.Scalar],
    ]
    type SettingsClass = type[ps.Settings]
    type RuntimeModule = ModuleType
    type LazyScalar = t.Scalar | bytes | date | time
    type LazyCollection = Mapping[str, LazyScalar] | Sequence[LazyScalar]
    ModuleExportValue = tp.JsonValue | bytes | date | time
    type ModuleExport = (
        ModuleExportValue
        | LazyCollection
        | RuntimeModule
        | type[BaseException | Enum]
        | Callable[..., ModuleExportValue | LazyCollection]
    )
    type LazyGetattr = Callable[[str], ModuleExport]
    type LazyDir = Callable[[], Sequence[str]]
    type LazyNamespaceValue = ModuleExport | t.StrSequence | LazyGetattr | LazyDir

    type ValidatorCallable = Callable[[ScalarOrModel], ScalarOrModel]

    type MapperCallable = Callable[
        [tp.JsonValue],
        tp.JsonValue,
    ]
    MapperInput = MapperCallable | tp.JsonValue
    StrictValue = (
        t.Scalar
        | ConfigurationMapping
        | t.JsonList
        | tuple[tp.JsonValue | t.Scalar, ...]
    )
    type PaginationMeta = Mapping[str, int | bool]

    type GuardInput = (
        type[BaseException | Enum]
        | AbstractSet[t.Scalar]
        | JsonPayload
        | bytearray
        | bytes
        | Callable[..., tp.JsonValue | mp.BaseModel]
        | Callable[[], RegisterableService]
        | Path
        | t.Scalar
        | t.JsonMapping
        | Enum
        | frozenset[str]
        | GenericAlias
        | Mapping[str, RuntimeData]
        | tp.JsonValue
        | ModuleType
        | mp.BaseModel
        | pc.Container
        | pcx.Context
        | ph.Dispatcher
        | ph.Handle
        | ph.Middleware
        | pl.HasLogger
        | pl.Logger
        | pr.Registry
        | p.Model
        | prt.ResultLike[RuntimeData]
        | ps.Settings
        | RegisterableService
        | Sequence[RuntimeData]
        | tuple[tp.JsonValue, ...]
        | tuple[type, ...]
        | type
        | TypeAliasType
        | tzinfo
        | UnionType
    )

    UserOverridesMapping = Mapping[
        str,
        t.Scalar | t.JsonMapping | t.ScalarList,
    ]
