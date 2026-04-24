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

    type JsonPayloadLeaf = t.Scalar | Path | tp.JsonValue | mp.BaseModel
    type JsonPayloadCollectionValue = (
        JsonPayloadLeaf | Mapping[str, JsonPayloadLeaf] | Sequence[JsonPayloadLeaf]
    )
    type JsonPayload = (
        JsonPayloadLeaf
        | Mapping[str, JsonPayloadCollectionValue]
        | Sequence[JsonPayloadCollectionValue]
    )
    type RegistryDict[T] = MutableMapping[str, T]
    type DomainModelCarrier = mp.BaseModel | p.Model
    type ScalarOrModel = t.Scalar | mp.BaseModel
    type ModelClass[T: mp.BaseModel] = type[T]
    type LogArgument = tp.JsonValue | p.Model
    type LogValue = LogArgument | Exception
    type LogResult = prt.Result[bool]
    type MetadataMapping = Mapping[str, JsonPayload]
    type MutableMetadataMapping = MutableMapping[str, JsonPayload]
    type RuntimeData = tp.JsonValue | mp.BaseModel
    type BootstrapInput = mp.BaseModel | t.JsonMapping
    type SettingsInput = t.SettingsValue | mp.BaseModel
    type ServiceValue = (
        JsonPayload
        | mp.BaseModel
        | pl.Logger
        | ps.Settings
        | pcx.Context
        | ph.Dispatcher
    )
    type UserOverridesMapping = Mapping[str, JsonPayload]
    type RegisterableService = (
        ServiceValue
        | Callable[
            ...,
            ServiceValue,
        ]
    )
    type FactoryCallable = Callable[[], RegisterableService]
    type ResourceCallable = Callable[[], RegisterableService]
    type ModelInput = tp.JsonValue | prt.HasModelDump | Mapping[str, JsonPayload]
    type ConfigModelInput = prt.HasModelDump | Mapping[str, JsonPayload]
    type MetadataInput = prt.HasModelDump | Mapping[str, tp.JsonValue] | None
    type ServiceMap = Mapping[str, RegisterableService]
    type FactoryMap = Mapping[str, FactoryCallable]
    type ResourceMap = Mapping[str, ResourceCallable]
    type ContextHookCallable = Callable[[t.Scalar], JsonPayload]
    type ContextHookMap = Mapping[str, Sequence[ContextHookCallable]]

    type HandlerCallable = Callable[
        ...,
        mp.BaseModel | prt.ResultLike[ScalarOrModel],
    ]
    type DispatchableHandler = (
        mp.BaseModel
        | ph.DispatchMessage
        | ph.Handle
        | ph.Execute
        | ph.AutoDiscoverableHandler
        | Callable[
            ...,
            mp.BaseModel | JsonPayload | prt.ResultLike[JsonPayload],
        ]
    )
    type ResolvedHandlerCallable = Callable[
        ...,
        mp.BaseModel | JsonPayload | prt.ResultLike[JsonPayload],
    ]
    type RoutedHandlerCallable = Callable[
        [p.Routable],
        JsonPayload | prt.ResultLike[JsonPayload],
    ]
    type RegistrablePlugin = ScalarOrModel | Callable[..., ScalarOrModel]
    type LoggerFactory = Callable[..., pl.OutputLogger] | None
    type LoggerWrapperFactory = Callable[[], type[pl.Logger]]

    type SortableObjectType = str | int | float
    type ValueAdapter[T] = mp.TypeAdapter[T]
    type MessageTypeSpecifier = type | str | UnionType | GenericAlias | TypeAliasType
    type IncEx = AbstractSet[str] | Mapping[str, AbstractSet[str] | bool]

    type ConfigurationMapping = Mapping[str, t.Scalar]
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
    type LazyScalar = t.Scalar | bytes | date | time
    type LazyCollection = Mapping[str, LazyScalar] | Sequence[LazyScalar]
    type ModuleExportValue = tp.JsonValue | bytes | date | time
    type ModuleExport = (
        ModuleExportValue
        | LazyCollection
        | ModuleType
        | type[BaseException | Enum]
        | Callable[..., ModuleExportValue | LazyCollection]
    )
    type LazyGetattr = Callable[[str], ModuleExport]
    type LazyDir = Callable[[], Sequence[str]]

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
        | Mapping[str, JsonPayload]
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
        | prt.ResultLike[JsonPayload]
        | ps.Settings
        | RegisterableService
        | Sequence[JsonPayload]
        | tuple[tp.JsonValue, ...]
        | tuple[type, ...]
        | type
        | TypeAliasType
        | tzinfo
        | UnionType
    )
