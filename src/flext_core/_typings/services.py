"""FlextTypesServices - service, mapping, and runtime helper type aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, Set as AbstractSet
from datetime import date, time, tzinfo
from enum import Enum
from pathlib import Path
from types import GenericAlias, ModuleType, UnionType
from typing import TypeAliasType

from flext_core._protocols.base import FlextProtocolsBase as p
from flext_core._protocols.context import FlextProtocolsContext as pcx
from flext_core._protocols.handler import FlextProtocolsHandler as ph
from flext_core._protocols.logging import FlextProtocolsLogging as pl
from flext_core._protocols.result import FlextProtocolsResult as prt
from flext_core._protocols.settings import FlextProtocolsSettings as ps
from flext_core._typings.base import FlextTypingBase as t
from flext_core._typings.containers import FlextTypingContainers as tc
from flext_core._typings.pydantic import FlextTypesPydantic as tp


class FlextTypesServices(tc):
    """Type aliases for service registration and runtime mappings."""

    type SettingsOverrideLeaf = tc.JsonPayloadLeaf | p.BaseModel
    type SettingsOverrideCollectionValue = (
        SettingsOverrideLeaf
        | t.MappingKV[str, SettingsOverrideLeaf]
        | t.SequenceOf[SettingsOverrideLeaf]
    )
    type SettingsOverride = (
        SettingsOverrideLeaf
        | t.MappingKV[str, SettingsOverrideCollectionValue]
        | t.SequenceOf[SettingsOverrideCollectionValue]
    )
    type SettingsOverridesMapping = t.MappingKV[str, SettingsOverride | None]
    type RegistryDict[T] = MutableMapping[str, T]
    type DomainModelCarrier = tp.BaseModelType | p.BaseModel
    type ModelClass[T: tp.BaseModelType] = type[T]
    type LogArgument = tc.JsonPayload | p.BaseModel
    type LogValue = LogArgument | Exception
    type LogResult = prt.Result[bool]
    type MetadataMapping = t.MappingKV[str, tc.JsonPayload]
    type MutableMetadataMapping = MutableMapping[str, tc.JsonPayload]
    type RuntimeData = tp.JsonValue | tp.BaseModelType
    type BootstrapInput = tp.BaseModelType | t.JsonMapping
    type ServiceValue = (
        tc.JsonPayload
        | tp.BaseModelType
        | pl.Logger
        | ps.Settings
        | pcx.Context
        | ph.Dispatcher
    )
    type RegisterableService = ServiceValue | Callable[..., ServiceValue]
    type FactoryCallable = Callable[[], RegisterableService]
    type ResourceCallable = Callable[[], RegisterableService]
    type ModelInput = tp.JsonValue | prt.HasModelDump | t.MappingKV[str, tc.JsonPayload]
    type ConfigModelInput = prt.HasModelDump | t.MappingKV[str, tc.JsonPayload]
    type MetadataInput = (
        prt.HasModelDump | t.MappingKV[str, tc.JsonPayload | None] | None
    )
    type ServiceMap = t.MappingKV[str, RegisterableService]
    type FactoryMap = t.MappingKV[str, FactoryCallable]
    type ResourceMap = t.MappingKV[str, ResourceCallable]
    type ContextHookCallable = Callable[[t.Scalar], tc.JsonPayload]
    type ContextHookMap = t.MappingKV[str, t.SequenceOf[ContextHookCallable]]

    type HandlerCallable = Callable[
        ..., tp.BaseModelType | prt.ResultLike[tc.ScalarOrModel]
    ]
    type DispatchableHandler = (
        tp.BaseModelType
        | ph.DispatchMessage
        | ph.Handle
        | ph.Execute
        | ph.AutoDiscoverableHandler
        | Callable[
            ...,
            tp.BaseModelType | tc.JsonPayload | prt.ResultLike[tc.JsonPayload] | None,
        ]
    )
    type ResolvedHandlerCallable = Callable[
        ..., tp.BaseModelType | tc.JsonPayload | prt.ResultLike[tc.JsonPayload] | None
    ]
    type RoutedHandlerCallable = Callable[
        [p.Routable], tc.JsonPayload | prt.ResultLike[tc.JsonPayload] | None
    ]
    type LoggerFactory = Callable[..., pl.OutputLogger] | None
    type LoggerWrapperFactory = Callable[[], type[pl.Logger]]

    type SortableObjectType = str | int | float
    type ValueAdapter[T] = tp.TypeAdapterType[T]
    type MessageTypeSpecifier = type | str | UnionType | GenericAlias | TypeAliasType
    type IncEx = (
        AbstractSet[int]
        | AbstractSet[str]
        | t.MappingKV[
            int,
            bool
            | AbstractSet[int]
            | AbstractSet[str]
            | t.MappingKV[int, bool]
            | t.MappingKV[str, bool],
        ]
        | t.MappingKV[
            str,
            bool
            | AbstractSet[int]
            | AbstractSet[str]
            | t.MappingKV[int, bool]
            | t.MappingKV[str, bool],
        ]
    )

    type ConfigurationMapping = t.MappingKV[str, t.Scalar]
    type MutableConfigurationMapping = MutableMapping[str, t.Scalar]
    type ScopedContainerRegistry = MutableMapping[str, t.MutableJsonMapping]
    type SettingsClass = type[ps.SettingsType]
    type LazyScalar = t.Scalar | bytes | date | time
    type LazyCollection = t.MappingKV[str, LazyScalar] | t.SequenceOf[LazyScalar]
    type ModuleExportValue = tp.JsonValue | bytes | date | time
    type ModuleExport = (
        ModuleExportValue
        | LazyCollection
        | ModuleType
        | type[BaseException | Enum]
        | Callable[..., ModuleExportValue | LazyCollection]
    )
    type LazyGetattr = Callable[[str], ModuleExport]
    type LazyDir = Callable[[], t.SequenceOf[str]]

    type ValidatorCallable = Callable[[tc.ScalarOrModel], tc.ScalarOrModel]

    type MapperCallable = Callable[[tp.JsonValue], tp.JsonValue]
    MapperInput = MapperCallable | tp.JsonValue
    StrictValue = (
        t.Scalar
        | ConfigurationMapping
        | t.JsonList
        | tuple[tp.JsonValue | t.Scalar, ...]
    )
    type PaginationMeta = t.MappingKV[str, int | bool]

    type GuardInput = (
        type[BaseException | Enum]
        | AbstractSet[t.Scalar]
        | tc.JsonPayload
        | bytearray
        | bytes
        | Callable[..., tp.JsonValue | tp.BaseModelType]
        | Callable[[], RegisterableService]
        | Path
        | t.Scalar
        | t.JsonMapping
        | Enum
        | frozenset[str]
        | GenericAlias
        | t.MappingKV[str, tc.JsonPayload]
        | tp.JsonValue
        | ModuleType
        | tp.BaseModelType
        | pcx.Context
        | ph.Dispatcher
        | ph.Handle
        | ph.Middleware
        | pl.HasLogger
        | pl.Logger
        | prt.HasModelDump
        | p.BaseModel
        | prt.ResultLike[tc.JsonPayload]
        | ps.Settings
        | RegisterableService
        | t.SequenceOf[tc.JsonPayload]
        | tuple[tp.JsonValue, ...]
        | tuple[type, ...]
        | type
        | TypeAliasType
        | tzinfo
        | UnionType
    )
