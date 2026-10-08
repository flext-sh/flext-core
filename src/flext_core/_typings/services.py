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

from flext_core._protocols import (
    FlextProtocolsBase,
    FlextProtocolsContainer,
    FlextProtocolsContext,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsRegistry,
    FlextProtocolsResult,
    FlextProtocolsSettings,
)
from flext_core._typings.base import FlextTypingBase
from flext_core._typings.pydantic import FlextTypesPydantic


class FlextTypesServices:
    """Type aliases for service registration and runtime mappings."""

    type JsonPayloadLeaf = (
        FlextTypingBase.Scalar
        | Path
        | FlextTypesPydantic.JsonValue
        | FlextTypesPydantic.BaseModelType
    )
    type JsonPayloadCollectionValue = (
        JsonPayloadLeaf
        | FlextTypingBase.MappingKV[str, JsonPayloadLeaf]
        | FlextTypingBase.SequenceOf[JsonPayloadLeaf]
    )
    type JsonPayload = (
        JsonPayloadLeaf
        | FlextTypingBase.MappingKV[str, JsonPayloadCollectionValue]
        | FlextTypingBase.SequenceOf[JsonPayloadCollectionValue]
    )
    type SettingsOverrideLeaf = JsonPayloadLeaf | FlextProtocolsBase.Model
    type SettingsOverrideCollectionValue = (
        SettingsOverrideLeaf
        | FlextTypingBase.MappingKV[str, SettingsOverrideLeaf]
        | FlextTypingBase.SequenceOf[SettingsOverrideLeaf]
    )
    type SettingsOverride = (
        SettingsOverrideLeaf
        | FlextTypingBase.MappingKV[str, SettingsOverrideCollectionValue]
        | FlextTypingBase.SequenceOf[SettingsOverrideCollectionValue]
    )
    type SettingsOverridesMapping = FlextTypingBase.MappingKV[
        str,
        SettingsOverride | None,
    ]
    type RegistryDict[T] = MutableMapping[str, T]
    type DomainModelCarrier = (
        FlextTypesPydantic.BaseModelType | FlextProtocolsBase.Model
    )
    type ScalarOrModel = FlextTypingBase.Scalar | FlextTypesPydantic.BaseModelType
    type ModelClass[T: FlextTypesPydantic.BaseModelType] = type[T]
    type LogArgument = JsonPayload | FlextProtocolsBase.Model
    type LogValue = LogArgument | Exception
    type LogResult = FlextProtocolsResult.Result[bool]
    type MetadataMapping = FlextTypingBase.MappingKV[str, JsonPayload]
    type MutableMetadataMapping = MutableMapping[str, JsonPayload]
    type RuntimeData = FlextTypesPydantic.JsonValue | FlextTypesPydantic.BaseModelType
    type BootstrapInput = FlextTypesPydantic.BaseModelType | FlextTypingBase.JsonMapping
    type ServiceClass = type[object]
    type ServiceValue = (
        JsonPayload
        | FlextTypesPydantic.BaseModelType
        | FlextProtocolsLogging.Logger
        | FlextProtocolsSettings.Settings
        | FlextProtocolsContext.Context
        | FlextProtocolsHandler.Dispatcher
    )
    type UserOverridesMapping = FlextTypingBase.MappingKV[str, JsonPayload]
    # Keep registerable-service shape to a single top-level union in this layer.
    # This avoids ``no_inline_union`` violations while preserving the contract
    # that services may be values, factories, or class references.
    type RegisterableServiceValue = ServiceValue | Callable[..., ServiceValue]
    type RegisterableService = RegisterableServiceValue | ServiceClass
    type FactoryCallable = Callable[[], RegisterableService]
    type ResourceCallable = Callable[[], RegisterableService]
    type ModelInput = (
        FlextTypesPydantic.JsonValue
        | FlextProtocolsResult.HasModelDump
        | FlextTypingBase.MappingKV[str, JsonPayload]
    )
    type ConfigModelInput = (
        FlextProtocolsResult.HasModelDump | FlextTypingBase.MappingKV[str, JsonPayload]
    )
    type MetadataInput = (
        FlextProtocolsResult.HasModelDump
        | FlextTypingBase.MappingKV[str, JsonPayload | None]
        | None
    )
    type ServiceMap = FlextTypingBase.MappingKV[str, RegisterableService]
    type FactoryMap = FlextTypingBase.MappingKV[str, FactoryCallable]
    type ResourceMap = FlextTypingBase.MappingKV[str, ResourceCallable]
    type ContextHookCallable = Callable[[FlextTypingBase.Scalar], JsonPayload]
    type ContextHookMap = FlextTypingBase.MappingKV[
        str,
        FlextTypingBase.SequenceOf[ContextHookCallable],
    ]

    type HandlerCallable = Callable[
        ...,
        FlextTypesPydantic.BaseModelType
        | FlextProtocolsResult.ResultView[ScalarOrModel],
    ]
    type DispatchableHandler = (
        FlextTypesPydantic.BaseModelType
        | FlextProtocolsHandler.DispatchMessage
        | FlextProtocolsHandler.Handle
        | FlextProtocolsHandler.Execute
        | FlextProtocolsHandler.AutoDiscoverableHandler
        | Callable[
            [FlextProtocolsBase.Routable],
            FlextTypesPydantic.BaseModelType
            | JsonPayload
            | FlextProtocolsResult.ResultView[JsonPayload]
            | None,
        ]
    )
    type ResolvedHandlerCallable = Callable[
        ...,
        FlextTypesPydantic.BaseModelType
        | JsonPayload
        | FlextProtocolsResult.ResultView[JsonPayload]
        | None,
    ]
    type RoutedHandlerCallable = Callable[
        [FlextProtocolsBase.Routable],
        JsonPayload | FlextProtocolsResult.ResultView[JsonPayload] | None,
    ]
    type RegistrablePlugin = ScalarOrModel | Callable[..., ScalarOrModel]
    type LoggerFactory = Callable[..., FlextProtocolsLogging.OutputLogger] | None
    type LoggerWrapperFactory = Callable[[], type[FlextProtocolsLogging.Logger]]
    type LoggerProcessor = Callable[..., FlextTypesPydantic.JsonValue]

    type SortableObjectType = str | int | float
    type ValueAdapter[T] = FlextTypesPydantic.TypeAdapter[T]
    type MessageTypeSpecifier = type | str | UnionType | GenericAlias | TypeAliasType
    type IncEx = (
        AbstractSet[str] | FlextTypingBase.MappingKV[str, AbstractSet[str] | bool]
    )

    type ConfigurationMapping = FlextTypingBase.MappingKV[str, FlextTypingBase.Scalar]
    type MutableConfigurationMapping = MutableMapping[str, FlextTypingBase.Scalar]
    type ScopedContainerRegistry = MutableMapping[
        str,
        FlextTypingBase.MutableJsonMapping,
    ]
    type SettingsClass = type[FlextProtocolsSettings.SettingsType]
    type LazyScalar = FlextTypingBase.Scalar | bytes | date | time
    type LazyCollection = (
        FlextTypingBase.MappingKV[str, LazyScalar]
        | FlextTypingBase.SequenceOf[LazyScalar]
    )
    type ModuleExportValue = FlextTypesPydantic.JsonValue | bytes | date | time
    type ModuleExport = (
        ModuleExportValue
        | LazyCollection
        | ModuleType
        | type[BaseException | Enum]
        | Callable[..., ModuleExportValue | LazyCollection]
    )
    type LazyGetattr = Callable[[str], ModuleExport]
    type LazyDir = Callable[[], FlextTypingBase.SequenceOf[str]]

    type ValidatorCallable = Callable[[ScalarOrModel], ScalarOrModel]

    type MapperCallable = Callable[
        [FlextTypesPydantic.JsonValue],
        FlextTypesPydantic.JsonValue,
    ]
    MapperInput = MapperCallable | FlextTypesPydantic.JsonValue
    StrictValue = (
        FlextTypingBase.Scalar
        | ConfigurationMapping
        | FlextTypingBase.JsonList
        | tuple[FlextTypesPydantic.JsonValue | FlextTypingBase.Scalar, ...]
    )
    type PaginationMeta = FlextTypingBase.MappingKV[str, int | bool]

    type GuardInput = (
        type[BaseException | Enum]
        | AbstractSet[FlextTypingBase.Scalar]
        | JsonPayload
        | bytearray
        | bytes
        | Callable[..., FlextTypesPydantic.JsonValue | FlextTypesPydantic.BaseModelType]
        | Callable[[], RegisterableService]
        | Path
        | FlextTypingBase.Scalar
        | FlextTypingBase.JsonMapping
        | Enum
        | frozenset[str]
        | GenericAlias
        | FlextTypingBase.MappingKV[str, JsonPayload]
        | FlextTypesPydantic.JsonValue
        | ModuleType
        | FlextTypesPydantic.BaseModelType
        | FlextProtocolsContainer.Container
        | FlextProtocolsContext.Context
        | FlextProtocolsHandler.Dispatcher
        | FlextProtocolsHandler.Handle
        | FlextProtocolsHandler.Middleware
        | FlextProtocolsLogging.HasLogger
        | FlextProtocolsLogging.Logger
        | FlextProtocolsResult.HasModelDump
        | FlextProtocolsRegistry.Registry
        | FlextProtocolsBase.Model
        | FlextProtocolsResult.Result[JsonPayload]
        | FlextProtocolsSettings.Settings
        | RegisterableService
        | FlextTypingBase.SequenceOf[JsonPayload]
        | tuple[FlextTypesPydantic.JsonValue, ...]
        | FlextTypingBase.VariadicTuple[type]
        | type
        | TypeAliasType
        | tzinfo
        | UnionType
    )
