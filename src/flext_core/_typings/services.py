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
from datetime import date, time
from enum import Enum
from pathlib import Path
from types import GenericAlias, ModuleType, UnionType
from typing import TypeAliasType

from flext_core import (
    FlextModelsPydantic as mp,
    FlextProtocolsBase as p,
    FlextProtocolsContainer,
    FlextProtocolsContext,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsResult,
    FlextProtocolsService,
    FlextProtocolsSettings,
    FlextTypingBase as t,
    FlextTypingContainers,
)


class FlextTypesServices:
    """Type aliases for service registration and runtime mappings."""

    type RegistryDict[T] = MutableMapping[str, T]
    type ModelCarrier = mp.BaseModel
    type ProtocolModelCarrier = p.Model
    type DomainModelCarrier = ModelCarrier | ProtocolModelCarrier
    type ContainerCarrier = FlextTypingContainers.ConfigMap | FlextTypingContainers.Dict
    type RecursiveMappingCarrier = t.RecursiveContainerMapping
    type RecursiveSequenceCarrier = t.RecursiveContainerList
    type RecursiveTupleCarrier = tuple[t.RecursiveContainer, ...]
    type ModelClass[T: ModelCarrier] = type[T]
    type LogArgument = t.RecursiveContainer | p.Model
    type LogValue = FlextTypesServices.LogArgument | Exception
    type LogResult = FlextProtocolsResult.Result[bool]

    type ScalarOrModel = t.Container | ModelCarrier
    type ValueOrModel = t.RecursiveContainer | ModelCarrier
    type PresentValueOrModel = (
        t.Container
        | RecursiveMappingCarrier
        | RecursiveSequenceCarrier
        | RecursiveTupleCarrier
        | ModelCarrier
    )
    type RuntimeData = ValueOrModel | FlextTypesServices.MetadataValue
    type RuntimeAtomic = (
        ValueOrModel | ContainerCarrier | FlextTypingContainers.ObjectList
    )

    type BootstrapInput = ModelCarrier | RecursiveMappingCarrier | None

    type RegisterableService = (
        t.Container
        | ModelCarrier
        | ContainerCarrier
        | FlextProtocolsLogging.Logger
        | Mapping[
            str,
            t.Container | t.RecursiveContainer,
        ]
        | Sequence[t.Container | t.RecursiveContainer]
        | Callable[
            ...,
            t.Container
            | ModelCarrier
            | ContainerCarrier
            | FlextProtocolsLogging.Logger,
        ]
        | Callable[..., FlextTypesServices.RegisterableService]
    )
    type FactoryCallable = Callable[[], RegisterableService]
    type ResourceCallable = Callable[[], RegisterableService]
    type MetadataValue = (
        t.Scalar
        | Mapping[
            str,
            t.Scalar | Sequence[t.Scalar],
        ]
        | Sequence[t.Scalar]
    )
    type MetadataOrValue = MetadataValue | t.RecursiveContainer
    type MetadataAttributeValue = MetadataValue
    type ModelInput = (
        t.RecursiveContainer
        | ModelCarrier
        | FlextTypingContainers.ConfigMap
        | Mapping[
            str,
            FlextTypesServices.ValueOrModel,
        ]
    )
    type ConfigModelInput = (
        ModelCarrier
        | FlextTypingContainers.ConfigMap
        | Mapping[str, FlextTypesServices.RuntimeAtomic]
    )
    type MetadataInput = (
        ModelCarrier | FlextTypingContainers.ConfigMap | Mapping[str, t.Scalar] | None
    )
    type HandlerCallable = Callable[
        ...,
        ModelCarrier | FlextProtocolsResult.ResultLike[RuntimeAtomic] | None,
    ]
    type HandlerLike = Callable[
        ...,
        ModelCarrier | FlextProtocolsResult.ResultLike[RuntimeAtomic],
    ]
    type DispatchableHandler = (
        ModelCarrier
        | Callable[
            ...,
            ModelCarrier
            | FlextTypesServices.RuntimeAtomic
            | FlextProtocolsResult.ResultLike[FlextTypesServices.RuntimeAtomic]
            | None,
        ]
    )
    type HandlerProtocolVariant = (
        FlextTypesServices.DispatchableHandler
        | FlextProtocolsHandler.DispatchMessage
        | FlextProtocolsHandler.Handle
        | FlextProtocolsHandler.Execute
        | FlextProtocolsHandler.AutoDiscoverableHandler
    )
    type ResolvedHandlerCallable = Callable[
        ...,
        ModelCarrier
        | FlextTypesServices.RuntimeAtomic
        | FlextProtocolsResult.ResultLike[FlextTypesServices.RuntimeAtomic]
        | None,
    ]
    type RoutedHandlerCallable = Callable[
        [p.Routable],
        FlextTypesServices.RuntimeAtomic
        | FlextProtocolsResult.ResultLike[FlextTypesServices.RuntimeAtomic]
        | None,
    ]
    type RegisteredHandler = tuple[
        FlextTypesServices.HandlerProtocolVariant,
        FlextTypesServices.ResolvedHandlerCallable,
    ]
    type AutoHandlerRegistration = tuple[
        FlextTypesServices.HandlerProtocolVariant,
        FlextTypesServices.ResolvedHandlerCallable,
        tuple[FlextTypesServices.MessageTypeSpecifier, ...],
    ]
    type RegistrablePlugin = (
        FlextTypesServices.ScalarOrModel
        | Callable[..., FlextTypesServices.ScalarOrModel]
    )
    type LoggerFactory = Callable[..., FlextProtocolsLogging.OutputLogger] | None
    type LoggerWrapperFactory = Callable[[], type[FlextProtocolsLogging.Logger]]
    type StructlogProcessor = Callable[
        ...,
        t.RecursiveContainerMapping,
    ]
    type ContextHookCallable = Callable[
        [t.Scalar],
        FlextTypesServices.ValueOrModel | None,
    ]
    type ContextHookMap = Mapping[str, Sequence[ContextHookCallable]]

    type SortableObjectType = str | int | float
    type TypeHintSpecifier = type | str | UnionType | GenericAlias | TypeAliasType
    type ValueAdapter[T] = mp.TypeAdapter[T]
    type TypeOriginSpecifier = TypeHintSpecifier
    type GenericTypeArgument = str | type[t.Scalar]
    type MessageTypeSpecifier = type | str | UnionType | GenericAlias | TypeAliasType
    type IncEx = set[str] | Mapping[str, set[str] | bool]

    type LazyImportIndex = Mapping[str, str | t.StrSequence]
    type ConfigurationMapping = Mapping[str, t.Scalar]
    type ResultErrorData = Mapping[str, t.Container]
    type FlatContainerMapping = Mapping[str, t.Container]
    type MutableFlatContainerMapping = MutableMapping[str, t.Container]
    type MutableConfigurationMapping = MutableMapping[str, t.Scalar]
    type ScopedContainerRegistry = MutableMapping[
        str,
        MutableMapping[str, t.Container],
    ]
    type ScopedScalarRegistry = MutableMapping[
        str,
        MutableMapping[str, t.Scalar],
    ]
    type ServiceMap = Mapping[str, RegisterableService]
    type FactoryMap = Mapping[str, FactoryCallable]
    type ResourceMap = Mapping[str, ResourceCallable]
    type SettingsClass = type[FlextProtocolsSettings.Settings]
    type RuntimeModule = ModuleType
    type LazyScalar = t.Scalar | bytes | date | time
    type LazyCollection = Mapping[str, LazyScalar] | Sequence[LazyScalar]
    type ModuleExportValue = t.Container | bytes | date | time
    type ModuleExport = (
        ModuleExportValue
        | LazyCollection
        | RuntimeModule
        | type[BaseException | Enum]
        | Callable[..., ModuleExportValue | LazyCollection | None]
    )
    type LazyGetattr = Callable[[str], ModuleExport]
    type LazyDir = Callable[[], list[str]]
    type LazyNamespaceValue = ModuleExport | t.StrSequence | LazyGetattr | LazyDir

    type ValidatorCallable = Callable[[ScalarOrModel | None], ScalarOrModel | None]

    type MapperCallable = Callable[
        [t.RecursiveContainer],
        t.RecursiveContainer,
    ]
    type MapperInput = MapperCallable | t.RecursiveContainer
    type StrictValue = t.Scalar | ConfigurationMapping | Sequence[t.Container] | None
    type PaginationMeta = Mapping[str, int | bool]

    type GuardInput = (
        t.Scalar
        | Path
        | RecursiveSequenceCarrier
        | Sequence[AbstractSet[t.RecursiveContainer]]
        | Mapping[str, t.RecursiveContainer | ModelCarrier]
        | Mapping[
            str,
            AbstractSet[t.RecursiveContainer] | t.RecursiveContainer,
        ]
        | Mapping[
            int | str,
            AbstractSet[t.RecursiveContainer] | t.RecursiveContainer | ModelCarrier,
        ]
        | RecursiveTupleCarrier
        | tuple[type, ...]
        | AbstractSet[t.RecursiveContainer]
        | type
        | ModelCarrier
        | FlextTypingContainers.ConfigMap
        | FlextTypingContainers.Dict
        | FlextTypingContainers.ObjectList
        | RegisterableService
        | Callable[..., t.RecursiveContainer]
        | FlextProtocolsContext.Context
        | FlextProtocolsSettings.Settings
        | FlextProtocolsHandler.Dispatcher
        | FlextProtocolsResult.ResultLike[RuntimeAtomic]
        | None
    )

    type ProtocolSubject = (
        GuardInput
        | p.Flushable
        | FlextProtocolsHandler.AutoDiscoverableHandler
        | FlextProtocolsContainer.ProviderLike[ModelCarrier | t.Container]
        | FlextProtocolsService.DispatchableService
        | FlextProtocolsResult.SuccessCheckable
        | FlextProtocolsResult.StructuredError
        | FlextProtocolsResult.ErrorDomainProtocol
        | FlextProtocolsSettings.Configurable
        | FlextProtocolsHandler.Handle
        | FlextProtocolsHandler.Execute
    )

    type UserOverridesMapping = Mapping[
        str,
        t.Scalar | FlextTypingContainers.ConfigMap | t.ScalarList,
    ]
