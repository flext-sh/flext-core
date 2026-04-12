"""FlextTypesServices - service, mapping, and runtime helper type aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from datetime import date, time
from enum import Enum
from pathlib import Path
from types import GenericAlias, ModuleType, UnionType
from typing import TypeAliasType

from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._protocols.base import FlextProtocolsBase
from flext_core._protocols.container import FlextProtocolsContainer
from flext_core._protocols.context import FlextProtocolsContext
from flext_core._protocols.handler import FlextProtocolsHandler
from flext_core._protocols.logging import FlextProtocolsLogging
from flext_core._protocols.result import FlextProtocolsResult
from flext_core._protocols.service import FlextProtocolsService
from flext_core._protocols.settings import FlextProtocolsSettings
from flext_core._typings.base import FlextTypingBase
from flext_core._typings.containers import FlextTypingContainers


class FlextTypesServices:
    """Type aliases for service registration and runtime mappings."""

    type RegistryDict[T] = MutableMapping[str, T]
    type ModelCarrier = FlextModelsPydantic.BaseModel
    type ModelClass[T: ModelCarrier] = type[T]

    type ScalarOrModel = FlextTypingBase.Scalar | Path | ModelCarrier
    type ValueOrModel = FlextTypingBase.RecursiveContainer | ModelCarrier
    type RuntimeData = ValueOrModel | FlextTypesServices.MetadataValue
    type RuntimeAtomic = FlextTypingBase.Container | ModelCarrier

    type BootstrapInput = (
        ModelCarrier | FlextTypingBase.RecursiveContainerMapping | None
    )

    type RegisterableService = (
        FlextTypingBase.Container
        | ModelCarrier
        | FlextProtocolsLogging.Logger
        | Mapping[
            str,
            FlextTypingBase.Container | FlextTypingBase.RecursiveContainer,
        ]
        | Sequence[FlextTypingBase.Container | FlextTypingBase.RecursiveContainer]
        | Callable[
            ...,
            FlextTypingBase.Container | ModelCarrier | FlextProtocolsLogging.Logger,
        ]
        | Callable[..., FlextTypesServices.RegisterableService]
    )
    type FactoryCallable = Callable[[], RegisterableService]
    type ResourceCallable = Callable[[], RegisterableService]
    type MetadataValue = (
        FlextTypingBase.Scalar
        | Mapping[
            str,
            FlextTypingBase.Scalar | Sequence[FlextTypingBase.Scalar],
        ]
        | Sequence[FlextTypingBase.Scalar]
    )
    type MetadataOrValue = MetadataValue | FlextTypingBase.RecursiveContainer
    type MetadataAttributeValue = MetadataValue
    type ModelInput = (
        FlextTypingBase.RecursiveContainer
        | ModelCarrier
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
        ModelCarrier
        | FlextTypingContainers.ConfigMap
        | Mapping[str, FlextTypingBase.Scalar]
        | None
    )
    type HandlerCallable = Callable[..., ModelCarrier | None]
    type HandlerLike = Callable[..., ModelCarrier]
    type DispatchableHandler = (
        ModelCarrier
        | Callable[..., ModelCarrier | FlextTypesServices.RuntimeAtomic | None]
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
        ModelCarrier | FlextTypesServices.RuntimeAtomic | None,
    ]
    type RoutedHandlerCallable = Callable[
        [FlextProtocolsBase.Routable],
        FlextTypesServices.RuntimeAtomic
        | FlextProtocolsResult.Result[FlextTypesServices.RuntimeAtomic]
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
        FlextTypingBase.RecursiveContainerMapping,
    ]
    type ContextHookCallable = Callable[
        [FlextTypingBase.Scalar],
        FlextTypesServices.ValueOrModel | None,
    ]
    type ContextHookMap = Mapping[str, Sequence[ContextHookCallable]]

    type SortableObjectType = str | int | float
    type TypeHintSpecifier = (
        type
        | str
        | UnionType
        | GenericAlias
        | TypeAliasType
        | Callable[[FlextTypingBase.Scalar], FlextTypingBase.Scalar]
    )
    type ValueAdapter[T] = FlextModelsPydantic.TypeAdapter[T]
    type TypeOriginSpecifier = TypeHintSpecifier
    type GenericTypeArgument = str | type[FlextTypingBase.Scalar]
    type MessageTypeSpecifier = str | type
    type IncEx = set[str] | Mapping[str, set[str] | bool]

    type LazyImportIndex = Mapping[str, str | Sequence[str]]
    type ConfigurationMapping = Mapping[str, FlextTypingBase.Scalar]
    type ResultErrorData = Mapping[str, FlextTypingBase.Container]
    type FlatContainerMapping = Mapping[str, FlextTypingBase.Container]
    type MutableFlatContainerMapping = MutableMapping[str, FlextTypingBase.Container]
    type MutableConfigurationMapping = MutableMapping[str, FlextTypingBase.Scalar]
    type ScopedContainerRegistry = MutableMapping[
        str,
        MutableMapping[str, FlextTypingBase.Container],
    ]
    type ScopedScalarRegistry = MutableMapping[
        str,
        MutableMapping[str, FlextTypingBase.Scalar],
    ]
    type ServiceMap = Mapping[str, RegisterableService]
    type FactoryMap = Mapping[str, FactoryCallable]
    type ResourceMap = Mapping[str, ResourceCallable]
    type SettingsClass = type[FlextProtocolsSettings.Settings]
    type RuntimeModule = ModuleType
    type LazyScalar = FlextTypingBase.Scalar | bytes | date | time
    type LazyCollection = Mapping[str, LazyScalar] | Sequence[LazyScalar]
    type ModuleExportValue = FlextTypingBase.Container | bytes | date | time
    type ModuleExport = (
        ModuleExportValue
        | LazyCollection
        | RuntimeModule
        | type[BaseException | Enum]
        | Callable[..., ModuleExportValue | LazyCollection | None]
    )
    type LazyGetattr = Callable[[str], ModuleExport]
    type LazyDir = Callable[[], list[str]]
    type LazyNamespaceValue = ModuleExport | Sequence[str] | LazyGetattr | LazyDir

    type ValidatorCallable = Callable[[ScalarOrModel | None], ScalarOrModel | None]

    type MapperCallable = Callable[
        [FlextTypingBase.RecursiveContainer],
        FlextTypingBase.RecursiveContainer,
    ]
    type MapperInput = MapperCallable | FlextTypingBase.RecursiveContainer
    type StrictValue = (
        FlextTypingBase.Scalar
        | ConfigurationMapping
        | Sequence[FlextTypingBase.Container]
        | None
    )
    type PaginationMeta = Mapping[str, int | bool]

    type GuardInput = (
        FlextTypingBase.Scalar
        | Path
        | FlextTypingBase.RecursiveContainerList
        | Mapping[str, FlextTypingBase.RecursiveContainer | ModelCarrier]
        | tuple[FlextTypingBase.RecursiveContainer, ...]
        | tuple[type, ...]
        | set[FlextTypingBase.RecursiveContainer]
        | ModelCarrier
        | FlextTypingContainers.ConfigMap
        | RegisterableService
        | FlextProtocolsContext.Context
        | FlextProtocolsSettings.Settings
        | FlextProtocolsHandler.Dispatcher
        | FlextProtocolsResult.Result[RuntimeAtomic]
        | None
    )

    type ProtocolSubject = (
        GuardInput
        | FlextProtocolsBase.Flushable
        | FlextProtocolsHandler.AutoDiscoverableHandler
        | FlextProtocolsContainer.ProviderLike[ModelCarrier | FlextTypingBase.Container]
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
        FlextTypingBase.Scalar
        | FlextTypingContainers.ConfigMap
        | FlextTypingBase.ScalarList,
    ]

    type RegistrationKwarg = (
        RuntimeData
        | FlextProtocolsSettings.Settings
        | FlextProtocolsContext.Context
        | Mapping[str, RegisterableService]
        | Mapping[str, FactoryCallable]
        | Mapping[str, ResourceCallable]
    )
