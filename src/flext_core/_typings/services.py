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
from typing import TYPE_CHECKING, TypeAliasType

from structlog.typing import BindableLogger

from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._typings.base import FlextTypingBase
from flext_core._typings.containers import FlextTypingContainers
from flext_core._utilities.pydantic import FlextUtilitiesPydantic

if TYPE_CHECKING:
    from flext_core import m, p

    type _ModelCarrier = m.BaseModel
    type _LoggerCarrier = p.Logger
    type _OutputLoggerCarrier = p.OutputLogger
    type _DispatchMessageCarrier = p.DispatchMessage
    type _HandleCarrier = p.Handle
    type _ExecuteCarrier = p.Execute
    type _AutoDiscoverableHandlerCarrier = p.AutoDiscoverableHandler
    type _RoutableCarrier = p.Routable
    type _ResultCarrier[T] = p.Result[T]
    type _SettingsCarrier = p.Settings
    type _ContextCarrier = p.Context
    type _DispatcherCarrier = p.Dispatcher
    type _FlushableCarrier = p.Flushable
    type _ProviderLikeCarrier[T] = p.ProviderLike[T]
    type _DispatchableServiceCarrier = p.DispatchableService
    type _SuccessCheckableCarrier = p.SuccessCheckable
    type _StructuredErrorCarrier = p.StructuredError
    type _ErrorDomainCarrier = p.ErrorDomainProtocol
    type _ConfigurableCarrier = p.Configurable
else:
    type _ModelCarrier = FlextModelsPydantic.BaseModel
    type _LoggerCarrier = BindableLogger
    type _OutputLoggerCarrier = BindableLogger
    type _DispatchMessageCarrier = Callable[
        ...,
        FlextModelsPydantic.BaseModel | FlextTypingBase.Container | None,
    ]
    type _HandleCarrier = _DispatchMessageCarrier
    type _ExecuteCarrier = _DispatchMessageCarrier
    type _AutoDiscoverableHandlerCarrier = Callable[[type], bool]
    type _RoutableCarrier = FlextModelsPydantic.BaseModel | FlextTypingBase.Container
    type _ResultCarrier[T] = FlextModelsPydantic.BaseModel | T
    type _SettingsCarrier = FlextModelsPydantic.BaseModel
    type _ContextCarrier = (
        FlextModelsPydantic.BaseModel | FlextTypingBase.ContainerMapping
    )
    type _DispatcherCarrier = Callable[
        [_RoutableCarrier],
        FlextModelsPydantic.BaseModel | FlextTypingBase.Container | None,
    ]
    type _FlushableCarrier = Callable[[], None]
    type _ProviderLikeCarrier[T] = Callable[[], T]
    type _DispatchableServiceCarrier = Callable[
        [FlextModelsPydantic.BaseModel],
        FlextModelsPydantic.BaseModel,
    ]
    type _SuccessCheckableCarrier = FlextModelsPydantic.BaseModel
    type _StructuredErrorCarrier = FlextModelsPydantic.BaseModel
    type _ErrorDomainCarrier = Enum
    type _ConfigurableCarrier = FlextModelsPydantic.BaseModel


class FlextTypesServices:
    """Type aliases for service registration and runtime mappings."""

    type RegistryDict[T] = MutableMapping[str, T]
    type ModelCarrier = _ModelCarrier
    type ModelClass[T: _ModelCarrier] = type[T]

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
        | _LoggerCarrier
        | Mapping[
            str,
            FlextTypingBase.Container | FlextTypingBase.RecursiveContainer,
        ]
        | Sequence[FlextTypingBase.Container | FlextTypingBase.RecursiveContainer]
        | Callable[
            ...,
            FlextTypingBase.Container | ModelCarrier | _LoggerCarrier,
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
        | _DispatchMessageCarrier
        | _HandleCarrier
        | _ExecuteCarrier
        | _AutoDiscoverableHandlerCarrier
    )
    type ResolvedHandlerCallable = Callable[
        ...,
        ModelCarrier | FlextTypesServices.RuntimeAtomic | None,
    ]
    type RoutedHandlerCallable = Callable[
        [_RoutableCarrier],
        FlextTypesServices.RuntimeAtomic
        | _ResultCarrier[FlextTypesServices.RuntimeAtomic]
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
    type LoggerFactory = Callable[..., _OutputLoggerCarrier] | None
    type LoggerWrapperFactory = Callable[[], type[_LoggerCarrier]]
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
    type ValueAdapter[T] = FlextUtilitiesPydantic.TypeAdapter[T]
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
    type SettingsClass = type[_SettingsCarrier]
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
        | _ContextCarrier
        | _SettingsCarrier
        | _DispatcherCarrier
        | _ResultCarrier[RuntimeAtomic]
        | None
    )

    type ProtocolSubject = (
        GuardInput
        | _FlushableCarrier
        | _AutoDiscoverableHandlerCarrier
        | _ProviderLikeCarrier[ModelCarrier | FlextTypingBase.Container]
        | _DispatchableServiceCarrier
        | _SuccessCheckableCarrier
        | _StructuredErrorCarrier
        | _ErrorDomainCarrier
        | _ConfigurableCarrier
        | _HandleCarrier
        | _ExecuteCarrier
    )

    type UserOverridesMapping = Mapping[
        str,
        FlextTypingBase.Scalar
        | FlextTypingContainers.ConfigMap
        | FlextTypingBase.ScalarList,
    ]

    type RegistrationKwarg = (
        RuntimeData
        | _SettingsCarrier
        | _ContextCarrier
        | Mapping[str, RegisterableService]
        | Mapping[str, FactoryCallable]
        | Mapping[str, ResourceCallable]
    )
