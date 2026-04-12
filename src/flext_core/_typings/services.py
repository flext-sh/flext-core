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

from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._typings.base import FlextTypingBase
from flext_core._typings.containers import FlextTypingContainers

if TYPE_CHECKING:
    from flext_core import p


class FlextTypesServices:
    """Type aliases for service registration and runtime mappings."""

    type RegistryDict[T] = MutableMapping[str, T]

    type ScalarOrModel = FlextTypingBase.Scalar | Path | FlextModelsPydantic.BaseModel
    type ValueOrModel = (
        FlextTypingBase.RecursiveContainer | FlextModelsPydantic.BaseModel
    )
    type RuntimeData = ValueOrModel | FlextTypesServices.MetadataValue
    type RuntimeAtomic = FlextTypingBase.Container | FlextModelsPydantic.BaseModel

    type BootstrapInput = (
        FlextModelsPydantic.BaseModel | FlextTypingBase.RecursiveContainerMapping | None
    )

    type RegisterableService = (
        FlextTypingBase.Container
        | FlextModelsPydantic.BaseModel
        | p.Logger
        | Mapping[
            str,
            FlextTypingBase.Container | FlextTypingBase.RecursiveContainer,
        ]
        | Sequence[FlextTypingBase.Container | FlextTypingBase.RecursiveContainer]
        | Callable[
            ...,
            FlextTypingBase.Container | FlextModelsPydantic.BaseModel | p.Logger,
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
        | FlextModelsPydantic.BaseModel
        | Mapping[
            str,
            FlextTypesServices.ValueOrModel,
        ]
    )
    type ConfigModelInput = (
        FlextModelsPydantic.BaseModel
        | FlextTypingContainers.ConfigMap
        | Mapping[str, FlextTypesServices.RuntimeAtomic]
    )
    type MetadataInput = (
        FlextModelsPydantic.BaseModel
        | FlextTypingContainers.ConfigMap
        | Mapping[str, FlextTypingBase.Scalar]
        | None
    )
    type HandlerCallable = Callable[..., FlextModelsPydantic.BaseModel | None]
    type HandlerLike = Callable[..., FlextModelsPydantic.BaseModel]
    type DispatchableHandler = (
        FlextModelsPydantic.BaseModel
        | Callable[
            ..., FlextModelsPydantic.BaseModel | FlextTypesServices.RuntimeAtomic | None
        ]
    )
    type HandlerProtocolVariant = (
        FlextTypesServices.DispatchableHandler
        | p.DispatchMessage
        | p.Handle
        | p.Execute
        | p.AutoDiscoverableHandler
    )
    type ResolvedHandlerCallable = Callable[
        ...,
        FlextModelsPydantic.BaseModel | FlextTypesServices.RuntimeAtomic | None,
    ]
    type RoutedHandlerCallable = Callable[
        [p.Routable],
        FlextTypesServices.RuntimeAtomic
        | p.Result[FlextTypesServices.RuntimeAtomic]
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
    type LoggerFactory = Callable[..., p.OutputLogger] | None
    type LoggerWrapperFactory = Callable[[], type[p.Logger]]
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
    type SettingsClass = type[p.Settings]
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
        | Mapping[
            str, FlextTypingBase.RecursiveContainer | FlextModelsPydantic.BaseModel
        ]
        | tuple[FlextTypingBase.RecursiveContainer, ...]
        | tuple[type, ...]
        | set[FlextTypingBase.RecursiveContainer]
        | FlextModelsPydantic.BaseModel
        | FlextTypingContainers.ConfigMap
        | RegisterableService
        | p.Context
        | p.Settings
        | p.Dispatcher
        | p.Result[RuntimeAtomic]
        | None
    )

    type ProtocolSubject = (
        GuardInput
        | p.Flushable
        | p.AutoDiscoverableHandler
        | p.ProviderLike[FlextModelsPydantic.BaseModel | FlextTypingBase.Container]
        | p.DispatchableService
        | p.SuccessCheckable
        | p.StructuredError
        | p.ErrorDomainProtocol
        | p.Configurable
        | p.Handle
        | p.Execute
    )

    type UserOverridesMapping = Mapping[
        str,
        FlextTypingBase.Scalar
        | FlextTypingContainers.ConfigMap
        | FlextTypingBase.ScalarList,
    ]

    type RegistrationKwarg = (
        RuntimeData
        | p.Settings
        | p.Context
        | Mapping[str, RegisterableService]
        | Mapping[str, FactoryCallable]
        | Mapping[str, ResourceCallable]
    )
