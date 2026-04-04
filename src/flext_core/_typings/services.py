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

from pydantic import BaseModel

from flext_core import FlextTypingBase, FlextTypingContainers

if TYPE_CHECKING:
    from flext_core import FlextDispatcher, m, p


class FlextTypesServices:
    """Type aliases for service registration and runtime mappings."""

    type RegistryDict[T] = MutableMapping[str, T]

    type ScalarOrModel = FlextTypingBase.Scalar | Path | BaseModel
    type ValueOrModel = FlextTypingBase.RecursiveContainer | BaseModel
    type RuntimeData = ValueOrModel | FlextTypesServices.MetadataValue
    type RuntimeAtomic = FlextTypingBase.Container | BaseModel

    type BootstrapInput = BaseModel | FlextTypingBase.RecursiveContainerMapping | None

    type RegisterableService = (
        FlextTypingBase.Container
        | BaseModel
        | Mapping[
            str,
            FlextTypingBase.Container | FlextTypingBase.RecursiveContainer,
        ]
        | Sequence[FlextTypingBase.Container | FlextTypingBase.RecursiveContainer]
        | Callable[..., FlextTypingBase.Container | BaseModel]
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
    type ConfigModelInput = BaseModel | FlextTypingContainers.ConfigMap
    type MetadataInput = (
        BaseModel
        | FlextTypingContainers.ConfigMap
        | Mapping[str, FlextTypingBase.Scalar]
        | None
    )
    type HandlerCallable = Callable[..., BaseModel | None]
    type HandlerLike = Callable[..., BaseModel]
    type DispatchableHandler = (
        BaseModel | Callable[..., BaseModel | FlextTypesServices.RuntimeAtomic | None]
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
        BaseModel | FlextTypesServices.RuntimeAtomic | None,
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
    type StructlogProcessor = Callable[
        ...,
        FlextTypingBase.RecursiveContainerMapping,
    ]

    type SortableObjectType = str | int | float
    type TypeHintSpecifier = (
        type
        | str
        | UnionType
        | GenericAlias
        | TypeAliasType
        | Callable[[FlextTypingBase.Scalar], FlextTypingBase.Scalar]
    )
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
    type LazyScalar = FlextTypingBase.Scalar | bytes | date | time
    type LazyCollection = Mapping[str, LazyScalar] | Sequence[LazyScalar]
    type ModuleExportValue = FlextTypingBase.Container | bytes | date | time
    type ModuleExport = (
        ModuleExportValue
        | LazyCollection
        | ModuleType
        | type[BaseException | Enum]
        | Callable[..., ModuleExportValue | LazyCollection | None]
    )

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
        | Mapping[str, FlextTypingBase.RecursiveContainer | BaseModel]
        | tuple[FlextTypingBase.RecursiveContainer, ...]
        | tuple[type, ...]
        | set[FlextTypingBase.RecursiveContainer]
        | BaseModel
        | FlextTypingContainers.ConfigMap
        | RegisterableService
        | p.Context
        | p.Settings
        | p.Dispatcher
        | FlextDispatcher
        | p.Result[RuntimeAtomic]
        | None
    )

    type ProtocolSubject = (
        GuardInput
        | p.Flushable
        | p.AutoDiscoverableHandler
        | p.ProviderLike[BaseModel | FlextTypingBase.Container]
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
        | m.ContainerConfig
        | Mapping[str, m.ServiceRegistration]
        | Mapping[str, m.FactoryRegistration]
        | Mapping[str, m.ResourceRegistration]
    )
