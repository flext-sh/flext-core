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
)
from datetime import date, time, tzinfo
from enum import Enum
from types import GenericAlias, ModuleType, UnionType
from typing import TypeAliasType

from flext_core import (
    FlextModelsPydantic as mp,
    FlextProtocolsBase as p,
    FlextProtocolsContainer as pc,
    FlextProtocolsContext as pcx,
    FlextProtocolsHandler as ph,
    FlextProtocolsLogging as pl,
    FlextProtocolsRegistry as pr,
    FlextProtocolsResult as prt,
    FlextProtocolsSettings as ps,
    FlextTypingBase as t,
)


class FlextTypesServices:
    """Type aliases for service registration and runtime mappings."""

    type JsonValue = t.JsonValue
    type JsonMapping = Mapping[str, t.JsonValue]
    type JsonSequence = Sequence[t.JsonValue]
    type JsonLikeValue = (
        t.Container | Mapping[str, JsonLikeValue] | Sequence[JsonLikeValue]
    )
    type JsonLikeMapping = Mapping[str, JsonLikeValue]
    type JsonLikeSequence = Sequence[JsonLikeValue]
    type JsonPayload = t.JsonValue | mp.BaseModel
    type RegistryDict[T] = MutableMapping[str, T]
    type ModelCarrier = mp.BaseModel
    type ProtocolModelCarrier = p.Model
    type DomainModelCarrier = ModelCarrier | ProtocolModelCarrier
    type ContainerCarrier = Mapping[str, t.Container]
    type ScalarOrModel = t.Scalar | mp.BaseModel
    type ModelClass[T: ModelCarrier] = type[T]
    type LogArgument = t.Container | p.Model
    type LogValue = FlextTypesServices.LogArgument | Exception
    type LogResult = prt.Result[bool]
    type MetadataValue = t.JsonValue
    type MetadataOrValue = MetadataValue | t.Container
    type MetadataAttributeValue = MetadataValue
    type ValueOrModel = t.Container | ModelCarrier
    type RuntimeAtomic = ValueOrModel
    type RuntimeData = RuntimeAtomic | JsonLikeValue
    type PresentValueOrModel = t.Container | ModelCarrier
    type BootstrapInput = ModelCarrier | Mapping[str, t.Container]
    type RegisterableService = (
        t.Container
        | ModelCarrier
        | ContainerCarrier
        | pl.Logger
        | Mapping[str, t.Container]
        | Sequence[t.Container]
        | Callable[
            ...,
            t.Container | ModelCarrier | ContainerCarrier | pl.Logger,
        ]
    )
    type FactoryCallable = Callable[[], RegisterableService]
    type ResourceCallable = Callable[[], RegisterableService]
    type ModelInput = t.Container | ModelCarrier | Mapping[str, ValueOrModel]
    type ConfigModelInput = ModelCarrier | Mapping[str, RuntimeAtomic]
    type MetadataInput = ModelCarrier | JsonLikeMapping
    type ServiceMap = Mapping[str, RegisterableService]
    type FactoryMap = Mapping[str, FactoryCallable]
    type ResourceMap = Mapping[str, ResourceCallable]
    type ContextHookCallable = Callable[[t.Scalar], ValueOrModel]
    type ContextHookMap = Mapping[str, Sequence[ContextHookCallable]]

    type HandlerCallable = Callable[
        ...,
        ModelCarrier | prt.ResultLike[ScalarOrModel],
    ]
    type HandlerLike = Callable[
        ...,
        ModelCarrier | prt.ResultLike[ScalarOrModel],
    ]
    type DispatchableHandler = (
        ModelCarrier
        | Callable[
            ...,
            ModelCarrier | RuntimeAtomic | prt.ResultLike[RuntimeAtomic],
        ]
    )
    type HandlerProtocolVariant = DispatchableHandler
    type ResolvedHandlerCallable = Callable[
        ...,
        ModelCarrier | RuntimeAtomic | prt.ResultLike[RuntimeAtomic],
    ]
    type RoutedHandlerCallable = Callable[
        [p.Routable],
        RuntimeAtomic | prt.ResultLike[RuntimeAtomic],
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
    type TypeHintSpecifier = type | str | UnionType | GenericAlias | TypeAliasType
    type ValueAdapter[T] = mp.TypeAdapter[T]
    type TypeOriginSpecifier = TypeHintSpecifier
    type GenericTypeArgument = str | type[t.Scalar]
    type MessageTypeSpecifier = type | str | UnionType | GenericAlias | TypeAliasType
    type IncEx = set[str] | Mapping[str, set[str] | bool]

    type LazyImportIndex = Mapping[str, str | t.StrSequence]
    type ConfigurationMapping = Mapping[str, t.Scalar]
    type ResultErrorData = Mapping[str, t.Container]
    type MutableConfigurationMapping = MutableMapping[str, t.Scalar]
    type ScopedContainerRegistry = MutableMapping[
        str,
        MutableMapping[str, t.Container],
    ]
    type ScopedScalarRegistry = MutableMapping[
        str,
        MutableMapping[str, t.Scalar],
    ]
    type SettingsClass = type[ps.Settings]
    type RuntimeModule = ModuleType
    type LazyScalar = t.Scalar | bytes | date | time
    type LazyCollection = Mapping[str, LazyScalar] | Sequence[LazyScalar]
    type ModuleExportValue = t.Container | bytes | date | time
    type ModuleExport = (
        ModuleExportValue
        | LazyCollection
        | RuntimeModule
        | type[BaseException | Enum]
        | Callable[..., ModuleExportValue | LazyCollection]
    )
    type LazyGetattr = Callable[[str], ModuleExport]
    type LazyDir = Callable[[], list[str]]
    type LazyNamespaceValue = ModuleExport | t.StrSequence | LazyGetattr | LazyDir

    type ValidatorCallable = Callable[[ScalarOrModel], ScalarOrModel]

    type MapperCallable = Callable[
        [t.Container],
        t.Container,
    ]
    type MapperInput = MapperCallable | t.Container
    type StrictValue = t.Scalar | ConfigurationMapping | Sequence[t.Container]
    type PaginationMeta = Mapping[str, int | bool]

    type GuardInput = (
        type[BaseException | Enum]
        | t.Container
        | ContainerCarrier
        | Mapping[str, ValueOrModel]
        | Sequence[t.Container]
        | bytes
        | bytearray
        | ModelCarrier
        | ProtocolModelCarrier
        | RegisterableService
        | Callable[..., t.Container | ModelCarrier]
        | Callable[[], RegisterableService]
        | type
        | tuple[type, ...]
        | tuple[t.Container, ...]
        | Enum
        | ModuleType
        | GenericAlias
        | UnionType
        | TypeAliasType
        | tzinfo
        | frozenset[str]
        | pc.Container
        | pcx.Context
        | ph.Dispatcher
        | ph.Handle
        | ph.Middleware
        | pl.Logger
        | pl.HasLogger
        | pr.Registry
        | prt.ResultLike[RuntimeAtomic]
        | ps.Settings
    )

    type UserOverridesMapping = Mapping[
        str,
        t.Scalar | Mapping[str, t.Container] | t.ScalarList,
    ]
