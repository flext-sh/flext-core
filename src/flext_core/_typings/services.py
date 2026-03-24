"""FlextTypesServices - service, mapping, and runtime helper type aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path
from types import GenericAlias, ModuleType, UnionType
from typing import TYPE_CHECKING, Protocol, TypeAliasType

from pydantic import BaseModel

from flext_core import FlextTypingBase, FlextTypingContainers

if TYPE_CHECKING:
    from flext_core import FlextDispatcher, p

    class _ServiceInstance(Protocol):
        """Protocol for arbitrary service instances (adapters, services, etc.).

        Satisfied by any class instance that has ``__dict__`` — this matches
        the runtime validator in ``FlextContainer`` (``hasattr(v, '__dict__')``).
        Only used during type-checking; never instantiated at runtime.
        """


class FlextTypesServices:
    """Type aliases for service registration and runtime mappings."""

    type RegistryDict[T] = MutableMapping[str, T]

    type ScalarOrModel = FlextTypingBase.Scalar | BaseModel
    type ValueOrModel = FlextTypingBase.NormalizedValue | BaseModel
    type RuntimeData = ValueOrModel | FlextTypesServices.MetadataValue
    type RuntimeAtomic = FlextTypingBase.Container | BaseModel

    type BootstrapInput = (
        BaseModel | Mapping[str, FlextTypingBase.NormalizedValue] | None
    )

    if TYPE_CHECKING:
        type RegisterableService = (
            FlextTypingBase.Container
            | BaseModel
            | _ServiceInstance
            | Mapping[
                str,
                FlextTypingBase.Container | FlextTypingBase.NormalizedValue,
            ]
            | Sequence[FlextTypingBase.Container | FlextTypingBase.NormalizedValue]
            | Callable[..., FlextTypingBase.Container | BaseModel]
            | Callable[..., FlextTypesServices.RegisterableService]
        )
    else:
        type RegisterableService = (
            FlextTypingBase.Container
            | BaseModel
            | Mapping[
                str,
                FlextTypingBase.Container | FlextTypingBase.NormalizedValue,
            ]
            | Sequence[FlextTypingBase.Container | FlextTypingBase.NormalizedValue]
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
    type MetadataOrValue = MetadataValue | FlextTypingBase.NormalizedValue
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
    type LoggerFactory = Callable[[], p.Logger] | p.Logger | None
    type StructlogProcessor = Callable[
        ...,
        Mapping[str, FlextTypingBase.NormalizedValue],
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
    type ModuleExport = (
        FlextTypingBase.Container
        | ModuleType
        | type
        | Callable[..., FlextTypingBase.Container]
    )

    type ValidatorCallable = Callable[[ScalarOrModel | None], ScalarOrModel | None]

    type MapperCallable = Callable[
        [FlextTypingBase.NormalizedValue],
        FlextTypingBase.NormalizedValue,
    ]
    type MapperInput = MapperCallable | FlextTypingBase.NormalizedValue
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
        | Sequence[FlextTypingBase.NormalizedValue]
        | Mapping[str, FlextTypingBase.NormalizedValue | BaseModel]
        | tuple[FlextTypingBase.NormalizedValue, ...]
        | tuple[type, ...]
        | set[FlextTypingBase.NormalizedValue]
        | BaseModel
        | FlextTypingContainers.ConfigMap
        | RegisterableService
        | p.Context
        | p.Settings
        | p.Dispatcher
        | FlextDispatcher
        | p.ResultLike[RuntimeAtomic]
        | None
    )
