"""FlextTypesServices - service, mapping, and runtime helper type aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import GenericAlias, ModuleType, UnionType
from typing import TYPE_CHECKING, Protocol, TypeAliasType, runtime_checkable

from pydantic import BaseModel

from flext_core import FlextTypingBase, FlextTypingContainers

if TYPE_CHECKING:
    from flext_core import p


class FlextTypesServices:
    """Type aliases for service registration and runtime mappings."""

    type ScalarOrModel = FlextTypingBase.Scalar | BaseModel

    @runtime_checkable
    class DispatchableService(Protocol):
        """Structural protocol for dispatch-capable service objects in the DI container.

        Matches FlextDispatcher and similar services that expose a dispatch method.
        Parameter uses Protocol bound since dispatch implementations accept varying
        message protocols (Routable, Command, Query).
        """

        def dispatch(self, message: BaseModel, /) -> BaseModel:
            """Dispatch a message and return the result."""
            ...

    type ValueOrModel = FlextTypingBase.NormalizedValue | BaseModel

    type BootstrapInput = (
        BaseModel | Mapping[str, FlextTypingBase.NormalizedValue] | None
    )

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
        | FlextTypesServices.DispatchableService
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
    type RegistrablePlugin = ScalarOrModel | Callable[..., ScalarOrModel]
    type LoggerFactory = Callable[[], p.Logger] | p.Logger | None
    type StructlogProcessor = Callable[
        ..., Mapping[str, FlextTypingBase.NormalizedValue]
    ]

    # Other Types
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
    type ServiceMap = Mapping[str, RegisterableService]
    type ModuleExport = (
        FlextTypingBase.Container
        | ModuleType
        | type
        | Callable[..., FlextTypingBase.Container]
    )

    # --- MAPPER / CACHE / CONVERSION CONSOLIDATED TYPES ---
    type ValidatorCallable = Callable[[ScalarOrModel | None], ScalarOrModel | None]

    type MapperCallable = Callable[
        [FlextTypingBase.NormalizedValue],
        FlextTypingBase.NormalizedValue,
    ]
    type MapperInput = MapperCallable | FlextTypingBase.NormalizedValue
    type StrictValue = (
        FlextTypingBase.Scalar
        | ConfigurationMapping
        | list[FlextTypingBase.Container]
        | None
    )
    type PaginationMeta = dict[str, int | bool]

    # GuardInput uses forward references to avoid circular imports
    # These are resolved at runtime via the protocol facade
    type GuardInput = (
        FlextTypingBase.Scalar
        | Path
        | list[FlextTypingBase.NormalizedValue]
        | Mapping[str, FlextTypingBase.NormalizedValue | BaseModel]
        | tuple[FlextTypingBase.NormalizedValue, ...]
        | BaseModel
        | FlextTypingContainers.ConfigMap
        | RegisterableService
        | None
    )
