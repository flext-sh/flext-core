"""FlextTypesServices - service, mapping, and runtime helper type aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from flext_core._typings.base import FlextTypingBase
from flext_core._typings.core import FlextTypesCore


class FlextTypesServices(FlextTypesCore):
    """Type aliases for service registration and runtime mappings."""

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

    type RegisterableService = (
        FlextTypingBase.Container
        | BaseModel
        | Mapping[
            str,
            FlextTypingBase.Container | FlextTypesCore.ContainerValue,
        ]
        | Sequence[FlextTypingBase.Container | FlextTypesCore.ContainerValue]
        | Callable[..., FlextTypingBase.Container | BaseModel]
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
    type MetadataAttributeValue = MetadataValue
    type HandlerCallable = Callable[[BaseModel], BaseModel]
    type HandlerLike = Callable[..., BaseModel]
    type RegistrablePlugin = (
        FlextTypingBase.Scalar
        | BaseModel
        | Callable[..., FlextTypingBase.Scalar | BaseModel]
    )

    # Other Types
    type SortableObjectType = str | int | float
    type TypeHintSpecifier = (
        type | str | Callable[[FlextTypingBase.Scalar], FlextTypingBase.Scalar]
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
    type MapperCallable = Callable[
        [FlextTypesCore.NormalizedValue], FlextTypesCore.NormalizedValue
    ]
    type StrictValue = (
        FlextTypingBase.Scalar
        | ConfigurationMapping
        | list[FlextTypingBase.Container]
        | None
    )
    type PaginationMeta = dict[str, int | bool]
