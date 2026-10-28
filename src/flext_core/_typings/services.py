"""FlextTypesServices - service, mapping, and runtime helper type aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Generator,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
    Set as AbstractSet,
)
from concurrent.futures import Future
from datetime import date, datetime, time, tzinfo
from enum import Enum
from logging import Formatter, Handler, LogRecord
from pathlib import Path
from types import GenericAlias, ModuleType, TracebackType, UnionType
from typing import BinaryIO, TextIO, TypeAliasType

from flext_core import c
from flext_core._typings import FlextTypesPydantic, FlextTypingBase


class FlextTypesServices:
    """Type aliases for service registration and runtime mappings."""

    type JsonPayload = (
        FlextTypesPydantic.JsonValue
        | FlextTypesPydantic.SerializeAsAny[FlextTypesPydantic.BaseModelInstance]
        | Path
        | datetime
    )

    type ValidationInput[TModel] = (
        FlextTypesServices.JsonPayload
        | bytes
        | bytearray
        | FlextTypesServices.ModelInput[
            TModel | FlextTypesServices.MetadataMapping | bytes | bytearray
        ]
        | Sequence[
            FlextTypesServices.JsonPayload
            | FlextTypesServices.ModelInput[
                TModel | FlextTypesServices.MetadataMapping | bytes | bytearray
            ]
        ]
    )

    type ServiceLeaf[T] = FlextTypingBase.Scalar | FlextTypesServices.JsonPayload | T
    type ServiceValue[T] = (
        FlextTypesServices.ServiceLeaf[T]
        | Mapping[str, FlextTypesServices.JsonPayload]
        | Sequence[FlextTypesServices.JsonPayload]
    )
    type RegisterableService[T] = (
        FlextTypesServices.ServiceValue[T]
        | Callable[..., FlextTypesServices.ServiceValue[T]]
    )
    type FactoryCallable[T] = Callable[[], FlextTypesServices.ServiceValue[T]]
    type ResourceCallable[T] = FlextTypesServices.FactoryCallable[T]
    type ServiceMap[T] = Mapping[str, FlextTypesServices.RegisterableService[T]]
    type FactoryMap[T] = Mapping[str, FlextTypesServices.FactoryCallable[T]]
    type ResourceMap[T] = Mapping[str, FlextTypesServices.ResourceCallable[T]]

    type OperationMetricsStream = Generator[FlextTypesServices.MutableMetadataMapping]

    type SettingsOverrideLeaf[T] = FlextTypesServices.JsonPayload | T
    type SettingsOverrideCollectionValue[T] = (
        FlextTypesServices.SettingsOverrideLeaf[T]
        | FlextTypingBase.JsonMapping
        | FlextTypingBase.JsonList
    )
    type SettingsOverride[T] = (
        FlextTypesServices.SettingsOverrideLeaf[T]
        | FlextTypingBase.JsonMapping
        | FlextTypingBase.JsonList
    )
    type SettingsOverridesMapping[T] = Mapping[
        str, FlextTypesServices.SettingsOverride[T] | None
    ]

    type RegistryDict[T] = MutableMapping[str, T]
    type DomainModelCarrier[T] = FlextTypesPydantic.BaseModelType | T
    type ScalarOrModel = FlextTypingBase.Scalar | FlextTypesPydantic.BaseModelType
    type ModelClass[T: FlextTypesPydantic.BaseModelType] = type[T]

    type LogLevel = c.LogLevel
    type LoggingTraceback = TracebackType
    type LoggingExceptionInfo = tuple[
        type[BaseException], BaseException, LoggingTraceback | None
    ]
    type LogArgument[T] = (
        FlextTypesServices.JsonPayload
        | FlextTypingBase.JsonMapping
        | Sequence[FlextTypesServices.JsonPayload]
        | T
    )
    type LogValue[T] = (
        FlextTypesServices.LogArgument[T]
        | BaseException
        | FlextTypesServices.LoggingExceptionInfo
    )
    type LogPositionalValue[T] = (
        FlextTypesServices.LogValue[T | Mapping[str, Scalar]]
        | Sequence[tuple[FlextTypesPydantic.JsonValue, ...]]
    )
    type LoggingNativeRecord = LogRecord
    type LoggingNativeFormatter = Formatter
    type LoggingNativeHandlers = list[Handler]
    type LoggingEventValue[T] = (
        FlextTypesServices.LogValue[T]
        | type[BaseException]
        | FlextTypesServices.LoggingTraceback
        | tuple[FlextTypesServices.LogPositionalValue[T], ...]
    )
    type LoggingEvent[T] = MutableMapping[str, FlextTypesServices.LoggingEventValue[T]]
    type LoggingContext[T] = dict[str, FlextTypesServices.LoggingEventValue[T]]
    type LoggingWriteCompletion = Future[int]

    type TextStream = TextIO
    type BinaryStream = BinaryIO
    type TextStreamIterator = Iterator[str]

    type MetadataMapping = Mapping[str, FlextTypesServices.JsonPayload]
    type MutableMetadataMapping = MutableMapping[str, FlextTypesServices.JsonPayload]
    type BootstrapInput = FlextTypesPydantic.BaseModelType | FlextTypingBase.JsonMapping
    type ServiceClass = type
    type UserOverridesMapping = Mapping[str, FlextTypesServices.JsonPayload]

    type ModelInput[T] = (
        FlextTypesPydantic.JsonValue
        | T
        | Mapping[str, FlextTypesPydantic.JsonValue | T]
    )
    type ConfigModelInput[T] = T | Mapping[str, FlextTypesPydantic.JsonValue]
    type MetadataInput[T] = (
        T | Mapping[str, FlextTypesServices.JsonPayload | None] | None
    )

    type ContextHookCallable = Callable[
        [FlextTypingBase.Scalar], FlextTypesServices.JsonPayload
    ]
    type ContextHookMap = Mapping[str, Sequence[ContextHookCallable]]

    type HandlerCallable[T] = Callable[..., FlextTypesPydantic.BaseModelType | T]
    type DispatchableHandler[T, RoutableT, ResultT] = (
        FlextTypesPydantic.BaseModelType
        | T
        | Callable[
            [RoutableT],
            FlextTypesPydantic.BaseModelType
            | FlextTypesServices.JsonPayload
            | ResultT
            | None,
        ]
    )
    type ResolvedHandlerCallable[ResultT] = Callable[
        ...,
        FlextTypesPydantic.BaseModelType
        | FlextTypesServices.JsonPayload
        | ResultT
        | None,
    ]
    type RoutedHandlerCallable[RoutableT, ResultT] = Callable[
        [RoutableT], FlextTypesServices.JsonPayload | ResultT | None
    ]
    type DispatcherRouteEntry[HandlerT, CallableT] = tuple[
        HandlerT,
        CallableT,
        tuple[FlextTypingBase.TypeHintSpecifier, ...],
        bool,
        int,
    ]

    type RegistrablePlugin = (
        FlextTypesServices.ScalarOrModel
        | Callable[..., FlextTypesServices.ScalarOrModel]
    )
    type SortableObjectType = str | int | float
    type ValueAdapter[T] = FlextTypesPydantic.TypeAdapter[T]
    type MessageTypeSpecifier = type | str | UnionType | GenericAlias | TypeAliasType
    type IncEx = AbstractSet[str] | Mapping[str, AbstractSet[str] | bool]

    type ConfigurationMapping = Mapping[str, FlextTypingBase.Scalar]
    type MutableConfigurationMapping = MutableMapping[str, FlextTypingBase.Scalar]
    type ScopedContainerRegistry = MutableMapping[
        str, FlextTypingBase.MutableJsonMapping
    ]
    type SettingsClass[T] = type[T]

    type LazyScalar = FlextTypingBase.Scalar | bytes | date | time
    type LazyCollection = FlextTypingBase.JsonMapping | FlextTypingBase.JsonList
    type ModuleExportValue = FlextTypesPydantic.JsonValue | bytes | date | time
    type ModuleExport = (
        FlextTypesServices.ModuleExportValue
        | FlextTypesServices.LazyCollection
        | ModuleType
        | type[BaseException | Enum]
        | Callable[..., FlextTypesPydantic.JsonValue]
    )
    type LazyGetattr = Callable[[str], FlextTypesServices.ModuleExport]
    type LazyDir = Callable[[], Sequence[str]]

    type ValidatorCallable = Callable[
        [FlextTypesServices.ScalarOrModel], FlextTypesServices.ScalarOrModel
    ]
    type MapperCallable = Callable[
        [FlextTypesPydantic.JsonValue], FlextTypesPydantic.JsonValue
    ]
    type MapperInput = FlextTypesServices.MapperCallable | FlextTypesPydantic.JsonValue
    type StrictValue = (
        FlextTypingBase.Scalar
        | FlextTypesServices.ConfigurationMapping
        | FlextTypingBase.JsonList
        | tuple[FlextTypesPydantic.JsonValue | FlextTypingBase.Scalar, ...]
    )
    type PaginationMeta = Mapping[str, int | bool]

    type GuardInput[T] = (
        type[BaseException | Enum]
        | AbstractSet[FlextTypingBase.Scalar]
        | FlextTypesServices.RegisterableService[T]
        | bytearray
        | bytes
        | Path
        | FlextTypingBase.Scalar
        | FlextTypingBase.JsonMapping
        | Enum
        | frozenset[str]
        | GenericAlias
        | FlextTypesServices.JsonPayload
        | ModuleType
        | T
        | tuple[FlextTypesPydantic.JsonValue, ...]
        | FlextTypingBase.VariadicTuple[type]
        | type
        | TypeAliasType
        | tzinfo
        | UnionType
    )


__all__: list[str] = ["FlextTypesServices"]
