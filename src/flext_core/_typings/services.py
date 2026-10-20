# AUTO-GENERATED FILE — regenerate through `make gen` (source of truth: flext-infra templates)
"""FlextTypesServices - service, mapping, and runtime helper type aliases.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Generator,
    Iterator,
    MutableMapping,
    Set as AbstractSet,
)
from concurrent.futures import Future
from datetime import date, time, tzinfo
from enum import Enum
from logging import Formatter, Handler, LogRecord
from pathlib import Path
from types import GenericAlias, ModuleType, TracebackType, UnionType
from typing import Any, BinaryIO, TextIO, TypeAliasType

from flext_core import c
from flext_core._typings import FlextTypesPydantic, FlextTypingBase


class FlextTypesServices:
    """Type aliases for service registration and runtime mappings."""

    type JsonPayload = FlextTypesPydantic.JsonValue

    type ServiceLeaf[T] = (
        FlextTypingBase.Scalar | FlextTypesPydantic.JsonValue | Path | T
    )
    type ServiceValue[T] = (
        FlextTypesServices.ServiceLeaf[T]
        | FlextTypingBase.JsonMapping
        | FlextTypingBase.JsonList
    )
    type RegisterableService[T] = (
        FlextTypesServices.ServiceValue[T] | Callable[..., Any]
    )
    type FactoryCallable[T] = Callable[..., Any]
    type ResourceCallable[T] = Callable[..., Any]
    type ServiceMap[T] = FlextTypingBase.MappingKV[
        str, FlextTypesServices.RegisterableService[T]
    ]
    type FactoryMap[T] = FlextTypingBase.MappingKV[
        str, FlextTypesServices.FactoryCallable[T]
    ]
    type ResourceMap[T] = FlextTypingBase.MappingKV[
        str, FlextTypesServices.ResourceCallable[T]
    ]

    type OperationMetricsStream = Generator[FlextTypingBase.JsonMapping]

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
    type SettingsOverridesMapping[T] = FlextTypingBase.MappingKV[
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
    type LogArgument[T] = FlextTypesServices.JsonPayload | T
    type LogValue[T] = (
        FlextTypesServices.LogArgument[T]
        | BaseException
        | FlextTypesServices.LoggingExceptionInfo
    )
    type LoggingNativeRecord = LogRecord
    type LoggingNativeFormatter = Formatter
    type LoggingNativeHandlers = list[Handler]
    type LoggingEventValue[T] = (
        FlextTypesServices.LogValue[T]
        | type[BaseException]
        | FlextTypesServices.LoggingTraceback
        | tuple[FlextTypesServices.LogValue[T], ...]
    )
    type LoggingEvent[T] = MutableMapping[str, FlextTypesServices.LoggingEventValue[T]]
    type LoggingContext[T] = FlextTypingBase.DictKV[
        str, FlextTypesServices.LoggingEventValue[T]
    ]
    type LoggingWriteCompletion = Future[int]

    type TextStream = TextIO
    type BinaryStream = BinaryIO
    type TextStreamIterator = Iterator[str]

    type MetadataMapping = FlextTypingBase.MappingKV[
        str, FlextTypesServices.JsonPayload
    ]
    type MutableMetadataMapping = MutableMapping[str, FlextTypesServices.JsonPayload]
    type RuntimeData = FlextTypesServices.JsonPayload
    type BootstrapInput = FlextTypesPydantic.BaseModelType | FlextTypingBase.JsonMapping
    type ServiceClass = type
    type UserOverridesMapping = FlextTypingBase.MappingKV[
        str, FlextTypesServices.JsonPayload
    ]

    type ModelInput[T] = (
        FlextTypesPydantic.JsonValue
        | T
        | FlextTypingBase.MappingKV[str, FlextTypesPydantic.JsonValue]
    )
    type ConfigModelInput[T] = (
        T | FlextTypingBase.MappingKV[str, FlextTypesPydantic.JsonValue]
    )
    type MetadataInput[T] = (
        T | FlextTypingBase.MappingKV[str, FlextTypesServices.JsonPayload | None] | None
    )

    type ContextHookCallable = Callable[
        [FlextTypingBase.Scalar], FlextTypesServices.JsonPayload
    ]
    type ContextHookMap = FlextTypingBase.MappingKV[
        str, FlextTypingBase.SequenceOf[ContextHookCallable]
    ]

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
    type IncEx = (
        AbstractSet[str] | FlextTypingBase.MappingKV[str, AbstractSet[str] | bool]
    )

    type ConfigurationMapping = FlextTypingBase.MappingKV[str, FlextTypingBase.Scalar]
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
        | Callable[..., Any]
    )
    type LazyGetattr = Callable[[str], FlextTypesServices.ModuleExport]
    type LazyDir = Callable[[], FlextTypingBase.SequenceOf[str]]

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
    type PaginationMeta = FlextTypingBase.MappingKV[str, int | bool]

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
