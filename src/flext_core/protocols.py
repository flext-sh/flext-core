# AUTO-GENERATED FILE — Regenerate with: make gen
"""Runtime-checkable structural typing protocols - Thin MRO facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TypeAlias

from flext_core._protocols import (
    FlextProtocolsBase,
    FlextProtocolsCollectionModels,
    FlextProtocolsConfig,
    FlextProtocolsContainer,
    FlextProtocolsContainers,
    FlextProtocolsContext,
    FlextProtocolsCqrs,
    FlextProtocolsDomainEvent,
    FlextProtocolsEntity,
    FlextProtocolsErrors,
    FlextProtocolsExceptionParams,
    FlextProtocolsFlextContext,
    FlextProtocolsFlextMixins,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsOptions,
    FlextProtocolsProjectMetadata,
    FlextProtocolsPydantic,
    FlextProtocolsRegistry,
    FlextProtocolsResult,
    FlextProtocolsService,
    FlextProtocolsSettings,
)


class FlextProtocols(
    FlextProtocolsBase,
    FlextProtocolsCollectionModels,
    FlextProtocolsConfig,
    FlextProtocolsContainer,
    FlextProtocolsContainers,
    FlextProtocolsContext,
    FlextProtocolsCqrs,
    FlextProtocolsDomainEvent,
    FlextProtocolsEntity,
    FlextProtocolsErrors,
    FlextProtocolsExceptionParams,
    FlextProtocolsFlextContext,
    FlextProtocolsFlextMixins,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsOptions,
    FlextProtocolsProjectMetadata,
    FlextProtocolsPydantic,
    FlextProtocolsRegistry,
    FlextProtocolsResult,
    FlextProtocolsService,
    FlextProtocolsSettings,
):
    """Runtime-checkable structural typing protocols for FLEXT framework.

    The ``p`` letter alias binds to this class; protocols compose via MRO so
    consumers can reference any contract as ``p.<Protocol>`` while
    implementations are checked structurally rather than by inheritance.
    """

    HasDomainEvents: TypeAlias = FlextProtocolsBase.HasDomainEvents
    ParseOptions: TypeAlias = FlextProtocolsConfig.ParseOptions
    RootDict: TypeAlias = FlextProtocolsContainer.RootDict
    MutableRootDict: TypeAlias = FlextProtocolsContainer.MutableRootDict
    ContainerType: TypeAlias = FlextProtocolsContainer.ContainerType
    ContextProxy: TypeAlias = FlextProtocolsContext.ContextProxy
    Handler: TypeAlias = FlextProtocolsHandler.Handler
    BaseModel: TypeAlias = FlextProtocolsPydantic.BaseModel
    RootModel: TypeAlias = FlextProtocolsPydantic.RootModel
    ResultState: TypeAlias = FlextProtocolsResult.ResultState
    ResultOps: TypeAlias = FlextProtocolsResult.ResultOps
    Result: TypeAlias = FlextProtocolsResult.Result
    HasModelDump: TypeAlias = FlextProtocolsResult.HasModelDump
    Service: TypeAlias = FlextProtocolsService.Service


p = FlextProtocols
__all__: list[str] = ["FlextProtocols", "p"]
