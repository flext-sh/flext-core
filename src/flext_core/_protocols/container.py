"""FlextProtocolsContainer - dependency injection protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from flext_core import t
from flext_core._protocols import (
    FlextProtocolsContext,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsRegistry,
    FlextProtocolsResult,
    FlextProtocolsService,
    FlextProtocolsSettings,
)
from flext_core._protocols._container import FlextProtocolsContainerPart03


class FlextProtocolsContainer(FlextProtocolsContainerPart03):
    """Public facade for FlextProtocolsContainer."""

    type ProtocolGuardInput = (
        t.JsonPayload
        | t.JsonMapping
        | Sequence[str]
        | t.TypeHintSpecifier
        | Callable[..., t.JsonPayload]
        | FlextProtocolsContainer.Container
        | FlextProtocolsContext.Context
        | FlextProtocolsHandler.Dispatcher
        | FlextProtocolsHandler.Handle
        | FlextProtocolsHandler.Middleware
        | FlextProtocolsLogging.Logger
        | FlextProtocolsResult.Result[t.JsonPayload]
        | FlextProtocolsSettings.Settings
        | FlextProtocolsService.Service[t.JsonPayload]
        | None
    )

    # The generic guard already builds registration values from these leaves.
    type GuardInput = t.GuardInput[
        FlextProtocolsContainer.RegistryCollaborator
        | FlextProtocolsContainer.Container
        | FlextProtocolsHandler.Handle
        | FlextProtocolsHandler.Middleware
        | FlextProtocolsLogging.HasLogger
        | FlextProtocolsResult.HasModelDump
        | FlextProtocolsRegistry.Registry
        | FlextProtocolsResult.Result[FlextProtocolsContainer.ServiceValue]
    ]


__all__: list[str] = ["FlextProtocolsContainer"]
