"""FlextProtocolsContainer - dependency injection protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable

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
from flext_core._protocols._container_parts import flextprotocolscontainer_part_03


class FlextProtocolsContainer(flextprotocolscontainer_part_03.FlextProtocolsContainer):
    """Public facade for FlextProtocolsContainer."""

    type ProtocolGuardInput = (
        t.JsonPayload
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

    type GuardInput = t.GuardInput[
        FlextProtocolsContainer.RegisterableService
        | FlextProtocolsContainer.Container
        | FlextProtocolsHandler.Handle
        | FlextProtocolsHandler.Middleware
        | FlextProtocolsLogging.HasLogger
        | FlextProtocolsResult.HasModelDump
        | FlextProtocolsRegistry.Registry
        | FlextProtocolsResult.Result[FlextProtocolsContainer.ServiceValue]
    ]


__all__: list[str] = ["FlextProtocolsContainer"]
