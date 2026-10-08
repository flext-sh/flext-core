"""Runtime-checkable structural typing protocols - Thin MRO facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._protocols import (
    FlextProtocolsBase,
    FlextProtocolsConfig,
    FlextProtocolsContainer,
    FlextProtocolsContext,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsProjectMetadata,
    FlextProtocolsPydantic,
    FlextProtocolsRegistry,
    FlextProtocolsResult,
    FlextProtocolsService,
    FlextProtocolsSettings,
)


class FlextProtocols(
    FlextProtocolsBase,
    FlextProtocolsConfig,
    FlextProtocolsContext,
    FlextProtocolsResult,
    FlextProtocolsSettings,
    FlextProtocolsContainer,
    FlextProtocolsService,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsProjectMetadata,
    FlextProtocolsPydantic,
    FlextProtocolsRegistry,
):
    """Runtime-checkable structural typing protocols for FLEXT framework."""


p = FlextProtocols

__all__: list[str] = ["FlextProtocols", "p"]
