"""Runtime-checkable structural typing protocols - Thin MRO facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._protocols import (
    FlextProtocolsBase,
    FlextProtocolsConfig,
    FlextProtocolsContext,
    FlextProtocolsDI,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsRegistry,
    FlextProtocolsResult,
    FlextProtocolsService,
)


class FlextProtocols(
    FlextProtocolsBase,
    FlextProtocolsContext,
    FlextProtocolsResult,
    FlextProtocolsConfig,
    FlextProtocolsDI,
    FlextProtocolsService,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsRegistry,
):
    """Runtime-checkable structural typing protocols for FLEXT framework."""


p = FlextProtocols
__all__ = ["FlextProtocols", "p"]
