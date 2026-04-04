"""Runtime-checkable structural typing protocols - Thin MRO facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    FlextProtocolsBase,
    FlextProtocolsConfig,
    FlextProtocolsContainer,
    FlextProtocolsContext,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsRegistry,
    FlextProtocolsResult,
    FlextProtocolsService,
)
from flext_core._models.namespace import FlextModelsNamespace


class FlextProtocols(
    FlextProtocolsBase,
    FlextProtocolsContext,
    FlextProtocolsResult,
    FlextProtocolsConfig,
    FlextProtocolsContainer,
    FlextProtocolsService,
    FlextProtocolsHandler,
    FlextProtocolsLogging,
    FlextProtocolsRegistry,
    FlextModelsNamespace,
):
    """Runtime-checkable structural typing protocols for FLEXT framework."""


p = FlextProtocols
__all__ = ["FlextProtocols", "p"]
