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
from flext_core._utilities.enforcement import FlextUtilitiesEnforcement


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
):
    """Runtime-checkable structural typing protocols for FLEXT framework."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Enforce protocols governance on subclasses."""
        super().__init_subclass__(**kwargs)
        FlextUtilitiesEnforcement.run_protocols(cls)


p = FlextProtocols
__all__ = ["FlextProtocols", "p"]
