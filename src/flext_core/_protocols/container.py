"""FlextProtocolsContainer - dependency injection protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._container import FlextProtocolsContainerOptions, FlextProtocolsContainerRuntime


# mro-wkii.17.26 (codex): compose focused contracts and retire numbered parts.
class FlextProtocolsContainer(
    FlextProtocolsContainerOptions, FlextProtocolsContainerRuntime
):
    """Dependency injection protocol facade."""


__all__: tuple[str, ...] = ("FlextProtocolsContainer",)
