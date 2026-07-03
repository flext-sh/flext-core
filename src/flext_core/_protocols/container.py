"""FlextProtocolsContainer - dependency injection protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._container_parts.flextprotocolscontainer_part_03 import (
    FlextProtocolsContainer as FlextProtocolsContainerPartFinal,
)


class FlextProtocolsContainer(FlextProtocolsContainerPartFinal):
    """Public facade for FlextProtocolsContainer."""


__all__: list[str] = ["FlextProtocolsContainer"]
