"""FlextProtocolsContainer - dependency injection protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._protocols._container_parts import flextprotocolscontainer_part_03


class FlextProtocolsContainer(flextprotocolscontainer_part_03.FlextProtocolsContainer):
    """Public facade for FlextProtocolsContainer."""


__all__: list[str] = ["FlextProtocolsContainer"]
