"""FlextProtocolsContext - context and bootstrap protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._protocols._context import flextprotocolscontext_part_03


class FlextProtocolsContext(flextprotocolscontext_part_03.FlextProtocolsContextPart03):
    """Public facade for FlextProtocolsContext."""


__all__: list[str] = ["FlextProtocolsContext"]
