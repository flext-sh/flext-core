"""FlextProtocolsContext - context and bootstrap protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._context_parts.flextprotocolscontext_part_03 import (
    FlextProtocolsContext as FlextProtocolsContextPartFinal,
)


class FlextProtocolsContext(FlextProtocolsContextPartFinal):
    """Public facade for FlextProtocolsContext."""


__all__: list[str] = ["FlextProtocolsContext"]
