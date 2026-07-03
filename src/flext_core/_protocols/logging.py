"""FlextProtocolsLogging - logging and related infrastructure protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._logging_parts.flextprotocolslogging_part_03 import (
    FlextProtocolsLogging as FlextProtocolsLoggingPartFinal,
)


class FlextProtocolsLogging(FlextProtocolsLoggingPartFinal):
    """Public facade for FlextProtocolsLogging."""


__all__: list[str] = ["FlextProtocolsLogging"]
