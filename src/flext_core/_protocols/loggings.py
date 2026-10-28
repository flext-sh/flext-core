"""FlextProtocolsLogging - logging and related infrastructure protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._protocols._logging import FlextProtocolsLoggingPart03


class FlextProtocolsLogging(FlextProtocolsLoggingPart03):
    """Public facade for FlextProtocolsLogging."""


__all__: list[str] = ["FlextProtocolsLogging"]
