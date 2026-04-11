"""FlextConstantsOutput - output/log message authorities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final


class FlextConstantsOutput:
    """Output and logging message constants for core runtime flows."""

    LOG_REGISTERED_AUTO_DISCOVERY_HANDLER: Final[str] = (
        "Registered auto-discovery handler"
    )
    LOG_REGISTERED_EVENT_SUBSCRIBER: Final[str] = "Registered event subscriber"
    LOG_REGISTERED_HANDLER: Final[str] = "Registered handler"


__all__ = ["FlextConstantsOutput"]
