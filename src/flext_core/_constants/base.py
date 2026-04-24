"""FlextConstantsBase - core primitive constants (SSOT, MRO facade for base scalars).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final


class FlextConstantsBase:
    """SSOT for base primitive constants used across the workspace."""

    NAME: Final[str] = "FLEXT"
    ZERO: Final[int] = 0

    PERCENTAGE_MULTIPLIER: Final[int] = 100
    MILLISECONDS_MULTIPLIER: Final[int] = 1000
    MICROSECONDS_MULTIPLIER: Final[int] = 1000000

    LOCALHOST: Final[str] = "localhost"
    LOOPBACK_IP: Final[str] = "127.0.0.1"
    MIN_PORT: Final[int] = 1
    MAX_PORT: Final[int] = 65535
    MAX_HOSTNAME_LENGTH: Final[int] = 253

    HTTP_STATUS_MIN: Final[int] = 100
    HTTP_STATUS_MAX: Final[int] = 599

    DEFAULT_PAGE_SIZE: Final[int] = 10
    MAX_PAGE_SIZE: Final[int] = 1000
    MIN_PAGE_SIZE: Final[int] = 1

    DEFAULT_BACKOFF_MULTIPLIER: Final[float] = 2.0
    DEFAULT_SIZE: Final[int] = 1000
    MAX_ITEMS: Final[int] = 10000
    DEFAULT_EMPTY_STRING: Final[str] = ""
