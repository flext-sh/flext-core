"""FlextConstantsTimeout - unified timeout constants (SSOT).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final


class FlextConstantsTimeout:
    """SSOT for all timeout-related constants."""

    DEFAULT_TIMEOUT_SECONDS: Final[int] = 30
    MIN_TIMEOUT_SECONDS: Final[int] = 1
    MAX_TIMEOUT_SECONDS: Final[int] = 3600
    CACHE_TTL: Final[int] = 300
    DEFAULT_RECOVERY_TIMEOUT_SECONDS: Final[int] = 60
    DEFAULT_MAX_DELAY_SECONDS: Final[float] = 60.0
