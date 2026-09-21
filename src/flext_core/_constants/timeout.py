"""FlextConstantsTimeout - unified timeout constants (SSOT).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar


class FlextConstantsTimeout:
    """SSOT for all timeout-related constants."""

    DEFAULT_TIMEOUT_SECONDS: ClassVar[int] = 30
    MIN_TIMEOUT_SECONDS: ClassVar[int] = 1
    MAX_TIMEOUT_SECONDS: ClassVar[int] = 3600
    CACHE_TTL: ClassVar[int] = 300
    DEFAULT_RECOVERY_TIMEOUT_SECONDS: ClassVar[int] = 60
    DEFAULT_MAX_DELAY_SECONDS: ClassVar[float] = 60.0
