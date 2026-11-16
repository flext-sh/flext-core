"""Facade for FlextUtilitiesLoggingConfig.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._logging.composition import FlextUtilitiesLoggingComposition
from flext_core._utilities._logging.performance import FlextUtilitiesLoggingPerformance


class FlextUtilitiesLoggingConfig(
    FlextUtilitiesLoggingComposition,
    FlextUtilitiesLoggingPerformance,
):
    """Compose logging configuration through its single-responsibility parts."""


__all__: list[str] = ["FlextUtilitiesLoggingConfig"]
