"""Facade for FlextUtilitiesLoggingContext.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._logging.context import FlextUtilitiesLoggingContextBinding


class FlextUtilitiesLoggingContext(FlextUtilitiesLoggingContextBinding):
    """Compose logging context through its single-responsibility part."""


__all__: list[str] = ["FlextUtilitiesLoggingContext"]
