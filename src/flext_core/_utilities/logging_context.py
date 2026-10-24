"""Facade for FlextUtilitiesLoggingContext.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._logging_context.logging_context_part_02 import (
    FlextUtilitiesLoggingContextPart02,
)


class FlextUtilitiesLoggingContext(FlextUtilitiesLoggingContextPart02):
    """Compose logging context through its internal part hierarchy."""


__all__: list[str] = ["FlextUtilitiesLoggingContext"]
