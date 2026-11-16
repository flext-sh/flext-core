"""Facade for FlextUtilitiesLoggingConfig.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._logging import FlextUtilitiesLoggingComposition


class FlextUtilitiesLoggingConfig(FlextUtilitiesLoggingComposition):
    """Compose logging configuration through its single-responsibility parts."""


__all__: list[str] = ["FlextUtilitiesLoggingConfig"]
