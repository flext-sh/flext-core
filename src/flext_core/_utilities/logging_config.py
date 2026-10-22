"""Facade for FlextUtilitiesLoggingConfig.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._utilities._logging_config.logging_config_part_03 import (
    FlextUtilitiesLoggingConfigPart03,
)


class FlextUtilitiesLoggingConfig(FlextUtilitiesLoggingConfigPart03):
    """Compose logging configuration through its internal part hierarchy."""


__all__: list[str] = ["FlextUtilitiesLoggingConfig"]
