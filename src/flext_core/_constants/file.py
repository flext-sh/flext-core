"""FlextConstantsFile - file system directory constants (SSOT).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique


class FlextConstantsFile:
    """SSOT for file system directory name identifiers."""

    @unique
    class Directory(StrEnum):
        """Standard directory name identifiers."""

        CONFIG = "settings"
        PLUGINS = "plugins"
        LOGS = "logs"
        DATA = "data"
        TEMP = "temp"
