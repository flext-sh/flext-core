"""FlextConstantsFile - file system directory constants (SSOT).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final


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

    # Workspace / project file names — canonical SSOT for path resolution
    PYPROJECT_FILENAME: Final[str] = "pyproject.toml"
    GIT_DIR_NAME: Final[str] = ".git"
    POETRY_LOCK_FILENAME: Final[str] = "poetry.lock"
