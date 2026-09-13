"""FlextConstantsEnvironment - deployment environment + env-var constants (SSOT).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final


class FlextConstantsEnvironment:
    """SSOT for environment tier + env-var configuration."""

    @unique
    class Environment(StrEnum):
        """Deployment environment types."""

        DEVELOPMENT = "development"
        STAGING = "staging"
        PRODUCTION = "production"
        TESTING = "testing"
        LOCAL = "local"

    ENV_PREFIX: Final[str] = "FLEXT_"
    """Root env prefix invariant (consumed forward by m for derived defaults).

    ``ENV_FILE_ENV_VAR``/``ENV_FILE_DEFAULT`` moved to their chain-law owner
    ``_settings.py`` (settings is the bottom layer); consume them via
    ``FlextSettings.ENV_FILE_*``. ``DEFAULT_APP_NAME``/``DEFAULT_TIMEZONE``
    were deleted: zero consumers (YAGNI).
    """
