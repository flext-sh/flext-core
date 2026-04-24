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
    ENV_FILE_DEFAULT: Final[str] = ".env"
    ENV_FILE_ENV_VAR: Final[str] = "FLEXT_ENV_FILE"
    ENV_NESTED_DELIMITER: Final[str] = "__"
    DEFAULT_APP_NAME: Final[str] = "flext"
