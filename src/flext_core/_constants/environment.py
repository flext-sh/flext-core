"""FlextConstantsEnvironment - deployment environment + env-var constants (SSOT).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique


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


__all__: tuple[str, ...] = ("FlextConstantsEnvironment",)
