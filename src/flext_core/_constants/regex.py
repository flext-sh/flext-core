"""FlextConstantsRegex - regex pattern constants (SSOT).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final


class FlextConstantsRegex:
    """SSOT for regex pattern string constants used across the workspace."""

    PATTERN_IDENTIFIER_WITH_UNDERSCORE: Final[str] = "^[a-zA-Z_][a-zA-Z0-9_]*$"
    "Pattern for identifiers that can start with underscore (context keys)."
    PATTERN_ISO8601_TIMESTAMP: Final[str] = (
        "^(\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}[Z+\\-][0-9:]*)?$"
    )
    "Pattern for ISO 8601 timestamps (optional, allows empty string)."
