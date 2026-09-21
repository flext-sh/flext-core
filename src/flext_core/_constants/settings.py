"""FlextConstantsSettings - runtime settings constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final

from pydantic import ConfigDict


class FlextConstantsSettings:
    """SSOT for runtime settings."""

    SHORT_UUID_LENGTH: Final[int] = 8

    ENV_FILE_ENV_VAR: Final[str] = "FLEXT_ENV_FILE"
    """Bootstrap env var that overrides the .env path (settings protocol owner)."""

    EXTRA_CONFIG_FORBID: Final = "forbid"
    EXTRA_CONFIG_IGNORE: Final = "ignore"
    SERIALIZATION_ISO8601: Final = "iso8601"
    SERIALIZATION_BASE64: Final = "base64"

    DOMAIN_MODEL_CONFIG: Final[ConfigDict] = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        validate_return=True,
        validate_default=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=False,
        extra="forbid",
    )
    """Domain model configuration defaults (SSOT; consumed via ``c.*``)."""
