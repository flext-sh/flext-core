"""FlextConstantsSettings - utilities and settings constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final

from flext_core import t


class FlextConstantsSettings:
    """Constants for utilities and settings."""

    LONG_UUID_LENGTH: Final[int] = 12
    SHORT_UUID_LENGTH: Final[int] = 8
    VERSION_MODULO: Final[int] = 100
    CACHE_ATTRIBUTE_NAMES: Final[t.VariadicTuple[str]] = (
        "_cache",
        "_ttl",
        "_cached_at",
        "_cached_value",
    )

    MAX_WORKERS_THRESHOLD: Final[int] = 50
    DEFAULT_ENABLE_CACHING: Final[bool] = True
    DEFAULT_ENABLE_TRACING: Final[bool] = False
    DEFAULT_DEBUG_MODE: Final[bool] = False
    DEFAULT_TRACE_MODE: Final[bool] = False

    EXTRA_CONFIG_FORBID: Final = "forbid"
    EXTRA_CONFIG_IGNORE: Final = "ignore"
    SERIALIZATION_ISO8601: Final = "iso8601"
    SERIALIZATION_BASE64: Final = "base64"
