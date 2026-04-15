"""FlextConstantsSettings - utilities, settings, and security constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final


class FlextConstantsSettings:
    """Constants for utilities and settings."""

    DEFAULT_ENCODING: Final[str] = "utf-8"

    @unique
    class Serialization(StrEnum):
        """Datetime and bytes serialization format identifiers."""

        ISO8601 = "iso8601"
        FLOAT = "float"
        BASE64 = "base64"
        UTF8 = "utf8"
        HEX = "hex"

    LONG_UUID_LENGTH: Final[int] = 12
    SHORT_UUID_LENGTH: Final[int] = 8
    VERSION_MODULO: Final[int] = 100
    CONTROL_CHARS_PATTERN: Final[str] = "[\\x00-\\x1F\\x7F]"
    CACHE_ATTRIBUTE_NAMES: Final[tuple[str, ...]] = (
        "_cache",
        "_ttl",
        "_cached_at",
        "_cached_value",
    )

    @unique
    class ConversionMode(StrEnum):
        """Conversion mode enumeration for type-safe conversion operations.

        DRY Pattern: StrEnum provides single source of truth for conversion modes.
        Use ConversionMode.TO_STR.value or ConversionMode.TO_STR directly.
        """

        TO_STR = "to_str"
        TO_STR_LIST = "to_str_list"
        NORMALIZE = "normalize"
        JOIN = "join"

    MAX_WORKERS_THRESHOLD: Final[int] = 50
    DEFAULT_ENABLE_CACHING: Final[bool] = True
    DEFAULT_ENABLE_TRACING: Final[bool] = False
    # DEFAULT_TIMEOUT inherited from FlextConstantsBase via FlextConstants MRO
    DEFAULT_DEBUG_MODE: Final[bool] = False
    DEFAULT_TRACE_MODE: Final[bool] = False

    @unique
    class LogLevel(StrEnum):
        """Standard log levels."""

        DEBUG = "DEBUG"
        INFO = "INFO"
        WARNING = "WARNING"
        ERROR = "ERROR"
        CRITICAL = "CRITICAL"

    @unique
    class Environment(StrEnum):
        """Environment types."""

        DEVELOPMENT = "development"
        STAGING = "staging"
        PRODUCTION = "production"
        TESTING = "testing"
        LOCAL = "local"

    @unique
    class ExtraConfig(StrEnum):
        """Pydantic model_config extra= field behavior identifiers."""

        FORBID = "forbid"
        IGNORE = "ignore"
        ALLOW = "allow"

    EXTRA_CONFIG_FORBID: Final = "forbid"
    EXTRA_CONFIG_IGNORE: Final = "ignore"
    SERIALIZATION_ISO8601: Final = "iso8601"
    SERIALIZATION_BASE64: Final = "base64"

    JWT_DEFAULT_ALGORITHM: Final[str] = "HS256"
    CREDENTIAL_BCRYPT_ROUNDS: Final[int] = 12
