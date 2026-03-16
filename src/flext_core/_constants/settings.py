"""FlextConstantsSettings - utilities, settings, and security constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final


class FlextConstantsSettings:
    """Constants for utilities and settings."""

    class Utilities:
        """Utility constants."""

        DEFAULT_ENCODING: Final[str] = "utf-8"
        "Default encoding for string operations."
        SERIALIZATION_ISO8601: Final = "iso8601"
        "ISO8601 format for datetime serialization."
        SERIALIZATION_FLOAT: Final = "float"
        "Float format for datetime serialization."
        SERIALIZATION_BASE64: Final = "base64"
        "Base64 format for bytes serialization."
        SERIALIZATION_UTF8: Final = "utf8"
        "UTF-8 format for bytes serialization."
        SERIALIZATION_HEX: Final = "hex"
        "Hex format for bytes serialization."
        MAX_TIMEOUT_SECONDS: Final[int] = 3600
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
            "Convert to string."
            TO_STR_LIST = "to_str_list"
            "Convert to list of strings."
            NORMALIZE = "normalize"
            "Normalize value."
            JOIN = "join"
            "Join values."

    class Settings:
        """Configuration defaults."""

        MAX_WORKERS_THRESHOLD: Final[int] = 50
        DEFAULT_ENABLE_CACHING: Final[bool] = True
        DEFAULT_ENABLE_TRACING: Final[bool] = False
        DEFAULT_TIMEOUT: Final[int] = 30
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

    class ModelConfig:
        """Pydantic model configuration defaults."""

        EXTRA_FORBID: Final = "forbid"
        "Extra fields behavior: forbid unknown fields."
        EXTRA_IGNORE: Final = "ignore"
        "Extra fields behavior: ignore unknown fields."
        EXTRA_ALLOW: Final = "allow"
        "Extra fields behavior: allow unknown fields."

    class Security:
        """Security constants."""

        JWT_DEFAULT_ALGORITHM: Final[str] = "HS256"
        CREDENTIAL_BCRYPT_ROUNDS: Final[int] = 12
