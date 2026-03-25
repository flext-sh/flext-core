"""FlextConstantsPlatform - platform, performance, and reliability constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final

from flext_core import FlextConstantsBase


class FlextConstantsPlatform:
    """Constants for platform and operational constraints."""

    ENV_PREFIX: Final[str] = "FLEXT_"
    ENV_FILE_DEFAULT: Final[str] = ".env"
    ENV_FILE_ENV_VAR: Final[str] = "FLEXT_ENV_FILE"
    ENV_NESTED_DELIMITER: Final[str] = "__"
    DEFAULT_APP_NAME: Final[str] = "flext"
    FLEXT_API_PORT: Final[int] = 8000
    DEFAULT_HOST: Final[str] = "localhost"
    DEFAULT_HTTP_PORT: Final[int] = 80
    MIME_TYPE_JSON: Final[str] = "application/json"
    PATTERN_EMAIL: Final[str] = (
        "^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"
    )
    PATTERN_URL: Final[str] = (
        "^https?://(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\\.)+[A-Z]{2,6}\\.?|localhost|\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3})(?::\\d+)?"
    )
    PATTERN_PHONE_NUMBER: Final[str] = "^\\+?[\\d\\s\\-\\(\\)]{10,20}$"

    MAX_RETRY_ATTEMPTS: Final[int] = 3
    # DEFAULT_TIMEOUT inherited from FlextConstantsBase via FlextConstants MRO
    CIRCUIT_BREAKER_THRESHOLD: Final[int] = 5
    HEADER_REQUEST_ID: Final[str] = "X-Request-ID"

    PATTERN_UUID: Final[str] = (
        "^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$"
    )
    PATTERN_PATH: Final[str] = '^[^<>"|?*\\x00-\\x1F]+$'
    PATTERN_IDENTIFIER: Final[str] = "^[a-zA-Z][a-zA-Z0-9_]*$"
    "Pattern for valid identifiers (handler names, resource types, etc.)."
    PATTERN_IDENTIFIER_WITH_UNDERSCORE: Final[str] = "^[a-zA-Z_][a-zA-Z0-9_]*$"
    "Pattern for identifiers that can start with underscore (context keys)."
    PATTERN_SIMPLE_IDENTIFIER: Final[str] = "^[a-zA-Z0-9]+$"
    "Pattern for simple alphanumeric identifiers."
    PATTERN_MODULE_PATH: Final[str] = "^[^:]+:[^:]+$"
    "Pattern for module:class paths (e.g., 'flext_core.dispatcher:FlextDispatcher')."
    PATTERN_ISO8601_TIMESTAMP: Final[str] = (
        "^(\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}[Z+\\-][0-9:]*)?$"
    )
    "Pattern for ISO 8601 timestamps (optional, allows empty string)."
    PATTERN_DN_STRING: Final[str] = "^(cn|ou|dc)=.*"
    "Pattern for LDAP DN strings (distinguished names)."
    EXT_PYTHON: Final[str] = ".py"
    EXT_YAML: Final[str] = ".yaml"
    EXT_JSON: Final[str] = ".json"
    EXT_TOML: Final[str] = ".toml"
    EXT_XML: Final[str] = ".xml"
    EXT_TXT: Final[str] = ".txt"
    EXT_MD: Final[str] = ".md"
    DIR_CONFIG: Final[str] = "config"
    DIR_PLUGINS: Final[str] = "plugins"
    DIR_LOGS: Final[str] = "logs"
    DIR_DATA: Final[str] = "data"
    DIR_TEMP: Final[str] = "temp"

    DEFAULT_DB_POOL_SIZE: Final[int] = 10
    MIN_DB_POOL_SIZE: Final[int] = 1
    MAX_DB_POOL_SIZE: Final[int] = 100
    MAX_RETRY_ATTEMPTS_LIMIT: Final[int] = 10
    DEFAULT_TIMEOUT_LIMIT: Final[int] = 300
    MIN_CURRENT_STEP: Final[int] = 0
    DEFAULT_INITIAL_DELAY_SECONDS: Final[float] = 1.0
    MAX_BATCH_SIZE: Final[int] = 10000
    DEFAULT_TIME_RANGE_SECONDS: Final[int] = 3600
    DEFAULT_TTL_SECONDS: Final[int] = 3600
    DEFAULT_VERSION: Final[int] = 1
    MIN_VERSION: Final[int] = 1
    # DEFAULT_PAGE_SIZE inherited from FlextConstantsBase via FlextConstants MRO
    HIGH_MEMORY_THRESHOLD_BYTES: Final[int] = 1073741824
    PLATFORM_MAX_TIMEOUT_SECONDS: Final[int] = (
        FlextConstantsBase.MAX_TIMEOUT_SECONDS_PERFORMANCE
    )
    MAX_BATCH_OPERATIONS: Final[int] = 1000
    # MAX_OPERATION_NAME_LENGTH inherited from FlextConstantsBase via FlextConstants MRO
    # EXPECTED_TUPLE_LENGTH inherited from FlextConstantsBase via FlextConstants MRO
    DEFAULT_EMPTY_STRING: Final[str] = ""

    DEFAULT_SIZE: Final[int] = 1000
    MAX_ITEMS: Final[int] = 10000
    MAX_VALIDATION_SIZE: Final[int] = 1000

    CLI_PERFORMANCE_CRITICAL_MS: Final[float] = 10000.0
    RECENT_THRESHOLD_MINUTES: Final[float] = 60.0
    VERY_RECENT_THRESHOLD_MINUTES: Final[float] = 5.0
    RECENT_THRESHOLD_SECONDS: Final[float] = 120.0
    VERSION_LOW_THRESHOLD: Final[int] = 5
    VERSION_MEDIUM_THRESHOLD: Final[int] = 20
    HEALTH_CHECK_STALE_MINUTES: Final[float] = 5.0
    FAILURE_RATE_WARNING_THRESHOLD: Final[float] = 0.25

    DEFAULT_MAX_RETRIES: Final[int] = 3
    DEFAULT_RETRY_DELAY_SECONDS: Final[int] = 1
    RETRY_BACKOFF_BASE: Final[float] = 2.0
    RETRY_BACKOFF_MAX: Final[float] = 60.0
    PLATFORM_DEFAULT_MAX_DELAY_SECONDS: Final[float] = 300.0
    "Default maximum delay in seconds for retry operations."
    RETRY_COUNT_MIN: Final[int] = 1
    DEFAULT_BACKOFF_STRATEGY: Final[str] = "exponential"
    BACKOFF_STRATEGY_EXPONENTIAL: Final[str] = "exponential"
    "Exponential backoff strategy."
    BACKOFF_STRATEGY_LINEAR: Final[str] = "linear"
    "Linear backoff strategy."
    PLATFORM_DEFAULT_FAILURE_THRESHOLD: Final[int] = 5
    DEFAULT_RECOVERY_TIMEOUT: Final[int] = (
        FlextConstantsBase.DEFAULT_RECOVERY_TIMEOUT_SECONDS
    )
    # DEFAULT_TIMEOUT_SECONDS inherited from FlextConstantsBase via FlextConstants MRO
    DEFAULT_RATE_LIMIT_WINDOW_SECONDS: Final[int] = 60
    DEFAULT_RATE_LIMIT_MAX_REQUESTS: Final[int] = 100
    DEFAULT_CIRCUIT_BREAKER_THRESHOLD: Final[int] = 5
    DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT: Final[int] = (
        FlextConstantsBase.DEFAULT_RECOVERY_TIMEOUT_SECONDS
    )
    DEFAULT_CIRCUIT_BREAKER_SUCCESS_THRESHOLD: Final[int] = 3

    @unique
    class CircuitBreakerState(StrEnum):
        """Circuit breaker states.

        DRY Pattern:
            StrEnum is the single source of truth. Use CircuitBreakerState.CLOSED.value
            or CircuitBreakerState.CLOSED directly - no base strings needed.
        """

        CLOSED = "closed"
        OPEN = "open"
        HALF_OPEN = "half_open"
