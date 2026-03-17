"""FlextConstantsBase - root and foundational constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final


class FlextConstantsBase:
    """Constants for root and foundational values."""

    NAME: Final[str] = "FLEXT"
    INITIAL_TIME: Final[float] = 0.0
    PERCENTAGE_MULTIPLIER: Final[int] = 100
    "Multiplier for percentage calculations (100 = 100%)."
    MILLISECONDS_MULTIPLIER: Final[int] = 1000
    "Multiplier to convert seconds to milliseconds."
    MICROSECONDS_MULTIPLIER: Final[int] = 1000000
    "Multiplier to convert seconds to microseconds."

    class Network:
        """Network configuration constants and limits."""

        LOOPBACK_IP: Final[str] = "127.0.0.1"
        LOCALHOST: Final[str] = "localhost"
        MIN_PORT: Final[int] = 1
        MAX_PORT: Final[int] = 65535
        DEFAULT_TIMEOUT: Final[int] = 30
        DEFAULT_CONNECTION_POOL_SIZE: Final[int] = 10
        MAX_CONNECTION_POOL_SIZE: Final[int] = 100
        MAX_HOSTNAME_LENGTH: Final[int] = 253
        HTTP_STATUS_MIN: Final[int] = 100
        HTTP_STATUS_MAX: Final[int] = 599

    class Messages:
        """User-facing message templates."""

        TYPE_MISMATCH: Final[str] = "Type mismatch"

    class Defaults:
        """Default values."""

        TIMEOUT: Final[int] = 30
        PAGE_SIZE: Final[int] = 100
        TIMEOUT_SECONDS: Final[int] = 30
        CACHE_TTL: Final[int] = 300
        DEFAULT_CACHE_TTL: Final[int] = CACHE_TTL
        DEFAULT_MAX_CACHE_SIZE: Final[int] = 100
        MAX_MESSAGE_LENGTH: Final[int] = 100
        DEFAULT_MIDDLEWARE_ORDER: Final[int] = 0
        OPERATION_TIMEOUT_SECONDS: Final[int] = 30
        DATABASE_URL: Final[str] = "sqlite:///:memory:"
        DEFAULT_DATABASE_URL: Final[str] = DATABASE_URL

    DEFAULT_TIMEOUT_SECONDS: Final[int] = 30
    MAX_TIMEOUT_SECONDS: Final[int] = 3600
    MIN_TIMEOUT_SECONDS: Final[int] = 1
    DEFAULT_MAX_CACHE_SIZE: Final[int] = 100
    DEFAULT_BATCH_SIZE: Final[int] = 1000
    DEFAULT_PAGE_SIZE: Final[int] = 10
    MAX_PAGE_SIZE: Final[int] = 1000
    MIN_PAGE_SIZE: Final[int] = 1
    DEFAULT_MAX_RETRY_ATTEMPTS: Final[int] = 3
    DEFAULT_WORKERS: Final[int] = 4
    DEFAULT_POOL_SIZE: Final[int] = 10
    MAX_POOL_SIZE: Final[int] = 100
    MIN_POOL_SIZE: Final[int] = 1
    MAX_NAME_LENGTH: Final[int] = 100
    MAX_OPERATION_NAME_LENGTH: Final[int] = 100
    MAX_PORT_NUMBER: Final[int] = 65535
    MIN_PORT_NUMBER: Final[int] = 1
    MAX_TIMEOUT_VALIDATION_SECONDS: Final[int] = 300
    MAX_RETRY_COUNT_VALIDATION: Final[int] = 10
    MAX_HOSTNAME_LENGTH_VALIDATION: Final[int] = 253
    MAX_WORKERS_VALIDATION: Final[int] = 100
    ZERO: Final[int] = 0
    EXPECTED_TUPLE_LENGTH: Final[int] = 2
    DEFAULT_FAILURE_THRESHOLD: Final[int] = 5
    PREVIEW_LENGTH: Final[int] = 50
    DEFAULT_RECOVERY_TIMEOUT_SECONDS: Final[int] = 60
    IDENTIFIER_LENGTH: Final[int] = 12
    MAX_BATCH_SIZE_LIMIT: Final[int] = 10000
    DEFAULT_BACKOFF_MULTIPLIER: Final[float] = 2.0
    DEFAULT_MAX_DELAY_SECONDS: Final[float] = 60.0
    MAX_TIMEOUT_SECONDS_PERFORMANCE: Final[int] = 600
    DEFAULT_HOUR_IN_SECONDS: Final[int] = 3600
