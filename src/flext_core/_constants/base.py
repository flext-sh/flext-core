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

    HTTP_STATUS_MIN: Final[int] = 100
    HTTP_STATUS_MAX: Final[int] = 599
    TYPE_MISMATCH: Final[str] = "Type mismatch"
    MAX_MESSAGE_LENGTH: Final[int] = 100
    DEFAULT_MIDDLEWARE_ORDER: Final[int] = 0

    DEFAULT_MAX_CACHE_SIZE: Final[int] = 100
    DEFAULT_BATCH_SIZE: Final[int] = 1000
    DEFAULT_MAX_RETRY_ATTEMPTS: Final[int] = 3
    DEFAULT_WORKERS: Final[int] = 4
    MAX_NAME_LENGTH: Final[int] = 100
    MAX_OPERATION_NAME_LENGTH: Final[int] = 100
    MAX_RETRY_COUNT_VALIDATION: Final[int] = 10
    MAX_WORKERS_VALIDATION: Final[int] = 100
    ZERO: Final[int] = 0
    EXPECTED_TUPLE_LENGTH: Final[int] = 2
    DEFAULT_FAILURE_THRESHOLD: Final[int] = 5
    PREVIEW_LENGTH: Final[int] = 50
    IDENTIFIER_LENGTH: Final[int] = 12
    MAX_BATCH_SIZE_LIMIT: Final[int] = 10000
    DEFAULT_BACKOFF_MULTIPLIER: Final[float] = 2.0
