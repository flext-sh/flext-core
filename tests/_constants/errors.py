"""Constants mixin for errors.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from typing import Final


class TestsFlextConstantsErrors:
    """Flat error and validation constants for flext-core tests."""

    USER_NOT_FOUND: Final[str] = "User not found"
    INVALID_EMAIL: Final[str] = "Invalid email address"
    VALUE_TOO_LOW: Final[str] = "Value must be positive"
    VALUE_TOO_HIGH: Final[str] = "Value must be <= 100"
    TEST_ERROR: Final[str] = "Test error"
    DISPATCHER_UNCONFIGURED: Final[str] = "dispatcher-unconfigured"
    DISPATCHER_FAIL: Final[str] = "dispatcher-fail"
    NO_USER_IDS_PROVIDED: Final[str] = "No user IDs provided"
    SUBCLASSES_MUST_IMPLEMENT_EXECUTE: Final[str] = (
        "Subclasses must implement execute()"
    )
    HANDLER_ID_CANNOT_BE_EMPTY: Final[str] = "Handler ID cannot be empty"
    PROCESSING_ERROR_DEFAULT: Final[str] = "Processing error"
    BAD_DICT_GET: Final[str] = "Bad dict get"
    BAD_LIST_ITERATION: Final[str] = "Bad list iteration"
    PLAIN_BOOM: Final[str] = "plain boom"
    CANNOT_INSTANTIATE: Final[str] = "Cannot instantiate"
    UNEXPECTED_MESSAGE_TYPE: Final[str] = "Unexpected message type"
    VALIDATION_FAILED_FOR_TEST: Final[str] = "Validation failed for test"
    ACTIVE_ERROR_MESSAGES: Final[frozenset[str]] = frozenset({
        USER_NOT_FOUND,
        INVALID_EMAIL,
        VALUE_TOO_LOW,
        VALUE_TOO_HIGH,
        TEST_ERROR,
        DISPATCHER_UNCONFIGURED,
        DISPATCHER_FAIL,
        NO_USER_IDS_PROVIDED,
        SUBCLASSES_MUST_IMPLEMENT_EXECUTE,
        HANDLER_ID_CANNOT_BE_EMPTY,
        PROCESSING_ERROR_DEFAULT,
        BAD_DICT_GET,
        BAD_LIST_ITERATION,
        PLAIN_BOOM,
        CANNOT_INSTANTIATE,
        UNEXPECTED_MESSAGE_TYPE,
        VALIDATION_FAILED_FOR_TEST,
    })

    MIN_LENGTH_DEFAULT: Final[int] = 3
    MAX_VALUE: Final[int] = 100
    ACTIVE_ERROR_MESSAGES_RE: Final[re.Pattern[str]] = re.compile(
        "|".join(re.escape(message) for message in ACTIVE_ERROR_MESSAGES),
        re.IGNORECASE,
    )
