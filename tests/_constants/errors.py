"""Constants mixin for errors.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar


class TestsFlextConstantsErrors:
    """Flat error and validation constants for flext-core tests."""

    USER_NOT_FOUND: ClassVar[str] = "User not found"
    INVALID_EMAIL: ClassVar[str] = "Invalid email address"
    VALUE_TOO_LOW: ClassVar[str] = "Value must be positive"
    VALUE_TOO_HIGH: ClassVar[str] = "Value must be <= 100"
    TEST_ERROR: ClassVar[str] = "Test error"
    NO_USER_IDS_PROVIDED: ClassVar[str] = "No user IDs provided"
    SUBCLASSES_MUST_IMPLEMENT_EXECUTE: ClassVar[str] = (
        "Subclasses must implement execute()"
    )
    HANDLER_ID_CANNOT_BE_EMPTY: ClassVar[str] = "Handler ID cannot be empty"
    PROCESSING_ERROR_DEFAULT: ClassVar[str] = "Processing error"
    BAD_DICT_GET: ClassVar[str] = "Bad dict get"
    BAD_LIST_ITERATION: ClassVar[str] = "Bad list iteration"
    CANNOT_INSTANTIATE: ClassVar[str] = "Cannot instantiate"
    UNEXPECTED_MESSAGE_TYPE: ClassVar[str] = "Unexpected message type"
    VALIDATION_FAILED_FOR_TEST: ClassVar[str] = "Validation failed for test"

    MIN_LENGTH_DEFAULT: ClassVar[int] = 3
    MAX_VALUE: ClassVar[int] = 100


__all__: list[str] = ["TestsFlextConstantsErrors"]
