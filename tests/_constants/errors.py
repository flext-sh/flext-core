"""Constants mixin for errors.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final

from flext_tests import c


class TestsFlextCoreConstantsErrors:
    class TestErrors:
        """Flext-core-specific error message patterns for validation."""

        DELIMITER_EMPTY: Final[str] = "Delimiter must be exactly one character"
        DELIMITER_MULTI: Final[str] = "Delimiter must be exactly one character"
        DELIMITER_WHITESPACE: Final[str] = "Delimiter cannot be a whitespace"
        SPLIT_EMPTY: Final[str] = "Split character cannot be empty"
        ESCAPE_EMPTY: Final[str] = "Escape character cannot be empty"
        SPLIT_ESCAPE_SAME: Final[str] = (
            "Split character and escape character cannot be the same"
        )
        FAILED_PARSE: Final[str] = "Failed to parse"
        FAILED_SPLIT: Final[str] = "Failed to split"
        FAILED_NORMALIZE: Final[str] = "Failed to normalize"
        FAILED_PIPELINE: Final[str] = "validation error"
        INVALID_REGEX: Final[str] = "Invalid regex pattern"
        USER_NOT_FOUND: Final[str] = "User not found"
        INVALID_EMAIL: Final[str] = "Invalid email address"
        VALUE_TOO_LOW: Final[str] = "Value must be positive"
        VALUE_TOO_HIGH: Final[str] = "Value must be <= 100"
        PROCESSING_ERROR: Final[str] = "Processing error occurred"

    class TestValidation:
        """Flext-core-specific validation constants for various checks."""

        MIN_LENGTH_DEFAULT: Final[int] = 3
        MAX_LENGTH_DEFAULT: Final[int] = 100
        MIN_VALUE: Final[int] = 0
        MAX_VALUE: Final[int] = 100

    class Exceptions(c):
        """Exception handling configuration for tests."""

        FailureLevel = c.FailureLevel
