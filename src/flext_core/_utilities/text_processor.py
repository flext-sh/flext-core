"""Utilities module - FlextUtilitiesTextProcessor.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
import re

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult

_logger = logging.getLogger(__name__)


class FlextUtilitiesTextProcessor:
    """Text processing utilities using railway composition."""

    @staticmethod
    def clean_text(text: str) -> FlextResult[str]:
        """Clean text by removing extra whitespace and control characters."""
        # Remove control characters except tab and newline
        cleaned = re.sub(FlextConstants.Utilities.CONTROL_CHARS_PATTERN, "", text)
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return FlextResult[str].ok(cleaned)

    @staticmethod
    def truncate_text(
        text: str,
        max_length: int = FlextConstants.Performance.BatchProcessing.DEFAULT_SIZE,
        suffix: str = "...",
    ) -> FlextResult[str]:
        """Truncate text to maximum length with suffix."""
        if len(text) <= max_length:
            return FlextResult[str].ok(text)

        truncated = text[: max_length - len(suffix)] + suffix
        return FlextResult[str].ok(truncated)

    @staticmethod
    def safe_string(text: str) -> FlextResult[str]:
        """Validate and clean text string.

        Fast fail: text must be non-empty string. Use FlextResult for error handling.

        Args:
            text: Text to validate and clean

        Returns:
            FlextResult[str]: Success with cleaned text, or failure if text is None/empty

        """
        # Fast fail: text cannot be None
        if text is None:
            return FlextResult[str].fail(
                "Text cannot be None. Use explicit empty string '' or handle None in calling code.",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )
        # Fast fail: text cannot be empty or whitespace-only
        stripped = text.strip()
        if not stripped:
            return FlextResult[str].fail(
                "Text cannot be empty or whitespace-only. Use explicit non-empty string.",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )
        return FlextResult[str].ok(stripped)


__all__ = ["FlextUtilitiesTextProcessor"]
