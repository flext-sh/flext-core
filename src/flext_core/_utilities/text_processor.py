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
    """Text processing primitive utilities.

    ⚠️ PRIMITIVE FUNCTIONS - Return raw values, raise exceptions for errors.
    DO NOT wrap in FlextResult - that belongs in utilities.py (Tier 2).
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing extra whitespace and control characters.

        ⚠️ PRIMITIVE FUNCTION - Returns str directly, NOT FlextResult[str].

        Args:
            text: Text to clean

        Returns:
            str: Cleaned text with normalized whitespace

        """
        # Remove control characters except tab and newline, normalize whitespace
        return re.sub(
            r"\s+",
            " ",
            re.sub(FlextConstants.Utilities.CONTROL_CHARS_PATTERN, "", text),
        ).strip()

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
    def safe_string(text: str) -> str:
        """Validate and clean text string.

        ⚠️ PRIMITIVE FUNCTION - Returns str directly, raises on validation failure.

        Fast fail: text must be non-empty string. Raises ValueError on invalid input.

        Args:
            text: Text to validate and clean

        Returns:
            str: Cleaned text with whitespace stripped

        Raises:
            ValueError: If text is None, empty, or whitespace-only

        """
        # Fast fail: text cannot be None
        if text is None:
            msg = "Text cannot be None. Use explicit empty string '' or handle None in calling code."
            raise ValueError(msg)
        # Fast fail: text cannot be empty or whitespace-only
        stripped = text.strip()
        if not stripped:
            msg = "Text cannot be empty or whitespace-only. Use explicit non-empty string."
            raise ValueError(msg)
        return stripped


__all__ = ["FlextUtilitiesTextProcessor"]
