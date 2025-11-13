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

# Module constants
MAX_PORT_NUMBER: int = 65535
MIN_PORT_NUMBER: int = 1
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
    def safe_string(
        text: str | None,
        default: str = FlextConstants.Performance.DEFAULT_EMPTY_STRING,
    ) -> str:
        """Convert text to safe string, handling None and empty values.

        Args:
            text: Text to make safe
            default: Default value if text is None or empty

        Returns:
            Safe string value

        """
        if not text:
            return default
        return text.strip()


__all__ = ["FlextUtilitiesTextProcessor"]
