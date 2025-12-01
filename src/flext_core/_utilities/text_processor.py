"""Primitive text helpers used by higher-level utilities.

The functions here intentionally return raw values and may raise on invalid
input; dispatcher-facing wrappers in ``flext_core.utilities`` apply
``FlextResult`` semantics when needed. Keeping this layer minimal reduces
cross-layer coupling while providing deterministic normalization behaviors.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re

from flext_core.constants import FlextConstants
from flext_core.result import FlextResult
from flext_core.runtime import FlextRuntime, StructlogLogger


class FlextUtilitiesTextProcessor:
    """Low-level text normalization helpers for CQRS utilities."""

    @property
    def logger(self) -> StructlogLogger:
        """Get logger instance using FlextRuntime (avoids circular imports).

        Returns structlog logger instance with all logging methods (debug, info, warning, error, etc).
        Uses same structure/config as FlextLogger but without circular import.
        """
        return FlextRuntime.get_logger(__name__)

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing extra whitespace and control characters.

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
    def safe_string(text: str | None) -> str:
        """Validate and clean text string.

        Args:
            text: Text to validate and clean

        Returns:
            str: Cleaned text with whitespace stripped

        Raises:
            ValueError: If text is ``None``, empty, or whitespace-only

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
