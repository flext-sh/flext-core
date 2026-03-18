"""Primitive text helpers used by higher-level utilities.

The functions here intentionally return raw values and may raise on invalid
input; dispatcher-facing wrappers in ``flext_core.utilities`` apply
``r`` semantics when needed. Keeping this layer minimal reduces
cross-layer coupling while providing deterministic normalization behaviors.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re

from flext_core import FlextRuntime, c, p, r


class FlextUtilitiesText:
    """Low-level text normalization helpers for CQRS utilities."""

    @property
    def logger(self) -> p.Logger:
        """Get structlog logger via FlextRuntime (infrastructure-level, no FlextLogger)."""
        return FlextRuntime.get_logger(__name__)

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing extra whitespace and control characters.

        Args:
            text: Text to clean

        Returns:
            str: Cleaned text with normalized whitespace

        """
        return re.sub(
            r"\s+",
            " ",
            re.sub(c.Utilities.CONTROL_CHARS_PATTERN, "", text),
        ).strip()

    @staticmethod
    def format_app_id(name: str) -> str:
        """Format application ID.

        Converts a name to a valid application ID by lowercasing
        and replacing spaces and underscores with hyphens.

        Args:
            name: Application name to format

        Returns:
            Formatted application ID (lowercase, hyphens)

        Example:
            app_id = u.format_app_id("My Application_Name")
            # → "my-application-name"

        """
        return name.lower().replace(" ", "-").replace("_", "-")

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
        if text is None:
            msg = "Text cannot be None. Use explicit empty string '' or handle None in calling code."
            raise ValueError(msg)
        stripped = text.strip()
        if not stripped:
            msg = "Text cannot be empty or whitespace-only. Use explicit non-empty string."
            raise ValueError(msg)
        return stripped

    @staticmethod
    def truncate_text(
        text: str,
        max_length: int = c.Performance.BatchProcessing.DEFAULT_SIZE,
        suffix: str = "...",
    ) -> r[str]:
        """Truncate text to maximum length with suffix."""
        if len(text) <= max_length:
            return r[str].ok(text)
        truncated = text[: max_length - len(suffix)] + suffix
        return r[str].ok(truncated)

    @staticmethod
    def normalize_alnum(text: str) -> str:
        """Strip non-alphanumeric characters and lowercase the result.

        Useful for fuzzy namespace matching where hyphens, underscores,
        and other punctuation should be ignored.

        Args:
            text: Text to normalize.

        Returns:
            Lowercase string with only alphanumeric characters.

        Example:
            >>> u.normalize_alnum("flext-ldap")
            'flextldap'

        """
        return "".join(ch for ch in text.lower() if ch.isalnum())


__all__ = ["FlextUtilitiesText"]
