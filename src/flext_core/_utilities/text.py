"""Primitive text helpers used by higher-level utilities.

The functions here intentionally return raw values and may raise on invalid
input; dispatcher-facing wrappers in ``flext_core.utilities`` apply
``p.Result`` semantics when needed. Keeping this layer minimal reduces
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
            re.sub(c.CONTROL_CHARS_PATTERN, "", text),
        ).strip()

    @staticmethod
    def format_app_id(name: str) -> str:
        """Format application ID by normalizing name to lowercase with hyphens.

        Converts spaces and underscores to hyphens and lowercases the result.

        Args:
            name: Application name to format.

        Returns:
            Formatted application ID (lowercase, hyphens).

        """
        return name.lower().replace(" ", "-").replace("_", "-")

    @staticmethod
    def safe_string(text: str | None) -> str:
        """Validate and clean text string, ensuring non-empty result.

        Args:
            text: Text to validate and clean.

        Returns:
            Cleaned text with whitespace stripped.

        Raises:
            ValueError: If text is None, empty, or whitespace-only.

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
        max_length: int = c.DEFAULT_SIZE,
        suffix: str = "...",
    ) -> r[str]:
        """Truncate text to maximum length and append suffix if needed.

        Args:
            text: Text to truncate.
            max_length: Maximum length including suffix (default: DEFAULT_SIZE).
            suffix: Suffix to append if truncated (default: "...").

        Returns:
            p.Result[str] with truncated text or original if already short enough.

        """
        if len(text) <= max_length:
            return r[str].ok(text)
        truncated = text[: max_length - len(suffix)] + suffix
        return r[str].ok(truncated)

    @staticmethod
    def normalize_alnum(text: str) -> str:
        """Strip non-alphanumeric characters and lowercase the result.

        Useful for fuzzy matching and case-insensitive namespace comparison
        where hyphens, underscores, and punctuation should be ignored.

        Args:
            text: Text to normalize.

        Returns:
            Lowercase string with only alphanumeric characters.

        """
        return "".join(ch for ch in text.lower() if ch.isalnum())


__all__ = ["FlextUtilitiesText"]
