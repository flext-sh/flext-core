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

from flext_core import r


class FlextUtilitiesText:
    """Low-level text normalization helpers for CQRS utilities."""

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

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize whitespace and strip control characters from text."""
        cleaned = text.translate(str.maketrans("", "", "\x00\r\n\t"))
        return re.sub(r" +", " ", cleaned).strip()

    @staticmethod
    def truncate_text(text: str, *, max_length: int) -> r[str]:
        """Truncate text with an ellipsis when it exceeds the target length."""
        if max_length <= 0:
            return r[str].fail("max_length must be greater than zero")
        if len(text) <= max_length:
            return r[str].ok(text)
        return r[str].ok(f"{text[:max_length]}...")


__all__ = ["FlextUtilitiesText"]
