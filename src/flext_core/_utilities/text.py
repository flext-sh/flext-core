"""Primitive text helpers used by higher-level utilities.

The functions here intentionally return raw values and may raise on invalid
input; dispatcher-facing wrappers in ``flext_core.utilities`` apply
``p.Result`` semantics when needed. Keeping this layer minimal reduces
cross-layer coupling while providing deterministic normalization behaviors.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import c


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
            raise ValueError(c.ERR_TEXT_NONE_NOT_ALLOWED)
        stripped = text.strip()
        if not stripped:
            raise ValueError(c.ERR_TEXT_EMPTY_NOT_ALLOWED)
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


__all__: list[str] = ["FlextUtilitiesText"]
