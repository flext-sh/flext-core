"""Consolidated regex patterns for infrastructure scripts.

Merges patterns from ``scripts/libs/patterns.py`` and
``scripts/libs/doc_patterns.py`` into a single namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from typing import ClassVar, TypeGuard

from flext_infra.typings import t


class FlextInfraUtilitiesPatterns:
    """Centralized regex patterns for infrastructure operations.

    Consolidates tooling patterns (mypy, stubs) and documentation
    patterns (markdown links, headings, TOC) into a single class.

    Usage via namespace::

        from flext_infra import u

        if u.Infra.matches("MYPY_HINT_RE", text):
            ...
    """

    MYPY_HINT_RE: ClassVar[re.Pattern[str]] = re.compile(
        r'note: Hint: "python3 -m pip install ([^"]+)"',
    )
    """Match mypy install hint messages, capturing the package name."""

    MYPY_STUB_RE: ClassVar[re.Pattern[str]] = re.compile(
        r'Library stubs not installed for "([^"]+)"',
    )
    """Match mypy missing stub messages, capturing the library name."""

    INTERNAL_PREFIXES: ClassVar[tuple[str, ...]] = ("flext_",)
    """Prefixes identifying internal FLEXT packages."""

    MARKDOWN_LINK_RE: ClassVar[re.Pattern[str]] = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    """Match markdown links capturing text (group 1) and URL (group 2)."""

    MARKDOWN_LINK_URL_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"\[[^\]]+\]\(([^)]+)\)"
    )
    """Match markdown links capturing only the URL (group 1)."""

    HEADING_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE
    )
    """Match any markdown heading (h1-h6), capturing the text."""

    HEADING_H2_H3_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"^(##|###)\s+(.+?)\s*$",
        re.MULTILINE,
    )
    """Match h2/h3 headings, capturing level (group 1) and text (group 2)."""

    ANCHOR_LINK_RE: ClassVar[re.Pattern[str]] = re.compile(r"\[([^\]]+)\]\(#([^)]+)\)")
    """Match internal anchor links, capturing text and anchor."""

    INLINE_CODE_RE: ClassVar[re.Pattern[str]] = re.compile(r"`[^`]*`")
    """Match inline code spans for stripping before analysis."""

    @staticmethod
    def _is_compiled_pattern(value: t.Infra.InfraValue) -> TypeGuard[re.Pattern[str]]:
        return isinstance(value, re.Pattern)

    @staticmethod
    def matches(pattern_name: str, text: str) -> bool:
        """Check if a pattern matches text.

        Args:
            pattern_name: Name of the pattern (e.g., 'MYPY_HINT_RE').
            text: Text to match against.

        Returns:
            True if the pattern matches, False otherwise.

        Example:
            >>> FlextInfraUtilitiesPatterns.matches("MYPY_HINT_RE", text)
            True

        """
        pattern_obj: t.Infra.InfraValue = getattr(
            FlextInfraUtilitiesPatterns, pattern_name, None
        )
        if not FlextInfraUtilitiesPatterns._is_compiled_pattern(pattern_obj):
            return False
        return pattern_obj.search(text) is not None


__all__ = ["FlextInfraUtilitiesPatterns"]
