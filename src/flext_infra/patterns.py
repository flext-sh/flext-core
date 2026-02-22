"""Consolidated regex patterns for infrastructure scripts.

Merges patterns from ``scripts/libs/patterns.py`` and
``scripts/libs/doc_patterns.py`` into a single namespace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re


class InfraPatterns:
    """Centralized regex patterns for infrastructure operations.

    Consolidates tooling patterns (mypy, stubs) and documentation
    patterns (markdown links, headings, TOC) into a single class.
    """

    # -- Tooling patterns (from patterns.py) --

    MYPY_HINT_RE: re.Pattern[str] = re.compile(
        r'note: Hint: "python3 -m pip install ([^"]+)"',
    )
    """Match mypy install hint messages, capturing the package name."""

    MYPY_STUB_RE: re.Pattern[str] = re.compile(
        r'Library stubs not installed for "([^"]+)"',
    )
    """Match mypy missing stub messages, capturing the library name."""

    INTERNAL_PREFIXES: tuple[str, ...] = ("flext_",)
    """Prefixes identifying internal FLEXT packages."""

    # -- Documentation patterns (from doc_patterns.py) --

    MARKDOWN_LINK_RE: re.Pattern[str] = re.compile(
        r"\[([^\]]+)\]\(([^)]+)\)",
    )
    """Match markdown links capturing text (group 1) and URL (group 2)."""

    MARKDOWN_LINK_URL_RE: re.Pattern[str] = re.compile(
        r"\[[^\]]+\]\(([^)]+)\)",
    )
    """Match markdown links capturing only the URL (group 1)."""

    HEADING_RE: re.Pattern[str] = re.compile(
        r"^#{1,6}\s+(.+?)\s*$",
        re.MULTILINE,
    )
    """Match any markdown heading (h1-h6), capturing the text."""

    HEADING_H2_H3_RE: re.Pattern[str] = re.compile(
        r"^(##|###)\s+(.+?)\s*$",
        re.MULTILINE,
    )
    """Match h2/h3 headings, capturing level (group 1) and text (group 2)."""

    ANCHOR_LINK_RE: re.Pattern[str] = re.compile(
        r"\[([^\]]+)\]\(#([^)]+)\)",
    )
    """Match internal anchor links, capturing text and anchor."""

    INLINE_CODE_RE: re.Pattern[str] = re.compile(r"`[^`]*`")
    """Match inline code spans for stripping before analysis."""


__all__ = ["InfraPatterns"]
