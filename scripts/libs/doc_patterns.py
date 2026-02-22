"""Shared regex patterns for documentation scripts."""

from __future__ import annotations

import re

MARKDOWN_LINK_RE: re.Pattern[str] = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
"""Match markdown links capturing both text (group 1) and URL (group 2).

Used by documentation fix and generation scripts.
"""

MARKDOWN_LINK_URL_RE: re.Pattern[str] = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
"""Match markdown links capturing only the URL (group 1).

Used by documentation audit scripts where only the target matters.
"""

HEADING_RE: re.Pattern[str] = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
"""Match any markdown heading (h1-h6), capturing the text (group 1)."""

HEADING_H2_H3_RE: re.Pattern[str] = re.compile(r"^(##|###)\s+(.+?)\s*$", re.MULTILINE)
"""Match h2/h3 headings only, capturing level (group 1) and text (group 2).

Used by TOC generation which only processes h2 and h3 headings.
"""

ANCHOR_LINK_RE: re.Pattern[str] = re.compile(r"\[([^\]]+)\]\(#([^)]+)\)")
"""Match internal anchor links ``[text](#anchor)``, capturing text and anchor."""

INLINE_CODE_RE: re.Pattern[str] = re.compile(r"`[^`]*`")
"""Match inline code spans for stripping before analysis."""

__all__ = [
    "ANCHOR_LINK_RE",
    "HEADING_H2_H3_RE",
    "HEADING_RE",
    "INLINE_CODE_RE",
    "MARKDOWN_LINK_RE",
    "MARKDOWN_LINK_URL_RE",
]
