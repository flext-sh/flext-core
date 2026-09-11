"""Centralized constants for the docs subpackage."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraConstantsDocs:
    """Docs infrastructure constants."""

    DEFAULT_DOCS_OUTPUT_DIR: Final[str] = ".reports/docs"
    DOCS_CONFIG_FILENAME: Final[str] = "docs_config.json"
    PYTHON_FENCE_RUFF_EXTEND_IGNORE: Final[t.StrSequence] = (
        "D100",
        "D101",
        "D102",
        "D103",
        "PLR2004",
        "S101",
        "INP001",
        "T201",
        "T203",
        "ANN001",
        "ANN002",
        "ANN003",
        "ANN201",
        "ANN202",
        "ANN204",
        "ANN205",
        "PLC0415",
    )
    """Rules ignored for executable docs snippets that are not full modules/tests."""
    PYTHON_FENCE_RE: Final[t.RegexPattern] = re.compile(
        r"^```python\s*\n(?P<body>.*?)^```\s*$", re.MULTILINE | re.DOTALL
    )
    """Regex matching ``python`` fenced blocks; ``body`` group yields contents."""

    PYTHON_FENCE_FIX_RE: Final[t.RegexPattern] = re.compile(
        r"^(?P<open>```python\s*\n)(?P<body>.*?)^```\s*$", re.MULTILINE | re.DOTALL
    )
    """Regex matching ``python`` fenced blocks for fix-in-place replacement."""

    FENCE_NOTEST_RE: Final[t.RegexPattern] = re.compile(
        r"^```(\S+)\s+notest\s*$", re.MULTILINE
    )
    """Regex matching fenced code blocks with a ``notest`` info qualifier."""

    MANUAL_TOC_RE: Final[t.RegexPattern] = re.compile(
        r"<!--\s*TOC\s+START\s*-->.*?<!--\s*TOC\s+END\s*-->", re.DOTALL
    )
    """Regex matching a manually inserted table-of-contents block."""

    FENCED_BLOCK_RE: Final[t.RegexPattern] = re.compile(
        r"^(?P<fence>```+|~~~+)[^\n]*\n.*?^(?P=fence)[ \t]*$\n?",
        re.MULTILINE | re.DOTALL,
    )
    """Match a whole fenced code block, backtick or tilde, with its info string."""

    # --- Markdown link/heading patterns ---
    MARKDOWN_LINK_RE: Final[t.RegexPattern] = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    """Match markdown links capturing text (group 1) and URL (group 2)."""
    MARKDOWN_LINK_URL_RE: Final[t.RegexPattern] = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    """Match markdown links capturing only the URL (group 1)."""
    HEADING_RE: Final[t.RegexPattern] = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
    """Match any markdown heading (h1-h6), capturing the text."""
    HEADING_H2_H3_RE: Final[t.RegexPattern] = re.compile(
        r"^(##|###)\s+(.+?)\s*$", re.MULTILINE
    )
    """Match h2/h3 headings, capturing level (group 1) and text (group 2)."""
    ANCHOR_LINK_RE: Final[t.RegexPattern] = re.compile(r"\[([^\]]+)\]\(#([^)]+)\)")
    """Match internal anchor links, capturing text and anchor."""
    INLINE_CODE_RE: Final[t.RegexPattern] = re.compile(r"`[^`]*`")
    """Match inline code spans for stripping before analysis."""
    STRING_LITERAL_RE: Final[t.RegexPattern] = re.compile(
        r"""["']([a-zA-Z0-9_\.]+)["']"""
    )
    """Match quoted string literals, capturing the content."""


__all__: list[str] = ["FlextInfraConstantsDocs"]
