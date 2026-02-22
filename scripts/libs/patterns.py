"""Shared regex patterns for mypy and stub tooling."""

from __future__ import annotations

import re

MYPY_HINT_RE: re.Pattern[str] = re.compile(
    r'note: Hint: "python3 -m pip install ([^"]+)"'
)

MYPY_STUB_RE: re.Pattern[str] = re.compile(r'Library stubs not installed for "([^"]+)"')

INTERNAL_PREFIXES: tuple[str, ...] = ("flext_",)

__all__ = [
    "INTERNAL_PREFIXES",
    "MYPY_HINT_RE",
    "MYPY_STUB_RE",
]
