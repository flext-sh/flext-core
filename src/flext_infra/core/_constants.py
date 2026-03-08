"""Centralized constants for the core subpackage."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final


class FlextInfraCoreConstants:
    """Core infrastructure constants."""

    EXEMPT_FILENAMES: Final[frozenset[str]] = frozenset({
        "__init__.py",
        "conftest.py",
        "__main__.py",
    })
    EXEMPT_PREFIXES: Final[frozenset[str]] = frozenset({"test_", "_"})
    ALIAS_NAMES: Final[frozenset[str]] = frozenset({
        "c",
        "t",
        "m",
        "p",
        "u",
        "r",
        "d",
        "e",
        "h",
        "s",
        "x",
        "tc",
    })
    DUNDER_ALLOWED: Final[frozenset[str]] = frozenset({"__all__", "__version__"})
    TYPEVAR_CALLABLES: Final[frozenset[str]] = frozenset({
        "TypeVar",
        "ParamSpec",
        "TypeVarTuple",
    })
    ENUM_BASES: Final[frozenset[str]] = frozenset({"StrEnum", "Enum", "IntEnum"})
    COLLECTION_CALLS: Final[frozenset[str]] = frozenset({
        "frozenset",
        "tuple",
        "dict",
        "list",
    })
    SKILLS_DIR: Final[Path] = Path(".claude/skills")
    REPORT_DEFAULT: Final[str] = ".claude/skills/{skill}/report.json"
    BASELINE_DEFAULT: Final[str] = ".claude/skills/{skill}/baseline.json"
    CACHE_TTL_SECONDS: Final[int] = 300
    MISSING_IMPORT_RE: Final[re.Pattern[str]] = re.compile(
        "Cannot find module `([^`]+)` \\[missing-import\\]"
    )
    MYPY_HINT_RE: Final[re.Pattern[str]] = re.compile(
        "note:\\s+(?:hint|note):\\s+.*?`(types-\\S+)`"
    )
    MYPY_STUB_RE: Final[re.Pattern[str]] = re.compile(
        "Library stubs not installed for ['\\\"](\\S+?)['\\\"]"
    )
    INTERNAL_PREFIXES: Final[tuple[str, ...]] = ("flext_", "flext-")


__all__ = ["FlextInfraCoreConstants"]
