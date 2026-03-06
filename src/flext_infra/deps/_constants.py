"""Centralized constants for the deps subpackage."""

from __future__ import annotations

import re
from typing import Final


class FlextInfraDepsConstants:
    """Deps infrastructure constants."""

    # From extra_paths.py
    PYRIGHT_BASE_ROOT: Final[list[str]] = [
        "scripts",
        "src",
        "typings",
        "typings/generated",
    ]
    MYPY_BASE_ROOT: Final[list[str]] = ["typings", "typings/generated", "src"]
    PYRIGHT_BASE_PROJECT: Final[list[str]] = [
        ".",
        "src",
        "tests",
        "examples",
        "scripts",
        "../typings",
        "../typings/generated",
    ]
    MYPY_BASE_PROJECT: Final[list[str]] = [
        ".",
        "../typings",
        "../typings/generated",
        "src",
    ]

    # From internal_sync.py
    GIT_REF_RE: Final[re.Pattern[str]] = re.compile(
        r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$"
    )
    GITHUB_REPO_URL_RE: Final[re.Pattern[str]] = re.compile(
        r"^(?:git@github\.com:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?|https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?)$",
    )
    PEP621_PATH_RE: Final[re.Pattern[str]] = re.compile(r"@\s*(?:file:)?(?P<path>.+)$")

    # From modernizer.py
    SKIP_DIRS: Final[frozenset[str]] = frozenset({
        ".archive",
        ".claude.disabled",
        ".flext-deps",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".sisyphus",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "htmlcov",
        "node_modules",
        "site",
        "vendor",
    })
    DEP_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
    RECENT_LINES_FOR_MARKER: Final[int] = 3
    RECENT_LINES_FOR_DEV_DEP: Final[int] = 4

    # From path_sync.py
    FLEXT_DEPS_DIR: Final[str] = ".flext-deps"
    PEP621_PATH_DEP_RE: Final[re.Pattern[str]] = re.compile(
        r"^(?P<name>[A-Za-z0-9_.-]+)\s*@\s*(?:file:(?://)?)?(?P<path>.+)$",
    )
    PEP621_NAME_RE: Final[re.Pattern[str]] = re.compile(
        r"^\s*(?P<name>[A-Za-z0-9_.-]+)"
    )


__all__ = ["FlextInfraDepsConstants"]
