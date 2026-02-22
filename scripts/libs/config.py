"""Shared configuration constants used by multiple scripts."""

from __future__ import annotations

COMMON_EXCLUDED_DIRS: frozenset[str] = frozenset({
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    ".reports",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
})
"""Common directories to exclude from analysis across all scripts."""

DOC_EXCLUDED_DIRS: frozenset[str] = COMMON_EXCLUDED_DIRS | {"site"}
"""Directories to exclude when analyzing documentation."""

PYPROJECT_SKIP_DIRS: frozenset[str] = COMMON_EXCLUDED_DIRS | {
    ".claude.disabled",
    ".flext-deps",
    ".sisyphus",
}
"""Directories to skip when scanning pyproject.toml files."""

CHECK_EXCLUDED_DIRS: frozenset[str] = COMMON_EXCLUDED_DIRS | {
    ".flext-deps",
    "reports",
}
"""Directories to exclude during quality checks."""

DEFAULT_CHECK_DIRS: tuple[str, ...] = ("src", "tests", "examples", "scripts")
"""Default directories to check in a project."""

DEFAULT_SRC_DIR: str = "src"
"""Default source directory for Python projects."""

GITHUB_REPO_URL: str = "https://github.com/flext-sh/flext"
"""Official GitHub repository URL for the FLEXT project."""

GITHUB_REPO_NAME: str = "flext-sh/flext"
"""GitHub repository name in owner/repo format."""

DEFAULT_ENCODING: str = "utf-8"
"""Default text encoding for file operations."""

VENV_BIN_REL: str = ".venv/bin"
"""Relative path to the virtualenv bin directory from workspace root."""

PYPROJECT_FILENAME: str = "pyproject.toml"
"""Standard filename for Python project configuration."""

MAKEFILE_FILENAME: str = "Makefile"
"""Standard filename for Makefile project markers."""

STATUS_PASS: str = "PASS"
"""Status string for checks that passed."""

STATUS_FAIL: str = "FAIL"
"""Status string for checks that failed."""

STATUS_OK: str = "OK"
"""Status string for successful operations."""

STATUS_WARN: str = "WARN"
"""Status string for operations with warnings."""


__all__ = [
    "CHECK_EXCLUDED_DIRS",
    "COMMON_EXCLUDED_DIRS",
    "DEFAULT_CHECK_DIRS",
    "DEFAULT_ENCODING",
    "DEFAULT_SRC_DIR",
    "DOC_EXCLUDED_DIRS",
    "GITHUB_REPO_NAME",
    "GITHUB_REPO_URL",
    "MAKEFILE_FILENAME",
    "PYPROJECT_FILENAME",
    "PYPROJECT_SKIP_DIRS",
    "STATUS_FAIL",
    "STATUS_OK",
    "STATUS_PASS",
    "STATUS_WARN",
    "VENV_BIN_REL",
]
