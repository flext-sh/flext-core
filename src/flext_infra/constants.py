"""Constants for flext-infra.

Defines configuration constants and enumerations for infrastructure services
including validation rules, check types, and workspace settings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final


class FlextInfraConstants:
    """Centralized constants for FLEXT infrastructure (Layer 0).

    Provides immutable, namespace-organized constants for infrastructure
    configuration, validation rules, check types, and workspace settings.

    Usage:
        >>> from flext_infra.constants import c
        >>> # Access constants via c.Status.PASS
        >>> # Access paths via c.Paths.VENV_BIN_REL
    """

    class Paths:
        """Path-related constants."""

        VENV_BIN_REL: Final[str] = ".venv/bin"
        """Relative path to the virtualenv bin directory from workspace root."""

        DEFAULT_SRC_DIR: Final[str] = "src"
        """Default source directory for Python projects."""

    class Files:
        """File-related constants."""

        PYPROJECT_FILENAME: Final[str] = "pyproject.toml"
        """Standard filename for Python project configuration."""

        MAKEFILE_FILENAME: Final[str] = "Makefile"
        """Standard filename for Makefile project markers."""

        BASE_MK: Final[str] = "base.mk"
        """Canonical base.mk filename."""

        GO_MOD: Final[str] = "go.mod"
        """Go module manifest filename."""

    class Gates:
        """Quality gate identifiers."""

        LINT: Final[str] = "lint"
        FORMAT: Final[str] = "format"
        PYREFLY: Final[str] = "pyrefly"
        MYPY: Final[str] = "mypy"
        PYRIGHT: Final[str] = "pyright"
        SECURITY: Final[str] = "security"
        MARKDOWN: Final[str] = "markdown"
        GO: Final[str] = "go"

        TYPE_ALIAS: Final[str] = "type"

        DEFAULT_CSV: Final[str] = (
            "lint,format,pyrefly,mypy,pyright,security,markdown,go"
        )

    class Status:
        """Status strings for check results."""

        PASS: Final[str] = "PASS"
        """Status string for checks that passed."""

        FAIL: Final[str] = "FAIL"
        """Status string for checks that failed."""

        OK: Final[str] = "OK"
        """Status string for successful operations."""

        WARN: Final[str] = "WARN"
        """Status string for operations with warnings."""

    class Excluded:
        """Directory exclusion sets for analysis."""

        COMMON_EXCLUDED_DIRS: Final[frozenset[str]] = frozenset({
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

        DOC_EXCLUDED_DIRS: Final[frozenset[str]] = COMMON_EXCLUDED_DIRS | {"site"}
        """Directories to exclude when analyzing documentation."""

        PYPROJECT_SKIP_DIRS: Final[frozenset[str]] = COMMON_EXCLUDED_DIRS | {
            ".claude.disabled",
            ".flext-deps",
            ".sisyphus",
        }
        """Directories to skip when scanning pyproject.toml files."""

        CHECK_EXCLUDED_DIRS: Final[frozenset[str]] = COMMON_EXCLUDED_DIRS | {
            ".flext-deps",
            "reports",
        }
        """Directories to exclude during quality checks."""

    class Check:
        """Check directory configuration."""

        DEFAULT_CHECK_DIRS: Final[tuple[str, ...]] = (
            "src",
            "tests",
            "examples",
            "scripts",
        )
        """Default directories to check in a project (root only uses scripts)."""

        CHECK_DIRS_SUBPROJECT: Final[tuple[str, ...]] = ("src", "tests", "examples")
        """Subprojects: type-check src/tests/examples only (scripts are workspace copies, run from root)."""

    class Github:
        """GitHub repository constants."""

        GITHUB_REPO_URL: Final[str] = "https://github.com/flext-sh/flext"
        """Official GitHub repository URL for the FLEXT project."""

        GITHUB_REPO_NAME: Final[str] = "flext-sh/flext"
        """GitHub repository name in owner/repo format."""

    class Encoding:
        """Encoding constants."""

        DEFAULT: Final[str] = "utf-8"
        """Default text encoding for file operations."""


c = FlextInfraConstants

__all__ = ["FlextInfraConstants", "c"]
