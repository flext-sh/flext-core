"""Constants for flext-infra.

Defines configuration constants and enumerations for infrastructure services
including validation rules, check types, and workspace settings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from typing import Final

from flext_core import FlextConstants
from flext_infra.basemk._constants import FlextInfraBasemkConstants
from flext_infra.check._constants import FlextInfraCheckConstants
from flext_infra.codegen._constants import FlextInfraCodegenConstants
from flext_infra.core._constants import FlextInfraCoreConstants
from flext_infra.deps._constants import FlextInfraDepsConstants
from flext_infra.docs._constants import FlextInfraDocsConstants
from flext_infra.github._constants import FlextInfraGithubConstants
from flext_infra.refactor._constants import FlextInfraRefactorConstants
from flext_infra.release._constants import FlextInfraReleaseConstants
from flext_infra.workspace._constants import FlextInfraWorkspaceConstants


class FlextInfraConstants(FlextConstants):
    """Centralized constants for FLEXT infrastructure (Layer 0).

    Provides immutable, namespace-organized constants for infrastructure
    configuration, validation rules, check types, and workspace settings.

    Usage:
        >>> from flext_infra import c
        >>> c.Infra.Status.PASS
        >>> c.Infra.Paths.VENV_BIN_REL
        >>> c.Infra.Codegen.EXCLUDED_PROJECTS
    """

    class Infra:
        """Infrastructure domain constants."""

        KNOWN_VERBS: Final[frozenset[str]] = frozenset({
            "build",
            "check",
            "dependencies",
            "docs",
            "preflight",
            "release",
            "tests",
            "validate",
            "workspace",
        })

        MIN_ARGV: Final[int] = 2
        """Minimum argv length for CLI dispatch."""

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

        class Encoding:
            """Encoding constants."""

            DEFAULT: Final[str] = "utf-8"
            """Default text encoding for file operations."""

        class Basemk(FlextInfraBasemkConstants):
            """Basemk constants via MRO."""

        class Codegen(FlextInfraCodegenConstants):
            """Codegen constants via MRO."""

        class Core(FlextInfraCoreConstants):
            """Core constants via MRO."""

        class Check(FlextInfraCheckConstants):
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

        class Deps(FlextInfraDepsConstants):
            """Deps constants via MRO."""

        class Docs(FlextInfraDocsConstants):
            """Docs constants via MRO."""

        class Github(FlextInfraGithubConstants):
            """GitHub repository constants."""

            GITHUB_REPO_URL: Final[str] = "https://github.com/flext-sh/flext"
            """Official GitHub repository URL for the FLEXT project."""

            GITHUB_REPO_NAME: Final[str] = "flext-sh/flext"
            """GitHub repository name in owner/repo format."""

        class Release(FlextInfraReleaseConstants):
            """Release constants via MRO."""

        class Workspace(FlextInfraWorkspaceConstants):
            """Workspace constants via MRO."""

        class Paths:
            """Path resolution constants for workspace navigation."""

            WORKSPACE_MARKERS: Final[frozenset[str]] = frozenset({
                ".git",
                "Makefile",
                "pyproject.toml",
            })
            """Filesystem markers used to detect workspace root directories."""

            VENV_BIN_REL: Final[str] = ".venv/bin"
            """Relative path to the virtualenv bin directory from workspace root."""

            DEFAULT_SRC_DIR: Final[str] = "src"
            """Default source directory for Python projects."""

        class Versioning:
            """Semantic versioning constants for version management."""

            PROJECT_SECTION: Final[str] = "[project]"
            """TOML section header for project metadata."""

            SEMVER_RE: Final[re.Pattern[str]] = re.compile(
                r"^(\d+)\.(\d+)\.(\d+)(?:-dev)?$",
            )
            """Regex pattern for parsing semantic version strings."""

            DEV_BRANCH_RE: Final[re.Pattern[str]] = re.compile(
                r"^(\d+\.\d+\.\d+)-dev$",
            )
            """Regex pattern for matching development branch names."""

            VALID_BUMP_TYPES: Final[frozenset[str]] = frozenset({
                "major",
                "minor",
                "patch",
            })
            """Allowed version bump type identifiers."""

        class Reporting:
            """Reporting service constants for .reports/ path management."""

            REPORTS_DIR_NAME: Final[str] = ".reports"
            """Standard directory name for report output."""

        class Refactor(FlextInfraRefactorConstants):
            """Refactor module constants via MRO."""


c = FlextInfraConstants

__all__ = ["FlextInfraConstants", "c"]
