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
from flext_infra.core._constants import (
    FlextInfraCoreConstants,
    FlextInfraSharedInfraConstants,
)
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

    class Infra(FlextInfraSharedInfraConstants):
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
        "Minimum argv length for CLI dispatch."

        class Toml:
            """TOML section/key names for pyproject.toml parsing."""

            TOOL: Final[str] = "tool"
            "Top-level [tool] section key."
            COVERAGE: Final[str] = "coverage"
            "Coverage tool section key."
            POETRY: Final[str] = "poetry"
            "Poetry tool subsection key."
            PROJECT: Final[str] = "project"
            "Top-level [project] section key."
            DEPENDENCIES: Final[str] = "dependencies"
            "Dependencies key within project or poetry sections."
            DEV_DEPENDENCIES: Final[str] = "dev-dependencies"
            "Poetry dev-dependencies key."
            DEPENDENCY_GROUPS: Final[str] = "dependency-groups"
            "PEP 735 dependency-groups section key."
            BUILD_SYSTEM: Final[str] = "build-system"
            "Build system section key."
            OPTIONAL_DEPENDENCIES: Final[str] = "optional-dependencies"
            "Optional dependencies key within [project]."
            EXTRAS: Final[str] = "extras"
            "Poetry extras key."
            GROUP: Final[str] = "group"
            "Poetry group key for dependency groups."
            NAME: Final[str] = "name"
            "Project/package name key."
            VERSION: Final[str] = "version"
            "Version key within project or tool sections."
            PYREFLY: Final[str] = "pyrefly"
            "Pyrefly tool section key."
            MYPY: Final[str] = "mypy"
            "Mypy tool section key."
            PYRIGHT: Final[str] = "pyright"
            "Pyright tool section key."
            PYTEST: Final[str] = "pytest"
            "Pytest tool section key."
            RUFF: Final[str] = "ruff"
            "Ruff tool section key."
            DEPTRY: Final[str] = "deptry"
            "Deptry tool section key."
            ISORT: Final[str] = "isort"
            "Isort tool section key (ruff.lint.isort)."
            INI_OPTIONS: Final[str] = "ini_options"
            "Pytest ini_options subsection key."
            LINT_SECTION: Final[str] = "lint"
            "Ruff lint subsection key."
            SEARCH_PATH: Final[str] = "search-path"
            "Pyrefly search-path config key."
            PROJECT_EXCLUDES: Final[str] = "project-excludes"
            "Pyrefly project-excludes config key."
            SUB_CONFIG: Final[str] = "sub-config"
            "Pyrefly sub-config key."
            PYTHON_VERSION_HYPHEN: Final[str] = "python-version"
            "Pyrefly/pyright python-version config key (hyphenated)."
            PYTHON_VERSION_UNDERSCORE: Final[str] = "python_version"
            "Mypy python_version config key (underscored)."
            EXTEND: Final[str] = "extend"
            "Ruff extend config key."
            KNOWN_FIRST_PARTY_HYPHEN: Final[str] = "known-first-party"
            "Ruff isort known-first-party key (hyphenated)."
            KNOWN_FIRST_PARTY_UNDERSCORE: Final[str] = "known_first_party"
            "Ruff isort known_first_party key (underscored)."
            IGNORE_ERRORS_IN_GENERATED: Final[str] = "ignore-errors-in-generated-code"
            "Pyrefly ignore-errors-in-generated-code key."
            MINVERSION: Final[str] = "minversion"
            "Pytest minversion config key."
            PYTHON_CLASSES: Final[str] = "python_classes"
            "Pytest python_classes config key."
            PYTHON_FILES: Final[str] = "python_files"
            "Pytest python_files config key."
            ADDOPTS: Final[str] = "addopts"
            "Pytest addopts config key."
            MARKERS: Final[str] = "markers"
            "Pytest markers config key."
            PLUGINS: Final[str] = "plugins"
            "Mypy plugins config key."
            DISABLE_ERROR_CODE: Final[str] = "disable_error_code"
            "Mypy disable_error_code config key."
            IGNORE: Final[str] = "ignore"
            "Pyrefly/sub-config ignore key."
            EXCLUDE: Final[str] = "exclude"
            "Generic exclude key."
            PATH: Final[str] = "path"
            "Path key within dependency entries."
            ERROR: Final[str] = "error"
            "Deptry JSON error field key."
            CODE: Final[str] = "code"
            "Deptry JSON code field key."
            MODULE: Final[str] = "module"
            "Deptry JSON module field key."
            DEV: Final[str] = "dev"
            "Development dependency group name."
            DOCS: Final[str] = "docs"
            "Documentation dependency group name."
            SECURITY: Final[str] = "security"
            "Security dependency group name."
            TEST: Final[str] = "test"
            "Test dependency group name."
            TYPING_LIBRARIES: Final[str] = "typing_libraries"
            "Project limits typing_libraries key."
            MODULE_TO_PACKAGE: Final[str] = "module_to_package"
            "Typing libraries module_to_package mapping key."
            PYTHON: Final[str] = "python"
            "Python config subsection key (in limits)."

        class Style:
            """ANSI color codes and terminal symbols (SSOT for output styling)."""

            # ANSI escape sequences (empty if colors disabled)
            RESET: Final[str] = "\x1b[0m"
            RED: Final[str] = "\x1b[31m"
            GREEN: Final[str] = "\x1b[32m"
            YELLOW: Final[str] = "\x1b[33m"
            BLUE: Final[str] = "\x1b[34m"
            BOLD: Final[str] = "\x1b[1m"

            # Unicode/ASCII symbols
            OK: Final[str] = "✓"
            FAIL: Final[str] = "✗"
            WARN: Final[str] = "⚠"
            SKIP: Final[str] = "–"
            ARROW: Final[str] = "→"
            BULLET: Final[str] = "•"
            PATH: Final[str] = "path"
            "Path key within dependency entries."
            ERROR: Final[str] = "error"
            "Deptry JSON error field key."
            CODE: Final[str] = "code"
            "Deptry JSON code field key."
            MODULE: Final[str] = "module"
            "Deptry JSON module field key."
            DEV: Final[str] = "dev"
            "Development dependency group name."
            DOCS: Final[str] = "docs"
            "Documentation dependency group name."
            SECURITY: Final[str] = "security"
            "Security dependency group name."
            TEST: Final[str] = "test"
            "Test dependency group name."

        class Cli:
            """CLI tool binary names used in subprocess calls."""

            GIT: Final[str] = "git"
            "Git version control binary."
            RUFF: Final[str] = "ruff"
            "Ruff linter/formatter binary."
            POETRY: Final[str] = "poetry"
            "Poetry package manager binary."
            SG: Final[str] = "sg"
            "ast-grep (sg) binary."
            DEPTRY: Final[str] = "deptry"
            "Deptry dependency checker binary."
            BANDIT: Final[str] = "bandit"
            "Bandit security linter binary."
            MARKDOWNLINT: Final[str] = "markdownlint"
            "Markdown linter binary."
            GOFMT: Final[str] = "gofmt"
            "Go formatter binary."
            OUTPUT_JSON: Final[str] = "json"
            "Common CLI output format flag value."
            MYPY: Final[str] = "mypy"
            "Mypy type checker binary."
            PYRIGHT: Final[str] = "pyright"
            "Pyright type checker binary."
            PYREFLY: Final[str] = "pyrefly"
            "Pyrefly type checker binary."
            GOVET: Final[str] = "go"
            "Go vet binary (invoked as 'go vet')."
            MAKE: Final[str] = "make"
            "Make build tool binary."
            GH: Final[str] = "gh"
            "GitHub CLI binary."

            class GitCmd:
                """Git subcommand constants."""

                CHECKOUT: Final[str] = "checkout"
                STATUS: Final[str] = "status"
                PUSH: Final[str] = "push"
                PULL: Final[str] = "pull"
                FETCH: Final[str] = "fetch"
                COMMIT: Final[str] = "commit"
                DIFF: Final[str] = "diff"
                ADD: Final[str] = "add"
                STASH: Final[str] = "stash"
                TAG: Final[str] = "tag"
                LOG: Final[str] = "log"
                REV_PARSE: Final[str] = "rev-parse"
                CONFIG: Final[str] = "config"
                BRANCH: Final[str] = "branch"
                INIT: Final[str] = "init"
                CLONE: Final[str] = "clone"
                REMOTE: Final[str] = "remote"
                VET: Final[str] = "vet"

            class RuffCmd:
                """Ruff subcommand constants."""

                CHECK: Final[str] = "check"
                FORMAT: Final[str] = "format"

            class PoetryCmd:
                """Poetry subcommand constants."""

                RUN: Final[str] = "run"
                ADD: Final[str] = "add"
                CHECK: Final[str] = "check"
                SHOW: Final[str] = "show"

            class SgCmd:
                """ast-grep subcommand constants."""

                SCAN: Final[str] = "scan"

            class GhCmd:
                """GitHub CLI subcommand constants."""

                PR: Final[str] = "pr"
                VIEW: Final[str] = "view"
                CREATE: Final[str] = "create"
                MERGE: Final[str] = "merge"
                SQUASH: Final[str] = "squash"
                LIST: Final[str] = "list"
                CHECKS: Final[str] = "checks"
                CLOSE: Final[str] = "close"
                UPDATE_BRANCH: Final[str] = "update-branch"
                RELEASE: Final[str] = "release"
                WORKFLOW: Final[str] = "workflow"

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
            GOVET: Final[str] = "govet"
            GOFMT: Final[str] = "gofmt"
            MARKDOWNLINT: Final[str] = "markdownlint"
            TYPE_ALIAS: Final[str] = "type"
            DEFAULT_CSV: Final[str] = (
                "lint,format,pyrefly,mypy,pyright,security,markdown,go"
            )

        class Status:
            """Status strings for check results."""

            PASS: Final[str] = "PASS"
            "Status string for checks that passed."
            FAIL: Final[str] = "FAIL"
            "Status string for checks that failed."
            OK: Final[str] = "OK"
            "Status string for successful operations."
            WARN: Final[str] = "WARN"
            "Status string for operations with warnings."

        class Defaults:
            """Default fallback values."""

            UNKNOWN: Final[str] = "unknown"
            "Fallback for missing identifiers."
            UNNAMED: Final[str] = "unnamed"
            "Default name for unnamed projects."

        class MatchModes:
            """Scanner match mode constants."""

            PRESENT: Final[str] = "present"
            "Match mode where presence indicates violations."
            ABSENT: Final[str] = "absent"
            "Match mode where absence indicates violations."

        class Modes:
            """Operation mode constants."""

            BASELINE: Final[str] = "baseline"
            "Baseline quality mode."
            STRICT: Final[str] = "strict"
            "Strict quality mode."

        class Severity:
            """Severity level constants for check/report results."""

            ERROR: Final[str] = "error"
            WARNING: Final[str] = "warning"
            NOTE: Final[str] = "note"
            LOW: Final[str] = "low"
            SKIP: Final[str] = "skip"

        class ReportKeys:
            """Common dictionary key names for reports and results."""

            STATUS: Final[str] = "status"
            FILE: Final[str] = "file"
            LINE: Final[str] = "line"
            MESSAGE: Final[str] = "message"
            NAME: Final[str] = "name"
            VERSION: Final[str] = "version"
            SUMMARY: Final[str] = "summary"
            TOTALS: Final[str] = "totals"
            TOTAL: Final[str] = "total"
            RULES: Final[str] = "rules"
            CONFIG: Final[str] = "config"
            RELEASE: Final[str] = "release"
            ACTION: Final[str] = "action"
            PATH: Final[str] = "path"
            SCOPE: Final[str] = "scope"
            VIOLATIONS: Final[str] = "violations"
            VIOLATIONS_COUNT: Final[str] = "violations_count"
            EXIT_CODE: Final[str] = "exit_code"
            RULE_ID: Final[str] = "rule_id"
            OK: Final[str] = "ok"
            ENABLED: Final[str] = "enabled"
            PROJECTS: Final[str] = "projects"
            WORKSPACE: Final[str] = "workspace"
            ROOT: Final[str] = "root"
            ID: Final[str] = "id"
            GENERATED_AT: Final[str] = "generated_at"
            TAG: Final[str] = "tag"
            URL: Final[str] = "url"
            STRATEGY: Final[str] = "strategy"
            CLASS_NESTING: Final[str] = "class_nesting"
            TARGET_NAMESPACE: Final[str] = "target_namespace"
            SOURCE_SYMBOL: Final[str] = "source_symbol"
            LOOSE_NAME: Final[str] = "loose_name"
            REWRITE_SCOPE: Final[str] = "rewrite_scope"
            CONFIDENCE: Final[str] = "confidence"
            MANUAL_REVIEW: Final[str] = "manual_review"
            FIX_ACTION: Final[str] = "fix_action"
            CURRENT_FILE: Final[str] = "current_file"
            VIOLATION_TYPE: Final[str] = "violation_type"
            SUGGESTED_FIX: Final[str] = "suggested_fix"
            HELPER_CONSOLIDATION: Final[str] = "helper_consolidation"
            FORBIDDEN_TARGETS: Final[str] = "forbidden_targets"
            PER_FILE_COUNTS: Final[str] = "per_file_counts"
            POST_CHECKS: Final[str] = "post_checks"
            SUGGESTIONS: Final[str] = "suggestions"

        class Verbs:
            """CLI subcommand / action verb constants."""

            CHECK: Final[str] = "check"
            VALIDATE: Final[str] = "validate"
            RELEASE: Final[str] = "release"
            WORKSPACE: Final[str] = "workspace"
            RUN: Final[str] = "run"
            STATUS: Final[str] = "status"
            CLOSE: Final[str] = "close"
            OPEN: Final[str] = "open"
            CHECKS: Final[str] = "checks"
            REBASE: Final[str] = "rebase"
            BUILD: Final[str] = "build"

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
            "Common directories to exclude from analysis across all scripts."
            DOC_EXCLUDED_DIRS: Final[frozenset[str]] = COMMON_EXCLUDED_DIRS | {"site"}
            "Directories to exclude when analyzing documentation."
            PYPROJECT_SKIP_DIRS: Final[frozenset[str]] = COMMON_EXCLUDED_DIRS | {
                ".claude.disabled",
                ".flext-deps",
                ".sisyphus",
            }
            "Directories to skip when scanning pyproject.toml files."
            CHECK_EXCLUDED_DIRS: Final[frozenset[str]] = COMMON_EXCLUDED_DIRS | {
                ".flext-deps",
                "reports",
            }
            "Directories to exclude during quality checks."

        class Encoding:
            """Encoding constants."""

            DEFAULT: Final[str] = "utf-8"
            "Default text encoding for file operations."

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
            "Default directories to check in a project (root only uses scripts)."
            CHECK_DIRS_SUBPROJECT: Final[tuple[str, ...]] = ("src", "tests", "examples")
            "Subprojects: type-check src/tests/examples only (scripts are workspace copies, run from root)."

        class Deps(FlextInfraDepsConstants):
            """Deps constants via MRO."""

        class Docs(FlextInfraDocsConstants):
            """Docs constants via MRO."""

        class Github(FlextInfraGithubConstants):
            """GitHub repository constants."""

            GITHUB_REPO_URL: Final[str] = "https://github.com/flext-sh/flext"
            "Official GitHub repository URL for the FLEXT project."
            GITHUB_REPO_NAME: Final[str] = "flext-sh/flext"
            "GitHub repository name in owner/repo format."

        class Release(FlextInfraReleaseConstants):
            """Release constants via MRO."""

        class Workspace(FlextInfraWorkspaceConstants):
            """Workspace constants via MRO."""

        class Versioning:
            """Semantic versioning constants for version management."""

            PROJECT_SECTION: Final[str] = "[project]"
            "TOML section header for project metadata."
            SEMVER_RE: Final[re.Pattern[str]] = re.compile(
                r"^(\d+)\.(\d+)\.(\d+)(?:-dev)?$",
            )
            "Regex pattern for parsing semantic version strings."
            DEV_BRANCH_RE: Final[re.Pattern[str]] = re.compile(
                r"^(\d+\.\d+\.\d+)-dev$",
            )
            "Regex pattern for matching development branch names."
            VALID_BUMP_TYPES: Final[frozenset[str]] = frozenset({
                "major",
                "minor",
                "patch",
            })
            "Allowed version bump type identifiers."

        class Reporting:
            """Reporting service constants for .reports/ path management."""

            REPORTS_DIR_NAME: Final[str] = ".reports"
            "Standard directory name for report output."

        class Refactor(FlextInfraRefactorConstants):
            """Refactor module constants via MRO."""


c = FlextInfraConstants
__all__ = ["FlextInfraConstants", "c"]
