"""Centralized constants for the workspace subpackage."""

from __future__ import annotations

from typing import Final


class FlextInfraWorkspaceConstants:
    """Workspace infrastructure constants."""

    MAKEFILE_REPLACEMENTS: Final[tuple[tuple[str, str], ...]] = (
        (
            'python3 "$(BASE_MK_DIR)/scripts/mode.py"',
            "python -m flext_infra workspace detect",
        ),
        (
            'python3 "$(WORKSPACE_ROOT)/scripts/sync.py"',
            "python -m flext_infra workspace sync",
        ),
        (
            'python3 "$(WORKSPACE_ROOT)/scripts/dependencies/sync_internal_deps.py"',
            "python -m flext_infra deps internal-sync",
        ),
        (
            'python "$(WORKSPACE_ROOT)/scripts/check/fix_pyrefly_config.py"',
            "python -m flext_infra check fix-pyrefly-config",
        ),
        (
            'python "$(WORKSPACE_ROOT)/scripts/check/workspace_check.py"',
            "python -m flext_infra check run",
        ),
        (
            '$(VENV_PYTHON) "$(BASE_MK_DIR)/scripts/core/pytest_diag_extract.py"',
            "$(VENV_PYTHON) -m flext_infra core pytest-diag",
        ),
        (
            'python3 "$(WORKSPACE_ROOT)/scripts/github/pr_manager.py"',
            "python3 -m flext_infra github pr",
        ),
    )
    GITIGNORE_REMOVE_EXACT: Final[frozenset[str]] = frozenset({
        "!scripts/",
        "!scripts/**",
        "scripts/",
        "/scripts/",
    })
    GITIGNORE_REQUIRED_PATTERNS: Final[tuple[str, ...]] = (
        ".reports/",
        ".venv/",
        "__pycache__/",
    )
    REQUIRED_GITIGNORE_ENTRIES: Final[list[str]] = [
        ".reports/",
        ".venv/",
        "__pycache__/",
    ]


__all__ = ["FlextInfraWorkspaceConstants"]
