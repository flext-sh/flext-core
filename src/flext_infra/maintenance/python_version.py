"""Python version enforcement service for FLEXT workspace.

Ensures Python version constraints are consistent across all workspace projects.
Creates ``.python-version`` files and verifies ``requires-python`` in each
project's ``pyproject.toml``.

Runtime version checking is handled automatically by
``flext_core._python_version_guard`` (imported on ``from flext_core import …``).
This service only manages the static ``.python-version`` files used by
pyenv / asdf / mise for interpreter selection.

Usage::

    from flext_infra.maintenance.python_version import PythonVersionEnforcer

    service = PythonVersionEnforcer()
    result = service.execute(check_only=True, verbose=True)
    if result.is_success:
        logger.info("Python version check passed")

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import override

import structlog

from flext_core import FlextService, r
from flext_infra.constants import c
from flext_infra.discovery import DiscoveryService

logger = structlog.get_logger(__name__)


class PythonVersionEnforcer(FlextService[int]):
    """Service for enforcing Python version constraints across workspace.

    Validates that all projects have consistent Python version requirements
    and that the runtime matches the workspace requirement.

    Attributes:
        check_only: If True, only verify without making changes.
        verbose: If True, print detailed output for each project.

    """

    check_only: bool = False
    verbose: bool = False

    @override
    def execute(self, *, check_only: bool = False, verbose: bool = False) -> r[int]:
        """Execute Python version enforcement.

        Args:
            check_only: If True, only verify without making changes.
            verbose: If True, print detailed output.

        Returns:
            FlextResult[int]: Exit code (0 for success, 1 for failure).

        """
        self.check_only = check_only
        self.verbose = verbose

        root = self._workspace_root_from_file(__file__)
        required_minor = self._read_required_minor(root)
        projects = self._discover_projects(root)
        mode = "Checking" if check_only else "Enforcing"

        logger.info(
            "python_version_enforcement_started",
            mode=mode,
            required_minor=required_minor,
            project_count=len(projects),
        )

        # Workspace root pyproject.toml
        if not self._ensure_python_version_file(root, required_minor):
            logger.error(
                "python_version_enforcement_failed",
                reason="missing_enforcement",
                required_minor=required_minor,
            )
            return r[int].fail("enforcement failed")

        for project in projects:
            if not self._ensure_python_version_file(project, required_minor):
                logger.error(
                    "python_version_enforcement_failed",
                    reason="missing_enforcement",
                    required_minor=required_minor,
                    project=project.name,
                )
                return r[int].fail("enforcement failed")

        logger.info(
            "python_version_enforcement_completed",
            project_count=len(projects),
            required_minor=required_minor,
        )
        return r[int].ok(0)

    def _workspace_root_from_file(self, file: str | Path) -> Path:
        """Resolve workspace root by walking up from file location.

        Finds the first directory containing .git, Makefile, and pyproject.toml.

        Args:
            file: Path to a file (usually __file__).

        Returns:
            Absolute Path to the workspace root.

        Raises:
            RuntimeError: If workspace root cannot be found.

        """
        current = Path(file).resolve()
        if current.is_file():
            current = current.parent

        # Walk up the directory tree
        for parent in [current] + list(current.parents):
            markers = {".git", c.Files.MAKEFILE_FILENAME, c.Files.PYPROJECT_FILENAME}
            if all((parent / marker).exists() for marker in markers):
                return parent

        msg = f"workspace root not found from {file}"
        raise RuntimeError(msg)

    def _read_required_minor(self, workspace_root: Path) -> int:
        """Read the required Python minor version from workspace pyproject.toml.

        Falls back to 13 if the field cannot be parsed.

        Args:
            workspace_root: Path to workspace root.

        Returns:
            int: Required Python minor version.

        """
        pyproject = workspace_root / c.Files.PYPROJECT_FILENAME
        if not pyproject.is_file():
            return 13
        content = pyproject.read_text(encoding="utf-8")
        match = re.search(r'requires-python\s*=\s*"[>!=]*(\d+)\.(\d+)', content)
        if match is None:
            return 13
        return int(match.group(2))

    def _discover_projects(self, workspace_root: Path) -> list[Path]:
        """Discover all Python projects in workspace.

        Args:
            workspace_root: Path to workspace root.

        Returns:
            list[Path]: List of project paths.

        """
        discovery = DiscoveryService()
        result = discovery.discover_projects(workspace_root)
        if result.is_failure:
            return []
        return [
            p.path
            for p in result.value
            if (p.path / c.Files.PYPROJECT_FILENAME).exists()
        ]

    def _ensure_python_version_file(self, project: Path, required_minor: int) -> bool:
        """Ensure pyproject.toml requires-python matches required minor version.

        Args:
            project: Path to project.
            required_minor: Required Python minor version.

        Returns:
            bool: True if validation passed, False otherwise.

        """
        local_minor = self._read_required_minor(project)

        # 1. Validate pyproject.toml
        if local_minor != required_minor:
            if self.check_only:
                logger.error(
                    "python_version_pyproject_wrong",
                    local_minor=local_minor,
                    project=project.name,
                )
            else:
                logger.error(
                    "python_version_pyproject_mismatch",
                    local_minor=local_minor,
                    required_minor=required_minor,
                    project=project.name,
                )
                logger.error(
                    "python_version_manual_update_required",
                    project=project.name,
                    file=f"{project.name}/pyproject.toml",
                )
            return False

        # 2. Validate current runtime
        runtime_minor = sys.version_info.minor
        if runtime_minor != required_minor:
            logger.error(
                "python_runtime_minor_mismatch",
                runtime_minor=runtime_minor,
                required_minor=required_minor,
                project=project.name,
            )
            return False

        if self.verbose:
            logger.info(
                "python_version_validated",
                required_minor=required_minor,
                project=project.name,
            )

        return True
