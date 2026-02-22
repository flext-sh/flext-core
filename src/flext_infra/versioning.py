"""Versioning service for semantic version management.

Wraps versioning operations with FlextResult error handling,
replacing bare functions with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping
from pathlib import Path

from flext_core.result import FlextResult, r

from flext_infra.constants import ic
from flext_infra.toml_io import TomlService

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-dev)?$")
_DEV_BRANCH_RE = re.compile(r"^(\d+\.\d+\.\d+)-dev$")
_VALID_BUMP_TYPES = frozenset({"major", "minor", "patch"})


class VersioningService:
    """Infrastructure service for semantic versioning operations.

    Provides FlextResult-wrapped version parsing, bumping, and
    project version management.
    """

    def __init__(self, toml: TomlService | None = None) -> None:
        self._toml = toml or TomlService()

    def parse_semver(
        self,
        version: str,
    ) -> FlextResult[tuple[int, int, int]]:
        """Parse a semantic version string into (major, minor, patch).

        Args:
            version: The version string to parse.

        Returns:
            FlextResult with version tuple.

        """
        match = _SEMVER_RE.match(version)
        if not match:
            return r[tuple[int, int, int]].fail(
                f"invalid semver: {version}",
            )
        return r[tuple[int, int, int]].ok(
            (int(match.group(1)), int(match.group(2)), int(match.group(3))),
        )

    def bump_version(
        self,
        version: str,
        bump_type: str,
    ) -> FlextResult[str]:
        """Bump a semantic version string.

        Args:
            version: The current version string.
            bump_type: One of "major", "minor", or "patch".

        Returns:
            FlextResult[str] with the bumped version.

        """
        if bump_type not in _VALID_BUMP_TYPES:
            return r[str].fail(f"invalid bump type: {bump_type}")

        result = self.parse_semver(version)
        if result.is_failure:
            return r[str].fail(result.error or "parse failed")

        major, minor, patch = result.value
        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        else:
            patch += 1

        return r[str].ok(f"{major}.{minor}.{patch}")

    def release_tag_from_branch(self, branch: str) -> FlextResult[str]:
        """Extract a release tag name from a branch name.

        Supports ``release/X.Y.Z`` and ``X.Y.Z-dev`` patterns.

        Args:
            branch: The branch name.

        Returns:
            FlextResult[str] with the tag name (e.g., "v1.2.3"),
            or failure if no pattern matches.

        """
        if branch.startswith("release/"):
            tag = f"v{branch.removeprefix('release/')}"
            return r[str].ok(tag)
        match = _DEV_BRANCH_RE.match(branch)
        if match:
            return r[str].ok(f"v{match.group(1)}")
        return r[str].fail(
            f"branch '{branch}' does not match release pattern",
        )

    def current_workspace_version(
        self,
        workspace_root: Path,
    ) -> FlextResult[str]:
        """Read the current version from the main pyproject.toml.

        Args:
            workspace_root: The root directory of the workspace.

        Returns:
            FlextResult[str] with the version string.

        """
        pyproject = workspace_root / ic.Files.PYPROJECT_FILENAME
        doc_result = self._toml.read_document(pyproject)
        if doc_result.is_failure:
            return r[str].fail(doc_result.error or "read failed")

        doc = doc_result.value
        project = doc.get("project")
        project_table = project if isinstance(project, MutableMapping) else None
        version = project_table.get("version") if project_table is not None else None
        if not isinstance(version, str) or not version.strip():
            return r[str].fail("version not found in pyproject.toml")
        return r[str].ok(version)

    def replace_project_version(
        self,
        project_path: Path,
        version: str,
    ) -> FlextResult[bool]:
        """Update the version field in a project's pyproject.toml.

        Args:
            project_path: Directory containing pyproject.toml.
            version: The new version string.

        Returns:
            FlextResult[bool] with True on success.

        """
        pyproject = project_path / ic.Files.PYPROJECT_FILENAME
        doc_result = self._toml.read_document(pyproject)
        if doc_result.is_failure:
            return r[bool].fail(doc_result.error or "read failed")

        doc = doc_result.value
        project = doc.get("project")
        if not isinstance(project, MutableMapping):
            return r[bool].fail(
                f"missing [project] table in {pyproject}",
            )

        project["version"] = version
        write_result = self._toml.write_document(pyproject, doc)
        if write_result.is_failure:
            return r[bool].fail(write_result.error or "write failed")
        return r[bool].ok(True)


__all__ = ["VersioningService"]
