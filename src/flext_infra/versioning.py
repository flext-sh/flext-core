"""Versioning service for semantic version management.

Wraps versioning operations with r error handling,
replacing bare functions with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import override

from flext_core import r, s
from flext_infra import c
from flext_infra.toml_io import FlextInfraTomlService


class FlextInfraVersioningService(s[str]):
    """Infrastructure service for semantic versioning operations.

    Provides r-wrapped version parsing, bumping, and
    project version management.
    """

    def __init__(self, toml: FlextInfraTomlService | None = None) -> None:
        """Initialize the versioning service."""
        super().__init__()
        self._toml = toml or FlextInfraTomlService()

    @staticmethod
    def _extract_project_version_from_text(content: str) -> str | None:
        in_project_section = False
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if line.startswith("[") and line.endswith("]"):
                in_project_section = line == c.Infra.Versioning.PROJECT_SECTION
                continue
            if not in_project_section or not line.startswith(c.Infra.Toml.VERSION):
                continue
            match = re.match(r"^version\s*=\s*[\"']([^\"']+)[\"']\s*$", line)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _has_project_table(content: str) -> bool:
        return any(
            raw_line.strip() == c.Infra.Versioning.PROJECT_SECTION
            for raw_line in content.splitlines()
        )

    @staticmethod
    def _replace_project_version_in_text(content: str, version: str) -> str | None:
        lines = content.splitlines(keepends=True)
        in_project_section = False
        updated_lines: list[str] = []
        replaced = False
        for raw_line in lines:
            line = raw_line.strip()
            if line.startswith("[") and line.endswith("]"):
                in_project_section = line == c.Infra.Versioning.PROJECT_SECTION
                updated_lines.append(raw_line)
                continue
            if (
                in_project_section
                and line.startswith(c.Infra.Toml.VERSION)
                and (not replaced)
            ):
                line_ending = "\n" if raw_line.endswith("\n") else ""
                updated_lines.append(f'version = "{version}"{line_ending}')
                replaced = True
                continue
            updated_lines.append(raw_line)
        if not replaced:
            return None
        return "".join(updated_lines)

    def bump_version(self, version: str, bump_type: str) -> r[str]:
        """Bump a semantic version string.

        Args:
            version: The current version string.
            bump_type: One of "major", "minor", or "patch".

        Returns:
            r[str] with the bumped version.

        """
        if bump_type not in c.Infra.Versioning.VALID_BUMP_TYPES:
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

    def current_workspace_version(self, workspace_root: Path) -> r[str]:
        """Read the current version from the main pyproject.toml.

        Args:
            workspace_root: The root directory of the workspace.

        Returns:
            r[str] with the version string.

        """
        pyproject = workspace_root / c.Infra.Files.PYPROJECT_FILENAME
        try:
            content = pyproject.read_text(encoding=c.Infra.Encoding.DEFAULT)
        except OSError as exc:
            return r[str].fail(f"read failed: {exc}")
        version = self._extract_project_version_from_text(content)
        if version is None or not version.strip():
            return r[str].fail("version not found in pyproject.toml")
        return r[str].ok(version)

    @override
    def execute(self) -> r[str]:
        """Execute versioning (default: empty string).

        Returns:
            r with empty string by default.

        """
        return r[str].ok("")

    def parse_semver(self, version: str) -> r[tuple[int, int, int]]:
        """Parse a semantic version string into (major, minor, patch).

        Args:
            version: The version string to parse.

        Returns:
            r with version tuple.

        """
        match = c.Infra.Versioning.SEMVER_RE.match(version)
        if not match:
            return r[tuple[int, int, int]].fail(f"invalid semver: {version}")
        return r[tuple[int, int, int]].ok((
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        ))

    def release_tag_from_branch(self, branch: str) -> r[str]:
        """Extract a release tag name from a branch name.

        Supports ``release/X.Y.Z`` and ``X.Y.Z-dev`` patterns.

        Args:
            branch: The branch name.

        Returns:
            r[str] with the tag name (e.g., "v1.2.3"),
            or failure if no pattern matches.

        """
        if branch.startswith("release/"):
            tag = f"v{branch.removeprefix('release/')}"
            return r[str].ok(tag)
        match = c.Infra.Versioning.DEV_BRANCH_RE.match(branch)
        if match:
            return r[str].ok(f"v{match.group(1)}")
        return r[str].fail(f"branch '{branch}' does not match release pattern")

    def replace_project_version(self, project_path: Path, version: str) -> r[bool]:
        """Update the version field in a project's pyproject.toml.

        Args:
            project_path: Directory containing pyproject.toml.
            version: The new version string.

        Returns:
            r[bool] with True on success.

        """
        pyproject = project_path / c.Infra.Files.PYPROJECT_FILENAME
        try:
            content = pyproject.read_text(encoding=c.Infra.Encoding.DEFAULT)
        except OSError as exc:
            return r[bool].fail(f"read failed: {exc}")
        if not self._has_project_table(content):
            return r[bool].fail(f"missing [project] table in {pyproject}")
        updated_content = self._replace_project_version_in_text(content, version)
        if updated_content is None:
            return r[bool].fail(f"missing [project] version in {pyproject}")
        try:
            _ = pyproject.write_text(updated_content, encoding=c.Infra.Encoding.DEFAULT)
        except OSError as exc:
            return r[bool].fail(f"write failed: {exc}")
        return r[bool].ok(True)


__all__ = ["FlextInfraVersioningService"]
